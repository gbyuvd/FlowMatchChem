import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

# ---------------- AdaLN + small Transformer block ----------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B, L, D], shift/scale: [B, 1, D]
    return x * (1 + scale) + shift

class AdaLayerNorm(nn.Module):
    """LayerNorm whose shift/scale are provided externally (AdaLN style)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # base gamma (learned single vector) to keep behaviour consistent
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        # standard layer norm on float for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            y = F.layer_norm(x.float(), (self.dim,))
        y = y.to(x.dtype) * self.weight[None, None, :]
        return y * (1 + scale) + shift

class AdaTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, mlp_ratio=4, dropout=0.1, cond_dim=128):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.ada_ln1 = AdaLayerNorm(dim)
        self.ada_ln2 = AdaLayerNorm(dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

        # projection from conditioning to modulation params
        # produce 6 vectors: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
        self.ada_proj = nn.Linear(cond_dim, 6 * dim)
        # init small
        nn.init.zeros_(self.ada_proj.weight)
        nn.init.zeros_(self.ada_proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: [B, L, D], c: [B, cond_dim]
        B, L, D = x.shape
        mod = self.ada_proj(c)[:, None, :]  # [B,1,6D]
        s_attn, sc_attn, g_attn, s_mlp, sc_mlp, g_mlp = mod.chunk(6, dim=-1)

        # Attention block
        x_skip = x
        x_norm = self.ada_ln1(x, shift=s_attn, scale=sc_attn)
        q = self.q(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, HD]
        k = self.k(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention (using torch function)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=0.0 if not self.training else self.dropout.p)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.attn_out(attn)
        x = x_skip + (1.0 + g_attn) * self.dropout(attn_out)  # gate_attn used as multiplicative gate

        # MLP block
        x_skip2 = x
        x_norm2 = self.ada_ln2(x, shift=s_mlp, scale=sc_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x_skip2 + (1.0 + g_mlp) * self.dropout(mlp_out)

        return x

# ---------------- Transformer backbone container ----------------
class TransformerFlowBackbone(nn.Module):
    """Transformer backbone with AdaLN time conditioning for flow-matching."""
    def __init__(self, hidden=320, n_layers=6, n_heads=8, mlp_ratio=4,
                 dropout=0.1, cond_dim=128, use_global_token=True):
        super().__init__()
        self.hidden = hidden
        self.use_global = use_global_token
        self.input_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.blocks = nn.ModuleList([
            AdaTransformerBlock(dim=hidden, n_heads=n_heads, mlp_ratio=mlp_ratio,
                                dropout=dropout, cond_dim=cond_dim)
            for _ in range(n_layers)
        ])
        self.global_pool = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.final_norm = nn.LayerNorm(hidden)

    def forward(self, inputs_embeds: torch.Tensor, time_cond: torch.Tensor = None):
        # inputs_embeds: [B, L, H]
        x = self.input_proj(inputs_embeds)
        B, L, D = x.shape
        if self.use_global:
            g = self.global_pool.expand(B, -1, -1)
            x = torch.cat([g, x], dim=1)  # [B, L+1, D]

        # cond vector
        if time_cond is None:
            c = torch.zeros(B, self.time_mlp[0].in_features, device=x.device)
        else:
            c = self.time_mlp(time_cond)  # [B, cond_dim]

        for block in self.blocks:
            x = block(x, c)

        if self.use_global:
            global_info, x = x[:, :1], x[:, 1:]
            x = x + 0.1 * global_info
        x = self.final_norm(x)
        return SimpleNamespace(last_hidden_state=x)
