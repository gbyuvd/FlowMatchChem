# fm3.py
# Flow Matching for Molecules (FM3) - Lightweight, modular, laptop-friendly
import math
import os
import json
from typing import Optional, Dict, Any
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


# ---------- scheduler --------------------------------------------------------
class OptimalTransportSchedule:
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, t: torch.Tensor):
        t = torch.clamp(t, self.eps, 1 - self.eps)
        alpha = t
        sigma = 1 - t
        d_alpha = torch.ones_like(t)
        d_sigma = -torch.ones_like(t)
        return alpha, sigma, d_alpha, d_sigma


# ---------- time embedding ---------------------------------------------------
class TimeAdditiveEmbedder(nn.Module):
    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        half = hidden_size // 2
        self.register_buffer('freqs', torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        self.offset = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, t: torch.Tensor):
        t = torch.clamp(t, 0.0, 1.0)
        args = t[:, None] * self.freqs[None].to(t)
        emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
        if emb.shape[-1] % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], -1)
        return 0.1 * self.mlp(emb)[:, None, :] + self.offset


# ---------- linear tail ------------------------------------------------------
class LinearTail(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.eps = eps

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, d_alpha: torch.Tensor):
        alpha_safe = torch.clamp(alpha, min=self.eps)
        coeff = d_alpha / alpha_safe
        return coeff[:, None, None] * x * torch.tanh(self.scale)


# ---------- lightweight Mamba backbone --------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.wg = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        gate = self.wg(x)
        hidden = self.act(self.w1(x))
        out = self.w2(self.dropout(hidden))
        return out * torch.sigmoid(gate)


class MambaBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        hidden = int(dim * mlp_mult)
        self.mlp = SwiGLU(dim, hidden, dropout=dropout)
        self.res_scale_attn = nn.Parameter(torch.tensor(0.1))
        self.res_scale_mlp = nn.Parameter(torch.tensor(0.1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, need_weights=False)
        x = x + self.res_scale_attn * self.drop(attn_out)
        x_conv = self.pw(self.dw_conv(x.transpose(1, 2))).transpose(1, 2)
        x = x + 0.1 * x_conv
        x_ln2 = self.ln2(x)
        mlp_out = self.mlp(x_ln2)
        x = x + self.res_scale_mlp * self.drop(mlp_out)
        return x


class MambaBackbone(nn.Module):
    def __init__(self, hidden: int = 320, n_layers: int = 4, n_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(hidden, n_heads, mlp_mult, dropout)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden, eps=1e-6)

    def forward(self, inputs_embeds: torch.Tensor):
        x = inputs_embeds
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        return SimpleNamespace(last_hidden_state=x)


# ---------- lightweight Simple backbone --------------------------------------
class SimpleFlowBackbone(nn.Module):
    def __init__(self, hidden: int = 320, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        res = x
        x = x.transpose(1, 2)  # [B, H, L]
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # [B, L, H]
        x = self.norm1(x + res)
        return SimpleNamespace(last_hidden_state=x)


# ---------- Backbone Registry -------------------------------------------------
BACKBONES = {
    'simple': SimpleFlowBackbone,
    'mamba': MambaBackbone,
}


# ---------- Main Model: FM3 --------------------------------------------------
class FM3(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1162,
        hidden: int = 320,
        backbone_type: str = 'simple',
        eps: float = 1e-5,
        dropout: float = 0.1,
        **backbone_kwargs  # ← catches n_layers, n_heads, etc.
    ):
        super().__init__()
        if backbone_type not in BACKBONES:
            raise ValueError(f"Backbone '{backbone_type}' not supported. Choose from {list(BACKBONES.keys())}")
        
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.backbone_type = backbone_type
        self.eps = eps

        self.embed = nn.Embedding(vocab_size, hidden)
        nn.init.normal_(self.embed.weight, std=0.02)

        # Pass ALL backbone_kwargs to the chosen backbone
        self.backbone = BACKBONES[backbone_type](
            hidden=hidden,
            dropout=dropout,
            **backbone_kwargs  # ← n_layers, n_heads, mlp_mult, kernel_size, etc.
        )
        self.time_emb = TimeAdditiveEmbedder(hidden)
        self.lin_tail = LinearTail(hidden, eps=eps)
        self.v_head = nn.Linear(hidden, hidden)
        nn.init.xavier_normal_(self.v_head.weight, gain=0.01)
        nn.init.zeros_(self.v_head.bias)
        self.output_norm = nn.LayerNorm(hidden)
        self.sched = OptimalTransportSchedule(eps=eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_emb(t)
        x_with_time = F.layer_norm(x_t + time_emb, (x_t.shape[-1],))
        h = self.backbone(x_with_time).last_hidden_state
        h = self.output_norm(h)
        v = self.v_head(h)
        return v

    @torch.no_grad()
    def sample_from_noise(self, noise: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Generate molecules from pure Gaussian noise.
        noise: [B, L, H] - samples from N(0, I)
        """
        return self.sample_euler(noise, steps)

    @torch.no_grad()
    def sample_euler(self, x: torch.Tensor, steps: int = 50) -> torch.Tensor:
        device = x.device
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=device)
            v = self.forward(x, t)
            x = x + v * dt
        return x

    # --- Explicit Sampling Helpers ---
    @torch.no_grad()
    def sample_from_noise(self, noise: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Generate molecules from pure Gaussian noise.
        noise: [B, L, H] - samples from N(0, I)
        """
        return self.sample_euler(noise, steps)

    @torch.no_grad()
    def sample_from_tokens(self, tokens: torch.Tensor, noise_scale: float = 0.1, steps: int = 50) -> torch.Tensor:
        """Generate similar molecules from existing ones (conditional).
        tokens: [B, L] - input token IDs
        """
        emb = self.embed(tokens)
        noise = torch.randn_like(emb) * noise_scale
        # Start from the molecule + small noise, then integrate forward
        return self.sample_euler(emb + noise, steps)

    # --- Hugging Face-style Save/Load ---
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        config = {
            'vocab_size': self.vocab_size,
            'hidden': self.hidden,
            'backbone_type': self.backbone_type,
            'eps': self.eps,
            'dropout': self.lin_tail.eps,  # approximate; better to store actual dropout if needed
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str, **override_kwargs):
        config_path = os.path.join(save_directory, "config.json")
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        with open(config_path) as f:
            config = json.load(f)
        config.update(override_kwargs)
        model = cls(**config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model