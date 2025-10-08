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


# ---------- Optimized Mamba-inspired components for vector field regression --
class RMSNorm(nn.Module):
    """RMSNorm for better gradient flow in regression tasks"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class GatedConvolution(nn.Module):
    """Gated convolution for local pattern extraction"""
    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size, 
                              padding=padding, dilation=dilation, groups=dim)
        self.gate_norm = nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        residual = x
        x = x.transpose(1, 2)  # [B, D, L]
        gates = self.conv(x).transpose(1, 2)  # [B, L, 2*D]
        x, gate = gates.chunk(2, dim=-1)
        x = x * torch.sigmoid(self.gate_norm(gate))
        return x + residual


class SelectiveMixing(nn.Module):
    """Selective information mixing inspired by Mamba's selective mechanism"""
    def __init__(self, dim: int, expand_ratio: float = 1.5, dropout: float = 0.1):
        super().__init__()
        inner_dim = int(dim * expand_ratio)
        
        # Input-dependent selection mechanism
        self.select_proj = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim * 3)  # Query, Key, Gate
        )
        
        # Value projection with residual
        self.value_proj = nn.Linear(dim, dim)
        
        # Learnable mixing weights
        self.mix_weight = nn.Parameter(torch.ones(dim) * 0.5)
        self.norm = RMSNorm(dim)
        
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.5)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # Input-dependent selection
        qkg = self.select_proj(x)
        q, k, g = qkg.chunk(3, dim=-1)
        
        # Selective attention (simplified)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply gating
        g = torch.sigmoid(g)
        v = self.value_proj(x) * g
        
        # Mix local and global information
        global_info = torch.bmm(attn, v)
        output = self.mix_weight * global_info + (1 - self.mix_weight) * x
        
        return self.norm(output)


class VectorFieldBlock(nn.Module):
    """Optimized block for vector field regression"""
    def __init__(self, dim: int, kernel_size: int = 7, expand_ratio: float = 2.0, 
                 dropout: float = 0.1, dilation: int = 1):
        super().__init__()
        
        # Multi-scale convolutions for capturing different frequencies
        self.conv_short = GatedConvolution(dim, kernel_size=3, dilation=1)
        self.conv_long = GatedConvolution(dim, kernel_size=kernel_size, dilation=dilation)
        
        # Selective mixing for adaptive information flow
        self.selective_mix = SelectiveMixing(dim, expand_ratio, dropout)
        
        # Smooth MLP for vector field prediction
        hidden_dim = int(dim * expand_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        
        # Learnable residual weights for stable training
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x: torch.Tensor):
        # Multi-scale local patterns
        x = x + self.alpha * self.conv_short(x)
        x = x + self.alpha * self.conv_long(x)
        
        # Selective global mixing
        x = x + self.beta * self.selective_mix(x)
        
        # Smooth transformation for vector field
        x = x + self.beta * self.mlp(x)
        
        return x


class VectorFieldBackbone(nn.Module):
    """Optimized Mamba-inspired backbone for vector field regression"""
    def __init__(self, hidden: int = 320, n_layers: int = 4, 
                 kernel_size: int = 7, expand_ratio: float = 2.0,
                 dropout: float = 0.1, use_multiscale: bool = True,
                 n_heads: int = 4, mlp_mult: int = 4, **kwargs):
        super().__init__()
        
        # Initial projection with smooth activation
        self.input_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU()
        )
        
        # Progressive dilation for multi-scale understanding
        dilations = [1, 2, 4, 8] if use_multiscale else [1] * n_layers
        dilations = (dilations * (n_layers // len(dilations) + 1))[:n_layers]
        
        self.blocks = nn.ModuleList([
            VectorFieldBlock(
                dim=hidden,
                kernel_size=kernel_size,
                expand_ratio=expand_ratio,
                dropout=dropout,
                dilation=dilations[i]
            )
            for i in range(n_layers)
        ])
        
        # Smooth output normalization for regression
        self.final_norm = nn.Sequential(
            RMSNorm(hidden),
            nn.LayerNorm(hidden)  # Double normalization for stability
        )
        
        # Global context aggregation
        self.global_pool = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        
    def forward(self, inputs_embeds: torch.Tensor):
        x = self.input_proj(inputs_embeds)
        
        # Add global context token
        B, L, D = x.shape
        global_token = self.global_pool.expand(B, -1, -1)
        x = torch.cat([global_token, x], dim=1)
        
        # Process through blocks
        for block in self.blocks:
            x = block(x)
        
        # Remove global token and use it for modulation
        global_info, x = x[:, :1], x[:, 1:]
        x = x + 0.1 * global_info
        
        x = self.final_norm(x)
        return SimpleNamespace(last_hidden_state=x)


# ---------- lightweight Simple backbone (unchanged) -------------------------
class SimpleFlowBackbone(nn.Module):
    def __init__(self, hidden: int = 320, kernel_size: int = 5, dropout: float = 0.1, **kwargs):
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
    'vectorf': VectorFieldBackbone,  # Optimized Mamba-inspired architecture
}


# ---------- Main Model: FM3 --------------------------------------------------
# fm3.py - Updated FM3 class initialization
class FM3(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1162,
        hidden: int = 320,
        backbone_type: str = 'simple',
        eps: float = 1e-5,
        dropout: float = 0.1,  # Add this parameter
        **backbone_kwargs
    ):
        super().__init__()
        if backbone_type not in BACKBONES:
            raise ValueError(f"Backbone '{backbone_type}' not supported. Choose from {list(BACKBONES.keys())}")
        
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.backbone_type = backbone_type
        self.eps = eps
        self.dropout = dropout  # Store dropout as instance attribute

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
        
        # Improved vector field head for regression
        self.v_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Use the dropout parameter
            nn.Linear(hidden * 2, hidden),
        )
        # Initialize with small weights for stable flow
        for layer in self.v_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.02)
                nn.init.zeros_(layer.bias)
        
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
            'dropout': self.dropout,  # Now this will work
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