import torch
from torch.optim import AdamW
import torch.nn.functional as F
from math import prod

class AdamWF(AdamW):
    """
    AdamW + Variance-Adaptive FlowNorm for 2D/3D param tensors — Gaussian kernel only.
    Compat with PyTorch and HuggingFace Trainer.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        flow_alpha=0.5,
        flow_strength=1e-4,
        kernel_size: int = 3,
        apply_interval: int = 2,
        var_beta: float = 0.99,
        var_eps: float = 1e-8,
        alpha_min: float = 0.1,
        alpha_max: float = 1.0,
        kernel_type: str = "gaussian",
        bilateral_tau: float = 0.1,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # --- FlowNorm hyperparameters ---
        self.flow_alpha = float(flow_alpha)
        self.flow_strength = float(flow_strength)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else (kernel_size | 1)
        self.apply_interval = int(apply_interval)
        self.var_beta = float(var_beta)
        self.var_eps = float(var_eps)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.kernel_type = str(kernel_type).lower()
        self.bilateral_tau = float(bilateral_tau)
        self._step_counter = 0

        # --- Store constructor defaults (for HF Trainer) ---
        self.defaults.update(dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            flow_alpha=flow_alpha,
            flow_strength=flow_strength,
            kernel_size=kernel_size,
            apply_interval=apply_interval,
            var_beta=var_beta,
            var_eps=var_eps,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            kernel_type=kernel_type,
            bilateral_tau=bilateral_tau,
        ))

        # --- Precompute gaussian kernels for reuse ---
        self._gauss_cache = {}
        self._gauss_cache[(2, self.kernel_size)] = self._make_gaussian_kernel(2, self.kernel_size)
        self._gauss_cache[(3, self.kernel_size)] = self._make_gaussian_kernel(3, self.kernel_size)

    def _make_gaussian_kernel(self, ndim: int, size: int, sigma: float = None):
        if sigma is None:
            sigma = max(0.5, size / 3.0)
        coords = torch.stack(torch.meshgrid(*[torch.arange(size)] * ndim, indexing="ij"), dim=-1).float()
        center = (size - 1) / 2.0
        dist2 = ((coords - center) ** 2).sum(dim=-1)
        g = torch.exp(-dist2 / (2 * sigma * sigma))
        g /= g.sum()
        return g

    def _get_gauss_cached(self, spatial_ndim: int, ksize: int):
        key = (spatial_ndim, ksize)
        if key not in self._gauss_cache:
            self._gauss_cache[key] = self._make_gaussian_kernel(spatial_ndim, ksize)
        return self._gauss_cache[key]

    def _get_kernel(self, spatial_ndim: int, device, dtype, ksize: int):
        g = self._get_gauss_cached(spatial_ndim, ksize)
        return g.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def _smooth_squared(self, g2: torch.Tensor):
        dims = g2.ndim
        if dims < 2:
            return g2
        spatial_ndim = 3 if dims >= 5 else 2
        ksize = self.kernel_size
        spatial = g2.shape[-spatial_ndim:]
        leading = g2.shape[:-spatial_ndim]
        batch = int(prod(leading)) if len(leading) > 0 else 1

        x = g2.reshape(batch, 1, *spatial)
        pad = ksize // 2
        k = self._get_kernel(spatial_ndim, g2.device, g2.dtype, ksize)
        sm = F.conv3d(x, k, padding=pad) if spatial_ndim == 3 else F.conv2d(x, k, padding=pad)
        return sm.reshape(*g2.shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_counter += 1
        apply_smooth = (self._step_counter % self.apply_interval == 0)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                if g.ndim < 2 or not apply_smooth:
                    continue

                state = self.state[p]
                if "exp_avg" not in state:
                    state["step"] = torch.tensor(0.0, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if group.get("amsgrad", False):
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                # --- FlowNorm smoothing ---
                g2 = g ** 2
                sm2 = self._smooth_squared(g2)
                v_prev = state.get("var_ema", torch.zeros_like(sm2))
                v_new = self.var_beta * v_prev + (1 - self.var_beta) * (sm2 - g2).abs()
                state["var_ema"] = v_new

                var_ratio = (v_new / (v_new.mean() + self.var_eps)).clamp(0, 10)
                adaptive_alpha = (self.flow_alpha * (var_ratio / (1 + var_ratio))).mean().clamp(
                    self.alpha_min, self.alpha_max
                )

                eps = 1e-12
                ratio = torch.sqrt((sm2 + eps) / (g2 + eps))
                smoothed_grad = g * ratio
                new_grad = (1.0 - adaptive_alpha) * g + adaptive_alpha * smoothed_grad
                reg = self.flow_strength * (sm2 - g2).abs().mean()
                p.grad.data = new_grad + new_grad * reg

        return super().step(closure)

    def state_dict(self):
        sd = super().state_dict()
        sd.update({
            "_step_counter": self._step_counter,
            "_kernel_size": self.kernel_size,
            "_flow_alpha": self.flow_alpha,
            "_flow_strength": self.flow_strength,
            "_apply_interval": self.apply_interval,
            "_var_beta": self.var_beta,
            "_alpha_range": (self.alpha_min, self.alpha_max),
            "_kernel_type": self.kernel_type,
        })
        return sd

    def load_state_dict(self, sd):
        self.kernel_size = int(sd.get("_kernel_size", self.kernel_size))
        self.flow_alpha = float(sd.get("_flow_alpha", self.flow_alpha))
        self.flow_strength = float(sd.get("_flow_strength", self.flow_strength))
        self.apply_interval = int(sd.get("_apply_interval", self.apply_interval))
        self._step_counter = int(sd.get("_step_counter", self._step_counter))
        self.var_beta = float(sd.get("_var_beta", self.var_beta))
        self.alpha_min, self.alpha_max = sd.get("_alpha_range", (self.alpha_min, self.alpha_max))
        self.kernel_type = sd.get("_kernel_type", self.kernel_type)
        super().load_state_dict({k: v for k, v in sd.items() if not k.startswith("_")})
