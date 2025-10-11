#flow.py
# -------- Scheduler ------------------
# Simplified from Meta's Implementation
#
# -------------------------------------
import torch
from torch import nn
import math

# -------- Scheduler --------
class PolynomialConvexScheduler:
    def __init__(self, n: float = 2.0):
        self.n = n
    def __call__(self, t: torch.Tensor):
        alpha_t = t.pow(self.n)
        d_alpha_t = self.n * t.pow(self.n - 1)
        return type("SchedOut", (), {"alpha_t": alpha_t, "d_alpha_t": d_alpha_t})


class CosineSmoothScheduler:
    """
    Smooth, bounded α(t) for stable flow matching.
    α(t) = 0.5 * (1 - cos(pi * t))
    dα/dt = 0.5 * pi * sin(pi * t)
    Range: α ∈ [0, 1], dα bounded ∈ [0, 1.57]
    """
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, t: torch.Tensor):
        t = torch.clamp(t, self.eps, 1 - self.eps)
        alpha_t = 0.5 * (1 - torch.cos(math.pi * t))
        d_alpha_t = 0.5 * math.pi * torch.sin(math.pi * t)
        return type("SchedOut", (), {"alpha_t": alpha_t, "d_alpha_t": d_alpha_t})


# -------- Source Distributions --------
class MaskedSourceDistribution:
    def __init__(self, mask_token: int): self.mask_token = mask_token
    def sample_like(self, x: torch.Tensor): return torch.full_like(x, self.mask_token)

class UniformSourceDistribution:
    def __init__(self, vocab_size: int): self.vocab_size = vocab_size
    def sample_like(self, x: torch.Tensor):
        return torch.randint_like(x, high=self.vocab_size)

# -------- Mixture Path --------
class MixtureDiscreteProbPath:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(self, t, x_0, x_1):
        # x_0, x_1: [B, L] long
        alpha = self.scheduler(t).alpha_t[:, None]
        bern = torch.rand_like(x_1.float())
        choose_x1 = bern < alpha
        x_t = torch.where(choose_x1, x_1, x_0)
        return type("SampleOut", (), {"x_t": x_t.long(), "t": t})


# -------- Builders --------
def get_path(scheduler_type="cosine", exponent=2.0):
    if scheduler_type == "cosine":
        return MixtureDiscreteProbPath(CosineSmoothScheduler())
    elif scheduler_type == "polynomial":
        return MixtureDiscreteProbPath(PolynomialConvexScheduler(exponent))
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

def get_source_distribution(source_distribution: str, vocab_size: int):
    if source_distribution == "mask":
        return MaskedSourceDistribution(mask_token=vocab_size - 1)
    elif source_distribution == "uniform":
        return UniformSourceDistribution(vocab_size=vocab_size)


def get_loss_function(name, path=None):
    from fm3.loss import MixturePathGeneralizedKL, StableMixturePathKL
    if name == "stable_kl":
        return StableMixturePathKL(path)
    elif name == "generalized_kl":
        return MixturePathGeneralizedKL(path)
    raise ValueError(f"Unknown loss {name}")
