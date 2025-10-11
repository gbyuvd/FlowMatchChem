#loss.py
# ------------- Loss ------------------
# Simplified from Meta's Implementation
#
# -------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

class MixturePathGeneralizedKL(nn.Module):
    def __init__(self, path, reduction="mean"):
        super().__init__()
        self.path, self.reduction = path, reduction

    def forward(self, logits, x_1, x_t, t):
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()

        log_p_x1 = torch.gather(log_p, -1, x_1.unsqueeze(-1)).squeeze(-1)
        p_xt = torch.gather(p, -1, x_t.unsqueeze(-1)).squeeze(-1)
        delta = (x_t == x_1).float()

        sched = self.path.scheduler(t)
        weight = torch.clamp(sched.d_alpha_t / (1 - sched.alpha_t), max=10.0).unsqueeze(-1)

        loss = -weight * (p_xt - delta + (1 - delta) * log_p_x1)
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss


class StableMixturePathKL(nn.Module):
    """
    Numerically stable discrete flow-matching loss with cosine schedule.
    """
    def __init__(self, path, reduction="mean", max_weight=10.0, eps=1e-4):
        super().__init__()
        self.path = path
        self.reduction = reduction
        self.max_weight = max_weight
        self.eps = eps

    def forward(self, logits, x_1, x_t, t):
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()

        log_p_x1 = torch.gather(log_p, -1, x_1.unsqueeze(-1)).squeeze(-1)
        p_xt = torch.gather(p, -1, x_t.unsqueeze(-1)).squeeze(-1)
        delta = (x_t == x_1).float()

        sched = self.path.scheduler(t)
        w = sched.d_alpha_t / ((1 - sched.alpha_t) + self.eps)
        w = torch.clamp(w, max=self.max_weight).unsqueeze(-1)

        loss = -w * (p_xt - delta + (1 - delta) * log_p_x1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

