"""
Numerically-stable Chebyshev LR scheduler implemented as a torch _LRScheduler subclass.
Maps global_step in [0, total_steps] to a smooth interpolation between lr_start and lr_end
using Chebyshev polynomials via the stable cosine identity.
"""
from typing import Optional
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class ChebyshevLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        lr_start: float = 1.0,
        lr_end: float = 0.0,
        last_epoch: int = -1,
        min_lr: float = 1e-12,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0 for ChebyshevLR")
        self.total_steps = int(total_steps)
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def _alpha_at_step(self, step: int) -> float:
        # t in [0,1]
        t = min(max(float(step) / float(max(1, self.total_steps)), 0.0), 1.0)
        # map to x in [1, -1] (start -> end)
        x = 1.0 - 2.0 * t
        # stable Chebyshev evaluation:
        # for |x| <= 1: T_n(x) = cos(n * arccos(x))
        n = self.total_steps
        if abs(x) <= 1.0:
            theta = math.acos(x)
            Tn = math.cos(n * theta)
        else:
            # numerical fallback (should rarely happen)
            x64 = float(x)
            T0 = 1.0
            T1 = x64
            for _ in range(2, n + 1):
                T2 = 2.0 * x64 * T1 - T0
                T0, T1 = T1, T2
            Tn = T1
        # map Tn from [-1,1] to [0,1]
        alpha = 0.5 * (Tn + 1.0)
        # clamp
        return min(max(alpha, 0.0), 1.0)

    def get_lr(self):
        step = max(0, self.last_epoch)
        alpha = self._alpha_at_step(step)
        out = []
        for base_lr in self.base_lrs:
            lr = float(self.lr_end) + (float(self.lr_start) - float(self.lr_end)) * alpha
            if lr < self.min_lr:
                lr = self.min_lr
            out.append(lr)
        return out

    @staticmethod
    def total_steps_from(epochs: int, steps_per_epoch: int) -> int:
        return int(max(1, int(epochs) * int(steps_per_epoch)))
