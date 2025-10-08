# Flow Matching for Molecules (FM3)

FM3 learns an Optimal-Transport vector field that pushes pure Gaussian noise into realistic 1D molecular embeddings (SELFIES/SMILES).
The field is parameterized by a Mamba-inspired multi-scale CNN that regresses the instantaneous velocity vθ(xt,t).

| File                       | What it does                                      |
| -------------------------- | ------------------------------------------------- |
| `fm3.py`                   | full model + schedulers + backbones               |
| `backbones.vectorf`        | Mamba-style selective mixing + dilated gated CNNs |
| `backbones.simple`         | 2-layer CNN baseline (fast, interpretable)        |
| `OptimalTransportSchedule` | closed-form α(t)=t, σ(t)=1–t, dα=1, dσ=–1         |
| `TimeAdditiveEmbedder`     | sinusoidal → MLP → residual (no param explosion)  |

# Early Test Result on Small Data (24K)
trained for 10 'epochs' (steps = 10 * len(dataloader)) using discrete trainer:
```
NUM_SAMPLES = 6
MAX_LEN = 12
STEPS = 200
MODE = "discrete"  # choose: "continuous" or "discrete"
TEMPERATURE = 0.35
```

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/8d4f3261-41f2-4387-bcf3-9226ef96cde3" />


