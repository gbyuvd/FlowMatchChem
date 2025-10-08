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

