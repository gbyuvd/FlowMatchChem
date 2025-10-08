# Flow Matching for Molecules (FM3)

FM3 learns an Optimal-Transport vector field that pushes pure Gaussian noise into realistic 1D molecular embeddings (SELFIES/SMILES).
The field is parameterized by a Mamba-inspired multi-scale CNN that regresses the instantaneous velocity vθ(xt,t).

For the 2.1M curated and filtered data from ChemBL34, COCONUTDB, and SuperNatural3, you can e-mail me gbyuvd@proton.me

| File                       | What it does                                      |
| -------------------------- | ------------------------------------------------- |
| `fm3.py`                   | full model + schedulers + backbones               |
| `backbones.vectorf`        | Mamba-style selective mixing + dilated gated CNNs |
| `backbones.simple`         | 2-layer CNN baseline (fast, interpretable)        |
| `OptimalTransportSchedule` | closed-form α(t)=t, σ(t)=1–t, dα=1, dσ=–1         |
| `TimeAdditiveEmbedder`     | sinusoidal → MLP → residual (no param explosion)  |

# Early Test Result on Small Data (24K)
trained for 10 'epochs' (steps = 10 * len(dataloader)) using discrete trainer and simple backbone:
```
NUM_SAMPLES = 6
MAX_LEN = 12
STEPS = 200
MODE = "discrete"  # choose: "continuous" or "discrete"
TEMPERATURE = 0.35
```

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/8d4f3261-41f2-4387-bcf3-9226ef96cde3" />

# References
```bibtex
@misc{lipman2024flowmatchingguidecode,
      title={Flow Matching Guide and Code}, 
      author={Yaron Lipman and Marton Havasi and Peter Holderrieth and Neta Shaul and Matt Le and Brian Karrer and Ricky T. Q. Chen and David Lopez-Paz and Heli Ben-Hamu and Itai Gat},
      year={2024},
      eprint={2412.06264},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.06264}, 
}

```

```bibtex
@misc{lipman2023flowmatchinggenerativemodeling,
      title={Flow Matching for Generative Modeling}, 
      author={Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matt Le},
      year={2023},
      eprint={2210.02747},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2210.02747}, 
}
```
