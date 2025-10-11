#transformer_backbone.py
# ------------ Utils ------------------
# Utilities for FM3: sampling, visualization, trajectory analysis
# -------------------------------------
import torch
import torch.nn.functional as F
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from IPython.display import display

# ------------------- GENERATION -------------------

@torch.no_grad()
def sample_molecules(
    model,
    tokenizer,
    num_samples=8,
    seq_len=64,
    steps=100,
    mode="discrete",
    temperature=1.0,
    device=None,
):
    """
    Generate molecules from a trained FM3 model.

    Args:
        model: trained FM3 instance
        tokenizer: FastChemTokenizerSelfies or compatible tokenizer
        num_samples: number of molecules to sample
        seq_len: sequence length
        steps: number of integration/sampling steps
        mode: "discrete" or "continuous"
        temperature: softmax temperature for discrete sampling
        device: torch.device (default: model's device)
    Returns:
        torch.LongTensor of shape [num_samples, seq_len]
    """
    device = device or next(model.parameters()).device
    model = model.to(device).eval()

    print(f"🧪 Sampling {num_samples} molecules via FM3 ({mode} mode)...")
    tokens = model.generate(
        tokenizer=tokenizer,
        num_samples=num_samples,
        seq_len=seq_len,
        steps=steps,
        mode=mode,
        temperature=temperature,
    )
    return tokens.cpu()


# ------------------- POSTPROCESS & VISUALIZE -------------------

def tokens_to_selfies(tok_list, tokenizer):
    """Convert a list of token IDs to a clean SELFIES string."""
    clean = []
    for t in tok_list:
        if t == tokenizer.eos_token_id:
            break
        if t in (tokenizer.pad_token_id, tokenizer.bos_token_id):
            continue
        clean.append(int(t))
    if not clean:
        return ""
    try:
        return tokenizer.decode(clean, skip_special_tokens=True)
    except Exception:
        return " ".join(map(str, clean))


def visualize_generated(tokens, tokenizer):
    """
    Convert token samples to SELFIES → SMILES → molecules and display grid.
    """
    generated_SELFIES, generated_mols = [], []

    for i in range(tokens.shape[0]):
        tok_list = tokens[i].cpu().numpy().tolist()
        pred_selfies = tokens_to_selfies(tok_list, tokenizer)
        pred_selfies_clean = pred_selfies.replace(" ", "").strip()

        if not pred_selfies_clean:
            print(f"Sample {i+1}: ❌ Empty SELFIES.")
            continue

        try:
            smi = sf.decoder(pred_selfies_clean)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Sample {i+1}: invalid SMILES '{smi}'")
                continue

            smi_canon = Chem.MolToSmiles(mol)
            generated_SELFIES.append(smi_canon)
            generated_mols.append(mol)
            print(f"✅ Sample {i+1}: {smi_canon}")

        except Exception as e:
            print(f"Sample {i+1}: Decoding error → {e}")

    if generated_mols:
        legends = [f"Sample {i+1}\n{smi}" for i, smi in enumerate(generated_SELFIES)]
        img = Draw.MolsToGridImage(
            generated_mols,
            molsPerRow=min(3, len(generated_mols)),
            subImgSize=(300, 300),
            legends=legends,
            useSVG=False
        )
        display(img)
        print(f"\n✓ Generated {len(generated_SELFIES)}/{tokens.shape[0]} valid molecules")
    else:
        print("⚠ No valid molecules generated.")

    return generated_mols, generated_SELFIES


# ------------------- CONTINUOUS TRAJECTORY INSPECTION -------------------

@torch.no_grad()
def analyze_continuous_trajectory(model, max_len=64, steps=50, noise_scale=0.1, device=None):
    """
    Analyze average embedding statistics along a continuous FM3 flow trajectory.
    """
    import numpy as np

    device = device or next(model.parameters()).device
    model = model.to(device).eval()

    print("\nAnalyzing continuous trajectory stats...")
    try:
        x0 = torch.randn(1, max_len, model.hidden, device=device) * noise_scale
        t_dense = torch.linspace(0.0, 1.0, steps, device=device)
        traj = [x0]

        for t in t_dense[1:]:
            v = model(traj[-1], t.repeat(1))
            traj.append(traj[-1] + v * (t_dense[1] - t_dense[0]))

        traj = torch.stack([x.detach() for x in traj])
        norms = traj.norm(dim=-1).mean(dim=(1, 2)).cpu()
        stds = traj.std(dim=-1).mean(dim=(1, 2)).cpu()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t_dense.cpu().numpy(), norms.numpy())
        plt.xlabel('t'); plt.ylabel('Avg embedding norm'); plt.title('Norm vs t'); plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(t_dense.cpu().numpy(), stds.numpy())
        plt.xlabel('t'); plt.ylabel('Avg embedding std'); plt.title('Std vs t'); plt.grid(True)

        plt.tight_layout()
        plt.show()

        print(f"Initial norm: {norms[0]:.4f}, Final norm: {norms[-1]:.4f} (Δ {float(norms[-1]-norms[0]):.4f})")

    except Exception as e:
        print(f"Trajectory analysis failed: {e}")
