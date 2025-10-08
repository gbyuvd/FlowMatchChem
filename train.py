from FastChemTokenizerHF import FastChemTokenizerSelfies
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm, random, pandas as pd, json, copy
from fm3 import FM3
# -----------------------------------------------------------------------------
# 1. Tokenizer
# -----------------------------------------------------------------------------
tokenizer = FastChemTokenizerSelfies.from_pretrained("./selftok_reordered")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------------------------------------------------
# 2. Dataset & Dataloader
# -----------------------------------------------------------------------------
class CFMSelfiesSingleDS(Dataset):
    """
    Unconditional CFM dataset from a single-column CSV.
    Returns a single encoded sequence `tok_x1` (1-D LongTensor with BOS/EOS/pad).
    """
    def __init__(self, csv_path: str, tokenizer, max_len: int = 128):
        df = pd.read_csv(csv_path)
        if "SELFIES" not in df.columns:
            raise KeyError("CSV must contain a column named 'SELFIES'")
        self.mols = df["SELFIES"].astype(str).tolist()
        self.tok = tokenizer
        self.max_l = max_len

    def __len__(self):
        return len(self.mols)

    def _encode(self, selfie: str) -> torch.Tensor:
        ids = self.tok.encode(selfie, add_special_tokens=False)
        ids = ids[:self.max_l - 2]
        full = [self.tok.bos_token_id] + ids + [self.tok.eos_token_id]
        full += [self.tok.pad_token_id] * (self.max_l - len(full))
        return torch.tensor(full, dtype=torch.long)

    def __getitem__(self, idx):
        return self._encode(self.mols[idx])


def cfm_single_collate_fn(batch):
    """
    Collate function for CFMSelfiesSingleDS.
    Pads variable-length sequences in a batch to the same length.
    """
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)  # or tokenizer.pad_token_id
    return padded_batch

# -----------------------------------------------------------------------------
# 2. Dataset & Dataloader
# -----------------------------------------------------------------------------
print("Loading data ....")
ds = CFMSelfiesSingleDS("./data/test.csv", tokenizer, max_len=64)

subset_size = int(len(ds) * 0.01)
indices = np.random.choice(len(ds), subset_size, replace=False)
subset_ds = Subset(ds, indices)

dl = DataLoader(
    subset_ds,
    batch_size=64,
    shuffle=True,
    collate_fn=cfm_single_collate_fn
)

# -----------------------------------------------------------------------------
# 3. Model & Optimizer
# -----------------------------------------------------------------------------
print("Init model ....")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FM3(
    vocab_size=len(tokenizer),
    hidden=320,
    backbone_type='vectorf',
    n_layers=4,
    dropout=0.1,
    kernel_size=7,
    expand_ratio=2.0,
    use_multiscale=True
).to(device)

optim = AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
max_grad_norm = 1.0

# Create EMA copy
ema = copy.deepcopy(model)
for p in ema.parameters():
    p.requires_grad_(False)
ema_decay = 0.999
def update_ema(model, ema, decay=ema_decay):
    with torch.no_grad():
        msd, esd = model.state_dict(), ema.state_dict()
        for k in msd.keys():
            esd[k].mul_(decay).add_(msd[k], alpha=1 - decay)

TEST_SEL = random.Random(42).sample(pd.read_csv("test.csv")["SELFIES"].tolist(), 3)

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------
losses, grad_norms = [], []
model.train()

print("Starting training ....")
max_steps = 3 * len(subset_ds) # 3 'epochs'
pbar = tqdm.trange(max_steps, desc='CFM')

# Track best model based on loss 
best_loss = float('inf')  # Initialize to infinity
best_model_path = "best_cfm_model"
ema_model_path = "ema_cfm_model"

data_iter = iter(dl)
for step in pbar:
    try:
        tok_batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dl)
        tok_batch = next(data_iter)

    tok_batch = tok_batch.to(device)
    B, L = tok_batch.shape

    # Time sampling
    t = torch.rand(B, device=device)

    # Embeddings and interpolation
    emb_layer = model.get_input_embeddings()
    x1 = emb_layer(tok_batch)          # [B, L, H] data
    x0 = torch.randn_like(x1)          # noise
    x_t = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v_target = x1 - x0

    # Forward and compute flow loss
    v_pred = model(x_t, t)
    flow_loss = F.mse_loss(v_pred, v_target)

    # Combine losses
    loss = flow_loss 

    # Backpropagation
    optim.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
    optim.step()

    # EMA update
    update_ema(model, ema)

    grad_norms.append(grad_norm.item())
    losses.append(loss.item())
    pbar.set_postfix(loss=f"{loss.item():.4f}", grad_norm=f"{grad_norm:.2f}")

    # -------------------------------------------------------------------------
    # Diagnostics every 100 steps
    # -------------------------------------------------------------------------
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            # ---------- 1. snap metrics on the *same* batch ----------
            x1_flat = x1.view(-1, x1.size(-1))          # [B*L, H]
            embed_matrix = model.get_input_embeddings().weight  # [V, H]
            dists = torch.cdist(x1_flat, embed_matrix)  # [B*L, V]
            snap_ids = dists.argmin(dim=-1)             # [B*L]
            tgt_flat = tok_batch.view(-1)               # [B*L]
            mask = tgt_flat != tokenizer.pad_token_id
            snap_acc = (snap_ids[mask] == tgt_flat[mask]).float().mean().item()
            avg_dist = dists.min(dim=-1).values[mask].mean().item()

            # ---------- 2. classic generation with EMA (unchanged) ----------
            test_selfie = TEST_SEL[0]
            test_tok = tokenizer.encode(test_selfie, add_special_tokens=False)
            test_tok = [tokenizer.bos_token_id] + test_tok + [tokenizer.eos_token_id]
            orig_len = len(test_tok)
            max_len = getattr(model, "max_len", 64)
            pad_len = max_len - len(test_tok)
            if pad_len < 0:
                test_tok = test_tok[:max_len]
            else:
                test_tok = test_tok + [tokenizer.pad_token_id] * pad_len
            test_tok = torch.tensor([test_tok], dtype=torch.long, device=device)

            emb_ref = ema.get_input_embeddings()(test_tok)
            x0_noise = torch.randn_like(emb_ref)
            x1_hat = ema.sample_euler(x0_noise, steps=50)

            B, L, H = x1_hat.shape
            x_flat = x1_hat.view(B * L, H)
            embedding_weight = ema.get_input_embeddings().weight.to(x1_hat.device)
            dist = torch.cdist(x_flat, embedding_weight)
            tok_flat = dist.argmin(dim=-1)
            tok_pred = tok_flat.view(B, L)

            pred_ids = tok_pred[0].cpu().tolist()
            if tokenizer.eos_token_id in pred_ids:
                eos_pos = pred_ids.index(tokenizer.eos_token_id)
                pred_ids = pred_ids[:eos_pos]
            pred_ids = [tid for tid in pred_ids if tid not in
                        [tokenizer.pad_token_id, tokenizer.bos_token_id]]
            selfies_out = tokenizer.decode(pred_ids, skip_special_tokens=True)

            # ---------- 3. pretty print ----------
            print(f"\nStep {step}  |  loss {loss.item():.4f}  |  snap-acc {snap_acc:.2%}  |  avg-dist {avg_dist:.3f}")
            print(f"Sampled: {selfies_out}")
            
            # ---------- 4. Save best model based on loss (lower is better) ----------
            if loss.item() < best_loss:
                best_loss = loss.item()
                print(f"🎉 New best loss: {best_loss:.4f}, saving best model...")
                
                model.save_pretrained(best_model_path)

                # Save tokenizer with the model
                tokenizer.save_pretrained(best_model_path)
                
                # Also save the EMA model
                ema.save_pretrained(ema_model_path)
                tokenizer.save_pretrained(ema_model_path)

        model.train()

# -----------------------------------------------------------------------------
# 5. Final Diagnostics & Saving
# -----------------------------------------------------------------------------
print(f"\n📊 Final Results:")
print(f"Best loss achieved: {best_loss:.4f}")
print(f"Total steps: {len(losses)}")
print(f"Final loss: {losses[-1]:.4f}")

# Plot diagnostics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(losses)
ax1.set_title("CFM Training Loss (Flow + CE)")
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

ax2.plot(grad_norms)
ax2.axhline(y=max_grad_norm, color='r', linestyle='--', label='Clip threshold')
ax2.set_title("Gradient Norms")
ax2.set_xlabel("Step")
ax2.set_ylabel("Grad Norm")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_diagnostics.png", dpi=150, bbox_inches='tight')
plt.close()

# Save final models with decoder head
print(" Saving final models...")

# Save main model with decoder head
final_model_path = "final_cfm_model"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Save EMA model with decoder head  
final_ema_path = "final_ema_model"
ema.save_pretrained(final_ema_path)
tokenizer.save_pretrained(final_ema_path)

# Save training metadata
training_metadata = {
    'final_loss': losses[-1],
    'best_loss': best_loss,
    'total_steps': len(losses),
    'model_config': {
        'vocab_size': len(tokenizer),
        'hidden': 320,
        'backbone_type': 'vectorf',
        'n_layers': 4,
        'dropout': 0.1,
        'kernel_size': 7,
        'expand_ratio': 2.0,
        'use_multiscale': True
    },
    'training_config': {
        'learning_rate': 7e-4,
        'weight_decay': 1e-5,
        'max_grad_norm': 1.0,
        'ema_decay': ema_decay,
        'aux_loss_weight': 0.5
    },
    'tokenizer_config': {
        'pad_token': tokenizer.pad_token,
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token
    }
}

with open('training_metadata.json', 'w') as f:
    json.dump(training_metadata, f, indent=2)

print(f" Training complete!")
print(f"   • Best model saved to: {best_model_path}")
print(f"   • EMA model saved to: {ema_model_path}")
print(f"   • Final model saved to: {final_model_path}")
print(f"   • Final EMA saved to: {final_ema_path}")
print(f"   • Training metadata saved to: training_metadata.json")
print(f"   • Diagnostics plot saved to: training_diagnostics.png")

# -----------------------------------------------------------------------------
# 6. Model Loading Example (for verification)
# -----------------------------------------------------------------------------
print("\n🔍 Verifying saved models...")

# Load best model
try:
    loaded_model = FM3.from_pretrained(best_model_path)
    print(" Best model loaded successfully!")
except Exception as e:
    print(f" Error loading best model: {e}")

# Load EMA model  
try:
    loaded_ema = FM3.from_pretrained(ema_model_path)
    print(" EMA model loaded successfully!")
except Exception as e:
    print(f"Error loading EMA model: {e}")

print("\n All done! Your models are ready for inference.")