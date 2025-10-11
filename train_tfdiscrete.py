import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tqdm, os, random
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from types import SimpleNamespace

from fm3.flow import get_path, get_source_distribution, get_loss_function
from fm3.fm3 import FM3
from fm3.tokenizer import FastChemTokenizerSelfies
from fm3.training import step

# ---------------- Dataset ----------------
class SelfiesDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        df = pd.read_csv(csv_path)
        assert "SELFIES" in df.columns
        self.samples = df["SELFIES"].astype(str).tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        ids = self.tok.encode(seq, add_special_tokens=False)[:self.max_len-2]
        ids = [self.tok.bos_token_id] + ids + [self.tok.eos_token_id]
        ids += [self.tok.pad_token_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

# ---------------- Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = FastChemTokenizerSelfies.from_pretrained("./tokenizer_vocab/selftok_reordered")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = SelfiesDataset("./data/test.csv", tokenizer, max_len=64)
dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

path = get_path("cosine")
source = get_source_distribution("mask", vocab_size=len(tokenizer))
loss_fn = get_loss_function("stable_kl", path)

model = FM3(vocab_size=len(tokenizer), cond_dim=128, n_layers=4, n_heads=4, backbone_type="tf").to(device)
opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.amp.GradScaler()

# ---------------- Training ----------------
max_steps = int(len(dl) * 1)
print("🚀 Starting discrete flow training for FM3")

iterator = iter(dl)

for step_idx in tqdm.trange(max_steps):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(dl)
        batch = next(iterator)

    x_1 = batch.to(device)
    x_0 = source.sample_like(x_1)
    t = torch.rand(x_1.shape[0], device=device)
    x_t = path.sample(t, x_0, x_1).x_t

    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        logits = model(x_t, t, return_logits=True)
        loss = loss_fn(logits, x_1, x_t, t)

    opt.zero_grad()
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    if step_idx % 100 == 0:
        print(f"[{step_idx}] loss = {loss.item():.4f}")

torch.save(model.state_dict(), "fm3_discrete.pt")
print("✅ Done. Saved to fm3_discrete.pt")
