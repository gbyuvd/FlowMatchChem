#training.py
# ------------ Training ---------------
# Simplified from Meta's Implementation
# Discrete Training Step
# -------------------------------------
import torch
from torch.cuda.amp import autocast

def step(state, loss_fn, path, scaler, iterator, device,
         source_distribution, logger, training=True, optim_params=None, time_epsilon=0.0):

    model, optimizer = state.model, state.optimizer
    model.train() if training else model.eval()

    batch = next(iterator)
    x_1 = batch["input_ids"].to(device)

    with torch.no_grad():
        x_0 = source_distribution.sample_like(x_1)
        t = torch.rand(x_1.shape[0], device=device) * (1.0 - time_epsilon)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    with autocast(enabled=True):
        logits = model(x_t=path_sample.x_t, time=path_sample.t)
        loss = loss_fn(logits, x_1, path_sample.x_t, path_sample.t)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.detach()
