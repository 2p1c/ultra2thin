"""Train v3 hybrid model — flexible input + U-Net decoder."""

import os, sys, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from simulation_v2 import generate_point_dataset
from network_v3 import HybridThicknessPredictor

torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_variable(batch):
    max_n = max(s[0].shape[0] for s in batch)
    T = batch[0][0].shape[1]
    B = len(batch)
    sigs = torch.zeros(B, max_n, T)
    pos = torch.zeros(B, max_n, 2)
    src = torch.zeros(B, 2)
    mask = torch.ones(B, max_n, dtype=torch.bool)
    tmaps = torch.zeros(B, 32, 32)
    for i, (s, p, t, sp, tmap) in enumerate(batch):
        n = s.shape[0]
        sigs[i,:n] = torch.from_numpy(s)
        pos[i,:n] = torch.from_numpy(p)
        src[i] = torch.from_numpy(sp)
        mask[i,:n] = False
        tmaps[i] = torch.from_numpy(tmap)
    return sigs, pos, src, mask, tmaps


@torch.no_grad()
def validate(model, loader, crit):
    model.eval()
    total, n = 0.0, 0
    for sigs, pos, src, mask, tmaps in loader:
        sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
        src = src.to(DEVICE); mask = mask.to(DEVICE)
        tmaps = tmaps.to(DEVICE)
        pred = model(sigs, pos, src, mask)
        loss = crit(pred, tmaps)
        total += loss.item() * sigs.size(0); n += sigs.size(0)
    return total / max(n, 1)


def main():
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)

    n_train, n_val = 4000, 500
    d_model, n_heads, n_layers = 128, 4, 3
    batch_size = 16

    print(f"\nGenerating {n_train} training + {n_val} validation samples...", flush=True)
    tr_data = generate_point_dataset(n_train, 9, 64)
    val_data = generate_point_dataset(n_val, 9, 64)
    tr_samples = list(zip(*tr_data))
    val_samples = list(zip(*val_data))
    tr_loader = DataLoader(tr_samples, batch_size=batch_size, shuffle=True, collate_fn=collate_variable)
    val_loader = DataLoader(val_samples, batch_size=batch_size, collate_fn=collate_variable)

    # Baseline: predict 2.0 everywhere
    bl = float(np.mean((np.stack([t for t in val_data[4]]) - 2.0) ** 2))
    print(f"Baseline MSE: {bl:.6f}", flush=True)

    model = HybridThicknessPredictor(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    crit = nn.MSELoss()
    best_loss, best_state = float("inf"), None

    # ── Phase 1 ─────────────────────────────────────────────────────
    print(f"\n=== Phase 1 (40 epochs, lr=2e-3) ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, steps_per_epoch=len(tr_loader), epochs=40,
        pct_start=0.08, div_factor=25.0, final_div_factor=1000.0)

    for ep in range(1, 41):
        model.train(); tr_loss, n_tr = 0.0, 0
        for sigs, pos, src, mask, tmaps in tr_loader:
            sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
            src = src.to(DEVICE); mask = mask.to(DEVICE)
            tmaps = tmaps.to(DEVICE)
            pred = model(sigs, pos, src, mask)
            loss = crit(pred, tmaps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tr_loss += loss.item() * sigs.size(0); n_tr += sigs.size(0)
        val_loss = validate(model, val_loader, crit)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 8 == 0 or ep == 1:
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  best={best_loss:.6f}", flush=True)
    print(f"Phase 1 best: {best_loss:.6f}  (+{(bl-best_loss)/bl*100:.1f}%)", flush=True)

    # ── Phase 2 ─────────────────────────────────────────────────────
    print(f"\n=== Phase 2 (30 epochs, lr=3e-4) ===", flush=True)
    model.load_state_dict(best_state)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    for ep in range(1, 31):
        model.train(); tr_loss, n_tr = 0.0, 0
        for sigs, pos, src, mask, tmaps in tr_loader:
            sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
            src = src.to(DEVICE); mask = mask.to(DEVICE)
            tmaps = tmaps.to(DEVICE)
            pred = model(sigs, pos, src, mask)
            loss = crit(pred, tmaps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tr_loss += loss.item() * sigs.size(0); n_tr += sigs.size(0)
        val_loss = validate(model, val_loader, crit)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 6 == 0 or ep == 1:
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  best={best_loss:.6f}", flush=True)
    print(f"Phase 2 best: {best_loss:.6f}  (+{(bl-best_loss)/bl*100:.1f}%)", flush=True)

    torch.save({"model_state_dict": best_state, "d_model": d_model, "n_heads": n_heads,
                "n_layers": n_layers, "val_loss": best_loss}, "checkpoints/model_v3.pt")
    print(f"\nSaved checkpoints/model_v3.pt  |  Best val: {best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
