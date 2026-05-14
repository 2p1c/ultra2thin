"""Full training for v2 point-wise thickness predictor."""

import os, sys, random
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from simulation_v2 import generate_point_dataset
from network_v2 import PointThicknessPredictor

torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_variable(batch):
    max_n = max(s[0].shape[0] for s in batch)
    T = batch[0][0].shape[1]
    B = len(batch)
    sigs = torch.zeros(B, max_n, T)
    pos = torch.zeros(B, max_n, 2)
    thick = torch.zeros(B, max_n)
    src = torch.zeros(B, 2)
    mask = torch.ones(B, max_n, dtype=torch.bool)
    for i, (s, p, t, sp, _) in enumerate(batch):
        n = s.shape[0]
        sigs[i,:n] = torch.from_numpy(s)
        pos[i,:n] = torch.from_numpy(p)
        thick[i,:n] = torch.from_numpy(t)
        src[i] = torch.from_numpy(sp)
        mask[i,:n] = False
    return sigs, pos, thick, src, mask


@torch.no_grad()
def validate(model, loader, crit):
    model.eval()
    total, n = 0.0, 0
    for sigs, pos, thick, src, mask in loader:
        sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
        thick, src = thick.to(DEVICE), src.to(DEVICE)
        mask = mask.to(DEVICE)
        pred = model(sigs, pos, src, mask)
        loss = crit(pred[~mask], thick[~mask])
        nv = (~mask).sum().item()
        total += loss.item() * nv; n += nv
    return total / max(n, 1)


def main():
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)

    # ── Hyperparams ─────────────────────────────────────────────────
    n_train = 4000
    n_val = 500
    d_model = 96
    n_heads = 4
    n_layers = 3
    batch_size = 16
    lr_phase1 = 2e-3
    epochs_p1 = 40
    lr_phase2 = 3e-4
    epochs_p2 = 30

    # ── Data ────────────────────────────────────────────────────────
    print(f"\nGenerating {n_train} training + {n_val} validation samples...", flush=True)
    (s1, p1, t1, sp1, _) = generate_point_dataset(n_train, 9, 64)
    (s2, p2, t2, sp2, _) = generate_point_dataset(n_val, 9, 64)
    bl = float(np.mean((np.concatenate([x for x in t2]) - 2.0) ** 2))
    print(f"Baseline MSE: {bl:.6f}", flush=True)

    tr_samples = list(zip(s1, p1, t1, sp1, [None]*len(s1)))
    val_samples = list(zip(s2, p2, t2, sp2, [None]*len(s2)))
    tr_loader = DataLoader(tr_samples, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_variable)
    val_loader = DataLoader(val_samples, batch_size=batch_size,
                            collate_fn=collate_variable)

    # ── Model ───────────────────────────────────────────────────────
    model = PointThicknessPredictor(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    crit = nn.MSELoss()
    best_loss = float("inf")
    best_state = None

    # ── Phase 1 ─────────────────────────────────────────────────────
    print(f"\n=== Phase 1 ({epochs_p1} epochs, lr={lr_phase1}) ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_phase1, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr_phase1, steps_per_epoch=len(tr_loader), epochs=epochs_p1,
        pct_start=0.08, div_factor=25.0, final_div_factor=1000.0)

    for ep in range(1, epochs_p1 + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        for sigs, pos, thick, src, mask in tr_loader:
            sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
            thick, src = thick.to(DEVICE), src.to(DEVICE)
            mask = mask.to(DEVICE)
            pred = model(sigs, pos, src, mask)
            loss = crit(pred[~mask], thick[~mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            nv = (~mask).sum().item()
            tr_loss += loss.item() * nv; n_tr += nv
        val_loss = validate(model, val_loader, crit)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 8 == 0 or ep == 1:
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  best={best_loss:.6f}", flush=True)
    print(f"Phase 1 best: {best_loss:.6f}  (+{(bl-best_loss)/bl*100:.1f}%)", flush=True)

    # ── Phase 2 ─────────────────────────────────────────────────────
    print(f"\n=== Phase 2 ({epochs_p2} epochs, lr={lr_phase2}) ===", flush=True)
    model.load_state_dict(best_state)  # resume from best
    opt = torch.optim.AdamW(model.parameters(), lr=lr_phase2, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_p2)

    for ep in range(1, epochs_p2 + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        for sigs, pos, thick, src, mask in tr_loader:
            sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
            thick, src = thick.to(DEVICE), src.to(DEVICE)
            mask = mask.to(DEVICE)
            pred = model(sigs, pos, src, mask)
            loss = crit(pred[~mask], thick[~mask])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            nv = (~mask).sum().item()
            tr_loss += loss.item() * nv; n_tr += nv
        val_loss = validate(model, val_loader, crit)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 6 == 0 or ep == 1:
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  best={best_loss:.6f}", flush=True)
    print(f"Phase 2 best: {best_loss:.6f}  (+{(bl-best_loss)/bl*100:.1f}%)", flush=True)

    # ── Save ────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": best_state,
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
        "val_loss": best_loss,
    }, "checkpoints/model_v2.pt")
    print(f"\nSaved checkpoints/model_v2.pt  |  Best val loss: {best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
