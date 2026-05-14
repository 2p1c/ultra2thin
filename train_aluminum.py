"""Train DenseThicknessPredictor on Lamb-wave dispersion data.

A0 Lamb wave propagation — dispersion c_p(f,d) encodes thickness along
source-receiver path. ~60x more thickness-sensitive than through-thickness TOF.
Per-point CNN+Fourier encoding → reshape → U-Net → 32x32 thickness map.
"""

import os, random, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from simulation_aluminum import (generate_terrace_map, D_NOMINAL, N_TIME, GRID_H, GRID_W)
from simulation_lamb import simulate_lamb_signals
from network_dense import DenseThicknessPredictor

torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_grid_positions(grid_h, grid_w):
    xs = np.linspace(0, GRID_W - 1, grid_w)
    ys = np.linspace(0, GRID_H - 1, grid_h)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)


def make_one_sample(rng, grid_h, grid_w):
    tmap = generate_terrace_map(rng=rng)
    sx = rng.uniform(0, GRID_W * 0.2)
    sy = rng.uniform(GRID_H * 0.1, GRID_H * 0.9)
    src = np.array([sx, sy], dtype=np.float32)
    pts = build_grid_positions(grid_h, grid_w)
    sigs, thick = simulate_lamb_signals(tmap, src, pts)
    return sigs, pts, thick, src, tmap


class PlateDataset:
    def __init__(self, n_samples, grid_h=32, grid_w=32, seed=0):
        self.n_samples = n_samples
        self.grid_h, self.grid_w = grid_h, grid_w
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        local_rng = np.random.RandomState(self.rng.randint(0, 2 ** 31 - 1) + idx)
        return make_one_sample(local_rng, self.grid_h, self.grid_w)


def collate_dense(batch):
    B = len(batch)
    N = batch[0][0].shape[0]
    T = batch[0][0].shape[1]
    sigs = torch.zeros(B, N, T)
    pos = torch.zeros(B, N, 2)
    src = torch.zeros(B, 2)
    tmaps = torch.zeros(B, 1, GRID_H, GRID_W)
    for i, (s, p, _, sp, tmap) in enumerate(batch):
        sigs[i] = torch.from_numpy(s)
        pos[i] = torch.from_numpy(p)
        src[i] = torch.from_numpy(sp)
        tmaps[i, 0] = torch.from_numpy(tmap)
    return sigs, pos, src, tmaps


def main():
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)

    # ── Hyperparams ──────────────────────────────────────────────────
    grid_h, grid_w = 32, 32     # 1024 dense measurement points
    d_model = 48
    n_train, n_val = 500, 120
    batch_size = 12
    epochs_p1, lr_p1 = 25, 2e-3
    epochs_p2, lr_p2 = 15, 2e-4

    N = grid_h * grid_w
    print(f"Grid: {grid_h}x{grid_w} = {N} pts  |  d_model={d_model}  |  "
          f"batch={batch_size}  |  train={n_train} val={n_val}", flush=True)

    # ── Data ─────────────────────────────────────────────────────────
    tr_ds = PlateDataset(n_train, grid_h, grid_w, seed=42)
    val_ds = PlateDataset(n_val, grid_h, grid_w, seed=123)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_dense)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            collate_fn=collate_dense)

    val_rng = np.random.RandomState(123)
    val_tmaps = []
    for i in range(n_val):
        local_rng = np.random.RandomState(val_rng.randint(0, 2 ** 31 - 1) + i)
        val_tmaps.append(generate_terrace_map(rng=local_rng))
    bl = float(np.mean((np.stack(val_tmaps) - D_NOMINAL) ** 2))
    print(f"Baseline MSE (predict {D_NOMINAL:.1f} mm): {bl:.6f}", flush=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = DenseThicknessPredictor(
        d_model=d_model, grid_h=grid_h, grid_w=grid_w,
        out_h=GRID_H, out_w=GRID_W).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    crit = nn.MSELoss()
    best_loss, best_state = float("inf"), None

    # ── Phase 1 ───────────────────────────────────────────────────────
    print(f"\n=== Phase 1 ({epochs_p1} epochs, lr={lr_p1}) ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_p1, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr_p1, steps_per_epoch=len(tr_loader), epochs=epochs_p1,
        pct_start=0.1, div_factor=25.0, final_div_factor=1000.0)

    for ep in range(1, epochs_p1 + 1):
        t_ep = time.time()
        model.train()
        tr_loss, n_tr = 0.0, 0
        for sigs, pos, src, tmaps in tr_loader:
            sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
            src, tmaps = src.to(DEVICE), tmaps.to(DEVICE)
            pred = model(sigs, pos, src)
            loss = crit(pred, tmaps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tr_loss += loss.item() * sigs.size(0); n_tr += sigs.size(0)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for sigs, pos, src, tmaps in val_loader:
                sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
                src, tmaps = src.to(DEVICE), tmaps.to(DEVICE)
                pred = model(sigs, pos, src)
                val_loss += crit(pred, tmaps).item() * sigs.size(0)
        val_loss /= len(val_ds)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 3 == 0 or ep == 1:
            imp = (bl - val_loss) / bl * 100
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  "
                  f"best={best_loss:.6f}  +{imp:.1f}%  {time.time()-t_ep:.1f}s",
                  flush=True)
    imp = (bl - best_loss) / bl * 100
    print(f"Phase 1 best: {best_loss:.6f}  (+{imp:.1f}%)", flush=True)

    # ── Phase 2 ───────────────────────────────────────────────────────
    print(f"\n=== Phase 2 ({epochs_p2} epochs, lr={lr_p2}) ===", flush=True)
    model.load_state_dict(best_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_p2, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_p2)

    for ep in range(1, epochs_p2 + 1):
        t_ep = time.time()
        model.train()
        tr_loss, n_tr = 0.0, 0
        for sigs, pos, src, tmaps in tr_loader:
            sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
            src, tmaps = src.to(DEVICE), tmaps.to(DEVICE)
            pred = model(sigs, pos, src)
            loss = crit(pred, tmaps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step()
            tr_loss += loss.item() * sigs.size(0); n_tr += sigs.size(0)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for sigs, pos, src, tmaps in val_loader:
                sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
                src, tmaps = src.to(DEVICE), tmaps.to(DEVICE)
                pred = model(sigs, pos, src)
                val_loss += crit(pred, tmaps).item() * sigs.size(0)
        val_loss /= len(val_ds)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 3 == 0 or ep == 1:
            imp = (bl - val_loss) / bl * 100
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  "
                  f"best={best_loss:.6f}  +{imp:.1f}%  {time.time()-t_ep:.1f}s",
                  flush=True)
    imp = (bl - best_loss) / bl * 100
    print(f"Phase 2 best: {best_loss:.6f}  (+{imp:.1f}%)", flush=True)

    torch.save({
        "model_state_dict": best_state,
        "d_model": d_model, "grid_h": grid_h, "grid_w": grid_w,
        "val_loss": best_loss, "baseline_mse": bl,
    }, "checkpoints/model_aluminum.pt")
    print(f"\nSaved checkpoints/model_aluminum.pt  |  Best val MSE: {best_loss:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
