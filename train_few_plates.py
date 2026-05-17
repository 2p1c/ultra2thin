"""Train with few plates × many source positions — realistic experiment setup.

6 training plates + 2 validation plates, each measured at 200 different
source positions.  Same network architecture and training schedule as
train_aluminum.py, only the data distribution differs.

Compare with train_aluminum.py (500 unique plates/epoch) to measure the
impact of limited plate diversity on generalisation.
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


def make_one_sample(rng, tmap, grid_h, grid_w):
    """One source-position configuration on a given plate."""
    sx = rng.uniform(0, GRID_W * 0.2)
    sy = rng.uniform(GRID_H * 0.1, GRID_H * 0.9)
    src = np.array([sx, sy], dtype=np.float32)
    pts = build_grid_positions(grid_h, grid_w)
    sigs, thick = simulate_lamb_signals(tmap, src, pts)
    return sigs, pts, thick, src, tmap


class FewPlateDataset:
    """Fixed set of plates × many source configurations each."""

    def __init__(self, plates, configs_per_plate, grid_h=32, grid_w=32, seed=0):
        self.plates = plates
        self.n_plates = len(plates)
        self.configs_per_plate = configs_per_plate
        self.n_samples = self.n_plates * configs_per_plate
        self.grid_h, self.grid_w = grid_h, grid_w
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        plate_idx = idx % self.n_plates
        config_idx = idx // self.n_plates
        local_rng = np.random.RandomState(
            self.rng.randint(0, 2 ** 31 - 1) + plate_idx * 10000 + config_idx)
        return make_one_sample(local_rng, self.plates[plate_idx],
                               self.grid_h, self.grid_w)


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
    grid_h, grid_w = 32, 32
    d_model = 48
    n_train_plates = 6
    n_val_plates = 2
    configs_per_plate = 80           # source positions per plate
    batch_size = 12
    epochs_p1, lr_p1 = 20, 2e-3
    epochs_p2, lr_p2 = 12, 2e-4

    n_train = n_train_plates * configs_per_plate
    n_val = n_val_plates * configs_per_plate
    print(f"=== Few-Plate Setup ===")
    print(f"Train: {n_train_plates} plates × {configs_per_plate} configs = {n_train} samples")
    print(f"Val:   {n_val_plates} plates × {configs_per_plate} configs = {n_val} samples")
    print(f"Grid: {grid_h}x{grid_w}  |  d_model={d_model}  |  batch={batch_size}",
          flush=True)

    # ── Generate fixed plate set ─────────────────────────────────────
    rng = np.random.RandomState(42)
    print("\nGenerating fixed plate set...", flush=True)
    train_plates = [generate_terrace_map(rng=rng) for _ in range(n_train_plates)]
    val_plates = [generate_terrace_map(rng=rng) for _ in range(n_val_plates)]
    for i, p in enumerate(train_plates):
        print(f"  Train plate {i}: [{p.min():.1f}, {p.max():.1f}] mm", flush=True)
    for i, p in enumerate(val_plates):
        print(f"  Val plate {i}:   [{p.min():.1f}, {p.max():.1f}] mm", flush=True)

    bl = float(np.mean((np.stack(val_plates) - D_NOMINAL) ** 2))
    print(f"Baseline MSE: {bl:.6f}", flush=True)

    # ── Datasets ─────────────────────────────────────────────────────
    tr_ds = FewPlateDataset(train_plates, configs_per_plate, grid_h, grid_w, seed=42)
    val_ds = FewPlateDataset(val_plates, configs_per_plate, grid_h, grid_w, seed=123)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_dense)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            collate_fn=collate_dense)

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
        "train_plates": [p.tolist() for p in train_plates],
        "val_plates": [p.tolist() for p in val_plates],
    }, "checkpoints/model_few_plates.pt")
    print(f"\nSaved checkpoints/model_few_plates.pt  |  "
          f"Best val MSE: {best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
