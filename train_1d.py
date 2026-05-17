"""Train LineThicknessPredictor on 1D Lamb-wave cross-section data.

Mimics COMSOL 2D cross-section: measurement points along a 1D line,
thickness varies along x only.  Same architecture as v2 but 1D geometry.
"""

import os, random, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from simulation_1d import (generate_terrace_profile, simulate_lamb_signals_1d,
                            D_NOMINAL, N_TIME, N_CELLS)
from network_1d import LineThicknessPredictor

torch.manual_seed(42); np.random.seed(42); random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_variable(batch):
    """Variable-length measurement points per sample."""
    max_n = max(s[0].shape[0] for s in batch)
    T = batch[0][0].shape[1]
    B = len(batch)
    sigs = torch.zeros(B, max_n, T)
    pos = torch.zeros(B, max_n, 1)
    thick = torch.zeros(B, max_n)
    src = torch.zeros(B, 1)
    mask = torch.ones(B, max_n, dtype=torch.bool)
    for i, (s, p, t, sp, _) in enumerate(batch):
        n = s.shape[0]
        sigs[i, :n] = torch.from_numpy(s)
        pos[i, :n, 0] = torch.from_numpy(p)
        thick[i, :n] = torch.from_numpy(t)
        src[i, 0] = float(sp)
        mask[i, :n] = False
    return sigs, pos, thick, src, mask


def make_one_sample(rng, dense=True):
    """Random profile + measurement grid + random source."""
    profile = generate_terrace_profile(rng=rng, n_cells=N_CELLS)

    sx = rng.uniform(0, N_CELLS * 0.15)
    src = np.float32(sx)

    if dense:
        # Fixed 64-point grid covering the full line
        pts = np.linspace(0, N_CELLS - 1, N_CELLS).astype(np.float32)
    else:
        n_pts = rng.randint(32, 65)
        pts = np.sort(rng.uniform(N_CELLS * 0.05, N_CELLS * 0.95, n_pts))
        pts = pts.astype(np.float32)

    sigs, thick = simulate_lamb_signals_1d(profile, src, pts)
    return sigs, pts, thick, src, profile


class ProfileDataset:
    def __init__(self, n_samples, seed=0):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        local_rng = np.random.RandomState(self.rng.randint(0, 2 ** 31 - 1) + idx)
        return make_one_sample(local_rng, dense=True)


@torch.no_grad()
def validate(model, loader, crit):
    model.eval()
    total, n_valid = 0.0, 0
    for sigs, pos, thick, src, mask in loader:
        sigs, pos = sigs.to(DEVICE), pos.to(DEVICE)
        thick, src = thick.to(DEVICE), src.to(DEVICE)
        mask = mask.to(DEVICE)
        pred = model(sigs, pos, src, mask)
        loss = crit(pred[~mask], thick[~mask])
        nv = (~mask).sum().item()
        total += loss.item() * nv; n_valid += nv
    return total / max(n_valid, 1)


def main():
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)

    # Hyperparams — dense 64pt grid
    d_model, n_heads, n_layers = 160, 4, 5
    n_train, n_val = 4000, 800
    batch_size = 32
    epochs_p1, lr_p1 = 40, 1.5e-3
    epochs_p2, lr_p2 = 25, 1e-4
    decoder_hid = 160

    print(f"=== Iter 1: dense 64pt grid + 1D CNN decoder ===", flush=True)
    print(f"1D line: {N_CELLS} fixed points  |  d_model={d_model}  |  "
          f"train={n_train} val={n_val}  |  batch={batch_size}  |  "
          f"n_layers={n_layers}  |  decoder_hid={decoder_hid}", flush=True)

    # Data
    tr_ds = ProfileDataset(n_train, seed=42)
    val_ds = ProfileDataset(n_val, seed=123)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_variable)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            collate_fn=collate_variable)

    # Baseline — using dense 64pt grid
    val_rng = np.random.RandomState(123)
    pts_all = np.linspace(0, N_CELLS - 1, N_CELLS).astype(np.float32)
    val_thicks_all = []
    for i in range(n_val):
        local_rng = np.random.RandomState(val_rng.randint(0, 2 ** 31 - 1) + i)
        profile = generate_terrace_profile(rng=local_rng, n_cells=N_CELLS)
        val_thicks_all.append(profile)
    bl = float(np.mean((np.stack(val_thicks_all) - D_NOMINAL) ** 2))
    print(f"Baseline MSE (predict {D_NOMINAL:.1f} mm): {bl:.6f}", flush=True)

    # Model — dense mode with 1D CNN decoder
    model = LineThicknessPredictor(
        n_time=N_TIME, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, dense=True, decoder_hid=decoder_hid).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    crit = nn.MSELoss()
    best_loss, best_state = float("inf"), None

    # Phase 1
    print(f"\n=== Phase 1 ({epochs_p1} epochs, lr={lr_p1}) ===", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_p1, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr_p1, steps_per_epoch=len(tr_loader), epochs=epochs_p1,
        pct_start=0.1, div_factor=25.0, final_div_factor=1000.0)

    for ep in range(1, epochs_p1 + 1):
        t_ep = time.time()
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
        if ep % 3 == 0 or ep == 1:
            imp = (bl - val_loss) / bl * 100
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  "
                  f"best={best_loss:.6f}  +{imp:.1f}%  {time.time()-t_ep:.1f}s",
                  flush=True)
    imp = (bl - best_loss) / bl * 100
    print(f"Phase 1 best: {best_loss:.6f}  (+{imp:.1f}%)", flush=True)

    # Phase 2
    print(f"\n=== Phase 2 ({epochs_p2} epochs, lr={lr_p2}) ===", flush=True)
    model.load_state_dict(best_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_p2, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_p2)

    for ep in range(1, epochs_p2 + 1):
        t_ep = time.time()
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
        if ep % 3 == 0 or ep == 1:
            imp = (bl - val_loss) / bl * 100
            print(f"  E{ep:2d}  tr={tr_loss/n_tr:.6f}  val={val_loss:.6f}  "
                  f"best={best_loss:.6f}  +{imp:.1f}%  {time.time()-t_ep:.1f}s",
                  flush=True)
    imp = (bl - best_loss) / bl * 100
    print(f"Phase 2 best: {best_loss:.6f}  (+{imp:.1f}%)", flush=True)

    torch.save({
        "model_state_dict": best_state,
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
        "val_loss": best_loss, "baseline_mse": bl,
    }, "checkpoints/model_1d.pt")
    print(f"\nSaved checkpoints/model_1d.pt  |  Best val MSE: {best_loss:.6f}",
          flush=True)


if __name__ == "__main__":
    main()
