"""Generate visualisation for trained dense aluminum model."""

import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from network_dense import DenseThicknessPredictor
from simulation_aluminum import (generate_terrace_map, D_NOMINAL, GRID_H, GRID_W, N_TIME)
from simulation_lamb import simulate_lamb_signals

# Load trained model
ckpt = torch.load("checkpoints/model_aluminum.pt", map_location="cpu",
                  weights_only=False)
model = DenseThicknessPredictor(
    d_model=ckpt["d_model"], grid_h=ckpt["grid_h"], grid_w=ckpt["grid_w"],
    out_h=GRID_H, out_w=GRID_W)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded model: val_loss={ckpt['val_loss']:.6f}")

# Build dense grid positions
grid_h, grid_w = ckpt["grid_h"], ckpt["grid_w"]
xs = np.linspace(0, GRID_W - 1, grid_w)
ys = np.linspace(0, GRID_H - 1, grid_h)
xx, yy = np.meshgrid(xs, ys)
pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

os.makedirs("web_results/images", exist_ok=True)
rng = np.random.RandomState(123)
samples = []
maes, corrs = [], []

for idx in range(24):
    tmap = generate_terrace_map(rng=rng)
    src = np.array([rng.uniform(0, 4), rng.uniform(3, 29)], dtype=np.float32)
    sigs, _ = simulate_lamb_signals(tmap, src, pts)

    sig_t = torch.from_numpy(sigs).unsqueeze(0)
    pos_t = torch.from_numpy(pts).unsqueeze(0)
    src_t = torch.from_numpy(src).unsqueeze(0)
    with torch.no_grad():
        pred_map = model(sig_t, pos_t, src_t).squeeze(0).squeeze(0).numpy()

    mae = float(np.mean(np.abs(pred_map - tmap)))
    corr = float(np.corrcoef(tmap.ravel(), pred_map.ravel())[0, 1])
    maes.append(mae); corrs.append(corr)

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    vmin, vmax = 1.0, 4.0

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(tmap, cmap="RdYlBu_r", aspect="equal", vmin=vmin, vmax=vmax,
                   origin="lower")
    ax.set_title(f"Ground Truth [{tmap.min():.1f}, {tmap.max():.1f}] mm",
                 fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(pred_map, cmap="RdYlBu_r", aspect="equal", vmin=vmin,
                   vmax=vmax, origin="lower")
    ax.set_title(f"Predicted [{pred_map.min():.1f}, {pred_map.max():.1f}] mm",
                 fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    err = pred_map - tmap
    emax = max(abs(err.min()), abs(err.max()), 0.1)
    im = ax.imshow(err, cmap="RdBu_r", aspect="equal", vmin=-emax, vmax=emax,
                   origin="lower")
    ax.set_title(f"Error (MAE={mae:.4f} mm)", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Cross-section
    ax = fig.add_subplot(gs[0, 3])
    row = GRID_H // 2
    ax.plot(tmap[row, :], "k-", lw=1.5, label="GT")
    ax.plot(pred_map[row, :], "r--", lw=1.5, label="Pred")
    ax.set_title(f"Cross-section (row {row})", fontsize=9, fontweight="bold")
    ax.set_xlabel("x [grid]"); ax.set_ylabel("Thickness [mm]")
    ax.legend(fontsize=7); ax.set_ylim(0.5, 4.5)

    # Dense measurement grid
    ax = fig.add_subplot(gs[1, :2])
    ax.scatter(pts[:, 0], pts[:, 1], c="cyan", s=1, alpha=0.3)
    ax.scatter([src[0]], [src[1]], c="red", s=120, marker="*", zorder=10,
               edgecolors="darkred")
    ax.set_title(f"{grid_h}x{grid_w} = {grid_h*grid_w} Measurement Points + Source",
                 fontsize=9, fontweight="bold")
    ax.set_xlim(-2, 34); ax.set_ylim(-2, 34); ax.set_aspect("equal")

    # Signals (first few)
    ax = fig.add_subplot(gs[1, 2:])
    t_us = np.arange(sigs.shape[1]) / 25e6 * 1e6
    step = max(1, len(sigs) // 12)
    for i in range(0, min(len(sigs), 12 * step), step):
        ax.plot(t_us, sigs[i] + (i // step) * 0.15, lw=0.3, alpha=0.7)
    ax.set_title(f"Signals ({len(sigs)} total, shown {min(12, len(sigs)//step)})",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Time [us]")

    fig.suptitle(f"Sample #{idx}  |  {grid_h}x{grid_w}  |  "
                 f"MAE={mae:.4f}mm  Corr={corr:.3f}",
                 fontsize=10, fontweight="bold")
    fig.savefig(f"web_results/images/sample_{idx}.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    samples.append({
        "id": idx, "grid": f"{grid_h}x{grid_w}",
        "mae": round(mae, 4), "corr": round(corr, 4),
        "src": [round(float(src[0]), 1), round(float(src[1]), 1)],
        "gt_range": [round(float(tmap.min()), 2), round(float(tmap.max()), 2)],
        "pred_range": [round(float(pred_map.min()), 2),
                       round(float(pred_map.max()), 2)],
    })

# ── Summary ──────────────────────────────────────────────────────────
summary = {
    "mean_mae": round(np.mean(maes), 4),
    "std_mae": round(np.std(maes), 4),
    "mean_corr": round(np.mean(corrs), 4),
    "std_corr": round(np.std(corrs), 4),
    "val_loss": ckpt["val_loss"],
    "grid": f"{grid_h}x{grid_w}",
}

with open("web_results/samples.json", "w") as f:
    json.dump({"samples": samples, "summary": summary}, f, indent=2)

print(f"\n=== Results ({grid_h}x{grid_w} dense grid) ===")
print(f"  Mean MAE:  {summary['mean_mae']:.4f} +/- {summary['std_mae']:.4f} mm")
print(f"  Mean Corr: {summary['mean_corr']:.4f} +/- {summary['std_corr']:.4f}")
print(f"  Val MSE:   {summary['val_loss']:.6f}")
print(f"Generated {len(samples)} samples in web_results/")
