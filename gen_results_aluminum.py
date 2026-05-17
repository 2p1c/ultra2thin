"""Generate comprehensive visualisation data for the Lamb-wave model.

Produces:
  1. Training signal gallery — 10 random plates, signals at key measurement pts
  2. Prediction gallery — 24 comparison images (GT / pred / error / x-section)
  3. Detail samples.json with per-sample metrics
"""

import os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from network_dense import DenseThicknessPredictor
from simulation_aluminum import generate_terrace_map, D_NOMINAL, GRID_H, GRID_W
from simulation_lamb import simulate_lamb_signals, path_averaged_thickness

# ── Load model ─────────────────────────────────────────────────────────────
ckpt = torch.load("checkpoints/model_aluminum.pt", map_location="cpu",
                  weights_only=False)
model = DenseThicknessPredictor(
    d_model=ckpt["d_model"], grid_h=ckpt["grid_h"], grid_w=ckpt["grid_w"],
    out_h=GRID_H, out_w=GRID_W)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded model: val_loss={ckpt['val_loss']:.6f}")

# ── Build dense grid positions ────────────────────────────────────────────
grid_h, grid_w = ckpt["grid_h"], ckpt["grid_w"]
xs = np.linspace(0, GRID_W - 1, grid_w)
ys = np.linspace(0, GRID_H - 1, grid_h)
xx, yy = np.meshgrid(xs, ys)
GRID_PTS = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

os.makedirs("web_results/images", exist_ok=True)
rng = np.random.RandomState(42)
FS = 25e6

# ═══════════════════════════════════════════════════════════════════════════
# 1. Training signal gallery — raw input signals
# ═══════════════════════════════════════════════════════════════════════════

print("Generating training signal gallery...")
train_signal_samples = []

for idx in range(10):
    tmap = generate_terrace_map(rng=rng)
    sx = rng.uniform(0, GRID_W * 0.2)
    sy = rng.uniform(GRID_H * 0.1, GRID_H * 0.9)
    src = np.array([sx, sy], dtype=np.float32)
    sigs, thick_local = simulate_lamb_signals(tmap, src, GRID_PTS)
    d_eff = path_averaged_thickness(tmap, src, GRID_PTS)

    # Pick 5 representative points: near, mid, far, thin, thick
    dx_mm = (GRID_PTS[:, 0] - sx) * 100 / GRID_W
    dy_mm = (GRID_PTS[:, 1] - sy) * 100 / GRID_H
    dist_mm = np.sqrt(dx_mm ** 2 + dy_mm ** 2)
    sort_dist = np.argsort(dist_mm)

    indices = [
        sort_dist[0],                          # nearest
        sort_dist[len(sort_dist) // 5],        # near-mid
        sort_dist[len(sort_dist) // 2],        # mid
        sort_dist[len(sort_dist) * 3 // 4],    # far-mid
        sort_dist[-1],                          # farthest
    ]

    signals_data = []
    for i in indices:
        signals_data.append({
            "index": int(i),
            "x": round(float(GRID_PTS[i, 0]), 1),
            "y": round(float(GRID_PTS[i, 1]), 1),
            "dist_mm": round(float(dist_mm[i]), 1),
            "thickness": round(float(thick_local[i]), 3),
            "d_eff": round(float(d_eff[i]), 3),
            "signal": sigs[i].tolist(),
        })

    # Also save the full thickness map
    train_signal_samples.append({
        "id": idx,
        "src": [round(float(sx), 1), round(float(sy), 1)],
        "tmap": tmap.tolist(),
        "tmap_range": [round(float(tmap.min()), 2), round(float(tmap.max()), 2)],
        "signals": signals_data,
    })

    # Plot a summary figure for this training sample
    fig = plt.figure(figsize=(12, 4.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.2, 1.6])

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(tmap, cmap="RdYlBu_r", aspect="equal", vmin=1.0, vmax=4.0,
                   origin="lower")
    ax.scatter([sx], [sy], c="red", s=80, marker="*", zorder=10,
               edgecolors="darkred")
    for sd in signals_data:
        ax.scatter([sd["x"]], [sd["y"]], c="cyan", s=25, zorder=9,
                   edgecolors="white", linewidths=0.4)
    ax.set_title(f"Plate #{idx}", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1])
    t_us = np.arange(256) / FS * 1e6
    for j, sd in enumerate(signals_data):
        sig_arr = np.array(sd["signal"])
        ax.plot(t_us, sig_arr + j * 0.4, lw=0.5,
                label=f'd={sd["thickness"]:.1f}mm L={sd["dist_mm"]:.0f}mm')
    ax.set_xlabel("Time [μs]"); ax.set_title("Selected signals", fontsize=10)
    ax.legend(fontsize=5.5, loc="upper right")

    ax = fig.add_subplot(gs[2])
    # Show all 1024 signals as a heatmap (rows sorted by distance)
    n_show = min(60, len(sort_dist))
    show_idx = sort_dist[:n_show]
    heatmap = sigs[show_idx]
    ax.imshow(heatmap, aspect="auto", cmap="RdBu_r", origin="upper",
              extent=[0, 256 / FS * 1e6, 0, n_show],
              vmin=-1, vmax=1)
    ax.set_xlabel("Time [μs]"); ax.set_ylabel("Point (sorted by dist)")
    ax.set_title(f"Signal heatmap ({n_show} pts)", fontsize=10)

    fig.suptitle(f"Training Sample #{idx}  |  Source ({sx:.1f}, {sy:.1f})  |  "
                 f"Thickness [{tmap.min():.1f}–{tmap.max():.1f}] mm",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"web_results/images/train_signal_{idx}.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Prediction gallery — 24 full comparisons + signal overlay
# ═══════════════════════════════════════════════════════════════════════════

print("Generating prediction gallery...")
pred_rng = np.random.RandomState(123)
pred_samples = []
maes, corrs = [], []

for idx in range(24):
    tmap = generate_terrace_map(rng=pred_rng)
    src = np.array([pred_rng.uniform(0, 4), pred_rng.uniform(3, 29)],
                   dtype=np.float32)
    sigs, _ = simulate_lamb_signals(tmap, src, GRID_PTS)

    sig_t = torch.from_numpy(sigs).unsqueeze(0)
    pos_t = torch.from_numpy(GRID_PTS).unsqueeze(0)
    src_t = torch.from_numpy(src).unsqueeze(0)
    with torch.no_grad():
        pred_map = model(sig_t, pos_t, src_t).squeeze(0).squeeze(0).numpy()

    mae = float(np.mean(np.abs(pred_map - tmap)))
    corr = float(np.corrcoef(tmap.ravel(), pred_map.ravel())[0, 1])
    maes.append(mae); corrs.append(corr)

    # Pick a representative signal (mid-thickness point)
    thick_local_val = tmap[int(GRID_H//2), int(GRID_W//2)]
    mid_idx = (GRID_H // 2) * GRID_W + (GRID_W // 2)

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
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
    emax = max(abs(err.min()), abs(err.max()), 0.05)
    im = ax.imshow(err, cmap="RdBu_r", aspect="equal", vmin=-emax, vmax=emax,
                   origin="lower")
    ax.set_title(f"Error (MAE={mae:.4f} mm)", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Cross-section at middle row
    ax = fig.add_subplot(gs[0, 3])
    row_idx = GRID_H // 2
    ax.plot(tmap[row_idx, :], "k-", lw=1.5, label="GT")
    ax.plot(pred_map[row_idx, :], "r--", lw=1.5, label="Pred")
    ax.fill_between(range(GRID_W), tmap[row_idx, :], pred_map[row_idx, :],
                    alpha=0.2, color="gray")
    ax.set_title(f"X-section row {row_idx}", fontsize=9, fontweight="bold")
    ax.set_xlabel("x [grid]"); ax.set_ylabel("Thickness [mm]")
    ax.legend(fontsize=7); ax.set_ylim(0.5, 4.5)

    # Measurement grid + source
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(GRID_PTS[:, 0], GRID_PTS[:, 1], c="cyan", s=0.5, alpha=0.3)
    ax.scatter([src[0]], [src[1]], c="red", s=100, marker="*", zorder=10,
               edgecolors="darkred")
    ax.set_title("32×32 Measurement Grid", fontsize=9, fontweight="bold")
    ax.set_xlim(-2, 34); ax.set_ylim(-2, 34); ax.set_aspect("equal")

    # Scatter: predicted vs true at all points
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(tmap.ravel(), pred_map.ravel(), c="steelblue", s=2, alpha=0.3)
    ax.plot([1, 4.5], [1, 4.5], "r--", lw=1)
    ax.set_xlabel("True thickness [mm]"); ax.set_ylabel("Predicted [mm]")
    ax.set_title(f"Corr = {corr:.4f}", fontsize=9, fontweight="bold")
    ax.set_xlim(0.8, 4.7); ax.set_ylim(0.8, 4.7)

    # Signal at centre point
    ax = fig.add_subplot(gs[1, 2])
    t_us = np.arange(sigs.shape[1]) / FS * 1e6
    ax.plot(t_us, sigs[mid_idx], "steelblue", lw=0.6)
    ax.set_title(f"Signal at ({GRID_W//2},{GRID_H//2})", fontsize=9,
                 fontweight="bold")
    ax.set_xlabel("Time [μs]")

    # Column cross-section
    ax = fig.add_subplot(gs[1, 3])
    col_idx = GRID_W // 2
    ax.plot(tmap[:, col_idx], "k-", lw=1.5, label="GT")
    ax.plot(pred_map[:, col_idx], "r--", lw=1.5, label="Pred")
    ax.set_title(f"X-section col {col_idx}", fontsize=9, fontweight="bold")
    ax.set_xlabel("y [grid]"); ax.set_ylabel("Thickness [mm]")
    ax.legend(fontsize=7); ax.set_ylim(0.5, 4.5)

    fig.suptitle(f"Prediction #{idx}  |  MAE={mae:.4f}mm  Corr={corr:.3f}",
                 fontsize=11, fontweight="bold")
    fig.savefig(f"web_results/images/pred_{idx}.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    pred_samples.append({
        "id": idx,
        "mae": round(mae, 4), "corr": round(corr, 4),
        "src": [round(float(src[0]), 1), round(float(src[1]), 1)],
        "gt_range": [round(float(tmap.min()), 2), round(float(tmap.max()), 2)],
        "pred_range": [round(float(pred_map.min()), 2),
                       round(float(pred_map.max()), 2)],
        "mse": round(float(np.mean((pred_map - tmap) ** 2)), 6),
    })

# ═══════════════════════════════════════════════════════════════════════════
# 3. Save all data
# ═══════════════════════════════════════════════════════════════════════════

summary = {
    "mean_mae": round(np.mean(maes), 4),
    "std_mae": round(np.std(maes), 4),
    "mean_corr": round(np.mean(corrs), 4),
    "std_corr": round(np.std(corrs), 4),
    "val_loss": ckpt["val_loss"],
    "grid": f"{grid_h}x{grid_w}",
}

output = {
    "train_signals": train_signal_samples,
    "predictions": pred_samples,
    "summary": summary,
}

with open("web_results/data.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n=== Results ===")
print(f"  Mean MAE:  {summary['mean_mae']:.4f} ± {summary['std_mae']:.4f} mm")
print(f"  Mean Corr: {summary['mean_corr']:.4f} ± {summary['std_corr']:.4f}")
print(f"  Val MSE:   {summary['val_loss']:.6f}")
print(f"  Training signal samples: {len(train_signal_samples)}")
print(f"  Prediction samples:      {len(pred_samples)}")
