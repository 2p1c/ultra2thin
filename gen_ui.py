"""Generate UI images after training completes."""
import os, torch, numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from network_v2 import PointThicknessPredictor
from simulation_v2 import generate_defect_map, simulate_point_signals

ckpt = torch.load('checkpoints/model_v2.pt', map_location='cpu', weights_only=False)
model = PointThicknessPredictor(d_model=ckpt['d_model'], n_heads=ckpt['n_heads'], n_layers=ckpt['n_layers'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

os.makedirs('web_v2/images', exist_ok=True)
samples = []
maes, corrs = [], []

for idx in range(30):
    tmap = generate_defect_map()
    n_pts = np.random.randint(16, 49)
    n_cols = int(np.sqrt(n_pts)); n_rows = (n_pts + n_cols - 1) // n_cols
    xs = np.linspace(1, 31, n_cols); ys = np.linspace(1, 31, n_rows)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)[:n_pts].astype(np.float32)
    src = np.array([np.random.uniform(0, 4), np.random.uniform(3, 29)], dtype=np.float32)
    sigs, gt = simulate_point_signals(tmap, src, pts)
    sig_t = torch.from_numpy(sigs).unsqueeze(0)
    with torch.no_grad():
        pred = model(sig_t, torch.from_numpy(pts).unsqueeze(0), torch.from_numpy(src).unsqueeze(0)).squeeze(0).numpy()

    mae = float(np.mean(np.abs(pred - gt)))
    corr = float(np.corrcoef(gt, pred)[0, 1])
    maes.append(mae); corrs.append(corr)

    # Interpolate
    grid_x, grid_y = np.mgrid[0:32, 0:32]
    pred_2d = griddata(pts, pred, (grid_x, grid_y), method='cubic', fill_value=2.0)

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(tmap, cmap='RdYlBu_r', aspect='equal', vmin=0.8, vmax=4.5, origin='lower')
    ax.set_title(f'Ground Truth [{tmap.min():.1f},{tmap.max():.1f}]mm', fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(pred_2d, cmap='RdYlBu_r', aspect='equal', vmin=0.8, vmax=4.5, origin='lower')
    ax.set_title(f'Predicted [{pred_2d.min():.1f},{pred_2d.max():.1f}]mm', fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    err = pred_2d - tmap
    emax = max(abs(err.min()), abs(err.max()), 0.1)
    im = ax.imshow(err, cmap='RdBu_r', aspect='equal', vmin=-emax, vmax=emax, origin='lower')
    ax.set_title(f'Error (MAE={mae:.4f})', fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, :2])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pred, cmap='RdYlBu_r', s=40, vmin=0.8, vmax=4.5,
                    edgecolors='white', linewidths=0.5, zorder=5)
    ax.scatter([src[0]], [src[1]], c='red', s=100, marker='*', zorder=10, edgecolors='darkred')
    ax.set_title(f'{n_pts} Points + Source  |  Corr={corr:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlim(-2, 34); ax.set_ylim(-2, 34); ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, fraction=0.03, label='Thickness [mm]')

    ax = fig.add_subplot(gs[1, 2])
    t_axis = np.arange(sigs.shape[1]) / 25e6 * 1e6
    for i in range(min(16, len(sigs))):
        ax.plot(t_axis, sigs[i] + i * 0.12, lw=0.4, alpha=0.7)
    ax.set_title(f'Signals (first {min(16, len(sigs))})', fontsize=9, fontweight='bold')
    ax.set_xlabel('Time [us]')

    fig.suptitle(f'Sample #{idx}  |  {n_pts} pts  |  MAE={mae:.4f}mm  Corr={corr:.3f}',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'web_v2/images/sample_{idx}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    samples.append({'id': idx, 'n_pts': n_pts, 'mae': round(mae, 4), 'corr': round(corr, 4),
                    'src': [round(float(src[0]), 1), round(float(src[1]), 1)],
                    'gt_range': [round(float(tmap.min()), 2), round(float(tmap.max()), 2)]})

with open('web_v2/samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

print(f'Generated {len(samples)} samples')
print(f'Mean MAE: {np.mean(maes):.4f}  Mean Corr: {np.mean(corrs):.4f}')
