"""
Flexible-grid Born scattering — precomputes kernel once at import.
Fast per-point signal generation via vectorized computation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

C_REF = 5400.0
D_REF = 2.0
FC = 500e3
FS = 25e6
PLATE_W = 0.1
GRID_H, GRID_W = 32, 32
N_TIME = 128

# ── Precompute Born kernel once at import ────────────────────────────────

def _build_born_kernel():
    """Build pixel-to-pixel Born scattering kernel (n_freqs, n_pix, n_pix)."""
    n_pix = GRID_H * GRID_W
    dx_phys = PLATE_W / GRID_W

    py, px = np.mgrid[0:GRID_H, 0:GRID_W]
    pix_x = px.ravel().astype(np.float32) * dx_phys
    pix_y = py.ravel().astype(np.float32) * dx_phys

    dx = pix_x[:, np.newaxis] - pix_x[np.newaxis, :]
    dy = pix_y[:, np.newaxis] - pix_y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2) + 1e-6

    # Source wavelet spectrum
    t = np.arange(N_TIME) / FS
    n_cycles = 5
    T_burst = n_cycles / FC
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / T_burst))
    env[t > T_burst] = 0.0
    src_t = env * np.sin(2.0 * np.pi * FC * t).astype(np.float32)
    src_f = np.fft.rfft(src_t).astype(np.complex64)
    n_freqs = len(src_f)
    freqs = np.fft.rfftfreq(N_TIME, 1.0 / FS).astype(np.float32)
    cutoff = min(FC * 3.0, FS / 2.5)
    src_f[int(cutoff / (FS/2) * (n_freqs-1)):] = 0.0

    omega = 2.0 * np.pi * freqs
    k_wave = omega[:, np.newaxis, np.newaxis] / C_REF

    amp = 1.0 / np.sqrt(dist)
    phase = np.exp(-1j * omega[:, np.newaxis, np.newaxis] * dist[np.newaxis, :, :] / C_REF)
    A = (k_wave**2) * amp[np.newaxis, :, :] * phase
    A = A.astype(np.complex64)
    s = 50.0 / max(np.sqrt(np.mean(np.abs(A)**2)), 1e-8)
    A *= s

    return A, src_f, freqs, n_freqs

_KERNEL, _SRC_F, _FREQS, _N_FREQS = _build_born_kernel()
print(f"[sim_v2] Born kernel precomputed: {_KERNEL.shape} ({_KERNEL.nbytes/1e6:.0f} MB)")


# ── Thickness map generation ─────────────────────────────────────────────

def generate_defect_map(n_defects=None, min_size=2, max_size=8,
                         min_depth=0.2, max_depth=1.5):
    thickness = np.full((GRID_H, GRID_W), D_REF, dtype=np.float32)
    if n_defects is None:
        n_defects = np.random.randint(1, 4)
    for _ in range(n_defects):
        cx = np.random.uniform(0.15 * GRID_W, 0.85 * GRID_W)
        cy = np.random.uniform(0.15 * GRID_H, 0.85 * GRID_H)
        delta = np.random.choice([-1, 1]) * np.random.uniform(min_depth, max_depth)
        yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
        dx = xx - cx; dy = yy - cy

        shape = np.random.choice(['ellipse', 'circle', 'square'])
        if shape == 'ellipse':
            rx = np.random.uniform(min_size, max_size)
            ry = np.random.uniform(min_size, max_size)
            theta = np.random.uniform(0, np.pi)
            d2 = ((dx*np.cos(theta)+dy*np.sin(theta))/rx)**2 + \
                 ((-dx*np.sin(theta)+dy*np.cos(theta))/ry)**2
            thickness += delta * np.exp(-d2 * 0.5)
        elif shape == 'circle':
            r = np.random.uniform(min_size, max_size)
            d2 = (dx**2 + dy**2) / r**2
            thickness += delta * np.exp(-d2 * 0.5)
        else:  # square
            half = np.random.uniform(min_size, max_size)
            theta = np.random.uniform(0, np.pi)
            dxr = dx*np.cos(theta) + dy*np.sin(theta)
            dyr = -dx*np.sin(theta) + dy*np.cos(theta)
            edge_x = 1.0 / (1.0 + np.exp((np.abs(dxr) - half) * 2))
            edge_y = 1.0 / (1.0 + np.exp((np.abs(dyr) - half) * 2))
            thickness += delta * edge_x * edge_y

    thickness = gaussian_filter(thickness, sigma=1.0)
    return np.clip(thickness, 0.8, 4.5).astype(np.float32)


# ── Signal simulation ────────────────────────────────────────────────────

def simulate_point_signals(thickness_map, source_pos, measure_positions):
    """Vectorized: compute scattered signal at all measurement points simultaneously.

    Args:
        thickness_map: (32, 32) thickness in mm
        source_pos: (sx, sy) in grid coords
        measure_positions: (N, 2) grid coords

    Returns:
        signals: (N, N_TIME) scattered signals
        thickness_at_points: (N,) true thickness
    """
    H, W = thickness_map.shape
    n_pix = H * W
    N = len(measure_positions)

    delta_d = (thickness_map.ravel() - D_REF).astype(np.float32)

    # Source pixel index
    sx = int(np.clip(int(source_pos[0]), 0, W-1))
    sy = int(np.clip(int(source_pos[1]), 0, H-1))
    src_idx = sy * W + sx

    # Measurement pixel indices
    mx = np.clip(measure_positions[:, 0].astype(int), 0, W-1)
    my = np.clip(measure_positions[:, 1].astype(int), 0, H-1)
    meas_idx = my * W + mx  # (N,)

    # For each frequency, compute R[f, n] = S[f] * K[f, meas_n, :] @ delta_d
    # Vectorize: R[f, :] = S[f] * K[f, meas_indices, :] @ delta_d
    # This is: for each f, take rows of K for measurement indices, dot with delta_d
    R = np.zeros((N, _N_FREQS), dtype=np.complex64)

    for f_idx in range(_N_FREQS):
        K_rows = _KERNEL[f_idx, meas_idx, :]  # (N, n_pix)
        R[:, f_idx] = _SRC_F[f_idx] * (K_rows @ delta_d)

    signals = np.fft.irfft(R, n=N_TIME, axis=1).real.astype(np.float32)

    # Normalize
    mx_val = np.max(np.abs(signals))
    if mx_val > 0:
        signals /= mx_val

    # Ground truth thickness at measurement points
    thickness_at_points = thickness_map[my, mx].astype(np.float32)

    return signals, thickness_at_points


# ── Dataset generation ───────────────────────────────────────────────────

def generate_point_dataset(n_samples, min_points=9, max_points=64):
    """Generate variable-grid dataset.

    Returns:
        all_signals:     list of (N_i, N_TIME) float32 arrays
        all_positions:   list of (N_i, 2) float32 arrays
        all_thickness:   list of (N_i,) float32 arrays
        all_source_pos:  list of (2,) float32 arrays
        all_tmaps:       list of (32, 32) float32 arrays
    """
    all_signals, all_positions, all_thickness = [], [], []
    all_source_pos, all_tmaps = [], []

    for _ in range(n_samples):
        tmap = generate_defect_map()

        # Random source on left edge
        sx = np.random.uniform(0, GRID_W * 0.15)
        sy = np.random.uniform(GRID_H * 0.1, GRID_H * 0.9)
        src = np.array([sx, sy], dtype=np.float32)

        # Random grid of measurement points
        n_pts = np.random.randint(min_points, max_points + 1)
        n_cols = int(np.sqrt(n_pts))
        n_rows = (n_pts + n_cols - 1) // n_cols

        x_base = np.linspace(GRID_W * 0.1, GRID_W * 0.9, n_cols)
        y_base = np.linspace(GRID_H * 0.1, GRID_H * 0.9, n_rows)
        xx, yy = np.meshgrid(x_base, y_base)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)[:n_pts]

        spacing = (GRID_W * 0.8) / max(n_cols - 1, 1)
        jitter = np.random.uniform(-0.08 * spacing, 0.08 * spacing, pts.shape)
        pts = np.clip(pts + jitter, [0, 0], [GRID_W-1, GRID_H-1]).astype(np.float32)

        sigs, thick = simulate_point_signals(tmap, src, pts)

        all_signals.append(sigs)
        all_positions.append(pts)
        all_thickness.append(thick)
        all_source_pos.append(src)
        all_tmaps.append(tmap)

    return all_signals, all_positions, all_thickness, all_source_pos, all_tmaps


def generate_dense_dataset(n_samples, grid_cols=14, grid_rows=14):
    """Generate fixed dense-grid dataset covering full plate.

    grid_cols × grid_rows measurement points, evenly spaced over 100mm×100mm.
    Source position varies randomly on the left edge.

    Returns: same format as generate_point_dataset.
    """
    n_pts = grid_cols * grid_rows
    x_base = np.linspace(0, GRID_W - 1, grid_cols)
    y_base = np.linspace(0, GRID_H - 1, grid_rows)
    xx, yy = np.meshgrid(x_base, y_base)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

    all_signals, all_positions, all_thickness = [], [], []
    all_source_pos, all_tmaps = [], []

    for _ in range(n_samples):
        tmap = generate_defect_map()
        sx = np.random.uniform(0, GRID_W * 0.1)
        sy = np.random.uniform(GRID_H * 0.1, GRID_H * 0.9)
        src = np.array([sx, sy], dtype=np.float32)
        sigs, thick = simulate_point_signals(tmap, src, pts)
        all_signals.append(sigs)
        all_positions.append(pts)
        all_thickness.append(thick)
        all_source_pos.append(src)
        all_tmaps.append(tmap)

    return all_signals, all_positions, all_thickness, all_source_pos, all_tmaps


if __name__ == "__main__":
    import time
    t0 = time.time()
    sigs, pos, thick, src, tmaps = generate_dense_dataset(5, 14, 14)
    n = sigs[0].shape[0]
    spacing = 100 / 13
    print(f"14×14={n} pts over 100mm×100mm, spacing={spacing:.1f}mm: {time.time()-t0:.2f}s")

