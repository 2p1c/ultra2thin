"""
1D line simulation — 2D COMSOL cross-section style.

Plate length along x, thickness varies along x only.  Source at position xs,
measurement points along the line at positions xi.  A0 Lamb waves propagate
in 1D (dispersion same as 2D, geometric attenuation ~1/sqrt(r)).

Mimics COMSOL 2D cross-section simulation: waves in the x-z plane produce
surface signals that encode local thickness via frequency-dependent phase
velocity c_p(f, d).
"""

import numpy as np
from lamb_dispersion import get_phase_velocity

# Geometry
LINE_LENGTH = 0.1       # m (100 mm)
N_CELLS = 64             # discretisation points along the line
D_NOMINAL = 4.0          # mm
D_MIN = 1.0              # mm

# Signal parameters (compatible with existing setup)
FS = 25e6
N_TIME = 256
FC_LOW, FC_HIGH = 200e3, 700e3


def _build_source_pulse():
    fc = (FC_LOW + FC_HIGH) / 2
    bw = FC_HIGH - FC_LOW
    sigma = 1.0 / (np.pi * bw)
    t0 = 4.0 * sigma
    t = np.arange(N_TIME) / FS
    env = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    pulse = env * np.sin(2.0 * np.pi * fc * (t - t0))
    return (pulse / np.max(np.abs(pulse))).astype(np.float32)


_SOURCE = _build_source_pulse()
_SOURCE_FFT = np.fft.rfft(_SOURCE).astype(np.complex64)
_SPEC_FREQS = np.fft.rfftfreq(N_TIME, 1.0 / FS)
print(f"[sim_1d] Source pulse ready  |  "
      f"{N_TIME} samples @ {FS / 1e6:.0f} MHz")


def generate_terrace_profile(n_defects=None, n_cells=N_CELLS, rng=None):
    """Generate a 1D thickness profile with stepped flat-bottom holes.

    Each defect is a set of concentric rectangular steps (1D: wider → narrower,
    deeper at centre).  Total profile is nominal 4mm with terrace-shaped dips.

    Returns (n_cells,) float32 array: thickness in mm.
    """
    if rng is None:
        rng = np.random
    thickness = np.full(n_cells, D_NOMINAL, dtype=np.float32)

    if n_defects is None:
        n_defects = rng.randint(1, 3)

    for _ in range(n_defects):
        cx = rng.uniform(0.15 * n_cells, 0.85 * n_cells)
        n_steps = rng.randint(2, 4)
        step_depths = np.sort(rng.uniform(0.3, 1.8, n_steps))[::-1]
        cumulative = np.cumsum(step_depths)
        base_half_w = rng.uniform(3, 12)  # half-width in cells

        for step_idx, total_depth in enumerate(cumulative):
            scale = (n_steps - step_idx) / n_steps
            hw = base_half_w * scale
            left = max(0, int(cx - hw))
            right = min(n_cells, int(cx + hw) + 1)
            new_thick = max(D_MIN, D_NOMINAL - total_depth)
            thickness[left:right] = np.minimum(thickness[left:right], new_thick)

    # Mild smoothing
    from scipy.ndimage import gaussian_filter1d
    thickness = gaussian_filter1d(thickness, sigma=0.8)
    return np.clip(thickness, D_MIN, D_NOMINAL).astype(np.float32)


def path_averaged_thickness_1d(profile, src_pos, meas_pos, n_samples=32):
    """Straight-line average thickness between source and each receiver (1D).

    Args:
        profile: (N_cells,) thickness [mm]
        src_pos: scalar x in [0, N_cells-1]
        meas_pos: (N,) x positions in [0, N_cells-1]
        n_samples: sampling points along each segment

    Returns:
        d_eff: (N,) path-averaged thickness [mm]
    """
    N = len(meas_pos)
    alphas = np.linspace(0, 1, n_samples).reshape(1, -1)    # (1, S)
    sx = np.float64(src_pos)
    mx = meas_pos.astype(np.float64).reshape(-1, 1)          # (N, 1)
    xs = sx + alphas * (mx - sx)                              # (N, S)
    xs_clip = np.clip(xs, 0, N_CELLS - 1.001)
    xi = xs_clip.astype(int)
    xf = xs_clip - xi
    xi1 = np.clip(xi + 1, 0, N_CELLS - 1)
    v0 = profile[xi]; v1 = profile[xi1]
    d_along = v0 * (1 - xf) + v1 * xf                         # linear interp
    return d_along.mean(axis=1).astype(np.float32)


def simulate_lamb_signals_1d(profile, source_pos, measure_positions):
    """Simulate A0 Lamb wave signals at 1D measurement points.

    Args:
        profile: (N_cells,) thickness profile [mm]
        source_pos: scalar x in [0, N_CELLS-1]
        measure_positions: (N,) x positions

    Returns:
        signals: (N, N_TIME) float32
        thickness_at_points: (N,) float32
    """
    dx_mm = LINE_LENGTH / N_CELLS * 1e3    # mm per cell

    N = len(measure_positions)
    sx_mm = float(source_pos) * dx_mm
    mx_mm = measure_positions.astype(np.float64) * dx_mm

    L_mm = np.abs(mx_mm - sx_mm)  # (N,) mm  — 1D distance

    d_eff = path_averaged_thickness_1d(profile, source_pos, measure_positions)

    mx_idx = np.clip(measure_positions.astype(int), 0, N_CELLS - 1)
    thick_local = profile[mx_idx].astype(np.float32)

    n_freqs = len(_SPEC_FREQS)
    signals = np.zeros((N, N_TIME), dtype=np.float32)

    for i in range(N):
        if L_mm[i] < 0.1:
            signals[i] = _SOURCE.copy()
            continue

        cp = get_phase_velocity(_SPEC_FREQS, d_eff[i])
        phase = -2.0 * np.pi * _SPEC_FREQS * (L_mm[i] * 1e-3) / cp
        spec = _SOURCE_FFT * np.exp(1j * phase)
        signals[i] = np.fft.irfft(spec, n=N_TIME).real.astype(np.float32)

    # Geometric attenuation (2D cross-section: cylindrical ~1/sqrt(r))
    atten = 1.0 / np.sqrt(L_mm + 1.0)
    signals *= atten[:, np.newaxis].astype(np.float32)

    mx_val = np.max(np.abs(signals), axis=1, keepdims=True)
    mx_val = np.maximum(mx_val, 1e-8)
    signals /= mx_val

    return signals, thick_local


if __name__ == "__main__":
    import time
    rng = np.random.RandomState(0)
    profile = generate_terrace_profile(rng=rng)
    print(f"Profile: [{profile.min():.2f}, {profile.max():.2f}] mm  "
          f"({len(profile)} cells)")

    # 64 measurement points along the line
    pts = np.linspace(0, N_CELLS - 1, N_CELLS).astype(np.float32)
    src = np.float32(N_CELLS * 0.15)

    t0 = time.time()
    sigs, thick = simulate_lamb_signals_1d(profile, src, pts)
    print(f"Signals: {sigs.shape}  |  Thickness: [{thick.min():.2f}, {thick.max():.2f}] mm")
    print(f"Time: {time.time() - t0:.2f}s for {len(pts)} points")

    for i in [0, 20, 50]:
        print(f"  Pt #{i}: x={pts[i]:.0f}  thick={thick[i]:.2f}mm  "
              f"d_eff={path_averaged_thickness_1d(profile, src, pts[i:i+1])[0]:.2f}mm")
