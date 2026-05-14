"""
Lamb-wave (A0 mode) signal simulation for thickness measurement.

Source generates broadband pulse (200-700 kHz) on plate surface.
A0 Lamb waves propagate laterally — dispersion (c_p(f,d)) encodes
local thickness along the source-receiver path.

~60x more thickness-sensitive than through-thickness time-of-flight.
"""

import numpy as np
from lamb_dispersion import get_phase_velocity

# Plate geometry
GRID_H, GRID_W = 32, 32
PLATE_W = 0.1        # m (100 mm)

# Signal parameters
FS = 25e6            # Hz
N_TIME = 256
FC_LOW, FC_HIGH = 200e3, 700e3

# Source pulse: same as simulation_aluminum.py
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
print(f"[sim_lamb] Source pulse ready  |  "
      f"{N_TIME} samples @ {FS / 1e6:.0f} MHz  |  "
      f"{N_TIME / FS * 1e6:.1f} us window")


def path_averaged_thickness(tmap, src_pos, meas_pos, n_samples=32):
    """Compute straight-line average thickness between source and each receiver.

    Args:
        tmap: (32, 32) thickness map [mm]
        src_pos: (sx, sy) grid coords
        meas_pos: (N, 2) grid coords
        n_samples: samples along each path

    Returns:
        d_eff: (N,) effective (mean) thickness along each path [mm]
    """
    N = len(meas_pos)
    alphas = np.linspace(0, 1, n_samples).reshape(1, -1, 1)   # (1, S, 1)
    sx = np.float64(src_pos[0])
    sy = np.float64(src_pos[1])
    mx = meas_pos[:, 0].astype(np.float64).reshape(-1, 1, 1)   # (N, 1, 1)
    my = meas_pos[:, 1].astype(np.float64).reshape(-1, 1, 1)

    # Sample positions along each path: (N, S, 2)
    xs = sx + alphas * (mx - sx)                                 # (N, S, 1)
    ys = sy + alphas * (my - sy)                                 # (N, S, 1)
    xs = xs.squeeze(-1)                                           # (N, S)
    ys = ys.squeeze(-1)

    # Clip to grid bounds
    xs_clip = np.clip(xs, 0, GRID_W - 1.001)
    ys_clip = np.clip(ys, 0, GRID_H - 1.001)

    # Bilinear interpolation on thickness map
    xi = xs_clip.astype(int)
    yi = ys_clip.astype(int)
    xf = xs_clip - xi
    yf = ys_clip - yi

    xi1 = np.clip(xi + 1, 0, GRID_W - 1)
    yi1 = np.clip(yi + 1, 0, GRID_H - 1)

    v00 = tmap[yi, xi]
    v10 = tmap[yi, xi1]
    v01 = tmap[yi1, xi]
    v11 = tmap[yi1, xi1]

    d_along = (v00 * (1 - xf) * (1 - yf) +
               v10 * xf * (1 - yf) +
               v01 * (1 - xf) * yf +
               v11 * xf * yf)

    return d_along.mean(axis=1).astype(np.float32)               # (N,)


def simulate_lamb_signals(tmap, source_pos, measure_positions):
    """Simulate A0 Lamb wave signals at measurement points.

    Each receiver signal is a dispersed version of the source pulse:
      S_r(f) = S_src(f) * exp(-j * 2*pi*f * L / c_p(f, d_eff))

    where L = in-plane source-receiver distance, and d_eff = path-averaged
    thickness.

    Args:
        tmap: (32, 32) thickness map [mm]
        source_pos: (sx, sy) grid coords
        measure_positions: (N, 2) grid coords

    Returns:
        signals: (N, N_TIME) float32
        thickness_at_points: (N,) float32  — local thickness at each receiver
    """
    dx_mm = PLATE_W / GRID_W * 1e3    # mm per grid unit (~3.125)

    N = len(measure_positions)
    sx_mm = float(source_pos[0]) * dx_mm
    sy_mm = float(source_pos[1]) * dx_mm
    mx_mm = measure_positions[:, 0].astype(np.float64) * dx_mm
    my_mm = measure_positions[:, 1].astype(np.float64) * dx_mm

    # In-plane distance
    dx = mx_mm - sx_mm
    dy = my_mm - sy_mm
    L_mm = np.sqrt(dx ** 2 + dy ** 2)  # (N,) mm

    # Path-averaged effective thickness
    d_eff = path_averaged_thickness(tmap, source_pos, measure_positions)  # (N,) mm

    # Local thickness at each receiver
    mx_idx = np.clip(measure_positions[:, 0].astype(int), 0, GRID_W - 1)
    my_idx = np.clip(measure_positions[:, 1].astype(int), 0, GRID_H - 1)
    thick_local = tmap[my_idx, mx_idx].astype(np.float32)

    # Generate signals via frequency-domain phase shift
    n_freqs = len(_SPEC_FREQS)
    signals = np.zeros((N, N_TIME), dtype=np.float32)

    for i in range(N):
        if L_mm[i] < 0.1:
            # Co-located: copy source pulse with small delay
            signals[i] = _SOURCE.copy()
            continue

        # Phase velocity at each frequency for this path's effective thickness
        cp = get_phase_velocity(_SPEC_FREQS, d_eff[i])          # (n_freqs,) m/s

        # Phase shift: phi(f) = -2*pi*f * L / c_p(f)
        phase = -2.0 * np.pi * _SPEC_FREQS * (L_mm[i] * 1e-3) / cp  # (n_freqs,)

        spec = _SOURCE_FFT * np.exp(1j * phase)  # complex
        signals[i] = np.fft.irfft(spec, n=N_TIME).real.astype(np.float32)

    # Attenuation: geometric spreading
    atten = 1.0 / np.sqrt(L_mm + 1.0)
    signals *= atten[:, np.newaxis].astype(np.float32)

    # Normalise per signal
    mx_val = np.max(np.abs(signals), axis=1, keepdims=True)
    mx_val = np.maximum(mx_val, 1e-8)
    signals /= mx_val

    return signals, thick_local


if __name__ == "__main__":
    import time
    from simulation_aluminum import generate_terrace_map

    rng = np.random.RandomState(0)
    tmap = generate_terrace_map(rng=rng)
    print(f"Thickness map: [{tmap.min():.2f}, {tmap.max():.2f}] mm")

    # Dense 32×32 measurement grid
    xs = np.linspace(0, GRID_W - 1, 32)
    ys = np.linspace(0, GRID_H - 1, 32)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    src = np.array([2.0, 16.0], dtype=np.float32)

    t0 = time.time()
    sigs, thick = simulate_lamb_signals(tmap, src, pts)
    dt = time.time() - t0
    print(f"Signals: {sigs.shape}  |  Thickness range: [{thick.min():.2f}, {thick.max():.2f}] mm")
    print(f"Time: {dt:.2f}s for {len(pts)} points")

    # Show signal variation with thickness
    for i in [0, 500, 800]:
        print(f"  Point #{i}: thick={thick[i]:.2f}mm  "
              f"d_eff={path_averaged_thickness(tmap, src, pts[i:i+1])[0]:.2f}mm  "
              f"sig_max={sigs[i].max():.4f}")
