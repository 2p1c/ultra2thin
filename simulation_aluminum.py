"""
Aluminum plate through-thickness pulse-echo simulation.

Physics: laser point source on the back (flat) side generates broadband ultrasound
(200-700 kHz). Waves travel through the plate thickness, reflect from the top
surface (or bottom of flat-bottom holes), and return. Terrace-shaped defects
(stepped flat-bottom holes) are machined from the top surface.

Key differences from simulation_v2.py:
  - Through-thickness (not in-plane) wave propagation
  - Image-method path length: sqrt(dx^2 + dy^2 + 4*d^2)
  - Aluminum: v = 6300 m/s, nominal thickness 4 mm
  - Terrace defects: concentric rectangular stepped flat-bottom holes
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# ── Material & geometry constants ─────────────────────────────────────────
V_AL = 6300.0        # m/s, longitudinal wave in aluminum
D_NOMINAL = 4.0      # mm, nominal plate thickness
D_MIN = 1.0          # mm, minimum remaining thickness
PLATE_W = 0.1        # m (100 mm), plate side length
GRID_H, GRID_W = 32, 32

# ── Signal parameters ─────────────────────────────────────────────────────
FC_LOW = 200e3       # Hz
FC_HIGH = 700e3      # Hz
FS = 25e6            # Hz sampling rate
N_TIME = 256         # time samples


# ── Broadband source pulse (precomputed at import) ────────────────────────

def _build_source_pulse():
    """Gaussian-enveloped sinusoid: centre 450 kHz, -3dB bandwidth ~500 kHz."""
    fc = (FC_LOW + FC_HIGH) / 2        # 450 kHz
    bw = FC_HIGH - FC_LOW               # 500 kHz
    sigma = 1.0 / (np.pi * bw)          # time-domain sigma
    t0 = 4.0 * sigma                    # centre the pulse
    t = np.arange(N_TIME) / FS
    env = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    pulse = env * np.sin(2.0 * np.pi * fc * (t - t0))
    return (pulse / np.max(np.abs(pulse))).astype(np.float32)

_SOURCE = _build_source_pulse()
_T_AXIS = np.arange(N_TIME) / FS  # seconds
_SPEC_FREQS = np.fft.rfftfreq(N_TIME, 1.0 / FS)
_SOURCE_FFT = np.fft.rfft(_SOURCE).astype(np.complex64)

print(f"[sim_al] Source pulse ready: {N_TIME} samples @ {FS/1e6:.0f} MHz  "
      f"({N_TIME/FS*1e6:.1f} us window)")


# ── Terrace defect generation ─────────────────────────────────────────────

def generate_terrace_map(n_defects=None, rng=None):
    """Generate a thickness map with terrace-shaped (stepped) flat-bottom holes.

    Each terrace is a set of concentric rectangular steps. The deepest step
    (thinnest remaining material) is at the centre, with progressively
    shallower steps outward — resembling rice terraces.

    Returns (32, 32) float32 array: remaining thickness in mm [D_MIN, D_NOMINAL].
    """
    if rng is None:
        rng = np.random
    thickness = np.full((GRID_H, GRID_W), D_NOMINAL, dtype=np.float32)

    if n_defects is None:
        n_defects = rng.randint(1, 4)

    for _ in range(n_defects):
        cx = rng.uniform(0.15 * GRID_W, 0.85 * GRID_W)
        cy = rng.uniform(0.15 * GRID_H, 0.85 * GRID_H)
        n_steps = rng.randint(2, 5)

        # Each step removes material cumulatively
        step_depths = np.sort(rng.uniform(0.3, 1.8, n_steps))[::-1]
        cumulative = np.cumsum(step_depths)

        base_half_x = rng.uniform(2.5, 8.0)
        base_half_y = rng.uniform(2.0, 6.0)
        theta = rng.uniform(0, np.pi)

        yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
        dx = xx - cx
        dy = yy - cy
        dxr = dx * np.cos(theta) + dy * np.sin(theta)
        dyr = -dx * np.sin(theta) + dy * np.cos(theta)

        for step_idx, total_depth in enumerate(cumulative):
            scale = (n_steps - step_idx) / n_steps
            hx = base_half_x * scale
            hy = base_half_y * scale
            in_step = (np.abs(dxr) <= hx) & (np.abs(dyr) <= hy)
            new_thick = D_NOMINAL - total_depth
            if new_thick < D_MIN:
                new_thick = D_MIN
            thickness[in_step] = np.minimum(thickness[in_step], new_thick)

    # Mild smoothing for realistic transitions (preserves terrace character)
    thickness = gaussian_filter(thickness, sigma=0.4)
    return np.clip(thickness, D_MIN, D_NOMINAL).astype(np.float32)


# ── Signal simulation ─────────────────────────────────────────────────────

def simulate_point_signals(thickness_map, source_pos, measure_positions):
    """Through-thickness pulse-echo signals via image method.

    Path length from source (sx,sy,0) reflecting off top surface at z=d
    to receiver (mx,my,0):
        L = sqrt( dx^2 + dy^2 + (2d)^2 )
        tau = L / v

    Args:
        thickness_map: (32, 32) remaining thickness [mm]
        source_pos: (sx, sy) in grid coords [0, 31]
        measure_positions: (N, 2) in grid coords

    Returns:
        signals: (N, N_TIME) time-domain scattered signals
        thickness_at_points: (N,) true thickness at each measurement point
    """
    dx_phys = PLATE_W / GRID_W * 1e3      # mm per grid unit (~3.125)
    v_mm_us = V_AL * 1e-3                  # mm/us (6.3)

    # Source in physical mm
    sx_mm = float(source_pos[0]) * dx_phys
    sy_mm = float(source_pos[1]) * dx_phys

    # Measurement points in physical mm
    mx_mm = measure_positions[:, 0].astype(np.float64) * dx_phys
    my_mm = measure_positions[:, 1].astype(np.float64) * dx_phys

    # Thickness at each measurement point
    mx_idx = np.clip(measure_positions[:, 0].astype(int), 0, GRID_W - 1)
    my_idx = np.clip(measure_positions[:, 1].astype(int), 0, GRID_H - 1)
    thick = thickness_map[my_idx, mx_idx].astype(np.float32)

    # Path length via image method
    dx = mx_mm - sx_mm
    dy = my_mm - sy_mm
    path_mm = np.sqrt(dx ** 2 + dy ** 2 + (2.0 * thick.astype(np.float64)) ** 2)
    tau_us = path_mm / v_mm_us                    # microsecond delays

    # Generate signals by frequency-domain phase shift
    N = len(measure_positions)
    omega = 2.0 * np.pi * _SPEC_FREQS.astype(np.float64)  # rad/s
    # Phase shift for each delay: exp(-j * omega * tau)
    phase_shift = np.exp(-1j * omega[np.newaxis, :] * tau_us[:, np.newaxis] * 1e-6)

    spec = _SOURCE_FFT.astype(np.complex128)[np.newaxis, :] * phase_shift
    signals = np.fft.irfft(spec, n=N_TIME, axis=1).real.astype(np.float32)

    # Attenuation: signal weakens with path length
    atten = 1.0 / (1.0 + path_mm / 5.0)
    signals *= atten[:, np.newaxis].astype(np.float32)

    # Add weak multiple reflection (~25% amplitude, double path)
    phase2 = np.exp(-1j * omega[np.newaxis, :] * (tau_us[:, np.newaxis] * 2e-6))
    spec2 = _SOURCE_FFT.astype(np.complex128)[np.newaxis, :] * phase2 * 0.25
    reverb = np.fft.irfft(spec2, n=N_TIME, axis=1).real.astype(np.float32)
    reverb *= (atten[:, np.newaxis] ** 2).astype(np.float32)
    signals += reverb

    # Normalise per sample
    mx_val = np.max(np.abs(signals), axis=1, keepdims=True)
    mx_val = np.maximum(mx_val, 1e-8)
    signals /= mx_val

    return signals, thick


# ── Dataset generation ────────────────────────────────────────────────────

def generate_plate_dataset(plate_maps, n_configs_per_plate, min_pts=16, max_pts=36,
                           rng=None):
    """Generate measurement configurations across multiple plates.

    Args:
        plate_maps: list of (32,32) thickness maps
        n_configs_per_plate: number of source+grid configs per plate
        min_pts, max_pts: measurement point count range

    Returns:
        all_signals, all_positions, all_thickness, all_source_pos, all_tmaps
    """
    if rng is None:
        rng = np.random

    all_signals, all_positions, all_thickness = [], [], []
    all_source_pos, all_tmaps = [], []

    for tmap in plate_maps:
        for _ in range(n_configs_per_plate):
            # Random source on left portion of the plate
            sx = rng.uniform(0, GRID_W * 0.2)
            sy = rng.uniform(GRID_H * 0.1, GRID_H * 0.9)
            src = np.array([sx, sy], dtype=np.float32)

            # Random measurement grid
            n_pts = rng.randint(min_pts, max_pts + 1)
            n_cols = int(np.sqrt(n_pts))
            n_rows = (n_pts + n_cols - 1) // n_cols

            x_base = np.linspace(GRID_W * 0.1, GRID_W * 0.9, n_cols)
            y_base = np.linspace(GRID_H * 0.1, GRID_H * 0.9, n_rows)
            xx, yy = np.meshgrid(x_base, y_base)
            pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)[:n_pts]

            spacing = (GRID_W * 0.8) / max(n_cols - 1, 1)
            jitter = rng.uniform(-0.08 * spacing, 0.08 * spacing, pts.shape)
            pts = np.clip(pts + jitter, [0, 0], [GRID_W - 1, GRID_H - 1]).astype(np.float32)

            sigs, thick = simulate_point_signals(tmap, src, pts)

            all_signals.append(sigs)
            all_positions.append(pts)
            all_thickness.append(thick)
            all_source_pos.append(src)
            all_tmaps.append(tmap)

    return all_signals, all_positions, all_thickness, all_source_pos, all_tmaps


# ── Quick sanity check ────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    rng = np.random.RandomState(0)
    t0 = time.time()

    # Generate one plate, test signal generation
    tmap = generate_terrace_map(n_defects=2, rng=rng)
    print(f"Terrace map range: [{tmap.min():.2f}, {tmap.max():.2f}] mm")

    src = np.array([2.0, 16.0], dtype=np.float32)
    pts = np.stack([
        np.linspace(2, 30, 6),
        np.linspace(2, 30, 6),
    ], axis=-1).astype(np.float32)
    sigs, thick = simulate_point_signals(tmap, src, pts)
    print(f"Signals: {sigs.shape}  |  Thickness at pts: {thick}")
    print(f"Signal range: [{sigs.min():.4f}, {sigs.max():.4f}]")
    print(f"Time: {time.time() - t0:.2f}s")
