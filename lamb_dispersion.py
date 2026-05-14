"""
Rayleigh-Lamb dispersion solver for A0 mode in aluminum plates.

Precomputes c_p(f, d) lookup table — phase velocity as function of
frequency and plate thickness.  Used by simulation_lamb.py for
Lamb-wave-based thickness measurement simulation.
"""

import numpy as np
from scipy.optimize import brentq

# Aluminum material constants
V_L = 6300.0      # m/s  longitudinal wave
V_T = 3120.0      # m/s  shear wave
RHO = 2700.0      # kg/m^3
KAPPA = V_L / V_T # ~2.019

# Grid for precomputation
N_F = 160          # frequency points
N_D = 100          # thickness points
F_MIN = 5e3        # Hz (avoid DC singularity)
F_MAX = 1.2e6      # Hz
D_MIN = 0.8        # mm
D_MAX = 5.0        # mm


def _a0_dispersion_residual(K, Omega, kappa):
    """Residual of the antisymmetric Rayleigh-Lamb equation.

    Dimensionless variables:
      Omega = omega * h / V_T
      K     = k * h
      kappa = V_L / V_T

    Returns f(K) where f(K)=0 at the A0 root.
    """
    # P^2 = Omega^2/kappa^2 - K^2,  Q^2 = Omega^2 - K^2
    Psq = Omega ** 2 / kappa ** 2 - K ** 2
    Qsq = Omega ** 2 - K ** 2

    if Psq <= 0 or Qsq <= 0:
        return 1e10  # no propagating solution

    P = np.sqrt(Psq)
    Q = np.sqrt(Qsq)

    # A0: 4*K^2*P*Q*tan(Q) + (Q^2 - K^2)^2 * tan(P) = 0
    # Handle poles of tan: they occur at pi/2 + n*pi
    # For A0 mode root-finding, we stay in the first branch
    tanP = np.tan(P)
    tanQ = np.tan(Q)

    return 4.0 * K ** 2 * P * Q * tanQ + (Qsq - K ** 2) ** 2 * tanP


def _solve_a0(Omega, kappa, K_guess):
    """Find A0 wavenumber K for given Omega. Returns K or NaN if no root."""
    # A0 root lies between 0 and Omega (c_p between V_T and infinity)
    # Actually A0 c_p < V_T, so K > Omega
    # Upper bound: K cannot exceed Omega * kappa (c_p >= V_L would require K < Omega/kappa)
    # For A0 at low Omega, K >> Omega

    K_min = Omega * 1.001        # c_p < V_T
    K_max = Omega * kappa * 1.5  # generous upper bound

    # Check there's a sign change
    try:
        f_min = _a0_dispersion_residual(K_min, Omega, kappa)
        f_max = _a0_dispersion_residual(K_max, Omega, kappa)
    except (ValueError, ZeroDivisionError):
        return np.nan

    # If no sign change, expand bounds
    for _ in range(5):
        if f_min * f_max < 0:
            break
        K_max *= 1.3
        try:
            f_max = _a0_dispersion_residual(K_max, Omega, kappa)
        except (ValueError, ZeroDivisionError):
            return np.nan

    if f_min * f_max >= 0:
        return np.nan

    try:
        K = brentq(_a0_dispersion_residual, K_min, K_max,
                   args=(Omega, kappa), xtol=1e-8, maxiter=100)
        return K
    except (ValueError, RuntimeError):
        return np.nan


def compute_dispersion_table():
    """Precompute phase velocity c_p(f, d) lookup table.

    Returns:
      freqs: (N_F,) Hz
      thicknesses: (N_D,) mm
      cp_table: (N_F, N_D) m/s  — phase velocity
    """
    freqs = np.linspace(F_MIN, F_MAX, N_F)
    thicknesses = np.linspace(D_MIN, D_MAX, N_D)
    cp_table = np.full((N_F, N_D), np.nan, dtype=np.float64)

    print(f"[disp] Computing A0 dispersion: {N_F}x{N_D} grid...", flush=True)
    for i, f in enumerate(freqs):
        for j, d in enumerate(thicknesses):
            h = d * 1e-3 / 2.0  # half-thickness in meters
            Omega = 2.0 * np.pi * f * h / V_T

            if Omega < 1e-4:
                # Very low frequency: use asymptotic flexural wave formula
                # c_p = sqrt(2*pi*f * d/2 * V_T / sqrt(3))
                cp_table[i, j] = np.sqrt(2.0 * np.pi * f * h * V_T / np.sqrt(3.0))
                continue

            # Initial guess from flexural wave approximation
            K_flex = Omega ** 0.5 * (3.0 * (KAPPA ** 2 - 1.0) / KAPPA ** 2) ** 0.25
            K_guess = max(K_flex, Omega * 1.1)

            K = _solve_a0(Omega, KAPPA, K_guess)

            if np.isnan(K):
                # Fall back to flexural approximation
                cp_table[i, j] = np.sqrt(2.0 * np.pi * f * h * V_T / np.sqrt(3.0))
            else:
                cp_table[i, j] = 2.0 * np.pi * f * h / K  # m/s

        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{N_F} frequencies done", flush=True)

    # Fill any remaining NaNs by interpolation
    nan_mask = np.isnan(cp_table)
    if nan_mask.any():
        print(f"  Filling {nan_mask.sum()} NaN values by interpolation", flush=True)
        from scipy.interpolate import griddata
        fi, di = np.meshgrid(freqs, thicknesses, indexing='ij')
        valid = ~nan_mask
        cp_table[nan_mask] = griddata(
            (fi[valid], di[valid]), cp_table[valid],
            (fi[nan_mask], di[nan_mask]), method='linear')

    print(f"  Done.  c_p range: [{np.nanmin(cp_table):.0f}, "
          f"{np.nanmax(cp_table):.0f}] m/s", flush=True)
    return freqs, thicknesses, cp_table


# ── Precompute at import ───────────────────────────────────────────────────
_FREQS, _THICKNESSES, _CP_TABLE = compute_dispersion_table()

# Build fast interpolator
from scipy.interpolate import RegularGridInterpolator
_CP_INTERP = RegularGridInterpolator(
    (_FREQS, _THICKNESSES), _CP_TABLE,
    bounds_error=False, fill_value=None)


def get_phase_velocity(freq_hz, thickness_mm):
    """Interpolate A0 phase velocity c_p(f, d).

    Args:
      freq_hz: frequency in Hz (scalar or array)
      thickness_mm: thickness in mm (scalar or array)

    Returns:
      phase velocity in m/s, broadcast to shape of inputs.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    thickness_mm = np.asarray(thickness_mm, dtype=np.float64)
    scalar = freq_hz.ndim == 0 and thickness_mm.ndim == 0

    # Broadcast to common shape
    f_b = np.broadcast_to(freq_hz, np.broadcast_shapes(freq_hz.shape, thickness_mm.shape))
    t_b = np.broadcast_to(thickness_mm, f_b.shape)

    # Clip to table range
    f_b = np.clip(f_b, F_MIN, F_MAX)
    t_b = np.clip(t_b, D_MIN, D_MAX)

    pts = np.stack([f_b.ravel(), t_b.ravel()], axis=-1)
    result = _CP_INTERP(pts).reshape(f_b.shape)

    if scalar:
        return float(result)
    return result


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot dispersion curves
    d_plot = [1.0, 1.5, 2.0, 3.0, 4.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for d in d_plot:
        idx = np.argmin(np.abs(_THICKNESSES - d))
        cp = _CP_TABLE[:, idx]
        ax1.plot(_FREQS / 1e3, cp, label=f"d={d} mm")
    ax1.set_xlabel("Frequency [kHz]")
    ax1.set_ylabel("Phase velocity c_p [m/s]")
    ax1.set_title("A0 Lamb Wave Dispersion — Aluminum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sensitivity: dc_p/dd
    for d in d_plot:
        idx_d = np.argmin(np.abs(_THICKNESSES - d))
        idx_ref = np.argmin(np.abs(_THICKNESSES - 4.0))
        dc = _CP_TABLE[:, idx_d] - _CP_TABLE[:, idx_ref]
        ax2.plot(_FREQS / 1e3, np.abs(dc), label=f"d={d} vs 4.0mm")
    ax2.set_xlabel("Frequency [kHz]")
    ax2.set_ylabel("|c_p(f,d) - c_p(f,4mm)| [m/s]")
    ax2.set_title("Thickness Sensitivity of c_p")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dispersion_curves.png", dpi=100)
    print("Saved dispersion_curves.png")
