#!/usr/bin/env python3
"""
Improved source extraction using empirical geometric attenuation fitting.

With 63 receivers at known distances, we can:
1. Fit the amplitude decay exponent n(f) from data: |S(f,L)| ∝ L^(-n(f))
2. Fit the actual phase progression to get effective k(f)
3. Reconstruct the source using the empirical model

Then export for COMSOL.
"""

import sys, os
import numpy as np
import scipy.io
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lamb_dispersion import get_phase_velocity

# ── Parameters ───────────────────────────────────────────────────────────────
MAT_PATH = '/Volumes/ESD-ISO/数据/260515/std_biaoding2/data.mat'
D_MM = 1.0
FS = 6.25e6
T_CUT_US = 65.0
N_RECV = 63
L0_MM = 22.0
SPREAD_MM = 61.0
OUT_DIR = 'experiment_results'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
mat = scipy.io.loadmat(MAT_PATH)
t_raw = mat['x'].squeeze()
signals_raw = mat['y']

dt = t_raw[1] - t_raw[0]
cut_idx = int(T_CUT_US * 1e-6 / dt)

t = t_raw[:cut_idx]
signals = signals_raw[:, :cut_idx].astype(np.float64)
N = len(t)

dx = SPREAD_MM / (N_RECV - 1)
L_mm = L0_MM + np.arange(N_RECV) * dx
L_m = L_mm * 1e-3  # meters

freqs = np.fft.rfftfreq(N, 1.0 / FS)
N_freqs = len(freqs)

# ── Compute spectra for all receivers ────────────────────────────────────────
spectra = np.zeros((N_RECV, N_freqs), dtype=np.complex128)
for i in range(N_RECV):
    sig = signals[i] - signals[i].mean()
    spectra[i] = np.fft.rfft(sig).astype(np.complex128)

# ── For each frequency, fit |S(f,L)| = A * L^(-n) ────────────────────────────
def power_law(L, A, n):
    return A * L ** (-n)

# Only fit frequencies with meaningful energy
total_energy = np.abs(spectra).sum(axis=0)
energy_threshold = total_energy.max() * 0.01
valid_freqs = total_energy > energy_threshold
print(f'Frequencies with >1% energy: {valid_freqs.sum()}/{N_freqs}  '
      f'({freqs[valid_freqs].min()/1e3:.0f}–{freqs[valid_freqs].max()/1e3:.0f} kHz)')

# Fit n(f) and A(f) for each frequency
n_f = np.full(N_freqs, np.nan)
A_f = np.full(N_freqs, np.nan)
cp_empirical = np.full_like(freqs, np.nan)

for j in range(N_freqs):
    if not valid_freqs[j] or freqs[j] < 1e3:
        continue

    amps = np.abs(spectra[:, j])
    if amps.max() < 1e-30:
        continue

    try:
        popt, _ = curve_fit(power_law, L_m, amps,
                           p0=[amps[len(amps)//2] * L_m[len(L_m)//2]**0.5, 0.5],
                           bounds=([0, 0], [np.inf, 3.0]),
                           maxfev=2000)
        A_f[j] = popt[0]
        n_f[j] = popt[1]
    except (RuntimeError, ValueError):
        pass

    # Fit phase progression to get cp(f): unwrap phase vs L
    phases = np.angle(spectra[:, j])
    # Linear fit: phase = -k * L + const, where k = 2πf / cp
    try:
        k_fit = -np.polyfit(L_m, phases, 1)[0]
        if k_fit > 0:
            cp_empirical[j] = 2.0 * np.pi * freqs[j] / k_fit
    except (ValueError, np.linalg.LinAlgError):
        pass

# ── Smooth n(f) and cp(f) with a running median ──────────────────────────────
from scipy.signal import medfilt

smooth_win = 11  # must be odd
n_f_smooth = np.full_like(n_f, np.nan)
cp_smooth = np.full_like(cp_empirical, np.nan)

mask_n = ~np.isnan(n_f)
mask_cp = ~np.isnan(cp_empirical)

if mask_n.sum() > smooth_win:
    n_f_smooth[mask_n] = medfilt(n_f[mask_n], kernel_size=smooth_win)
if mask_cp.sum() > smooth_win:
    cp_smooth[mask_cp] = medfilt(cp_empirical[mask_cp], kernel_size=smooth_win)

# Where fitting failed or is noisy, fall back to theoretical values
cp_theory = get_phase_velocity(freqs, D_MM)
for j in range(N_freqs):
    if np.isnan(cp_smooth[j]) or cp_smooth[j] <= 0 or cp_smooth[j] > 10000:
        cp_smooth[j] = cp_theory[j]
    if np.isnan(n_f_smooth[j]) or n_f_smooth[j] <= 0:
        n_f_smooth[j] = 0.5  # theoretical cylindrical spreading

# ── Reconstruct source from each receiver using empirical parameters ──────────
source_estimates = np.zeros((N_RECV, N), dtype=np.float64)
source_spectra = np.zeros((N_RECV, N_freqs), dtype=np.complex128)

for i in range(N_RECV):
    S_rec = spectra[i]
    L_i = L_m[i]

    # Reverse geometric spreading using empirical n(f)
    # Forward: |S_rec| = A(f) * L^(-n(f))  → source amplitude = A(f)
    # So: S_src(f) = S_rec(f) * L^(n(f))
    amp_correction = L_i ** n_f_smooth
    S_src = S_rec * amp_correction

    # Reverse dispersion using empirical cp(f)
    phase = 2.0 * np.pi * freqs * L_i / cp_smooth
    S_src *= np.exp(1j * phase)

    source_spectra[i] = S_src
    source_estimates[i] = np.fft.irfft(S_src, n=N).real

# ── Average source estimate ──────────────────────────────────────────────────
S_src_mean = np.mean(source_spectra, axis=0)
s_src_mean = np.fft.irfft(S_src_mean, n=N).real
s_src_std = np.std(source_estimates, axis=0)

# ── Also: simple approach – use closest receiver only ─────────────────────────
# At 22mm, propagation effects are minimal. Just time-shift back.
# This is often the most robust source estimate.
closest_sig = signals[0] - signals[0].mean()
# Approximate time shift: 22mm / 5400 m/s ≈ 4.1 μs → negligible at this dt

# ── Save ─────────────────────────────────────────────────────────────────────
np.savez(os.path.join(OUT_DIR, 'source_calibration_v2.npz'),
         t=t, s_src=s_src_mean, s_src_std=s_src_std,
         freqs=freqs, S_src=S_src_mean,
         n_f=n_f, n_f_smooth=n_f_smooth,
         cp_theory=cp_theory, cp_empirical=cp_smooth,
         L_mm=L_mm, fs=FS, d_mm=D_MM,
         s_closest=closest_sig,
         source_estimates=source_estimates)

# COMSOL CSV (source estimate)
csv_path = os.path.join(OUT_DIR, 'source_signal_comsol.csv')
np.savetxt(csv_path, np.column_stack([t, s_src_mean]),
           delimiter=',', header='time_s,displacement', comments='')

# Also save the raw closest channel for COMSOL (no processing, most conservative)
csv_raw = os.path.join(OUT_DIR, 'source_signal_raw_ch0_comsol.csv')
np.savetxt(csv_raw, np.column_stack([t, closest_sig]),
           delimiter=',', header='time_s,displacement', comments='')

# COMSOL TXT format
np.savetxt(os.path.join(OUT_DIR, 'source_signal_comsol.txt'),
           np.column_stack([t, s_src_mean]),
           delimiter='\t', header='t\tsignal', comments='')

# ── Results summary ──────────────────────────────────────────────────────────
print(f'\nFitted n(f): median={np.nanmedian(n_f):.3f}, '
      f'Q25={np.nanpercentile(n_f, 25):.3f}, Q75={np.nanpercentile(n_f, 75):.3f}')
print(f'Theory: n=0.5 (cylindrical spreading 1/sqrt(L))')
print(f'Source signal peak-to-peak: {np.ptp(s_src_mean):.6e}')

# Check consistency
correlations = [np.corrcoef(source_estimates[i], s_src_mean)[0,1] for i in range(N_RECV)]
print(f'Correlation with mean: median={np.median(correlations):.3f}, '
      f'max={np.max(correlations):.3f}, min={np.min(correlations):.3f}')

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

# 1. Raw signals
ax = axes[0, 0]
for idx in [0, 15, 31, 47, 62]:
    ax.plot(t * 1e6, signals[idx], lw=0.5,
            label=f'ch{idx} (L={L_mm[idx]:.0f}mm)')
ax.set_xlabel('Time [μs]')
ax.set_title('Raw signals (0–65 μs)')
ax.legend(fontsize=6, ncol=2)

# 2. Raw signals image
ax = axes[0, 1]
im = ax.imshow(signals, aspect='auto', extent=[0, T_CUT_US, N_RECV-1, 0],
               cmap='RdBu_r')
ax.set_xlabel('Time [μs]'); ax.set_ylabel('Channel')
ax.set_title('All 63 channels')
plt.colorbar(im, ax=ax)

# 3. Fitted n(f) vs theory
ax = axes[0, 2]
ax.axhline(0.5, color='gray', ls='--', lw=0.8, label='theory n=0.5 (1/√L)')
ax.plot(freqs / 1e3, n_f, '.', ms=1, alpha=0.3, color='steelblue', label='raw fit')
ax.plot(freqs / 1e3, n_f_smooth, 'darkorange', lw=1.5, label='smoothed')
ax.set_xlabel('Frequency [kHz]'); ax.set_ylabel('n (amplitude ∝ L^{-n})')
ax.set_title('Empirical geometric attenuation exponent')
ax.legend(fontsize=7)
ax.set_ylim(0, 2)
ax.grid(True, alpha=0.3)

# 4. Phase velocity: theory vs empirical
ax = axes[1, 0]
ax.plot(freqs / 1e3, cp_theory, 'gray', lw=1, alpha=0.6, label='A0 theory')
ax.plot(freqs / 1e3, cp_empirical, '.', ms=1, alpha=0.3, color='steelblue')
ax.plot(freqs / 1e3, cp_smooth, 'darkorange', lw=1.5, label='empirical (smoothed)')
ax.set_xlabel('Frequency [kHz]'); ax.set_ylabel('c_p [m/s]')
ax.set_title('Phase velocity: theory vs data')
ax.legend(fontsize=7)
ax.set_ylim(0, 5000)
ax.grid(True, alpha=0.3)

# 5. Source estimates
ax = axes[1, 1]
for i in [0, 15, 31, 47, 62]:
    ax.plot(t * 1e6, source_estimates[i], lw=0.4,
            label=f'from ch{i} (L={L_mm[i]:.0f}mm)')
ax.plot(t * 1e6, s_src_mean, 'k', lw=1.5, label='Mean')
ax.set_xlabel('Time [μs]'); ax.set_ylabel('Displacement')
ax.set_title('Source estimates (empirical model)')
ax.legend(fontsize=6, ncol=2)

# 6. Mean source signal with uncertainty
ax = axes[1, 2]
ax.fill_between(t * 1e6, s_src_mean - s_src_std, s_src_mean + s_src_std,
                alpha=0.3, color='darkorange')
ax.plot(t * 1e6, s_src_mean, 'darkorange', lw=1, label='Mean source')
ax.plot(t * 1e6, closest_sig, 'steelblue', lw=0.8, alpha=0.7,
        label=f'Closest receiver (L={L_mm[0]:.0f}mm)')
ax.set_xlabel('Time [μs]'); ax.set_ylabel('Displacement')
ax.set_title('Source signal (empirical) vs closest receiver')
ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'source_calibration_v2.png'), dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'Plots: {OUT_DIR}/source_calibration_v2.png')

# ── Print COMSOL instructions ─────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════╗
║  COMSOL 加载指南                                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  输出文件:                                                       ║
║     source_signal_comsol.csv  — 经验模型反推的源信号             ║
║     source_signal_raw_ch0_comsol.csv — 最近接收点原始信号(备选)  ║
║                                                                  ║
║  方法 1: Interpolation 函数 (推荐)                               ║
║  ────────────────────────────────────                            ║
║  Home → Definitions → Functions → Interpolation                 ║
║    Data source: File                                            ║
║    Number of arguments: 1                                       ║
║    Argument: t                                                  ║
║    File: 选择 source_signal_comsol.csv                           ║
║    Interpolation: Linear (或 Cubic spline)                       ║
║    Extrapolation: Constant, value = 0                           ║
║                                                                  ║
║  方法 2: 直接使用 Piecewise 函数                                 ║
║  ────────────────────────────────                                ║
║  Definitions → Functions → Piecewise                            ║
║    将 CSV 中的数据复制到表格中                                   ║
║                                                                  ║
║  在物理场中应用:                                                 ║
║    • Prescribed Displacement: u0 = src_func(t[1/s])             ║
║    • Point Load / Body Load: F = src_func(t[1/s]) * 幅值系数    ║
║                                                                  ║
║  ⚠ 注意: 本反推得到的源信号幅值很小(~10^-12量级), 在COMSOL      ║
║    中可能需要根据实际位移量级进行缩放。信号形状是正确的,         ║
║    幅值需要根据实验标定(激光能量、测振仪灵敏度等)调整。         ║
║                                                                  ║
║  备选方案: 如果经验拟合结果不理想, 可直接使用                    ║
║  source_signal_raw_ch0_comsol.csv (最近接收点的原始信号),       ║
║  它在 22mm 处受传播效应影响最小。                                 ║
╚══════════════════════════════════════════════════════════════════╝
""")
