#!/usr/bin/env python3
"""
Process experimental laser-ultrasound calibration data to extract equivalent source.

63 receivers in a line, known distances from source. Each receiver signal is
inverted through reverse dispersion + geometric spreading to estimate the
source time-domain signal. All 63 estimates are averaged for the final result.

Usage:
  python process_experiment.py
"""

import sys, os
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lamb_dispersion import get_phase_velocity

# ── Parameters ───────────────────────────────────────────────────────────────
MAT_PATH = '/Volumes/ESD-ISO/数据/260515/std_biaoding2/data.mat'
D_MM = 1.0           # plate thickness [mm]
FS = 6.25e6          # sampling rate [Hz]
T_CUT_US = 65.0      # cut time [μs] to remove edge reflections
N_RECV = 63          # number of receivers
L0_MM = 22.0         # distance from source to first receiver [mm]
SPREAD_MM = 61.0     # distance from first to last receiver [mm]
OUT_DIR = 'experiment_results'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
mat = scipy.io.loadmat(MAT_PATH)
t_raw = mat['x'].squeeze()       # (1000,) time axis [s]
signals_raw = mat['y']           # (63, 1000)

dt = t_raw[1] - t_raw[0]
cut_idx = int(T_CUT_US * 1e-6 / dt)
print(f'dt = {dt*1e6:.3f} μs, fs = {1/dt*1e-6:.3f} MHz')
print(f'Cut at {T_CUT_US} μs → index {cut_idx}')

# Trim to 0–65 μs
t = t_raw[:cut_idx]
signals = signals_raw[:, :cut_idx].astype(np.float64)
N = len(t)

# Compute receiver positions
dx = SPREAD_MM / (N_RECV - 1)
L_mm = L0_MM + np.arange(N_RECV) * dx   # distance from source for each receiver

print(f'Receiver distances: {L_mm[0]:.1f} – {L_mm[-1]:.1f} mm, dx = {dx:.3f} mm')
print(f'FFT length: {N} samples, freq resolution: {FS/N/1e3:.2f} kHz')
print(f'Freq range: 0 – {FS/2/1e3:.1f} kHz')

# ── Process each channel ─────────────────────────────────────────────────────
freqs = np.fft.rfftfreq(N, 1.0 / FS)
cp = get_phase_velocity(freqs, D_MM)  # (N_freqs,) m/s

source_estimates = np.zeros((N_RECV, N), dtype=np.float64)
source_spectra = np.zeros((N_RECV, len(freqs)), dtype=np.complex128)

for i in range(N_RECV):
    sig = signals[i] - signals[i].mean()  # remove DC
    S_rec = np.fft.rfft(sig).astype(np.complex128)

    L_m = L_mm[i] * 1e-3

    # Reverse geometric spreading: multiply by sqrt(L)
    S_src = S_rec * np.sqrt(L_m)

    # Reverse dispersion: add back the phase delay
    phase = 2.0 * np.pi * freqs * L_m / cp
    S_src *= np.exp(1j * phase)

    source_spectra[i] = S_src
    source_estimates[i] = np.fft.irfft(S_src, n=N).real.astype(np.float64)

# ── Average across receivers ─────────────────────────────────────────────────
S_src_mean = np.mean(source_spectra, axis=0)
S_src_std = np.std(source_spectra, axis=0)
s_src_mean = np.fft.irfft(S_src_mean, n=N).real.astype(np.float64)

# Also compute time-domain mean and std
s_src_std_across = np.std(source_estimates, axis=0)

print(f'\nSource spectrum peak at {freqs[np.argmax(np.abs(S_src_mean))]/1e3:.1f} kHz')
print(f'Source time signal peak-to-peak: {np.ptp(s_src_mean):.6f}')

# ── Save results ─────────────────────────────────────────────────────────────
# NPZ for Python use
np.savez(os.path.join(OUT_DIR, 'source_calibration.npz'),
         t=t, s_src=s_src_mean, s_src_std=s_src_std_across,
         freqs=freqs, S_src=S_src_mean, S_src_std=S_src_std,
         fs=FS, d_mm=D_MM, L_mm=L_mm,
         source_estimates=source_estimates)

# COMSOL-compatible CSV: time [s], displacement
csv_path = os.path.join(OUT_DIR, 'source_signal_comsol.csv')
np.savetxt(csv_path,
           np.column_stack([t, s_src_mean]),
           delimiter=',', header='time_s,displacement', comments='')
print(f'COMSOL CSV saved to: {csv_path}')

# Also save as piecewise interpolation table (2 columns: t, signal)
# COMSOL can read this directly as an Interpolation function
tsv_path = os.path.join(OUT_DIR, 'source_signal_comsol.txt')
np.savetxt(tsv_path,
           np.column_stack([t, s_src_mean]),
           delimiter='\t', header='t\tsignal', comments='')
print(f'COMSOL TXT saved to: {tsv_path}')

# ── Diagnostic plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Raw signals (first/last/middle)
ax = axes[0, 0]
for idx in [0, 15, 31, 47, 62]:
    ax.plot(t_raw * 1e6, signals_raw[idx], lw=0.5,
            label=f'ch{idx} (L={L_mm[idx]:.0f}mm)')
ax.axvline(T_CUT_US, color='red', ls='--', lw=0.8)
ax.set_xlabel('Time [μs]'); ax.set_ylabel('Displacement')
ax.set_title('Raw signals with 65 μs cut')
ax.legend(fontsize=6, ncol=2)

# 2. All channels as image
ax = axes[0, 1]
im = ax.imshow(signals_raw[:, :cut_idx], aspect='auto',
               extent=[0, T_CUT_US, N_RECV-1, 0], cmap='RdBu_r')
ax.set_xlabel('Time [μs]'); ax.set_ylabel('Channel')
ax.set_title('All channels (0–65 μs)')
plt.colorbar(im, ax=ax)

# 3. A0 dispersion curve
ax = axes[0, 2]
ax.plot(freqs / 1e3, cp, 'steelblue', lw=1.2)
ax.set_xlabel('Frequency [kHz]'); ax.set_ylabel('c_p [m/s]')
ax.set_title(f'A0 phase velocity (d={D_MM}mm)')
ax.grid(True, alpha=0.3)

# 4. Individual source estimates
ax = axes[1, 0]
for i in [0, 15, 31, 47, 62]:
    ax.plot(t * 1e6, source_estimates[i], lw=0.4,
            label=f'from ch{i} (L={L_mm[i]:.0f}mm)')
ax.plot(t * 1e6, s_src_mean, 'k', lw=1.5, label='Mean')
ax.set_xlabel('Time [μs]'); ax.set_ylabel('Displacement')
ax.set_title('Source estimates from each channel')
ax.legend(fontsize=6, ncol=2)

# 5. Mean source signal with std band
ax = axes[1, 1]
ax.fill_between(t * 1e6,
                s_src_mean - s_src_std_across,
                s_src_mean + s_src_std_across,
                alpha=0.3, color='darkorange')
ax.plot(t * 1e6, s_src_mean, 'darkorange', lw=1)
ax.set_xlabel('Time [μs]'); ax.set_ylabel('Displacement')
ax.set_title('Mean source signal ±1σ across receivers')

# 6. Source amplitude spectrum
ax = axes[1, 2]
ax.loglog(freqs / 1e3, np.abs(S_src_mean), 'darkorange', lw=1)
ax.fill_between(freqs / 1e3,
                np.maximum(np.abs(S_src_mean) - S_src_std, 1e-30),
                np.abs(S_src_mean) + S_src_std,
                alpha=0.3, color='darkorange')
ax.set_xlabel('Frequency [kHz]'); ax.set_ylabel('|S(f)|')
ax.set_title('Source amplitude spectrum')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'source_calibration.png'), dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'Plots saved to: {OUT_DIR}/source_calibration.png')

# ── Print COMSOL instructions ─────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║  COMSOL 加载指南                                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  方法 1: Interpolation 函数 (推荐)                           ║
║  ──────────────────────────────────                          ║
║  Definitions → Functions → Interpolation                    ║
║    - Data source: File                                      ║
║    - 选择 source_signal_comsol.txt                          ║
║    - Arguments: t (s)                                       ║
║    - Extrapolation: 填零或 constant=0                       ║
║                                                              ║
║  使用时引用:  src(t)  或用具体函数名                          ║
║                                                              ║
║  方法 2: Piecewise 函数                                      ║
║  ────────────────────                                        ║
║  Definitions → Functions → Piecewise                        ║
║    - 手动复制 CSV 中的 (t, signal) 数据对                    ║
║                                                              ║
║  应用: 在 Physics 中将该函数设为:                             ║
║    - 点载荷: Point Load →  x/y 分量 = src(t) * 单位方向      ║
║    - 位移边界: Prescribed Displacement → u0 = src(t)         ║
║                                                              ║
║  注意: 信号幅值取决于接收信号的标定 (激光测振仪输出为         ║
║  速度/位移?), 需要在 COMSOL 中做量纲匹配。                    ║
╚══════════════════════════════════════════════════════════════╝
""")
