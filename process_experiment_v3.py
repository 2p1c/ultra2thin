#!/usr/bin/env python3
"""
Robust source extraction from multi-distance experimental data.

Strategy:
  1. Zero-pad & Tukey-window each receiver signal for clean FFT
  2. Bandpass to isolate the A0-dominant frequency band
  3. For each receiver, invert dispersion + geometric spreading
  4. Weighted average (1/L weight — closer receivers more reliable)
  5. Also provide the raw closest-receiver signal as comparison

Output: time-domain source signal + COMSOL-ready CSV/TXT files.
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
D_MM = 1.0
FS = 6.25e6
T_CUT_US = 65.0
N_RECV = 63
L0_MM = 22.0
SPREAD_MM = 61.0

# Processing params
ZERO_PAD_FACTOR = 4       # zero-pad to 4x signal length for smooth spectra
BANDPASS_LOW = 200e3      # Hz — matches vibrometer hardware filter
BANDPASS_HIGH = 700e3     # Hz — matches vibrometer hardware filter
OUT_DIR = 'experiment_results'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────────────────
mat = scipy.io.loadmat(MAT_PATH)
t_raw = mat['x'].squeeze()
signals_raw = mat['y']

dt = t_raw[1] - t_raw[0]
cut_idx = int(T_CUT_US * 1e-6 / dt)

t_original = t_raw[:cut_idx].copy()
signals = signals_raw[:, :cut_idx].astype(np.float64)
N_orig = len(t_original)

dx = SPREAD_MM / (N_RECV - 1)
L_mm = L0_MM + np.arange(N_RECV) * dx
L_m = L_mm * 1e-3

# ── Zero-pad and window each signal ──────────────────────────────────────────
N_pad = N_orig * ZERO_PAD_FACTOR
from scipy.signal.windows import tukey
window = tukey(N_orig, alpha=0.2)

signals_windowed = np.zeros((N_RECV, N_pad), dtype=np.float64)
for i in range(N_RECV):
    sig = signals[i] - signals[i].mean()
    signals_windowed[i, :N_orig] = sig * window

t_padded = np.arange(N_pad) / FS
freqs = np.fft.rfftfreq(N_pad, 1.0 / FS)
N_freqs = len(freqs)

# ── Bandpass mask ────────────────────────────────────────────────────────────
bp_mask = (freqs >= BANDPASS_LOW) & (freqs <= BANDPASS_HIGH)
print(f'Bandpass: {BANDPASS_LOW/1e3:.0f}–{BANDPASS_HIGH/1e3:.0f} kHz  '
      f'({bp_mask.sum()}/{N_freqs} freq bins)')

# ── Compute spectra ──────────────────────────────────────────────────────────
spectra = np.fft.rfft(signals_windowed, axis=1).astype(np.complex128)

# ── A0 phase velocity ────────────────────────────────────────────────────────
cp = get_phase_velocity(freqs, D_MM)

# ── Invert each receiver to source ───────────────────────────────────────────
# Weights: closer receivers get higher weight (less accumulated phase error)
weights = 1.0 / L_mm
weights /= weights.sum()

source_spectra = np.zeros((N_RECV, N_freqs), dtype=np.complex128)

for i in range(N_RECV):
    S_rec = spectra[i]
    L_i = L_m[i]

    # Reverse geometric spreading
    S_src = S_rec * np.sqrt(L_i)

    # Reverse A0 dispersion
    phase = 2.0 * np.pi * freqs * L_i / cp
    S_src *= np.exp(1j * phase)

    # Apply bandpass (tapered edges)
    bp_taper = np.ones(N_freqs)
    # Soft low-end rolloff
    low_idx = np.searchsorted(freqs, BANDPASS_LOW)
    bp_taper[:low_idx] = np.linspace(0, 1, low_idx)
    # Soft high-end rolloff
    high_idx = np.searchsorted(freqs, BANDPASS_HIGH)
    roll_width = min(50, N_freqs - high_idx)
    bp_taper[high_idx:high_idx + roll_width] = np.linspace(1, 0, roll_width)
    bp_taper[high_idx + roll_width:] = 0

    source_spectra[i] = S_src * bp_taper

# ── Weighted average ─────────────────────────────────────────────────────────
S_src_weighted = np.average(source_spectra, axis=0, weights=weights)
s_src_weighted = np.fft.irfft(S_src_weighted, n=N_pad).real
s_src_weighted = s_src_weighted[:N_orig]  # trim back to original length

# Unweighted average for comparison
S_src_mean = np.mean(source_spectra, axis=0)
s_src_mean = np.fft.irfft(S_src_mean, n=N_pad).real[:N_orig]

# Source estimates from each receiver (trimmed)
source_estimates = np.array([
    np.fft.irfft(source_spectra[i], n=N_pad).real[:N_orig]
    for i in range(N_RECV)
])

# ── Closest receiver (minimal processing) ────────────────────────────────────
closest_raw = signals[0] - signals[0].mean()

# ── Consistency check ────────────────────────────────────────────────────────
corrs = [np.corrcoef(source_estimates[i], s_src_weighted)[0, 1]
         for i in range(N_RECV)]
print(f'Correlation with weighted mean: median={np.median(corrs):.3f}, '
      f'IQR=[{np.percentile(corrs,25):.3f}, {np.percentile(corrs,75):.3f}]')

# Weighted consistency: are closer receivers more consistent?
for i in [0, 10, 30, 50, 62]:
    print(f'  ch{i} (L={L_mm[i]:.0f}mm, w={weights[i]:.3f}): corr={corrs[i]:.4f}')

# ── Save outputs ─────────────────────────────────────────────────────────────
np.savez(os.path.join(OUT_DIR, 'source_calibration_v3.npz'),
         t=t_original, t_padded=t_padded,
         s_src_weighted=s_src_weighted,
         s_src_mean=s_src_mean,
         s_closest=closest_raw,
         source_estimates=source_estimates,
         weights=weights, L_mm=L_mm,
         freqs=freqs, S_src_weighted=S_src_weighted,
         cp=cp, fs=FS, d_mm=D_MM)

# COMSOL files
csv_path = os.path.join(OUT_DIR, 'source_signal_comsol.csv')
np.savetxt(csv_path,
           np.column_stack([t_original, s_src_weighted]),
           delimiter=',', header='time_s,displacement', comments='')

csv_raw = os.path.join(OUT_DIR, 'source_signal_raw_ch0_comsol.csv')
np.savetxt(csv_raw,
           np.column_stack([t_original, closest_raw]),
           delimiter=',', header='time_s,displacement', comments='')

np.savetxt(os.path.join(OUT_DIR, 'source_signal_comsol.txt'),
           np.column_stack([t_original, s_src_weighted]),
           delimiter='\t', header='t\tsignal', comments='')

# ── Summary ──────────────────────────────────────────────────────────────────
print(f'\nSource signal (weighted): peak-to-peak = {np.ptp(s_src_weighted):.4e}')
print(f'Closest receiver raw:     peak-to-peak = {np.ptp(closest_raw):.4e}')
peak_freq = freqs[np.argmax(np.abs(S_src_weighted))]
print(f'Peak frequency: {peak_freq/1e3:.1f} kHz')

# ── High-quality visualization ───────────────────────────────────────────────
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12})

fig = plt.figure(figsize=(18, 11))

# --- Top row: Raw data overview ---
ax1 = fig.add_subplot(3, 3, (1, 2))
for idx in [0, 10, 20, 31, 41, 51, 62]:
    ax1.plot(t_original * 1e6, signals[idx], lw=0.5,
             label=f'ch{idx} L={L_mm[idx]:.0f}mm')
ax1.set_xlabel('Time [μs]')
ax1.set_ylabel('Displacement [m]')
ax1.set_title('Raw receiver signals (7 of 63 shown)')
ax1.legend(fontsize=6, ncol=4, loc='upper right')

ax2 = fig.add_subplot(3, 3, 3)
im = ax2.imshow(signals, aspect='auto',
                extent=[0, T_CUT_US, N_RECV-1, 0],
                cmap='RdBu_r', interpolation='bilinear')
ax2.set_xlabel('Time [μs]')
ax2.set_ylabel('Receiver index')
ax2.set_title('All 63 channels (waterfall)')
plt.colorbar(im, ax=ax2, label='Displacement [m]')

# --- Middle row: Source estimation ---
ax3 = fig.add_subplot(3, 3, (4, 5))
# Plot individual estimates (faint) + weighted mean (bold)
for i in range(0, N_RECV, 3):
    alpha = 0.15 + 0.15 * weights[i] / weights.max()
    ax3.plot(t_original * 1e6, source_estimates[i], lw=0.3,
             alpha=alpha, color='steelblue')
ax3.plot(t_original * 1e6, s_src_weighted, 'darkorange', lw=2.0,
         label=f'Weighted mean ({N_RECV} receivers)')
ax3.plot(t_original * 1e6, closest_raw, 'gray', lw=0.8, alpha=0.7,
         label=f'Closest receiver (L={L_mm[0]:.0f}mm)')
ax3.set_xlabel('Time [μs]')
ax3.set_ylabel('Displacement [m]')
ax3.set_title('Source time-domain signal (individual estimates + weighted mean)')
ax3.legend(fontsize=9)

# --- Bottom row: Detailed source signal ---
ax4 = fig.add_subplot(3, 3, (7, 8))
std_est = np.std(source_estimates, axis=0)
ax4.fill_between(t_original * 1e6,
                 s_src_weighted - std_est,
                 s_src_weighted + std_est,
                 alpha=0.2, color='darkorange', label='±1σ across receivers')
ax4.plot(t_original * 1e6, s_src_weighted, 'darkorange', lw=1.5)
ax4.set_xlabel('Time [μs]')
ax4.set_ylabel('Displacement [m]')
ax4.set_title('Extracted source signal with uncertainty')
ax4.legend(fontsize=9)

# --- Side panels ---
# Source spectrum
ax5 = fig.add_subplot(3, 3, 6)
# Compute spectrum of the trimmed weighted source for display
S_display = np.abs(np.fft.rfft(s_src_weighted * np.hanning(N_orig)))
f_display = np.fft.rfftfreq(N_orig, 1.0 / FS)
ax5.semilogy(f_display / 1e3, S_display, 'darkorange', lw=1.2)
ax5.axvline(BANDPASS_LOW / 1e3, color='gray', ls='--', lw=0.8, label='BP')
ax5.axvline(BANDPASS_HIGH / 1e3, color='gray', ls='--', lw=0.8)
ax5.set_xlabel('Frequency [kHz]')
ax5.set_ylabel('|S(f)|')
ax5.set_title('Source amplitude spectrum')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# A0 dispersion curve
ax6 = fig.add_subplot(3, 3, 9)
ax6.plot(freqs / 1e3, cp / 1e3, 'steelblue', lw=1.2)
ax6.fill_between([BANDPASS_LOW/1e3, BANDPASS_HIGH/1e3], 0, 6,
                 alpha=0.1, color='darkorange')
ax6.set_xlabel('Frequency [kHz]')
ax6.set_ylabel('c_p [km/s]')
ax6.set_title(f'A0 phase velocity (Al, d={D_MM}mm) — used band in orange')
ax6.set_ylim(0, 6)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'source_calibration_v3.png')
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Plot: {plot_path}')

# ── Also generate a clean standalone source plot ─────────────────────────────
fig2, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(t_original * 1e6, s_src_weighted, 'darkorange', lw=1.5)
ax.fill_between(t_original * 1e6,
                s_src_weighted - std_est,
                s_src_weighted + std_est,
                alpha=0.25, color='darkorange')
ax.set_xlabel('Time [μs]', fontsize=14)
ax.set_ylabel('Equivalent displacement [m]', fontsize=14)
ax.set_title(f'Extracted Laser Source Signal  (Al plate, d={D_MM}mm, '
             f'{N_RECV} receivers @ L={L_mm[0]:.0f}–{L_mm[-1]:.0f}mm)',
             fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', lw=0.5)
fig2.tight_layout()
clean_plot = os.path.join(OUT_DIR, 'source_signal_clean.png')
fig2.savefig(clean_plot, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Clean plot: {clean_plot}')
