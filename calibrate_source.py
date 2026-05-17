#!/usr/bin/env python3
"""
Extract equivalent Lamb-wave source spectrum from a real time-domain signal.

Input:  time-domain displacement signal + calibration parameters (d, L, fs)
Output: source spectrum (complex), source-position time-domain signal, plots

Physics:
  S_rec(f) = S_src(f) · 1/sqrt(L) · exp(-j·2πf·L / c_p(f,d))

  Step 1 — remove geometric spreading + reverse dispersion:
    S_src(f) = S_rec(f) · sqrt(L) · exp(+j·2πf·L / c_p(f,d))

  Step 2 — source-position signal (what the laser actually generates):
    s_src(t) = IFFT(S_src(f))

Usage:
  python calibrate_source.py signal.csv --d 4.0 --L 50 --fs 25e6
  python calibrate_source.py signal.npy --d 3.0 --L 80 --fs 10e6 --plot
"""

import argparse, sys, os
import numpy as np

# Add repo root to path so we can import lamb_dispersion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lamb_dispersion import get_phase_velocity


def load_signal(path, fs=None, time_col=0, disp_col=1, skip_rows=0):
    """Load time-domain signal from CSV or NPY.

    CSV: time_col=column index for time [s], disp_col=column for displacement
         If time column is absent (single-column data), synthesize time axis from fs.
    NPY: raw 1D array of displacement values; fs must be provided.

    Returns:
        signal: (N,) float64 displacement array
        fs: sampling rate [Hz]
        t: (N,) time axis [s]
    """
    if path.endswith('.npy'):
        sig = np.load(path).astype(np.float64)
        if fs is None:
            raise ValueError('--fs required for .npy input')
        t = np.arange(len(sig)) / fs
        return sig, fs, t

    # CSV
    data = np.loadtxt(path, delimiter=',', skiprows=skip_rows, dtype=np.float64)
    if data.ndim == 1:
        if fs is None:
            raise ValueError('Single-column CSV: --fs required')
        sig = data
        t = np.arange(len(sig)) / fs
    elif data.shape[1] >= 2:
        t = data[:, time_col]
        sig = data[:, disp_col]
        if fs is None:
            fs = 1.0 / np.median(np.diff(t))
            print(f'[load] Inferred fs = {fs/1e6:.3f} MHz from time column')
    else:
        raise ValueError(f'Unexpected CSV shape: {data.shape}')
    return sig.astype(np.float64), fs, t.astype(np.float64)


def resample(signal, fs_in, fs_out=25e6, N_out=256):
    """Resample signal to target fs and length via FFT."""
    N_in = len(signal)
    freqs_in = np.fft.rfftfreq(N_in, 1.0 / fs_in)
    spec = np.fft.rfft(signal)

    freqs_out = np.fft.rfftfreq(N_out, 1.0 / fs_out)
    # Interpolate spectrum onto new frequency grid (complex linear interp)
    spec_real = np.interp(freqs_out, freqs_in, spec.real, left=0, right=0)
    spec_imag = np.interp(freqs_out, freqs_in, spec.imag, left=0, right=0)
    spec_out = spec_real + 1j * spec_imag

    t_out = np.arange(N_out) / fs_out
    signal_out = np.fft.irfft(spec_out, n=N_out).real.astype(np.float64)
    return signal_out, fs_out, t_out


def calibrate(signal, L_mm, d_mm, fs, smooth_window=0):
    """Extract equivalent source spectrum from a real received signal.

    Args:
        signal: (N,) time-domain signal
        L_mm: source-receiver distance [mm]
        d_mm: calibration plate thickness [mm]
        fs: sampling rate [Hz]
        smooth_window: Savitzky-Golay window for amplitude smoothing (0 = off)

    Returns:
        S_src: (N_freq,) complex source spectrum
        freqs: (N_freq,) frequency axis [Hz]
        s_src: (N,) source-position time-domain signal
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    S_rec = np.fft.rfft(signal).astype(np.complex128)

    # Optional: smooth amplitude spectrum (preserve phase)
    if smooth_window > 0 and smooth_window % 2 == 0:
        smooth_window += 1
    if smooth_window >= 3:
        from scipy.signal import savgol_filter
        mag = np.abs(S_rec)
        mag_smooth = savgol_filter(mag, window_length=smooth_window, polyorder=3)
        mag_smooth = np.maximum(mag_smooth, 1e-30)
        S_rec = (mag_smooth / mag) * S_rec

    # Look up A0 phase velocity at each frequency for this thickness
    cp = get_phase_velocity(freqs, d_mm)                 # (N_freq,) m/s

    L_m = L_mm * 1e-3                                     # mm → m

    # Step 1: remove geometric spreading (the signal carries 1/sqrt(L))
    S_rec = S_rec * np.sqrt(L_m)

    # Step 2: reverse the dispersion phase delay
    #   phi(f) = 2πf · L / c_p(f,d)   (positive: backwards in time)
    phase_correction = 2.0 * np.pi * freqs * L_m / cp
    S_src = S_rec * np.exp(1j * phase_correction)

    # Step 3: source-position time-domain signal (what the laser generates)
    s_src = np.fft.irfft(S_src, n=N).real.astype(np.float64)

    return S_src, freqs, s_src


def simulate(S_src, freqs, L_mm, d_mm, N_time=None):
    """Generate signal at target distance using calibrated source spectrum.

    Args:
        S_src: (N_freq,) complex source spectrum (pure, no geometric spreading baked in)
        freqs: (N_freq,) frequency axis [Hz]
        L_mm: target source-receiver distance [mm]
        d_mm: target plate thickness [mm]
        N_time: output time samples (default: 2 * len(freqs) - 1)

    Returns:
        signal: (N_time,) simulated time-domain signal
    """
    if N_time is None:
        N_time = 2 * len(freqs) - 1

    cp = get_phase_velocity(freqs, d_mm)
    L_m = L_mm * 1e-3

    # Forward dispersion
    phase = -2.0 * np.pi * freqs * L_m / cp
    S_target = S_src * np.exp(1j * phase)

    # Apply geometric spreading for target distance
    if L_m > 1e-9:
        S_target = S_target / np.sqrt(L_m)

    return np.fft.irfft(S_target, n=N_time).real.astype(np.float32)


def main():
    p = argparse.ArgumentParser(
        description='Extract equivalent Lamb-wave source spectrum from real signal')
    p.add_argument('input', help='Input signal file (.csv or .npy)')
    p.add_argument('--d', type=float, required=True,
                   help='Calibration plate thickness [mm]')
    p.add_argument('--L', type=float, required=True,
                   help='Source-receiver distance [mm]')
    p.add_argument('--fs', type=float, default=None,
                   help='Sampling rate [Hz] (inferred from CSV time column if omitted)')
    p.add_argument('--time-col', type=int, default=0,
                   help='CSV column index for time (default: 0)')
    p.add_argument('--disp-col', type=int, default=1,
                   help='CSV column index for displacement (default: 1)')
    p.add_argument('--skip-rows', type=int, default=0,
                   help='CSV header rows to skip')
    p.add_argument('--resample-fs', type=float, default=None,
                   help='Resample to this fs [Hz] before calibration')
    p.add_argument('--resample-N', type=int, default=256,
                   help='Resample to this many time samples (default: 256)')
    p.add_argument('--smooth', type=int, default=0,
                   help='Savitzky-Golay window for amplitude smoothing (0=off)')
    p.add_argument('--out', type=str, default=None,
                   help='Output .npz path (default: <input_stem>_source.npz)')
    p.add_argument('--simulate', type=float, nargs=2, default=None,
                   metavar=('L_TARGET', 'D_TARGET'),
                   help='Also simulate at target (L_mm, d_mm) for validation')
    p.add_argument('--plot', action='store_true',
                   help='Save diagnostic plots')
    args = p.parse_args()

    # ── Load ──────────────────────────────────────────────────────────
    signal, fs, t = load_signal(args.input, fs=args.fs,
                                time_col=args.time_col,
                                disp_col=args.disp_col,
                                skip_rows=args.skip_rows)
    print(f'[load] {len(signal)} samples @ {fs/1e6:.3f} MHz  '
          f'({len(signal)/fs*1e6:.1f} μs)')

    # ── Optional resample ─────────────────────────────────────────────
    if args.resample_fs is not None:
        signal, fs, t = resample(signal, fs, fs_out=args.resample_fs,
                                 N_out=args.resample_N)
        print(f'[resample] → {len(signal)} samples @ {fs/1e6:.3f} MHz')

    # ── Calibrate ─────────────────────────────────────────────────────
    S_src, freqs, s_src = calibrate(signal, args.L, args.d, fs,
                                    smooth_window=args.smooth)
    print(f'[calibrate] d={args.d:.1f}mm  L={args.L:.1f}mm  '
          f'N_freqs={len(freqs)}  ({freqs[0]/1e3:.0f}–{freqs[-1]/1e3:.0f} kHz)')

    # ── Save ──────────────────────────────────────────────────────────
    out_path = args.out or os.path.splitext(args.input)[0] + '_source.npz'
    np.savez(out_path,
             S_src=S_src, freqs=freqs, s_src=s_src,
             fs=fs, d_mm=args.d, L_mm=args.L,
             input_file=os.path.basename(args.input))
    print(f'[save] → {out_path}')

    # ── Optional: validate by round-trip simulation ───────────────────
    if args.simulate is not None:
        L_tgt, d_tgt = args.simulate
        sim_sig = simulate(S_src, freqs, L_tgt, d_tgt)
        sim_path = os.path.splitext(out_path)[0] + '_simulated.npy'
        np.save(sim_path, sim_sig)
        print(f'[simulate] L={L_tgt:.1f}mm d={d_tgt:.1f}mm → {sim_path}')

    # ── Plot ──────────────────────────────────────────────────────────
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Input signal
        ax = axes[0, 0]
        ax.plot(t * 1e6, signal, 'steelblue', lw=0.6)
        ax.set_title(f'Input signal (d={args.d}mm, L={args.L}mm)')
        ax.set_xlabel('Time [μs]'); ax.set_ylabel('Displacement')

        # Input spectrum
        ax = axes[0, 1]
        ax.semilogy(freqs / 1e3, np.abs(np.fft.rfft(signal)), 'steelblue', lw=0.8)
        ax.set_title('Input amplitude spectrum')
        ax.set_xlabel('Frequency [kHz]'); ax.set_ylabel('|S(f)|')

        # Phase velocity used
        ax = axes[0, 2]
        cp = get_phase_velocity(freqs, args.d)
        ax.plot(freqs / 1e3, cp, 'steelblue', lw=1)
        ax.set_title(f'A0 phase velocity (d={args.d}mm)')
        ax.set_xlabel('Frequency [kHz]'); ax.set_ylabel('c_p [m/s]')
        ax.grid(True, alpha=0.3)

        # Source spectrum
        ax = axes[1, 0]
        ax.semilogy(freqs / 1e3, np.abs(S_src), 'darkorange', lw=0.8)
        ax.set_title('Source spectrum |S_src(f)|')
        ax.set_xlabel('Frequency [kHz]'); ax.set_ylabel('|S(f)|')

        # Source time-domain signal
        ax = axes[1, 1]
        t_src = np.arange(len(s_src)) / fs
        ax.plot(t_src * 1e6, s_src, 'darkorange', lw=0.6)
        ax.set_title('Source-position signal')
        ax.set_xlabel('Time [μs]')

        # Overlay: input vs simulated at calibration distance
        if args.simulate is not None:
            ax = axes[1, 2]
            L_tgt, d_tgt = args.simulate
            t_sim = np.arange(len(sim_sig)) / fs
            ax.plot(t_sim * 1e6, sim_sig, 'darkorange', lw=0.6,
                    label=f'sim (L={L_tgt}mm, d={d_tgt}mm)')
            ax.set_title('Simulated signal at target')
            ax.set_xlabel('Time [μs]')
            ax.legend(fontsize=7)
        else:
            axes[1, 2].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.splitext(out_path)[0] + '_plots.png'
        fig.savefig(plot_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'[plot] → {plot_path}')


if __name__ == '__main__':
    main()
