# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Ultra2Thin — neural thickness prediction on metal plates from sparse ultrasonic scattered-signal measurements. A Born scattering physics simulation generates synthetic training data (32×32 thickness maps with defects → signals at measurement points). The model maps variable-grid point measurements + source position → thickness estimates, either per-point (v2) or as a full 2D map (v3).

## Commands

```bash
# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Train v3 model (hybrid: point encoder + U-Net decoder → full thickness map)
python train_v3.py

# Train v2 model (point-wise prediction with spatial interpolation)
python train_v2.py

# Generate UI images after training (requires checkpoints/model_v2.pt or model_v3.pt)
python gen_ui.py      # v2: per-point pred + cubic interpolation → 2D map
python gen_ui_v3.py   # v3: direct 2D map output via U-Net decoder

# View results — open the static HTML in a browser
open web_v3/index.html   # or web_v2/index.html
```

There are no tests, linters, or build steps. Training saves checkpoints to `checkpoints/`.

## Architecture (version evolution)

All versions share `simulation_v2.py` for data generation. The model architecture progressed across three versions:

| | v1 (earlier, not in repo) | v2 | v3 (current) |
|---|---|---|---|
| Input | fixed 4×4 grid, fixed source | variable N-point grid, variable source | variable N-point grid, variable source |
| Encoder | — | CNN(signal) + Fourier geo features | same as v2 |
| Aggregation | — | Transformer self-attention across points | same as v2 |
| Output head | direct 32×32 map | per-point thickness → `scipy.interpolate.griddata` cubic interpolation → 32×32 | GridScatter (Gaussian splat) → U-Net decoder → 32×32 map |

**Key files:**

- `simulation_v2.py` — Born scattering physics engine. Precomputes a frequency-domain scattering kernel once at import. Exports `generate_defect_map()` (random thickness maps with defects), `simulate_point_signals()` (maps + source + measurement positions → time-domain signals), and two dataset generators (`generate_point_dataset` for variable grids, `generate_dense_dataset` for fixed grids).
- `network_v2.py` — Point-wise predictor. 1D CNN encodes each signal, Fourier features encode per-point geometry (absolute pos, relative-to-source, distance, direction). Transformer lets points share context. Output: per-point thickness scalar.
- `network_v3.py` — Hybrid predictor. Reuses v2's encoder + Transformer, then `GridScatter` (Gaussian-weighted splatting of point features onto a 32×32 grid) feeds a lightweight U-Net decoder that outputs the full thickness map directly. This avoids v2's post-hoc interpolation.
- `train_v2.py` / `train_v3.py` — Two-phase training: Phase 1 (OneCycleLR, lr=2e-3, 40 epochs), Phase 2 (CosineAnnealing, lr=3e-4, 30 epochs). Both use AdamW with gradient clipping. Saves best checkpoint by validation MSE.
- `gen_ui.py` / `gen_ui_v3.py` — Run trained models on random samples, generate matplotlib comparison figures (ground truth / prediction / error / signals), write images + `samples.json` to `web_v*/`.
- `web_v*/index.html` — Static viewer with slider, auto-play, metric cards. Loads `samples.json` and corresponding images.

**Physics constants** (in `simulation_v2.py`): wave speed `C_REF = 5400 m/s`, reference thickness `D_REF = 2.0 mm`, center frequency `FC = 500 kHz`, sampling rate `FS = 25 MHz`, plate size 100 mm, grid resolution 32×32.

**Data flow:** `generate_defect_map()` → thickness map (32×32) → `simulate_point_signals()` → per-point time signals (N×128) + ground truth thicknesses → model → thickness predictions → metrics (MSE, MAE, correlation).

## Notes

- The Born kernel is precomputed at module import and consumes significant memory (~100 MB for the (N_freqs, 1024, 1024) tensor). First import of `simulation_v2` is slow.
- Both v2 and v3 use the same data format from `generate_point_dataset()` — the v3 model just ignores the per-point thickness labels and targets the full map instead.
- Training uses variable-length padding in the batch collate — `mask=True` marks padded positions.
- All random seeds are hardcoded (42). There's no config system — hyperparameters are inline in the training scripts.
