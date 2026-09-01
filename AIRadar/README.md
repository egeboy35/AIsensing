# AIRadar

AI radar and ISAC (integrated sensing and communication) research core of the
[AIsensing](https://github.com/lkk688/AIsensing) project: dataset generation,
model training pipelines, and the reusable `AIRadarLib` Python package.

## Installation

From this directory (`AIRadar/`):

```bash
pip install -e .
```

or, using flit as described in the root README:

```bash
pip install flit
flit install --symlink
```

Both commands install the `AIRadarLib` package together with its core
dependencies (NumPy, SciPy, Matplotlib, PyTorch, einops, scikit-learn).
GPU-enabled PyTorch is recommended for model training; see
[pytorch.org](https://pytorch.org) for platform-specific builds.

Optional extras:

```bash
pip install -e ".[sionna]"   # TensorFlow + Sionna based waveform helpers
```

## AIRadarLib modules

| Module | Purpose |
|---|---|
| `signal_processing` | Radar DSP blocks: FMCW dechirping, range/Doppler processing |
| `radar_det` | Target detection (CA-CFAR and related detectors) |
| `waveform_utils` | Chirp/waveform generation helpers |
| `channel_simulation` | FMCW channel and target echo simulation |
| `datautil` | Signal conversion and data utilities |
| `target_utils` | Synthetic radar target generation and label bookkeeping |
| `AIradar_autopara` | Automatic radar parameter design (`RadarParameterDesigner`) |
| `modeling_RadarNet`, `modeling_TimeNet`, `modeling_transformer` | Neural detector architectures (CNN, temporal, axial-attention transformer) |
| `ofdm_decoder` | OFDM demodulation/decoding helpers for ISAC experiments |
| `pretrain_dataset` | Dataset classes for model pre-training |
| `visualization` | Plotting utilities for RD maps, detections, and results |

## Experiment scripts

The top-level `AIRadar/*.py` scripts are research entry points for dataset
generation (`AIradar_dataset*.py`), model training (`AIradar_train*.py`),
joint communication-radar models (`AIradar_comm_*.py`), and the configurable
ISAC experiment framework (`isac_experiment/`). See the root
[README](https://github.com/lkk688/AIsensing#readme) for recommended workflows.
