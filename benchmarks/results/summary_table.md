Radar config `config_phaser` (Phaser_10GHz_DevKit): 500 MHz BW, 64 chirps x 1000 samples, RD map 64 x 667 bins, range bin 0.150 m, velocity bin 0.468 m/s.

16 frames per SNR point, 1 target(s) per frame, base seed 20260819, SNR referenced to **target** power. Association gate: +/-2 range bins (0.300 m), +/-1 Doppler bins (0.468 m/s). Quantization RMSE floor: 0.043 m / 0.135 m/s.

### Stage 1: threshold calibration (threshold -> measured Pfa)

16 target-free frames per scene kind. Threshold axis: dB above the local mean noise power; converted to each detector's own knob by the closed forms in benchmarks/detectors.py. Grid: 6, 7, 8, 9, 10, 11, 12, 13 dB.

* target Pfa 1e-03: 42240 eligible cells x 16 frames = 675,840 cell trials -> 675.8 expected false alarms, **measurable** (10 events needs 1 frames).
* target Pfa 1e-04: 42240 eligible cells x 16 frames = 675,840 cell trials -> 67.6 expected false alarms, **measurable** (10 events needs 3 frames).

Scene kind `noise_only` (source scenario `clutter_off`):

| detector | knob | threshold (dB) | knob value | false alarms | measured Pfa /cell |
|---|---|---|---|---|---|
| `cfar_numpy_go` | `magnitude_warp_exponent` | 6.00 | 1.41063 | 8132 | 1.20e-02 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 7.00 | 1.26225 | 3335 | 4.93e-03 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 8.00 | 1.14212 | 1010 | 1.49e-03 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 9.00 | 1.04286 | 185 | 2.74e-04 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 9.49 (as shipped) | 1 | 66 | 9.77e-05 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 10.00 | 0.959477 | 24 | 3.55e-05 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 11.00 | 0.88844 | 1 | 1.48e-06 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 12.00 | 0.827197 | 0 | 0 (none observed) |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 13.00 | 0.773853 | 0 | 0 (none observed) |
| `cfar_advanced_go` | `pfa` | 6.00 | 0.368196 | 8267 | 1.22e-02 |
| `cfar_advanced_go` | `pfa` | 7.00 | 0.284347 | 3380 | 5.00e-03 |
| `cfar_advanced_go` | `pfa` | 8.00 | 0.205413 | 1006 | 1.49e-03 |
| `cfar_advanced_go` | `pfa` | 9.00 | 0.136444 | 188 | 2.78e-04 |
| `cfar_advanced_go` | `pfa` | 10.00 | 0.0815562 | 20 | 2.96e-05 |
| `cfar_advanced_go` | `pfa` | 11.00 | 0.0426944 | 1 | 1.48e-06 |
| `cfar_advanced_go` | `pfa` | 12.00 | 0.0189212 | 0 | 0 (none observed) |
| `cfar_advanced_go` | `pfa` | 13.00 | 0.00680334 | 0 | 0 (none observed) |
| `cfar_advanced_go` | `pfa` | 16.66 (as shipped) | 1e-05 | 0 | 0 (none observed) |
| `cfar_custom_datasetv8` | `threshold_offset` | 6.00 | 8.50682 | 8164 | 1.20e-02 |
| `cfar_custom_datasetv8` | `threshold_offset` | 7.00 | 9.50682 | 3351 | 4.94e-03 |
| `cfar_custom_datasetv8` | `threshold_offset` | 8.00 | 10.5068 | 1015 | 1.50e-03 |
| `cfar_custom_datasetv8` | `threshold_offset` | 9.00 | 11.5068 | 186 | 2.74e-04 |
| `cfar_custom_datasetv8` | `threshold_offset` | 10.00 | 12.5068 | 25 | 3.68e-05 |
| `cfar_custom_datasetv8` | `threshold_offset` | 11.00 | 13.5068 | 1 | 1.47e-06 |
| `cfar_custom_datasetv8` | `threshold_offset` | 12.00 | 14.5068 | 0 | 0 (none observed) |
| `cfar_custom_datasetv8` | `threshold_offset` | 12.49 (as shipped) | 15 | 0 | 0 (none observed) |
| `cfar_custom_datasetv8` | `threshold_offset` | 13.00 | 15.5068 | 0 | 0 (none observed) |

Scene kind `clutter_only` (source scenario `clutter_strong`):

| detector | knob | threshold (dB) | knob value | false alarms | measured Pfa /cell |
|---|---|---|---|---|---|
| `cfar_numpy_go` | `magnitude_warp_exponent` | 6.00 | 1.41063 | 8105 | 1.20e-02 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 7.00 | 1.26225 | 3364 | 4.98e-03 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 8.00 | 1.14212 | 1075 | 1.59e-03 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 9.00 | 1.04286 | 255 | 3.77e-04 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 9.49 (as shipped) | 1 | 131 | 1.94e-04 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 10.00 | 0.959477 | 84 | 1.24e-04 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 11.00 | 0.88844 | 55 | 8.14e-05 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 12.00 | 0.827197 | 43 | 6.36e-05 |
| `cfar_numpy_go` | `magnitude_warp_exponent` | 13.00 | 0.773853 | 37 | 5.47e-05 |
| `cfar_advanced_go` | `pfa` | 6.00 | 0.368196 | 8019 | 1.19e-02 |
| `cfar_advanced_go` | `pfa` | 7.00 | 0.284347 | 3310 | 4.90e-03 |
| `cfar_advanced_go` | `pfa` | 8.00 | 0.205413 | 1045 | 1.55e-03 |
| `cfar_advanced_go` | `pfa` | 9.00 | 0.136444 | 243 | 3.60e-04 |
| `cfar_advanced_go` | `pfa` | 10.00 | 0.0815562 | 76 | 1.12e-04 |
| `cfar_advanced_go` | `pfa` | 11.00 | 0.0426944 | 50 | 7.40e-05 |
| `cfar_advanced_go` | `pfa` | 12.00 | 0.0189212 | 41 | 6.07e-05 |
| `cfar_advanced_go` | `pfa` | 13.00 | 0.00680334 | 32 | 4.73e-05 |
| `cfar_advanced_go` | `pfa` | 16.66 (as shipped) | 1e-05 | 17 | 2.52e-05 |
| `cfar_custom_datasetv8` | `threshold_offset` | 6.00 | 8.50682 | 8134 | 1.20e-02 |
| `cfar_custom_datasetv8` | `threshold_offset` | 7.00 | 9.50682 | 3377 | 4.97e-03 |
| `cfar_custom_datasetv8` | `threshold_offset` | 8.00 | 10.5068 | 1077 | 1.59e-03 |
| `cfar_custom_datasetv8` | `threshold_offset` | 9.00 | 11.5068 | 256 | 3.77e-04 |
| `cfar_custom_datasetv8` | `threshold_offset` | 10.00 | 12.5068 | 85 | 1.25e-04 |
| `cfar_custom_datasetv8` | `threshold_offset` | 11.00 | 13.5068 | 55 | 8.10e-05 |
| `cfar_custom_datasetv8` | `threshold_offset` | 12.00 | 14.5068 | 43 | 6.33e-05 |
| `cfar_custom_datasetv8` | `threshold_offset` | 12.49 (as shipped) | 15 | 40 | 5.89e-05 |
| `cfar_custom_datasetv8` | `threshold_offset` | 13.00 | 15.5068 | 37 | 5.45e-05 |

Solved operating points:

| detector | scene kind | target Pfa | solved threshold (dB) | knob value | status |
|---|---|---|---|---|---|
| `cfar_numpy_go` | noise_only | 1e-03 | 8.237 | 1.116954 | interpolated |
| `cfar_numpy_go` | noise_only | 1e-04 | 9.493 | 1.000010 | interpolated |
| `cfar_advanced_go` | noise_only | 1e-03 | 8.237 | 0.187977 | interpolated |
| `cfar_advanced_go` | noise_only | 1e-04 | 9.457 | 0.109457 | interpolated |
| `cfar_custom_datasetv8` | noise_only | 1e-03 | 8.237 | 10.743809 | interpolated |
| `cfar_custom_datasetv8` | noise_only | 1e-04 | 9.502 | 12.009011 | interpolated |
| `cfar_numpy_go` | clutter_only | 1e-03 | 8.323 | 1.108096 | interpolated |
| `cfar_numpy_go` | clutter_only | 1e-04 | 10.513 | 0.921639 | interpolated |
| `cfar_advanced_go` | clutter_only | 1e-03 | 8.299 | 0.183547 | interpolated |
| `cfar_advanced_go` | clutter_only | 1e-04 | 10.280 | 0.069032 | interpolated |
| `cfar_custom_datasetv8` | clutter_only | 1e-03 | 8.321 | 10.827986 | interpolated |
| `cfar_custom_datasetv8` | clutter_only | 1e-04 | 10.516 | 13.023093 | interpolated |

the sweep uses the noise-only solve. The clutter-only curve is reported beside it: under clutter_strong the clutter scatterers are real peaks, so there is a floor on the achievable peak density that no threshold removes.

### Stage 2a (headline): all detectors at the calibrated common operating point, Pfa = 1e-04 /cell

Thresholds in use: `cfar_advanced_go` pfa=0.109457 (9.46 dB); `cfar_custom_datasetv8` threshold_offset=12.009011 (9.50 dB); `cfar_numpy_go` magnitude_warp_exponent=1.000010 (9.49 dB).

#### `clutter_off` -- `apply_realistic_effects=False`, `clutter_intensity=1` -- clutter disabled; noise referenced to target-only power, which required +0.00 dB mean correction to the requested SNR.

| detector | thr (dB) | SNR in (dB) | target-bin SNR (dB) | Pd | FA/frame | FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cfar_advanced_go` | 9.46 | -42 | 2.3 | 0.062 | 5.56 | 1.32e-04 | 0.0603 | 0.3276 | 1 | 89 | 15 |
| `cfar_advanced_go` | 9.46 | -39 | 4.2 | 0.062 | 6.00 | 1.42e-04 | 0.0367 | 0.1470 | 1 | 96 | 15 |
| `cfar_advanced_go` | 9.46 | -36 | 7.3 | 0.250 | 5.12 | 1.21e-04 | 0.0720 | 0.1578 | 4 | 82 | 12 |
| `cfar_advanced_go` | 9.46 | -33 | 10.3 | 0.938 | 5.62 | 1.33e-04 | 0.0904 | 0.1704 | 15 | 90 | 1 |
| `cfar_advanced_go` | 9.46 | -30 | 13.7 | 1.000 | 5.81 | 1.38e-04 | 0.0897 | 0.1678 | 16 | 93 | 0 |
| `cfar_advanced_go` | 9.46 | -27 | 16.9 | 1.000 | 5.62 | 1.33e-04 | 0.0539 | 0.1763 | 16 | 90 | 0 |
| `cfar_advanced_go` | 9.46 | -24 | 19.7 | 1.000 | 5.31 | 1.26e-04 | 0.0651 | 0.1396 | 16 | 85 | 0 |
| `cfar_custom_datasetv8` | 9.50 | -42 | 2.3 | 0.000 | 5.31 | 1.25e-04 | n/a | n/a | 0 | 85 | 16 |
| `cfar_custom_datasetv8` | 9.50 | -39 | 4.2 | 0.062 | 5.69 | 1.34e-04 | 0.0367 | 0.1470 | 1 | 91 | 15 |
| `cfar_custom_datasetv8` | 9.50 | -36 | 7.3 | 0.250 | 4.75 | 1.12e-04 | 0.0720 | 0.1578 | 4 | 76 | 12 |
| `cfar_custom_datasetv8` | 9.50 | -33 | 10.3 | 0.938 | 5.25 | 1.24e-04 | 0.0904 | 0.1704 | 15 | 84 | 1 |
| `cfar_custom_datasetv8` | 9.50 | -30 | 13.7 | 1.000 | 5.44 | 1.28e-04 | 0.0897 | 0.1678 | 16 | 87 | 0 |
| `cfar_custom_datasetv8` | 9.50 | -27 | 16.9 | 1.000 | 5.75 | 1.36e-04 | 0.0539 | 0.1763 | 16 | 92 | 0 |
| `cfar_custom_datasetv8` | 9.50 | -24 | 19.7 | 1.000 | 4.75 | 1.12e-04 | 0.0651 | 0.1396 | 16 | 76 | 0 |
| `cfar_numpy_go` | 9.49 | -42 | 2.3 | 0.000 | 5.31 | 1.26e-04 | n/a | n/a | 0 | 85 | 16 |
| `cfar_numpy_go` | 9.49 | -39 | 4.2 | 0.062 | 5.62 | 1.33e-04 | 0.0367 | 0.1470 | 1 | 90 | 15 |
| `cfar_numpy_go` | 9.49 | -36 | 7.3 | 0.250 | 4.75 | 1.12e-04 | 0.0720 | 0.1578 | 4 | 76 | 12 |
| `cfar_numpy_go` | 9.49 | -33 | 10.3 | 0.938 | 5.50 | 1.30e-04 | 0.0904 | 0.1704 | 15 | 88 | 1 |
| `cfar_numpy_go` | 9.49 | -30 | 13.7 | 1.000 | 5.56 | 1.32e-04 | 0.0897 | 0.1678 | 16 | 89 | 0 |
| `cfar_numpy_go` | 9.49 | -27 | 16.9 | 1.000 | 5.81 | 1.38e-04 | 0.0539 | 0.1763 | 16 | 93 | 0 |
| `cfar_numpy_go` | 9.49 | -24 | 19.7 | 1.000 | 4.88 | 1.15e-04 | 0.0651 | 0.1396 | 16 | 78 | 0 |

#### `clutter_strong` -- `apply_realistic_effects=True`, `clutter_intensity=10000` -- clutter/target RCS power +15.5 dB; noise referenced to target-only power, which required +11.24 dB mean correction to the requested SNR.

| detector | thr (dB) | SNR in (dB) | target-bin SNR (dB) | Pd | FA/frame | FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cfar_advanced_go` | 9.46 | -42 | 2.3 | 0.062 | 5.81 | 1.38e-04 | 0.0603 | 0.3276 | 1 | 93 | 15 |
| `cfar_advanced_go` | 9.46 | -39 | 4.2 | 0.062 | 6.81 | 1.61e-04 | 0.0367 | 0.1470 | 1 | 109 | 15 |
| `cfar_advanced_go` | 9.46 | -36 | 7.3 | 0.250 | 6.50 | 1.54e-04 | 0.0720 | 0.1578 | 4 | 104 | 12 |
| `cfar_advanced_go` | 9.46 | -33 | 10.2 | 0.875 | 8.06 | 1.91e-04 | 0.0904 | 0.1723 | 14 | 129 | 2 |
| `cfar_advanced_go` | 9.46 | -30 | 13.5 | 1.000 | 9.44 | 2.23e-04 | 0.0897 | 0.1678 | 16 | 151 | 0 |
| `cfar_advanced_go` | 9.46 | -27 | 16.5 | 1.000 | 11.44 | 2.71e-04 | 0.0539 | 0.1763 | 16 | 183 | 0 |
| `cfar_advanced_go` | 9.46 | -24 | 19.0 | 1.000 | 14.31 | 3.39e-04 | 0.0651 | 0.1396 | 16 | 229 | 0 |
| `cfar_custom_datasetv8` | 9.50 | -42 | 2.3 | 0.000 | 5.75 | 1.36e-04 | n/a | n/a | 0 | 92 | 16 |
| `cfar_custom_datasetv8` | 9.50 | -39 | 4.2 | 0.000 | 6.44 | 1.52e-04 | n/a | n/a | 0 | 103 | 16 |
| `cfar_custom_datasetv8` | 9.50 | -36 | 7.3 | 0.250 | 6.31 | 1.49e-04 | 0.0720 | 0.1578 | 4 | 101 | 12 |
| `cfar_custom_datasetv8` | 9.50 | -33 | 10.2 | 0.938 | 8.00 | 1.89e-04 | 0.0904 | 0.1704 | 15 | 128 | 1 |
| `cfar_custom_datasetv8` | 9.50 | -30 | 13.5 | 1.000 | 9.62 | 2.27e-04 | 0.0897 | 0.1678 | 16 | 154 | 0 |
| `cfar_custom_datasetv8` | 9.50 | -27 | 16.5 | 1.000 | 12.50 | 2.95e-04 | 0.0539 | 0.1763 | 16 | 200 | 0 |
| `cfar_custom_datasetv8` | 9.50 | -24 | 19.0 | 1.000 | 14.81 | 3.49e-04 | 0.0651 | 0.1396 | 16 | 237 | 0 |
| `cfar_numpy_go` | 9.49 | -42 | 2.3 | 0.000 | 5.75 | 1.36e-04 | n/a | n/a | 0 | 92 | 16 |
| `cfar_numpy_go` | 9.49 | -39 | 4.2 | 0.000 | 6.38 | 1.51e-04 | n/a | n/a | 0 | 102 | 16 |
| `cfar_numpy_go` | 9.49 | -36 | 7.3 | 0.250 | 6.31 | 1.49e-04 | 0.0720 | 0.1578 | 4 | 101 | 12 |
| `cfar_numpy_go` | 9.49 | -33 | 10.2 | 0.938 | 8.19 | 1.94e-04 | 0.0904 | 0.1704 | 15 | 131 | 1 |
| `cfar_numpy_go` | 9.49 | -30 | 13.5 | 1.000 | 9.69 | 2.29e-04 | 0.0897 | 0.1678 | 16 | 155 | 0 |
| `cfar_numpy_go` | 9.49 | -27 | 16.5 | 1.000 | 12.62 | 2.99e-04 | 0.0539 | 0.1763 | 16 | 202 | 0 |
| `cfar_numpy_go` | 9.49 | -24 | 19.0 | 1.000 | 15.00 | 3.55e-04 | 0.0651 | 0.1396 | 16 | 240 | 0 |

### Stage 2b: the same sweep with the thresholds as configured in the repo

Kept because it is what a user of this repository actually gets. It is **not** a detector comparison: the three effective thresholds are `cfar_advanced_go` 16.66 dB, `cfar_custom_datasetv8` 12.49 dB, `cfar_numpy_go` 9.49 dB, so the Pd order mostly follows the threshold order.

#### `clutter_off` -- `apply_realistic_effects=False`, `clutter_intensity=1` -- clutter disabled; noise referenced to target-only power, which required +0.00 dB mean correction to the requested SNR.

| detector | thr (dB) | SNR in (dB) | target-bin SNR (dB) | Pd | FA/frame | FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cfar_advanced_go` | 16.66 | -42 | 2.3 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | 16.66 | -39 | 4.2 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | 16.66 | -36 | 7.3 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | 16.66 | -33 | 10.3 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | 16.66 | -30 | 13.7 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | 16.66 | -27 | 16.9 | 0.500 | 0.00 | 0 (none observed) | 0.0637 | 0.1631 | 8 | 0 | 8 |
| `cfar_advanced_go` | 16.66 | -24 | 19.7 | 1.000 | 0.00 | 0 (none observed) | 0.0651 | 0.1396 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | 12.49 | -42 | 2.3 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | 12.49 | -39 | 4.2 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | 12.49 | -36 | 7.3 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | 12.49 | -33 | 10.3 | 0.125 | 0.00 | 0 (none observed) | 0.0880 | 0.0960 | 2 | 0 | 14 |
| `cfar_custom_datasetv8` | 12.49 | -30 | 13.7 | 0.938 | 0.00 | 0 (none observed) | 0.0926 | 0.1667 | 15 | 0 | 1 |
| `cfar_custom_datasetv8` | 12.49 | -27 | 16.9 | 1.000 | 0.00 | 0 (none observed) | 0.0539 | 0.1763 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | 12.49 | -24 | 19.7 | 1.000 | 0.00 | 0 (none observed) | 0.0651 | 0.1396 | 16 | 0 | 0 |
| `cfar_numpy_go` | 9.49 | -42 | 2.3 | 0.000 | 5.31 | 1.26e-04 | n/a | n/a | 0 | 85 | 16 |
| `cfar_numpy_go` | 9.49 | -39 | 4.2 | 0.062 | 5.62 | 1.33e-04 | 0.0367 | 0.1470 | 1 | 90 | 15 |
| `cfar_numpy_go` | 9.49 | -36 | 7.3 | 0.250 | 4.75 | 1.12e-04 | 0.0720 | 0.1578 | 4 | 76 | 12 |
| `cfar_numpy_go` | 9.49 | -33 | 10.3 | 0.938 | 5.50 | 1.30e-04 | 0.0904 | 0.1704 | 15 | 88 | 1 |
| `cfar_numpy_go` | 9.49 | -30 | 13.7 | 1.000 | 5.56 | 1.32e-04 | 0.0897 | 0.1678 | 16 | 89 | 0 |
| `cfar_numpy_go` | 9.49 | -27 | 16.9 | 1.000 | 5.81 | 1.38e-04 | 0.0539 | 0.1763 | 16 | 93 | 0 |
| `cfar_numpy_go` | 9.49 | -24 | 19.7 | 1.000 | 4.88 | 1.15e-04 | 0.0651 | 0.1396 | 16 | 78 | 0 |

#### `clutter_strong` -- `apply_realistic_effects=True`, `clutter_intensity=10000` -- clutter/target RCS power +15.5 dB; noise referenced to target-only power, which required +11.24 dB mean correction to the requested SNR.

| detector | thr (dB) | SNR in (dB) | target-bin SNR (dB) | Pd | FA/frame | FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cfar_advanced_go` | 16.66 | -42 | 2.3 | 0.000 | 0.00 | 0 (none observed) | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | 16.66 | -39 | 4.2 | 0.000 | 0.06 | 1.48e-06 | n/a | n/a | 0 | 1 | 16 |
| `cfar_advanced_go` | 16.66 | -36 | 7.3 | 0.000 | 0.31 | 7.40e-06 | n/a | n/a | 0 | 5 | 16 |
| `cfar_advanced_go` | 16.66 | -33 | 10.2 | 0.000 | 0.44 | 1.04e-05 | n/a | n/a | 0 | 7 | 16 |
| `cfar_advanced_go` | 16.66 | -30 | 13.5 | 0.000 | 0.94 | 2.22e-05 | n/a | n/a | 0 | 15 | 16 |
| `cfar_advanced_go` | 16.66 | -27 | 16.5 | 0.500 | 1.81 | 4.29e-05 | 0.0637 | 0.1631 | 8 | 29 | 8 |
| `cfar_advanced_go` | 16.66 | -24 | 19.0 | 1.000 | 2.44 | 5.77e-05 | 0.0651 | 0.1396 | 16 | 39 | 0 |
| `cfar_custom_datasetv8` | 12.49 | -42 | 2.3 | 0.000 | 0.12 | 2.95e-06 | n/a | n/a | 0 | 2 | 16 |
| `cfar_custom_datasetv8` | 12.49 | -39 | 4.2 | 0.000 | 0.38 | 8.84e-06 | n/a | n/a | 0 | 6 | 16 |
| `cfar_custom_datasetv8` | 12.49 | -36 | 7.3 | 0.000 | 0.69 | 1.62e-05 | n/a | n/a | 0 | 11 | 16 |
| `cfar_custom_datasetv8` | 12.49 | -33 | 10.2 | 0.125 | 1.62 | 3.83e-05 | 0.0880 | 0.0960 | 2 | 26 | 14 |
| `cfar_custom_datasetv8` | 12.49 | -30 | 13.5 | 0.875 | 2.56 | 6.04e-05 | 0.0909 | 0.1680 | 14 | 41 | 2 |
| `cfar_custom_datasetv8` | 12.49 | -27 | 16.5 | 1.000 | 3.44 | 8.10e-05 | 0.0539 | 0.1763 | 16 | 55 | 0 |
| `cfar_custom_datasetv8` | 12.49 | -24 | 19.0 | 1.000 | 5.56 | 1.31e-04 | 0.0651 | 0.1396 | 16 | 89 | 0 |
| `cfar_numpy_go` | 9.49 | -42 | 2.3 | 0.000 | 5.75 | 1.36e-04 | n/a | n/a | 0 | 92 | 16 |
| `cfar_numpy_go` | 9.49 | -39 | 4.2 | 0.000 | 6.38 | 1.51e-04 | n/a | n/a | 0 | 102 | 16 |
| `cfar_numpy_go` | 9.49 | -36 | 7.3 | 0.250 | 6.31 | 1.49e-04 | 0.0720 | 0.1578 | 4 | 101 | 12 |
| `cfar_numpy_go` | 9.49 | -33 | 10.2 | 0.938 | 8.19 | 1.94e-04 | 0.0904 | 0.1704 | 15 | 131 | 1 |
| `cfar_numpy_go` | 9.49 | -30 | 13.5 | 1.000 | 9.69 | 2.29e-04 | 0.0897 | 0.1678 | 16 | 155 | 0 |
| `cfar_numpy_go` | 9.49 | -27 | 16.5 | 1.000 | 12.62 | 2.99e-04 | 0.0539 | 0.1763 | 16 | 202 | 0 |
| `cfar_numpy_go` | 9.49 | -24 | 19.0 | 1.000 | 15.00 | 3.55e-04 | 0.0651 | 0.1396 | 16 | 240 | 0 |

### Stage 3: ROC -- Pd vs *measured* noise-only Pfa, scenario `clutter_off`

Each row is one point of the threshold grid. The Pfa column is the density measured on the target-free frames of stage 1, not a design parameter.

At input SNR -33 dB (target-bin SNR 10.3 dB):

| detector | threshold (dB) | knob value | measured Pfa /cell | Pd | FA/frame on target frames | of which near the target |
|---|---|---|---|---|---|---|
| `cfar_advanced_go` | 6.00 | 0.368196 | 1.22e-02 | 1.000 | 516.38 | 0.62 |
| `cfar_advanced_go` | 7.00 | 0.284347 | 5.00e-03 | 1.000 | 212.62 | 0.31 |
| `cfar_advanced_go` | 8.00 | 0.205413 | 1.49e-03 | 1.000 | 61.88 | 0.00 |
| `cfar_advanced_go` | 9.00 | 0.136444 | 2.78e-04 | 1.000 | 12.75 | 0.00 |
| `cfar_advanced_go` | 10.00 | 0.0815562 | 2.96e-05 | 0.938 | 1.88 | 0.00 |
| `cfar_advanced_go` | 11.00 | 0.0426944 | 1.48e-06 | 0.438 | 0.00 | 0.00 |
| `cfar_advanced_go` | 12.00 | 0.0189212 | 0 (none observed) | 0.188 | 0.00 | 0.00 |
| `cfar_advanced_go` | 13.00 | 0.00680334 | 0 (none observed) | 0.062 | 0.00 | 0.00 |
| `cfar_custom_datasetv8` | 6.00 | 8.50682 | 1.20e-02 | 1.000 | 512.06 | 0.69 |
| `cfar_custom_datasetv8` | 7.00 | 9.50682 | 4.94e-03 | 1.000 | 213.38 | 0.19 |
| `cfar_custom_datasetv8` | 8.00 | 10.5068 | 1.50e-03 | 1.000 | 62.56 | 0.12 |
| `cfar_custom_datasetv8` | 9.00 | 11.5068 | 2.74e-04 | 1.000 | 12.62 | 0.00 |
| `cfar_custom_datasetv8` | 10.00 | 12.5068 | 3.68e-05 | 0.938 | 1.88 | 0.00 |
| `cfar_custom_datasetv8` | 11.00 | 13.5068 | 1.47e-06 | 0.438 | 0.06 | 0.00 |
| `cfar_custom_datasetv8` | 12.00 | 14.5068 | 0 (none observed) | 0.188 | 0.00 | 0.00 |
| `cfar_custom_datasetv8` | 13.00 | 15.5068 | 0 (none observed) | 0.062 | 0.00 | 0.00 |
| `cfar_numpy_go` | 6.00 | 1.41063 | 1.20e-02 | 1.000 | 510.00 | 0.69 |
| `cfar_numpy_go` | 7.00 | 1.26225 | 4.93e-03 | 1.000 | 212.62 | 0.19 |
| `cfar_numpy_go` | 8.00 | 1.14212 | 1.49e-03 | 1.000 | 62.38 | 0.12 |
| `cfar_numpy_go` | 9.00 | 1.04286 | 2.74e-04 | 1.000 | 12.56 | 0.00 |
| `cfar_numpy_go` | 10.00 | 0.959477 | 3.55e-05 | 0.938 | 1.88 | 0.00 |
| `cfar_numpy_go` | 11.00 | 0.88844 | 1.48e-06 | 0.438 | 0.06 | 0.00 |
| `cfar_numpy_go` | 12.00 | 0.827197 | 0 (none observed) | 0.188 | 0.00 | 0.00 |
| `cfar_numpy_go` | 13.00 | 0.773853 | 0 (none observed) | 0.062 | 0.00 | 0.00 |

At input SNR -27 dB (target-bin SNR 16.9 dB):

| detector | threshold (dB) | knob value | measured Pfa /cell | Pd | FA/frame on target frames | of which near the target |
|---|---|---|---|---|---|---|
| `cfar_advanced_go` | 6.00 | 0.368196 | 1.22e-02 | 1.000 | 521.31 | 0.25 |
| `cfar_advanced_go` | 7.00 | 0.284347 | 5.00e-03 | 1.000 | 213.19 | 0.00 |
| `cfar_advanced_go` | 8.00 | 0.205413 | 1.49e-03 | 1.000 | 63.69 | 0.00 |
| `cfar_advanced_go` | 9.00 | 0.136444 | 2.78e-04 | 1.000 | 13.25 | 0.00 |
| `cfar_advanced_go` | 10.00 | 0.0815562 | 2.96e-05 | 1.000 | 1.69 | 0.00 |
| `cfar_advanced_go` | 11.00 | 0.0426944 | 1.48e-06 | 1.000 | 0.12 | 0.00 |
| `cfar_advanced_go` | 12.00 | 0.0189212 | 0 (none observed) | 1.000 | 0.00 | 0.00 |
| `cfar_advanced_go` | 13.00 | 0.00680334 | 0 (none observed) | 1.000 | 0.00 | 0.00 |
| `cfar_custom_datasetv8` | 6.00 | 8.50682 | 1.20e-02 | 1.000 | 523.31 | 0.38 |
| `cfar_custom_datasetv8` | 7.00 | 9.50682 | 4.94e-03 | 1.000 | 215.19 | 0.19 |
| `cfar_custom_datasetv8` | 8.00 | 10.5068 | 1.50e-03 | 1.000 | 64.69 | 0.00 |
| `cfar_custom_datasetv8` | 9.00 | 11.5068 | 2.74e-04 | 1.000 | 13.44 | 0.00 |
| `cfar_custom_datasetv8` | 10.00 | 12.5068 | 3.68e-05 | 1.000 | 2.00 | 0.00 |
| `cfar_custom_datasetv8` | 11.00 | 13.5068 | 1.47e-06 | 1.000 | 0.19 | 0.00 |
| `cfar_custom_datasetv8` | 12.00 | 14.5068 | 0 (none observed) | 1.000 | 0.00 | 0.00 |
| `cfar_custom_datasetv8` | 13.00 | 15.5068 | 0 (none observed) | 1.000 | 0.00 | 0.00 |
| `cfar_numpy_go` | 6.00 | 1.41063 | 1.20e-02 | 1.000 | 521.31 | 0.38 |
| `cfar_numpy_go` | 7.00 | 1.26225 | 4.93e-03 | 1.000 | 214.06 | 0.19 |
| `cfar_numpy_go` | 8.00 | 1.14212 | 1.49e-03 | 1.000 | 64.25 | 0.00 |
| `cfar_numpy_go` | 9.00 | 1.04286 | 2.74e-04 | 1.000 | 13.25 | 0.00 |
| `cfar_numpy_go` | 10.00 | 0.959477 | 3.55e-05 | 1.000 | 2.00 | 0.00 |
| `cfar_numpy_go` | 11.00 | 0.88844 | 1.48e-06 | 1.000 | 0.19 | 0.00 |
| `cfar_numpy_go` | 12.00 | 0.827197 | 0 (none observed) | 1.000 | 0.00 | 0.00 |
| `cfar_numpy_go` | 13.00 | 0.773853 | 0 (none observed) | 1.000 | 0.00 | 0.00 |

### How much of the measured false-alarm rate is the target's own sidelobes

Every in-gate detection beyond the one matched pair counts as a false positive, so a target sidelobe that survives non-maximum suppression is reported as a false alarm. Split at the highest SNR point (-24 dB) using a wider +/-8 range bin, +/-3 Doppler bin neighbourhood (which changes no count -- it only labels them):

| detector | variant | scenario | FP total | FP near a target | share | FP/frame far from any target |
|---|---|---|---|---|---|---|
| `cfar_advanced_go` | as_shipped | clutter_off | 0 | 0 | n/a | 0.00 |
| `cfar_advanced_go` | as_shipped | clutter_strong | 39 | 0 | 0.0% | 2.44 |
| `cfar_custom_datasetv8` | as_shipped | clutter_off | 0 | 0 | n/a | 0.00 |
| `cfar_custom_datasetv8` | as_shipped | clutter_strong | 89 | 0 | 0.0% | 5.56 |
| `cfar_numpy_go` | as_shipped | clutter_off | 78 | 0 | 0.0% | 4.88 |
| `cfar_numpy_go` | as_shipped | clutter_strong | 240 | 0 | 0.0% | 15.00 |
| `cfar_advanced_go` | calibrated_pfa1e-4 | clutter_off | 85 | 0 | 0.0% | 5.31 |
| `cfar_advanced_go` | calibrated_pfa1e-4 | clutter_strong | 229 | 0 | 0.0% | 14.31 |
| `cfar_custom_datasetv8` | calibrated_pfa1e-4 | clutter_off | 76 | 0 | 0.0% | 4.75 |
| `cfar_custom_datasetv8` | calibrated_pfa1e-4 | clutter_strong | 237 | 0 | 0.0% | 14.81 |
| `cfar_numpy_go` | calibrated_pfa1e-4 | clutter_off | 78 | 0 | 0.0% | 4.88 |
| `cfar_numpy_go` | calibrated_pfa1e-4 | clutter_strong | 240 | 0 | 0.0% | 15.00 |

Runtime: 1014.5 s wall clock for 224 swept frames plus 16 calibration frames per scene kind, 2976 detector calls (49.6 s simulation, cfar_advanced_go 322.4 s, cfar_custom_datasetv8 177.3 s, cfar_numpy_go 181.5 s).

Environment: Python 3.14.2, numpy 2.5.2, scipy 1.18.0, matplotlib 3.11.1, torch not installed, tqdm not installed, Windows-11-10.0.26200-SP0, device cpu.

Provenance: generated from commit `2a1b4fae3fdd` on branch `research/benchmark-harness`, worktree DIRTY (tracked diff sha256 22a42d4f3fc7). That commit is the parent of the commit carrying this file, because the results are generated before they are committed.
