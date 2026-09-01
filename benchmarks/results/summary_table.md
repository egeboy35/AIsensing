Radar config `config_phaser` (Phaser_10GHz_DevKit): 500 MHz BW, 64 chirps x 1000 samples, RD map 64 x 667 bins, range bin 0.150 m, velocity bin 0.468 m/s.

16 frames per SNR point, 1 target(s) per frame, base seed 20260819. Association gate: +/-2 range bins (0.300 m), +/-1 Doppler bins (0.468 m/s). Quantization RMSE floor: 0.043 m / 0.135 m/s.

### Scenario `clutter_off`

`apply_realistic_effects=False`, `clutter_intensity=1` -- clutter disabled.

| detector | SNR in (dB) | peak SNR (dB) | Pd | FA/frame | FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| `cfar_advanced_go` | -50 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -45 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -40 | 9.3 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -35 | 11.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -30 | 15.9 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -25 | 20.6 | 0.938 | 0.00 | 0.00e+00 | 0.0528 | 0.1595 | 15 | 0 | 1 |
| `cfar_advanced_go` | -20 | 25.4 | 1.000 | 0.00 | 0.00e+00 | 0.0423 | 0.1396 | 16 | 0 | 0 |
| `cfar_advanced_go` | -15 | 30.6 | 1.000 | 0.00 | 0.00e+00 | 0.0401 | 0.1358 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | -50 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -45 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -40 | 9.3 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -35 | 11.2 | 0.062 | 0.00 | 0.00e+00 | 0.0342 | 0.1288 | 1 | 0 | 15 |
| `cfar_custom_datasetv8` | -30 | 15.9 | 0.938 | 0.00 | 0.00e+00 | 0.0926 | 0.1667 | 15 | 0 | 1 |
| `cfar_custom_datasetv8` | -25 | 20.6 | 1.000 | 0.00 | 0.00e+00 | 0.0512 | 0.1612 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | -20 | 25.4 | 1.000 | 0.00 | 0.00e+00 | 0.0423 | 0.1396 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | -15 | 30.6 | 1.000 | 0.00 | 0.00e+00 | 0.0401 | 0.1358 | 16 | 0 | 0 |
| `cfar_numpy_go` | -50 | 9.2 | 0.000 | 5.31 | 1.26e-04 | n/a | n/a | 0 | 85 | 16 |
| `cfar_numpy_go` | -45 | 9.2 | 0.000 | 5.62 | 1.33e-04 | n/a | n/a | 0 | 90 | 16 |
| `cfar_numpy_go` | -40 | 9.3 | 0.000 | 4.75 | 1.12e-04 | n/a | n/a | 0 | 76 | 16 |
| `cfar_numpy_go` | -35 | 11.2 | 0.438 | 5.50 | 1.30e-04 | 0.1154 | 0.1937 | 7 | 88 | 9 |
| `cfar_numpy_go` | -30 | 15.9 | 1.000 | 5.56 | 1.32e-04 | 0.0897 | 0.1678 | 16 | 89 | 0 |
| `cfar_numpy_go` | -25 | 20.6 | 1.000 | 5.81 | 1.38e-04 | 0.0512 | 0.1612 | 16 | 93 | 0 |
| `cfar_numpy_go` | -20 | 25.4 | 1.000 | 4.88 | 1.15e-04 | 0.0423 | 0.1396 | 16 | 78 | 0 |
| `cfar_numpy_go` | -15 | 30.6 | 1.000 | 4.94 | 1.17e-04 | 0.0401 | 0.1358 | 16 | 79 | 0 |

### Scenario `clutter_default`

`apply_realistic_effects=True`, `clutter_intensity=1` -- clutter/target RCS power -24.5 dB.

| detector | SNR in (dB) | peak SNR (dB) | Pd | FA/frame | FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| `cfar_advanced_go` | -50 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -45 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -40 | 9.3 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -35 | 11.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -30 | 15.8 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -25 | 20.6 | 0.938 | 0.00 | 0.00e+00 | 0.0528 | 0.1595 | 15 | 0 | 1 |
| `cfar_advanced_go` | -20 | 25.4 | 1.000 | 0.00 | 0.00e+00 | 0.0423 | 0.1396 | 16 | 0 | 0 |
| `cfar_advanced_go` | -15 | 30.6 | 1.000 | 0.00 | 0.00e+00 | 0.0401 | 0.1358 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | -50 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -45 | 9.2 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -40 | 9.3 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -35 | 11.2 | 0.062 | 0.00 | 0.00e+00 | 0.0342 | 0.1288 | 1 | 0 | 15 |
| `cfar_custom_datasetv8` | -30 | 15.8 | 0.938 | 0.00 | 0.00e+00 | 0.0926 | 0.1667 | 15 | 0 | 1 |
| `cfar_custom_datasetv8` | -25 | 20.6 | 1.000 | 0.00 | 0.00e+00 | 0.0512 | 0.1612 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | -20 | 25.4 | 1.000 | 0.00 | 0.00e+00 | 0.0423 | 0.1396 | 16 | 0 | 0 |
| `cfar_custom_datasetv8` | -15 | 30.6 | 1.000 | 0.00 | 0.00e+00 | 0.0401 | 0.1358 | 16 | 0 | 0 |
| `cfar_numpy_go` | -50 | 9.2 | 0.000 | 5.31 | 1.26e-04 | n/a | n/a | 0 | 85 | 16 |
| `cfar_numpy_go` | -45 | 9.2 | 0.000 | 5.62 | 1.33e-04 | n/a | n/a | 0 | 90 | 16 |
| `cfar_numpy_go` | -40 | 9.3 | 0.000 | 4.75 | 1.12e-04 | n/a | n/a | 0 | 76 | 16 |
| `cfar_numpy_go` | -35 | 11.2 | 0.438 | 5.50 | 1.30e-04 | 0.1154 | 0.1937 | 7 | 88 | 9 |
| `cfar_numpy_go` | -30 | 15.8 | 1.000 | 5.56 | 1.32e-04 | 0.0897 | 0.1678 | 16 | 89 | 0 |
| `cfar_numpy_go` | -25 | 20.6 | 1.000 | 5.81 | 1.38e-04 | 0.0512 | 0.1612 | 16 | 93 | 0 |
| `cfar_numpy_go` | -20 | 25.4 | 1.000 | 4.94 | 1.17e-04 | 0.0423 | 0.1396 | 16 | 79 | 0 |
| `cfar_numpy_go` | -15 | 30.6 | 1.000 | 4.94 | 1.17e-04 | 0.0401 | 0.1358 | 16 | 79 | 0 |

### Scenario `clutter_strong`

`apply_realistic_effects=True`, `clutter_intensity=10000` -- clutter/target RCS power +15.5 dB.

| detector | SNR in (dB) | peak SNR (dB) | Pd | FA/frame | FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| `cfar_advanced_go` | -50 | 9.1 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -45 | 9.1 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -40 | 8.7 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -35 | 8.9 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -30 | 10.0 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -25 | 12.5 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_advanced_go` | -20 | 15.2 | 0.312 | 0.00 | 0.00e+00 | 0.0313 | 0.1289 | 5 | 0 | 11 |
| `cfar_advanced_go` | -15 | 20.1 | 0.500 | 0.75 | 1.78e-05 | 0.0284 | 0.1450 | 8 | 12 | 8 |
| `cfar_custom_datasetv8` | -50 | 9.1 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -45 | 9.1 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -40 | 8.7 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -35 | 8.9 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -30 | 10.0 | 0.000 | 0.00 | 0.00e+00 | n/a | n/a | 0 | 0 | 16 |
| `cfar_custom_datasetv8` | -25 | 12.5 | 0.375 | 0.00 | 0.00e+00 | 0.0524 | 0.2554 | 6 | 0 | 10 |
| `cfar_custom_datasetv8` | -20 | 15.2 | 0.625 | 0.56 | 1.33e-05 | 0.0790 | 0.1530 | 10 | 9 | 6 |
| `cfar_custom_datasetv8` | -15 | 20.1 | 0.938 | 3.06 | 7.22e-05 | 0.0383 | 0.1354 | 15 | 49 | 1 |
| `cfar_numpy_go` | -50 | 9.1 | 0.000 | 5.31 | 1.26e-04 | n/a | n/a | 0 | 85 | 16 |
| `cfar_numpy_go` | -45 | 9.1 | 0.000 | 5.56 | 1.32e-04 | n/a | n/a | 0 | 89 | 16 |
| `cfar_numpy_go` | -40 | 8.7 | 0.000 | 4.75 | 1.12e-04 | n/a | n/a | 0 | 76 | 16 |
| `cfar_numpy_go` | -35 | 8.9 | 0.062 | 5.44 | 1.29e-04 | 0.1841 | 0.1288 | 1 | 87 | 15 |
| `cfar_numpy_go` | -30 | 10.0 | 0.125 | 5.56 | 1.32e-04 | 0.0858 | 0.2756 | 2 | 89 | 14 |
| `cfar_numpy_go` | -25 | 12.5 | 0.500 | 6.44 | 1.52e-04 | 0.1092 | 0.2338 | 8 | 103 | 8 |
| `cfar_numpy_go` | -20 | 15.2 | 0.875 | 7.25 | 1.72e-04 | 0.0908 | 0.1834 | 14 | 116 | 2 |
| `cfar_numpy_go` | -15 | 20.1 | 0.938 | 11.31 | 2.68e-04 | 0.0383 | 0.1354 | 15 | 181 | 1 |

Runtime: 347.3 s wall clock for 384 simulated frames (72.7 s simulation, cfar_advanced_go 91.8 s, cfar_custom_datasetv8 91.2 s, cfar_numpy_go 91.7 s).

Environment: Python 3.14.2, numpy 2.4.2, scipy 1.18.0, matplotlib 3.11.1, torch not installed, Windows-11-10.0.26200-SP0, device cpu. Repo commit `8a7502f366f1`.
