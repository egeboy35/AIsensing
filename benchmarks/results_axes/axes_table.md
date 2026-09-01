Radar config `config_phaser` (Phaser_10GHz_DevKit): baseline RD map 64 x 667 bins, range bin 0.150 m, velocity bin 0.468 m/s.

32 frames per SNR point, 1 target per frame, base seed 20260822, SNR referenced to **target** power, clutter disabled. Association gate held fixed in physical units at +/-0.300 m and +/-0.468 m/s and converted to bins per configuration, because two axes change the bin grid.

**The common operating point.** Every configuration below is calibrated on its own target-free frames to the same measured rate of **4.243 false alarms per frame** (= 1e-04 per cell on the baseline 64 x 667 grid, which is the first study's headline point). A rate per frame rather than per cell, because the zero-padding and chirp-count axes change the number of cells while covering the same physical volume; per-cell equality would hand the finer grid more false alarms per frame. 20 target-free frames per configuration give 85 expected false-alarm events at that rate.

**Did the calibration hold?** At the bottom of the sweep (-45.0 dB), where nothing is being detected and the false-alarm count is not reduced by matched detections, the configurations realise 3.62 to 5.16 false alarms per frame (mean 4.14) against the 4.24 they were solved for. That residual is the sampling error of solving a threshold on 20 target-free frames; converted through each calibration curve's own slope it is at most 0.09 dB of threshold (mean 0.03 dB), which is the resolution at which any claim below should be read.

**How to read the result columns.** `FA/frame at floor` is the check above, per row. `Pd@-36.0 dB` is the detection rate at the fixed SNR point nearest the baseline's Pd = 0.50 crossing, and `Pd@-27.0 dB` is the top of the sweep, where a configuration that throws targets away shows a ceiling below 1. `shift` is the dB of sensitivity the configuration buys: the SNR at which the baseline reaches Pd = 0.50 minus the SNR at which this configuration reaches it, so **positive means it detects the same target further down**. Intervals are 95% paired-bootstrap intervals over 2000 resamples of the 32 scenes.

### What the measurement says

**4 of the 8 axes move the curve** by more than their own 95% interval (`cfar_training_cells`, `range_zero_padding`, `coherent_chirps`, `noncoherent_looks`); the other 4 do not at this budget (`cfar_guard_cells`, `nms_kernel`, `mtd_filter`, `cfar_averaging`).

**Doubling the dwell time, two ways.** Doubling the chirp count -- coherent integration, one detection per frame as before -- buys +2.90 dB [+1.31, +4.25]. Spending the same extra dwell on a second look integrated non-coherently buys +2.76 dB [+1.61, +4.06].

The intervals overlap, so this study cannot rank the two at equal dwell; it can only say that both are worth more than every other axis it swept. The non-coherent route needs no change to the waveform at all -- it is arithmetic on range-Doppler maps the pipeline already computes -- while the coherent route changes the Doppler grid and the frame rate.

**What "flat" means here.** The largest movement anywhere among the flat axes is `num_guard` = 8 at +0.19 dB [+0.00, +1.62], an interval that spans zero; resolving a difference that small at this operating point would need about 114 frames per SNR point instead of 32. Read those rows as "this study cannot distinguish these settings", not as "these settings are identical" -- the upper end of each interval is what an unmeasured effect could still be worth.

**The moving-target filter.** `mtd=True` -- which the dataset pipeline enables whenever `apply_realistic_effects` is set -- moves the Pd = 0.50 crossing by -0.17 dB [-1.50, +0.30], and caps Pd at 0.906 at the top of the sweep against 1.000 without it, because 3 of the 32 drawn targets are slower than the 1 m/s it discards. It buys no sensitivity at this operating point and costs every slow target.

**Non-maximum suppression is not about detection, it is about counting.** Disabling it (`nms_kernel_size=1`) does not move the Pd curve (-0.07 dB [-0.75, +0.00]), but over the swept frames it produces +1.84 false alarms per frame relative to the shipped kernel at the same target-free rate -- the target's own main lobe, reported as several detections instead of one.

### Summary: which axis moves the curve

Ranked by **span**: the dB between the best and the worst value swept on that axis. A large span means the knob matters -- whether or not the shipped value can be improved on, getting it wrong costs that many dB. All shifts are relative to the axis's own baseline value.

| axis | knob | baseline | best value (shift dB) | worst value (shift dB) | span (dB) | any shift resolved? | cost |
|---|---|---|---|---|---|---|---|
| `coherent_chirps` | `N_chirps` | 64 | 128 (+2.90) | 32 (-2.62) | 5.53 | yes | simulation and detection cost scale with N_chirps; dwell time doubles per doubling |
| `noncoherent_looks` | `looks` | 1 | 8 (+6.43) | 2 (+2.76) | 3.67 | yes | L simulated frames per detection; detector cost unchanged |
| `cfar_training_cells` | `num_train` | 10 | 6 (+0.12) | 2 (-1.50) | 1.62 | yes | num_train=2 runs ~5x faster than num_train=10; num_train=16 ~1.9x slower |
| `range_zero_padding` | `zero_pad_factor` | 2 | 1 (+0.36) | 4 (-0.25) | 0.62 | yes | detector cost is proportional to the number of range bins |
| `cfar_guard_cells` | `num_guard` | 4 | 8 (+0.19) | 0 (+0.00) | 0.19 | no | window grows as (2*(train+guard)+1)^2; guard=12 costs ~2.4x the baseline convolution |
| `nms_kernel` | `nms_kernel_size` | 5 | 3 (+0.00) | 1 (-0.07) | 0.07 | no | a maximum filter over the whole map; negligible next to the CFAR convolution |
| `cfar_averaging` | `method` | GO | CA (+0.06) | SO (+0.02) | 0.05 | no | identical (the same convolutions, combined differently) |
| `mtd_filter` | `mtd` | off | on (-0.17) | on (-0.17) | 0.00 | no | free (a filter on the detection list) |

### `cfar_guard_cells` -- `num_guard`

Do the guard cells cover the target's own main lobe? With a Hann range window and zero_pad_factor=2 the main lobe is about +/-4 range bins wide, so a guard block narrower than that feeds target energy into the noise estimate and the detector masks itself.

Cost: window grows as (2*(train+guard)+1)^2; guard=12 costs ~2.4x the baseline convolution. Baseline value: `4`.

| num_guard | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 9.68 | 4.28 | 0.406 | -0.031 [-0.094, +0.000] | 1.000 | -35.44 | +0.00 [-0.75, +0.70] | not resolved (needs ~112 frames/point) |
| 2 | 9.66 | 4.41 | 0.406 | -0.031 [-0.094, +0.000] | 1.000 | -35.55 | +0.04 [-0.43, +1.00] | not resolved (needs ~112 frames/point) |
| **4** | 9.67 | 4.00 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| 6 | 9.61 | 4.12 | 0.406 | -0.031 [-0.094, +0.000] | 1.000 | -35.50 | +0.00 [-0.50, +0.83] | not resolved (needs ~122 frames/point) |
| 8 | 9.61 | 4.00 | 0.469 | +0.031 [+0.000, +0.094] | 1.000 | -35.81 | +0.19 [+0.00, +1.62] | not resolved (needs ~114 frames/point) |
| 12 | 9.59 | 4.34 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.70 | +0.12 [+0.00, +1.31] | not resolved (dPd is exactly 0 here) |

### `cfar_training_cells` -- `num_train`

Classical CFAR loss: a noise estimate averaged over N cells needs a higher threshold than a known noise level for the same false-alarm rate. The shipped num_train=10 gives 580 training cells per GO branch, where that loss should already be negligible -- so this axis is a test of whether the shipped value is buying anything for its cost.

Cost: num_train=2 runs ~5x faster than num_train=10; num_train=16 ~1.9x slower. Baseline value: `10`.

| num_train | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| 2 | 9.77 | 4.28 | 0.219 | -0.219 [-0.375, -0.094] | 1.000 | -34.00 | -1.50 [-3.25, -0.38] | loss |
| 4 | 9.65 | 4.41 | 0.312 | -0.125 [-0.250, -0.031] | 1.000 | -35.18 | -0.25 [-1.25, +0.30] | not resolved (needs ~32 frames/point) |
| 6 | 9.62 | 4.28 | 0.438 | +0.000 [-0.094, +0.094] | 1.000 | -35.67 | +0.12 [-0.31, +1.25] | not resolved (dPd is exactly 0 here) |
| **10** | 9.67 | 4.00 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| 16 | 9.61 | 4.25 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.67 | +0.10 [+0.00, +1.20] | not resolved (dPd is exactly 0 here) |

### `nms_kernel` -- `nms_kernel_size`

Non-maximum suppression thins the peak list, so it lowers the measured false-alarm rate and buys threshold back. 1 disables it entirely (the repo's `if nms_kernel_size > 1` guard), which makes every threshold crossing a detection.

Cost: a maximum filter over the whole map; negligible next to the CFAR convolution. Baseline value: `5`.

| nms_kernel_size | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | 9.70 | 3.97 | 0.406 | -0.031 [-0.094, +0.000] | 1.000 | -35.36 | -0.07 [-0.75, +0.00] | not resolved (needs ~112 frames/point) |
| 3 | 9.68 | 4.00 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.50 | +0.00 [+0.00, +0.00] | not resolved (dPd is exactly 0 here) |
| **5** | 9.67 | 4.00 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| 7 | 9.67 | 4.00 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.50 | +0.00 [+0.00, +0.00] | not resolved (dPd is exactly 0 here) |
| 9 | 9.67 | 4.00 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.50 | +0.00 [-0.02, +0.00] | not resolved (dPd is exactly 0 here) |
| 15 | 9.67 | 4.06 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.50 | +0.00 [-0.02, +0.00] | not resolved (dPd is exactly 0 here) |

### `range_zero_padding` -- `zero_pad_factor`

Zero-padding the range FFT adds no information, but it reduces straddle (scalloping) loss when a target falls between bins. It also doubles the cell count, which at a fixed false-alarm rate per frame costs threshold.

Cost: detector cost is proportional to the number of range bins. Baseline value: `2`.

| zero_pad_factor | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | 9.27 | 4.28 | 0.500 | +0.062 [+0.000, +0.156] | 1.000 | -36.00 | +0.36 [+0.02, +1.73] | gain |
| **2** | 9.67 | 4.00 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| 4 | 9.84 | 3.97 | 0.344 | -0.094 [-0.188, +0.000] | 1.000 | -35.17 | -0.25 [-1.33, +0.29] | not resolved (needs ~38 frames/point) |

### `coherent_chirps` -- `N_chirps`

Coherent integration: the Doppler FFT sums N_chirps samples in phase, so the target-to-noise ratio in the map should rise 3 dB per doubling. The dwell time rises with it -- compare against noncoherent_looks, which spends the same extra dwell without phase coherence.

Cost: simulation and detection cost scale with N_chirps; dwell time doubles per doubling. Baseline value: `64`.

| N_chirps | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| 32 | 9.19 | 5.16 | 0.062 | -0.375 [-0.562, -0.188] | 1.000 | -32.88 | -2.62 [-3.59, -1.09] | loss |
| **64** | 9.67 | 4.00 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| 128 | 9.96 | 3.62 | 0.812 | +0.375 [+0.125, +0.594] | 1.000 | -38.33 | +2.90 [+1.31, +4.25] | gain |

### `noncoherent_looks` -- `looks`

Non-coherent integration: average the power maps of L successive frames of the same scene, then detect once. Costs L times the dwell, exactly like doubling the chirp count -- so the two axes are directly comparable per unit of time spent.

Cost: L simulated frames per detection; detector cost unchanged. Baseline value: `1`.

| looks | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | 9.67 | 4.00 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| 2 | 6.30 | 4.28 | 0.844 | +0.406 [+0.250, +0.594] | 1.000 | -38.25 | +2.76 [+1.61, +4.06] | gain |
| 4 | 3.99 | 4.31 | 1.000 | +0.562 [+0.405, +0.719] | 1.000 | -40.18 | +4.67 [+3.78, +6.19] | gain |
| 8 | 2.33 | 4.25 | 1.000 | +0.562 [+0.405, +0.719] | 1.000 | -42.00 | +6.43 [+5.38, +7.76] | gain |

### `mtd_filter` -- `mtd`

The dataset pipeline turns `_cfar_2d_custom`'s moving-target filter on whenever apply_realistic_effects is set. It discards every detection with |v| < 1 m/s, which removes false alarms near zero Doppler -- and also removes any real target that happens to be slow.

Cost: free (a filter on the detection list). Baseline value: `off`.

| mtd | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| **off** | 9.67 | 4.00 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| on | 9.63 | 3.94 | 0.375 | -0.062 [-0.156, +0.000] | 0.906 | -35.25 | -0.17 [-1.50, +0.30] | not resolved (needs ~59 frames/point) |

### `cfar_averaging` -- `method`

GO-CFAR takes the larger of the two training-branch means, which raises the threshold to protect against clutter edges. In homogeneous noise that protection is a loss. Measured on cfar_2d_numpy, the one shipped detector whose `method` argument is reachable (`_cfar_2d_custom` hard-codes GO).

Cost: identical (the same convolutions, combined differently). Baseline value: `GO`.

| method | calibrated thr (dB) | FA/frame at floor | Pd@-36.0 dB | dPd [95% CI] | Pd@-27.0 dB | SNR at Pd=0.50 (dB) | shift (dB) [95% CI] | verdict |
|---|---|---|---|---|---|---|---|---|
| **GO** | 9.67 | 3.97 | 0.438 | -- | 1.000 | -35.50 | -- | baseline |
| CA | 9.72 | 4.03 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.62 | +0.06 [+0.00, +1.02] | not resolved (dPd is exactly 0 here) |
| SO | 9.83 | 4.31 | 0.438 | +0.000 [+0.000, +0.000] | 1.000 | -35.57 | +0.02 [+0.00, +0.60] | not resolved (dPd is exactly 0 here) |

### Calibrated thresholds and what they cost

Every row is a solve on target-free frames, so the threshold column is a *result*, not a setting. A configuration that produces fewer peaks per frame for the same threshold gets its threshold lowered until the rate matches, which is where part of its gain comes from.

| config | detector | knob | calibrated knob value | effective thr (dB) | status |
|---|---|---|---|---|---|
| `cfar_custom_datasetv8__train10_guard0_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.18500 | 9.68 | interpolated |
| `cfar_custom_datasetv8__train10_guard12_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.09513 | 9.59 | interpolated |
| `cfar_custom_datasetv8__train10_guard2_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.16640 | 9.66 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms15_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.17463 | 9.67 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms1_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.20468 | 9.70 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms3_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.18215 | 9.68 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp1_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 11.77288 | 9.27 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp2_nc128_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.46375 | 9.96 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp2_nc32_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 11.69833 | 9.19 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.18069 | 9.67 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp2_nc64_looks1_mtd1` | `cfar_custom_datasetv8` | `threshold_offset` | 12.13796 | 9.63 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp2_nc64_looks2_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 8.81028 | 6.30 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp2_nc64_looks4_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 6.49358 | 3.99 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp2_nc64_looks8_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 4.84056 | 2.33 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms5_zp4_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.35178 | 9.84 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms7_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.18069 | 9.67 | interpolated |
| `cfar_custom_datasetv8__train10_guard4_nms9_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.18069 | 9.67 | interpolated |
| `cfar_custom_datasetv8__train10_guard6_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.11598 | 9.61 | interpolated |
| `cfar_custom_datasetv8__train10_guard8_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.11426 | 9.61 | interpolated |
| `cfar_custom_datasetv8__train16_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.11460 | 9.61 | interpolated |
| `cfar_custom_datasetv8__train2_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.27999 | 9.77 | interpolated |
| `cfar_custom_datasetv8__train4_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.15421 | 9.65 | interpolated |
| `cfar_custom_datasetv8__train6_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_custom_datasetv8` | `threshold_offset` | 12.12820 | 9.62 | interpolated |
| `cfar_numpy_ca__train10_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_numpy_ca` | `magnitude_warp_exponent` | 0.98121 | 9.72 | interpolated |
| `cfar_numpy_go__train10_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_numpy_go` | `magnitude_warp_exponent` | 0.98535 | 9.67 | interpolated |
| `cfar_numpy_so__train10_guard4_nms5_zp2_nc64_looks1_mtd0` | `cfar_numpy_so` | `magnitude_warp_exponent` | 0.97262 | 9.83 | interpolated |

### The error bar

A Pd measured over 32 frames with one target each has a binomial standard error of 0.088 at Pd = 0.5, and is quantized to 1/32 = 0.0312. That is the error bar on an **absolute** Pd, and it is why no absolute Pd in this study should be read to better than about +/-0.18.

The differences are better resolved than that, because they are paired: every configuration is measured on the same physical scenes and the same noise draws, so scene difficulty cancels. The bracketed intervals are the bootstrap of that paired difference, and they are the numbers to read. Where an interval spans zero the study **cannot resolve that axis** at this budget, and the verdict column states the per-point frame budget that would.

Total: 26 configurations, 14176 detector calls.

Provenance: everything above depends only on the seeds. Wall-clock timings, the environment and the git state are deliberately kept out of this block and written to `results_axes/axes_run_meta.json` instead, so that two runs of the study produce byte-identical copies of every other artifact.
