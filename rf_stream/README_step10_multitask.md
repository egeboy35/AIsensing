# RF Stream PHY — Step 10 & Multi-Task Learning Documentation

**Project:** AIsensing / GlobeCom 2026 — Multi-Task PHY with Neural Gate  
**Hardware:** ADALM-PlutoSDR (AD9361, 70 MHz – 6 GHz, 20 MHz BW)  
**Platform:** TX = Jetson Orin (ip:192.168.3.2), RX = x86 Ubuntu (ip:192.168.2.2)

---

## 1. System Overview

```
┌─────────────────────────────┐       RF cable / OTA antenna
│   Jetson Orin (TX)          │       (2.4 GHz, 3 MHz BW)
│   rf_stream_tx_step6phy.py  │──────────────────────────────────►
│   PlutoSDR @ 192.168.3.2    │                                    │
└─────────────────────────────┘                                    ▼
                                                    ┌─────────────────────────────┐
                                                    │   x86 Ubuntu (RX)           │
                                                    │   rf_stream_rx_step10phy.py │
                                                    │   PlutoSDR @ 192.168.2.2    │
                                                    └─────────────────────────────┘
                                                                   │ --save_npz
                                                                   ▼
                                                         NPZ captures (rawiq, meta)
                                                                   │
                                                                   ▼
                                                        train_multitask_v2.py
                                                     (gate + mod + SNR, phase-aug)
```

### PHY Frame Structure
```
│ gap_long │ STF (×6) │ LTF (×8) │ OFDM data symbols │ gap_long │
```
- **STF:** Schmidl-Cox sync, 6 repetitions (`--stf_repeats 6`)
- **LTF:** Long training field, 8 repetitions (`--ltf_symbols 8`) for channel estimation
- **Data:** OFDM, N_FFT=64, CP=16, 48 data subcarriers, 4 pilots at ±7, ±21
- **Modulations:** QPSK (2 bps), QAM16 (4 bps)
- **Repeat factor:** `--repeat 4` (4× bit repetition for diversity)
- **Packet:** 64-byte payload + MAGIC(AIS1) + seq + CRC32 = 78 bytes framed
- **fc for OTA:** 2400 MHz (changed from 2300 MHz in step8/9)

---

## 2. File Inventory

| File | Role |
|---|---|
| `rf_stream_tx_step6phy.py` | TX: all modulations, non-cyclic streaming, IQ-swap compensation |
| `rf_stream_rx_step10phy.py` | RX Step10: MMSE eq + pilot_weight blending + auto_z_th |
| `train_multitask_v2.py` | Multi-task trainer v2: gate + mod + SNR, phase-aug, cnn/attn arch |
| `analyze_runs_offline.py` | Offline BER/EVM analysis from saved NPZ directories |
| `generate_multitask_figures.py` | Publication figures for multitask results |
| `generate_paper_figures.py` | Publication figures for BER/gate ROC |
| `multitask_model_v2/` | Trained model checkpoints + metrics JSON files |
| `paper/globecom2026_multitask_phy.tex` | GlobeCom 2026 paper source |
| `paper_figures/` | All PDF figures referenced by the paper |

---

## 3. Step 10 RX — What's New vs Step 9

**File:** `rf_stream_rx_step10phy.py`

### Three New Features

**1. MMSE Equalization (`--equalization mmse`)**
```
H_mmse[k] = H[k]* / (|H[k]|² + noise_var)
```
Noise variance estimated from LTF SNR measurement. Reduces noise enhancement
at subcarriers with poor channel response (dominant benefit at low SNR / OTA).
ZF mode (`--equalization zf`) gives step8/9-equivalent behavior.

**2. Pilot Correction Weight (`--pilot_weight 0.0–1.0`)**
```
H_eff[k] = (1 - w) × H_ltf[k] + w × H_pilot_corrected[k]
```
- `w=0.0`: pure LTF channel estimate (step8 behavior)
- `w=1.0`: full per-symbol pilot residual correction (step9 behavior)
- `w=0.5`: optimal blend for OTA — corrects time-varying multipath without
  amplifying pilot noise on static cable segments

**3. Modulation-Aware Energy Threshold (`--auto_z_th`)**
```
QAM16 PAPR ≈ 3 dB higher than QPSK → lower average energy per sample
Scale factors: qpsk×1.0,  qam16×0.65,  bpsk×1.2
```
Prevents QAM16 OTA packets from being missed by the energy detector.
Use with default `--energy_z_th 8.0` — the effective threshold becomes 5.2 for QAM16.

### Equalization Pipeline (Stage 1 + Stage 2)
```
Stage 1: MMSE bulk equalization
  noise_var = 10^(-SNR_dB/10)          (from LTF power ratio)
  H_mmse[k] = H[k]* / (|H[k]|² + noise_var)
  Ye = Y[k] / H_mmse[k]               for all used subcarriers

Stage 2: pilot-weight residual correction
  residual = Ye[pilots] / (pilot_sign × pilot_vals)
  if mean|residual - 1| < bypass_th → skip (flat channel)
  else:
    phase_res = unwrap(angle(residual))   → linear interpolate → all 64 bins
    Ye_corr = Ye / exp(j × phase_res)    × pilot_weight + Ye × (1-pilot_weight)
```

---

## 4. Multi-Task Learning — Architecture

**File:** `train_multitask_v2.py`

### MT-PreamCNN (CNN baseline)
```
Input: 800 IQ samples → interleaved real/imag → shape [1600]
Conv1D(1, 64, 7) → BN → ReLU → MaxPool(4)     [400 → 100 steps]
Conv1D(64, 128, 5) → BN → ReLU → MaxPool(4)   [100 → 25 steps]
Conv1D(128, 256, 3) → BN → ReLU               [25 steps, d=256]
AdaptiveAvgPool(50) → [50 × 256]
GlobalAvgPool → [256]                           ← encoder output
├── Gate head:  Linear(256,1) → sigmoid         gate_p ∈ [0,1]
├── Mod head:   Linear(256,2) → softmax         P(QPSK), P(QAM16)
└── SNR head:   Linear(256,1)                   normalized SNR
Params: 361,891
```

### MT-PreamCNN-Attn (attention model — main contribution)
```
Identical CNN encoder → [50 × 256]
4-head MHSA block:
  MultiheadAttention(d=128, nhead=4, dropout=0.1)
  LayerNorm + residual
  FFN(128→512→128) + LayerNorm
  Project back → [50 × 256]
GlobalAvgPool → [256]                           ← encoder output
├── Gate head  (same as CNN)
├── Mod head   (same as CNN)
└── SNR head   (same as CNN)
Params: 428,195  (+18% over CNN baseline)
```

### Phase-Rotation Augmentation (`--phase_aug`)
```python
# Applied per sample per training step (not at inference)
θ ~ U[0, 2π)
x̃[n] = x[n] · exp(jθ)   # rotates the IQ plane uniformly

# Equivalent in real-valued interleaved format:
I_new =  I·cos(θ) - Q·sin(θ)
Q_new =  I·sin(θ) + Q·cos(θ)
```
**Why:** Schmidl-Cox CFO correction removes frequency drift but leaves a random
initial phase offset θ₀ per capture. On CoaxSweep (+330 Hz CFO) vs AirLink
(−3000 Hz CFO), the phase trajectories are opposite, inverting phase-sensitive
features and collapsing modulation accuracy. Phase-aug forces the encoder to rely
exclusively on amplitude structure (single ring for QPSK vs three rings for QAM16),
which is phase-invariant.

### Training Modes
| Mode | Train set | Primary eval | Use case |
|---|---|---|---|
| `zeroshot` | CoaxSweep only | CoaxSweep val (20%) | Baseline; AirLink eval is unseen |
| `mixed` | CoaxSweep + 80% AirLink | AirLink held-out (20%) | Best OTA performance |
| `coax` | CoaxSweep only | CoaxSweep val | Cable-only deployment |

### Loss Function
```
L = w_gate × L_BCE(gate) + w_mod × L_CE(mod) + w_snr × L_MSE(snr)
w_gate=1.0,  w_mod=0.5,  w_snr=0.1
```
Mod labels: `mod_label=-1` for gate=0 (fail) captures — excluded from mod CE loss.
**Critical:** WiFi-triggered fail NPZs have `bps=2` from TX config; if fails were
included in mod eval, they inflate QAM16 accuracy (bug fixed at lines 124–128
of `train_multitask_v2.py`).

---

## 5. Datasets

### CoaxSweep (cable, high-SNR reference)

| Collection | Runs | Samples | Notes |
|---|---|---|---|
| ber_sweep/ (v1) | 5 runs | ~3,000 | April 28-29, 2300 MHz, qpsk+qam16 |
| ber_sweep_v2/ | 6 runs | ~3,500 | April 29, 2300 MHz |
| ber_sweep_v3/ | 6 runs | ~3,000 | April 29, 2300 MHz |
| ber_sweep_v4/ | 3 runs | ~4,500 | May 1, 2300 MHz |
| **Total** | **19 dirs** | **18,455** | ok=13,477, fail=4,978 |

Data root: `rf_stream/ber_sweep_v{2,3,4}/run_*` and `rf_stream/ber_sweep/run_*`

### AirLink (OTA antenna, 2400 MHz)

| Collection | Distance | Environment | Runs | OK NPZs | Fail NPZs |
|---|---|---|---|---|---|
| AirLink-I QPSK | ~40 cm | Office, no WiFi | 1 | ~800 | ~200 |
| AirLink-II QPSK | ~40 cm | Office, low WiFi | 2 | ~400 | ~600 |
| AirLink-I QAM16 | ~40 cm | Office | 1 | ~481 | ~300 |
| AirLink-III QPSK | ~50 cm | 2.4 GHz WiFi congested | 5 | 332 | 4,794 |
| **Total** | | | **9 dirs** | **2,123** | **5,342** |

Data root (on local machine, not committed):
```
/tmp/airlink_qpsk/run_20260502_215517
/tmp/airlink2_qpsk/run_20260502_231843
/tmp/airlink2_qpsk/run_20260502_232533
/tmp/airlink_qam16/run_20260502_215103
/tmp/airlink3_qpsk/run_20260503_12*    (5 runs, WiFi-congested)
```

**AirLink-III captures** (airlink3_qpsk) are especially valuable:
- 4,794 WiFi-triggered fail NPZs serve as hard-negative gate examples
- These dramatically improved mixed gate AUC (0.89 → 0.98) during training

### NPZ File Format
Each capture is saved as `{timestamp}_{ok|fail}.npz` with keys:
```python
npz["rxw"]       # np.complex64 [262144] — full raw IQ window
npz["meta_json"] # JSON string with: cap, status, bps, cfo_hz, snr_db, modulation
```
The training script loads `rxw[0:800]` (post-CFO-corrected preamble window).

---

## 6. Launch Commands

### TX (on Jetson Orin) — Step 6

```bash
# QPSK OTA @ 2400 MHz
nohup python3 rf_stream_tx_step6phy.py \
  --uri ip:192.168.3.2 --fc 2400e6 --fs 3e6 \
  --modulation qpsk --tx_gain 0 --repeat 4 \
  --ltf_symbols 8 --stf_repeats 6 \
  > /tmp/tx_qpsk.log 2>&1 &

# QAM16 OTA @ 2400 MHz
nohup python3 rf_stream_tx_step6phy.py \
  --uri ip:192.168.3.2 --fc 2400e6 --fs 3e6 \
  --modulation qam16 --tx_gain 0 --repeat 4 \
  --ltf_symbols 8 --stf_repeats 6 \
  > /tmp/tx_qam16.log 2>&1 &
```

### RX — Step 10 (on x86)

```bash
# QPSK OTA — validated configuration (40–50 cm, 2400 MHz)
python3 rf_stream/rf_stream_rx_step10phy.py \
  --uri ip:192.168.2.2 --fc 2400e6 --fs 3e6 \
  --modulation qpsk --repeat 4 --ltf_symbols 8 --stf_repeats 6 \
  --rx_gain 30 --equalization mmse --pilot_weight 0.5 --auto_z_th \
  --save_npz --save_fail_npz --fail_npz_prob 1.0 \
  --out_root /tmp/airlink_qpsk

# QAM16 OTA — auto_z_th scales threshold ×0.65 for QAM16
python3 rf_stream/rf_stream_rx_step10phy.py \
  --uri ip:192.168.2.2 --fc 2400e6 --fs 3e6 \
  --modulation qam16 --repeat 4 --ltf_symbols 8 --stf_repeats 6 \
  --rx_gain 30 --equalization mmse --pilot_weight 0.5 --auto_z_th \
  --save_npz --save_fail_npz --fail_npz_prob 1.0 \
  --out_root /tmp/airlink_qam16

# 8-gain BER sweep (cable or OTA)
python3 rf_stream/rf_stream_rx_step10phy.py \
  --uri ip:192.168.2.2 --fc 2400e6 --fs 3e6 \
  --modulation qpsk --repeat 4 --ltf_symbols 8 --stf_repeats 6 \
  --rx_gain 60 \
  --rx_gain_sweep "60,55,50,45,40,35,30,25" --gain_step_s 15 \
  --max_caps 1200 \
  --equalization mmse --pilot_weight 0.5 --auto_z_th \
  --save_npz \
  --out_root /tmp/sweep_step10_qpsk

# Quick sanity check (50 packets, no sweep)
python3 rf_stream/rf_stream_rx_step10phy.py \
  --uri ip:192.168.2.2 --fc 2400e6 --fs 3e6 \
  --modulation qpsk --repeat 4 --ltf_symbols 8 --stf_repeats 6 \
  --rx_gain 30 --max_caps 50 \
  --equalization mmse --pilot_weight 0.5 --auto_z_th \
  --out_root /tmp/test_step10
```

### Multi-Task Training (on x86 with GPU)

```bash
# Train all 4 baseline models (no phase-aug, ~40 min total on GPU)
for arch in cnn attn; do
  for mode in zeroshot mixed; do
    python3 rf_stream/train_multitask_v2.py \
      --arch $arch --mode $mode \
      --airlink_dirs /tmp/airlink_qpsk/run_* /tmp/airlink2_qpsk/run_* \
                     /tmp/airlink_qam16/run_* /tmp/airlink3_qpsk/run_20260503_12* \
      --epochs 80
  done
done

# Train all 4 models WITH phase-rotation augmentation (--phase_aug, ~40 min total)
for arch in cnn attn; do
  for mode in zeroshot mixed; do
    python3 rf_stream/train_multitask_v2.py \
      --arch $arch --mode $mode \
      --airlink_dirs /tmp/airlink_qpsk/run_* /tmp/airlink2_qpsk/run_* \
                     /tmp/airlink_qam16/run_* /tmp/airlink3_qpsk/run_20260503_12* \
      --epochs 80 --phase_aug
  done
done

# Or use the batch script (sequential, logs to /tmp/training_phaseaug.log):
bash /tmp/run_phase_aug.sh
```

### AirLink Data Collection Sequence

```bash
# Step 1: Start TX on Jetson (run continuously during collection)
nohup python3 rf_stream_tx_step6phy.py \
  --uri ip:192.168.3.2 --fc 2400e6 --fs 3e6 \
  --modulation qpsk --tx_gain 0 --repeat 4 \
  --ltf_symbols 8 --stf_repeats 6 > /tmp/tx.log 2>&1 &

# Step 2: Collect on RX (run 3–5 times, each ~300 good packets)
python3 rf_stream/rf_stream_rx_step10phy.py \
  --uri ip:192.168.2.2 --fc 2400e6 --fs 3e6 \
  --modulation qpsk --repeat 4 --ltf_symbols 8 --stf_repeats 6 \
  --rx_gain 0 --equalization mmse --pilot_weight 0.5 --auto_z_th \
  --save_npz --save_fail_npz --fail_npz_prob 1.0 \
  --max_caps 400 \
  --out_root /tmp/airlink_qpsk

# Step 3: Verify collection
python3 -c "
import glob, numpy as np
ok = glob.glob('/tmp/airlink_qpsk/**/*_ok.npz', recursive=True)
fail = glob.glob('/tmp/airlink_qpsk/**/*fail*.npz', recursive=True)
print(f'OK: {len(ok)}, Fail: {len(fail)}')
"
```

---

## 7. Experimental Results

### Hardware Configuration (Step 10 OTA)

| Parameter | Value |
|---|---|
| Center frequency (fc) | 2400 MHz |
| Sample rate (fs) | 3 MHz |
| FFT size (N_FFT) | 64 |
| Cyclic prefix (N_CP) | 16 samples |
| Data subcarriers | 48 |
| Pilot subcarriers | 4 (bins ±7, ±21) |
| STF repetitions | 6 |
| LTF repetitions | 8 |
| Repeat factor | 4 |
| TX gain | 0 dB (max) |
| Equalization | MMSE |
| Pilot weight | 0.50 |
| Auto z-threshold | Enabled |
| TX–RX separation | 40 cm (Jetson Orin → local panel antenna) |
| Measured CFO | −3000 Hz (stable Pluto TCXO offset) |
| Measured SNR | 17–28 dB across gain levels |

### OTA Decode Results (Step 10, 2400 MHz, 40 cm)

**QPSK OTA — 1200 packets, 8-gain sweep:**

| RX Gain | Decoded | Decode Rate |
|---|---|---|
| 60 dB | 103/150 | 68.7% |
| 55 dB | 129/150 | 86.0% |
| 50 dB | 130/150 | 86.7% |
| 45 dB | 129/150 | 86.0% |
| 40 dB | 128/150 | 85.3% |
| 35 dB | 129/150 | 86.2% |
| 30 dB | 116/150 | 77.3% |
| 25 dB | 67/150 | 44.7% — worst gain |
| **Overall** | **931/1200** | **77.6%** |

**QAM16 OTA — 1200 packets, 8-gain sweep:**

| RX Gain | Decoded | Decode Rate |
|---|---|---|
| 25 dB | 70/150 | 46.8% — worst gain |
| 60 dB | 143/150 | 95.3% |
| Best gains (40–55 dB) | ~143/150 | ~95.5% |
| **Overall** | **1054/1200** | **87.8%** |

> BER=0 on all successfully decoded packets (CRC-pass). Failures are from
> undetected packets (energy/NCC gate missed), not bit errors.

> Gain=25 dB is worst for both modulations: below the effective AGC floor,
> packet energy barely exceeds noise floor after MMSE.

### Multi-Task Model Results — Baseline (No Augmentation)

**Dataset:** 18,455 CoaxSweep + 7,465 AirLink = 25,920 total samples  
**Eval splits:** CoaxSweep val (20%), AirLink-ZeroShot (full AirLink), AirLink-HeldOut (20%)

| Model | Domain | Gate AUC | Mod Acc | SNR RMSE |
|---|---|---|---|---|
| MT-PreamCNN (zeroshot) | CoaxSweep | 0.9972 | 100.0% | 0.91 dB |
| MT-PreamCNN (zeroshot) | AirLink | 0.8397 | 32.69% ⚠ | 4.48 dB |
| MT-PreamCNN (mixed) | CoaxSweep | 0.9960 | 100.0% | 1.09 dB |
| MT-PreamCNN (mixed) | AirLink held-out | 0.9826 | 71.96% | 1.70 dB |
| MT-PreamCNN-Attn (zeroshot) | CoaxSweep | 0.9977 | 100.0% | 0.83 dB |
| MT-PreamCNN-Attn (zeroshot) | AirLink | 0.8851 | 48.94% | 4.36 dB |
| MT-PreamCNN-Attn (mixed) | CoaxSweep | 0.9969 | 100.0% | 0.84 dB |
| MT-PreamCNN-Attn (mixed) | AirLink held-out | 0.9820 | 74.61% | 1.54 dB |

> ⚠ CNN zeroshot 32.69% is BELOW the 50% random baseline — the CNN predicts
> QAM16 for all AirLink samples (matching the 32.1% QAM16 fraction exactly).
> This is **modulation collapse** from CFO sign reversal: CoaxSweep +330 Hz vs
> AirLink −3000 Hz inverts phase-roll features that the CNN encoded as
> modulation-discriminative.

### Multi-Task Model Results — With Phase-Rotation Augmentation

| Model | Mode | Baseline Mod | +Phase Aug Mod | Δ |
|---|---|---|---|---|
| MT-PreamCNN | Zero-shot | 32.69% | **62.93%** | +30.2 pp |
| MT-PreamCNN | Mixed | 71.96% | **89.62%** | +17.7 pp |
| MT-PreamCNN-Attn | Zero-shot | 48.94% | **61.00%** | +12.1 pp |
| MT-PreamCNN-Attn | Mixed | 74.61% | **92.27%** | +17.7 pp ← best |

**Phase-aug also improves SNR estimation:**

| Model | Mode | Baseline RMSE | +Phase Aug RMSE | Δ |
|---|---|---|---|---|
| MT-PreamCNN-Attn | Mixed | 1.54 dB | **1.37 dB** | −11% |
| MT-PreamCNN | Mixed | 1.70 dB | **1.44 dB** | −15% |

**CoaxSweep accuracy is maintained at 100% modulation and ≤1.03 dB SNR RMSE
for all phase-aug models.**

### Model Checkpoints

```
rf_stream/multitask_model_v2/
  mt_preamcnn_cnn_zeroshot.pt
  mt_preamcnn_cnn_zeroshot_phaseaug.pt      ← 62.93% AirLink mod
  mt_preamcnn_cnn_mixed.pt
  mt_preamcnn_cnn_mixed_phaseaug.pt         ← 89.62% AirLink mod
  mt_preamcnn_attn_zeroshot.pt
  mt_preamcnn_attn_zeroshot_phaseaug.pt     ← 61.00% AirLink mod
  mt_preamcnn_attn_mixed.pt
  mt_preamcnn_attn_mixed_phaseaug.pt        ← 92.27% AirLink mod (best)
  metrics_{model}_[phaseaug].json           ← full metric dicts per split
```

---

## 8. Key Lessons Learned

### Step 10 PHY

**MMSE over ZF for OTA:**  
ZF equalization divides by H[k], which can be very small at nulls (deep fades),
amplifying noise dramatically. MMSE regularizes with `noise_var = 10^(-SNR/10)`.
On cable (high SNR), ZF ≈ MMSE. On OTA (SNR 17–28 dB), MMSE gives measurably
better BER at deep-fade subcarriers.

**pilot_weight=0.50 is the optimal blend:**  
Weight=0.0 (LTF only) misses inter-symbol channel drift at OTA distances.
Weight=1.0 (full pilot correction) amplifies pilot noise on static segments.
Blend at 0.50 is empirically the sweet spot validated on both cable and 40 cm OTA.

**Frequency change: 2300 → 2400 MHz:**  
2400 MHz is inside the 2.4 GHz ISM band. This helps collect real WiFi-environment
interference captures (AirLink-III) without special TX permission. The −3000 Hz
CFO offset is a stable property of the Pluto TCXO at this frequency.

**auto_z_th is essential for QAM16 OTA:**  
QAM16 has ~3 dB higher PAPR than QPSK. The same energy threshold that easily
catches QPSK packets may miss QAM16 at longer distances. The 0.65× scale factor
(effective threshold = 5.2 instead of 8.0) corrects for this.

**Gain=25 dB is the weak point:**  
At the lowest gain step, the RX signal is near the ADC noise floor. Both QPSK
and QAM16 suffer the most at 25 dB (44.7% and 46.8% respectively). This is a
fundamental SNR limit, not a PHY bug.

### Multi-Task Learning

**Modulation Collapse — Root Cause:**  
Schmidl-Cox CFO correction removes frequency drift but leaves a random initial
phase offset θ₀ per capture. CoaxSweep (+330 Hz median CFO) and AirLink
(−3000 Hz median CFO) produce opposite IQ phase trajectories. CNN features
encoding phase roll are perfectly modulation-discriminative on CoaxSweep but
are anti-discriminative on AirLink — the CNN learns "phase roll direction =
modulation order" which is domain-specific.

**Phase-Rotation Augmentation Solves Modulation Collapse:**  
Randomly rotating each training sample by θ ~ U[0, 2π) forces the encoder to
discard phase-sensitive features and rely exclusively on amplitude structure
(single ring for QPSK, three rings for QAM16). Both architectures gain exactly
+17.7 pp in mixed mode — the benefit is architecture-independent.

**WiFi Fail NPZs are Hard-Negative Gate Examples:**  
The 4,794 WiFi-triggered fail NPZs from AirLink-III dramatically improved
mixed gate AUC from ≤0.89 to ≥0.98. These are packets where the energy
detector fired on WiFi interference that resembles the STF preamble structure.
Having real OTA interference as gate=0 training examples is far more effective
than synthetic noise.

**mod_label Fix — Critical Bug:**  
WiFi fail NPZs have `bps=2` (QPSK) from the TX configuration even though the
captured packet is noise/WiFi. If fail NPZs are included in mod evaluation,
they inflate mod_acc by contributing ~5,000 incorrectly-labeled QPSK samples.
Fix: gate=0 (fail) captures must get `mod_label=-1` (excluded from mod CE loss).

**Attention Advantage is SNR, Not Modulation:**  
The attention model's primary benefit over CNN is SNR estimation robustness.
Under mixed training, MT-PreamCNN-Attn achieves 0.84 dB coax SNR RMSE vs
1.09 dB for CNN — a 23% reduction — while CNN degrades as OTA data dilutes
coax-specific kernel responses. The modulation accuracy difference (mixed:
74.61% vs 71.96%) is smaller and disappears under phase-aug (both gain +17.7 pp).

**Zero-Shot Asymmetry (CNN −30 pp vs Attn −12 pp without aug):**  
Global MHSA attends to both phase and amplitude positions simultaneously,
so it cannot be as decisively inverted by a single CFO sign change as a
local-receptive-field CNN. With phase-aug, both architectures converge to
~61-63% zero-shot, suggesting the augmentation fully corrects the remaining
phase reliance in both.

---

## 9. Configuration Reference

### Step 10 Validated Configuration (OTA @ 40–50 cm)
```bash
--fc 2400e6          # 2.4 GHz ISM band (was 2300 MHz in step8/9)
--fs 3e6             # 3 MHz sample rate
--repeat 4           # 4× bit repetition
--ltf_symbols 8      # 8 LTF repetitions (better channel est vs 4)
--stf_repeats 6      # 6 STF repetitions (better sync vs 2)
--equalization mmse  # MMSE eq (step10 default)
--pilot_weight 0.5   # balanced blend (0.0=LTF-only, 1.0=full-pilot)
--auto_z_th          # scale energy threshold by modulation PAPR
--rx_gain 0          # TX gain=0; use low RX gain for OTA
```

### Train_multitask_v2.py — Key Hyperparameters
| Arg | Default | Description |
|---|---|---|
| `--arch` | `attn` | Model architecture: `cnn` or `attn` |
| `--mode` | `coax` | Training mode: `zeroshot`, `mixed`, `coax` |
| `--phase_aug` | off | Enable random phase-rotation augmentation |
| `--epochs` | 80 | Training epochs |
| `--batch_size` | 64 | Mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--embed_dim` | 256 | CNN/Attn encoder feature dimension |
| `--w_gate` | 1.0 | Gate loss weight |
| `--w_mod` | 0.5 | Modulation loss weight |
| `--w_snr` | 0.1 | SNR loss weight |

### PHY Constants (must match TX and RX exactly)
```python
N_FFT      = 64
N_CP       = 16
SYMBOL_LEN = 80        # N_FFT + N_CP
DATA_SUBCARRIERS = 48  # ±[1..26] excluding pilots
PILOT_SUBCARRIERS = [-21, -7, 7, 21]
PILOT_VALS = [1, 1, 1, -1]
pilot_sign = +1 if sym_idx%2==0 else -1   # alternating per symbol
MAGIC      = b"AIS1"
```

---

## 10. Paper References

**Paper:** `rf_stream/paper/globecom2026_multitask_phy.tex`  
**Figures:** `rf_stream/paper_figures/fig*.pdf`

| Figure | Content | Script |
|---|---|---|
| fig_mt_arch.pdf | MT-PreamCNN-Attn architecture diagram | manual |
| fig12_mt_roc_ablation.pdf | Gate ROC curves per model | generate_multitask_figures.py |
| fig13_mt_loss_curves.pdf | Training loss curves | generate_multitask_figures.py |
| fig14_mt_snr_scatter.pdf | SNR prediction scatter | generate_multitask_figures.py |
| fig15_mt_summary_bar.pdf | Per-task bar chart, all models | generate_multitask_figures.py |

```bash
# Regenerate all multitask figures
python3 rf_stream/generate_multitask_figures.py \
  --out_dir rf_stream/paper_figures

# Build paper PDF (run twice for references)
cd rf_stream/paper
pdflatex globecom2026_multitask_phy.tex
pdflatex globecom2026_multitask_phy.tex
```
