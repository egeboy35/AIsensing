# Pilot-geometry experiment (OFDM path of `sdradi/sdr_video_commv2.py`)

**Question.** The root `README.md` (lines 223-226) reports ~50% BER on dual-Pluto
TX/RX "even when sync appears healthy" and attributes it to a host/driver
transport limitation. Static analysis suggested a DSP defect instead: the OFDM
receiver interpolates its channel estimate and fits its timing estimator over a
pilot grid that is **non-monotonic in true frequency**. This experiment tests
that hypothesis against the *actual pipeline code*, in simulation, with
controlled channels.

Everything here is additive: no file outside `experiments/pilot_geometry/` was
modified.

## 1. Verified code anatomy (exact references)

All references are to `sdradi/sdr_video_commv2.py` at the commit this branch
forked from.

| What | Where | Verified finding |
|---|---|---|
| Declared pilot plan | line 68 | `pilot_carriers=(-21, -7, 7, 21)` in `OFDMConfig` — **ignored** by the implementation (`pilot_values`, line 69, is ignored too; pilots are seed-42 BPSK, lines 916-918) |
| Actual pilot placement | lines 920-954 (`_setup_carriers`), selection at 947-950 | pilots at raw FFT bins `[1, 14, 38, 51]` = signed frequencies `[+1, +14, -26, -13]` — non-monotonic in frequency; confirmed by executing the class |
| Per-symbol channel estimate | lines 1040-1059 | LS at pilots (1046), then `np.interp` of magnitude (1051) and unwrapped phase (1054-1056) over `np.arange(64)` with the raw-index pilot grid as abscissa |
| Equalizer | lines 1074-1078 | ZF using the interpolated estimate |
| Timing estimator | lines 1115-1149 (`estimate_delay`), polyfit at 1145 | pilot phase slope fitted against the same raw-index grid; delay formula at 1146 assumes phase linear in that coordinate |
| Fine-time sync loop | lines 2134-2181 | 3 iterations (2136), converged if \|est\|<0.1 (2144), abort if \|est\|>20 (2148), integer `np.round` shift applied cumulatively (2154-2158) |
| README transport claim | root `README.md` lines 223-226 | as quoted above |

Why the geometry matters: `np.interp`/`np.polyfit` are perfectly well-defined on
`[1, 14, 38, 51]` (it *is* increasing as raw indices), but a physical channel is
smooth in **signed frequency**. For any *fractional* timing residual τ the
per-carrier phase is `-2π f τ/64` with signed `f`; expressed on the raw-index
axis this function has a branch offset of `2πτ` between the positive-frequency
bins (1-26) and negative-frequency bins (38-63). Interpolating across the
guard band between raw bins 14 and 38 therefore blends values from **opposite
band edges** (f=+14 and f=-26), and the phase-slope fit acquires the wrong sign
and magnitude. For *integer* circular shifts the raw axis happens to be a valid
coordinate (`e^{-j2πkd/64}` is raw-linear), which is why the bug hides in
clean integer-aligned tests.

## 2. Method: same code, two geometries

- **Stock** variant: `OFDMTransceiver` exactly as shipped.
- **Corrected** variant (`CorrectedGeometryRx` in `run_experiments.py`):
  a subclass that pre-multiplies the received stream by `(-1)^n` and re-expresses
  the pilot/data index tables in fftshifted coordinates
  `k' = (k_raw + 32) mod 64 = f_signed + 32` (pilot abscissa `[6, 19, 33, 46]`,
  monotone in frequency). Because `FFT{(-1)^n x[n]}[k] = X[(k+32) mod 64]`, the
  **unmodified** parent `demodulate`/`estimate_delay` bodies then execute their
  own `np.interp`, `np.unwrap`, `np.polyfit` and ZF lines on a frequency-ordered
  grid. Same 4 physical pilots, same pilot symbols (reordered to keep the tx
  pairing), same interpolator, same code path. Every FFT window starts at an
  even stream offset (symbol length 80, CP 16), so one global `(-1)^n` is exact.
- Transmit side is always the stock modulator; only the receiver differs.
- Channels: fractional delay applied as a signed-frequency phase ramp over the
  full stream (the physically correct model of a bandlimited timing residual);
  multipath as integer-tap FIR with all delays < CP; AWGN at 25 dB unless noted.
- Deterministic (`--seed`, default 12345); ≥1120 OFDM symbols (80 frames) per
  measurement point. Full run ≈ 4.7 min (numpy 2.5.2, Python 3.14).

Reproduce with:

```
python experiments/pilot_geometry/run_experiments.py            # full run
python experiments/pilot_geometry/run_experiments.py --frames 20  # quick
```

## 3. Results

### E1 — Geometry audit (no channel)

Stock: 12/48 data carriers (25%) are interpolated across the guard band
(blending f=+14 with f=-26), and another 12/48 (25%) sit in the
extrapolation-clamp zone beyond raw bin 51. Corrected grid: 0% guard-crossing;
the same 25% clamp zone remains (f=+15..+26 clamp to f=+14) because the four
*shipped* pilot positions simply do not bracket the band — the declared
`(-21, -7, 7, 21)` plan would. (`results/e1_geometry.csv`)

### E2 — BER vs residual fractional timing offset, 25 dB SNR

`results/e2_ber_vs_timing_{qpsk,qam16}.csv`, `results/e2_ber_vs_timing.png`.
Selected points (BER):

| τ (samples) | QPSK stock | QPSK corrected | QPSK stock+loop | QPSK corr+loop | 16QAM stock | 16QAM corrected |
|---|---|---|---|---|---|---|
| 0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.3 | 0.0238 | 0.0000 | 0.0238 | 0.0000 | 0.0639 | 0.0079 |
| 0.5 | **0.0668** | **0.0000** | 0.0668 | 0.0000 | **0.1125** | **0.0311** |
| 1.0 | 0.0471 | 0.0468 | 0.0000 | 0.0000 | 0.0658 | 0.0661 |
| 1.5 | 0.1419 | 0.0741 | 0.0668 | 0.0003 | 0.1680 | 0.0861 |
| 2.0 | **0.2289** | 0.0936 | **0.2289** | **0.0000** | 0.2407 | 0.1046 |

- The stock receiver has a **noise-independent BER floor for any non-integer
  residual** (6.7% at τ=0.5 QPSK; 23% at τ=2.0), caused purely by the
  guard-band-crossing interpolation. The corrected geometry is error-free up to
  \|τ\|≈0.55 and, **combined with the sync loop, essentially error-free
  (QPSK ≤ 3.1e-4) over the whole 0–2.0 sweep**, while stock+loop stays at its
  floor wherever its own estimator rounds to zero (τ=0.3-0.7, 1.3-1.7, and —
  notably — an uncorrected 2-sample offset at τ=2.0). For 16QAM the
  corrected+loop combination retains up to ~3.2% BER at half-sample residuals
  (τ=0.5/1.5): the loop removes only the integer part, and the remaining
  0.5-sample fraction costs margin in the clamp zone discussed in E1.
- Without the loop, corrected is not magic: for \|τ\|>0.8 its clamp zone
  (a consequence of the shipped pilot positions, see E1) also produces errors.
  The loop removes the integer part, which is exactly why a *sane estimator*
  matters (E4).

### E3 — BER vs multipath severity, 25 dB SNR

`results/e3_ber_vs_multipath.csv`, `.png`. Two regimes, honestly different:

- **Integer-tap-only channels (frac=0.0):** the raw-index axis is then a
  mathematically valid coordinate, and the two geometries merely have their
  interpolation/clamp weaknesses in different places relative to the channel
  ripple. Corrected wins most rows but **stock strictly wins 4 of 16, with 3
  exact ties** (e.g. d=4 2-tap: 0.030 vs 0.077). The hypothesis's phrase
  "under any frequency-selective residual" is **too strong** for
  exactly-integer, CP-circular channels.
- **Same channels plus a 0.5-sample fractional residual (frac=0.5) — the
  realistic hardware condition:** corrected wins **16 of 16** rows, typically
  by 2-5x (e.g. d=10 2-tap: stock 0.130 vs corr 0.027; d=1 3-tap: 0.071 vs
  0.001).

  *Label note: the multipath taps are not power-normalized (total power up to
  1.25), so the effective SNR at the labelled 25 dB points is ~26 dB. Both
  variants see the identical stream, so all stock-vs-corrected comparisons are
  unaffected; only the absolute SNR label is ~1 dB optimistic.*

### E4 — `estimate_delay` audit and sync-loop convergence

`results/e4_delay_estimator.csv`, `e4_syncloop.csv`, `e4_delay_estimator.png`.

- Feeding known delays through the **actual** `estimate_delay`: for fractional
  τ the stock estimator returns ≈ **-0.52·τ — wrong sign and half magnitude**
  (true +0.25 → -0.129; true +0.50 → -0.261). The corrected-grid version tracks
  the ideal line (+0.242, +0.486); its `np.unwrap` aliasing limit on the
  13/14/13-spaced grid is **asymmetric**: it stays on the ideal line up to
  +2.25 but already breaks at -2.25 (est -0.42), i.e. the negative-side limit
  is ~-2.1. The stock grid aliases already at \|τ\| ≈ 1.3 because of its
  24-bin gap.
- Loop convergence (policy of lines 2136-2174 around the actual estimator,
  25 dB): stock converges only for initial offsets within ≈ ±1 sample and even
  then leaves the 6.7% floor at ±0.5; at ±2 it leaves the offset entirely
  uncorrected (BER 0.23). Corrected converges to \|resid\| ≤ 0.5 with BER 0.000
  for all initial offsets within ±2, and also at +2.5; at -2.5 it exceeds its
  asymmetric unwrap limit and mis-shifts (residual -4.5, BER 0.56).
  **Beyond ≈ ±3 both estimators alias and
  the loop mis-shifts; the resulting misalignment lands at 55-58% BER — i.e.
  the exact "~50% BER while sync looks healthy" symptom of the README.**
- The hypothesis's specific claim that the meaningless slope drives the loop
  "by up to ±20 samples" is **refuted**: the broken estimator *shrinks*
  estimates (slope -0.52), so the loop under-corrects rather than running away;
  worst observed mis-shift was 2 samples in 3 iterations. The ±20 guard at
  line 2148 was never hit.

### E5 — Honest null checks

`results/e5_null_checks.csv`. With perfect timing and a flat channel the stock
geometry is **fine**: AWGN sweep 5-25 dB gives statistically identical BER for
both variants (e.g. 10 dB: 0.00209 vs 0.00212; ≥15 dB: both 0), and a pure
common-phase rotation of 20-40° is absorbed by both (BER 0). The defect is
invisible in exactly the clean-loopback conditions a quick bench test would
use.

## 4. Conclusions

**Does the evidence support "pilot geometry is a plausible root cause of the
reported ~50% BER"? — Partially, but substantially: it is a demonstrated,
severe receiver defect and a plausible major contributor; it is not proven to
be *the* root cause of the hardware number.**

Supported (demonstrated on the actual pipeline code, deterministic, ≥1120
symbols/point):

1. The implementation ignores its own configured pilot plan and derives a
   pilot grid that is non-monotonic in true frequency (E1; hypothesis (a)
   confirmed, lines 947-950).
2. For any non-integer timing residual — unavoidable on real hardware with
   independent TX/RX clocks — the stock channel interpolation corrupts ~25% of
   data carriers and produces a noise-independent BER floor of ~7% (QPSK) to
   ~11% (16QAM) at τ=0.5, and up to ~24% within ±2 samples (E2; hypothesis (b)
   confirmed for fractional/timing effects, lines 1051-1059).
3. The stock `estimate_delay` is genuinely broken for the case it exists for:
   fractional residuals give estimates of wrong sign and half magnitude, and
   its unwrap aliases at \|τ\|>1.3, so the fine-sync loop cannot correct
   offsets ≥2 samples and leaves 6-23% BER floors uncorrected; misalignments
   of ≥3 samples produce the ~50% BER signature (E4; hypothesis (c) confirmed
   in mechanism, lines 1145-1146 + 2136-2174).
4. The same code run on the same four pilots in frequency-ordered coordinates
   removes essentially all of this: ≤3.1e-4 BER (QPSK) across the entire
   0-2-sample sweep with the sync loop (16QAM retains up to ~3.2% at
   half-sample residuals via the E1 clamp zone), and 2-5x lower BER under
   multipath+residual.

Corrections to the hypothesis (refuted details — stating them is part of the
point):

- "under **any** frequency-selective ... residual" is too strong: for
  exactly-integer-tap, CP-circular channels with no fractional residual the raw
  grid is a valid coordinate and stock occasionally beats corrected (E3).
- The "drives the loop by up to ±20 samples" mechanism is wrong: the estimator
  *attenuates* (slope ≈ -0.52), so the failure mode is under-correction and
  stuck offsets, not runaway (E4). The ~50% BER state is reached via aliasing
  for offsets ≥ ~3 samples, or accumulation of the uncorrected floors, not via
  a ±20-sample excursion.

What this does NOT prove (no hardware in this experiment):

- That pilot geometry *is* the cause of the specific dual-Pluto observation.
  Real links add CFO, sample-clock drift, filter delays, IIO buffer behavior,
  and the coarse correlation sync — none exercised here. A genuine host/driver
  transport fault (the README's explanation) could coexist; the two are not
  mutually exclusive.
- That fixing the geometry alone recovers the video link: the shipped pilot
  *positions* (not bracketing the band, E1's 25% clamp zone) still cost margin
  for 16QAM at ±0.5-sample residuals even with corrected interpolation; using
  the declared `(-21, -7, 7, 21)` plan would address that but was out of scope
  ("same 4 pilots").
- Hardware validation is the required next step: on a Pluto pair, log
  `estimate_delay` outputs and per-carrier EVM vs signed frequency; the
  defect's fingerprint would be an EVM hump over data carriers f=+15..+26
  (raw 15-26) and estimator readings anti-correlated with applied timing steps.

## 5. Incidental findings (while reading the pipeline)

- `OFDMConfig` is decorated `@dataclass` twice (lines 61-62); harmless.
- `pilot_values` (line 69) is ignored, like `pilot_carriers`: pilots are
  regenerated as seed-42 BPSK at lines 916-918.
- The demodulator's SNR estimate (lines 1095-1102) hard-slices with `np.sign`,
  which is only valid for QPSK; for 16QAM the reported `snr_est_db` metric is
  wrong (bits are unaffected).
- `demodulate` processes every 80-sample chunk in the buffer, so trailing
  garbage after the payload is demodulated into appended garbage bits; callers
  must truncate by expected length (metrics like `snr_est` are diluted).

## Files

- `run_experiments.py` — single entry point (argparse; seeds fixed; ~5 min).
- `results/e1_geometry.csv`
- `results/e2_ber_vs_timing_qpsk.csv`, `results/e2_ber_vs_timing_qam16.csv`,
  `results/e2_ber_vs_timing.png`
- `results/e3_ber_vs_multipath.csv`, `results/e3_ber_vs_multipath.png`
- `results/e4_delay_estimator.csv`, `results/e4_syncloop.csv`,
  `results/e4_delay_estimator.png`
- `results/e5_null_checks.csv`
