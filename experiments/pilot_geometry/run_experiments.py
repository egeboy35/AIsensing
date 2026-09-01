#!/usr/bin/env python
"""
Pilot-geometry experiment for sdradi/sdr_video_commv2.py (OFDM path).

HYPOTHESIS UNDER TEST
---------------------
The OFDM receiver places its 4 pilots at raw FFT bins [1, 14, 38, 51]
(sdr_video_commv2.py, OFDMTransceiver._setup_carriers, lines 920-954).
In true (signed) baseband frequency these bins are [+1, +14, -26, -13] --
NON-monotonic in frequency.  The per-symbol channel estimator
(demodulate, lines 1048-1059) linearly interpolates pilot magnitude and
unwrapped phase over the RAW index axis, and estimate_delay
(lines 1115-1149) polyfits pilot phase against the same raw axis.
Both are therefore computed in a coordinate frame in which a physical
(fractional-delay / frequency-selective) channel is NOT smooth, which is
suspected to be a root cause of the ~50% BER on hardware that the root
README (lines 223-226) attributes to a host/driver transport limitation.

METHOD
------
All experiments run the ACTUAL pipeline code (OFDMTransceiver.modulate /
demodulate / estimate_delay imported from sdradi/sdr_video_commv2.py).
The "corrected" variant does NOT reimplement anything: it is a subclass
that pre-multiplies the received stream by (-1)^n before calling the
unmodified parent methods.  Because FFT{(-1)^n x[n]}[k] = X[(k+N/2) mod N]
(an fftshift realized in the time domain), the parent's np.fft.fft then
produces the fftshift-ordered spectrum, and by giving the subclass
frequency-sorted pilot/data index tables the SAME interpolation, unwrap,
polyfit and equalizer lines execute on a grid that is monotonic in true
frequency.  Same 4 physical pilots, same interpolator, same code --
only the geometry differs.  (Every FFT window in the stream starts at an
even offset -- symbol length 80, CP 16 -- so one global (-1)^n works.)

Transmit side is always the stock modulator; the two variants only differ
at the receiver, which is where the hypothesis lives.

EXPERIMENTS
-----------
E1  Geometry audit (pure analysis, no channel).
E2  BER vs residual fractional timing offset (signed-frequency phase
    ramp), stock vs corrected, QPSK + 16QAM, 25 dB SNR; both without and
    with the fine-time-sync loop policy of lines 2136-2174.
E3  BER vs multipath delay spread (2-tap and 3-tap, CP-contained),
    stock vs corrected, with and without an extra 0.5-sample residual.
E4  estimate_delay audit: estimated vs true delay through the ACTUAL
    estimate_delay (noiseless), plus sync-loop convergence audit.
E5  Honest null checks: ideal channel (AWGN-only SNR sweep) and pure
    common-phase-error rotation -- cases where stock should be fine.

Determinism: master seed fixed via --seed; every experiment derives its
own child rng.  Runtime target < 10 min (measured ~1-3 min).
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Import the ACTUAL pipeline (simulation mode: adi/cv2/torch warnings OK)
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "sdradi"))

import sdr_video_commv2 as pipeline  # noqa: E402

OFDMConfig = pipeline.OFDMConfig
OFDMTransceiver = pipeline.OFDMTransceiver


# ----------------------------------------------------------------------
# Corrected-geometry receiver (same code, frequency-monotonic coordinates)
# ----------------------------------------------------------------------
class CorrectedGeometryRx(OFDMTransceiver):
    """OFDMTransceiver whose rx-side interpolation runs on a grid that is
    monotonic in true (signed) frequency.

    No pipeline method body is copied or edited.  demodulate() and
    estimate_delay() are the unmodified parent implementations; they are
    fed the stream multiplied by (-1)^n, which fftshifts every FFT the
    parent takes, and the pilot/data index tables are re-expressed in
    fftshifted coordinates k' = (k_raw + N/2) mod N = f_signed + N/2.
    Physical pilots, pilot symbols, interpolator, unwrap, polyfit and
    equalizer are all identical to stock.
    """

    def __init__(self, config=None):
        super().__init__(config)
        N = self.config.fft_size
        raw_pilots = self.pilot_indices.copy()          # [1, 14, 38, 51]
        shifted = (raw_pilots + N // 2) % N             # [33, 46, 6, 19]
        order = np.argsort(shifted)                     # frequency order
        self.pilot_indices = shifted[order]             # [6, 19, 33, 46]
        self.pilot_symbols = self.pilot_symbols[order]  # keep tx pairing
        # keep the ORIGINAL gather order so recovered data symbols come
        # out in the same order the stock transmitter placed them
        self.data_indices = (self.data_indices + N // 2) % N

    @staticmethod
    def _fftshift_in_time(signal):
        return signal * ((-1.0) ** np.arange(len(signal)))

    def demodulate(self, signal):
        return super().demodulate(self._fftshift_in_time(signal))

    def estimate_delay(self, signal):
        return super().estimate_delay(self._fftshift_in_time(signal))


# ----------------------------------------------------------------------
# Channel / helper functions
# ----------------------------------------------------------------------
def signed_freq(k, n=64):
    k = np.asarray(k)
    return np.where(k < n // 2, k, k - n)


def fractional_delay(signal, tau):
    """Ideal bandlimited delay by tau samples: phase ramp over the SIGNED
    frequencies of the full stream (the physically correct model of a
    residual timing offset of a bandlimited baseband waveform)."""
    n = len(signal)
    f = np.fft.fftfreq(n) * n
    return np.fft.ifft(np.fft.fft(signal) * np.exp(-2j * np.pi * f * tau / n))


def awgn(signal, snr_db, rng, ref_power):
    npow = ref_power / (10.0 ** (snr_db / 10.0))
    noise = (rng.standard_normal(len(signal)) +
             1j * rng.standard_normal(len(signal))) * np.sqrt(npow / 2.0)
    return signal + noise


def multipath(signal, taps):
    """Linear convolution with an integer-spaced FIR, trimmed to length.
    All tap delays < CP (16), so per-symbol the channel is circular."""
    h = np.zeros(max(d for d, _ in taps) + 1, dtype=complex)
    for d, a in taps:
        h[d] += a
    return np.convolve(signal, h)[: len(signal)]


def ber(tx_bits, rx_bits):
    n = len(tx_bits)
    return float(np.mean(np.asarray(rx_bits[:n]) != np.asarray(tx_bits)))


def make_payload(trx, num_frames, rng):
    bits = rng.integers(0, 2, num_frames * trx.config.bits_per_frame)
    tx = trx.modulate(bits)
    return bits, tx


def sync_loop(trx, stream, payload_start):
    """Faithful transcription of the fine-time-sync policy of
    sdr_video_commv2.py lines 2136-2174 (3 iterations, |est|<0.1
    converged, |est|>20 abort, integer np.round shift, re-slice),
    driving the ACTUAL trx.estimate_delay. Returns final payload_start."""
    ps = payload_start
    for _ in range(3):
        est = trx.estimate_delay(stream[ps:])
        if abs(est) < 0.1:
            break
        if abs(est) > 20.0:
            break
        int_shift = int(np.round(est))
        if int_shift == 0:
            break
        ps += int_shift
        if ps < 0:
            ps = 0
        if ps >= len(stream):
            break
    return ps


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------
# E1 -- geometry audit
# ----------------------------------------------------------------------
def experiment_e1(outdir):
    print("\n=== E1: pilot geometry audit (pure analysis) ===")
    stock = OFDMTransceiver(OFDMConfig())
    corr = CorrectedGeometryRx(OFDMConfig())
    N = stock.config.fft_size

    raw = stock.pilot_indices
    print(f"stock pilot bins (raw FFT index): {list(raw)}")
    print(f"  true signed frequencies       : {list(signed_freq(raw, N))}")
    print(f"config pilot_carriers (line 68) : {stock.config.pilot_carriers}"
          f"  <- declared but IGNORED by _setup_carriers")
    print(f"corrected interp grid (fftshift coords): {list(corr.pilot_indices)}"
          f" -> signed f {list(corr.pilot_indices - N // 2)}")

    rows = []
    for name, trx, to_f in (
        ("stock", stock, lambda k: signed_freq(k, N)),
        ("corrected", corr, lambda k: np.asarray(k) - N // 2),
    ):
        xp = trx.pilot_indices  # interp abscissa actually used by the code
        guard_raw = set(range(27, 38))  # unused bins around Nyquist (+ bin 32)
        n_cross = n_clamp = 0
        for d in trx.data_indices:
            if d < xp[0] or d > xp[-1]:
                n_clamp += 1
                seg = "clamp"
            else:
                j = np.searchsorted(xp, d) - 1
                lo, hi = xp[j], xp[j + 1]
                # does this interpolation segment span the guard band /
                # Nyquist wrap in PHYSICAL frequency terms?
                if name == "stock":
                    crosses = any(g in guard_raw for g in range(lo + 1, hi))
                else:
                    # corrected axis is fftshifted; guard is k' in [0,5]+[59,63]
                    crosses = False  # by construction: grid is freq-sorted
                if crosses:
                    n_cross += 1
                    seg = "cross-guard"
                else:
                    seg = "ok"
            rows.append([name, int(d), int(signed_freq(d, N)) if name == "stock"
                         else int(d - N // 2), seg])
        nd = len(trx.data_indices)
        print(f"{name:9s}: data carriers whose interp segment crosses the "
              f"guard band: {n_cross}/{nd} ({100*n_cross/nd:.0f}%), "
              f"in extrapolation-clamp zone: {n_clamp}/{nd} "
              f"({100*n_clamp/nd:.0f}%)")
    write_csv(os.path.join(outdir, "e1_geometry.csv"),
              ["variant", "data_carrier_code_index", "signed_freq", "segment"],
              rows)


# ----------------------------------------------------------------------
# E2 -- BER vs residual fractional timing offset
# ----------------------------------------------------------------------
def experiment_e2(outdir, seed, num_frames, snr_db=25.0):
    print(f"\n=== E2: BER vs residual timing offset "
          f"(SNR {snr_db:.0f} dB, {num_frames*14} OFDM symbols/point) ===")
    taus = np.round(np.arange(0.0, 2.01, 0.1), 3)
    prepad = 200  # room for negative loop shifts; mimics mid-capture payload
    results = {}
    for mod_order, mod_name in ((4, "qpsk"), (16, "qam16")):
        cfg_kw = dict(mod_order=mod_order)
        stock = OFDMTransceiver(OFDMConfig(**cfg_kw))
        corr = CorrectedGeometryRx(OFDMConfig(**cfg_kw))
        rng = np.random.default_rng(seed + mod_order)
        bits, tx = make_payload(stock, num_frames, rng)
        ref_pow = float(np.mean(np.abs(tx) ** 2))
        rows = []
        for tau in taus:
            stream = np.concatenate([np.zeros(prepad, complex),
                                     tx,
                                     np.zeros(160, complex)])
            stream = fractional_delay(stream, tau)
            stream = awgn(stream, snr_db, np.random.default_rng(
                seed + mod_order + int(tau * 1000)), ref_pow)
            row = [tau]
            for trx in (stock, corr):
                rx_bits, _ = trx.demodulate(stream[prepad:])
                row.append(ber(bits, rx_bits))
            for trx in (stock, corr):
                ps = sync_loop(trx, stream, prepad)
                rx_bits, _ = trx.demodulate(stream[ps:])
                row.append(ber(bits, rx_bits))
            rows.append(row)
            print(f"  {mod_name} tau={tau:4.2f}  "
                  f"noloop stock={row[1]:.4f} corr={row[2]:.4f}   "
                  f"withloop stock={row[3]:.4f} corr={row[4]:.4f}")
        results[mod_name] = np.array(rows)
        write_csv(os.path.join(outdir, f"e2_ber_vs_timing_{mod_name}.csv"),
                  ["tau_samples", "ber_stock_noloop", "ber_corrected_noloop",
                   "ber_stock_withloop", "ber_corrected_withloop"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (mod_name, arr) in zip(axes, results.items()):
        ax.plot(arr[:, 0], arr[:, 1], "o-", color="#c0392b",
                label="stock geometry")
        ax.plot(arr[:, 0], arr[:, 2], "s-", color="#27ae60",
                label="corrected geometry")
        ax.plot(arr[:, 0], arr[:, 3], "o--", color="#e67e22", alpha=0.7,
                label="stock + sync loop")
        ax.plot(arr[:, 0], arr[:, 4], "s--", color="#2980b9", alpha=0.7,
                label="corrected + sync loop")
        ax.set_title(f"E2 {mod_name.upper()} @ {snr_db:.0f} dB SNR")
        ax.set_xlabel("residual timing offset (samples)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("BER")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "e2_ber_vs_timing.png"), dpi=140)
    plt.close(fig)
    return results


# ----------------------------------------------------------------------
# E3 -- BER vs frequency-selective (multipath) severity
# ----------------------------------------------------------------------
def experiment_e3(outdir, seed, num_frames, snr_db=25.0):
    print(f"\n=== E3: BER vs multipath delay spread "
          f"(SNR {snr_db:.0f} dB, {num_frames*14} OFDM symbols/point) ===")
    stock = OFDMTransceiver(OFDMConfig())
    corr = CorrectedGeometryRx(OFDMConfig())
    rng = np.random.default_rng(seed + 300)
    bits, tx = make_payload(stock, num_frames, rng)
    ref_pow = float(np.mean(np.abs(tx) ** 2))
    delays = [1, 2, 3, 4, 6, 8, 10, 12]
    rows = []
    for d in delays:
        for ntaps in (2, 3):
            if ntaps == 2:
                taps = [(0, 1.0), (d, 0.5 * np.exp(1j * np.pi / 4))]
            else:
                taps = [(0, 1.0),
                        (max(1, d // 2), 0.4 * np.exp(1j * 2.1)),
                        (d, 0.3 * np.exp(-1j * 0.7))]
            for frac in (0.0, 0.5):
                stream = multipath(tx, taps)
                if frac:
                    stream = fractional_delay(stream, frac)
                stream = awgn(stream, snr_db, np.random.default_rng(
                    seed + 300 + d * 100 + ntaps * 10 + int(frac * 2)),
                    ref_pow)
                b_stock = ber(bits, stock.demodulate(stream)[0])
                b_corr = ber(bits, corr.demodulate(stream)[0])
                rows.append([d, ntaps, frac, b_stock, b_corr])
                print(f"  d={d:2d} taps={ntaps} frac={frac:.1f}  "
                      f"stock={b_stock:.4f}  corr={b_corr:.4f}")
    write_csv(os.path.join(outdir, "e3_ber_vs_multipath.csv"),
              ["max_tap_delay", "num_taps", "extra_frac_offset",
               "ber_stock", "ber_corrected"], rows)

    arr = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, frac in zip(axes, (0.0, 0.5)):
        for ntaps, mk in ((2, "o"), (3, "^")):
            m = (arr[:, 1] == ntaps) & (arr[:, 2] == frac)
            ax.plot(arr[m, 0], arr[m, 3], mk + "-", color="#c0392b",
                    label=f"stock {ntaps}-tap")
            ax.plot(arr[m, 0], arr[m, 4], mk + "-", color="#27ae60",
                    label=f"corrected {ntaps}-tap")
        ax.set_title(f"E3 multipath, extra fractional offset = {frac}")
        ax.set_xlabel("max tap delay (samples)")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("BER")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "e3_ber_vs_multipath.png"), dpi=140)
    plt.close(fig)
    return arr


# ----------------------------------------------------------------------
# E4 -- estimate_delay audit + sync-loop convergence
# ----------------------------------------------------------------------
def experiment_e4(outdir, seed, num_frames, snr_db=25.0):
    print("\n=== E4: estimate_delay audit (actual pipeline estimator) ===")
    stock = OFDMTransceiver(OFDMConfig())
    corr = CorrectedGeometryRx(OFDMConfig())
    rng = np.random.default_rng(seed + 400)
    bits, tx = make_payload(stock, max(num_frames, 2), rng)
    ref_pow = float(np.mean(np.abs(tx) ** 2))

    taus = np.round(np.arange(-3.0, 3.001, 0.125), 4)
    rows = []
    for tau in taus:
        stream = fractional_delay(tx, tau)  # noiseless estimator audit
        rows.append([tau, stock.estimate_delay(stream),
                     corr.estimate_delay(stream)])
    write_csv(os.path.join(outdir, "e4_delay_estimator.csv"),
              ["true_delay", "est_stock", "est_corrected"], rows)
    arr = np.array(rows)
    i25 = np.argmin(np.abs(taus - 0.25))
    i50 = np.argmin(np.abs(taus - 0.5))
    print(f"  true +0.25 -> stock {arr[i25,1]:+.3f}, corr {arr[i25,2]:+.3f}")
    print(f"  true +0.50 -> stock {arr[i50,1]:+.3f}, corr {arr[i50,2]:+.3f}")

    # sync-loop convergence audit (policy of lines 2136-2174, with noise)
    print("  sync-loop convergence (initial offset -> |final residual|, BER):")
    prepad = 200
    init = np.round(np.arange(-4.0, 4.001, 0.5), 3)
    lrows = []
    for tau in init:
        stream = np.concatenate([np.zeros(prepad, complex), tx,
                                 np.zeros(160, complex)])
        stream = fractional_delay(stream, tau)
        stream = awgn(stream, snr_db, np.random.default_rng(
            seed + 400 + int(tau * 100)), ref_pow)
        row = [tau]
        for trx in (stock, corr):
            ps = sync_loop(trx, stream, prepad)
            resid = tau - (ps - prepad)
            b = ber(bits, trx.demodulate(stream[ps:])[0])
            row += [resid, b]
        lrows.append(row)
        print(f"    init={row[0]:+5.2f}  stock resid={row[1]:+5.2f} "
              f"ber={row[2]:.4f}   corr resid={row[3]:+5.2f} ber={row[4]:.4f}")
    write_csv(os.path.join(outdir, "e4_syncloop.csv"),
              ["initial_offset", "residual_stock", "ber_stock",
               "residual_corrected", "ber_corrected"], lrows)

    larr = np.array(lrows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(arr[:, 0], arr[:, 0], "k:", alpha=0.6, label="ideal (est = true)")
    ax.plot(arr[:, 0], arr[:, 1], "o-", color="#c0392b", ms=3,
            label="stock estimate_delay")
    ax.plot(arr[:, 0], arr[:, 2], "s-", color="#27ae60", ms=3,
            label="corrected geometry")
    ax.set_xlabel("true delay (samples)")
    ax.set_ylabel("estimated delay (samples)")
    ax.set_title("E4 estimate_delay: estimated vs true (noiseless)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.axhline(0.5, color="k", ls=":", alpha=0.5)
    ax.plot(larr[:, 0], np.abs(larr[:, 1]), "o-", color="#c0392b",
            label="stock |final residual|")
    ax.plot(larr[:, 0], np.abs(larr[:, 3]), "s-", color="#27ae60",
            label="corrected |final residual|")
    ax.set_xlabel("initial timing offset (samples)")
    ax.set_ylabel("|residual| after sync loop (samples)")
    ax.set_title(f"E4 fine-sync loop convergence @ {snr_db:.0f} dB")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "e4_delay_estimator.png"), dpi=140)
    plt.close(fig)
    return arr, larr


# ----------------------------------------------------------------------
# E5 -- honest null checks
# ----------------------------------------------------------------------
def experiment_e5(outdir, seed, num_frames):
    print(f"\n=== E5: null checks (ideal timing/channel; "
          f"{num_frames*14} OFDM symbols/point) ===")
    stock = OFDMTransceiver(OFDMConfig())
    corr = CorrectedGeometryRx(OFDMConfig())
    rng = np.random.default_rng(seed + 500)
    bits, tx = make_payload(stock, num_frames, rng)
    ref_pow = float(np.mean(np.abs(tx) ** 2))
    rows = []
    for snr in (5, 8, 10, 12, 15, 20, 25):
        stream = awgn(tx, snr, np.random.default_rng(seed + 500 + snr),
                      ref_pow)
        b_stock = ber(bits, stock.demodulate(stream)[0])
        b_corr = ber(bits, corr.demodulate(stream)[0])
        rows.append(["awgn", snr, b_stock, b_corr])
        print(f"  AWGN {snr:2d} dB: stock={b_stock:.5f}  corr={b_corr:.5f}")
    # pure common phase error (all carriers rotated equally): the stock
    # flat interpolation handles this fine -- it must NOT be blamed.
    for cpe_deg in (20.0, 40.0):
        stream = awgn(tx * np.exp(1j * np.deg2rad(cpe_deg)), 25,
                      np.random.default_rng(seed + 555), ref_pow)
        b_stock = ber(bits, stock.demodulate(stream)[0])
        b_corr = ber(bits, corr.demodulate(stream)[0])
        rows.append([f"cpe{cpe_deg:.0f}deg@25dB", 25, b_stock, b_corr])
        print(f"  CPE {cpe_deg:.0f} deg @25 dB: stock={b_stock:.5f}  "
              f"corr={b_corr:.5f}")
    write_csv(os.path.join(outdir, "e5_null_checks.csv"),
              ["case", "snr_db", "ber_stock", "ber_corrected"], rows)
    return rows


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--frames", type=int, default=80,
                    help="OFDM frames per measurement point "
                         "(80 frames = 1120 OFDM symbols)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    print(f"pipeline module: {pipeline.__file__}")
    print(f"numpy {np.__version__}, seed {args.seed}, "
          f"{args.frames} frames (= {args.frames*14} OFDM symbols) per point")

    experiment_e1(args.outdir)
    experiment_e2(args.outdir, args.seed, args.frames)
    experiment_e3(args.outdir, args.seed, args.frames)
    experiment_e4(args.outdir, args.seed, args.frames)
    experiment_e5(args.outdir, args.seed, args.frames)

    print(f"\nTotal runtime: {time.time()-t0:.1f} s. "
          f"Outputs in {args.outdir}")


if __name__ == "__main__":
    main()
