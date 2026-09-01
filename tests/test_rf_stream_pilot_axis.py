"""Guard the frequency-axis convention of the rf_stream OFDM receivers.

Context
-------
The rf_stream PHY labels its subcarriers k in -26..+26 and stores them at array
positions sc_to_bin(k) = k mod 64, so PILOT_BINS is [43, 57, 7, 21] and sorting
it gives [7, 21, 43, 57]. Read as raw DFT bins those would be the signed
frequencies [+7, +21, -21, -7] -- non-monotonic -- which looks like the defect
that was fixed in sdradi/sdr_video_commv2.py (pilots at raw bins [1, 14, 38, 51]).

It is not the same situation here. The rf_stream transmitter builds
    x = ifft(ifftshift(X)) * sqrt(N)     with X indexed by sc_to_bin(k)
and the receiver reads it back with
    Y = fftshift(fft(...))               (extract_ofdm_symbol)
The ifftshift/fftshift pair rotates the spectrum by N/2, so array position j
holds the physical baseband frequency

    f_phys(j) = j - N_FFT // 2                          (monotonic in j)

not bin_to_sc(j). Sorted PILOT_BINS [7, 21, 43, 57] therefore sits at physical
frequencies [-25, -11, +11, +25], which is strictly increasing, and
pilot_residual_correction()'s interpolation over the raw array index is already
an interpolation over physical frequency (an affine reparametrisation, which
linear interpolation is invariant to).

These tests pin that convention down so the "obvious" change -- re-sorting the
pilots by their subcarrier LABELS (PILOT_SUBCARRIERS / bin_to_sc) -- cannot be
applied without a red test. Measured on this code, that change raises the
noiseless QPSK symbol-error rate at a half-sample timing residual from 0.0000
to 0.1250, and the QAM16 SER at a quarter-sample residual from 0.0000 to 0.1139.

Deterministic, no hardware, no torch, no TensorFlow; runs in well under a second.

Run from the repository root:
    python -m pytest tests/test_rf_stream_pilot_axis.py
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RF_STREAM = os.path.join(_REPO_ROOT, "rf_stream")
if _RF_STREAM not in sys.path:
    sys.path.insert(0, _RF_STREAM)

import rf_stream_rx_step8phy as rx8
import rf_stream_rx_step9phy as rx9
import rf_stream_rx_step10phy as rx10
import rf_stream_tx_step6phy as tx6

N_FFT = rx10.N_FFT
N_CP = rx10.N_CP
SYMBOL_LEN = rx10.SYMBOL_LEN
PILOT_VALS = np.array([1, 1, 1, -1], dtype=np.complex64)

# Physical signed baseband frequency of each raw DFT bin, and of each position
# of the fftshift-ordered arrays the receiver actually works with.
DFT_FREQS = np.fft.fftfreq(N_FFT) * N_FFT
PHYS_FREQS = np.arange(N_FFT) - N_FFT // 2

SEED = 12345
TAU = 0.509  # fractional timing residual, in samples


def _fractional_delay_symbol(sym, tau):
    """Ideal bandlimited (cyclic) fractional delay of one CP-OFDM symbol.

    The 64-sample core is delayed by a phase ramp over the true DFT-bin
    frequencies and the cyclic prefix is rebuilt, which is exactly what a
    residual sample-timing offset does to a CP-OFDM symbol.
    """
    core = sym[N_CP:]
    delayed = np.fft.ifft(
        np.fft.fft(core) * np.exp(-2j * np.pi * DFT_FREQS * tau / N_FFT))
    return np.concatenate([delayed[-N_CP:], delayed]).astype(np.complex64)


def _tx_symbol(sym_idx=0, mod="qpsk", seed=SEED):
    """One payload OFDM symbol from the transmitter's own generator."""
    rng = np.random.default_rng(seed)
    table = tx6.make_constellation(mod)
    data = table[rng.integers(0, len(table), tx6.N_DATA)]
    return tx6.create_ofdm_symbol(data, PILOT_VALS, sym_idx)


def _extract(sym):
    padded = np.concatenate([sym, np.zeros(SYMBOL_LEN, dtype=np.complex64)])
    return rx10.extract_ofdm_symbol(padded, 0)


def _read_source(filename):
    """Read one rf_stream receiver's source text."""
    with open(os.path.join(_RF_STREAM, filename), encoding="utf-8") as fh:
        return fh.read()


def test_array_position_is_physical_frequency():
    """Position j of the receiver's arrays holds physical frequency j - N/2.

    Located model-free: excite one subcarrier label at a time and find the
    resulting tone by the raw DFT of the time-domain samples.
    """
    for k in [k for k in range(-26, 27) if k != 0]:
        X = np.zeros(N_FFT, dtype=np.complex64)
        j = tx6.sc_to_bin(k)
        X[j] = 1.0
        x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
        bin_found = int(np.argmax(np.abs(np.fft.fft(x))))
        f_phys = bin_found if bin_found < N_FFT // 2 else bin_found - N_FFT
        assert f_phys == j - N_FFT // 2, (
            f"subcarrier label {k:+d} is stored at array position {j} and lands "
            f"at physical frequency {f_phys:+d}, but j - N/2 = {j - N_FFT // 2:+d}. "
            f"The ifftshift/fftshift convention has changed; every "
            f"cross-subcarrier interpolation in the receivers depends on it."
        )
        # The label axis (the array position read directly as a signed index)
        # is a different coordinate: it differs from the physical axis by the
        # N/2 rotation the ifftshift/fftshift pair introduces.


def test_sorted_pilot_bins_are_monotonic_in_physical_frequency():
    """Sorting PILOT_BINS ascending is ascending in physical frequency."""
    order = np.argsort(rx10.PILOT_BINS)
    freqs = PHYS_FREQS[np.asarray(rx10.PILOT_BINS)[order]]
    assert list(freqs) == [-25, -11, 11, 25], list(freqs)
    assert np.all(np.diff(freqs) > 0), (
        f"pilot physical frequencies {list(freqs)} are not monotonically "
        f"increasing when PILOT_BINS is sorted ascending; "
        f"pilot_residual_correction()'s np.interp axis would be invalid."
    )


def test_timing_residual_is_linear_in_physical_frequency():
    """A real fractional delay is a phase ramp in (position - N/2), not in labels."""
    sym = _tx_symbol()
    Y_ref = _extract(sym)
    Y_del = _extract(_fractional_delay_symbol(sym, TAU))
    used = np.asarray(rx10.USED_BINS)
    phase = np.angle(Y_del[used] / Y_ref[used])

    slope, offset = np.polyfit(PHYS_FREQS[used], phase, 1)
    resid_phys = float(np.max(np.abs(phase - (slope * PHYS_FREQS[used] + offset))))
    assert resid_phys < 1e-4, (
        f"residual phase is not linear in (array position - N/2): max fit "
        f"residual {resid_phys:.3e} rad."
    )
    assert abs(slope - (-2 * np.pi * TAU / N_FFT)) < 1e-4, (
        f"phase slope {slope:+.5f} rad/bin does not match the "
        f"{-2 * np.pi * TAU / N_FFT:+.5f} rad/bin expected for a "
        f"{TAU}-sample delay."
    )

    label_axis = np.array([b - N_FFT if b >= N_FFT // 2 else b for b in used])
    s2, o2 = np.polyfit(label_axis, phase, 1)
    resid_label = float(np.max(np.abs(phase - (s2 * label_axis + o2))))
    assert resid_label > 0.5, (
        f"the subcarrier-label axis unexpectedly also fits the timing residual "
        f"(max residual {resid_label:.3e} rad); the two axes are supposed to "
        f"differ by an N/2 rotation."
    )


def _residual_phase_error(rxmod, tau, pilot_sign=1):
    """Phase error pilot_residual_correction() leaves, per physical frequency.

    Builds the LTF-equalized symbol a pure timing residual tau would produce,
    feeds it to the real routine, and returns {physical frequency: error}.
    """
    Ye = np.zeros(rxmod.N_FFT, dtype=np.complex64)
    truth = np.exp(-2j * np.pi * PHYS_FREQS * tau / rxmod.N_FFT)
    for b in rxmod.USED_BINS:
        Ye[b] = truth[b]
    Ye[rxmod.PILOT_BINS] *= (pilot_sign * PILOT_VALS)
    H_res = rxmod.pilot_residual_correction(Ye, pilot_sign, PILOT_VALS)
    return {int(PHYS_FREQS[b]): float(np.angle(H_res[b] * np.conj(truth[b])))
            for b in rxmod.USED_BINS}


def test_correction_is_exact_inside_the_pilot_span():
    """Inside |f| <= 25 the timing residual is removed to numerical precision.

    This is the property the raw-array-index axis buys. Re-sorting the pilots by
    subcarrier label instead scrambles the interpolation and leaves errors of
    order 1 rad here.
    """
    for rxmod in (rx10, rx9):
        for tau in (0.25, TAU, 1.0, -TAU):
            err = _residual_phase_error(rxmod, tau)
            inside = {f: e for f, e in err.items() if abs(f) <= 25}
            worst = max(abs(e) for e in inside.values())
            assert worst < 1e-5, (
                f"{rxmod.__name__}: at tau={tau:+.3f} samples the residual "
                f"correction leaves up to {worst:.4f} rad inside the pilot span "
                f"(|f| <= 25), where the four pilots bracket the data. Expected "
                f"an exact fit -- a linear ramp interpolated linearly between "
                f"its own samples. A large value means the interpolation axis is "
                f"no longer the physical-frequency axis."
            )


def test_only_the_outermost_subcarriers_are_extrapolated():
    """Beyond the outermost pilots the error is bounded by nearest-neighbour hold.

    The four pilots span |f| <= 25 while the payload occupies |f| <= 31, so the
    12 subcarriers at |f| = 26..31 are held at the edge pilot value. That is a
    pilot-placement limit, not an axis error, and it is bounded by |slope| * 6.
    """
    tau = TAU
    err = _residual_phase_error(rx10, tau)
    outside = {f: e for f, e in err.items() if abs(f) > 25}
    assert len(outside) == 12, sorted(outside)
    slope = 2 * np.pi * tau / N_FFT
    worst = max(abs(e) for e in outside.values())
    assert worst <= slope * 6 + 1e-6, (
        f"extrapolated error {worst:.4f} rad exceeds the |slope| * 6 = "
        f"{slope * 6:.4f} rad a nearest-neighbour edge hold can produce."
    )


def test_common_phase_error_is_corrected_exactly():
    """A pure common-phase error is removed exactly, on any axis."""
    for theta in (0.4, -1.2, 2.5):
        Ye = np.zeros(N_FFT, dtype=np.complex64)
        for b in rx10.USED_BINS:
            Ye[b] = np.exp(1j * theta)
        Ye[rx10.PILOT_BINS] *= PILOT_VALS
        H_res = rx10.pilot_residual_correction(Ye, 1, PILOT_VALS)
        worst = max(abs(float(np.angle(H_res[b] * np.exp(-1j * theta))))
                    for b in rx10.USED_BINS)
        assert worst < 1e-5, (theta, worst)


def test_step9_and_step10_corrections_agree():
    """The two receivers carry the same routine; keep them in step."""
    rng = np.random.default_rng(SEED)
    for trial in range(64):
        tau = rng.uniform(-2.0, 2.0)
        theta = rng.uniform(-np.pi, np.pi)
        sign = 1 if trial % 2 == 0 else -1
        Ye = np.zeros(N_FFT, dtype=np.complex64)
        for b in rx10.USED_BINS:
            Ye[b] = np.exp(1j * (-2 * np.pi * PHYS_FREQS[b] * tau / N_FFT + theta))
        Ye[rx10.PILOT_BINS] *= (sign * PILOT_VALS)
        a = rx10.pilot_residual_correction(Ye, sign, PILOT_VALS)
        b = rx9.pilot_residual_correction(Ye, sign, PILOT_VALS)
        assert np.max(np.abs(a - b)) < 1e-6, trial


def test_occupied_physical_band_is_shifted_from_the_documented_plan():
    """Pin where the payload actually lands in the physical spectrum.

    README_step10_multitask.md documents ``DATA_SUBCARRIERS = 48  # +-[1..26]``
    with ``PILOT_SUBCARRIERS = [-21, -7, 7, 21]``, i.e. the 802.11a layout with
    a null at DC. ``sc_to_bin(k) = (k + 64) % 64`` produces raw DFT indices,
    but the symbol is then built with ``ifft(ifftshift(X))``, which expects X in
    fftshifted order. The net effect is a rotation by N/2, so subcarrier label
    k is transmitted at physical baseband frequency ``(k % 64) - 32``: label +1
    goes out at -31, label -21 at +11. The link is unaffected because the
    receiver applies the inverse rotation (``fftshift(fft(...))`` with the same
    bin tables), but the occupied band is +-[6, 31] rather than +-[1, 26], and
    the empty region sits at |f| <= 5 instead of only at DC.

    This test records the placement as measured, so a change in either the
    tables or the shift convention is caught rather than discovered later.
    """
    labels = [int(k) for k in tx6.DATA_SUBCARRIERS] +              [int(k) for k in tx6.PILOT_SUBCARRIERS]
    physical = sorted(((k % N_FFT) - N_FFT // 2) for k in labels)

    assert len(physical) == 52
    assert min(abs(f) for f in physical) == 6, physical
    assert max(abs(f) for f in physical) == 31, physical
    # Eleven physical frequencies around DC carry nothing, DC included.
    empty_centre = [f for f in range(-5, 6) if f not in physical]
    assert len(empty_centre) == 11
    assert 0 in empty_centre
    # The documented plan would have put energy at |f| = 1 and none beyond 26.
    assert 1 not in physical
    assert 31 in physical and -31 in physical


def test_step8_has_no_cross_subcarrier_interpolation():
    """Step 8 corrects a scalar CPE only, so no frequency axis is involved."""
    assert not hasattr(rx8, "pilot_residual_correction")
    src = _read_source("rf_stream_rx_step8phy.py")
    assert "np.interp" not in src
    assert "polyfit" not in src


def test_no_polyfit_delay_estimator_in_rf_stream_receivers():
    """There is no pilot-phase-slope delay estimator here to get wrong."""
    for name in ("rf_stream_rx_step8phy.py", "rf_stream_rx_step9phy.py",
                 "rf_stream_rx_step10phy.py"):
        src = _read_source(name)
        assert "polyfit" not in src, name
