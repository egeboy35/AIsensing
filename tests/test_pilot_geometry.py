"""Regression tests for the signed-frequency pilot geometry fix in the OFDM RX.

The OFDM receiver in sdradi/sdr_video_commv2.py (and v3) places its pilots at
raw FFT bins [1, 14, 38, 51], which in true (signed) baseband frequency are
[+1, +14, -26, -13] -- non-monotonic. A physical channel (fractional timing
residual, multipath) is smooth in signed frequency, so both the per-symbol
pilot channel interpolation in demodulate() and the pilot-phase-slope fit in
estimate_delay() must run over the signed-frequency axis (attributes
carrier_freqs / pilot_freqs / pilot_freq_order set up by _setup_carriers).

On the previous code (interpolation and polyfit over the raw FFT-index axis):
  * estimate_delay returned approximately -0.54 * tau for a fractional timing
    residual tau -- exactly -830/1538 * tau for this pilot set, so -0.271 at
    tau = +0.5: wrong sign and roughly half the magnitude. The fine-time-sync
    loop is then fed an estimate that rounds to a zero shift, so it exits
    immediately and leaves the offset uncorrected (it under-corrects rather
    than diverging);
  * demodulate had a noise-independent BER floor (~6.6e-2 for QPSK at
    tau = 0.5, noiseless), because the interpolated channel estimate blended
    opposite band edges across the Nyquist guard band.

These tests exercise the actual pipeline classes (no reimplementation) and
fail on that previous code while passing with the fix.

Run from the repository root:
    python -m pytest tests/test_pilot_geometry.py
"""

import os
import sys

# Keep any transitively imported matplotlib headless (myadiclass imports
# matplotlib.pyplot at module level when the SDR stack is installed).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

# Make sdradi/ importable regardless of the pytest invocation directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SDRADI = os.path.join(_REPO_ROOT, "sdradi")
if _SDRADI not in sys.path:
    sys.path.insert(0, _SDRADI)

import sdr_video_commv2 as pipeline_v2
import sdr_video_commv3 as pipeline_v3

SEED = 12345
TAU = 0.5  # fractional timing residual, in samples


def _fractional_delay(signal, tau):
    """Delay a bandlimited baseband stream by tau samples.

    Implemented as a phase ramp over the SIGNED frequencies of the whole
    stream (ideal bandlimited delay) -- the physically correct model of a
    residual timing offset after coarse synchronization. Because every OFDM
    symbol here is bandlimited to the used carriers and the cyclic prefix
    absorbs the shift, each FFT window then sees the per-carrier phase
    exp(-2j*pi*f*tau/N) with f the signed carrier frequency.
    """
    n = len(signal)
    f = np.fft.fftfreq(n) * n
    return np.fft.ifft(np.fft.fft(signal) * np.exp(-2j * np.pi * f * tau / n))


def _make_payload(trx, num_frames, seed=SEED):
    """Random bits + the pipeline's own modulator output (deterministic)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_frames * trx.config.bits_per_frame)
    return bits, trx.modulate(bits)


def _estimator_check(pipeline_module):
    trx = pipeline_module.OFDMTransceiver(pipeline_module.OFDMConfig())
    _, tx = _make_payload(trx, num_frames=2)
    rx = _fractional_delay(tx, TAU)
    est = trx.estimate_delay(rx)
    # Fixed code tracks the true delay (est ~ +0.5). The previous code
    # returned ~ -0.54 * tau (-0.271 here), far outside this window.
    assert abs(est - TAU) < 0.15, (
        f"estimate_delay returned {est:+.3f} for a true fractional delay of "
        f"{TAU:+.2f} samples; expected within +/-0.15. A negative estimate "
        f"means the pilot phase slope was fit over the raw FFT-index axis "
        f"(non-monotonic in signed frequency) -- the pre-fix regression."
    )


def test_estimate_delay_tracks_fractional_delay():
    """v2 estimate_delay must track a +0.5-sample fractional delay (noiseless)."""
    _estimator_check(pipeline_v2)


def test_estimate_delay_tracks_fractional_delay_v3():
    """Same estimator regression check for sdr_video_commv3.py."""
    _estimator_check(pipeline_v3)


def test_demod_ber_under_fractional_residual():
    """Noiseless demod with a 0.5-sample residual must be essentially error-free.

    The pilot-based channel interpolation inside demodulate() is the normal
    correction path for a fractional residual: interpolated over signed
    frequency it equalizes the residual almost perfectly (BER ~ 0), while the
    previous raw-index interpolation left a noise-independent BER floor of
    ~6.6e-2 (QPSK, noiseless) by blending opposite band edges across the
    Nyquist guard band.
    """
    trx = pipeline_v2.OFDMTransceiver(pipeline_v2.OFDMConfig())
    tx_bits, tx = _make_payload(trx, num_frames=20)  # 280 OFDM symbols
    rx = _fractional_delay(tx, TAU)
    rx_bits, _ = trx.demodulate(rx)
    n = len(tx_bits)
    ber = float(np.mean(np.asarray(rx_bits[:n]) != tx_bits))
    assert ber < 1e-2, (
        f"BER {ber:.4f} under a {TAU}-sample fractional residual (noiseless); "
        f"expected < 1e-2. A floor near 6.6e-2 is the pre-fix regression "
        f"(pilot interpolation over the raw FFT-index axis)."
    )
