"""OFDM modulate/demodulate loopback tests for sdradi/sdr_video_comm.py.

Note: OFDMTransceiver.__init__ reseeds the *global* numpy RNG
(np.random.seed(42)) to make its pilots reproducible, so every test here
uses an independent np.random.default_rng instance for its own data.
"""

import numpy as np
import pytest

svc = pytest.importorskip("sdr_video_comm")


def _random_bits(rng, n):
    return rng.integers(0, 2, size=n)


def test_ofdm_loopback_identity_channel_zero_ber():
    cfg = svc.OFDMConfig(mod_order=4)
    trx = svc.OFDMTransceiver(cfg)
    rng = np.random.default_rng(1)

    bits = _random_bits(rng, 3 * cfg.bits_per_frame)
    signal = trx.modulate(bits)
    assert len(signal) == 3 * cfg.samples_per_frame

    rx_bits, metrics = trx.demodulate(signal)
    assert len(rx_bits) >= len(bits)
    ber = np.mean(rx_bits[: len(bits)] != bits)
    assert ber == 0.0
    # The SNR estimator is biased even for a perfect loopback, so only
    # assert that the metric exists and is finite - never its magnitude.
    assert np.isfinite(metrics["snr_est_db"])


def test_ofdm_flat_channel_equalized_zero_ber():
    cfg = svc.OFDMConfig(mod_order=4)
    trx = svc.OFDMTransceiver(cfg)
    rng = np.random.default_rng(2)

    bits = _random_bits(rng, 3 * cfg.bits_per_frame)
    signal = trx.modulate(bits)

    # Flat channel: attenuation + constant phase rotation.  The
    # pilot-based LS equalizer must remove it completely.
    channel = 0.5 * np.exp(1j * np.pi / 4)
    rx_bits, _ = trx.demodulate(signal * channel)
    ber = np.mean(rx_bits[: len(bits)] != bits)
    assert ber == 0.0


def test_ofdm_pilots_deterministic_across_instances():
    trx_a = svc.OFDMTransceiver(svc.OFDMConfig(mod_order=4))
    trx_b = svc.OFDMTransceiver(svc.OFDMConfig(mod_order=4))

    assert np.array_equal(trx_a.pilot_indices, trx_b.pilot_indices)
    assert np.array_equal(trx_a.pilot_symbols, trx_b.pilot_symbols)
    # Pilots are BPSK: unit magnitude, purely real.
    assert np.all(np.abs(trx_a.pilot_symbols) == 1.0)
    assert np.all(trx_a.pilot_symbols.imag == 0.0)
