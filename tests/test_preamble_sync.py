"""Preamble generation and synchronization tests for SDRVideoLink.

The link object is constructed without any SDR hardware.  SDRConfig() is
passed explicitly so construction never depends on a CWD-relative tuned
config JSON file.
"""

import numpy as np
import pytest

svc = pytest.importorskip("sdr_video_comm")


def _make_link():
    return svc.SDRVideoLink(
        sdr_config=svc.SDRConfig(),
        ofdm_config=svc.OFDMConfig(mod_order=4),
        fec_config=svc.FECConfig(enabled=False),
    )


def test_preamble_is_fixed_and_reproducible():
    link = _make_link()
    pre_a = link._generate_preamble()
    pre_b = link._generate_preamble()

    # 16-sample block repeated 20 times, from a fixed RandomState(12345).
    assert len(pre_a) == 320
    assert np.array_equal(pre_a, pre_b)
    # Unit-magnitude QPSK blocks.
    assert np.allclose(np.abs(pre_a), 1.0)


def test_synchronize_recovers_delayed_payload():
    link = _make_link()
    cfg = link.ofdm_config
    rng = np.random.default_rng(5)

    bits = rng.integers(0, 2, size=cfg.bits_per_frame)
    tx = np.concatenate([link._generate_preamble(), link.ofdm.modulate(bits)])

    # Channel: 123-sample delay, flat gain/rotation, light AWGN.
    channel = 0.5 * np.exp(1j * np.pi / 4)
    rx = np.concatenate([np.zeros(123, dtype=complex), tx * channel])
    noise = (
        rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx))
    ) * (0.005 / np.sqrt(2))
    rx = rx + noise

    payload, meta = link._synchronize(rx)
    assert meta["sync_success"] is True
    # Preamble energy 320 * |0.5| = 160 correlation peak, far above the
    # configured threshold of 40.
    assert meta["peak_val"] > 2 * cfg.sync_threshold

    rx_bits, _ = link.ofdm.demodulate(payload)
    assert len(rx_bits) >= len(bits)
    ber = np.mean(rx_bits[: len(bits)] != bits)
    assert ber == 0.0
