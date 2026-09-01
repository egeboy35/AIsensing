"""OTFS modulate/demodulate loopback test for sdradi/sdr_video_comm.py.

The ISFFT/SFFT pair is an exact inverse, so an identity channel must give
a bit-exact round trip.  channel_est is passed explicitly because the
internal estimator path references OFDM-only config fields and raises
AttributeError on OTFSConfig.
"""

import numpy as np
import pytest

svc = pytest.importorskip("sdr_video_comm")


def test_otfs_loopback_identity_zero_ber():
    cfg = svc.OTFSConfig(mod_order=4)
    trx = svc.OTFSTransceiver(cfg)
    rng = np.random.default_rng(3)

    # Exactly one full delay-Doppler frame: 64 x 256 QPSK = 32768 bits.
    assert cfg.bits_per_frame == 32768
    bits = rng.integers(0, 2, size=cfg.bits_per_frame)

    signal = trx.modulate(bits)
    # MMSE equalization with H = 1 is a positive real scaling, which
    # QPSK minimum-distance demapping is invariant to.
    rx_bits, _ = trx.demodulate(signal, channel_est=np.ones(cfg.N_delay, dtype=complex))

    assert len(rx_bits) >= len(bits)
    ber = np.mean(rx_bits[: len(bits)] != bits)
    assert ber == 0.0
