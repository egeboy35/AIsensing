"""Convolutional FEC (K=7 Viterbi + block interleaver) round-trip tests.

The decoder output is longer than the encoder input (tail bits and
interleaver padding), so comparisons always use a prefix slice.  Input is
kept small (512 bits) because the Viterbi decoder is a pure-Python loop.
"""

import numpy as np
import pytest

svc = pytest.importorskip("sdr_video_comm")

NUM_BITS = 512


def _make_codec():
    cfg = svc.FECConfig(enabled=True, fec_type=svc.FECType.CONVOLUTIONAL)
    return svc.FECCodec(cfg)


def test_conv_fec_clean_roundtrip():
    codec = _make_codec()
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=NUM_BITS)

    coded = codec.encode(bits)
    # Rate 1/2 plus tail and interleaver padding: at least 2x expansion.
    assert len(coded) >= 2 * NUM_BITS

    decoded = codec.decode(coded)
    assert len(decoded) >= NUM_BITS
    assert np.array_equal(decoded[:NUM_BITS], bits)


def test_conv_fec_corrects_scattered_bit_flips():
    codec = _make_codec()
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=NUM_BITS)

    coded = codec.encode(bits)
    corrupted = coded.copy()
    for idx in (10, 250, 500, 700, 900):
        corrupted[idx] = 1 - corrupted[idx]

    decoded = codec.decode(corrupted)
    assert np.array_equal(decoded[:NUM_BITS], bits)
