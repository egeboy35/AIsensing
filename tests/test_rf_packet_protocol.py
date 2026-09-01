"""Packet protocol and QPSK mapping tests for sdradi/pluto_test/rf_image_transfer.py.

rf_image_transfer imports matplotlib at module top (and selects Agg
itself); 'import adi' only happens inside the hardware entry points, so
importing the module is hardware-free.
"""

import numpy as np
import pytest

rf = pytest.importorskip("rf_image_transfer")


def test_packet_build_parse_roundtrip_and_crc():
    rng = np.random.default_rng(7)
    payload = bytes(rng.integers(0, 256, size=48, dtype=np.uint8))

    bits, frame_len = rf.build_packet_bits(seq=42, total_pkts=100, payload=payload, repeat=1)
    # Frame layout: MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | payload | CRC32(4)
    assert frame_len == 10 + len(payload) + 4
    assert len(bits) == 8 * frame_len

    frame_bytes = rf.bits_to_bytes(bits)
    valid, seq, total, rx_payload = rf.parse_packet_data(frame_bytes)
    assert valid
    assert seq == 42
    assert total == 100
    assert rx_payload == payload

    # XOR-corrupt one payload byte: CRC32 covers header + payload.
    corrupted = bytearray(frame_bytes)
    corrupted[10] ^= 0xFF
    valid_bad, _, _, _ = rf.parse_packet_data(bytes(corrupted))
    assert not valid_bad


def test_create_chunks_covers_data_exactly():
    rng = np.random.default_rng(8)
    data = bytes(rng.integers(0, 256, size=100, dtype=np.uint8))

    chunks = list(rf.create_chunks(data, 32))
    assert len(chunks) == 4
    assert [idx for idx, _, _ in chunks] == [0, 1, 2, 3]
    assert all(total == 4 for _, total, _ in chunks)
    assert [len(payload) for _, _, payload in chunks] == [32, 32, 32, 4]
    assert b"".join(payload for _, _, payload in chunks) == data


def test_qpsk_gray_mapping_roundtrip_and_power():
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, size=96).astype(np.uint8)

    symbols = rf.mapping_qpsk_gray(bits)
    assert len(symbols) == 48
    # Constellation normalized by 1/sqrt(2): unit mean symbol power.
    assert abs(np.mean(np.abs(symbols) ** 2) - 1.0) < 1e-6

    rx_bits = rf.demapping_qpsk_gray(symbols)
    assert np.array_equal(np.asarray(rx_bits).ravel()[: len(bits)], bits)
