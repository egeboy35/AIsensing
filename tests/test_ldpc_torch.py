"""5G NR LDPC encode/decode round-trip (torch-only, marked slow).

sdr_video_comm is imported inside the test function so that collection of
this module never touches torch.  LDPC_AVAILABLE is decided when
sdr_video_comm is first imported: with torch installed the sdr_ldpc
backend loads and the base-graph CSVs resolve relative to sdr_ldpc.py.

This mirrors sdradi/test_ldpc.py with pytest assertions.  CPU decoding of
a n=16384 block takes tens of seconds, hence the slow marker
(deselect with -m "not slow").
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_ldpc_encode_decode_roundtrip():
    pytest.importorskip("torch")
    import sdr_video_comm as svc

    if not svc.LDPC_AVAILABLE:
        pytest.skip("sdr_ldpc LDPC backend not available")

    cfg = svc.FECConfig(
        enabled=True,
        fec_type=svc.FECType.LDPC,
        code_rate="1/2",
        num_bits_per_symbol=4,
    )
    coder = svc.LDPC5GCoder(cfg)
    assert coder.k == 8192
    assert coder.n == 16384

    rng = np.random.default_rng(6)
    info = rng.integers(0, 2, size=coder.k)

    coded = coder.encode(info)
    assert len(coded) == coder.n

    # Flip two coded bits; belief propagation on the +-10 LLRs the decoder
    # builds internally must correct them completely.
    corrupted = coded.copy()
    for idx in (0, 100):
        corrupted[idx] = 1 - corrupted[idx]

    decoded = coder.decode(corrupted)
    assert len(decoded) == coder.k
    assert np.array_equal(decoded, info)
