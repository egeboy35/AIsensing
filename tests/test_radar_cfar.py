"""CA-CFAR detection tests for AIRadarLib.radar_det.cfar_2d_numpy.

radar_det.py imports torch at module top even though cfar_2d_numpy is
pure numpy/scipy, so the whole module is torch-gated.  The seed is fixed:
the noise-only zero-detection claim is only guaranteed for this draw
(the CFAR threshold is +12 dB over the local mean in the dB domain).
"""

import numpy as np
import pytest

pytest.importorskip("torch", reason="AIRadarLib.radar_det imports torch at module top")
radar_det = pytest.importorskip("AIRadarLib.radar_det")

CFAR_KWARGS = {
    "num_train": 8,
    "num_guard": 4,
    "range_res": 0.5,
    "doppler_res": 0.25,
    "max_range": 100,
    "max_speed": 50,
    "method": "CA",
}
TARGET_DOPPLER, TARGET_RANGE = 80, 100


def _noise_map(sigma=0.05):
    rng = np.random.default_rng(7)
    # Layout: [num_rx, 2 (real/imag), doppler, range] - not complex.
    return rng.normal(0.0, sigma, size=(2, 2, 128, 256))


def test_cfar_detects_injected_target():
    rd_map = _noise_map()
    # Amplitude-10 target on both rx antennas at one delay-Doppler cell.
    rd_map[:, 0, TARGET_DOPPLER, TARGET_RANGE] += 10.0

    detections = radar_det.cfar_2d_numpy(rd_map, **CFAR_KWARGS)
    assert len(detections) == 1

    det = detections[0]
    assert det["doppler_idx"] == TARGET_DOPPLER
    assert det["range_idx"] == TARGET_RANGE
    assert det["range_m"] == 50.0  # 100 * 0.5 m/bin
    assert det["velocity_mps"] == 4.0  # (80 - 64) * 0.25 m/s per bin


def test_cfar_rejects_noise_only_map():
    detections = radar_det.cfar_2d_numpy(_noise_map(), **CFAR_KWARGS)

    # Primary claim: nothing detected at the (would-be) target cell.
    assert not any(
        d["doppler_idx"] == TARGET_DOPPLER and d["range_idx"] == TARGET_RANGE
        for d in detections
    )
    # With this fixed seed, the map is entirely detection-free.
    assert len(detections) == 0
