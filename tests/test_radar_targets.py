"""Target mask and target generation invariants for AIRadarLib.target_utils.

generate_radar_targets draws from the stdlib random module, so
determinism requires random.seed (numpy seeding has no effect on it).
"""

import random

import numpy as np
import pytest

target_utils = pytest.importorskip("AIRadarLib.target_utils")


def test_create_target_mask_position_and_sparsity():
    num_doppler, num_range = 128, 256
    mask = target_utils.create_target_mask(
        [{"distance": 25.0, "velocity": 5.0}],
        num_doppler,
        num_range,
        range_resolution=0.5,
        velocity_resolution=0.25,
    )

    assert mask.shape == (num_doppler, num_range, 1)

    # Target cell: range bin 25/0.5 = 50, doppler bin 64 + 5/0.25 = 84.
    assert mask[84, 50, 0] == 1.0
    # A far-away cell stays empty.
    assert mask[10, 200, 0] == 0.0
    # The Gaussian blob is sparse: under 1% of all cells are hot.
    assert mask.sum() < 0.01 * num_doppler * num_range


def test_generate_radar_targets_spherical_consistency():
    random.seed(123)
    targets = target_utils.generate_radar_targets(
        num_targets=5, min_range=1, max_range=30, range_factor=1.0
    )

    assert len(targets) == 5
    for target in targets:
        # range_factor=1.0 keeps the documented [min_range, max_range] bound.
        assert 1.0 <= target["distance"] <= 30.0
        # Spherical -> Cartesian consistency: |position| == distance.
        position = np.asarray(target["position"], dtype=float)
        assert np.linalg.norm(position) == pytest.approx(
            target["distance"], abs=1e-9
        )
