"""Unit tests for the benchmark metric functions and the repo-parameter mirror.

The metric tests use hand-computed cases only -- no reference implementation, no
golden files -- so a change in the association rule has to break a number a human
can check by hand.
"""

import math
import os
import sys

# Self-contained bootstrap: make the repository root importable and keep
# matplotlib headless, without relying on a conftest.py. Keeping this here
# rather than in tests/conftest.py avoids colliding with the shared
# conftest that the hardware-free pytest suite introduces separately.
os.environ.setdefault("MPLBACKEND", "Agg")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest

from benchmarks.metrics import (
    FrameResult,
    Match,
    Point,
    aggregate,
    associate,
    gate_distance,
    quantization_rmse_floor,
    rmse,
)

# --------------------------------------------------------------------------- #
# association / confusion counts
# --------------------------------------------------------------------------- #


def test_hand_computed_confusion_case():
    """Two targets, three detections, gate +/-2 range bins and +/-1 Doppler bin.

    T0 (bin 100, 32) has two detections inside its gate: D0 (101, 32) at gate
    distance 0.5 and D1 (100, 32) at distance 0.  Greedy nearest-first must take
    D1, leaving D0 as a false alarm.  T1 (bin 200, 10) has nothing inside its gate.
    D2 (bin 500, 5) is far from everything.

    Expected: TP = 1, FP = 2 (D0 and D2), FN = 1 (T1).
    """
    targets = [
        Point(range_bin=100, doppler_bin=32, range_m=15.0, velocity_mps=0.0),
        Point(range_bin=200, doppler_bin=10, range_m=30.0, velocity_mps=-5.0),
    ]
    detections = [
        Point(range_bin=101, doppler_bin=32, range_m=15.15, velocity_mps=0.0),
        Point(range_bin=100, doppler_bin=32, range_m=15.0, velocity_mps=0.0),
        Point(range_bin=500, doppler_bin=5, range_m=75.0, velocity_mps=-7.0),
    ]

    result = associate(targets, detections, gate_range_bins=2, gate_doppler_bins=1)

    assert result.true_positives == 1
    assert result.false_positives == 2
    assert result.false_negatives == 1
    assert [m.target_index for m in result.matches] == [0]
    assert [m.detection_index for m in result.matches] == [1]
    assert result.matches[0].gate_distance == pytest.approx(0.0)
    assert result.unmatched_targets == [1]
    assert result.unmatched_detections == [0, 2]


def test_association_is_one_to_one():
    """One target, three in-gate detections: exactly one match, two false alarms."""
    targets = [Point(50, 16, 10.0, 1.0)]
    detections = [Point(49, 16, 9.9, 1.0), Point(50, 17, 10.0, 1.4), Point(51, 16, 10.1, 1.0)]
    result = associate(targets, detections, gate_range_bins=2, gate_doppler_bins=1)
    assert result.true_positives == 1
    assert result.false_positives == 2
    assert result.false_negatives == 0


def test_gate_boundary_is_inclusive_and_rejects_outside():
    target = Point(100, 30, 15.0, 0.0)
    inside = Point(102, 31, 15.3, 0.5)
    outside_range = Point(103, 30, 15.45, 0.0)
    outside_doppler = Point(100, 32, 15.0, 1.0)

    assert gate_distance(target, inside, 2, 1) == pytest.approx(math.sqrt(2.0))
    assert gate_distance(target, outside_range, 2, 1) is None
    assert gate_distance(target, outside_doppler, 2, 1) is None


def test_zero_gate_requires_exact_bin_match():
    target = Point(10, 4, 1.5, 0.0)
    assert gate_distance(target, Point(10, 4, 1.5, 0.0), 0, 0) == pytest.approx(0.0)
    assert gate_distance(target, Point(11, 4, 1.65, 0.0), 0, 0) is None


def test_negative_gate_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        associate([Point(1, 1, 1.0, 1.0)], [], gate_range_bins=-1, gate_doppler_bins=1)


# --------------------------------------------------------------------------- #
# RMSE
# --------------------------------------------------------------------------- #


def test_known_rmse_case():
    """Two matched pairs with range errors +0.3 m and -0.5 m, velocity 0.0 and +0.4.

    range RMSE    = sqrt((0.3^2 + 0.5^2) / 2) = sqrt(0.17)  = 0.4123105626
    velocity RMSE = sqrt((0.0^2 + 0.4^2) / 2) = sqrt(0.08)  = 0.2828427125
    range bias    = (+0.3 - 0.5) / 2                        = -0.1
    """
    targets = [Point(10, 5, 10.0, 1.0), Point(50, 20, 50.0, 2.0)]
    detections = [Point(10, 5, 10.3, 1.0), Point(51, 20, 49.5, 2.4)]
    result = associate(targets, detections, gate_range_bins=2, gate_doppler_bins=1)
    assert result.true_positives == 2

    errors_r = sorted(m.range_error_m for m in result.matches)
    assert errors_r == pytest.approx([-0.5, 0.3])

    summary = aggregate([result], eligible_cells_per_frame=1000)
    assert summary["range_rmse_m"] == pytest.approx(math.sqrt(0.17))
    assert summary["velocity_rmse_mps"] == pytest.approx(math.sqrt(0.08))
    assert summary["range_bias_m"] == pytest.approx(-0.1)
    assert summary["velocity_bias_mps"] == pytest.approx(0.2)
    assert summary["pd"] == pytest.approx(1.0)


def test_rmse_of_empty_list_is_nan():
    assert math.isnan(rmse([]))
    assert rmse([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))


def test_quantization_floor():
    assert quantization_rmse_floor(0.3) == pytest.approx(0.3 / math.sqrt(12.0))
    assert quantization_rmse_floor(1.0) == pytest.approx(0.28867513459)


# --------------------------------------------------------------------------- #
# degenerate cases
# --------------------------------------------------------------------------- #


def test_empty_detections():
    """No detections at all: Pd = 0, no false alarms, RMSE undefined."""
    targets = [Point(10, 5, 10.0, 1.0), Point(50, 20, 50.0, 2.0)]
    result = associate(targets, [], gate_range_bins=2, gate_doppler_bins=1)
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 2
    assert result.matches == []

    summary = aggregate([result], eligible_cells_per_frame=42240)
    assert summary["pd"] == pytest.approx(0.0)
    assert summary["false_positives"] == 0
    assert summary["false_alarms_per_frame"] == pytest.approx(0.0)
    assert math.isnan(summary["range_rmse_m"])
    assert math.isnan(summary["velocity_rmse_mps"])
    assert math.isnan(summary["precision"])


def test_empty_targets_gives_undefined_pd_and_counts_all_detections_as_false_alarms():
    result = associate([], [Point(10, 5, 10.0, 1.0), Point(11, 5, 10.15, 1.0)], 2, 1)
    assert result.true_positives == 0
    assert result.false_positives == 2
    summary = aggregate([result], eligible_cells_per_frame=1000)
    assert math.isnan(summary["pd"])
    assert summary["false_alarms_per_frame"] == pytest.approx(2.0)
    assert summary["false_alarm_rate_per_cell"] == pytest.approx(2.0 / 1000)


def test_aggregate_over_multiple_frames():
    frames = [
        FrameResult(
            num_targets=1,
            num_detections=2,
            matches=[Match(0, 0, 0.1, -0.2, 0.0)],
            unmatched_targets=[],
            unmatched_detections=[1],
        ),
        FrameResult(
            num_targets=1,
            num_detections=2,
            matches=[],
            unmatched_targets=[0],
            unmatched_detections=[0, 1],
        ),
    ]
    summary = aggregate(frames, eligible_cells_per_frame=1000)
    assert summary["frames"] == 2
    assert summary["ground_truth_targets"] == 2
    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 3
    assert summary["false_negatives"] == 1
    assert summary["pd"] == pytest.approx(0.5)
    assert summary["precision"] == pytest.approx(0.25)
    assert summary["false_alarms_per_frame"] == pytest.approx(1.5)
    assert summary["false_alarm_rate_per_cell"] == pytest.approx(3.0 / 2000)
    assert summary["range_rmse_m"] == pytest.approx(0.1)
    assert summary["velocity_rmse_mps"] == pytest.approx(0.2)


def test_aggregate_rejects_nonpositive_denominator():
    with pytest.raises(ValueError, match="must be positive"):
        aggregate([], eligible_cells_per_frame=0)


# --------------------------------------------------------------------------- #
# the harness must stay faithful to the repo code it measures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def repo():
    from benchmarks.repo_shim import load_repo_modules

    return load_repo_modules()


def test_repo_numpy_detectors_do_not_touch_stubbed_modules(repo):
    from benchmarks.repo_shim import assert_no_stub_dependency

    radar_det = repo["radar_det"]
    assert_no_stub_dependency(
        radar_det.cfar_2d_numpy,
        radar_det.cfar_2d_advanced,
        repo["AIRadarDataset"]._cfar_2d_custom,
    )
    with pytest.raises(AssertionError, match="stubbed module"):
        assert_no_stub_dependency(radar_det.cfar_2d_pytorch)


@pytest.mark.parametrize("config_name", ["config_phaser", "config1", "config2"])
def test_radar_params_matches_repo_dataset_init(repo, config_name, monkeypatch, tmp_path):
    """RadarParams must reproduce AIRadarDataset.__init__ attribute for attribute.

    The harness calls the repo's unbound methods against RadarParams instead of a
    real dataset object (whose ``__init__`` force-generates and HDF5-saves a whole
    dataset).  This test builds one real instance with the save step stubbed out and
    compares every derived quantity.
    """
    from benchmarks.scenarios import RadarParams

    dataset_cls = repo["AIRadarDataset"]
    monkeypatch.setattr(dataset_cls, "save_dataset", lambda self: None)

    real = dataset_cls(
        num_samples=1,
        config_name=config_name,
        max_targets=1,
        apply_realistic_effects=False,
        save_path=str(tmp_path / "unused"),
    )
    mirror = RadarParams(
        repo["RADAR_CONFIGS"][config_name], max_targets=1, apply_realistic_effects=False
    )

    for attr in (
        "fc",
        "B",
        "T",
        "Nc",
        "Ns",
        "fs",
        "num_rx",
        "slope",
        "lambda_c",
        "zero_pad",
        "R_max",
        "num_range_bins",
        "num_doppler_bins",
        "range_resolution",
        "velocity_resolution",
        "max_unambiguous_velocity",
        "v_max",
    ):
        assert getattr(mirror, attr) == pytest.approx(getattr(real, attr)), attr

    np.testing.assert_allclose(mirror.range_axis, real.range_axis)
    np.testing.assert_allclose(mirror.velocity_axis, real.velocity_axis)
    np.testing.assert_allclose(mirror.t_fast, real.t_fast)
    np.testing.assert_allclose(mirror.t_slow, real.t_slow)
    assert mirror.cfar_params == real.cfar_params


def test_ground_truth_quantization_matches_repo_target_mask(repo):
    """Our GT bin indices must be the cells the repo's own target mask lights up."""
    from benchmarks.scenarios import RadarParams, quantize_to_bins

    params = RadarParams(repo["RADAR_CONFIGS"]["config_phaser"], max_targets=1)
    targets = [
        {"range": 41.7, "velocity": -3.2, "rcs": 10.0, "azimuth": 0.0, "elevation": 0.0},
        {"range": 12.05, "velocity": 7.9, "rcs": 20.0, "azimuth": 0.0, "elevation": 0.0},
    ]
    mask = repo["AIRadarDataset"].create_target_mask(params, targets)
    for target in targets:
        r_bin, d_bin = quantize_to_bins(params, target["range"], target["velocity"])
        assert mask[d_bin, r_bin] == 1.0


def test_detector_velocity_formula_matches_velocity_axis(repo):
    """All three detectors infer velocity from (doppler_idx - N//2) * doppler_res."""
    from benchmarks.detectors import check_velocity_convention
    from benchmarks.scenarios import RadarParams

    for config_name in ("config_phaser", "config1", "config2"):
        check_velocity_convention(RadarParams(repo["RADAR_CONFIGS"][config_name]))


def test_complex_rd_map_reproduces_repo_db_map(repo):
    """simulate_frame asserts internally; this pins the behaviour as a test."""
    from benchmarks.scenarios import RadarParams, simulate_frame

    params = RadarParams(repo["RADAR_CONFIGS"]["config_phaser"], max_targets=1)
    frame = simulate_frame(params, trial=0, snr_index=0, snr_db=-25.0, base_seed=7)
    np.testing.assert_allclose(
        20 * np.log10(np.abs(frame.rd_map_complex) + 1e-6), frame.rd_map_db, atol=1e-9
    )
    assert frame.rd_map_detector_input.shape == (1, 2, params.Nc, params.num_range_bins)
    assert len(frame.targets) == 1


def test_simulation_is_reproducible_for_a_fixed_seed(repo):
    from benchmarks.scenarios import RadarParams, simulate_frame

    params = RadarParams(repo["RADAR_CONFIGS"]["config_phaser"], max_targets=1)
    a = simulate_frame(params, trial=3, snr_index=1, snr_db=-30.0, base_seed=11)
    b = simulate_frame(params, trial=3, snr_index=1, snr_db=-30.0, base_seed=11)
    np.testing.assert_array_equal(a.rd_map_db, b.rd_map_db)
    assert a.targets == b.targets

    c = simulate_frame(params, trial=3, snr_index=1, snr_db=-30.0, base_seed=12)
    assert not np.array_equal(a.rd_map_db, c.rd_map_db)


def test_scenario_targets_are_shared_across_snr_points(repo):
    """The SNR sweep must be paired: same scene, different noise realization."""
    from benchmarks.scenarios import RadarParams, simulate_frame

    params = RadarParams(repo["RADAR_CONFIGS"]["config_phaser"], max_targets=1)
    low = simulate_frame(params, trial=2, snr_index=0, snr_db=-45.0, base_seed=5)
    high = simulate_frame(params, trial=2, snr_index=5, snr_db=-15.0, base_seed=5)
    assert low.targets == high.targets
    assert not np.array_equal(low.rd_map_db, high.rd_map_db)


def test_eligible_cell_count_is_within_the_map(repo):
    from benchmarks.detectors import build_detectors
    from benchmarks.scenarios import RadarParams

    params = RadarParams(repo["RADAR_CONFIGS"]["config_phaser"], max_targets=1)
    total_cells = params.num_range_bins * params.num_doppler_bins
    for detector in build_detectors(params):
        eligible = detector.eligible_cells(params)
        assert 0 < eligible <= total_cells


def test_different_trials_draw_different_scenes(repo):
    """Guard the statistical meaning of every Pd number in the benchmark.

    ``draw_scenario`` keys its RNG on ``(base_seed, 1, trial)``. If the ``trial``
    component were dropped, every "independent" frame at an SNR point would be the
    same scene repeated, and the resulting Pd would be one scene's outcome dressed
    up as an average. That mutation used to pass the whole suite, so assert the
    property directly: distinct trials must yield distinct target draws.
    """
    from benchmarks.scenarios import RadarParams, draw_scenario

    params = RadarParams(repo["RADAR_CONFIGS"]["config_phaser"], max_targets=1)
    scenes = []
    for trial in range(8):
        targets, _ = draw_scenario(params, trial=trial, base_seed=4242)
        scenes.append(
            tuple((round(t["range"], 6), round(t["velocity"], 6)) for t in targets)
        )

    assert len(set(scenes)) >= 7, (
        "draw_scenario returned the same scene for different trials "
        f"({len(set(scenes))} distinct out of {len(scenes)}); the per-SNR frames are "
        "then not independent samples and Pd loses its meaning."
    )
    # Same trial must still be reproducible from the same seed.
    again, _ = draw_scenario(params, trial=3, base_seed=4242)
    assert scenes[3] == tuple(
        (round(t["range"], 6), round(t["velocity"], 6)) for t in again
    )


def test_calibration_index_outside_the_evaluation_range(repo):
    """Calibration must not be measured on an evaluation noise realisation.

    Frames are seeded on ``(base_seed, 2, trial, snr_index)``, so a calibration
    index inside ``0..len(snr_points)-1`` would tune each detector's threshold on
    exactly the noise the sweep then scores. The default must sit outside that
    range, and the runner must refuse an overlapping one.
    """
    from benchmarks.run_benchmark import DEFAULT_SNR_DB, build_parser

    args = build_parser().parse_args([])
    assert not (0 <= args.calibration_snr_index < len(DEFAULT_SNR_DB)), (
        f"default --calibration-snr-index {args.calibration_snr_index} overlaps the "
        f"evaluation range 0..{len(DEFAULT_SNR_DB) - 1}"
    )


def test_target_free_frames_carry_no_targets(repo):
    """``measure_curve`` counts every returned peak as a false alarm, which is only
    true if the calibration frames really contain no targets."""
    from benchmarks.scenarios import RadarParams, simulate_target_free_frame

    params = RadarParams(repo["RADAR_CONFIGS"]["config_phaser"], max_targets=1)
    for kind in ("noise_only", "clutter_only"):
        frame = simulate_target_free_frame(
            params, trial=0, snr_index=1000, snr_db=-30.0, base_seed=99, kind=kind
        )
        assert frame.targets == () or len(frame.targets) == 0, kind
