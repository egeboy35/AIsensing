"""Unit tests for the axis study: the Pd -> dB interpolation, its error bars,
non-coherent integration, and the guards that keep the comparison paired.

The interpolation and bootstrap cases are hand-computed -- no golden files, no
reference implementation -- so a change in how a dB of sensitivity is defined has
to break a number a human can check on paper.
"""

import math
import os
import sys

# Self-contained bootstrap: make the repository root importable and keep
# matplotlib headless, without relying on a conftest.py (this branch
# deliberately does not ship one; see tests/test_benchmark_metrics.py).
os.environ.setdefault("MPLBACKEND", "Agg")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest

from benchmarks.sensitivity import (
    binomial_se,
    bootstrap_pd_difference,
    bootstrap_shift,
    frames_to_resolve,
    resample_indices,
    snr_at_pd,
)

# --------------------------------------------------------------------------- #
# Pd -> SNR interpolation
# --------------------------------------------------------------------------- #


def test_snr_at_pd_hand_computed_interpolation():
    """Pd crosses 0.5 a quarter of the way from -36 to -34.5 dB.

    Between the two bracketing points Pd runs 0.40 -> 0.80, so reaching 0.50 needs
    (0.50 - 0.40) / (0.80 - 0.40) = 0.25 of the 1.5 dB step: -36 + 0.375 = -35.625.
    """
    snr = [-39.0, -37.5, -36.0, -34.5, -33.0]
    pd = [0.0, 0.1, 0.4, 0.8, 1.0]
    result = snr_at_pd(snr, pd, 0.5)
    assert result.status == "interpolated"
    assert result.snr_db == pytest.approx(-35.625)
    assert result.bracket_snr_db == (-36.0, -34.5)
    assert result.bracket_pd == (0.4, 0.8)


def test_snr_at_pd_exact_grid_hit_returns_that_point():
    result = snr_at_pd([-10.0, -8.0, -6.0], [0.0, 0.5, 1.0], 0.5)
    assert result.status == "interpolated"
    assert result.snr_db == pytest.approx(-8.0)


def test_snr_at_pd_takes_the_last_upward_crossing():
    """A dip back below the level after an early crossing must not win.

    The curve crosses 0.5 twice: between index 0 and 1, and again between index 3
    and 4.  The transition into the saturated region is the second one.
    """
    snr = [-40.0, -38.0, -36.0, -34.0, -32.0]
    pd = [0.4, 0.6, 0.5, 0.4, 1.0]
    result = snr_at_pd(snr, pd, 0.5)
    assert result.status == "interpolated"
    # between -34 (0.4) and -32 (1.0): (0.5-0.4)/(1.0-0.4) = 1/6 of 2 dB
    assert result.snr_db == pytest.approx(-34.0 + 2.0 / 6.0)


def test_snr_at_pd_refuses_to_extrapolate_below_the_grid():
    result = snr_at_pd([-30.0, -28.0], [0.7, 1.0], 0.5)
    assert result.status == "below_grid"
    assert result.snr_db is None


def test_snr_at_pd_refuses_to_extrapolate_above_the_grid():
    result = snr_at_pd([-30.0, -28.0], [0.0, 0.3], 0.5)
    assert result.status == "above_grid"
    assert result.snr_db is None


def test_snr_at_pd_rejects_bad_input():
    with pytest.raises(ValueError):
        snr_at_pd([-30.0, -32.0], [0.0, 1.0], 0.5)  # not increasing
    with pytest.raises(ValueError):
        snr_at_pd([-30.0], [0.0], 0.5)  # too short
    with pytest.raises(ValueError):
        snr_at_pd([-30.0, -28.0], [0.0, 1.0], 0.0)  # degenerate level
    with pytest.raises(ValueError):
        snr_at_pd([-30.0, -28.0], [0.0], 0.5)  # length mismatch


# --------------------------------------------------------------------------- #
# error bars
# --------------------------------------------------------------------------- #


def test_binomial_se_at_half_over_sixteen_frames():
    assert binomial_se(0.5, 16) == pytest.approx(0.125)
    assert binomial_se(1.0, 16) == 0.0


def test_binomial_se_rejects_bad_input():
    with pytest.raises(ValueError):
        binomial_se(0.5, 0)
    with pytest.raises(ValueError):
        binomial_se(1.5, 16)


def test_frames_to_resolve_hand_computed():
    """A difference exactly at one standard error needs z**2 times the frames.

    difference = 0.10, se = 0.10 at 16 frames, so n = 16 * (1.96 * 0.10 / 0.10)**2
    = 16 * 3.8415 = 61.5 -> 62 frames.
    """
    assert frames_to_resolve(0.10, 0.10, 16) == 62


def test_frames_to_resolve_never_reports_fewer_than_were_run():
    assert frames_to_resolve(1.0, 0.01, 16) == 16


def test_frames_to_resolve_of_no_difference_is_undefined():
    assert frames_to_resolve(0.0, 0.1, 16) is None


def test_frames_to_resolve_rejects_bad_input():
    with pytest.raises(ValueError):
        frames_to_resolve(0.1, 0.1, 0)
    with pytest.raises(ValueError):
        frames_to_resolve(0.1, -0.1, 16)


# --------------------------------------------------------------------------- #
# the paired bootstrap
# --------------------------------------------------------------------------- #


def test_resample_indices_are_reproducible_and_in_range():
    a = resample_indices(16, 50, seed=3)
    b = resample_indices(16, 50, seed=3)
    c = resample_indices(16, 50, seed=4)
    assert a.shape == (50, 16)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= 0 and a.max() <= 15


def test_bootstrap_shift_of_a_configuration_against_itself_is_exactly_zero():
    snr = [-40.0, -38.0, -36.0]
    detected = np.array([[0, 0], [1, 0], [1, 1]])
    targets = np.ones_like(detected)
    indices = resample_indices(2, 100, seed=1)
    result = bootstrap_shift(snr, detected, targets, detected, targets, indices)
    assert result["shift_db_median"] == 0.0
    assert result["shift_db_lo"] == 0.0
    assert result["shift_db_hi"] == 0.0


def test_bootstrap_shift_sign_is_positive_when_the_variant_is_more_sensitive():
    """A variant whose curve is shifted one grid step to the left gains +2 dB.

    Both curves are deterministic (every frame agrees), so every resample gives the
    same answer and the interval collapses onto it.
    """
    snr = [-40.0, -38.0, -36.0, -34.0]
    baseline = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    variant = np.array([[0, 0], [1, 1], [1, 1], [1, 1]])
    targets = np.ones_like(baseline)
    indices = resample_indices(2, 100, seed=1)
    result = bootstrap_shift(snr, baseline, targets, variant, targets, indices)
    assert result["shift_db_median"] == pytest.approx(2.0)
    assert result["shift_db_lo"] == pytest.approx(2.0)
    assert result["shift_db_hi"] == pytest.approx(2.0)
    assert result["undefined_fraction"] == 0.0


def test_bootstrap_shift_reports_undefined_resamples_instead_of_hiding_them():
    snr = [-40.0, -38.0]
    detected = np.array([[0, 0], [0, 0]])  # never detects: no crossing anywhere
    targets = np.ones_like(detected)
    indices = resample_indices(2, 25, seed=1)
    result = bootstrap_shift(snr, detected, targets, detected, targets, indices)
    assert result["undefined_fraction"] == 1.0
    assert result["shift_db_median"] is None


def test_bootstrap_pd_difference_recovers_the_observed_difference():
    baseline = np.array([1, 0, 0, 0])
    variant = np.array([1, 1, 1, 0])
    targets = np.ones(4, dtype=int)
    indices = resample_indices(4, 500, seed=2)
    result = bootstrap_pd_difference(baseline, targets, variant, targets, indices)
    assert result["pd_difference"] == pytest.approx(0.5)
    assert result["pd_difference_lo"] <= 0.5 <= result["pd_difference_hi"]
    assert result["pd_difference_se"] > 0


# --------------------------------------------------------------------------- #
# axis plumbing
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def repo():
    from benchmarks.repo_shim import load_repo_modules

    return load_repo_modules()


def test_method_specs_register_without_touching_the_shipped_ones(repo):
    from benchmarks.axes import NUMPY_CA, NUMPY_GO, NUMPY_SO, register_method_specs
    from benchmarks.detectors import SPECS, SPECS_BY_NAME

    before = {spec.name: spec for spec in SPECS}
    added = register_method_specs()
    assert set(added) == {NUMPY_CA, NUMPY_SO}
    assert NUMPY_CA in SPECS_BY_NAME and NUMPY_SO in SPECS_BY_NAME
    for name, spec in before.items():
        assert SPECS_BY_NAME[name] is spec
    assert register_method_specs().keys() == added.keys()  # idempotent
    assert SPECS_BY_NAME[NUMPY_CA].knob == SPECS_BY_NAME[NUMPY_GO].knob


def test_build_params_routes_every_knob_to_where_the_repo_reads_it(repo):
    from benchmarks.axes import BASELINE, Knobs, build_params

    config = repo["RADAR_CONFIGS"]["config_phaser"]
    knobs = Knobs(num_train=3, num_guard=7, nms_kernel_size=9, zero_pad_factor=4, num_chirps=32)
    params = build_params(config, knobs)
    assert params.cfar_params["num_train"] == 3
    assert params.cfar_params["num_guard"] == 7
    assert params.cfar_params["nms_kernel_size"] == 9
    assert params.num_doppler_bins == 32
    assert params.zero_pad == 4 * params.Ns
    # the repo's own config dict must not be mutated
    assert config["N_chirps"] == 64
    assert config["cfar_params"]["num_train"] == 10
    baseline = build_params(config, BASELINE)
    assert baseline.cfar_params["num_train"] == 10
    assert baseline.num_doppler_bins == 64


def test_gate_is_held_fixed_in_physical_units_across_grids(repo):
    from benchmarks.axes import BASELINE, build_params, resolve_gates

    config = repo["RADAR_CONFIGS"]["config_phaser"]
    base = build_params(config, BASELINE)
    gate_m = 2 * base.range_bin_spacing
    gate_mps = base.velocity_bin_spacing

    for zero_pad, chirps in ((1, 64), (2, 64), (4, 64), (2, 128)):
        params = build_params(
            config, type(BASELINE)(zero_pad_factor=zero_pad, num_chirps=chirps)
        )
        gate_r, gate_d = resolve_gates(params, gate_m, gate_mps)
        assert gate_r * params.range_bin_spacing == pytest.approx(gate_m, rel=1e-9)
        assert gate_d * params.velocity_bin_spacing == pytest.approx(gate_mps, rel=1e-9)


def test_calibration_indices_and_trials_are_disjoint_from_evaluation():
    from benchmarks.axes import (
        CALIBRATION_TRIAL_BASE,
        assert_calibration_is_disjoint,
        calibration_noise_index,
        evaluation_noise_index,
    )

    assert_calibration_is_disjoint(num_snr_points=13, trials=16, looks=8)
    evaluation = {evaluation_noise_index(i, k) for i in range(13) for k in range(8)}
    assert len(evaluation) == 13 * 8
    assert evaluation.isdisjoint({calibration_noise_index(k) for k in range(8)})
    with pytest.raises(AssertionError):
        assert_calibration_is_disjoint(1, CALIBRATION_TRIAL_BASE + 1, 1)
    with pytest.raises(ValueError):
        evaluation_noise_index(0, 64)


def test_merge_curves_deduplicates_and_orders_by_threshold():
    from benchmarks.axes import _merge_curves
    from benchmarks.calibration import CalibrationCurve, CalibrationPoint

    def curve(thresholds):
        return CalibrationCurve(
            detector="d",
            knob="k",
            scene_kind="noise_only",
            frames=4,
            eligible_cells_per_frame=100,
            points=tuple(
                CalibrationPoint(t, 1.0, 3, 3 / 400.0) for t in thresholds
            ),
        )

    merged = _merge_curves([curve([8.0, 10.0]), curve([9.0, 10.0])], params=None)
    assert [p.effective_threshold_db for p in merged.points] == [8.0, 9.0, 10.0]


# --------------------------------------------------------------------------- #
# non-coherent integration
# --------------------------------------------------------------------------- #


def _fake_frame(rd_complex):
    from benchmarks.scenarios import Frame, detector_input_from_rd, repo_db_map

    return Frame(
        rd_map_complex=rd_complex,
        rd_map_db=repo_db_map(rd_complex),
        rd_map_detector_input=detector_input_from_rd(rd_complex),
        targets=(),
        snr_db=-30.0,
        trial=0,
        peak_snr_db=float("nan"),
        kind="noise_only",
    )


def test_single_look_integration_returns_the_frame_untouched():
    from benchmarks.axes import integrate_looks

    frame = _fake_frame(np.array([[1 + 1j, 2 - 3j]]))
    assert integrate_looks([frame], params=None) is frame


def test_integration_averages_power_not_amplitude():
    """Two looks of magnitude 3 and 4 integrate to sqrt((9 + 16)/2) = 3.5355.

    Averaging the amplitudes would give 3.5, so the two are distinguishable.
    """
    from benchmarks.axes import integrate_looks

    a = _fake_frame(np.array([[3.0 + 0j]]))
    b = _fake_frame(np.array([[0.0 + 4.0j]]))
    merged = integrate_looks([a, b], params=None)
    assert abs(merged.rd_map_complex[0, 0]) == pytest.approx(math.sqrt(12.5))
    assert abs(merged.rd_map_complex[0, 0]) != pytest.approx(3.5, abs=1e-6)
    assert merged.targets == ()
    assert merged.kind == "noise_only"


def test_integration_rejects_an_empty_look_list():
    from benchmarks.axes import integrate_looks

    with pytest.raises(ValueError):
        integrate_looks([], params=None)


def test_integrated_frame_reproduces_a_single_look_to_float_precision():
    from benchmarks.axes import integrate_looks

    rng = np.random.default_rng(0)
    rd = rng.normal(size=(8, 16)) + 1j * rng.normal(size=(8, 16))
    frame = _fake_frame(rd)
    merged = integrate_looks([frame, frame], params=None)
    assert np.allclose(merged.rd_map_complex, rd, rtol=1e-12, atol=1e-12)


def test_detector_output_is_invariant_to_the_carried_phase(repo):
    """The phase of the integrated map is arbitrary; no benchmarked detector sees it.

    ``integrate_looks`` carries the first look's phase purely to fill the complex
    layout ``cfar_2d_numpy`` expects.  If any detector depended on it the choice
    would be a hidden parameter of the study, so this asserts it does not.
    """
    from benchmarks.axes import BASELINE, build_params, integrate_looks
    from benchmarks.detectors import build_detector
    from benchmarks.scenarios import scene_reference, simulate_frame

    params = build_params(repo["RADAR_CONFIGS"]["config_phaser"], BASELINE)
    _, _, reference = scene_reference(params, 0, 4242)
    looks = [
        simulate_frame(params, 0, k, -30.0, 4242, snr_reference="target", reference=reference)
        for k in range(2)
    ]
    merged = integrate_looks(looks, params)
    scrambled = _fake_frame(
        np.abs(merged.rd_map_complex) * np.exp(1j * np.angle(looks[1].rd_map_complex))
    )
    detector = build_detector(params, "cfar_numpy_go", 1.0, "test")
    assert detector.run(merged, params) == detector.run(scrambled, params)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def test_fa_rate_error_db_converts_a_rate_miss_into_dB_of_threshold():
    """A curve falling one decade per dB, missed by one decade, is 1 dB of threshold.

    The curve gives 40 false alarms per frame at 8 dB and 4 at 9 dB. The target is 4
    and the solve landed on 9 dB, but the configuration actually realised 40, so its
    threshold effectively sits 1 dB too low -- reported positive, "ran hot".
    """
    from benchmarks.run_axes import fa_rate_error_db

    result = {
        "calibration": {
            "frames": 10,
            "target_fa_per_frame": 4.0,
            "solved_effective_threshold_db": 8.5,
            "curve": [
                {"effective_threshold_db": 8.0, "false_alarms": 400},
                {"effective_threshold_db": 9.0, "false_alarms": 40},
            ],
        }
    }
    assert fa_rate_error_db(result, 40.0) == pytest.approx(1.0)
    assert fa_rate_error_db(result, 4.0) == pytest.approx(0.0)
    assert fa_rate_error_db(result, 0.4) == pytest.approx(-1.0)
    assert fa_rate_error_db(result, 0.0) is None


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    from benchmarks.run_axes import TABLE_BEGIN, TABLE_END, update_readme

    readme = tmp_path / "README.md"
    readme.write_text(
        f"head\n{TABLE_BEGIN}\nold table\n{TABLE_END}\ntail\n", encoding="utf-8"
    )
    assert update_readme("new table", str(readme)) is True
    text = readme.read_text(encoding="utf-8")
    assert "old table" not in text
    assert "new table" in text
    assert text.startswith("head\n")
    assert text.endswith("tail\n")


def test_update_readme_refuses_a_file_without_markers(tmp_path):
    from benchmarks.run_axes import update_readme

    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    assert update_readme("new table", str(readme)) is False
    assert readme.read_text(encoding="utf-8") == "no markers here\n"
    assert update_readme("new table", str(tmp_path / "missing.md")) is False


# --------------------------------------------------------------------------- #
# end to end, on the smallest budget that still exercises the whole path
# --------------------------------------------------------------------------- #


def test_measure_config_is_reproducible_and_calibrates_to_the_requested_rate(repo):
    from benchmarks.axes import Knobs, measure_config

    settings = {
        "config": "config_phaser",
        "seed": 20260822,
        "trials": 2,
        "snr_db": [-33.0],
        "snr_reference": "target",
        "threshold_grid_db": [8.0, 9.0, 10.0],
        "calibration_frames": 2,
        "calibration_snr_db": -30.0,
        "target_fa_per_frame": 4.0,
        "gate_range_m": 0.3,
        "gate_velocity_mps": 0.47,
    }
    knobs = Knobs()
    first = measure_config(knobs, settings)
    second = measure_config(knobs, settings)
    for key in ("detected", "targets", "false_alarms", "truth_digest", "rows"):
        assert first[key] == second[key]
    assert first["calibration"]["status"] == "interpolated"
    assert first["calibration"]["target_fa_per_frame"] == 4.0
    # the solved per-cell target must be the requested rate divided by the cells
    assert first["calibration"]["target_pfa_per_cell"] == pytest.approx(
        4.0 / first["calibration"]["eligible_cells_per_frame"]
    )
    assert first["association"]["gate_range_bins"] == 2
    assert first["association"]["gate_doppler_bins"] == 1


def test_albersheim_matches_a_hand_computed_point():
    """Pin the closed form against arithmetic a reader can redo on paper.

    For Pd = 0.5 the logit term B = ln(0.5/0.5) = 0, so the expression collapses
    to -5 log10(N) + (6.2 + 4.54/sqrt(N + 0.44)) * log10(ln(0.62/Pfa)), which is
    short enough to check by hand.
    """
    import math

    from benchmarks.sensitivity import albersheim_snr_db

    pfa = 1e-4
    a = math.log(0.62 / pfa)
    for n in (1, 8):
        expected = -5.0 * math.log10(n) + (6.2 + 4.54 / math.sqrt(n + 0.44)) * math.log10(a)
        assert abs(albersheim_snr_db(0.5, pfa, n) - expected) < 1e-9


def test_albersheim_refuses_to_extrapolate():
    """Outside the range the approximation is stated for it must return None,
    not an unmarked number that a reader would take as a prediction."""
    from benchmarks.sensitivity import albersheim_snr_db

    assert albersheim_snr_db(0.99, 1e-4, 8) is None      # Pd above 0.9
    assert albersheim_snr_db(0.5, 1e-9, 8) is None       # Pfa below 1e-7
    assert albersheim_snr_db(0.5, 1e-4, 100000) is None  # N above 8096
    assert albersheim_snr_db(0.5, 1e-4, 8) is not None   # inside the range


def test_noncoherent_gain_is_positive_and_sublinear():
    """Non-coherent integration must gain, but less than the 10 log10(N) a
    coherent sum would give -- that ordering is the physical content."""
    import math

    from benchmarks.sensitivity import noncoherent_gain_db

    for n in (2, 4, 8):
        gain = noncoherent_gain_db(0.5, 1e-4, n)
        assert gain is not None
        assert 0.0 < gain < 10.0 * math.log10(n), (n, gain)
