"""Second study: if the detector choice does not move the curve, what does?

The first study (``benchmarks/run_benchmark.py``) ended in a null result -- at a
common *measured* false-alarm rate the three shipped CFAR detectors are
indistinguishable, and their apparent ranking was an artefact of three different
shipped thresholds.  That answers "which detector", and immediately raises "then
what actually buys sensitivity?".

This module measures that, on the same footing and with the same machinery:

* every configuration is **re-calibrated on target-free frames** before its Pd is
  read, so no comparison here can repeat the threshold artefact the first study
  diagnosed;
* the common operating point is a **false-alarm rate per frame**, not per cell,
  because two of the axes (range zero-padding, chirp count) change how many cells
  the map has while covering the same physical range/velocity volume.  Holding the
  per-cell density fixed would silently grant the finer grid twice the false
  alarms per frame.  ``target_pfa = target_fa_per_frame / eligible_cells`` is
  handed to the *existing* solver in :mod:`benchmarks.calibration`, so the
  denominator cancels and what is actually held constant is the measured number of
  false alarms per frame;
* calibration uses its own scene draws **and** its own noise draws, disjoint from
  the evaluation set on both counts (:data:`CALIBRATION_TRIAL_BASE`,
  :data:`CALIBRATION_INDEX_BASE`), which is strictly stronger than the first
  study's guard;
* every configuration sees the **same physical scenes** -- the target draw depends
  only on ``(seed, trial)`` and on ``R_max``/``v_max``, which none of these axes
  change -- so all comparisons are paired.  :func:`measure_config` records a hash
  of the drawn truth and the runner asserts every configuration agrees.

The axes and why each is here are in :data:`AXES`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from dataclasses import dataclass

import numpy as np

from benchmarks.calibration import CalibrationCurve, measure_curve
from benchmarks.detectors import (
    SPECS_BY_NAME,
    DetectorSpec,
    _common_kwargs,
    _to_points,
    build_detector,
    warp_magnitude,
)
from benchmarks.metrics import Point, aggregate, associate
from benchmarks.repo_shim import load_repo_modules
from benchmarks.scenarios import (
    Frame,
    RadarParams,
    detector_input_from_rd,
    measure_peak_snr_db,
    measure_target_bin_snr_linear,
    repo_db_map,
    scene_reference,
    simulate_frame,
    simulate_target_free_frame,
)

#: Trial indices used for the calibration scenes.  Disjoint from the evaluation
#: trials ``0..trials-1``, so calibration never sees an evaluation scene.
CALIBRATION_TRIAL_BASE = 5000
#: Noise-realisation indices used for the calibration frames.
CALIBRATION_INDEX_BASE = 100000
#: Stride between the noise indices of consecutive SNR points.  Must exceed the
#: largest ``looks`` value so that no two (SNR point, look) pairs collide.
LOOK_STRIDE = 64

CUSTOM = "cfar_custom_datasetv8"
NUMPY_GO = "cfar_numpy_go"
NUMPY_CA = "cfar_numpy_ca"
NUMPY_SO = "cfar_numpy_so"


# --------------------------------------------------------------------------- #
# extra detector specs: the same repo function at its other averaging modes
# --------------------------------------------------------------------------- #


def _build_numpy_method(method: str):
    """``cfar_2d_numpy`` at ``method``, reachable at any threshold by the warp.

    ``benchmarks.detectors`` only registers the GO variant because that is what the
    repo's own pipeline uses.  ``cfar_2d_numpy`` takes ``method`` in ``{'CA', 'GO',
    'SO'}`` and the magnitude-warp reparameterisation of its hard-coded +12 dB
    offset is valid for all three: CA is a linear functional of the dB map, GO and
    SO are a max/min of two such functionals, and all of those are
    translation-equivariant and positively homogeneous, which is exactly what the
    warp argument needs.
    """

    def _build(params, exponent: float, extra: dict):
        repo = load_repo_modules()
        fn = repo["radar_det"].cfar_2d_numpy
        kwargs = {**_common_kwargs(params), "method": method, "estimate_aoa": False}
        exponent = float(exponent)
        identity = exponent == 1.0

        def run(frame, _params):
            data = frame.rd_map_detector_input
            if not identity:
                data = warp_magnitude(data, exponent)
            return _to_points(fn(data, **kwargs))

        recorded = {
            **kwargs,
            "hard_coded_offset_db": 12.0,
            "magnitude_warp_exponent": exponent,
            "equivalent_offset_db": 12.0 / exponent,
        }
        return run, recorded

    return _build


def register_method_specs() -> dict[str, DetectorSpec]:
    """Add the CA and SO variants of ``cfar_2d_numpy`` to the shared registry.

    :func:`benchmarks.calibration.measure_curve` resolves detectors by name against
    ``benchmarks.detectors.SPECS_BY_NAME``, so a new spec has to be visible there.
    Registration is additive and idempotent; the three shipped specs are untouched.
    """
    base = SPECS_BY_NAME[NUMPY_GO]
    added = {}
    for name, method in ((NUMPY_CA, "CA"), (NUMPY_SO, "SO")):
        if name not in SPECS_BY_NAME:
            spec = dataclasses.replace(
                base,
                name=name,
                description=(
                    f"cfar_2d_numpy at method={method!r}: the same repo function and "
                    "the same dB-domain threshold, with the noise estimate taken as "
                    + (
                        "the mean of the whole training region"
                        if method == "CA"
                        else "the smaller of the two training branches"
                    )
                    + " instead of the larger."
                ),
                _build=_build_numpy_method(method),
            )
            SPECS_BY_NAME[name] = spec
        added[name] = SPECS_BY_NAME[name]
    return added


# --------------------------------------------------------------------------- #
# the axis grid
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Knobs:
    """One fully specified configuration of the detection chain."""

    detector: str = CUSTOM
    num_train: int = 10
    num_guard: int = 4
    nms_kernel_size: int = 5
    zero_pad_factor: int = 2
    num_chirps: int = 64
    #: Frames integrated non-coherently before detection (1 = the shipped chain).
    looks: int = 1
    #: ``_cfar_2d_custom``'s moving-target filter (drops ``|v| < 1 m/s``).
    mtd: bool = False

    @property
    def key(self) -> str:
        return (
            f"{self.detector}__train{self.num_train}_guard{self.num_guard}"
            f"_nms{self.nms_kernel_size}_zp{self.zero_pad_factor}"
            f"_nc{self.num_chirps}_looks{self.looks}_mtd{int(self.mtd)}"
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


BASELINE = Knobs()


@dataclass(frozen=True)
class Axis:
    """One knob swept over a small defensible set, plus why it is worth sweeping."""

    name: str
    knob: str
    question: str
    #: ``(printed value, knobs)`` in sweep order.
    members: tuple[tuple[str, Knobs], ...]
    #: The printed value that the other members are compared against.
    baseline_value: str
    #: Extra cost of a member relative to the baseline, stated so a gain can be
    #: read against what it costs.
    cost_note: str


AXES: tuple[Axis, ...] = (
    Axis(
        name="cfar_guard_cells",
        knob="num_guard",
        question=(
            "Do the guard cells cover the target's own main lobe? With a Hann range "
            "window and zero_pad_factor=2 the main lobe is about +/-4 range bins "
            "wide, so a guard block narrower than that feeds target energy into the "
            "noise estimate and the detector masks itself."
        ),
        members=tuple(
            (str(g), dataclasses.replace(BASELINE, num_guard=g))
            for g in (0, 2, 4, 6, 8, 12)
        ),
        baseline_value="4",
        cost_note="window grows as (2*(train+guard)+1)^2; guard=12 costs ~2.4x the baseline convolution",
    ),
    Axis(
        name="cfar_training_cells",
        knob="num_train",
        question=(
            "Classical CFAR loss: a noise estimate averaged over N cells needs a "
            "higher threshold than a known noise level for the same false-alarm "
            "rate. The shipped num_train=10 gives 580 training cells per GO branch, "
            "where that loss should already be negligible -- so this axis is a test "
            "of whether the shipped value is buying anything for its cost."
        ),
        members=tuple(
            (str(t), dataclasses.replace(BASELINE, num_train=t))
            for t in (2, 4, 6, 10, 16)
        ),
        baseline_value="10",
        cost_note="num_train=2 runs ~5x faster than num_train=10; num_train=16 ~1.9x slower",
    ),
    Axis(
        name="nms_kernel",
        knob="nms_kernel_size",
        question=(
            "Non-maximum suppression thins the peak list, so it lowers the measured "
            "false-alarm rate and buys threshold back. 1 disables it entirely "
            "(the repo's `if nms_kernel_size > 1` guard), which makes every "
            "threshold crossing a detection."
        ),
        members=tuple(
            (str(k), dataclasses.replace(BASELINE, nms_kernel_size=k))
            for k in (1, 3, 5, 7, 9, 15)
        ),
        baseline_value="5",
        cost_note="a maximum filter over the whole map; negligible next to the CFAR convolution",
    ),
    Axis(
        name="range_zero_padding",
        knob="zero_pad_factor",
        question=(
            "Zero-padding the range FFT adds no information, but it reduces straddle "
            "(scalloping) loss when a target falls between bins. It also doubles the "
            "cell count, which at a fixed false-alarm rate per frame costs threshold."
        ),
        members=tuple(
            (str(z), dataclasses.replace(BASELINE, zero_pad_factor=z)) for z in (1, 2, 4)
        ),
        baseline_value="2",
        cost_note="detector cost is proportional to the number of range bins",
    ),
    Axis(
        name="coherent_chirps",
        knob="N_chirps",
        question=(
            "Coherent integration: the Doppler FFT sums N_chirps samples in phase, "
            "so the target-to-noise ratio in the map should rise 3 dB per doubling. "
            "The dwell time rises with it -- compare against noncoherent_looks, "
            "which spends the same extra dwell without phase coherence."
        ),
        members=tuple(
            (str(n), dataclasses.replace(BASELINE, num_chirps=n)) for n in (32, 64, 128)
        ),
        baseline_value="64",
        cost_note="simulation and detection cost scale with N_chirps; dwell time doubles per doubling",
    ),
    Axis(
        name="noncoherent_looks",
        knob="looks",
        question=(
            "Non-coherent integration: average the power maps of L successive frames "
            "of the same scene, then detect once. Costs L times the dwell, exactly "
            "like doubling the chirp count -- so the two axes are directly "
            "comparable per unit of time spent."
        ),
        members=tuple(
            (str(n), dataclasses.replace(BASELINE, looks=n)) for n in (1, 2, 4, 8)
        ),
        baseline_value="1",
        cost_note="L simulated frames per detection; detector cost unchanged",
    ),
    Axis(
        name="mtd_filter",
        knob="mtd",
        question=(
            "The dataset pipeline turns `_cfar_2d_custom`'s moving-target filter on "
            "whenever apply_realistic_effects is set. It discards every detection "
            "with |v| < 1 m/s, which removes false alarms near zero Doppler -- and "
            "also removes any real target that happens to be slow."
        ),
        members=(
            ("off", BASELINE),
            ("on", dataclasses.replace(BASELINE, mtd=True)),
        ),
        baseline_value="off",
        cost_note="free (a filter on the detection list)",
    ),
    Axis(
        name="cfar_averaging",
        knob="method",
        question=(
            "GO-CFAR takes the larger of the two training-branch means, which raises "
            "the threshold to protect against clutter edges. In homogeneous noise "
            "that protection is a loss. Measured on cfar_2d_numpy, the one shipped "
            "detector whose `method` argument is reachable (`_cfar_2d_custom` "
            "hard-codes GO)."
        ),
        members=(
            ("GO", dataclasses.replace(BASELINE, detector=NUMPY_GO)),
            ("CA", dataclasses.replace(BASELINE, detector=NUMPY_CA)),
            ("SO", dataclasses.replace(BASELINE, detector=NUMPY_SO)),
        ),
        baseline_value="GO",
        cost_note="identical (the same convolutions, combined differently)",
    ),
)


def unique_configs(axes: tuple[Axis, ...] = AXES) -> dict[str, Knobs]:
    """Every distinct configuration the axes need, in a deterministic order."""
    out: dict[str, Knobs] = {}
    for axis in axes:
        for _, knobs in axis.members:
            out.setdefault(knobs.key, knobs)
    return out


# --------------------------------------------------------------------------- #
# configuration -> radar parameters
# --------------------------------------------------------------------------- #


def build_params(base_config: dict, knobs: Knobs) -> RadarParams:
    """A :class:`RadarParams` for one configuration.

    ``num_chirps`` goes into a copy of the repo's own config dict; the CFAR window
    knobs go into ``cfar_params``, which is where every detector adapter reads them
    from, so no detector code needs to know this study exists.
    """
    config = dict(base_config)
    config["N_chirps"] = int(knobs.num_chirps)
    cfar = dict(base_config.get("cfar_params", {}))
    cfar.update(
        num_train=int(knobs.num_train),
        num_guard=int(knobs.num_guard),
        nms_kernel_size=int(knobs.nms_kernel_size),
    )
    return RadarParams(
        config,
        zero_pad_factor=int(knobs.zero_pad_factor),
        max_targets=1,
        apply_realistic_effects=False,
        clutter_intensity=1.0,
        cfar_params=cfar,
    )


def resolve_gates(params: RadarParams, gate_range_m: float, gate_velocity_mps: float):
    """Association gate in bins, holding the gate fixed in *physical* units.

    Two axes change the bin grid.  Keeping the gate at a constant number of bins
    would silently widen or narrow it in metres and metres per second, so it is
    pinned in physical units and converted back per configuration.  A gate can
    never be smaller than one bin, so a grid coarser than the requested gate gets a
    wider gate than asked for; :func:`measure_config` records the realised gate in
    physical units for exactly that reason.
    """
    gate_r = max(1, round(gate_range_m / params.range_bin_spacing))
    gate_d = max(1, round(gate_velocity_mps / params.velocity_bin_spacing))
    return gate_r, gate_d


def blanked_eligible_cells(detector, params: RadarParams, mtd: bool) -> int:
    """Eligible cells after ``_cfar_2d_custom``'s moving-target filter.

    ``Detector.eligible_cells`` mirrors the range/velocity gates but not the MTD
    filter, which blanks every cell with ``|v| < 1 m/s``.  Reported so the per-cell
    density column stays honest; nothing in the calibration depends on it, because
    the common operating point is a rate per frame.
    """
    cells = detector.eligible_cells(params)
    if not mtd:
        return cells
    centre = params.num_doppler_bins // 2
    dv = params.velocity_bin_spacing
    kept = sum(
        1
        for d in range(params.num_doppler_bins)
        if 1.0 <= abs((d - centre) * dv) < params.v_max
    )
    total = sum(
        1
        for d in range(params.num_doppler_bins)
        if abs((d - centre) * dv) < params.v_max
    )
    return round(cells * kept / total) if total else cells


# --------------------------------------------------------------------------- #
# non-coherent integration
# --------------------------------------------------------------------------- #


def integrate_looks(looks: list[Frame], params: RadarParams) -> Frame:
    """Average the power maps of several looks into one frame.

    ``|X|**2`` averaged over looks, then re-magnitude-ed: the classical
    non-coherent (square-law) integrator, applied to the range-Doppler maps the
    repo simulator produces.  The phase of the first look is carried through
    because the packed ``[1, 2, doppler, range]`` layout wants a complex map; every
    detector benchmarked here consumes only ``|X|`` (``_cfar_2d_custom`` is handed
    the dB map directly and ``cfar_2d_numpy`` takes ``np.abs`` first), and AoA
    estimation -- the one consumer of phase -- is off, so the choice is
    inconsequential.  A unit test asserts detector output is invariant to it.

    A single look is returned untouched, so the ``looks=1`` baseline is bit-for-bit
    the frame the first study measured rather than a round trip through
    ``sqrt(abs(x)**2)``.
    """
    if not looks:
        raise ValueError("need at least one look")
    if len(looks) == 1:
        return looks[0]
    reference = looks[0]
    stack = np.stack([np.abs(f.rd_map_complex) ** 2 for f in looks], axis=0)
    magnitude = np.sqrt(stack.mean(axis=0))
    phase = np.exp(1j * np.angle(reference.rd_map_complex))
    rd_complex = magnitude * phase
    rd_db = repo_db_map(rd_complex)
    targets = list(reference.targets)
    return Frame(
        rd_map_complex=rd_complex,
        rd_map_db=rd_db,
        rd_map_detector_input=detector_input_from_rd(rd_complex),
        targets=reference.targets,
        snr_db=reference.snr_db,
        trial=reference.trial,
        peak_snr_db=measure_peak_snr_db(params, rd_db, targets),
        target_bin_snr_linear=measure_target_bin_snr_linear(rd_complex, targets),
        snr_correction_db=reference.snr_correction_db,
        kind=reference.kind,
    )


def evaluation_noise_index(snr_index: int, look: int) -> int:
    """Noise-realisation index for one (SNR point, look) pair."""
    if not 0 <= look < LOOK_STRIDE:
        raise ValueError(f"look must be in 0..{LOOK_STRIDE - 1}")
    return snr_index * LOOK_STRIDE + look


def calibration_noise_index(look: int) -> int:
    """Noise-realisation index for one calibration look."""
    if not 0 <= look < LOOK_STRIDE:
        raise ValueError(f"look must be in 0..{LOOK_STRIDE - 1}")
    return CALIBRATION_INDEX_BASE + look


def assert_calibration_is_disjoint(num_snr_points: int, trials: int, looks: int) -> None:
    """Fail loudly if calibration could share a scene or a noise draw with the sweep.

    The first study guarded the noise index only; this study also separates the
    scene draw, so both halves of the guard are checked here.
    """
    eval_indices = {
        evaluation_noise_index(i, k) for i in range(num_snr_points) for k in range(looks)
    }
    cal_indices = {calibration_noise_index(k) for k in range(looks)}
    overlap = eval_indices & cal_indices
    if overlap:
        raise AssertionError(
            f"calibration and evaluation share noise indices {sorted(overlap)[:5]}"
        )
    eval_trials = set(range(trials))
    cal_trials = set(range(CALIBRATION_TRIAL_BASE, CALIBRATION_TRIAL_BASE + trials))
    if eval_trials & cal_trials:
        raise AssertionError("calibration and evaluation share scene trials")


# --------------------------------------------------------------------------- #
# one configuration, end to end
# --------------------------------------------------------------------------- #


def truth_digest(frames: list[Frame]) -> str:
    """Hash of the continuous ground truth, to prove two configurations agree.

    Bin indices deliberately excluded: the zero-padding and chirp-count axes change
    the grid, so only the physical truth can be compared.
    """
    parts = []
    for frame in frames:
        for target in frame.targets:
            parts.append(
                f"{frame.trial}:{target.range_m:.9f}:{target.velocity_mps:.9f}"
                f":{target.rcs_dbsm:.9f}"
            )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


#: Most refinement steps the threshold search may add to the calibration grid.
MAX_GRID_EXTENSIONS = 8


def _merge_curves(curves: list[CalibrationCurve], params) -> CalibrationCurve:
    """One curve carrying every measured point, ordered by threshold."""
    by_threshold = {}
    for curve in curves:
        for point in curve.points:
            by_threshold[round(point.effective_threshold_db, 9)] = point
    merged = CalibrationCurve(
        detector=curves[0].detector,
        knob=curves[0].knob,
        scene_kind=curves[0].scene_kind,
        frames=curves[0].frames,
        eligible_cells_per_frame=curves[0].eligible_cells_per_frame,
        points=tuple(by_threshold[k] for k in sorted(by_threshold)),
    )
    object.__setattr__(merged, "_params", params)
    return merged


def calibrate_to_rate(params, knobs: Knobs, frames: list[Frame], settings: dict):
    """Measure the false-alarm curve and solve it for the common rate per frame.

    The measurement is :func:`benchmarks.calibration.measure_curve`, unchanged.  The
    only addition is a bracket search, needed because the axes move the operating
    threshold by several dB in either direction -- eight-look non-coherent
    integration lands about 7 dB below a single look, since averaging eight power
    maps thins the noise tail -- so one fixed grid would either miss the crossing or
    waste measurements on every configuration.  Two cases:

    *   the target rate is coarser than the whole grid (or no grid point produced a
        single false alarm): extend 2 dB below the current minimum;
    *   the target rate is finer than the lowest grid point that produced any false
        alarm, while a higher grid point produced none: the crossing is inside that
        gap and the gap is bisected.  Extending upward there would be useless -- the
        higher point measured zero because the density fell below what the frame
        budget can see, not because the threshold was too low.

    Every measured point is kept and reported, so the search leaves a full audit
    trail in ``axes_calibration.csv``.

    Returns ``(curve, solution, grid_used)``.
    """
    grid = sorted(float(g) for g in settings["threshold_grid_db"])
    target = float(settings["target_fa_per_frame"])
    measured = [
        measure_curve(
            params,
            knobs.detector,
            frames,
            grid,
            scene_kind="noise_only",
            include_shipped=False,
            mtd=knobs.mtd,
        )
    ]
    curve = measured[0]
    target_pfa = target / curve.eligible_cells_per_frame
    solution = curve.solve(target_pfa)
    for _ in range(MAX_GRID_EXTENSIONS):
        if solution.status == "interpolated":
            break
        if solution.status == "clamped_high":
            top = max(p.effective_threshold_db for p in curve.points if p.detections > 0)
            above = [t for t in grid if t > top]
            extra = [0.5 * (top + min(above))] if above else [max(grid) + 1.0]
        else:  # clamped_low or unreachable: the threshold has to come down
            extra = [min(grid) - 2.0, min(grid) - 1.0]
        if any(abs(e - g) < 1e-9 for e in extra for g in grid):
            break
        measured.append(
            measure_curve(
                params,
                knobs.detector,
                frames,
                extra,
                scene_kind="noise_only",
                include_shipped=False,
                mtd=knobs.mtd,
            )
        )
        grid = sorted(grid + extra)
        curve = _merge_curves(measured, params)
        solution = curve.solve(target_pfa)
    return curve, solution, grid


def measure_config(knobs: Knobs, settings: dict) -> dict:
    """Calibrate one configuration to the common false-alarm rate, then sweep it.

    Returns a plain dict (no numpy scalars, no dataclasses) so the whole thing can
    cross a process boundary.
    """
    started = time.perf_counter()
    register_method_specs()
    repo = load_repo_modules()
    base_config = repo["RADAR_CONFIGS"][settings["config"]]
    seed = int(settings["seed"])
    trials = int(settings["trials"])
    snr_points = [float(s) for s in settings["snr_db"]]
    looks = int(knobs.looks)

    assert_calibration_is_disjoint(len(snr_points), trials, looks)

    params = build_params(base_config, knobs)
    gate_r, gate_d = resolve_gates(
        params, settings["gate_range_m"], settings["gate_velocity_mps"]
    )

    # ---- stage 1: calibrate on target-free frames -------------------------- #
    sim_seconds = 0.0
    cal_frames: list[Frame] = []
    for index in range(int(settings["calibration_frames"])):
        trial = CALIBRATION_TRIAL_BASE + index
        t0 = time.perf_counter()
        _, _, reference = scene_reference(params, trial, seed)
        looks_frames = [
            simulate_target_free_frame(
                params,
                trial,
                calibration_noise_index(k),
                float(settings["calibration_snr_db"]),
                seed,
                kind="noise_only",
                snr_reference=settings["snr_reference"],
                reference=reference,
            )
            for k in range(looks)
        ]
        cal_frames.append(integrate_looks(looks_frames, params))
        sim_seconds += time.perf_counter() - t0

    detector_calls = 0
    t0 = time.perf_counter()
    curve, solution, grid_used = calibrate_to_rate(params, knobs, cal_frames, settings)
    calibration_seconds = time.perf_counter() - t0
    detector_calls += len(curve.points) * len(cal_frames)

    eligible = curve.eligible_cells_per_frame
    target_fa_per_frame = float(settings["target_fa_per_frame"])
    target_pfa = target_fa_per_frame / eligible
    if not np.isfinite(solution.native_value):
        raise RuntimeError(
            f"{knobs.key}: no threshold on the grid produced a false alarm, so the "
            f"common operating point of {target_fa_per_frame:.3f} FA/frame cannot be "
            "solved for. Widen --threshold-grid-db."
        )

    detector = build_detector(
        params, knobs.detector, solution.native_value, "axes", mtd=knobs.mtd
    )

    # ---- stage 2: sweep at the calibrated threshold ------------------------ #
    references = {}
    for trial in range(trials):
        t0 = time.perf_counter()
        _, _, references[trial] = scene_reference(params, trial, seed)
        sim_seconds += time.perf_counter() - t0

    rows = []
    detected = np.zeros((len(snr_points), trials), dtype=np.int64)
    truth_count = np.zeros((len(snr_points), trials), dtype=np.int64)
    false_alarms = np.zeros((len(snr_points), trials), dtype=np.int64)
    digest = None
    slow_targets = 0
    detector_seconds = 0.0

    for snr_index, snr_db in enumerate(snr_points):
        frames = []
        for trial in range(trials):
            t0 = time.perf_counter()
            looks_frames = [
                simulate_frame(
                    params,
                    trial,
                    evaluation_noise_index(snr_index, k),
                    snr_db,
                    seed,
                    snr_reference=settings["snr_reference"],
                    reference=references[trial],
                )
                for k in range(looks)
            ]
            frames.append(integrate_looks(looks_frames, params))
            sim_seconds += time.perf_counter() - t0
        if digest is None:
            digest = truth_digest(frames)
            slow_targets = sum(
                1 for f in frames for t in f.targets if abs(t.velocity_mps) < 1.0
            )

        results = []
        for trial, frame in enumerate(frames):
            truth = [
                Point(t.range_bin, t.doppler_bin, t.range_m, t.velocity_mps)
                for t in frame.targets
            ]
            t0 = time.perf_counter()
            detections = detector.run(frame, params)
            detector_seconds += time.perf_counter() - t0
            detector_calls += 1
            result = associate(truth, detections, gate_r, gate_d)
            results.append(result)
            detected[snr_index, trial] = result.true_positives
            truth_count[snr_index, trial] = result.num_targets
            false_alarms[snr_index, trial] = result.false_positives

        agg = aggregate(results, eligible_cells_per_frame=eligible)
        finite = [f.target_bin_snr_linear for f in frames if np.isfinite(f.target_bin_snr_linear)]
        mean_linear = float(np.mean(finite)) if finite else float("nan")
        rows.append(
            {
                "config": knobs.key,
                "snr_db": snr_db,
                "target_bin_snr_db": (
                    float(10.0 * np.log10(mean_linear))
                    if np.isfinite(mean_linear) and mean_linear > 0
                    else float("nan")
                ),
                "mean_peak_snr_db": float(np.mean([f.peak_snr_db for f in frames])),
                **agg,
            }
        )

    return {
        "key": knobs.key,
        "knobs": knobs.as_dict(),
        "detector": knobs.detector,
        "radar": {
            "num_doppler_bins": params.num_doppler_bins,
            "num_range_bins": params.num_range_bins,
            "range_bin_spacing_m": params.range_bin_spacing,
            "velocity_bin_spacing_mps": params.velocity_bin_spacing,
            "v_max_mps": float(params.v_max),
            "r_max_m": float(params.R_max),
        },
        "association": {
            "gate_range_bins": gate_r,
            "gate_doppler_bins": gate_d,
            "gate_range_m": gate_r * params.range_bin_spacing,
            "gate_velocity_mps": gate_d * params.velocity_bin_spacing,
        },
        "calibration": {
            "frames": curve.frames,
            "eligible_cells_per_frame": eligible,
            "eligible_cells_after_mtd": blanked_eligible_cells(
                detector, params, knobs.mtd
            ),
            "target_fa_per_frame": target_fa_per_frame,
            "target_pfa_per_cell": target_pfa,
            "expected_false_alarms": target_fa_per_frame * curve.frames,
            "threshold_grid_db": grid_used,
            "solved_native_value": solution.native_value,
            "solved_effective_threshold_db": solution.effective_threshold_db,
            "status": solution.status,
            "knob": curve.knob,
            "curve": [p.as_dict() for p in curve.points],
            "measured_fa_per_frame": [
                p.detections / curve.frames for p in curve.points
            ],
        },
        "rows": rows,
        "detected": detected.tolist(),
        "targets": truth_count.tolist(),
        "false_alarms": false_alarms.tolist(),
        "truth_digest": digest,
        "slow_targets": slow_targets,
        "timing_s": {
            "total": time.perf_counter() - started,
            "simulation": sim_seconds,
            "calibration_detector": calibration_seconds,
            "sweep_detector": detector_seconds,
            "detector_calls": detector_calls,
        },
    }


def measure_worker(payload: dict) -> dict:
    """Process-pool entry point: rebuild the :class:`Knobs` and measure it."""
    return measure_config(Knobs(**payload["knobs"]), payload["settings"])
