"""AIRadar classical-detector benchmark: one entry point, real numbers.

Three stages, in order:

1. **Calibration.** Measure every detector's per-cell false-alarm density as a
   function of its own threshold knob, on target-free scenes (noise-only, and
   separately clutter-only), and solve each knob for one common target density.
   The measured curve is recorded so the solve can be audited.
2. **Sweep.** Pd / false alarms / RMSE vs SNR for every detector, both at the
   calibrated common operating point (the headline comparison) and at the settings
   the repository ships (kept because that is also useful information).
3. **ROC.** Pd against *measured* false-alarm density at fixed SNR points, which is
   the only fully honest comparison between detectors whose threshold parameters do
   not mean the same thing.

Usage
-----
    python -m benchmarks.run_benchmark                       # default sweep
    python -m benchmarks.run_benchmark --trials 4 --quick    # fast smoke run
    python -m benchmarks.run_benchmark --update-readme       # refresh README tables

Everything is CPU-only and torch-free; see ``benchmarks/repo_shim.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata

import numpy as np

from benchmarks.calibration import (
    CalibrationCurve,
    measurability,
    measure_curve,
)
from benchmarks.detectors import SPECS, build_detectors
from benchmarks.metrics import Point, aggregate, associate, quantization_rmse_floor
from benchmarks.repo_shim import REPO_ROOT, load_repo_modules
from benchmarks.scenarios import RadarParams

#: Input SNR sweep, in dB, referenced per ``--snr-reference``.  Chosen to bracket
#: the Pd transition of every variant measured here (peak SNR ~2 dB to ~28 dB).
DEFAULT_SNR_DB = [-42.0, -39.0, -36.0, -33.0, -30.0, -27.0, -24.0]

#: SNR points at which the full threshold grid is swept to build a ROC.
DEFAULT_ROC_SNR_DB = [-33.0, -27.0]

#: Common threshold axis for the calibration sweep: dB above the local mean noise
#: power.  See :mod:`benchmarks.detectors` for the per-detector conversion.
DEFAULT_THRESHOLD_GRID_DB = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]

#: Target false-alarm densities to solve for.  Both are measurable at the frame
#: budget used here -- ``benchmarks.calibration.measurability`` states the
#: arithmetic and the results file records it.
DEFAULT_TARGET_PFA = [1e-3, 1e-4]

#: Scenario axis: name -> (apply_realistic_effects, clutter_intensity).
#:
#: ``clutter_default`` is the ``AIRadarDataset.__init__`` default and is **not** in
#: the default sweep: measurement showed its clutter lands ~24 dB below the primary
#: target, and that every Pd/TP/FN/RMSE cell of its table was byte-equal to
#: ``clutter_off``.  It is kept selectable via ``--scenarios`` so that measurement
#: can be reproduced.
SCENARIOS: dict[str, tuple[bool, float]] = {
    "clutter_off": (False, 1.0),
    "clutter_default": (True, 1.0),
    "clutter_strong": (True, 1e4),
}
DEFAULT_SCENARIOS = ["clutter_off", "clutter_strong"]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
README_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
TABLE_BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
TABLE_END = "<!-- END GENERATED RESULTS -->"

AS_SHIPPED = "as_shipped"


# --------------------------------------------------------------------------- #
# environment / provenance
# --------------------------------------------------------------------------- #


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", REPO_ROOT, *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def provenance_report() -> dict:
    """Identify the code that produced these numbers, dirty tree included.

    A bare commit hash is not enough: the results are generated *before* they are
    committed, so the recorded ``head_commit`` is necessarily the parent of the
    commit that carries this file.  ``worktree_dirty`` plus ``tracked_diff_sha256``
    (the SHA-256 of ``git diff HEAD``) pin down what the tree actually contained;
    if the tree was clean, the head commit alone identifies the code.
    """
    head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    diff = _git("diff", "HEAD")
    dirty = bool(status and status.strip())
    return {
        "head_commit": head.strip() if head else None,
        "head_commit_is_parent_of_results_commit": True,
        "branch": branch.strip() if branch else None,
        "worktree_dirty": dirty,
        "tracked_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8", "replace")).hexdigest() if dirty and diff else None
        ),
        "status_porcelain_sha256": (
            hashlib.sha256(status.encode("utf-8", "replace")).hexdigest() if dirty else None
        ),
        "note": (
            "head_commit identifies the tree the run started from. Because results are "
            "generated before being committed, the commit that contains this file is the "
            "child of head_commit. When worktree_dirty is true the head commit alone does "
            "NOT identify the code: tracked_diff_sha256 pins the uncommitted delta."
        ),
    }


def environment_report() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "matplotlib": _package_version("matplotlib"),
        "torch": _package_version("torch"),
        "tqdm": _package_version("tqdm"),
        "device": "cpu",
        "lint": "ruff check benchmarks tests, default rule set (the repository has no "
        "ruff configuration file, so no extra rules are enabled)",
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def sanitize_for_json(value):
    """Recursively replace non-finite floats with ``None`` so the file is valid JSON.

    ``json.dump`` happily writes bare ``NaN`` / ``Infinity`` tokens, which Python
    reads back but which are not JSON and which ``jq`` and most other parsers
    reject.  Undefined metrics (RMSE with no matched pairs, dB of a zero ratio) are
    therefore emitted as ``null``.
    """
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    return value


def _linear_mean_to_db(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    mean = sum(finite) / len(finite)
    if mean <= 0:
        return float("nan")
    return float(10.0 * math.log10(mean))


class _Clock:
    """Accumulates detector time and call counts."""

    def __init__(self) -> None:
        self.detector_time: dict[str, float] = {}
        self.calls = 0
        self.simulation = 0.0

    def run(self, detector, frame, params):
        t0 = time.perf_counter()
        detections = detector.run(frame, params)
        self.detector_time[detector.name] = self.detector_time.get(detector.name, 0.0) + (
            time.perf_counter() - t0
        )
        self.calls += 1
        return detections


# --------------------------------------------------------------------------- #
# stage 1: calibration
# --------------------------------------------------------------------------- #


def run_calibration(args, params_by_mode: dict[str, RadarParams], clock: _Clock) -> dict:
    from benchmarks.scenarios import scene_reference, simulate_target_free_frame

    grid = list(args.threshold_grid_db)
    targets = list(args.target_pfa)

    # Calibration must not reuse an evaluation noise draw. Frames are seeded on
    # (base_seed, 2, trial, snr_index), so an overlapping index would tune each
    # threshold on the same noise realisation the sweep later scores.
    eval_points = len(args.snr_db) if args.snr_db else len(DEFAULT_SNR_DB)
    if 0 <= args.calibration_snr_index < eval_points:
        raise SystemExit(
            f"--calibration-snr-index {args.calibration_snr_index} collides with the "
            f"evaluation range 0..{eval_points - 1}; calibration frames would share "
            "a noise realisation with the sweep. Pick an index outside that range "
            "(the default 1000 is safe)."
        )

    noise_source = "clutter_off" if "clutter_off" in params_by_mode else next(iter(params_by_mode))
    clutter_source = next(
        (m for m in params_by_mode if SCENARIOS[m][0] and m != "clutter_default"), None
    )

    sources = {"noise_only": noise_source}
    if clutter_source is not None:
        sources["clutter_only"] = clutter_source

    frames_by_kind: dict[str, list] = {}
    for kind, mode in sources.items():
        params = params_by_mode[mode]
        frames = []
        for trial in range(args.calibration_frames):
            t0 = time.perf_counter()
            _, _, reference = scene_reference(params, trial, args.seed)
            frames.append(
                simulate_target_free_frame(
                    params,
                    trial,
                    args.calibration_snr_index,
                    args.calibration_snr_db,
                    args.seed,
                    kind=kind,
                    snr_reference=args.snr_reference,
                    reference=reference,
                )
            )
            clock.simulation += time.perf_counter() - t0
        frames_by_kind[kind] = frames

    curves: list[CalibrationCurve] = []
    for kind, frames in frames_by_kind.items():
        params = params_by_mode[sources[kind]]
        for spec in SPECS:
            curve = measure_curve(
                params,
                spec.name,
                frames,
                grid,
                scene_kind=kind,
                mtd=args.mtd,
                on_call=lambda: setattr(clock, "calls", clock.calls + 1),
            )
            curves.append(curve)
            if not args.quiet:
                print(
                    f"[calibrate/{kind}] {spec.name}: "
                    + " ".join(
                        f"{p.effective_threshold_db:.2f}dB->{p.measured_pfa:.2e}"
                        for p in curve.points
                    ),
                    flush=True,
                )

    headline = targets[-1]
    applied: dict[str, dict] = {}
    for spec in SPECS:
        curve = next(c for c in curves if c.detector == spec.name and c.scene_kind == "noise_only")
        solution = curve.solve(headline)
        applied[spec.name] = {
            "knob": spec.knob,
            "native_value": solution.native_value,
            "effective_threshold_db": solution.effective_threshold_db,
            "target_pfa_per_cell": headline,
            "solved_on_scene_kind": "noise_only",
            "status": solution.status,
        }

    reference_cells = curves[0].eligible_cells_per_frame if curves else 0
    return {
        "threshold_grid_db": grid,
        "threshold_axis": (
            "dB above the local mean noise power; converted to each detector's own "
            "knob by the closed forms in benchmarks/detectors.py"
        ),
        "frames": args.calibration_frames,
        "scene_sources": sources,
        "snr_db_used": args.calibration_snr_db,
        "scale_invariance_note": (
            "the noise-only calibration is independent of the SNR point and of the "
            "clutter scenario: the residual maps at two SNR points are exactly "
            "proportional, and a dB-domain or ratio-domain CFAR is scale invariant, so "
            "the detection sets are identical (asserted by a test)"
        ),
        "target_pfa": targets,
        "headline_target_pfa": headline,
        "measurability": [
            measurability(reference_cells, args.calibration_frames, t) for t in targets
        ],
        "curves": [c.as_dict(targets) for c in curves],
        "applied": applied,
        "applied_note": (
            "the sweep uses the noise-only solve. The clutter-only curve is reported "
            "beside it: under clutter_strong the clutter scatterers are real peaks, so "
            "there is a floor on the achievable peak density that no threshold removes."
        ),
    }


# --------------------------------------------------------------------------- #
# stage 2+3: sweep and ROC
# --------------------------------------------------------------------------- #


def _row_from(detector, clutter_mode, snr_db, frame_results, peak_snrs, snr_lins, corrections, params):
    agg = aggregate(frame_results, eligible_cells_per_frame=detector.eligible_cells(params))
    return {
        "detector": detector.name,
        "variant": detector.variant,
        "clutter": clutter_mode,
        "snr_db": snr_db,
        "knob": detector.knob,
        "knob_value": detector.knob_value,
        "effective_threshold_db": detector.effective_threshold_db,
        "mean_peak_snr_db": float(np.mean(peak_snrs)) if peak_snrs else float("nan"),
        "target_bin_snr_db": _linear_mean_to_db(snr_lins),
        "mean_snr_correction_db": float(np.mean(corrections)) if corrections else 0.0,
        **agg,
    }


def run_sweep(args) -> dict:
    repo = load_repo_modules()
    configs = repo["RADAR_CONFIGS"]
    if args.config not in configs:
        raise SystemExit(f"unknown config {args.config!r}; available: {sorted(configs)}")

    snr_points = args.snr_db if args.snr_db else list(DEFAULT_SNR_DB)
    clutter_modes = args.scenarios
    roc_snr = [s for s in args.roc_snr_db if s in snr_points]

    from benchmarks.scenarios import (
        clutter_to_target_power_db,
        scene_reference,
        simulate_frame,
    )

    started = time.perf_counter()
    clock = _Clock()

    params_by_mode: dict[str, RadarParams] = {}
    references: dict[str, dict] = {}
    scenario_meta: dict[str, dict] = {}
    for clutter_mode in clutter_modes:
        enabled, intensity = SCENARIOS[clutter_mode]
        if clutter_mode == "clutter_strong":
            intensity = args.strong_clutter_intensity
        params = RadarParams(
            configs[args.config],
            zero_pad_factor=args.zero_pad_factor,
            max_targets=args.max_targets,
            apply_realistic_effects=enabled,
            clutter_intensity=intensity,
        )
        params_by_mode[clutter_mode] = params
        refs = {}
        for trial in range(args.trials):
            t0 = time.perf_counter()
            _, _, reference = scene_reference(params, trial, args.seed)
            clock.simulation += time.perf_counter() - t0
            refs[trial] = reference
        references[clutter_mode] = refs
        offsets = [r.scene_to_target_power_db for r in refs.values()]
        scenario_meta[clutter_mode] = {
            "apply_realistic_effects": enabled,
            "clutter_intensity": intensity,
            "clutter_to_target_power_db": clutter_to_target_power_db(
                params, args.trials, args.seed
            ),
            "scene_to_target_power_db": {
                "mean": float(np.mean(offsets)),
                "min": float(np.min(offsets)),
                "max": float(np.max(offsets)),
            },
        }

    calibration = run_calibration(args, params_by_mode, clock)
    settings = {k: v["native_value"] for k, v in calibration["applied"].items()}
    calibrated_label = "calibrated_pfa{:.0e}".format(calibration["headline_target_pfa"]).replace(
        "e-0", "e-"
    )

    rows: list[dict] = []
    frame_rows: list[dict] = []
    roc_rows: list[dict] = []
    detector_meta: dict[str, dict] = {}

    for clutter_mode in clutter_modes:
        params = params_by_mode[clutter_mode]
        variants = build_detectors(params, mtd=args.mtd, variant=AS_SHIPPED)
        variants += build_detectors(
            params, mtd=args.mtd, settings=settings, variant=calibrated_label
        )
        for det in variants:
            detector_meta.setdefault(
                det.key,
                {
                    "detector": det.name,
                    "variant": det.variant,
                    "source": det.source,
                    "description": det.description,
                    "knob": det.knob,
                    "knob_value": det.knob_value,
                    "effective_threshold_db": det.effective_threshold_db,
                    "range_gate_low_m": det.range_gate_low_m,
                    "eligible_cells_per_frame": det.eligible_cells(params),
                    "params": {k: _jsonable(v) for k, v in det.params.items()},
                },
            )

        roc_detectors: list = []
        if clutter_mode == clutter_modes[0]:
            for spec in SPECS:
                for effective_db in args.threshold_grid_db:
                    roc_detectors.append(
                        spec.build(
                            params,
                            spec.native_knob(effective_db, params),
                            "roc",
                            mtd=args.mtd,
                        )
                    )

        for snr_index, snr_db in enumerate(snr_points):
            frames = []
            for trial in range(args.trials):
                t0 = time.perf_counter()
                frames.append(
                    simulate_frame(
                        params,
                        trial,
                        snr_index,
                        snr_db,
                        args.seed,
                        snr_reference=args.snr_reference,
                        reference=references[clutter_mode][trial],
                    )
                )
                clock.simulation += time.perf_counter() - t0

            peak_snrs = [f.peak_snr_db for f in frames]
            snr_lins = [f.target_bin_snr_linear for f in frames]
            corrections = [f.snr_correction_db for f in frames]
            truths = [
                [Point(t.range_bin, t.doppler_bin, t.range_m, t.velocity_mps) for t in f.targets]
                for f in frames
            ]

            for det in variants:
                results = []
                for frame, truth in zip(frames, truths):
                    detections = clock.run(det, frame, params)
                    result = associate(
                        truth,
                        detections,
                        args.gate_range_bins,
                        args.gate_doppler_bins,
                        args.near_range_bins,
                        args.near_doppler_bins,
                    )
                    results.append(result)
                    frame_rows.append(
                        {
                            "clutter": clutter_mode,
                            "snr_db": snr_db,
                            "trial": frame.trial,
                            "detector": det.name,
                            "variant": det.variant,
                            "effective_threshold_db": round(det.effective_threshold_db, 4),
                            "peak_snr_db": round(frame.peak_snr_db, 3),
                            "target_bin_snr_linear": round(frame.target_bin_snr_linear, 4),
                            "snr_correction_db": round(frame.snr_correction_db, 4),
                            "num_targets": result.num_targets,
                            "num_detections": result.num_detections,
                            "tp": result.true_positives,
                            "fp": result.false_positives,
                            "fp_near_target": result.false_positives_near_target,
                            "fn": result.false_negatives,
                        }
                    )
                rows.append(
                    _row_from(
                        det,
                        clutter_mode,
                        snr_db,
                        results,
                        peak_snrs,
                        snr_lins,
                        corrections,
                        params,
                    )
                )

            if snr_db in roc_snr and roc_detectors:
                for det in roc_detectors:
                    results = []
                    for frame, truth in zip(frames, truths):
                        detections = clock.run(det, frame, params)
                        results.append(
                            associate(
                                truth,
                                detections,
                                args.gate_range_bins,
                                args.gate_doppler_bins,
                                args.near_range_bins,
                                args.near_doppler_bins,
                            )
                        )
                    agg = aggregate(
                        results, eligible_cells_per_frame=det.eligible_cells(params)
                    )
                    measured_pfa = _noise_only_pfa(
                        calibration, det.name, det.effective_threshold_db
                    )
                    roc_rows.append(
                        {
                            "detector": det.name,
                            "clutter": clutter_mode,
                            "snr_db": snr_db,
                            "knob": det.knob,
                            "knob_value": det.knob_value,
                            "effective_threshold_db": det.effective_threshold_db,
                            "pd": agg["pd"],
                            "true_positives": agg["true_positives"],
                            "false_negatives": agg["false_negatives"],
                            "measured_pfa_noise_only": measured_pfa,
                            "false_alarms_per_frame_on_target_frames": agg[
                                "false_alarms_per_frame"
                            ],
                            "false_alarms_near_target_per_frame": agg[
                                "false_positives_near_target"
                            ]
                            / max(1, agg["frames"]),
                            "target_bin_snr_db": _linear_mean_to_db(snr_lins),
                        }
                    )

            if not args.quiet:
                current = [r for r in rows if r["clutter"] == clutter_mode and r["snr_db"] == snr_db]
                print(
                    f"[{clutter_mode}] snr={snr_db:+.0f} dB "
                    f"tgt_bin_snr={current[0]['target_bin_snr_db']:5.1f} dB  "
                    + "  ".join(
                        f"{r['detector']}/{r['variant']}: Pd={r['pd']:.3f} "
                        f"FA/frame={r['false_alarms_per_frame']:.2f}"
                        for r in current
                    ),
                    flush=True,
                )

    wall = time.perf_counter() - started
    reference_params = params_by_mode[clutter_modes[0]]
    return {
        "config": args.config,
        "radar": reference_params.summary(),
        "sweep": {
            "snr_db": snr_points,
            "trials_per_point": args.trials,
            "scenarios": clutter_modes,
            "base_seed": args.seed,
            "max_targets": args.max_targets,
            "mtd": args.mtd,
            "snr_reference": args.snr_reference,
            "variants": [AS_SHIPPED, calibrated_label],
            "calibrated_variant": calibrated_label,
            "roc_snr_db": roc_snr,
            "roc_scenario": clutter_modes[0] if roc_snr else None,
        },
        "scenario_details": scenario_meta,
        "calibration": calibration,
        "association": {
            "gate_range_bins": args.gate_range_bins,
            "gate_doppler_bins": args.gate_doppler_bins,
            "gate_range_m": args.gate_range_bins * reference_params.range_bin_spacing,
            "gate_velocity_mps": args.gate_doppler_bins * reference_params.velocity_bin_spacing,
            "near_range_bins": args.near_range_bins,
            "near_doppler_bins": args.near_doppler_bins,
            "rule": (
                "rectangular gate in bin space, one-to-one, greedy by increasing "
                "normalized gate distance; every further in-gate detection is a false "
                "positive, so target sidelobes surviving NMS are counted as false "
                "alarms. The near gate splits that component out for reporting only."
            ),
        },
        "quantization_floor": {
            "range_rmse_m": quantization_rmse_floor(reference_params.range_bin_spacing),
            "velocity_rmse_mps": quantization_rmse_floor(reference_params.velocity_bin_spacing),
        },
        "detectors": detector_meta,
        "environment": environment_report(),
        "provenance": provenance_report(),
        "shim": load_repo_modules()["shim_report"].as_dict(),
        "json_encoding": {
            "non_finite_floats": "null",
            "note": (
                "undefined metrics (RMSE with no matched pairs, dB of a non-positive "
                "ratio, clutter/target ratio with clutter disabled) are emitted as JSON "
                "null, never as bare NaN/Infinity tokens."
            ),
        },
        "timing_s": {
            "wall_clock_total": wall,
            "simulation": clock.simulation,
            "detectors": clock.detector_time,
            "detector_calls": clock.calls,
            "frames_simulated": args.trials * len(snr_points) * len(clutter_modes),
        },
        "rows": rows,
        "roc": roc_rows,
        "frames": frame_rows,
    }


def _noise_only_pfa(calibration: dict, detector: str, effective_db: float) -> float | None:
    for curve in calibration["curves"]:
        if curve["detector"] != detector or curve["scene_kind"] != "noise_only":
            continue
        for point in curve["curve"]:
            if abs(point["effective_threshold_db"] - effective_db) < 1e-9:
                return point["measured_pfa_per_cell"]
    return None


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

SUMMARY_FIELDS = [
    "detector",
    "variant",
    "clutter",
    "snr_db",
    "knob",
    "knob_value",
    "effective_threshold_db",
    "target_bin_snr_db",
    "mean_peak_snr_db",
    "mean_snr_correction_db",
    "frames",
    "ground_truth_targets",
    "detections",
    "true_positives",
    "false_positives",
    "false_positives_near_target",
    "false_positives_far_from_target",
    "false_negatives",
    "pd",
    "precision",
    "false_alarms_per_frame",
    "false_alarm_rate_per_cell",
    "eligible_cells_per_frame",
    "range_rmse_m",
    "velocity_rmse_mps",
    "range_bias_m",
    "velocity_bias_mps",
]

CALIBRATION_FIELDS = [
    "detector",
    "scene_kind",
    "knob",
    "effective_threshold_db",
    "native_value",
    "is_shipped_setting",
    "frames",
    "eligible_cells_per_frame",
    "false_alarms",
    "measured_pfa_per_cell",
]

ROC_FIELDS = [
    "detector",
    "clutter",
    "snr_db",
    "target_bin_snr_db",
    "knob",
    "knob_value",
    "effective_threshold_db",
    "measured_pfa_noise_only",
    "pd",
    "true_positives",
    "false_negatives",
    "false_alarms_per_frame_on_target_frames",
    "false_alarms_near_target_per_frame",
]


def write_outputs(report: dict, outdir: str, make_figures: bool) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    written = []

    def _csv(name, fields, records):
        path = os.path.join(outdir, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({k: record.get(k) for k in fields})
        written.append(path)

    _csv("summary.csv", SUMMARY_FIELDS, report["rows"])
    if report["frames"]:
        _csv("frames.csv", list(report["frames"][0].keys()), report["frames"])
    if report["roc"]:
        _csv("roc.csv", ROC_FIELDS, report["roc"])

    calibration_records = []
    for curve in report["calibration"]["curves"]:
        for point in curve["curve"]:
            calibration_records.append(
                {
                    "detector": curve["detector"],
                    "scene_kind": curve["scene_kind"],
                    "knob": curve["knob"],
                    "frames": curve["frames"],
                    "eligible_cells_per_frame": curve["eligible_cells_per_frame"],
                    **point,
                }
            )
    _csv("calibration.csv", CALIBRATION_FIELDS, calibration_records)

    results_json = os.path.join(outdir, "results.json")
    slim = {k: v for k, v in report.items() if k != "frames"}
    with open(results_json, "w", encoding="utf-8") as fh:
        json.dump(sanitize_for_json(slim), fh, indent=2, allow_nan=False)
        fh.write("\n")
    written.append(results_json)

    table_md = os.path.join(outdir, "summary_table.md")
    with open(table_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    written.append(table_md)

    if make_figures:
        written.extend(write_figures(report, outdir))
    return written


def _fmt(value, digits=3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _pfa(value) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0 (none observed)"
    return f"{value:.2e}"


def _metric_table(report: dict, rows: list[dict]) -> list[str]:
    lines = [
        ("| detector | thr (dB) | SNR in (dB) | target-bin SNR (dB) | Pd | FA/frame | "
         "FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |"),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["detector"], r["snr_db"])):
        lines.append(
            "| `{d}` | {thr} | {snr:+.0f} | {tsnr} | {pd} | {fa} | {far} | {rr} | {vr} "
            "| {tp} | {fp} | {fn} |".format(
                d=r["detector"],
                thr=_fmt(r["effective_threshold_db"], 2),
                snr=r["snr_db"],
                tsnr=_fmt(r["target_bin_snr_db"], 1),
                pd=_fmt(r["pd"], 3),
                fa=_fmt(r["false_alarms_per_frame"], 2),
                far=_pfa(r["false_alarm_rate_per_cell"]),
                rr=_fmt(r["range_rmse_m"], 4),
                vr=_fmt(r["velocity_rmse_mps"], 4),
                tp=r["true_positives"],
                fp=r["false_positives"],
                fn=r["false_negatives"],
            )
        )
    return lines


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    radar = report["radar"]
    assoc = report["association"]
    sweep = report["sweep"]
    floor = report["quantization_floor"]
    cal = report["calibration"]

    lines.append(
        f"Radar config `{report['config']}` ({radar['config_name']}): "
        f"{radar['bandwidth_hz'] / 1e6:.0f} MHz BW, {radar['num_chirps']} chirps x "
        f"{radar['samples_per_chirp']} samples, RD map "
        f"{radar['num_doppler_bins']} x {radar['num_range_bins']} bins, "
        f"range bin {radar['range_bin_spacing_m']:.3f} m, "
        f"velocity bin {radar['velocity_bin_spacing_mps']:.3f} m/s."
    )
    lines.append("")
    lines.append(
        f"{sweep['trials_per_point']} frames per SNR point, {sweep['max_targets']} "
        f"target(s) per frame, base seed {sweep['base_seed']}, SNR referenced to "
        f"**{sweep['snr_reference']}** power. Association gate: "
        f"+/-{assoc['gate_range_bins']} range bins ({assoc['gate_range_m']:.3f} m), "
        f"+/-{assoc['gate_doppler_bins']} Doppler bins "
        f"({assoc['gate_velocity_mps']:.3f} m/s). Quantization RMSE floor: "
        f"{floor['range_rmse_m']:.3f} m / {floor['velocity_rmse_mps']:.3f} m/s."
    )
    lines.append("")

    # ---- calibration -------------------------------------------------------
    lines.append("### Stage 1: threshold calibration (threshold -> measured Pfa)")
    lines.append("")
    lines.append(
        f"{cal['frames']} target-free frames per scene kind. Threshold axis: "
        f"{cal['threshold_axis']}. Grid: "
        + ", ".join(f"{g:g}" for g in cal["threshold_grid_db"])
        + " dB."
    )
    lines.append("")
    for m in cal["measurability"]:
        verdict = "measurable" if m["measurable"] else "NOT measurable"
        lines.append(
            f"* target Pfa {m['target_pfa_per_cell']:.0e}: "
            f"{m['eligible_cells_per_frame']} eligible cells x {m['frames']} frames = "
            f"{m['cell_trials']:,} cell trials -> {m['expected_false_alarms']:.1f} "
            f"expected false alarms, **{verdict}** "
            f"(10 events needs {m['frames_for_ten_expected_events']:,} frames)."
        )
    lines.append("")

    for scene_kind in ("noise_only", "clutter_only"):
        curves = [c for c in cal["curves"] if c["scene_kind"] == scene_kind]
        if not curves:
            continue
        lines.append(f"Scene kind `{scene_kind}` (source scenario "
                     f"`{cal['scene_sources'][scene_kind]}`):")
        lines.append("")
        lines.append(
            "| detector | knob | threshold (dB) | knob value | false alarms | "
            "measured Pfa /cell |"
        )
        lines.append("|---|---|---|---|---|---|")
        for curve in curves:
            for point in curve["curve"]:
                tag = " (as shipped)" if point["is_shipped_setting"] else ""
                lines.append(
                    "| `{d}` | `{k}` | {thr:.2f}{tag} | {nv:.6g} | {fa} | {pfa} |".format(
                        d=curve["detector"],
                        k=curve["knob"],
                        thr=point["effective_threshold_db"],
                        tag=tag,
                        nv=point["native_value"],
                        fa=point["false_alarms"],
                        pfa=_pfa(point["measured_pfa_per_cell"]),
                    )
                )
        lines.append("")

    lines.append("Solved operating points:")
    lines.append("")
    lines.append(
        "| detector | scene kind | target Pfa | solved threshold (dB) | knob value | status |"
    )
    lines.append("|---|---|---|---|---|---|")
    for curve in cal["curves"]:
        for solution in curve["solutions"]:
            lines.append(
                "| `{d}` | {sk} | {t:.0e} | {thr} | {nv} | {st} |".format(
                    d=curve["detector"],
                    sk=curve["scene_kind"],
                    t=solution["target_pfa_per_cell"],
                    thr=_fmt(solution["effective_threshold_db"], 3),
                    nv=_fmt(solution["native_value"], 6),
                    st=solution["status"],
                )
            )
    lines.append("")
    lines.append(cal["applied_note"])
    lines.append("")

    # ---- headline calibrated tables ---------------------------------------
    calibrated = sweep["calibrated_variant"]
    lines.append(
        f"### Stage 2a (headline): all detectors at the calibrated common operating "
        f"point, Pfa = {cal['headline_target_pfa']:.0e} /cell"
    )
    lines.append("")
    applied = cal["applied"]
    lines.append(
        "Thresholds in use: "
        + "; ".join(
            f"`{name}` {info['knob']}={_fmt(info['native_value'], 6)} "
            f"({_fmt(info['effective_threshold_db'], 2)} dB)"
            for name, info in sorted(applied.items())
        )
        + "."
    )
    lines.append("")
    for clutter_mode in sweep["scenarios"]:
        detail = report["scenario_details"][clutter_mode]
        lines.append(f"#### `{clutter_mode}` -- {_scenario_caption(detail, sweep)}")
        lines.append("")
        lines.extend(
            _metric_table(
                report,
                [
                    r
                    for r in report["rows"]
                    if r["clutter"] == clutter_mode and r["variant"] == calibrated
                ],
            )
        )
        lines.append("")

    # ---- as-shipped tables ------------------------------------------------
    lines.append("### Stage 2b: the same sweep with the thresholds as configured in the repo")
    lines.append("")
    lines.append(
        "Kept because it is what a user of this repository actually gets. It is **not** "
        "a detector comparison: the three effective thresholds are "
        + ", ".join(
            f"`{m['detector']}` {_fmt(m['effective_threshold_db'], 2)} dB"
            for m in sorted(
                (m for m in report["detectors"].values() if m["variant"] == AS_SHIPPED),
                key=lambda m: m["detector"],
            )
        )
        + ", so the Pd order mostly follows the threshold order."
    )
    lines.append("")
    for clutter_mode in sweep["scenarios"]:
        detail = report["scenario_details"][clutter_mode]
        lines.append(f"#### `{clutter_mode}` -- {_scenario_caption(detail, sweep)}")
        lines.append("")
        lines.extend(
            _metric_table(
                report,
                [
                    r
                    for r in report["rows"]
                    if r["clutter"] == clutter_mode and r["variant"] == AS_SHIPPED
                ],
            )
        )
        lines.append("")

    # ---- ROC ---------------------------------------------------------------
    if report["roc"]:
        lines.append(
            "### Stage 3: ROC -- Pd vs *measured* noise-only Pfa, scenario "
            f"`{sweep['roc_scenario']}`"
        )
        lines.append("")
        lines.append(
            "Each row is one point of the threshold grid. The Pfa column is the density "
            "measured on the target-free frames of stage 1, not a design parameter."
        )
        lines.append("")
        for snr_db in sweep["roc_snr_db"]:
            subset = [r for r in report["roc"] if r["snr_db"] == snr_db]
            if not subset:
                continue
            lines.append(
                f"At input SNR {snr_db:+.0f} dB (target-bin SNR "
                f"{_fmt(subset[0]['target_bin_snr_db'], 1)} dB):"
            )
            lines.append("")
            lines.append(
                "| detector | threshold (dB) | knob value | measured Pfa /cell | Pd | "
                "FA/frame on target frames | of which near the target |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for r in sorted(subset, key=lambda r: (r["detector"], r["effective_threshold_db"])):
                lines.append(
                    "| `{d}` | {thr:.2f} | {nv:.6g} | {pfa} | {pd} | {fa} | {near} |".format(
                        d=r["detector"],
                        thr=r["effective_threshold_db"],
                        nv=r["knob_value"],
                        pfa=_pfa(r["measured_pfa_noise_only"]),
                        pd=_fmt(r["pd"], 3),
                        fa=_fmt(r["false_alarms_per_frame_on_target_frames"], 2),
                        near=_fmt(r["false_alarms_near_target_per_frame"], 2),
                    )
                )
            lines.append("")

    # ---- sidelobe accounting ----------------------------------------------
    lines.extend(_sidelobe_section(report))

    timing = report["timing_s"]
    lines.append(
        f"Runtime: {timing['wall_clock_total']:.1f} s wall clock for "
        f"{timing['frames_simulated']} swept frames plus "
        f"{report['calibration']['frames']} calibration frames per scene kind, "
        f"{timing['detector_calls']} detector calls "
        f"({timing['simulation']:.1f} s simulation, "
        + ", ".join(f"{k} {v:.1f} s" for k, v in sorted(timing["detectors"].items()))
        + ")."
    )
    lines.append("")
    env = report["environment"]
    prov = report["provenance"]
    lines.append(
        f"Environment: Python {env['python']}, numpy {env['numpy']}, scipy "
        f"{env['scipy']}, matplotlib {env['matplotlib']}, torch "
        f"{env['torch'] or 'not installed'}, tqdm {env['tqdm'] or 'not installed'}, "
        f"{env['platform']}, device {env['device']}."
    )
    lines.append("")
    lines.append(
        f"Provenance: generated from commit `{(prov['head_commit'] or 'unknown')[:12]}` "
        f"on branch `{prov['branch']}`, worktree "
        f"{'DIRTY (tracked diff sha256 ' + (prov['tracked_diff_sha256'] or '?')[:12] + ')' if prov['worktree_dirty'] else 'clean'}. "
        "That commit is the parent of the commit carrying this file, because the "
        "results are generated before they are committed."
    )
    lines.append("")
    return "\n".join(lines)


def _scenario_caption(detail: dict, sweep: dict) -> str:
    ratio = detail["clutter_to_target_power_db"]
    if ratio is None or ratio == float("-inf"):
        text = "clutter disabled"
    else:
        text = f"clutter/target RCS power {ratio:+.1f} dB"
    offset = detail["scene_to_target_power_db"]["mean"]
    if sweep["snr_reference"] == "target":
        extra = (
            f"; noise referenced to target-only power, which required "
            f"{offset:+.2f} dB mean correction to the requested SNR"
        )
    else:
        extra = (
            f"; noise referenced to whole-scene power, i.e. the target sits "
            f"{offset:+.2f} dB (mean) below the nominal SNR"
        )
    return (
        f"`apply_realistic_effects={detail['apply_realistic_effects']}`, "
        f"`clutter_intensity={detail['clutter_intensity']:g}` -- {text}{extra}."
    )


def _sidelobe_section(report: dict) -> list[str]:
    assoc = report["association"]
    sweep = report["sweep"]
    top_snr = max(sweep["snr_db"])
    lines = [
        "### How much of the measured false-alarm rate is the target's own sidelobes",
        "",
        ("Every in-gate detection beyond the one matched pair counts as a false positive, "
         "so a target sidelobe that survives non-maximum suppression is reported as a "
         f"false alarm. Split at the highest SNR point ({top_snr:+.0f} dB) using a wider "
         f"+/-{assoc['near_range_bins']} range bin, +/-{assoc['near_doppler_bins']} "
         "Doppler bin neighbourhood (which changes no count -- it only labels them):"),
        "",
        ("| detector | variant | scenario | FP total | FP near a target | share | "
         "FP/frame far from any target |"),
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        (r for r in report["rows"] if r["snr_db"] == top_snr),
        key=lambda r: (r["variant"], r["detector"], r["clutter"]),
    ):
        total = r["false_positives"]
        near = r["false_positives_near_target"]
        share = f"{100.0 * near / total:.1f}%" if total else "n/a"
        lines.append(
            f"| `{r['detector']}` | {r['variant']} | {r['clutter']} | {total} | {near} | "
            f"{share} | {r['false_positives_far_from_target'] / r['frames']:.2f} |"
        )
    lines.append("")
    return lines


def write_figures(report: dict, outdir: str) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures", file=sys.stderr)
        return []

    written = []
    rows = report["rows"]
    modes = report["sweep"]["scenarios"]
    calibrated = report["sweep"]["calibrated_variant"]
    styles = {"clutter_off": "-o", "clutter_default": "--s", "clutter_strong": ":^"}

    specs = [
        ("pd", "Probability of detection", "pd_vs_snr", (-0.05, 1.05)),
        ("false_alarms_per_frame", "False alarms per frame", "fa_per_frame_vs_snr", None),
        ("range_rmse_m", "Range RMSE (m)", "range_rmse_vs_snr", None),
        ("velocity_rmse_mps", "Velocity RMSE (m/s)", "velocity_rmse_vs_snr", None),
    ]
    for variant, suffix in ((calibrated, "calibrated"), (AS_SHIPPED, "as_shipped")):
        for key, ylabel, stem, ylim in specs:
            fig, ax = plt.subplots(figsize=(7.0, 4.2))
            for det in sorted({r["detector"] for r in rows}):
                for mode in modes:
                    sel = [
                        r
                        for r in rows
                        if r["detector"] == det
                        and r["clutter"] == mode
                        and r["variant"] == variant
                    ]
                    sel.sort(key=lambda r: r["snr_db"])
                    if not sel:
                        continue
                    ax.plot(
                        [r["snr_db"] for r in sel],
                        [r[key] for r in sel],
                        styles.get(mode, "-o"),
                        markersize=4,
                        label=f"{det} / {mode}",
                    )
            ax.set_xlabel(
                f"input SNR (dB) passed to simulate_fmcw_signal "
                f"({report['sweep']['snr_reference']}-referenced)"
            )
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            if key == "range_rmse_m":
                ax.axhline(
                    report["quantization_floor"]["range_rmse_m"],
                    color="gray",
                    lw=0.8,
                    ls=":",
                    label="quantization floor",
                )
            if key == "velocity_rmse_mps":
                ax.axhline(
                    report["quantization_floor"]["velocity_rmse_mps"],
                    color="gray",
                    lw=0.8,
                    ls=":",
                    label="quantization floor",
                )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            ax.set_title(f"{ylabel} vs input SNR - {variant}")
            fig.tight_layout()
            path = os.path.join(outdir, f"{stem}_{suffix}.png")
            fig.savefig(path, dpi=140)
            plt.close(fig)
            written.append(path)

    # calibration curves
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for curve in report["calibration"]["curves"]:
        pts = [p for p in curve["curve"] if not p["is_shipped_setting"] and p["measured_pfa_per_cell"] > 0]
        if not pts:
            continue
        ax.semilogy(
            [p["effective_threshold_db"] for p in pts],
            [p["measured_pfa_per_cell"] for p in pts],
            "-o" if curve["scene_kind"] == "noise_only" else "--s",
            markersize=4,
            label=f"{curve['detector']} / {curve['scene_kind']}",
        )
    for m in report["calibration"]["measurability"]:
        ax.axhline(m["target_pfa_per_cell"], color="gray", lw=0.8, ls=":")
    ax.set_xlabel("effective threshold above local mean noise power (dB)")
    ax.set_ylabel("measured false-alarm density per eligible cell")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("Calibration curves (target-free scenes)")
    fig.tight_layout()
    path = os.path.join(outdir, "calibration_curves.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # ROC
    if report["roc"]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for det in sorted({r["detector"] for r in report["roc"]}):
            for snr_db in report["sweep"]["roc_snr_db"]:
                sel = [
                    r
                    for r in report["roc"]
                    if r["detector"] == det
                    and r["snr_db"] == snr_db
                    and r["measured_pfa_noise_only"]
                ]
                sel.sort(key=lambda r: r["measured_pfa_noise_only"])
                if not sel:
                    continue
                ax.semilogx(
                    [r["measured_pfa_noise_only"] for r in sel],
                    [r["pd"] for r in sel],
                    "-o",
                    markersize=4,
                    label=f"{det} @ {snr_db:+.0f} dB",
                )
        ax.set_xlabel("measured noise-only false-alarm density per cell")
        ax.set_ylabel("Probability of detection")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
        ax.set_title("ROC at fixed SNR - scenario " + str(report["sweep"]["roc_scenario"]))
        fig.tight_layout()
        path = os.path.join(outdir, "roc_pd_vs_measured_pfa.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    return written


def update_readme(report: dict, readme_path: str = README_PATH) -> bool:
    if not os.path.exists(readme_path):
        print(f"README not found at {readme_path}; skipping injection", file=sys.stderr)
        return False
    with open(readme_path, encoding="utf-8") as fh:
        text = fh.read()
    if TABLE_BEGIN not in text or TABLE_END not in text:
        print(
            f"README missing {TABLE_BEGIN}/{TABLE_END} markers; skipping injection",
            file=sys.stderr,
        )
        return False
    head, rest = text.split(TABLE_BEGIN, 1)
    _, tail = rest.split(TABLE_END, 1)
    new = f"{head}{TABLE_BEGIN}\n\n{render_markdown(report)}\n{TABLE_END}{tail}"
    with open(readme_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    return True


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_benchmark",
        description=(
            "CPU-only detection benchmark for the classical CFAR detectors that ship "
            "in the AIsensing AIRadar code. Torch is not required."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config_phaser",
        help="RADAR_CONFIGS key from AIRadar/AIradar_datasetv8.py (FMCW configs only)",
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=None,
        dest="snr_db",
        help=f"input SNR sweep points in dB (default: {DEFAULT_SNR_DB})",
    )
    parser.add_argument("--trials", type=int, default=16, help="frames per SNR point")
    parser.add_argument("--seed", type=int, default=20260819, help="base seed")
    parser.add_argument(
        "--snr-reference",
        choices=("target", "scene"),
        default="target",
        help=(
            "'target': add the measured scene/target power ratio to the requested SNR so "
            "the axis is the primary target's SNR in every scenario. 'scene': the repo's "
            "own convention, where clutter also raises the injected noise."
        ),
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=1,
        help="max ground-truth targets per frame (1 keeps the SNR axis unambiguous)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=list(DEFAULT_SCENARIOS),
        help="scenario axis; see benchmarks.run_benchmark.SCENARIOS",
    )
    parser.add_argument(
        "--strong-clutter-intensity",
        type=float,
        default=SCENARIOS["clutter_strong"][1],
        help="clutter RCS scaling used by the clutter_strong scenario",
    )
    parser.add_argument("--zero-pad-factor", type=int, default=2, help="range FFT zero padding")
    parser.add_argument(
        "--gate-range-bins", type=int, default=2, help="association gate, range bins"
    )
    parser.add_argument(
        "--gate-doppler-bins", type=int, default=1, help="association gate, Doppler bins"
    )
    parser.add_argument(
        "--near-range-bins",
        type=int,
        default=8,
        help="wider neighbourhood used only to label sidelobe false alarms",
    )
    parser.add_argument(
        "--near-doppler-bins",
        type=int,
        default=3,
        help="wider neighbourhood used only to label sidelobe false alarms",
    )
    parser.add_argument(
        "--threshold-grid-db",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_GRID_DB),
        help="calibration grid, dB above the local mean noise power",
    )
    parser.add_argument(
        "--target-pfa",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_PFA),
        help="false-alarm densities to solve for; the last one is the headline",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=16,
        help="target-free frames per scene kind used to measure the Pfa curves",
    )
    parser.add_argument(
        "--calibration-snr-db",
        type=float,
        default=-30.0,
        help="SNR of the frames the calibration residuals come from (scale invariant)",
    )
    parser.add_argument(
        "--calibration-snr-index",
        type=int,
        default=1000,
        help=(
            "noise-seed index for the calibration frames. Frames are seeded on "
            "(base_seed, 2, trial, snr_index), so this MUST NOT fall inside the "
            "evaluation range 0..len(--snr-db)-1: sharing an index would calibrate "
            "the thresholds on the very noise realisation the sweep then measures."
        ),
    )
    parser.add_argument(
        "--roc-snr-db",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROC_SNR_DB),
        help="SNR points at which the whole threshold grid is swept for a ROC",
    )
    parser.add_argument(
        "--mtd",
        action="store_true",
        help="enable the moving-target filter of _cfar_2d_custom (drops |v| < 1 m/s)",
    )
    parser.add_argument("--outdir", default=RESULTS_DIR, help="results directory")
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="inject the generated tables into benchmarks/README.md",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="3 SNR points, 3 trials, 4 calibration frames: a smoke run, not a result",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-point progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.snr_db = args.snr_db or [-36.0, -30.0, -24.0]
        args.trials = min(args.trials, 3)
        args.calibration_frames = min(args.calibration_frames, 4)
        args.threshold_grid_db = [6.0, 9.0, 12.0]
        args.roc_snr_db = [-30.0]
    report = run_sweep(args)
    written = write_outputs(report, args.outdir, make_figures=not args.no_figures)
    if args.update_readme:
        update_readme(report)
    if not args.quiet:
        print("\n" + render_markdown(report))
        print("wrote:")
        for path in written:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
