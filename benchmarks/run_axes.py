"""Axis study entry point: what moves the detection curve, at equal false alarms.

    python -m benchmarks.run_axes                    # the full study
    python -m benchmarks.run_axes --quick            # smoke run, not a result
    python -m benchmarks.run_axes --update-readme    # refresh the README tables

Every configuration is re-calibrated to the same measured false-alarm rate per
frame before its Pd is read -- see :mod:`benchmarks.axes` for why that is a rate
per frame rather than per cell, and :mod:`benchmarks.sensitivity` for how a Pd
curve is turned into a dB of sensitivity.

Outputs land in ``benchmarks/results_axes/``.  The measurement content and the
run metadata are written to **separate** files: ``axes_results.json`` contains
only what the seeds determine, so two runs of this script must produce
byte-identical copies of it, while wall-clock timings and the environment go to
``axes_run_meta.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchmarks.axes import (
    AXES,
    BASELINE,
    Axis,
    Knobs,
    build_params,
    measure_config,
    measure_worker,
    register_method_specs,
    resolve_gates,
    unique_configs,
)
from benchmarks.detectors import build_detector
from benchmarks.repo_shim import load_repo_modules
from benchmarks.run_benchmark import (
    environment_report,
    provenance_report,
    sanitize_for_json,
)
from benchmarks.sensitivity import (
    CONFIDENCE,
    binomial_se,
    bootstrap_pd_difference,
    bootstrap_shift,
    frames_to_resolve,
    resample_indices,
    snr_at_pd,
)

#: 1.5 dB steps through the whole transition region, wide enough that a
#: configuration gaining or losing 6 dB still has its Pd = 0.5 crossing on the
#: grid.  The first study used 3 dB steps, which is too coarse to read a shift.
DEFAULT_SNR_DB = [
    -45.0, -43.5, -42.0, -40.5, -39.0, -37.5, -36.0, -34.5, -33.0, -31.5, -30.0, -28.5, -27.0,
]

#: Calibration grid, dB above the local mean noise power.
DEFAULT_THRESHOLD_GRID_DB = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

#: Detection probability the SNR axis is interpolated to.
DEFAULT_PD_LEVEL = 0.5

#: Per-cell density the common operating point is derived from, for continuity
#: with the first study's headline point.  It is converted to a rate per frame on
#: the baseline grid and *that* rate is what every configuration is calibrated to.
BASELINE_PFA_PER_CELL = 1e-4

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_axes")
README_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
TABLE_BEGIN = "<!-- BEGIN GENERATED AXES -->"
TABLE_END = "<!-- END GENERATED AXES -->"


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def baseline_reference(args) -> dict:
    """Grid, gate and eligible-cell count of the baseline configuration.

    The common operating point is ``BASELINE_PFA_PER_CELL`` times this cell count,
    expressed as a rate per frame, so it has to be resolved before any
    configuration runs.
    """
    register_method_specs()
    repo = load_repo_modules()
    configs = repo["RADAR_CONFIGS"]
    if args.config not in configs:
        raise SystemExit(f"unknown config {args.config!r}; available: {sorted(configs)}")
    params = build_params(configs[args.config], BASELINE)
    detector = build_detector(params, BASELINE.detector, 15.0, "reference")
    gate_r, gate_d = resolve_gates(
        params,
        args.gate_range_m if args.gate_range_m else 2 * params.range_bin_spacing,
        args.gate_velocity_mps if args.gate_velocity_mps else params.velocity_bin_spacing,
    )
    return {
        "config": args.config,
        "config_name": params.config.get("name"),
        "eligible_cells_per_frame": detector.eligible_cells(params),
        "num_doppler_bins": params.num_doppler_bins,
        "num_range_bins": params.num_range_bins,
        "range_bin_spacing_m": params.range_bin_spacing,
        "velocity_bin_spacing_mps": params.velocity_bin_spacing,
        "gate_range_m": gate_r * params.range_bin_spacing,
        "gate_velocity_mps": gate_d * params.velocity_bin_spacing,
        "bandwidth_hz": params.B,
        "chirp_duration_s": params.T,
        "samples_per_chirp": params.Ns,
    }


def _cost_hint(knobs: Knobs) -> float:
    """Rough relative cost, used only to start the slow configurations first."""
    window = (2 * (knobs.num_train + knobs.num_guard) + 1) ** 2
    base_window = (2 * (BASELINE.num_train + BASELINE.num_guard) + 1) ** 2
    return (
        (window / base_window)
        * (knobs.zero_pad_factor / BASELINE.zero_pad_factor)
        * (knobs.num_chirps / BASELINE.num_chirps)
        * (1.0 + 0.15 * (knobs.looks - 1))
    )


def run_configurations(args, settings: dict, quiet: bool) -> dict[str, dict]:
    configs = unique_configs(_selected_axes(args))
    order = sorted(configs, key=lambda k: -_cost_hint(configs[k]))
    results: dict[str, dict] = {}
    started = time.perf_counter()

    if args.workers <= 1:
        for key in order:
            results[key] = measure_config(configs[key], settings)
            if not quiet:
                _progress(results[key], len(results), len(order), started)
    else:
        payloads = [
            {"knobs": configs[key].as_dict(), "settings": settings} for key in order
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(measure_worker, payloads):
                results[result["key"]] = result
                if not quiet:
                    _progress(result, len(results), len(order), started)

    return {key: results[key] for key in configs}


def _progress(result: dict, done: int, total: int, started: float) -> None:
    pds = [row["pd"] for row in result["rows"]]
    print(
        f"[{done:2d}/{total}] {result['key']}  "
        f"thr={result['calibration']['solved_effective_threshold_db']:.2f} dB  "
        f"Pd={' '.join(f'{p:.2f}' for p in pds)}  "
        f"({result['timing_s']['total']:.0f} s, elapsed {time.perf_counter() - started:.0f} s)",
        flush=True,
    )


def _selected_axes(args) -> tuple[Axis, ...]:
    if not args.axes:
        return AXES
    wanted = set(args.axes)
    unknown = wanted - {a.name for a in AXES}
    if unknown:
        raise SystemExit(f"unknown axis {sorted(unknown)}; available: {[a.name for a in AXES]}")
    return tuple(a for a in AXES if a.name in wanted)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #


def fa_rate_error_db(result: dict, measured_fa_per_frame: float) -> float | None:
    """How far the realised false-alarm rate sits from the common one, in dB of threshold.

    The operating point is *solved for* on a finite number of target-free frames, so
    the rate a configuration actually realises differs from the target by the
    sampling error of that solve.  Quoting that residual as a percentage of a rate
    is not comparable with anything; converting it through the locally measured
    slope of the calibration curve puts it in the same units as the study's results,
    where it can be read directly against the dB shifts being claimed.

    Positive means the configuration ran *hot* -- more false alarms than the common
    rate, i.e. its threshold sits this many dB too low.
    """
    cal = result["calibration"]
    frames = cal["frames"]
    target = cal["target_fa_per_frame"]
    points = sorted(
        (p["effective_threshold_db"], p["false_alarms"] / frames)
        for p in cal["curve"]
        if p["false_alarms"] > 0
    )
    if len(points) < 2 or measured_fa_per_frame <= 0 or target <= 0:
        return None
    threshold = cal["solved_effective_threshold_db"]
    below = [p for p in points if p[0] <= threshold] or [points[0]]
    above = [p for p in points if p[0] > threshold] or [points[-1]]
    lo, hi = below[-1], above[0]
    if hi[0] == lo[0]:
        return None
    slope = (math.log10(hi[1]) - math.log10(lo[1])) / (hi[0] - lo[0])
    if slope == 0:
        return None
    return -(math.log10(measured_fa_per_frame) - math.log10(target)) / slope


def analyse(args, settings: dict, results: dict[str, dict]) -> dict:
    """Turn per-configuration counts into shifts, differences and verdicts."""
    snr_points = [float(s) for s in settings["snr_db"]]
    trials = int(settings["trials"])
    indices = resample_indices(trials, args.bootstrap_resamples, args.bootstrap_seed)
    level = float(args.pd_level)

    curves = {}
    for key, result in results.items():
        detected = np.asarray(result["detected"], dtype=np.int64)
        targets = np.asarray(result["targets"], dtype=np.int64)
        pd = np.where(targets.sum(axis=1) > 0, detected.sum(axis=1) / np.maximum(targets.sum(axis=1), 1), np.nan)
        curves[key] = {
            "pd": [float(v) for v in pd],
            "detected": detected,
            "targets": targets,
            "crossing": snr_at_pd(snr_points, pd, level),
        }

    selected = _selected_axes(args)
    baseline_key = BASELINE.key
    if baseline_key not in curves:  # a subset of axes that excludes the study baseline
        first = selected[0]
        baseline_key = dict(first.members)[first.baseline_value].key
    baseline_crossing = curves[baseline_key]["crossing"]
    if baseline_crossing.snr_db is None:
        reference_snr = snr_points[len(snr_points) // 2]
    else:
        reference_snr = min(snr_points, key=lambda s: abs(s - baseline_crossing.snr_db))
    reference_index = snr_points.index(reference_snr)

    axes_out = []
    for axis in selected:
        base_knobs = dict(axis.members)[axis.baseline_value]
        base_key = base_knobs.key
        members = []
        for value, knobs in axis.members:
            key = knobs.key
            entry = curves[key]
            shift = bootstrap_shift(
                snr_points,
                curves[base_key]["detected"],
                curves[base_key]["targets"],
                entry["detected"],
                entry["targets"],
                indices,
                level,
            )
            pd_diff = bootstrap_pd_difference(
                curves[base_key]["detected"][reference_index],
                curves[base_key]["targets"][reference_index],
                entry["detected"][reference_index],
                entry["targets"][reference_index],
                indices,
            )
            resolved_shift = (
                shift["shift_db_lo"] is not None
                and shift["undefined_fraction"] < 0.1
                and (shift["shift_db_lo"] > 0 or shift["shift_db_hi"] < 0)
            )
            resolved_pd = pd_diff["pd_difference_lo"] > 0 or pd_diff["pd_difference_hi"] < 0
            needed = frames_to_resolve(
                pd_diff["pd_difference"], pd_diff["pd_difference_se"], trials
            )
            members.append(
                {
                    "value": value,
                    "config": key,
                    "is_axis_baseline": key == base_key,
                    "knobs": knobs.as_dict(),
                    "solved_effective_threshold_db": results[key]["calibration"][
                        "solved_effective_threshold_db"
                    ],
                    "solved_native_value": results[key]["calibration"]["solved_native_value"],
                    "calibration_status": results[key]["calibration"]["status"],
                    "eligible_cells_per_frame": results[key]["calibration"][
                        "eligible_cells_per_frame"
                    ],
                    "measured_fa_per_frame_swept": float(
                        np.mean(
                            [
                                row["false_alarms_per_frame"]
                                for row in results[key]["rows"]
                            ]
                        )
                    ),
                    "measured_fa_per_frame_at_floor": float(
                        results[key]["rows"][0]["false_alarms_per_frame"]
                    ),
                    "fa_rate_error_db": fa_rate_error_db(
                        results[key],
                        float(results[key]["rows"][0]["false_alarms_per_frame"]),
                    ),
                    "pd_curve": entry["pd"],
                    "pd_at_reference": entry["pd"][reference_index],
                    "pd_at_top": entry["pd"][-1],
                    "snr_at_pd": entry["crossing"].as_dict(),
                    "shift_db": shift,
                    "pd_difference": pd_diff,
                    "resolved_shift": bool(resolved_shift),
                    "resolved_pd_difference": bool(resolved_pd),
                    "frames_to_resolve_pd_difference": needed,
                    "range_rmse_m": results[key]["rows"][reference_index]["range_rmse_m"],
                    "velocity_rmse_mps": results[key]["rows"][reference_index][
                        "velocity_rmse_mps"
                    ],
                }
            )
        usable = [
            m
            for m in members
            if not m["is_axis_baseline"] and m["shift_db"]["shift_db_median"] is not None
        ]
        best = max(usable, key=lambda m: m["shift_db"]["shift_db_median"], default=None)
        worst = min(usable, key=lambda m: m["shift_db"]["shift_db_median"], default=None)
        span = (
            best["shift_db"]["shift_db_median"] - worst["shift_db"]["shift_db_median"]
            if best and worst
            else None
        )
        axes_out.append(
            {
                "name": axis.name,
                "knob": axis.knob,
                "question": axis.question,
                "cost_note": axis.cost_note,
                "baseline_value": axis.baseline_value,
                "baseline_config": base_key,
                "members": members,
                "best_value": best["value"] if best else None,
                "best_shift_db": best["shift_db"]["shift_db_median"] if best else None,
                "best_is_resolved": bool(best["resolved_shift"]) if best else False,
                "worst_value": worst["value"] if worst else None,
                "worst_shift_db": worst["shift_db"]["shift_db_median"] if worst else None,
                "worst_is_resolved": bool(worst["resolved_shift"]) if worst else False,
                "span_db": span,
                "any_resolved": any(m["resolved_shift"] for m in members if not m["is_axis_baseline"]),
            }
        )

    floors = [
        m["measured_fa_per_frame_at_floor"] for axis in axes_out for m in axis["members"]
    ]
    errors = [
        abs(m["fa_rate_error_db"])
        for axis in axes_out
        for m in axis["members"]
        if m["fa_rate_error_db"] is not None
    ]
    return {
        "pd_level": level,
        "reference_snr_db": reference_snr,
        "reference_snr_index": reference_index,
        "baseline_config": baseline_key,
        "baseline_crossing": baseline_crossing.as_dict(),
        "false_alarm_audit": {
            "target_fa_per_frame": float(settings["target_fa_per_frame"]),
            "measured_at_sweep_floor": {
                "snr_db": snr_points[0],
                "min": float(np.min(floors)),
                "mean": float(np.mean(floors)),
                "max": float(np.max(floors)),
            },
            "threshold_equivalent_db": {
                "max_abs": float(np.max(errors)) if errors else None,
                "mean_abs": float(np.mean(errors)) if errors else None,
            },
            "note": (
                "measured on the lowest SNR point of the sweep, where every "
                "configuration detects almost nothing, so the count is not reduced by "
                "matched detections being scored as true positives. The residual "
                "spread is the sampling error of a threshold solved on a finite number "
                "of target-free frames; threshold_equivalent_db converts it through the "
                "locally measured slope of each calibration curve so it can be read "
                "against the dB shifts this study reports."
            ),
        },
        "bootstrap": {
            "resamples": int(args.bootstrap_resamples),
            "seed": int(args.bootstrap_seed),
            "confidence": CONFIDENCE,
            "paired": True,
            "note": (
                "resampling is over the physical scenes and is paired: every "
                "configuration is scored on the same resampled scene multiset, so "
                "scene-to-scene difficulty cancels out of the difference."
            ),
        },
        "single_pd_standard_error_at_half": binomial_se(0.5, trials),
        "axes": axes_out,
    }


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

CONFIG_FIELDS = [
    "axis",
    "value",
    "config",
    "is_axis_baseline",
    "detector",
    "num_train",
    "num_guard",
    "nms_kernel_size",
    "zero_pad_factor",
    "num_chirps",
    "looks",
    "mtd",
    "eligible_cells_per_frame",
    "solved_effective_threshold_db",
    "solved_native_value",
    "calibration_status",
    "measured_fa_per_frame_at_floor",
    "measured_fa_per_frame_swept",
    "fa_rate_error_db",
    "pd_at_reference_snr",
    "pd_at_top_snr",
    "pd_difference",
    "pd_difference_lo",
    "pd_difference_hi",
    "snr_at_pd_db",
    "snr_at_pd_status",
    "shift_db_median",
    "shift_db_lo",
    "shift_db_hi",
    "resolved_shift",
    "frames_to_resolve_pd_difference",
    "range_rmse_m",
    "velocity_rmse_mps",
]

SWEEP_FIELDS = [
    "config",
    "snr_db",
    "target_bin_snr_db",
    "mean_peak_snr_db",
    "frames",
    "ground_truth_targets",
    "true_positives",
    "false_positives",
    "false_negatives",
    "pd",
    "false_alarms_per_frame",
    "false_alarm_rate_per_cell",
    "eligible_cells_per_frame",
    "range_rmse_m",
    "velocity_rmse_mps",
]

CALIBRATION_FIELDS = [
    "config",
    "detector",
    "knob",
    "effective_threshold_db",
    "native_value",
    "false_alarms",
    "measured_fa_per_frame",
    "measured_pfa_per_cell",
]


def write_outputs(report: dict, outdir: str, make_figures: bool) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    written = []

    def _csv(name, fields, records):
        path = os.path.join(outdir, name)
        with open(path, "w", newline="\n", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow({k: record.get(k) for k in fields})
        written.append(path)

    config_rows = []
    for axis in report["analysis"]["axes"]:
        for member in axis["members"]:
            config_rows.append(
                {
                    "axis": axis["name"],
                    "value": member["value"],
                    "config": member["config"],
                    "is_axis_baseline": member["is_axis_baseline"],
                    **member["knobs"],
                    "eligible_cells_per_frame": member["eligible_cells_per_frame"],
                    "solved_effective_threshold_db": member["solved_effective_threshold_db"],
                    "solved_native_value": member["solved_native_value"],
                    "calibration_status": member["calibration_status"],
                    "measured_fa_per_frame_at_floor": member["measured_fa_per_frame_at_floor"],
                    "measured_fa_per_frame_swept": member["measured_fa_per_frame_swept"],
                    "fa_rate_error_db": member["fa_rate_error_db"],
                    "pd_at_reference_snr": member["pd_at_reference"],
                    "pd_at_top_snr": member["pd_at_top"],
                    "pd_difference": member["pd_difference"]["pd_difference"],
                    "pd_difference_lo": member["pd_difference"]["pd_difference_lo"],
                    "pd_difference_hi": member["pd_difference"]["pd_difference_hi"],
                    "snr_at_pd_db": member["snr_at_pd"]["snr_db"],
                    "snr_at_pd_status": member["snr_at_pd"]["status"],
                    "shift_db_median": member["shift_db"]["shift_db_median"],
                    "shift_db_lo": member["shift_db"]["shift_db_lo"],
                    "shift_db_hi": member["shift_db"]["shift_db_hi"],
                    "resolved_shift": member["resolved_shift"],
                    "frames_to_resolve_pd_difference": member[
                        "frames_to_resolve_pd_difference"
                    ],
                    "range_rmse_m": member["range_rmse_m"],
                    "velocity_rmse_mps": member["velocity_rmse_mps"],
                }
            )
    _csv("axes_configs.csv", CONFIG_FIELDS, config_rows)

    sweep_rows = [row for result in report["configs"].values() for row in result["rows"]]
    _csv("axes_summary.csv", SWEEP_FIELDS, sweep_rows)

    calibration_rows = []
    for key, result in report["configs"].items():
        frames = result["calibration"]["frames"]
        for point in result["calibration"]["curve"]:
            calibration_rows.append(
                {
                    "config": key,
                    "detector": result["detector"],
                    "knob": result["calibration"]["knob"],
                    "effective_threshold_db": point["effective_threshold_db"],
                    "native_value": point["native_value"],
                    "false_alarms": point["false_alarms"],
                    "measured_fa_per_frame": point["false_alarms"] / frames,
                    "measured_pfa_per_cell": point["measured_pfa_per_cell"],
                }
            )
    _csv("axes_calibration.csv", CALIBRATION_FIELDS, calibration_rows)

    frame_rows = []
    for key, result in report["configs"].items():
        detected = result["detected"]
        targets = result["targets"]
        false_alarms = result["false_alarms"]
        for i, snr_db in enumerate(report["settings"]["snr_db"]):
            for trial in range(len(detected[i])):
                frame_rows.append(
                    {
                        "config": key,
                        "snr_db": snr_db,
                        "trial": trial,
                        "targets": targets[i][trial],
                        "detected": detected[i][trial],
                        "false_alarms": false_alarms[i][trial],
                    }
                )
    _csv(
        "axes_frames.csv",
        ["config", "snr_db", "trial", "targets", "detected", "false_alarms"],
        frame_rows,
    )

    deterministic = {
        "study": report["study"],
        "settings": report["settings"],
        "baseline": report["baseline"],
        "analysis": report["analysis"],
        "configs": {
            key: {k: v for k, v in result.items() if k != "timing_s"}
            for key, result in report["configs"].items()
        },
    }
    path = os.path.join(outdir, "axes_results.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sanitize_for_json(deterministic), fh, indent=2, allow_nan=False, sort_keys=True)
        fh.write("\n")
    written.append(path)

    meta = {
        "timing_s": report["timing_s"],
        "per_config_timing_s": {
            key: result["timing_s"] for key, result in report["configs"].items()
        },
        "environment": environment_report(),
        "provenance": provenance_report(),
        "workers": report["workers"],
    }
    path = os.path.join(outdir, "axes_run_meta.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sanitize_for_json(meta), fh, indent=2, allow_nan=False)
        fh.write("\n")
    written.append(path)

    path = os.path.join(outdir, "axes_table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(report))
    written.append(path)

    if make_figures:
        written.extend(write_figures(report, outdir))
    return written


def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _signed(value, digits=2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.{digits}f}"


def _verdict(member: dict) -> str:
    if member["is_axis_baseline"]:
        return "baseline"
    shift = member["shift_db"]
    if member["snr_at_pd"]["status"] != "interpolated":
        return f"Pd={member['pd_at_reference']:.2f}, crossing {member['snr_at_pd']['status']}"
    if shift["undefined_fraction"] >= 0.1:
        return "crossing off-grid in {:.0%} of resamples".format(shift["undefined_fraction"])
    if member["resolved_shift"]:
        return "gain" if shift["shift_db_median"] > 0 else "loss"
    needed = member["frames_to_resolve_pd_difference"]
    if needed is None:
        return "not resolved (dPd is exactly 0 here)"
    return f"not resolved (needs ~{needed} frames/point)"


def _member(analysis: dict, axis_name: str, value: str) -> dict | None:
    for axis in analysis["axes"]:
        if axis["name"] != axis_name:
            continue
        for member in axis["members"]:
            if member["value"] == value:
                return member
    return None


def _shift_text(member: dict) -> str:
    shift = member["shift_db"]
    if shift["shift_db_median"] is None:
        return "not measurable"
    return (
        f"{_signed(shift['shift_db_median'])} dB "
        f"[{_signed(shift['shift_db_lo'])}, {_signed(shift['shift_db_hi'])}]"
    )


def _headline_lines(report: dict) -> list[str]:
    """The three-sentence version, written from the measurement rather than about it."""
    analysis = report["analysis"]
    settings = report["settings"]
    axes = analysis["axes"]
    moving = [a for a in axes if a["any_resolved"]]
    flat = [a for a in axes if not a["any_resolved"]]
    lines = ["### What the measurement says", ""]

    def names(items):
        return ", ".join(f"`{a['name']}`" for a in items) if items else "none"

    integration = {"coherent_chirps", "noncoherent_looks"}
    moving_names = {a["name"] for a in moving}
    lines.append(
        f"**{len(moving)} of the {len(axes)} axes move the curve** by more than their own "
        f"{CONFIDENCE:.0%} interval ({names(moving)}); the other {len(flat)} do not at this "
        f"budget ({names(flat)})."
    )
    if moving_names and moving_names <= integration:
        lines.append("")
        lines.append(
            "Every axis that moves it is integration -- how much signal is summed before "
            "the detector runs -- and none of them is a property of the detector. Every "
            "knob inside the CFAR itself (window geometry, averaging mode, non-maximum "
            "suppression) is flat once the false-alarm rate is held equal, which extends "
            "the first study's conclusion from the choice of detector to the tuning of it."
        )
    lines.append("")

    coherent = _member(analysis, "coherent_chirps", "128")
    non_coherent = _member(analysis, "noncoherent_looks", "2")
    if coherent and non_coherent:
        lines.append(
            f"**Doubling the dwell time, two ways.** Doubling the chirp count -- coherent "
            f"integration, one detection per frame as before -- buys {_shift_text(coherent)}. "
            f"Spending the same extra dwell on a second look integrated non-coherently buys "
            f"{_shift_text(non_coherent)}."
        )
        a, b = coherent["shift_db"], non_coherent["shift_db"]
        if a["shift_db_median"] is None or b["shift_db_median"] is None:
            verdict = (
                "One of the two has no measurable crossing, so no ranking between them "
                "is claimed."
            )
        elif a["shift_db_lo"] > b["shift_db_hi"]:
            verdict = (
                "The intervals do not overlap: at equal dwell, the coherent route is the "
                "better buy here."
            )
        elif b["shift_db_lo"] > a["shift_db_hi"]:
            verdict = (
                "The intervals do not overlap: at equal dwell, the non-coherent route is "
                "the better buy here."
            )
        else:
            verdict = (
                "The intervals overlap, so this study cannot rank the two at equal dwell; "
                "it can only say that both are worth more than every other axis it swept."
            )
        lines.append("")
        lines.append(
            verdict
            + " The non-coherent route needs no change to the waveform at all -- it is "
            "arithmetic on range-Doppler maps the pipeline already computes -- while the "
            "coherent route changes the Doppler grid and the frame rate."
        )
        lines.append("")

    largest_flat = None
    for axis in flat:
        for member in axis["members"]:
            if member["is_axis_baseline"] or member["shift_db"]["shift_db_median"] is None:
                continue
            if largest_flat is None or abs(member["shift_db"]["shift_db_median"]) > abs(
                largest_flat[1]["shift_db"]["shift_db_median"]
            ):
                largest_flat = (axis, member)
    if largest_flat is not None:
        axis, member = largest_flat
        needed = member["frames_to_resolve_pd_difference"]
        budget = (
            f"resolving a difference that small at this operating point would need about "
            f"{needed} frames per SNR point instead of {settings['trials']}"
            if needed
            else "its Pd difference at the reference point is exactly zero, so no frame "
            "budget resolves it"
        )
        lines.append(
            f"**What \"flat\" means here.** The largest movement anywhere among the flat "
            f"axes is `{axis['knob']}` = {member['value']} at {_shift_text(member)}, an "
            f"interval that spans zero; {budget}. Read those rows as \"this study cannot "
            f"distinguish these settings\", not as \"these settings are identical\" -- the "
            f"upper end of each interval is what an unmeasured effect could still be worth."
        )
        lines.append("")

    mtd = _member(analysis, "mtd_filter", "on")
    mtd_off = _member(analysis, "mtd_filter", "off")
    if mtd is not None and mtd_off is not None:
        slow = report["study"].get("slow_targets_below_1mps")
        ceiling = mtd_off["pd_at_top"] - mtd["pd_at_top"]
        text = (
            f"**The moving-target filter.** `mtd=True` -- which the dataset pipeline "
            f"enables whenever `apply_realistic_effects` is set -- moves the Pd = "
            f"{analysis['pd_level']:.2f} crossing by {_shift_text(mtd)}"
        )
        if ceiling > 0:
            text += (
                f", and caps Pd at {mtd['pd_at_top']:.3f} at the top of the sweep against "
                f"{mtd_off['pd_at_top']:.3f} without it, because {slow} of the "
                f"{settings['trials']} drawn targets are slower than the 1 m/s it discards. "
                f"It buys no sensitivity at this operating point and costs every slow target."
            )
        else:
            text += (
                f", and does not lower the Pd ceiling either -- but only because {slow} of "
                f"the {settings['trials']} drawn targets are slower than the 1 m/s it "
                f"discards. That is a property of this scene draw, not of the filter."
            )
        lines.append(text)
        lines.append("")

    nms_off = _member(analysis, "nms_kernel", "1")
    nms_base = _member(analysis, "nms_kernel", "5")
    if nms_off is not None and nms_base is not None:
        extra = (
            nms_off["measured_fa_per_frame_swept"] - nms_base["measured_fa_per_frame_swept"]
        )
        moved = "does not move" if not nms_off["resolved_shift"] else "moves"
        lines.append(
            f"**Non-maximum suppression is not about detection, it is about counting.** "
            f"Disabling it (`nms_kernel_size=1`) {moved} the Pd curve "
            f"({_shift_text(nms_off)}), but over the swept frames it produces "
            f"{extra:+.2f} false alarms per frame relative to the shipped kernel at the same "
            f"target-free rate -- the target's own main lobe, reported as several detections "
            f"instead of one."
        )
        lines.append("")
    return lines


def render_markdown(report: dict) -> str:
    settings = report["settings"]
    baseline = report["baseline"]
    analysis = report["analysis"]
    lines: list[str] = []

    lines.append(
        f"Radar config `{baseline['config']}` ({baseline['config_name']}): "
        f"baseline RD map {baseline['num_doppler_bins']} x {baseline['num_range_bins']} bins, "
        f"range bin {baseline['range_bin_spacing_m']:.3f} m, "
        f"velocity bin {baseline['velocity_bin_spacing_mps']:.3f} m/s."
    )
    lines.append("")
    lines.append(
        f"{settings['trials']} frames per SNR point, 1 target per frame, base seed "
        f"{settings['seed']}, SNR referenced to **target** power, clutter disabled. "
        f"Association gate held fixed in physical units at "
        f"+/-{baseline['gate_range_m']:.3f} m and "
        f"+/-{baseline['gate_velocity_mps']:.3f} m/s and converted to bins per "
        f"configuration, because two axes change the bin grid."
    )
    lines.append("")
    lines.append(
        f"**The common operating point.** Every configuration below is calibrated on "
        f"its own target-free frames to the same measured rate of "
        f"**{settings['target_fa_per_frame']:.3f} false alarms per frame** "
        f"(= {BASELINE_PFA_PER_CELL:.0e} per cell on the baseline "
        f"{baseline['num_doppler_bins']} x {baseline['num_range_bins']} grid, which is the "
        f"first study's headline point). A rate per frame rather than per cell, because "
        f"the zero-padding and chirp-count axes change the number of cells while covering "
        f"the same physical volume; per-cell equality would hand the finer grid more false "
        f"alarms per frame. {settings['calibration_frames']} target-free frames per "
        f"configuration give {settings['target_fa_per_frame'] * settings['calibration_frames']:.0f} "
        f"expected false-alarm events at that rate."
    )
    lines.append("")
    audit = analysis["false_alarm_audit"]
    lines.append(
        f"**Did the calibration hold?** At the bottom of the sweep "
        f"({audit['measured_at_sweep_floor']['snr_db']:+.1f} dB), where nothing is being "
        f"detected and the false-alarm count is not reduced by matched detections, the "
        f"configurations realise "
        f"{audit['measured_at_sweep_floor']['min']:.2f} to "
        f"{audit['measured_at_sweep_floor']['max']:.2f} false alarms per frame "
        f"(mean {audit['measured_at_sweep_floor']['mean']:.2f}) against the "
        f"{audit['target_fa_per_frame']:.2f} they were solved for. That residual is the "
        f"sampling error of solving a threshold on "
        f"{settings['calibration_frames']} target-free frames; converted through each "
        f"calibration curve's own slope it is at most "
        f"{audit['threshold_equivalent_db']['max_abs']:.2f} dB of threshold "
        f"(mean {audit['threshold_equivalent_db']['mean_abs']:.2f} dB), which is the "
        f"resolution at which any claim below should be read."
    )
    lines.append("")
    lines.append(
        f"**How to read the result columns.** `FA/frame at floor` is the check above, per "
        f"row. `Pd@{analysis['reference_snr_db']:+.1f} dB` is the detection rate at the "
        f"fixed SNR point nearest the baseline's Pd = {analysis['pd_level']:.2f} crossing, "
        f"and `Pd@{settings['snr_db'][-1]:+.1f} dB` is the top of the sweep, where a "
        f"configuration that throws targets away shows a ceiling below 1. `shift` is the dB "
        f"of sensitivity the configuration buys: the SNR at which the baseline reaches "
        f"Pd = {analysis['pd_level']:.2f} minus the SNR at which this configuration reaches "
        f"it, so **positive means it detects the same target further down**. Intervals are "
        f"{CONFIDENCE:.0%} paired-bootstrap intervals over "
        f"{analysis['bootstrap']['resamples']} resamples of the "
        f"{settings['trials']} scenes."
    )
    lines.append("")
    lines.extend(_headline_lines(report))
    lines.append("### Summary: which axis moves the curve")
    lines.append("")
    lines.append(
        "Ranked by **span**: the dB between the best and the worst value swept on that "
        "axis. A large span means the knob matters -- whether or not the shipped value "
        "can be improved on, getting it wrong costs that many dB. All shifts are "
        "relative to the axis's own baseline value."
    )
    lines.append("")
    lines.append(
        "| axis | knob | baseline | best value (shift dB) | worst value (shift dB) | "
        "span (dB) | any shift resolved? | cost |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    ranked = sorted(
        analysis["axes"],
        key=lambda a: -(a["span_db"] if a["span_db"] is not None else -99),
    )
    for axis in ranked:
        lines.append(
            f"| `{axis['name']}` | `{axis['knob']}` | {axis['baseline_value']} | "
            f"{axis['best_value']} ({_signed(axis['best_shift_db'])}) | "
            f"{axis['worst_value']} ({_signed(axis['worst_shift_db'])}) | "
            f"{_fmt(axis['span_db'], 2)} | "
            f"{'yes' if axis['any_resolved'] else 'no'} | {axis['cost_note']} |"
        )
    lines.append("")

    for axis in analysis["axes"]:
        lines.append(f"### `{axis['name']}` -- `{axis['knob']}`")
        lines.append("")
        lines.append(axis["question"])
        lines.append("")
        lines.append(
            f"Cost: {axis['cost_note']}. Baseline value: `{axis['baseline_value']}`."
        )
        lines.append("")
        lines.append(
            f"| {axis['knob']} | calibrated thr (dB) | FA/frame at floor | "
            f"Pd@{analysis['reference_snr_db']:+.1f} dB | dPd [{CONFIDENCE:.0%} CI] | "
            f"Pd@{settings['snr_db'][-1]:+.1f} dB | "
            f"SNR at Pd={analysis['pd_level']:.2f} (dB) | shift (dB) [{CONFIDENCE:.0%} CI] | verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for member in axis["members"]:
            pd_diff = member["pd_difference"]
            shift = member["shift_db"]
            crossing = member["snr_at_pd"]
            lines.append(
                "| {value} | {thr} | {fa} | {pd} | {dpd} | {top} | {snr} | {shift} | {verdict} |".format(
                    value=f"**{member['value']}**" if member["is_axis_baseline"] else member["value"],
                    thr=_fmt(member["solved_effective_threshold_db"], 2),
                    fa=_fmt(member["measured_fa_per_frame_at_floor"], 2),
                    pd=_fmt(member["pd_at_reference"], 3),
                    top=_fmt(member["pd_at_top"], 3),
                    dpd=(
                        "--"
                        if member["is_axis_baseline"]
                        else f"{pd_diff['pd_difference']:+.3f} "
                        f"[{pd_diff['pd_difference_lo']:+.3f}, {pd_diff['pd_difference_hi']:+.3f}]"
                    ),
                    snr=(
                        _fmt(crossing["snr_db"], 2)
                        if crossing["snr_db"] is not None
                        else crossing["status"]
                    ),
                    shift=(
                        "--"
                        if member["is_axis_baseline"]
                        else (
                            f"{_signed(shift['shift_db_median'])} "
                            f"[{_signed(shift['shift_db_lo'])}, {_signed(shift['shift_db_hi'])}]"
                            if shift["shift_db_median"] is not None
                            else "n/a"
                        )
                    ),
                    verdict=_verdict(member),
                )
            )
        lines.append("")

    lines.append("### Calibrated thresholds and what they cost")
    lines.append("")
    lines.append(
        "Every row is a solve on target-free frames, so the threshold column is a "
        "*result*, not a setting. A configuration that produces fewer peaks per frame "
        "for the same threshold gets its threshold lowered until the rate matches, "
        "which is where part of its gain comes from."
    )
    lines.append("")
    lines.append("| config | detector | knob | calibrated knob value | effective thr (dB) | status |")
    lines.append("|---|---|---|---|---|---|")
    for key, result in report["configs"].items():
        cal = result["calibration"]
        lines.append(
            f"| `{key}` | `{result['detector']}` | `{cal['knob']}` | "
            f"{_fmt(cal['solved_native_value'], 5)} | "
            f"{_fmt(cal['solved_effective_threshold_db'], 2)} | {cal['status']} |"
        )
    lines.append("")

    se = analysis["single_pd_standard_error_at_half"]
    lines.append("### The error bar")
    lines.append("")
    lines.append(
        f"A Pd measured over {settings['trials']} frames with one target each has a "
        f"binomial standard error of {se:.3f} at Pd = 0.5, and is quantized to "
        f"1/{settings['trials']} = {1 / settings['trials']:.4f}. That is the error bar on an "
        f"**absolute** Pd, and it is why no absolute Pd in this study should be read to "
        f"better than about +/-{2 * se:.2f}."
    )
    lines.append("")
    lines.append(
        "The differences are better resolved than that, because they are paired: every "
        "configuration is measured on the same physical scenes and the same noise draws, "
        "so scene difficulty cancels. The bracketed intervals are the bootstrap of that "
        "paired difference, and they are the numbers to read. Where an interval spans "
        "zero the study **cannot resolve that axis** at this budget, and the verdict "
        "column states the per-point frame budget that would."
    )
    lines.append("")
    lines.append(
        f"Total: {report['study']['configurations']} configurations, "
        f"{report['study']['detector_calls']} detector calls."
    )
    lines.append("")
    lines.append(
        "Provenance: everything above depends only on the seeds. Wall-clock timings, "
        "the environment and the git state are deliberately kept out of this block and "
        "written to `results_axes/axes_run_meta.json` instead, so that two runs of the "
        "study produce byte-identical copies of every other artifact."
    )
    return "\n".join(lines) + "\n"


def write_figures(report: dict, outdir: str) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is optional
        print("matplotlib not available; skipping figures", file=sys.stderr)
        return []

    written = []
    snr = report["settings"]["snr_db"]
    analysis = report["analysis"]

    for axis in analysis["axes"]:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for member in axis["members"]:
            style = "-o" if member["is_axis_baseline"] else "--s"
            width = 2.2 if member["is_axis_baseline"] else 1.2
            label = f"{axis['knob']}={member['value']}"
            if member["is_axis_baseline"]:
                label += " (baseline)"
            ax.plot(snr, member["pd_curve"], style, linewidth=width, markersize=4, label=label)
        ax.axhline(analysis["pd_level"], color="0.6", linewidth=0.8, linestyle=":")
        ax.axvline(analysis["reference_snr_db"], color="0.6", linewidth=0.8, linestyle=":")
        ax.set_xlabel("input SNR (dB, target-referenced)")
        ax.set_ylabel("Pd")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(
            f"{axis['name']} at {report['settings']['target_fa_per_frame']:.2f} FA/frame"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(outdir, f"axes_pd_{axis['name']}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    labels, values, lo, hi, colours = [], [], [], [], []
    for axis in analysis["axes"]:
        for member in axis["members"]:
            if member["is_axis_baseline"] or member["shift_db"]["shift_db_median"] is None:
                continue
            labels.append(f"{axis['knob']}={member['value']}")
            values.append(member["shift_db"]["shift_db_median"])
            lo.append(member["shift_db"]["shift_db_median"] - member["shift_db"]["shift_db_lo"])
            hi.append(member["shift_db"]["shift_db_hi"] - member["shift_db"]["shift_db_median"])
            colours.append("#1b7837" if member["resolved_shift"] else "#999999")
    if labels:
        fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(labels) + 1.6))
        positions = np.arange(len(labels))
        ax.barh(positions, values, xerr=[lo, hi], color=colours, height=0.65, capsize=2)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.0, color="0.2", linewidth=1.0)
        ax.set_xlabel(
            f"sensitivity gained at Pd={analysis['pd_level']:.2f} (dB, positive = better)"
        )
        ax.set_title("what actually moves the curve, at equal false alarms per frame")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, "axes_shift_summary.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def update_readme(markdown: str, readme_path: str = README_PATH) -> bool:
    """Replace the generated block in the README with ``markdown``.

    Takes rendered text rather than the report, so the same bytes that were written
    to ``results_axes/axes_table.md`` are what land in the README -- and so the
    injection can be redone from a finished run with ``--render-from`` without
    re-measuring anything.
    """
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
    new = f"{head}{TABLE_BEGIN}\n\n{markdown}\n{TABLE_END}{tail}"
    with open(readme_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    return True


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_axes",
        description=(
            "Second slice of the AIRadar detection benchmark: which knobs move the "
            "detection curve, with every configuration re-calibrated to a common "
            "measured false-alarm rate per frame."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config_phaser", help="RADAR_CONFIGS key (FMCW only)")
    parser.add_argument(
        "--snr-db", type=float, nargs="+", default=None, dest="snr_db",
        help=f"input SNR sweep points in dB (default: {DEFAULT_SNR_DB})",
    )
    parser.add_argument("--trials", type=int, default=32, help="frames per SNR point")
    parser.add_argument("--seed", type=int, default=20260822, help="base seed")
    parser.add_argument(
        "--axes", nargs="+", default=None,
        help=f"subset of axes to run; default all of {[a.name for a in AXES]}",
    )
    parser.add_argument(
        "--threshold-grid-db", type=float, nargs="+", default=list(DEFAULT_THRESHOLD_GRID_DB),
        help="calibration grid, dB above the local mean noise power",
    )
    parser.add_argument(
        "--target-fa-per-frame", type=float, default=None,
        help="common operating point; default 1e-4 per cell on the baseline grid",
    )
    parser.add_argument("--calibration-frames", type=int, default=20, help="target-free frames per configuration")
    parser.add_argument("--calibration-snr-db", type=float, default=-30.0, help="SNR of the calibration residuals")
    parser.add_argument("--gate-range-m", type=float, default=None, help="association gate, metres (default 2 baseline bins)")
    parser.add_argument("--gate-velocity-mps", type=float, default=None, help="association gate, m/s (default 1 baseline bin)")
    parser.add_argument("--pd-level", type=float, default=DEFAULT_PD_LEVEL, help="Pd the SNR axis is interpolated to")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000, help="paired bootstrap resamples")
    parser.add_argument("--bootstrap-seed", type=int, default=7, help="bootstrap RNG seed")
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="process pool size; results are independent of this value",
    )
    parser.add_argument("--outdir", default=RESULTS_DIR, help="results directory")
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    parser.add_argument("--update-readme", action="store_true", help="inject the tables into benchmarks/README.md")
    parser.add_argument("--quick", action="store_true", help="tiny smoke run, not a result")
    parser.add_argument("--quiet", action="store_true", help="suppress per-configuration progress")
    parser.add_argument(
        "--render-from",
        default=None,
        metavar="AXES_RESULTS_JSON",
        help=(
            "skip the measurement: re-render the tables (and the README block, with "
            "--update-readme) from a finished run's axes_results.json. The rendered "
            "text is a pure function of that file, which is what makes it safe to "
            "edit the wording without re-measuring."
        ),
    )
    return parser


def render_from_file(path: str) -> str:
    """Re-render the tables from a finished run's ``axes_results.json``."""
    with open(path, encoding="utf-8") as fh:
        return render_markdown(json.load(fh))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.render_from:
        markdown = render_from_file(args.render_from)
        path = os.path.join(os.path.dirname(os.path.abspath(args.render_from)), "axes_table.md")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(markdown)
        if args.update_readme:
            update_readme(markdown)
        if not args.quiet:
            print(markdown)
            print(f"wrote:\n  {path}")
        return 0
    if args.quick:
        args.snr_db = args.snr_db or [-37.5, -34.5, -31.5]
        args.trials = min(args.trials, 3)
        args.calibration_frames = min(args.calibration_frames, 3)
        args.threshold_grid_db = [8.0, 10.0, 12.0]
        args.bootstrap_resamples = min(args.bootstrap_resamples, 200)

    started = time.perf_counter()
    baseline = baseline_reference(args)
    target_fa = (
        args.target_fa_per_frame
        if args.target_fa_per_frame is not None
        else BASELINE_PFA_PER_CELL * baseline["eligible_cells_per_frame"]
    )
    settings = {
        "config": args.config,
        "seed": args.seed,
        "trials": args.trials,
        "snr_db": args.snr_db if args.snr_db else list(DEFAULT_SNR_DB),
        "snr_reference": "target",
        "threshold_grid_db": list(args.threshold_grid_db),
        "calibration_frames": args.calibration_frames,
        "calibration_snr_db": args.calibration_snr_db,
        "target_fa_per_frame": target_fa,
        "gate_range_m": baseline["gate_range_m"],
        "gate_velocity_mps": baseline["gate_velocity_mps"],
    }

    results = run_configurations(args, settings, args.quiet)

    clamped = [
        key
        for key, result in results.items()
        if result["calibration"]["status"] != "interpolated"
    ]
    if clamped:
        print(
            "WARNING: the calibration grid did not bracket the operating point for "
            f"{len(clamped)} configuration(s): {clamped}. Their thresholds are clamped "
            "to the end of the grid and their false-alarm rate is NOT the common one; "
            "widen --threshold-grid-db.",
            file=sys.stderr,
        )

    digests = {r["truth_digest"] for r in results.values()}
    if len(digests) != 1:
        raise SystemExit(
            "configurations did not see the same ground truth; the comparison would "
            f"not be paired (digests: {sorted(digests)})"
        )

    analysis = analyse(args, settings, results)
    report = {
        "study": {
            "name": "axis study: what moves the detection curve",
            "follows": "benchmarks/run_benchmark.py (the detector null result)",
            "held_constant": (
                "measured false alarms per frame, the scene draws, the noise draws, "
                "the association gate in physical units, and the frame budget"
            ),
            "truth_digest": next(iter(digests)),
            "slow_targets_below_1mps": next(iter(results.values()))["slow_targets"],
            "configurations": len(results),
            "detector_calls": sum(
                r["timing_s"]["detector_calls"] for r in results.values()
            ),
        },
        "settings": settings,
        "baseline": baseline,
        "analysis": analysis,
        "configs": results,
        "workers": args.workers,
        "timing_s": {
            "wall_clock_total": time.perf_counter() - started,
            "cpu_seconds_total": sum(r["timing_s"]["total"] for r in results.values()),
        },
    }

    written = write_outputs(report, args.outdir, make_figures=not args.no_figures)
    if args.update_readme:
        update_readme(render_markdown(report))
    if not args.quiet:
        print("\n" + render_markdown(report))
        print("wrote:")
        for path in written:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
