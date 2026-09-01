"""AIRadar classical-detector benchmark: one entry point, real numbers.

Sweeps time-domain SNR over a fixed set of seeded FMCW scenarios, runs every
registered detector on every frame, and writes machine-readable results plus
(optionally) figures and a Markdown table injected into ``benchmarks/README.md``.

Usage
-----
    python -m benchmarks.run_benchmark                       # default sweep
    python -m benchmarks.run_benchmark --trials 4 --quick    # fast smoke run
    python -m benchmarks.run_benchmark --update-readme       # refresh README table

Everything is CPU-only and torch-free; see ``benchmarks/repo_shim.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata

import numpy as np

from benchmarks.detectors import build_detectors
from benchmarks.metrics import Point, aggregate, associate, quantization_rmse_floor
from benchmarks.repo_shim import REPO_ROOT, load_repo_modules
from benchmarks.scenarios import RadarParams

DEFAULT_SNR_DB = [-50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0]

#: Scenario axis: name -> (AIRadarDataset.apply_realistic_effects, clutter_intensity).
#: ``clutter_default`` is what the repo's dataset pipeline ships; measurement showed
#: that at that setting the clutter+coupling returns land ~23 dB *below* the primary
#: target power, so it barely perturbs the range-Doppler map.  ``clutter_strong``
#: scales the clutter RCS by +40 dB to put clutter above the target and make the
#: axis actually exercise the detectors.
SCENARIOS: dict[str, tuple[bool, float]] = {
    "clutter_off": (False, 1.0),
    "clutter_default": (True, 1.0),
    "clutter_strong": (True, 1e4),
}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
README_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
TABLE_BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
TABLE_END = "<!-- END GENERATED RESULTS -->"


# --------------------------------------------------------------------------- #
# environment / provenance
# --------------------------------------------------------------------------- #


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return out.stdout.strip() or None


def environment_report() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "matplotlib": _package_version("matplotlib"),
        "torch": _package_version("torch"),
        "git_commit": _git_commit(),
        "device": "cpu",
    }


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


def run_sweep(args) -> dict:
    repo = load_repo_modules()
    configs = repo["RADAR_CONFIGS"]
    if args.config not in configs:
        raise SystemExit(
            f"unknown config {args.config!r}; available: {sorted(configs)}"
        )

    snr_points = args.snr_db if args.snr_db else list(DEFAULT_SNR_DB)
    clutter_modes = args.scenarios

    started = time.perf_counter()
    rows: list[dict] = []
    frame_rows: list[dict] = []
    detector_time: dict[str, float] = {}
    simulate_time = 0.0
    params_by_mode: dict[str, RadarParams] = {}
    detector_meta: dict[str, dict] = {}
    scenario_meta: dict[str, dict] = {}

    # Imported here so the shim has already put AIRadar on sys.path.
    from benchmarks.scenarios import clutter_to_target_power_db, simulate_frame

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
        scenario_meta[clutter_mode] = {
            "apply_realistic_effects": enabled,
            "clutter_intensity": intensity,
            "clutter_to_target_power_db": clutter_to_target_power_db(
                params, args.trials, args.seed
            ),
        }
        detectors = build_detectors(params, mtd=args.mtd, pfa=args.pfa)
        for det in detectors:
            detector_meta.setdefault(
                det.name,
                {
                    "source": det.source,
                    "description": det.description,
                    "params": {k: _jsonable(v) for k, v in det.params.items()},
                },
            )

        for snr_index, snr_db in enumerate(snr_points):
            per_detector_frames: dict[str, list] = {d.name: [] for d in detectors}
            peak_snrs: list[float] = []
            for trial in range(args.trials):
                t0 = time.perf_counter()
                frame = simulate_frame(params, trial, snr_index, snr_db, args.seed)
                simulate_time += time.perf_counter() - t0
                peak_snrs.append(frame.peak_snr_db)

                truth = [
                    Point(t.range_bin, t.doppler_bin, t.range_m, t.velocity_mps)
                    for t in frame.targets
                ]
                for det in detectors:
                    t0 = time.perf_counter()
                    detections = det.run(frame, params)
                    detector_time[det.name] = detector_time.get(det.name, 0.0) + (
                        time.perf_counter() - t0
                    )
                    result = associate(
                        truth, detections, args.gate_range_bins, args.gate_doppler_bins
                    )
                    per_detector_frames[det.name].append(result)
                    frame_rows.append(
                        {
                            "clutter": clutter_mode,
                            "snr_db": snr_db,
                            "trial": trial,
                            "detector": det.name,
                            "peak_snr_db": round(frame.peak_snr_db, 3),
                            "num_targets": result.num_targets,
                            "num_detections": result.num_detections,
                            "tp": result.true_positives,
                            "fp": result.false_positives,
                            "fn": result.false_negatives,
                        }
                    )

            for det in detectors:
                agg = aggregate(
                    per_detector_frames[det.name],
                    eligible_cells_per_frame=det.eligible_cells(params),
                )
                rows.append(
                    {
                        "detector": det.name,
                        "clutter": clutter_mode,
                        "snr_db": snr_db,
                        "mean_peak_snr_db": float(np.mean(peak_snrs)),
                        **agg,
                    }
                )
            if not args.quiet:
                print(
                    f"[clutter={clutter_mode}] snr={snr_db:+.0f} dB "
                    f"peak_snr={np.mean(peak_snrs):5.1f} dB  "
                    + "  ".join(
                        f"{r['detector']}: Pd={r['pd']:.3f} FA/frame={r['false_alarms_per_frame']:.2f}"
                        for r in rows[-len(detectors) :]
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
            "pfa": args.pfa,
        },
        "scenario_details": scenario_meta,
        "association": {
            "gate_range_bins": args.gate_range_bins,
            "gate_doppler_bins": args.gate_doppler_bins,
            "gate_range_m": args.gate_range_bins * reference_params.range_bin_spacing,
            "gate_velocity_mps": args.gate_doppler_bins
            * reference_params.velocity_bin_spacing,
            "rule": (
                "rectangular gate in bin space, one-to-one, greedy by increasing "
                "normalized gate distance"
            ),
        },
        "quantization_floor": {
            "range_rmse_m": quantization_rmse_floor(reference_params.range_bin_spacing),
            "velocity_rmse_mps": quantization_rmse_floor(
                reference_params.velocity_bin_spacing
            ),
        },
        "detectors": detector_meta,
        "environment": environment_report(),
        "shim": load_repo_modules()["shim_report"].as_dict(),
        "timing_s": {
            "wall_clock_total": wall,
            "simulation": simulate_time,
            "detectors": detector_time,
            "frames_simulated": args.trials * len(snr_points) * len(clutter_modes),
        },
        "rows": rows,
        "frames": frame_rows,
    }


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

SUMMARY_FIELDS = [
    "detector",
    "clutter",
    "snr_db",
    "mean_peak_snr_db",
    "frames",
    "ground_truth_targets",
    "detections",
    "true_positives",
    "false_positives",
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


def write_outputs(report: dict, outdir: str, make_figures: bool) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    written = []

    summary_csv = os.path.join(outdir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({k: row.get(k) for k in SUMMARY_FIELDS})
    written.append(summary_csv)

    frames_csv = os.path.join(outdir, "frames.csv")
    if report["frames"]:
        with open(frames_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report["frames"][0].keys()))
            writer.writeheader()
            writer.writerows(report["frames"])
        written.append(frames_csv)

    results_json = os.path.join(outdir, "results.json")
    slim = {k: v for k, v in report.items() if k != "frames"}
    with open(results_json, "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=2, default=_jsonable)
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
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    radar = report["radar"]
    assoc = report["association"]
    sweep = report["sweep"]
    floor = report["quantization_floor"]

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
        f"target(s) per frame, base seed {sweep['base_seed']}. Association gate: "
        f"+/-{assoc['gate_range_bins']} range bins ({assoc['gate_range_m']:.3f} m), "
        f"+/-{assoc['gate_doppler_bins']} Doppler bins "
        f"({assoc['gate_velocity_mps']:.3f} m/s). Quantization RMSE floor: "
        f"{floor['range_rmse_m']:.3f} m / {floor['velocity_rmse_mps']:.3f} m/s."
    )
    lines.append("")

    for clutter_mode in sweep["scenarios"]:
        detail = report["scenario_details"][clutter_mode]
        ratio = detail["clutter_to_target_power_db"]
        ratio_text = (
            "clutter disabled"
            if ratio == float("-inf")
            else f"clutter/target RCS power {ratio:+.1f} dB"
        )
        lines.append(f"### Scenario `{clutter_mode}`")
        lines.append("")
        lines.append(
            f"`apply_realistic_effects={detail['apply_realistic_effects']}`, "
            f"`clutter_intensity={detail['clutter_intensity']:g}` -- {ratio_text}."
        )
        lines.append("")
        lines.append(
            "| detector | SNR in (dB) | peak SNR (dB) | Pd | FA/frame | "
            "FA rate /cell | range RMSE (m) | vel RMSE (m/s) | TP | FP | FN |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = [r for r in report["rows"] if r["clutter"] == clutter_mode]
        rows.sort(key=lambda r: (r["detector"], r["snr_db"]))
        for r in rows:
            lines.append(
                "| `{d}` | {snr:+.0f} | {peak} | {pd} | {fa} | {far} | {rr} | {vr} "
                "| {tp} | {fp} | {fn} |".format(
                    d=r["detector"],
                    snr=r["snr_db"],
                    peak=_fmt(r["mean_peak_snr_db"], 1),
                    pd=_fmt(r["pd"], 3),
                    fa=_fmt(r["false_alarms_per_frame"], 2),
                    far=f"{r['false_alarm_rate_per_cell']:.2e}",
                    rr=_fmt(r["range_rmse_m"], 4),
                    vr=_fmt(r["velocity_rmse_mps"], 4),
                    tp=r["true_positives"],
                    fp=r["false_positives"],
                    fn=r["false_negatives"],
                )
            )
        lines.append("")

    timing = report["timing_s"]
    lines.append(
        f"Runtime: {timing['wall_clock_total']:.1f} s wall clock for "
        f"{timing['frames_simulated']} simulated frames "
        f"({timing['simulation']:.1f} s simulation, "
        + ", ".join(f"{k} {v:.1f} s" for k, v in sorted(timing["detectors"].items()))
        + ")."
    )
    lines.append("")
    env = report["environment"]
    lines.append(
        f"Environment: Python {env['python']}, numpy {env['numpy']}, scipy "
        f"{env['scipy']}, matplotlib {env['matplotlib']}, torch "
        f"{env['torch'] or 'not installed'}, {env['platform']}, device "
        f"{env['device']}. Repo commit `{(env['git_commit'] or 'unknown')[:12]}`."
    )
    lines.append("")
    return "\n".join(lines)


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
    detectors = sorted({r["detector"] for r in rows})
    modes = report["sweep"]["scenarios"]
    styles = {"clutter_off": "-o", "clutter_default": "--s", "clutter_strong": ":^"}

    specs = [
        ("pd", "Probability of detection", "roc_pd_vs_snr.png", (-0.05, 1.05)),
        ("false_alarms_per_frame", "False alarms per frame", "fa_per_frame_vs_snr.png", None),
        ("range_rmse_m", "Range RMSE (m)", "range_rmse_vs_snr.png", None),
        ("velocity_rmse_mps", "Velocity RMSE (m/s)", "velocity_rmse_vs_snr.png", None),
    ]
    for key, ylabel, fname, ylim in specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for det in detectors:
            for mode in modes:
                sel = [r for r in rows if r["detector"] == det and r["clutter"] == mode]
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
        ax.set_xlabel("input time-domain SNR (dB) passed to simulate_fmcw_signal")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        if key.endswith("rmse_m"):
            ax.axhline(
                report["quantization_floor"]["range_rmse_m"],
                color="gray",
                lw=0.8,
                ls=":",
                label="quantization floor",
            )
        if key.endswith("rmse_mps"):
            ax.axhline(
                report["quantization_floor"]["velocity_rmse_mps"],
                color="gray",
                lw=0.8,
                ls=":",
                label="quantization floor",
            )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title(f"{ylabel} vs input SNR - config {report['config']}")
        fig.tight_layout()
        path = os.path.join(outdir, fname)
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
    with open(readme_path, "w", encoding="utf-8") as fh:
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
        "--max-targets",
        type=int,
        default=1,
        help="max ground-truth targets per frame (1 keeps the SNR axis unambiguous)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIOS),
        default=["clutter_off", "clutter_default", "clutter_strong"],
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
        "--pfa", type=float, default=1e-5, help="design Pfa handed to cfar_2d_advanced"
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
        help="inject the generated table into benchmarks/README.md",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="3 SNR points and 3 trials: a smoke run, not a result",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-point progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.snr_db = args.snr_db or [-40.0, -30.0, -20.0]
        args.trials = min(args.trials, 3)
        args.scenarios = args.scenarios[:2]
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
