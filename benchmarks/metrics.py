"""Detection metrics for the AIRadar benchmark.

Pure functions over plain dataclasses -- no repo imports, no numpy RNG, no I/O --
so every number in the results table is reproducible from the association rule
alone.  See ``benchmarks/README.md`` for the prose definitions.

Association rule (one sentence): a detection matches a ground-truth target iff it
lies inside a rectangular gate of +/-``gate_range_bins`` range bins and
+/-``gate_doppler_bins`` Doppler bins around the target's nearest-bin position,
and matching is one-to-one, assigned greedily in order of increasing normalized
gate distance.

Consequence worth stating out loud: *every* in-gate detection beyond the one
matched pair is counted as a false positive.  Target sidelobes that survive
non-maximum suppression inside a target's own neighbourhood are therefore
reported as false alarms, not merged into the detection.  :func:`associate`
accepts an optional wider "near" gate purely so that this component can be
measured and reported separately -- it does not change any TP/FP/FN count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point:
    """A detection or a ground-truth target in bin + physical coordinates."""

    range_bin: int
    doppler_bin: int
    range_m: float
    velocity_mps: float


@dataclass(frozen=True)
class Match:
    target_index: int
    detection_index: int
    range_error_m: float
    velocity_error_mps: float
    gate_distance: float


@dataclass
class FrameResult:
    """Per-frame confusion counts and matched-pair errors."""

    num_targets: int
    num_detections: int
    matches: list[Match] = field(default_factory=list)
    unmatched_targets: list[int] = field(default_factory=list)
    unmatched_detections: list[int] = field(default_factory=list)
    #: Subset of ``unmatched_detections`` that lies inside the wider "near"
    #: neighbourhood of some ground-truth target (target sidelobes and split
    #: peaks).  Empty when no near gate was supplied.  Diagnostic only.
    near_target_false_positives: list[int] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_targets)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_detections)

    @property
    def false_positives_near_target(self) -> int:
        return len(self.near_target_false_positives)

    @property
    def false_positives_far_from_target(self) -> int:
        return len(self.unmatched_detections) - len(self.near_target_false_positives)


def gate_distance(
    target: Point,
    detection: Point,
    gate_range_bins: int,
    gate_doppler_bins: int,
) -> float | None:
    """Normalized distance inside the association gate, or ``None`` if outside.

    The gate is a rectangle in bin space.  Inside it, candidate pairs are ranked
    by ``sqrt((dr/Gr)^2 + (dd/Gd)^2)`` so that a pair one bin off in range and one
    bin off in Doppler is ranked by how much of each gate it consumes.  A gate
    dimension of 0 means "must match exactly on this axis".
    """
    dr = abs(int(detection.range_bin) - int(target.range_bin))
    dd = abs(int(detection.doppler_bin) - int(target.doppler_bin))
    if dr > gate_range_bins or dd > gate_doppler_bins:
        return None
    nr = dr / gate_range_bins if gate_range_bins > 0 else 0.0
    nd = dd / gate_doppler_bins if gate_doppler_bins > 0 else 0.0
    return math.sqrt(nr * nr + nd * nd)


def associate(
    targets: list[Point],
    detections: list[Point],
    gate_range_bins: int = 2,
    gate_doppler_bins: int = 1,
    near_range_bins: int | None = None,
    near_doppler_bins: int | None = None,
) -> FrameResult:
    """One-to-one greedy association of detections to ground-truth targets.

    Deterministic: candidate pairs are sorted by
    ``(gate_distance, target_index, detection_index)`` and accepted while both
    endpoints are still free.  Errors are measured against the target's
    *continuous* range/velocity, not its quantized bin centre.

    ``near_range_bins`` / ``near_doppler_bins`` define an optional wider
    rectangle, used only to split the false positives into "near a target"
    (sidelobe / split-peak residue) and "far from any target".  Supplying them
    changes no TP/FP/FN count.
    """
    if gate_range_bins < 0 or gate_doppler_bins < 0:
        raise ValueError("gate sizes must be non-negative")
    if (near_range_bins is None) != (near_doppler_bins is None):
        raise ValueError("near_range_bins and near_doppler_bins must be given together")
    if near_range_bins is not None and (
        near_range_bins < gate_range_bins or near_doppler_bins < gate_doppler_bins
    ):
        raise ValueError("the near gate must be at least as large as the association gate")

    candidates = []
    for ti, target in enumerate(targets):
        for di, detection in enumerate(detections):
            dist = gate_distance(target, detection, gate_range_bins, gate_doppler_bins)
            if dist is not None:
                candidates.append((dist, ti, di))
    candidates.sort()

    taken_targets: set[int] = set()
    taken_detections: set[int] = set()
    matches: list[Match] = []
    for dist, ti, di in candidates:
        if ti in taken_targets or di in taken_detections:
            continue
        taken_targets.add(ti)
        taken_detections.add(di)
        matches.append(
            Match(
                target_index=ti,
                detection_index=di,
                range_error_m=float(detections[di].range_m - targets[ti].range_m),
                velocity_error_mps=float(
                    detections[di].velocity_mps - targets[ti].velocity_mps
                ),
                gate_distance=float(dist),
            )
        )

    unmatched_detections = [
        i for i in range(len(detections)) if i not in taken_detections
    ]
    near: list[int] = []
    if near_range_bins is not None and targets:
        for di in unmatched_detections:
            detection = detections[di]
            for target in targets:
                dr = abs(int(detection.range_bin) - int(target.range_bin))
                dd = abs(int(detection.doppler_bin) - int(target.doppler_bin))
                if dr <= near_range_bins and dd <= near_doppler_bins:
                    near.append(di)
                    break

    return FrameResult(
        num_targets=len(targets),
        num_detections=len(detections),
        matches=matches,
        unmatched_targets=[i for i in range(len(targets)) if i not in taken_targets],
        unmatched_detections=unmatched_detections,
        near_target_false_positives=near,
    )


def rmse(values: list[float]) -> float:
    """Root-mean-square of ``values``; ``nan`` for an empty list."""
    if not values:
        return float("nan")
    return math.sqrt(sum(v * v for v in values) / len(values))


def aggregate(frames: list[FrameResult], eligible_cells_per_frame: int) -> dict:
    """Aggregate per-frame results into the reported metrics.

    * ``pd`` -- probability of detection: total TP / total ground-truth targets.
    * ``false_alarms_per_frame`` -- total FP / number of frames.
    * ``false_positives_near_target`` -- how many of those FPs sit inside the wider
      "near" gate of a ground-truth target, i.e. are target sidelobes or split peaks
      rather than noise peaks.  Zero unless ``associate`` was given a near gate.
    * ``false_alarm_rate_per_cell`` -- total FP / (frames * eligible cells).  This
      counts *post-NMS peaks*, not raw threshold crossings, so it is a peak-level
      false-alarm density and is NOT directly comparable to a CFAR design Pfa.
    * ``range_rmse_m`` / ``velocity_rmse_mps`` -- RMSE over matched pairs only,
      against the target's continuous truth.  Undefined (``nan``) with no matches.
    """
    if eligible_cells_per_frame <= 0:
        raise ValueError("eligible_cells_per_frame must be positive")
    n_frames = len(frames)
    total_targets = sum(f.num_targets for f in frames)
    total_tp = sum(f.true_positives for f in frames)
    total_fp = sum(f.false_positives for f in frames)
    total_fp_near = sum(f.false_positives_near_target for f in frames)
    total_fn = sum(f.false_negatives for f in frames)
    total_detections = sum(f.num_detections for f in frames)

    range_errors = [m.range_error_m for f in frames for m in f.matches]
    velocity_errors = [m.velocity_error_mps for f in frames for m in f.matches]

    return {
        "frames": n_frames,
        "ground_truth_targets": total_targets,
        "detections": total_detections,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_positives_near_target": total_fp_near,
        "false_positives_far_from_target": total_fp - total_fp_near,
        "false_negatives": total_fn,
        "pd": (total_tp / total_targets) if total_targets else float("nan"),
        "precision": (total_tp / total_detections) if total_detections else float("nan"),
        "false_alarms_per_frame": (total_fp / n_frames) if n_frames else float("nan"),
        "false_alarm_rate_per_cell": (
            total_fp / (n_frames * eligible_cells_per_frame) if n_frames else float("nan")
        ),
        "eligible_cells_per_frame": int(eligible_cells_per_frame),
        "range_rmse_m": rmse(range_errors),
        "velocity_rmse_mps": rmse(velocity_errors),
        "range_bias_m": (sum(range_errors) / len(range_errors)) if range_errors else float("nan"),
        "velocity_bias_mps": (
            (sum(velocity_errors) / len(velocity_errors)) if velocity_errors else float("nan")
        ),
    }


def quantization_rmse_floor(bin_spacing: float) -> float:
    """RMSE floor imposed by reporting a bin index as a physical value.

    A uniformly distributed truth inside a bin of width ``w`` that is reported at
    the bin centre has RMSE ``w / sqrt(12)``.  Any measured RMSE at or below this
    means the estimator is limited by the bin grid, not by noise.
    """
    return float(bin_spacing) / math.sqrt(12.0)
