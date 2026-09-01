"""Threshold calibration: measure Pfa vs threshold, then solve for a common Pfa.

Why this stage exists
---------------------
The three repo detectors ship at three different distances above the local noise
level (9.49, 12.49 and 16.66 dB -- see :mod:`benchmarks.detectors`).  Comparing
their Pd at those settings ranks the thresholds, not the detectors, and it also
pins the false-alarm columns of the two conservative detectors at exactly zero:
at 16.66 dB effective threshold the per-cell peak false-alarm probability is small
enough that no achievable frame budget can produce a single event, so "0.00
FA/frame" is a statement about the frame count, not about the detector.

So: measure each detector's per-cell false-alarm density as a function of its own
threshold knob on **target-free** scenes, then solve each detector's knob for one
common target density and compare there.

What "measurable" means here
----------------------------
A false-alarm density ``Pfa`` is only measurable if ``cells_per_frame * frames *
Pfa`` is comfortably above 1.  :func:`measurability` states that arithmetic for a
given budget; the reports label any target that fails it instead of printing a
silent zero.

Everything is measured on *post-NMS peaks*, because that is what the detectors
return and what the Pd/FA columns count.  It is a peak density, not a raw
threshold-crossing probability, and is therefore not directly comparable to a CFAR
design ``Pfa`` -- which is exactly why the design ``pfa`` argument of
``cfar_2d_advanced`` cannot be used as a common operating point and has to be
calibrated like everything else.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from benchmarks.detectors import SPECS_BY_NAME
from benchmarks.metrics import associate

#: Expected false-alarm events below which a measured density is not reported as a
#: number.  Ten events is a ~32% relative standard error on a Poisson count.
MIN_EXPECTED_EVENTS = 10.0


@dataclass(frozen=True)
class CalibrationPoint:
    """One (threshold, measured false-alarm density) sample."""

    effective_threshold_db: float
    native_value: float
    detections: int
    measured_pfa: float
    #: True when this point is the value the repository ships, rather than a grid point.
    is_shipped: bool = False

    def as_dict(self) -> dict:
        return {
            "effective_threshold_db": self.effective_threshold_db,
            "native_value": self.native_value,
            "false_alarms": self.detections,
            "measured_pfa_per_cell": self.measured_pfa,
            "is_shipped_setting": self.is_shipped,
        }


@dataclass(frozen=True)
class ThresholdSolution:
    """The knob value that hits a requested false-alarm density."""

    target_pfa: float
    effective_threshold_db: float
    native_value: float
    #: ``"interpolated"``, ``"clamped_low"`` (target coarser than the whole grid),
    #: ``"clamped_high"`` (target finer than any measurable grid point) or
    #: ``"unreachable"`` (no grid point produced a single false alarm).
    status: str
    bracket: tuple[float, float] | None
    bracket_pfa: tuple[float, float] | None

    def as_dict(self) -> dict:
        return {
            "target_pfa_per_cell": self.target_pfa,
            "effective_threshold_db": self.effective_threshold_db,
            "native_value": self.native_value,
            "status": self.status,
            "bracket_effective_threshold_db": list(self.bracket) if self.bracket else None,
            "bracket_measured_pfa": list(self.bracket_pfa) if self.bracket_pfa else None,
        }


@dataclass(frozen=True)
class CalibrationCurve:
    """A detector's measured false-alarm density over a threshold grid."""

    detector: str
    knob: str
    scene_kind: str
    frames: int
    eligible_cells_per_frame: int
    points: tuple[CalibrationPoint, ...]
    #: The radar parameters the curve was measured with, needed to invert a solved
    #: effective threshold back into the detector's own knob.  Set via
    #: ``object.__setattr__`` by :func:`measure_curve` so the curve stays a plain
    #: value object.
    _params: object = None

    @property
    def cell_trials(self) -> int:
        return self.frames * self.eligible_cells_per_frame

    @property
    def min_measurable_pfa(self) -> float:
        """Density corresponding to exactly one observed false alarm."""
        return 1.0 / self.cell_trials

    def solve(self, target_pfa: float) -> ThresholdSolution:
        """Solve for the threshold that gives ``target_pfa``, log-linearly.

        The measured densities fall roughly one decade per dB of threshold, so the
        interpolation is done in ``(effective_threshold_db, log10(pfa))`` between the
        two bracketing grid points.  Only grid points (not the shipped setting) take
        part, and only points with at least one observed false alarm, because
        ``log10(0)`` is not a number.
        """
        if not 0.0 < target_pfa < 1.0:
            raise ValueError("target_pfa must be in (0, 1)")
        usable = sorted(
            (p for p in self.points if not p.is_shipped and p.detections > 0),
            key=lambda p: p.effective_threshold_db,
        )
        spec = SPECS_BY_NAME[self.detector]
        if not usable:
            return ThresholdSolution(
                target_pfa, float("nan"), float("nan"), "unreachable", None, None
            )
        if target_pfa >= usable[0].measured_pfa:
            point = usable[0]
            return ThresholdSolution(
                target_pfa,
                point.effective_threshold_db,
                point.native_value,
                "clamped_low",
                None,
                (point.measured_pfa, point.measured_pfa),
            )
        if target_pfa <= usable[-1].measured_pfa:
            point = usable[-1]
            return ThresholdSolution(
                target_pfa,
                point.effective_threshold_db,
                point.native_value,
                "clamped_high",
                None,
                (point.measured_pfa, point.measured_pfa),
            )
        lo = usable[0]
        hi = usable[-1]
        for left, right in itertools.pairwise(usable):
            if left.measured_pfa >= target_pfa >= right.measured_pfa:
                lo, hi = left, right
                break
        span = math.log10(hi.measured_pfa) - math.log10(lo.measured_pfa)
        if span == 0:  # pragma: no cover - defensive
            effective = lo.effective_threshold_db
        else:
            frac = (math.log10(target_pfa) - math.log10(lo.measured_pfa)) / span
            effective = lo.effective_threshold_db + frac * (
                hi.effective_threshold_db - lo.effective_threshold_db
            )
        return ThresholdSolution(
            target_pfa,
            float(effective),
            float(spec.native_knob(effective, self._params)),
            "interpolated",
            (lo.effective_threshold_db, hi.effective_threshold_db),
            (lo.measured_pfa, hi.measured_pfa),
        )

    def as_dict(self, targets: list[float]) -> dict:
        return {
            "detector": self.detector,
            "knob": self.knob,
            "scene_kind": self.scene_kind,
            "frames": self.frames,
            "eligible_cells_per_frame": self.eligible_cells_per_frame,
            "cell_trials": self.cell_trials,
            "min_measurable_pfa_per_cell": self.min_measurable_pfa,
            "curve": [p.as_dict() for p in self.points],
            "solutions": [self.solve(t).as_dict() for t in targets],
        }


def measurability(eligible_cells_per_frame: int, frames: int, target_pfa: float) -> dict:
    """State the arithmetic behind "is this Pfa measurable at this budget?"."""
    cell_trials = eligible_cells_per_frame * frames
    expected = cell_trials * target_pfa
    return {
        "target_pfa_per_cell": target_pfa,
        "eligible_cells_per_frame": eligible_cells_per_frame,
        "frames": frames,
        "cell_trials": cell_trials,
        "expected_false_alarms": expected,
        "measurable": bool(expected >= MIN_EXPECTED_EVENTS),
        "frames_for_ten_expected_events": (
            math.ceil(MIN_EXPECTED_EVENTS / (eligible_cells_per_frame * target_pfa))
            if target_pfa > 0
            else None
        ),
    }


def measure_curve(
    params,
    detector_name: str,
    frames: list,
    grid_db: list[float],
    scene_kind: str,
    *,
    include_shipped: bool = True,
    mtd: bool = False,
    on_call=None,
) -> CalibrationCurve:
    """Run one detector at every grid threshold over target-free frames.

    ``frames`` must be target-free (``frame.targets == ()``), so every returned peak
    is a false alarm by construction and ``associate`` reduces to counting.
    """
    spec = SPECS_BY_NAME[detector_name]
    for frame in frames:
        if frame.targets:
            raise ValueError("calibration frames must be target-free")

    natives = [(float(g), spec.native_knob(g, params), False) for g in grid_db]
    if include_shipped:
        shipped = float(spec.shipped_knob(params))
        natives.append((spec.effective_threshold_db(shipped, params), shipped, True))
    natives.sort(key=lambda item: item[0])

    points: list[CalibrationPoint] = []
    eligible = None
    for effective_db, native, is_shipped in natives:
        detector = spec.build(params, native, "calibration", mtd=mtd)
        eligible = detector.eligible_cells(params)
        total = 0
        for frame in frames:
            detections = detector.run(frame, params)
            if on_call is not None:
                on_call()
            total += associate([], detections, 0, 0).false_positives
        points.append(
            CalibrationPoint(
                effective_threshold_db=float(effective_db),
                native_value=float(native),
                detections=total,
                measured_pfa=total / (len(frames) * eligible),
                is_shipped=is_shipped,
            )
        )

    curve = CalibrationCurve(
        detector=detector_name,
        knob=spec.knob,
        scene_kind=scene_kind,
        frames=len(frames),
        eligible_cells_per_frame=int(eligible or 0),
        points=tuple(points),
    )
    object.__setattr__(curve, "_params", params)
    return curve
