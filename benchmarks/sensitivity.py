"""Turning a Pd-vs-SNR curve into the two numbers an engineer asks for.

The first study reported Pd at fixed SNR points.  That is the right primitive but
the wrong summary: "Pd rose from 0.44 to 0.62 at -34.5 dB" is hard to act on,
while "this configuration detects the same target 1.8 dB further down" is
directly comparable against a link budget.

This module is pure arithmetic over already-measured counts -- no repo imports,
no simulation, no I/O -- so every number in the axis study is re-derivable from
``results_axes/axes_frames.csv`` alone.

Three pieces:

:func:`snr_at_pd`
    Interpolate the measured Pd curve to a fixed detection probability and return
    the SNR at which it is reached, with an explicit status when the grid does not
    bracket the crossing.  Never extrapolates.
:func:`bootstrap_shift`
    A **paired** resampling of the scenes, which is what makes small differences
    readable at a 16-frame budget: every configuration in this study sees the same
    physical scenes and the same noise draws, so the comparison is within-scene and
    its error bar is much tighter than the error bar on either absolute Pd.
:func:`frames_to_resolve`
    The honest counterpart: when a difference does not clear its own error bar,
    say how many frames would have been needed instead of reporting the difference
    as if it were real.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

#: Confidence level used for every interval reported by this module.
CONFIDENCE = 0.95
#: Two-sided normal quantile for :data:`CONFIDENCE`.
Z_SCORE = 1.959963984540054


@dataclass(frozen=True)
class SnrAtPd:
    """The SNR at which a measured Pd curve crosses a fixed level."""

    level: float
    #: ``None`` unless ``status == "interpolated"``.
    snr_db: float | None
    #: ``"interpolated"``: the grid brackets the crossing.
    #: ``"below_grid"``: Pd is already at or above ``level`` at the lowest SNR
    #: point, so the crossing is off the left edge of the sweep.
    #: ``"above_grid"``: Pd never reaches ``level``, so the crossing is off the
    #: right edge.
    status: str
    #: The two SNR points the answer was interpolated between.
    bracket_snr_db: tuple[float, float] | None = None
    bracket_pd: tuple[float, float] | None = None

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "snr_db": self.snr_db,
            "status": self.status,
            "bracket_snr_db": list(self.bracket_snr_db) if self.bracket_snr_db else None,
            "bracket_pd": list(self.bracket_pd) if self.bracket_pd else None,
        }


def snr_at_pd(snr_db, pd, level: float = 0.5) -> SnrAtPd:
    """Interpolate a Pd curve to ``level`` and return the SNR that reaches it.

    ``snr_db`` must be strictly increasing.  The answer is the **last** upward
    crossing of ``level``: for a saturating detection curve that is the edge of
    the saturated region, and taking the last rather than the first crossing makes
    the summary robust to a single lucky frame early in the sweep.

    Linear interpolation in (SNR, Pd) between the two bracketing grid points.  The
    curve is measured at a finite number of frames, so a smoother fit would only
    add a modelling assumption the data does not support; the interpolation error
    is bounded by the grid spacing and is dominated by the binomial error on Pd
    (see :func:`bootstrap_shift`).

    Extrapolation is refused.  If Pd is already at or above ``level`` at the first
    SNR point the status is ``"below_grid"``; if it never reaches ``level`` the
    status is ``"above_grid"``.  In both cases ``snr_db`` is ``None`` -- an
    unbracketed crossing is a statement about the sweep range, not a measurement.
    """
    snr = [float(s) for s in snr_db]
    values = [float(p) for p in pd]
    if len(snr) != len(values):
        raise ValueError("snr_db and pd must have the same length")
    if len(snr) < 2:
        raise ValueError("need at least two SNR points")
    if any(b <= a for a, b in itertools.pairwise(snr)):
        raise ValueError("snr_db must be strictly increasing")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")

    if values[0] >= level:
        return SnrAtPd(level, None, "below_grid")
    if max(values) < level:
        return SnrAtPd(level, None, "above_grid")

    index = max(
        i for i in range(len(snr) - 1) if values[i] < level <= values[i + 1]
    )
    lo_s, hi_s = snr[index], snr[index + 1]
    lo_p, hi_p = values[index], values[index + 1]
    crossing = lo_s + (level - lo_p) * (hi_s - lo_s) / (hi_p - lo_p)
    return SnrAtPd(level, float(crossing), "interpolated", (lo_s, hi_s), (lo_p, hi_p))


def binomial_se(p: float, n: int) -> float:
    """Standard error of a detection probability measured over ``n`` targets."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    return math.sqrt(p * (1.0 - p) / n)


def frames_to_resolve(
    difference: float,
    se_difference: float,
    frames_used: int,
    z: float = Z_SCORE,
) -> int | None:
    """Frames per SNR point needed for ``difference`` to clear its own error bar.

    The error bar shrinks as ``1/sqrt(frames)``, so requiring
    ``z * se(n) <= |difference|`` with ``se(n) = se_difference *
    sqrt(frames_used / n)`` gives
    ``n >= frames_used * (z * se_difference / |difference|)**2``.

    Returns ``None`` when the observed difference is exactly zero (no budget
    resolves a difference that is not there), and ``frames_used`` when the
    difference already clears the bar at the budget that was run.
    """
    if frames_used <= 0:
        raise ValueError("frames_used must be positive")
    if se_difference < 0:
        raise ValueError("se_difference must be non-negative")
    if difference == 0:
        return None
    needed = frames_used * (z * se_difference / abs(float(difference))) ** 2
    return max(int(frames_used), math.ceil(needed))


def _pd_from_counts(detected: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Pd per SNR point from ``[n_snr, n_frames]`` count matrices."""
    total_targets = targets.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total_targets > 0, detected.sum(axis=1) / total_targets, np.nan)


def resample_indices(n_frames: int, resamples: int, seed: int) -> np.ndarray:
    """Frame indices for a paired bootstrap, ``[resamples, n_frames]``.

    Drawn once and reused for every configuration, which is what makes the
    comparison paired: resample *r* uses the same multiset of physical scenes for
    the baseline and for the variant, so scene-to-scene difficulty cancels instead
    of adding variance to the difference.
    """
    if n_frames <= 0 or resamples <= 0:
        raise ValueError("n_frames and resamples must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_frames, size=(resamples, n_frames), dtype=np.int64)


def bootstrap_shift(
    snr_db,
    baseline_detected: np.ndarray,
    baseline_targets: np.ndarray,
    variant_detected: np.ndarray,
    variant_targets: np.ndarray,
    indices: np.ndarray,
    level: float = 0.5,
) -> dict:
    """Paired bootstrap of the SNR shift at a fixed Pd, in dB.

    ``*_detected`` and ``*_targets`` are ``[n_snr, n_frames]`` integer matrices of
    matched targets and ground-truth targets.  ``indices`` comes from
    :func:`resample_indices`.

    The reported ``shift_db`` is ``baseline_snr - variant_snr``: **positive means
    the variant is more sensitive** (it reaches the same detection probability at
    a lower SNR).

    Resamples in which either curve fails to bracket the crossing are dropped and
    counted in ``undefined_fraction``.  A large fraction means the SNR grid, not
    the frame budget, is the limiting factor and the interval must not be read.
    """
    shifts = []
    undefined = 0
    for row in indices:
        base_pd = _pd_from_counts(baseline_detected[:, row], baseline_targets[:, row])
        var_pd = _pd_from_counts(variant_detected[:, row], variant_targets[:, row])
        base = snr_at_pd(snr_db, base_pd, level)
        var = snr_at_pd(snr_db, var_pd, level)
        if base.snr_db is None or var.snr_db is None:
            undefined += 1
            continue
        shifts.append(base.snr_db - var.snr_db)
    total = len(indices)
    if not shifts:
        return {
            "shift_db_median": None,
            "shift_db_lo": None,
            "shift_db_hi": None,
            "shift_db_se": None,
            "undefined_fraction": 1.0,
            "resamples": total,
        }
    values = np.asarray(shifts, dtype=float)
    tail = 100.0 * (1.0 - CONFIDENCE) / 2.0
    return {
        "shift_db_median": float(np.median(values)),
        "shift_db_lo": float(np.percentile(values, tail)),
        "shift_db_hi": float(np.percentile(values, 100.0 - tail)),
        "shift_db_se": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "undefined_fraction": float(undefined / total),
        "resamples": total,
    }


def bootstrap_pd_difference(
    baseline_detected: np.ndarray,
    baseline_targets: np.ndarray,
    variant_detected: np.ndarray,
    variant_targets: np.ndarray,
    indices: np.ndarray,
) -> dict:
    """Paired bootstrap of ``Pd(variant) - Pd(baseline)`` at one SNR point.

    Inputs are ``[n_frames]`` count vectors for a single SNR point.
    """
    base = np.asarray(baseline_detected, dtype=float)
    base_t = np.asarray(baseline_targets, dtype=float)
    var = np.asarray(variant_detected, dtype=float)
    var_t = np.asarray(variant_targets, dtype=float)
    base_pd = base[indices].sum(axis=1) / np.maximum(base_t[indices].sum(axis=1), 1e-12)
    var_pd = var[indices].sum(axis=1) / np.maximum(var_t[indices].sum(axis=1), 1e-12)
    diff = var_pd - base_pd
    tail = 100.0 * (1.0 - CONFIDENCE) / 2.0
    observed = (
        var.sum() / max(var_t.sum(), 1e-12) - base.sum() / max(base_t.sum(), 1e-12)
    )
    return {
        "pd_difference": float(observed),
        "pd_difference_lo": float(np.percentile(diff, tail)),
        "pd_difference_hi": float(np.percentile(diff, 100.0 - tail)),
        "pd_difference_se": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
    }
