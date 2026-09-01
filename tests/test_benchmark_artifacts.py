"""Guard the committed benchmark artifacts against silent staleness.

The two studies quote numbers in benchmarks/README.md that are generated from the
committed CSVs. Nothing else in the suite reads those files, so if the code that
produces them changes -- or if someone edits a table by hand -- the README keeps
its old numbers, the figures keep looking authoritative and CI stays green.

These tests re-derive the headline claims from the committed per-frame data and
assert the committed summary tables still agree with them. They are cheap: no
simulation, no detector runs, only arithmetic over CSVs.
"""

import csv
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

RESULTS = os.path.join(_REPO_ROOT, "benchmarks", "results")
RESULTS_AXES = os.path.join(_REPO_ROOT, "benchmarks", "results_axes")


def _rows(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.relpath(path, _REPO_ROOT)} not committed")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_study1_summary_pd_matches_the_per_frame_counts():
    """Every Pd in summary.csv must be the true positives over the targets in frames.csv."""
    frames = _rows(os.path.join(RESULTS, "frames.csv"))
    summary = _rows(os.path.join(RESULTS, "summary.csv"))

    agg = {}
    for r in frames:
        key = (r["clutter"], r["variant"], r["detector"], r["snr_db"])
        tp, tgt = agg.get(key, (0, 0))
        agg[key] = (tp + int(r["tp"]), tgt + int(r["num_targets"]))

    checked = 0
    for r in summary:
        key = (r["clutter"], r["variant"], r["detector"], r["snr_db"])
        if key not in agg:
            continue
        tp, tgt = agg[key]
        assert tgt > 0, key
        assert abs(float(r["pd"]) - tp / tgt) < 1e-9, (
            f"summary.csv Pd for {key} is {r['pd']} but frames.csv gives {tp}/{tgt}. "
            "One of the two is stale."
        )
        checked += 1
    assert checked >= 40, f"only cross-checked {checked} rows; the join is not working"


def test_study1_detectors_agree_on_the_target_in_the_documented_proportion():
    """Pin the claim the first study's conclusion rests on.

    benchmarks/README.md states the three detectors detect the target in the same
    frames in 220 of 224 cases at the calibrated operating point, and that all three
    per-frame counts match in only 82. Those two numbers carry the whole "the
    ranking is a threshold artefact" argument, so they are asserted here.
    """
    frames = _rows(os.path.join(RESULTS, "frames.csv"))
    calibrated = [r for r in frames if r["variant"] == "calibrated_pfa1e-4"]
    detectors = sorted({r["detector"] for r in calibrated})
    assert len(detectors) == 3, detectors

    by_frame = {}
    for r in calibrated:
        by_frame.setdefault((r["clutter"], r["snr_db"], r["trial"]), {})[r["detector"]] = (
            r["tp"],
            r["fp"],
            r["fn"],
        )

    complete = [v for v in by_frame.values() if len(v) == 3]
    same_tp = sum(1 for v in complete if len({v[d][0] for d in detectors}) == 1)
    same_all = sum(1 for v in complete if len({v[d] for d in detectors}) == 1)

    assert len(complete) == 224, f"expected 224 comparable frames, found {len(complete)}"
    assert same_tp == 220, f"README says 220 frames agree on the target, data says {same_tp}"
    assert same_all == 82, f"README says 82 frames agree on all counts, data says {same_all}"


def test_study2_summary_pd_matches_the_per_frame_counts():
    """Same staleness guard for the axes study."""
    frames = _rows(os.path.join(RESULTS_AXES, "axes_frames.csv"))
    summary = _rows(os.path.join(RESULTS_AXES, "axes_summary.csv"))

    agg = {}
    for r in frames:
        key = (r["config"], r["snr_db"])
        tp, tgt = agg.get(key, (0, 0))
        agg[key] = (tp + int(r["detected"]), tgt + int(r["targets"]))

    checked = 0
    for r in summary:
        key = (r["config"], r["snr_db"])
        if key not in agg or not r.get("pd"):
            continue
        tp, tgt = agg[key]
        if tgt == 0:
            continue
        assert tp == int(r["true_positives"]), (
            f"axes_summary.csv true_positives for {key} is {r['true_positives']} but "
            f"axes_frames.csv sums to {tp}. One of the two is stale."
        )
        assert abs(float(r["pd"]) - tp / tgt) < 1e-9, (
            f"axes_summary.csv Pd for {key} is {r['pd']} but axes_frames.csv gives "
            f"{tp}/{tgt}. One of the two is stale."
        )
        checked += 1
    assert checked >= 100, f"only cross-checked {checked} rows; the join is not working"
