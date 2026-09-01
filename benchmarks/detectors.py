"""Detector adapters.

Every detector benchmarked here is repo code, called unmodified.  The adapters
only translate between the harness's frame representation and each function's
own calling convention, and normalize the returned dicts into
:class:`benchmarks.metrics.Point`.

The plug-in point for a *learned* detector is :class:`Detector`: supply a
``run`` callable with the signature ``(frame, params) -> list[Point]``.  A torch
model wrapper drops in here with no change to the metric or reporting code --
see the "Adding a detector" section of ``benchmarks/README.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from benchmarks.metrics import Point
from benchmarks.repo_shim import assert_no_stub_dependency, load_repo_modules


@dataclass
class Detector:
    """A named, configured detector under test."""

    name: str
    source: str
    description: str
    run: Callable[[object, object], list[Point]]
    params: dict = field(default_factory=dict)
    #: Lower range gate the detector itself applies, in metres (used to compute
    #: the eligible-cell denominator for the false-alarm rate).
    range_gate_low_m: float = 1.0

    def eligible_cells(self, params) -> int:
        """Number of RD-map cells this detector can possibly report a peak in.

        Mirrors the detector's own ``range_gate_low < range_m < max_range`` and
        ``abs(velocity) < max_speed`` filters, so the false-alarm rate has an
        honest denominator.
        """
        dr = params.range_bin_spacing
        dv = params.velocity_bin_spacing
        n_r = sum(
            1
            for r in range(params.num_range_bins)
            if self.range_gate_low_m < r * dr < params.R_max
        )
        centre = params.num_doppler_bins // 2
        n_d = sum(
            1
            for d in range(params.num_doppler_bins)
            if abs((d - centre) * dv) < params.v_max
        )
        return n_r * n_d


def _to_points(raw: list[dict]) -> list[Point]:
    return [
        Point(
            range_bin=int(d["range_idx"]),
            doppler_bin=int(d["doppler_idx"]),
            range_m=float(d["range_m"]),
            velocity_mps=float(d["velocity_mps"]),
        )
        for d in raw
    ]


def check_velocity_convention(params) -> None:
    """Verify the detectors' velocity formula matches the dataset's velocity axis.

    All three repo detectors report ``velocity = (doppler_idx - num_doppler//2) *
    doppler_res``, while the dataset quantizes ground truth against
    ``velocity_axis``.  If those two disagree the benchmark would silently measure
    an offset instead of an error.
    """
    centre = params.num_doppler_bins // 2
    dv = params.velocity_bin_spacing
    implied = (np.arange(params.num_doppler_bins) - centre) * dv
    if not np.allclose(implied, params.velocity_axis, atol=1e-9):
        raise AssertionError(
            "detector velocity formula does not match RadarParams.velocity_axis"
        )


def build_detectors(params, *, mtd: bool = False, pfa: float = 1e-5) -> list[Detector]:
    """Instantiate the three pure-numpy CFAR detectors that ship in the repo."""
    repo = load_repo_modules()
    radar_det = repo["radar_det"]
    dataset_cls = repo["AIRadarDataset"]

    cfar_2d_numpy = radar_det.cfar_2d_numpy
    cfar_2d_advanced = radar_det.cfar_2d_advanced
    cfar_2d_custom = dataset_cls._cfar_2d_custom

    assert_no_stub_dependency(cfar_2d_numpy, cfar_2d_advanced, cfar_2d_custom)
    check_velocity_convention(params)

    dr = params.range_bin_spacing
    dv = params.velocity_bin_spacing
    cfg = params.cfar_params

    numpy_kwargs = {
        "num_train": int(cfg.get("num_train", 10)),
        "num_guard": int(cfg.get("num_guard", 4)),
        "range_res": dr,
        "doppler_res": dv,
        "max_range": float(params.R_max),
        "max_speed": float(params.v_max),
        "method": "GO",
        "nms_kernel_size": int(cfg.get("nms_kernel_size", 5)),
        "estimate_aoa": False,
    }

    advanced_kwargs = {
        "num_train": int(cfg.get("num_train", 10)),
        "num_guard": int(cfg.get("num_guard", 4)),
        "range_res": dr,
        "doppler_res": dv,
        "max_range": float(params.R_max),
        "max_speed": float(params.v_max),
        "method": "GO",
        "pfa": float(pfa),
        "nms_kernel_size": int(cfg.get("nms_kernel_size", 5)),
        "estimate_aoa": False,
        "suppress_zero_doppler_width": 0,
        "min_snr_db": 6.0,
    }

    custom_kwargs = {
        "num_train": int(cfg.get("num_train", 10)),
        "num_guard": int(cfg.get("num_guard", 4)),
        "range_res": dr,
        "doppler_res": dv,
        "max_range": float(params.R_max),
        "max_speed": float(params.v_max),
        "threshold_offset": float(cfg.get("threshold_offset", 15)),
        "nms_kernel_size": int(cfg.get("nms_kernel_size", 5)),
        "mtd": bool(mtd),
    }

    return [
        Detector(
            name="cfar_numpy_go",
            source="AIRadar/AIRadarLib/radar_det.py::cfar_2d_numpy",
            description=(
                "GO-CFAR on the dB-magnitude map with a hard-coded +12 dB threshold "
                "offset (the function accepts no Pfa argument)."
            ),
            run=lambda frame, p: _to_points(
                cfar_2d_numpy(frame.rd_map_detector_input, **numpy_kwargs)
            ),
            params={
                **numpy_kwargs,
                "threshold_offset_db": 12.0,
                "note": "offset hard-coded in repo",
            },
            range_gate_low_m=1.0,
        ),
        Detector(
            name="cfar_advanced_go",
            source="AIRadar/AIRadarLib/radar_det.py::cfar_2d_advanced",
            description=(
                "GO-CFAR on linear power with the threshold multiplier derived from "
                "the requested Pfa, connected-component pruning, and the shipped "
                "min_snr_db=6 post-filter."
            ),
            run=lambda frame, p: _to_points(
                cfar_2d_advanced(frame.rd_map_detector_input, **advanced_kwargs)
            ),
            params={**advanced_kwargs},
            range_gate_low_m=1.0,
        ),
        Detector(
            name="cfar_custom_datasetv8",
            source="AIRadar/AIradar_datasetv8.py::AIRadarDataset._cfar_2d_custom",
            description=(
                "The detector the dataset pipeline actually calls: GO-CFAR on the dB "
                "map with the config's threshold_offset. Called unbound with self=None, "
                "which it never dereferences."
            ),
            run=lambda frame, p: _to_points(
                cfar_2d_custom(None, frame.rd_map_db, **custom_kwargs)
            ),
            params={**custom_kwargs},
            range_gate_low_m=0.5,
        ),
    ]
