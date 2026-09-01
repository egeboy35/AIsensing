"""Detector adapters, and the common threshold axis that makes them comparable.

Every detector benchmarked here is repo code, called unmodified.  The adapters
only translate between the harness's frame representation and each function's own
calling convention, and normalize the returned dicts into
:class:`benchmarks.metrics.Point`.

Why a common threshold axis
---------------------------
The three repo detectors do not share a threshold parameterization at all:

* ``cfar_2d_numpy`` has **no** threshold argument -- ``noise_est_dB + 12`` is
  hard-coded.
* ``_cfar_2d_custom`` takes ``threshold_offset`` in dB above a dB-domain noise
  estimate.
* ``cfar_2d_advanced`` takes a design ``pfa``, converts it to a linear-power
  multiplier, and then additionally applies a ``min_snr_db`` post-filter.

Compared "as shipped" they therefore run at three different distances above the
local noise level, and any Pd ranking mostly reports that distance rather than
anything about the detectors.  :class:`DetectorSpec` maps all three onto one axis
-- **effective threshold in dB above the local mean noise power** -- so a common
operating point can be solved for (see :mod:`benchmarks.calibration`).

The conversions are closed-form and each is verified by a test:

``_cfar_2d_custom`` / ``cfar_2d_numpy`` (dB domain)
    The noise estimate is an arithmetic mean of ``20*log10|x|`` values, i.e. an
    estimate of the *geometric* mean of power.  For complex-Gaussian noise
    ``E[10*log10|x|^2] = 10*log10(E|x|^2) - 10*gamma/ln(10)``, so an offset of
    ``theta`` dB above that estimate sits ``theta - 2.507`` dB above the mean noise
    power.  (GO-CFAR takes the larger of two such averages, which adds a further
    ~0.03 dB that this closed form deliberately does not model; an independent
    measurement of the shipped settings gives 9.52 / 12.52 dB against the 9.49 /
    12.49 dB predicted here.)

``cfar_2d_advanced`` (linear power domain)
    ``threshold = noise_mean * alpha(pfa, N)`` with
    ``alpha = N * (pfa**(-1/N) - 1)`` and ``N = w*(w - 2*num_guard - 1)`` training
    cells per GO branch, ``w = 2*(num_train + num_guard) + 1``.  The shipped
    ``min_snr_db`` post-filter demands another ``min_snr_db`` dB on top, so the
    effective threshold is ``10*log10(alpha) + min_snr_db``.

Reaching ``cfar_2d_numpy``'s threshold at all
---------------------------------------------
Its ``+12`` dB is a literal in repo code, and this directory is additive only.  The
threshold is still reachable *exactly*, without touching the repo, because the
detector is a linear operator in the dB domain: it declares a detection where
``mag_db > mean_dB(neighbourhood) + 12``.  Raising the input magnitudes to a power
``p`` multiplies both sides' dB values by ``p``, so the test becomes
``mag_db > mean_dB(neighbourhood) + 12/p`` -- i.e. feeding ``|x|**p`` is *identical*
to running the detector with an offset of ``12/p`` dB.  Non-maximum suppression and
the range/velocity gates are unaffected because ``p > 0`` is strictly monotone and
they act on orderings and indices.  :func:`warp_magnitude` does this, and a test
asserts that the resulting detection set equals ``_cfar_2d_custom``'s at
``threshold_offset=12/p`` on the same map -- a check that uses only repo code and no
reimplementation of CFAR.

The plug-in point for a *learned* detector is :class:`Detector`: supply a ``run``
callable with the signature ``(frame, params) -> list[Point]``.  A torch model
wrapper drops in here with no change to the metric or reporting code -- see the
"Adding a detector" section of ``benchmarks/README.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from benchmarks.metrics import Point
from benchmarks.repo_shim import assert_no_stub_dependency, load_repo_modules

#: ``10 * gamma / ln(10)``: how far the mean of ``10*log10(power)`` sits *below*
#: ``10*log10(mean power)`` for exponentially distributed power (Rayleigh magnitude).
DB_DOMAIN_MEAN_BIAS_DB = 10.0 * 0.5772156649015329 / math.log(10.0)

#: Magnitude floor used by :func:`warp_magnitude`, chosen 9 orders of magnitude above
#: the ``1e-12`` epsilon inside ``cfar_2d_numpy`` so the warp stays exactly monotone.
_WARP_MIN_MAGNITUDE = 1e-3


# --------------------------------------------------------------------------- #
# closed-form threshold conversions
# --------------------------------------------------------------------------- #


def go_training_cells(num_train: int, num_guard: int) -> int:
    """Training cells per GO/SO branch, as the repo's kernels actually count them.

    ``full_kernel`` is ``w x w`` with ``w = 2*(num_train + num_guard) + 1``; the guard
    block removes ``(2*num_guard + 1)**2`` cells; the horizontal branch then zeroes
    the ``2*num_guard + 1`` full rows through the guard block.  What survives is
    ``w**2 - (2*num_guard + 1)*w``.  The vertical branch is the transpose, so both
    branches have the same count.
    """
    w = 2 * (int(num_train) + int(num_guard)) + 1
    return int(w * (w - 2 * int(num_guard) - 1))


def alpha_from_pfa(num_cells: int, pfa: float) -> float:
    """``cfar_2d_advanced``'s own Pfa-to-multiplier formula, reproduced exactly."""
    n = max(1, int(num_cells))
    return float(n * (pfa ** (-1.0 / n) - 1.0))


def pfa_from_alpha(num_cells: int, alpha: float) -> float:
    """Inverse of :func:`alpha_from_pfa`."""
    n = max(1, int(num_cells))
    if alpha <= 0:
        return 1.0
    return float((1.0 + alpha / n) ** (-n))


def warp_magnitude(detector_input: np.ndarray, exponent: float) -> np.ndarray:
    """Return the ``[1, 2, doppler, range]`` map with magnitudes raised to ``exponent``.

    Phase is preserved and the whole map is rescaled by a constant gain (which a
    dB-domain CFAR is invariant to) so that the smallest magnitude stays far above the
    ``1e-12`` epsilon inside ``cfar_2d_numpy``.  See the module docstring: this is an
    exact reparameterization of that detector's hard-coded ``+12`` dB offset to
    ``12 / exponent`` dB.
    """
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    rd = detector_input[0, 0] + 1j * detector_input[0, 1]
    mag = np.abs(rd)
    peak = float(np.max(mag))
    if peak <= 0:  # pragma: no cover - defensive
        return detector_input
    mag_safe = np.maximum(mag, peak * 1e-30)
    warped = mag_safe**exponent
    warped *= _WARP_MIN_MAGNITUDE / float(np.min(warped))
    out = np.empty_like(detector_input)
    phase = np.exp(1j * np.angle(rd))
    out[0, 0] = (warped * phase).real
    out[0, 1] = (warped * phase).imag
    return out


# --------------------------------------------------------------------------- #
# detector objects
# --------------------------------------------------------------------------- #


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
    #: ``"as_shipped"`` or a calibrated-variant label.
    variant: str = "as_shipped"
    #: Name and value of the native threshold knob this instance runs at.
    knob: str = ""
    knob_value: float = float("nan")
    #: Closed-form threshold above the local mean noise power, dB.  Comparable
    #: across detectors; see the module docstring.
    effective_threshold_db: float = float("nan")

    @property
    def key(self) -> str:
        return f"{self.name}@{self.variant}"

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


@dataclass(frozen=True)
class DetectorSpec:
    """Static description of one repo detector, plus its threshold parameterization."""

    name: str
    source: str
    description: str
    #: Name of the native knob that moves the threshold.
    knob: str
    #: Value of that knob as the repository configures it.
    shipped_knob: Callable[[object], float]
    range_gate_low_m: float
    _effective_from_native: Callable[[float, object], float]
    _native_from_effective: Callable[[float, object], float]
    _build: Callable[[object, float, dict], Callable[[object, object], list[Point]]]

    def effective_threshold_db(self, native: float, params) -> float:
        return float(self._effective_from_native(native, params))

    def native_knob(self, effective_threshold_db: float, params) -> float:
        return float(self._native_from_effective(effective_threshold_db, params))

    def build(self, params, native: float, variant: str, *, mtd: bool = False) -> Detector:
        extra = {"mtd": bool(mtd)}
        run, recorded = self._build(params, native, extra)
        return Detector(
            name=self.name,
            source=self.source,
            description=self.description,
            run=run,
            params=recorded,
            range_gate_low_m=self.range_gate_low_m,
            variant=variant,
            knob=self.knob,
            knob_value=float(native),
            effective_threshold_db=self.effective_threshold_db(native, params),
        )


# --------------------------------------------------------------------------- #
# the three repo detectors
# --------------------------------------------------------------------------- #

#: Hard-coded dB offset inside ``cfar_2d_numpy``.
NUMPY_GO_OFFSET_DB = 12.0
#: ``min_snr_db`` post-filter value ``cfar_2d_advanced`` ships with.
ADVANCED_MIN_SNR_DB = 6.0


def _common_kwargs(params) -> dict:
    cfg = params.cfar_params
    return {
        "num_train": int(cfg.get("num_train", 10)),
        "num_guard": int(cfg.get("num_guard", 4)),
        "range_res": params.range_bin_spacing,
        "doppler_res": params.velocity_bin_spacing,
        "max_range": float(params.R_max),
        "max_speed": float(params.v_max),
        "nms_kernel_size": int(cfg.get("nms_kernel_size", 5)),
    }


def _numpy_effective(exponent: float, params) -> float:
    return NUMPY_GO_OFFSET_DB / float(exponent) - DB_DOMAIN_MEAN_BIAS_DB


def _numpy_native(effective_db: float, params) -> float:
    return NUMPY_GO_OFFSET_DB / (float(effective_db) + DB_DOMAIN_MEAN_BIAS_DB)


def _build_numpy(params, exponent: float, extra: dict):
    repo = load_repo_modules()
    fn = repo["radar_det"].cfar_2d_numpy
    kwargs = {**_common_kwargs(params), "method": "GO", "estimate_aoa": False}
    exponent = float(exponent)
    identity = exponent == 1.0

    def run(frame, _params):
        data = frame.rd_map_detector_input
        if not identity:
            data = warp_magnitude(data, exponent)
        return _to_points(fn(data, **kwargs))

    recorded = {
        **kwargs,
        "hard_coded_offset_db": NUMPY_GO_OFFSET_DB,
        "magnitude_warp_exponent": exponent,
        "equivalent_offset_db": NUMPY_GO_OFFSET_DB / exponent,
    }
    return run, recorded


def _advanced_effective(pfa: float, params) -> float:
    n = go_training_cells(
        params.cfar_params.get("num_train", 10), params.cfar_params.get("num_guard", 4)
    )
    return 10.0 * math.log10(alpha_from_pfa(n, float(pfa))) + ADVANCED_MIN_SNR_DB


def _advanced_native(effective_db: float, params) -> float:
    n = go_training_cells(
        params.cfar_params.get("num_train", 10), params.cfar_params.get("num_guard", 4)
    )
    alpha = 10.0 ** ((float(effective_db) - ADVANCED_MIN_SNR_DB) / 10.0)
    return pfa_from_alpha(n, alpha)


def _build_advanced(params, pfa: float, extra: dict):
    repo = load_repo_modules()
    fn = repo["radar_det"].cfar_2d_advanced
    kwargs = {
        **_common_kwargs(params),
        "method": "GO",
        "pfa": float(pfa),
        "estimate_aoa": False,
        "suppress_zero_doppler_width": 0,
        "min_snr_db": ADVANCED_MIN_SNR_DB,
    }

    def run(frame, _params):
        return _to_points(fn(frame.rd_map_detector_input, **kwargs))

    return run, dict(kwargs)


def _custom_effective(threshold_offset: float, params) -> float:
    return float(threshold_offset) - DB_DOMAIN_MEAN_BIAS_DB


def _custom_native(effective_db: float, params) -> float:
    return float(effective_db) + DB_DOMAIN_MEAN_BIAS_DB


def _build_custom(params, threshold_offset: float, extra: dict):
    repo = load_repo_modules()
    fn = repo["AIRadarDataset"]._cfar_2d_custom
    kwargs = {
        **_common_kwargs(params),
        "threshold_offset": float(threshold_offset),
        "mtd": bool(extra.get("mtd", False)),
    }

    def run(frame, _params):
        return _to_points(fn(None, frame.rd_map_db, **kwargs))

    return run, dict(kwargs)


SPECS: tuple[DetectorSpec, ...] = (
    DetectorSpec(
        name="cfar_numpy_go",
        source="AIRadar/AIRadarLib/radar_det.py::cfar_2d_numpy",
        description=(
            "GO-CFAR on the dB-magnitude map. The threshold offset is the hard-coded "
            "+12 dB; other operating points are reached by the exact magnitude warp "
            "described in benchmarks/detectors.py, which is equivalent to an offset "
            "of 12/exponent dB."
        ),
        knob="magnitude_warp_exponent",
        shipped_knob=lambda params: 1.0,
        range_gate_low_m=1.0,
        _effective_from_native=_numpy_effective,
        _native_from_effective=_numpy_native,
        _build=_build_numpy,
    ),
    DetectorSpec(
        name="cfar_advanced_go",
        source="AIRadar/AIRadarLib/radar_det.py::cfar_2d_advanced",
        description=(
            "GO-CFAR on linear power with the threshold multiplier derived from the "
            "requested Pfa, connected-component pruning, and the shipped "
            "min_snr_db=6 post-filter (which adds 6 dB to the effective threshold)."
        ),
        knob="pfa",
        shipped_knob=lambda params: 1e-5,
        range_gate_low_m=1.0,
        _effective_from_native=_advanced_effective,
        _native_from_effective=_advanced_native,
        _build=_build_advanced,
    ),
    DetectorSpec(
        name="cfar_custom_datasetv8",
        source="AIRadar/AIradar_datasetv8.py::AIRadarDataset._cfar_2d_custom",
        description=(
            "The detector the dataset pipeline actually calls: GO-CFAR on the dB map "
            "with the config's threshold_offset. Called unbound with self=None, which "
            "it never dereferences."
        ),
        knob="threshold_offset",
        shipped_knob=lambda params: float(params.cfar_params.get("threshold_offset", 15)),
        range_gate_low_m=0.5,
        _effective_from_native=_custom_effective,
        _native_from_effective=_custom_native,
        _build=_build_custom,
    ),
)

SPECS_BY_NAME = {spec.name: spec for spec in SPECS}


def _guard(params) -> None:
    repo = load_repo_modules()
    assert_no_stub_dependency(
        repo["radar_det"].cfar_2d_numpy,
        repo["radar_det"].cfar_2d_advanced,
        repo["AIRadarDataset"]._cfar_2d_custom,
    )
    check_velocity_convention(params)


def build_detectors(
    params,
    *,
    mtd: bool = False,
    pfa: float | None = None,
    settings: dict[str, float] | None = None,
    variant: str = "as_shipped",
) -> list[Detector]:
    """Instantiate the three pure-numpy CFAR detectors that ship in the repo.

    With no ``settings``, every detector runs at the knob value the repository
    configures (``variant="as_shipped"``).  ``settings`` maps detector name to a
    native knob value, e.g. ``{"cfar_custom_datasetv8": 11.9}``; ``pfa`` is a
    shorthand override for ``cfar_advanced_go`` kept for the original CLI flag.
    """
    _guard(params)
    resolved: dict[str, float] = {}
    for spec in SPECS:
        if settings is not None and spec.name in settings:
            resolved[spec.name] = float(settings[spec.name])
        elif pfa is not None and spec.name == "cfar_advanced_go":
            resolved[spec.name] = float(pfa)
        else:
            resolved[spec.name] = float(spec.shipped_knob(params))
    return [spec.build(params, resolved[spec.name], variant, mtd=mtd) for spec in SPECS]


def build_detector(params, name: str, native: float, variant: str, *, mtd: bool = False):
    """One detector, at one native knob value."""
    _guard(params)
    return SPECS_BY_NAME[name].build(params, native, variant, mtd=mtd)
