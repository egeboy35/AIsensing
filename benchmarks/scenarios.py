"""Scenario generation for the AIRadar detection benchmark.

The radar physics is **not** reimplemented here.  This module builds a parameter
holder whose attributes mirror ``AIRadarDataset.__init__``, then calls the repo's
own unbound methods against it:

* ``AIRadarDataset.generate_targets``          -- ground-truth target draws
* ``AIRadarDataset._generate_clutter_targets`` -- static + ground clutter
* ``AIRadarDataset._generate_coupling_target`` -- TX leakage at ~0 m
* ``AIRadarDataset.simulate_fmcw_signal``      -- FMCW beat signal + RD map
* ``AIRadarDataset.create_target_mask``        -- ground-truth bin quantization

None of those methods needs a fully constructed dataset object; they only read
attributes.  Constructing ``AIRadarDataset`` directly is avoided because its
``__init__`` unconditionally generates *and* HDF5-saves an entire dataset.

``tests/test_benchmark_metrics.py`` asserts that :class:`RadarParams` reproduces
the real ``AIRadarDataset.__init__`` derivation attribute for attribute, so this
parameter holder cannot silently drift from the repo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.constants import c

from benchmarks.repo_shim import load_repo_modules


class RadarParams:
    """Mirror of the FMCW parameter derivation in ``AIRadarDataset.__init__``.

    Every attribute set here exists on a real ``AIRadarDataset`` instance with the
    same value; the equivalence is enforced by a unit test.
    """

    def __init__(
        self,
        config: dict,
        zero_pad_factor: int = 2,
        max_targets: int = 1,
        clutter_intensity: float = 1.0,
        apply_realistic_effects: bool = False,
        cfar_params: dict | None = None,
    ):
        cfg = config
        self.config = cfg
        self.signal_type = cfg.get("signal_type", "FMCW")
        if self.signal_type != "FMCW":
            raise ValueError(
                f"this harness only covers the FMCW signal path, got {self.signal_type!r}"
            )
        self.fc = cfg.get("fc", 77e9)
        self.B = cfg.get("B", 1.5e9)
        self.T = cfg.get("T_chirp", 40e-6)
        self.Nc = cfg.get("N_chirps", 128)
        self.R_max = cfg.get("R_max", 100)
        self.hardware_model = cfg.get("hardware_model", "generic")
        self.num_rx = int(cfg.get("num_rx", 1))

        self.fs = cfg.get("fs", None)
        if self.fs is not None:
            self.Ns = int(self.fs * self.T)
        else:
            self.Ns = cfg.get("N_samples", 2048)
            self.fs = self.Ns / self.T

        self.max_targets = max_targets
        self.apply_realistic_effects = apply_realistic_effects
        self.clutter_intensity = clutter_intensity
        self.precision = "float32"
        self.drawfig = False

        default_cfar = {
            "num_train": 10,
            "num_guard": 4,
            "threshold_offset": 15,
            "nms_kernel_size": 5,
        }
        config_cfar = cfg.get("cfar_params", default_cfar)
        self.cfar_params = cfar_params if cfar_params is not None else config_cfar

        self.lambda_c = c / self.fc
        self.slope = self.B / self.T
        self.zero_pad = zero_pad_factor * self.Ns

        self.use_array_factor = bool(cfg.get("use_array_factor", False))
        self.array_N = int(cfg.get("array_N", 8))
        steering_angles = cfg.get("steering_angles", [0.0] * self.num_rx)
        if len(steering_angles) < self.num_rx:
            steering_angles = list(steering_angles) + [steering_angles[-1]] * (
                self.num_rx - len(steering_angles)
            )
        self.steering_angles = np.array(steering_angles, dtype=float)

        self.model_dc_offset = bool(cfg.get("model_dc_offset", False))
        self.model_iq_imbalance = bool(cfg.get("model_iq_imbalance", False))
        self.model_phase_noise = bool(cfg.get("model_phase_noise", False))
        self.quantize_adc = bool(cfg.get("quantize_adc", False))
        self.dc_scale = float(cfg.get("dc_scale", 0.02))
        self.iq_gain_std = float(cfg.get("iq_gain_std", 0.02))
        self.iq_phase_std_deg = float(cfg.get("iq_phase_std_deg", 2.0))
        self.phase_noise_std = float(cfg.get("phase_noise_std", 0.01))
        self.cfo_std_hz = float(cfg.get("cfo_std_hz", 200.0))
        self.adc_bits = int(cfg.get("adc_bits", 12))
        self.static_clutter_velocity_std = float(cfg.get("static_clutter_velocity_std", 0.0))
        self.ground_clutter_velocity_std = float(cfg.get("ground_clutter_velocity_std", 0.3))
        self.coupling_rcs_db = float(cfg.get("coupling_rcs_db", -10.0))

        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T

        # Nyquist clip, same as the repo (which also prints a warning here).
        f_nyquist = self.fs / 2.0
        r_max_nyquist = f_nyquist * c / (2 * self.slope)
        self.R_max = min(self.R_max, r_max_nyquist)
        self.R_max_physical = self.R_max

        range_res_fft = (c * self.fs) / (2 * self.slope * self.zero_pad)
        self.range_axis = np.arange(self.zero_pad // 2) * range_res_fft
        max_bin_idx = int(self.R_max / range_res_fft)
        self.num_range_bins = min(self.zero_pad // 2, max_bin_idx)
        self.range_axis = self.range_axis[: self.num_range_bins]
        self.num_doppler_bins = self.Nc

        self.velocity_axis = (
            np.fft.fftshift(np.fft.fftfreq(self.Nc, d=self.T)) * self.lambda_c / 2
        )
        self.range_resolution = c / (2 * self.B)
        self.velocity_resolution = self.lambda_c / (2 * self.Nc * self.T)
        self.max_unambiguous_velocity = self.lambda_c / (4 * self.T)
        self.v_max = self.max_unambiguous_velocity

    # -- derived quantities the harness itself needs --------------------------

    @property
    def range_bin_spacing(self) -> float:
        """Metres between adjacent range bins.

        Note this is *not* ``range_resolution``: with ``zero_pad_factor=2`` the FFT
        grid is oversampled, so the bin spacing is c/(4B) while the true
        (Rayleigh) range resolution is c/(2B).
        """
        return float(self.range_axis[1] - self.range_axis[0])

    @property
    def velocity_bin_spacing(self) -> float:
        return float(self.velocity_axis[1] - self.velocity_axis[0])

    def summary(self) -> dict:
        return {
            "config_name": self.config.get("name"),
            "signal_type": self.signal_type,
            "fc_hz": self.fc,
            "bandwidth_hz": self.B,
            "chirp_duration_s": self.T,
            "sample_rate_hz": self.fs,
            "samples_per_chirp": self.Ns,
            "num_chirps": self.Nc,
            "num_rx": self.num_rx,
            "zero_pad": self.zero_pad,
            "num_range_bins": self.num_range_bins,
            "num_doppler_bins": self.num_doppler_bins,
            "r_max_m": float(self.R_max),
            "v_max_mps": float(self.v_max),
            "range_resolution_m": float(self.range_resolution),
            "range_bin_spacing_m": self.range_bin_spacing,
            "velocity_resolution_mps": float(self.velocity_resolution),
            "velocity_bin_spacing_mps": self.velocity_bin_spacing,
            "max_targets": self.max_targets,
            "clutter_enabled": self.apply_realistic_effects,
            "cfar_params": dict(self.cfar_params),
        }


@dataclass(frozen=True)
class GroundTruthTarget:
    """One ground-truth target, in both continuous and bin coordinates."""

    range_m: float
    velocity_mps: float
    rcs_dbsm: float
    range_bin: int
    doppler_bin: int


@dataclass
class Frame:
    """One simulated radar frame ready for detection."""

    rd_map_complex: np.ndarray  # [num_doppler, num_range] complex
    rd_map_db: np.ndarray  # [num_doppler, num_range] float, repo's own dB map
    rd_map_detector_input: np.ndarray  # [1, 2, num_doppler, num_range]
    targets: tuple[GroundTruthTarget, ...]
    snr_db: float
    trial: int
    peak_snr_db: float  # max-in-window peak minus median noise floor; FLOORS, see below
    #: Noise-subtracted target-bin power ratio, linear.  ``nan`` with no targets.
    #: Unlike ``peak_snr_db`` this does not floor at the noise maximum: it is an
    #: unbiased estimate of (target power)/(noise+clutter power) and goes to ~0 as
    #: the target vanishes, so it can be averaged in the linear domain and stays
    #: monotone in the requested SNR.
    target_bin_snr_linear: float = float("nan")
    #: dB added to the requested ``snr_db`` before it was handed to the repo
    #: simulator, to reference the noise to target-only power instead of whole-scene
    #: power.  0.0 for the repo's own convention.
    snr_correction_db: float = 0.0
    #: ``"full"``, ``"noise_only"`` or ``"clutter_only"``.
    kind: str = "full"


def _seed_globals(*key: int) -> None:
    """Seed both global RNGs the repo code uses.

    ``simulate_fmcw_signal`` calls ``np.random.randn`` and the target/clutter
    generators call both ``random`` and ``np.random``, none of which accept an
    injected Generator.  Reproducibility therefore has to go through the legacy
    global seeds.
    """
    seed = int(np.random.SeedSequence(list(key)).generate_state(1, dtype=np.uint32)[0])
    np.random.seed(seed)
    random.seed(seed)


def rd_map_from_beat(params: RadarParams, beat_rx: np.ndarray) -> np.ndarray:
    """Range-Doppler complex map from a multi-RX beat cube.

    Byte-for-byte the FFT chain of ``AIRadarDataset.simulate_fmcw_signal``, kept
    separately only because the repo function returns the dB magnitude and throws
    the complex map away.  :func:`simulate_frame` asserts that
    ``20*log10(|this| + 1e-6)`` reproduces the repo's own dB map exactly, which is
    what makes the replication trustworthy; the same chain is then reused for the
    target-free calibration frames, which the repo function cannot produce (it adds
    no noise at all when the scene is empty, because it scales the noise by the
    scene power).
    """
    beat_sum = np.sum(beat_rx, axis=0)
    range_fft = np.fft.fft(beat_sum, n=params.zero_pad, axis=1)[:, : params.zero_pad // 2]
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
    return doppler_fft[:, : params.num_range_bins]


def detector_input_from_rd(rd_complex: np.ndarray) -> np.ndarray:
    """Pack a complex RD map into the ``[1, 2, doppler, range]`` layout the repo
    detectors in ``radar_det.py`` expect."""
    detector_input = np.empty((1, 2, *rd_complex.shape), dtype=np.float64)
    detector_input[0, 0] = rd_complex.real
    detector_input[0, 1] = rd_complex.imag
    return detector_input


def repo_db_map(rd_complex: np.ndarray) -> np.ndarray:
    """The repo's own dB-magnitude convention, ``20*log10(|x| + 1e-6)``."""
    return 20 * np.log10(np.abs(rd_complex) + 1e-6)


def quantize_to_bins(params: RadarParams, range_m: float, velocity_mps: float) -> tuple[int, int]:
    """Nearest-bin quantization, identical to ``AIRadarDataset.create_target_mask``."""
    range_bin = int(np.argmin(np.abs(params.range_axis - range_m)))
    doppler_bin = int(np.argmin(np.abs(params.velocity_axis - velocity_mps)))
    return range_bin, doppler_bin


def draw_scenario(params: RadarParams, trial: int, base_seed: int) -> tuple[list[dict], list[dict]]:
    """Draw the ground-truth targets (and clutter) for one trial.

    Depends only on ``(base_seed, trial)``, so the same physical scene is reused at
    every SNR point -- the SNR sweep is paired rather than independently sampled.
    """
    repo = load_repo_modules()
    dataset_cls = repo["AIRadarDataset"]

    _seed_globals(base_seed, 1, trial)
    targets = dataset_cls.generate_targets(params)

    sim_targets = list(targets)
    if params.apply_realistic_effects:
        sim_targets.extend(dataset_cls._generate_clutter_targets(params))
        sim_targets.append(dataset_cls._generate_coupling_target(params))
    return targets, sim_targets


def clutter_to_target_power_db(params: RadarParams, trials: int, base_seed: int) -> float:
    """Mean clutter-plus-coupling RCS power relative to primary-target RCS power, in dB.

    Reported per scenario so the clutter axis is quantified rather than asserted.
    ``-inf`` when clutter is disabled.
    """
    ratios = []
    for trial in range(max(1, trials)):
        targets, sim_targets = draw_scenario(params, trial, base_seed)
        target_power = sum(10 ** (t["rcs"] / 10) for t in targets)
        clutter_power = sum(10 ** (t["rcs"] / 10) for t in sim_targets[len(targets) :])
        if target_power > 0:
            ratios.append(clutter_power / target_power)
    if not ratios or max(ratios) == 0:
        return float("-inf")
    return float(10 * np.log10(np.mean(ratios)))


#: Values accepted by the ``snr_reference`` argument.
SNR_REFERENCES = ("scene", "target")


@dataclass(frozen=True)
class SceneReference:
    """Noise-free beat cubes for one drawn scene, and the powers derived from them.

    ``simulate_fmcw_signal`` sets the AWGN variance to
    ``mean(|windowed beat over the WHOLE scene|^2) / 10**(snr_db/10)``.  With clutter
    enabled, "the whole scene" includes the clutter and the TX-coupling return, so
    raising the clutter RCS raises the injected noise as well and the requested
    ``snr_db`` is not the primary target's SNR.  Recovering the primary target's own
    reference power needs a second, target-only, noise-free simulation -- which is
    exactly what this holder caches.

    Both cubes come from the repo simulator called with ``snr_db=inf`` (noise power
    ``signal_power / inf == 0.0``, so the AWGN term is identically zero).  The
    simulator is linear in the target list for the configs benchmarked here
    (``num_rx=1``, ``use_array_factor=False``, ``hardware_model='generic'``), which a
    test verifies, so ``beat_all - beat_primary`` is exactly the clutter-plus-coupling
    contribution.
    """

    beat_all: np.ndarray
    beat_primary: np.ndarray
    power_all: float
    power_primary: float

    @property
    def scene_to_target_power_db(self) -> float:
        """dB by which whole-scene mean power exceeds primary-target-only mean power.

        This is exactly the amount by which the repo's scene-referenced convention
        inflates the injected noise relative to a target-referenced one, so adding it
        to the requested ``snr_db`` holds the primary target's SNR fixed.
        """
        if self.power_primary <= 0:
            return 0.0
        return float(10 * np.log10(self.power_all / self.power_primary))


def scene_reference(
    params: RadarParams, trial: int, base_seed: int
) -> tuple[list[dict], list[dict], SceneReference]:
    """Draw one scene and simulate it noise-free, whole-scene and target-only."""
    repo = load_repo_modules()
    dataset_cls = repo["AIRadarDataset"]
    targets, sim_targets = draw_scenario(params, trial, base_seed)

    # Seeded for form only: with snr_db=inf the noise term is identically zero, so
    # these two calls are deterministic regardless of RNG state.
    _seed_globals(base_seed, 3, trial)
    beat_all, _ = dataset_cls.simulate_fmcw_signal(params, sim_targets, snr_db=np.inf)
    _seed_globals(base_seed, 4, trial)
    beat_primary, _ = dataset_cls.simulate_fmcw_signal(params, targets, snr_db=np.inf)

    return (
        targets,
        sim_targets,
        SceneReference(
            beat_all=beat_all,
            beat_primary=beat_primary,
            power_all=float(np.mean(np.abs(beat_all) ** 2)),
            power_primary=float(np.mean(np.abs(beat_primary) ** 2)),
        ),
    )


def _ground_truth(
    params: RadarParams, targets: list[dict]
) -> tuple[GroundTruthTarget, ...]:
    gt = []
    for t in targets:
        r_bin, d_bin = quantize_to_bins(params, t["range"], t["velocity"])
        gt.append(
            GroundTruthTarget(
                range_m=float(t["range"]),
                velocity_mps=float(t["velocity"]),
                rcs_dbsm=float(t["rcs"]),
                range_bin=r_bin,
                doppler_bin=d_bin,
            )
        )
    return tuple(gt)


def _resolve_correction(reference: SceneReference, snr_reference: str) -> float:
    if snr_reference not in SNR_REFERENCES:
        raise ValueError(
            f"snr_reference must be one of {SNR_REFERENCES}, got {snr_reference!r}"
        )
    if snr_reference == "scene":
        return 0.0
    return reference.scene_to_target_power_db


def simulate_frame(
    params: RadarParams,
    trial: int,
    snr_index: int,
    snr_db: float,
    base_seed: int,
    snr_reference: str = "scene",
    reference: SceneReference | None = None,
) -> Frame:
    """Simulate one frame with the repo's FMCW model and build detector inputs.

    ``snr_reference="scene"`` reproduces the repo exactly: the requested ``snr_db``
    is measured against whole-scene power.  ``snr_reference="target"`` adds
    :attr:`SceneReference.scene_to_target_power_db` to the requested value before
    calling the simulator, so that the *primary target's* SNR is what the sweep axis
    means and the clutter axis stops doubling as an SNR axis.  Nothing in the repo is
    modified either way; only the number handed to ``snr_db`` changes.
    """
    repo = load_repo_modules()
    dataset_cls = repo["AIRadarDataset"]

    if reference is None:
        targets, sim_targets, reference = scene_reference(params, trial, base_seed)
    else:
        targets, sim_targets = draw_scenario(params, trial, base_seed)

    correction_db = _resolve_correction(reference, snr_reference)

    _seed_globals(base_seed, 2, trial, snr_index)
    beat_rx, rd_map_db = dataset_cls.simulate_fmcw_signal(
        params, sim_targets, snr_db=snr_db + correction_db
    )

    rd_complex = rd_map_from_beat(params, beat_rx)
    reconstructed_db = repo_db_map(rd_complex)
    if not np.allclose(reconstructed_db, rd_map_db, rtol=0, atol=1e-9):
        raise AssertionError(
            "complex RD map does not reproduce the repo's dB map; the FFT chain "
            "in simulate_fmcw_signal must have changed"
        )

    gt = _ground_truth(params, targets)

    return Frame(
        rd_map_complex=rd_complex,
        rd_map_db=rd_map_db,
        rd_map_detector_input=detector_input_from_rd(rd_complex),
        targets=gt,
        snr_db=float(snr_db),
        trial=trial,
        peak_snr_db=measure_peak_snr_db(params, rd_map_db, list(gt)),
        target_bin_snr_linear=measure_target_bin_snr_linear(rd_complex, list(gt)),
        snr_correction_db=float(correction_db),
        kind="full",
    )


def simulate_target_free_frame(
    params: RadarParams,
    trial: int,
    snr_index: int,
    snr_db: float,
    base_seed: int,
    kind: str,
    snr_reference: str = "scene",
    reference: SceneReference | None = None,
) -> Frame:
    """Simulate one frame with the primary target(s) removed.

    ``kind="noise_only"`` subtracts the whole noise-free scene, leaving exactly the
    AWGN realization the repo simulator drew, at exactly the variance it chose for
    the corresponding target-present frame.  ``kind="clutter_only"`` subtracts only
    the primary target, leaving clutter + TX coupling + that same noise.

    This is the only way to get a target-free scene out of the repo model:
    ``simulate_fmcw_signal`` computes the noise variance as
    ``mean(|beat|^2) / snr_linear`` and skips the noise term entirely when the scene
    is empty, so calling it with ``targets=[]`` returns an all-zero map rather than a
    noise-only one.
    """
    if kind not in ("noise_only", "clutter_only"):
        raise ValueError(f"kind must be 'noise_only' or 'clutter_only', got {kind!r}")
    repo = load_repo_modules()
    dataset_cls = repo["AIRadarDataset"]

    if reference is None:
        _, sim_targets, reference = scene_reference(params, trial, base_seed)
    else:
        _, sim_targets = draw_scenario(params, trial, base_seed)

    correction_db = _resolve_correction(reference, snr_reference)

    _seed_globals(base_seed, 2, trial, snr_index)
    beat_rx, _ = dataset_cls.simulate_fmcw_signal(
        params, sim_targets, snr_db=snr_db + correction_db
    )
    subtract = reference.beat_all if kind == "noise_only" else reference.beat_primary
    residual = beat_rx - subtract

    rd_complex = rd_map_from_beat(params, residual)
    return Frame(
        rd_map_complex=rd_complex,
        rd_map_db=repo_db_map(rd_complex),
        rd_map_detector_input=detector_input_from_rd(rd_complex),
        targets=(),
        snr_db=float(snr_db),
        trial=trial,
        peak_snr_db=float("nan"),
        target_bin_snr_linear=float("nan"),
        snr_correction_db=float(correction_db),
        kind=kind,
    )


def measure_target_bin_snr_linear(
    rd_complex: np.ndarray,
    targets: list[GroundTruthTarget],
    guard_bins: int = 6,
) -> float:
    """Noise-subtracted target-to-background power ratio at the ground-truth bin.

    ``(P_target_bin - P_background) / P_background`` in the *linear* power domain,
    averaged over targets, where ``P_background`` is the mean power of cells at least
    ``guard_bins`` away from every target in both axes.  With clutter enabled the
    background includes the clutter, so this is a target-to-(noise+clutter) ratio.

    Two properties :func:`measure_peak_snr_db` does not have: it is an unbiased
    estimate rather than a maximum (so it does not floor at "the largest noise cell
    in the window" once the target is buried), and it is proportional to the
    requested SNR, so averaging it over trials in the linear domain and only then
    converting to dB gives a column that keeps decreasing instead of flattening.
    Negative values are possible, and meaningful, at very low SNR.
    """
    if not targets:
        return float("nan")
    power = np.abs(rd_complex) ** 2
    mask = np.ones(power.shape, dtype=bool)
    for t in targets:
        d0 = max(0, t.doppler_bin - guard_bins)
        d1 = min(power.shape[0], t.doppler_bin + guard_bins + 1)
        r0 = max(0, t.range_bin - guard_bins)
        r1 = min(power.shape[1], t.range_bin + guard_bins + 1)
        mask[d0:d1, r0:r1] = False
    if not mask.any():  # pragma: no cover - defensive
        return float("nan")
    background = float(np.mean(power[mask]))
    if background <= 0:  # pragma: no cover - defensive
        return float("nan")
    ratios = [
        (float(power[t.doppler_bin, t.range_bin]) - background) / background
        for t in targets
    ]
    return float(np.mean(ratios))


def measure_peak_snr_db(
    params: RadarParams,
    rd_map_db: np.ndarray,
    targets: list[GroundTruthTarget],
    guard_bins: int = 6,
) -> float:
    """Measured target-peak-to-noise-floor ratio in the range-Doppler map, in dB.

    ``max(rd_map_db)`` over a +/-``guard_bins`` window around each target, minus the
    median dB level of the cells outside every such window.

    **This column has a floor and must not be read below it.**  Once the target is
    weaker than the strongest noise cell in its own window, the "peak" is that noise
    cell, so the value stops tracking the target and settles at the
    max-of-window-minus-median level of pure noise -- about +9.2 dB for the default
    ``config_phaser`` sweep (13x13-cell window, Rayleigh magnitudes).  It is also not
    guaranteed monotone once clutter enters the median.  Kept because it is a direct
    reading of the map the detector sees, but the headline SNR column is derived from
    :func:`measure_target_bin_snr_linear`, which degrades gracefully.
    """
    if not targets:
        return float("nan")
    mask = np.ones(rd_map_db.shape, dtype=bool)
    peaks = []
    for t in targets:
        d0 = max(0, t.doppler_bin - guard_bins)
        d1 = min(rd_map_db.shape[0], t.doppler_bin + guard_bins + 1)
        r0 = max(0, t.range_bin - guard_bins)
        r1 = min(rd_map_db.shape[1], t.range_bin + guard_bins + 1)
        mask[d0:d1, r0:r1] = False
        peaks.append(float(np.max(rd_map_db[d0:d1, r0:r1])))
    if not mask.any():  # pragma: no cover - defensive
        return float("nan")
    noise_floor_db = float(np.median(rd_map_db[mask]))
    return float(np.mean(peaks) - noise_floor_db)
