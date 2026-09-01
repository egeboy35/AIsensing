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
    peak_snr_db: float  # measured target-peak-to-noise-floor in the RD map


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


def simulate_frame(
    params: RadarParams,
    trial: int,
    snr_index: int,
    snr_db: float,
    base_seed: int,
) -> Frame:
    """Simulate one frame with the repo's FMCW model and build detector inputs."""
    repo = load_repo_modules()
    dataset_cls = repo["AIRadarDataset"]

    targets, sim_targets = draw_scenario(params, trial, base_seed)

    _seed_globals(base_seed, 2, trial, snr_index)
    beat_rx, rd_map_db = dataset_cls.simulate_fmcw_signal(params, sim_targets, snr_db=snr_db)

    # Same FFT chain as simulate_fmcw_signal, but keeping the complex result --
    # the repo function only returns the dB magnitude, and cfar_2d_numpy /
    # cfar_2d_advanced take a complex (real, imag) map.  The equality assertion
    # below is what makes this replication trustworthy.
    beat_sum = np.sum(beat_rx, axis=0)
    range_fft = np.fft.fft(beat_sum, n=params.zero_pad, axis=1)[:, : params.zero_pad // 2]
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
    rd_complex = doppler_fft[:, : params.num_range_bins]

    reconstructed_db = 20 * np.log10(np.abs(rd_complex) + 1e-6)
    if not np.allclose(reconstructed_db, rd_map_db, rtol=0, atol=1e-9):
        raise AssertionError(
            "complex RD map does not reproduce the repo's dB map; the FFT chain "
            "in simulate_fmcw_signal must have changed"
        )

    detector_input = np.empty((1, 2, *rd_complex.shape), dtype=np.float64)
    detector_input[0, 0] = rd_complex.real
    detector_input[0, 1] = rd_complex.imag

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

    return Frame(
        rd_map_complex=rd_complex,
        rd_map_db=rd_map_db,
        rd_map_detector_input=detector_input,
        targets=tuple(gt),
        snr_db=float(snr_db),
        trial=trial,
        peak_snr_db=measure_peak_snr_db(params, rd_map_db, gt),
    )


def measure_peak_snr_db(
    params: RadarParams,
    rd_map_db: np.ndarray,
    targets: list[GroundTruthTarget],
    guard_bins: int = 6,
) -> float:
    """Measured target-peak-to-noise-floor ratio in the range-Doppler map, in dB.

    The noise floor is the median dB level over cells that are at least
    ``guard_bins`` away (in both axes) from every ground-truth target.  Reported
    as a diagnostic because the ``snr_db`` knob of ``simulate_fmcw_signal`` is a
    *time-domain* SNR: coherent range+Doppler integration adds tens of dB before
    the detector ever sees the data.
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
