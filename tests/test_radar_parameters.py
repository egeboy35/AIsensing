"""Golden-value tests for AIRadarLib.datautil radar parameter derivation.

All expected values are derived from first principles with the same
c = 3e8 m/s the implementation uses.  The function prints a summary with
Unicode glyphs; conftest.py reconfigures stdout to UTF-8 so this cannot
crash Windows cp125x consoles.
"""

import numpy as np
import pytest

datautil = pytest.importorskip("AIRadarLib.datautil")

C = 3e8  # speed of light used by the implementation


def test_radar_parameter_derivation_golden_values():
    sample_rate = 10e6
    chirp_duration = 100e-6
    center_freq = 77e9
    bandwidth = 500e6
    num_chirps = 128

    params = datautil.calculate_radar_parameters(
        sample_rate=sample_rate,
        chirp_duration=chirp_duration,
        center_freq=center_freq,
        bandwidth=bandwidth,
        num_chirps=num_chirps,
    )

    # Range resolution: c / (2B) = 0.3 m
    assert params["range_resolution"] == pytest.approx(C / (2 * bandwidth))
    assert params["range_resolution"] == pytest.approx(0.3)

    # Wavelength: c / fc
    wavelength = C / center_freq
    assert params["wavelength"] == pytest.approx(wavelength)

    # FMCW slope: B / T = 5e12 Hz/s
    assert params["fmcw_slope"] == pytest.approx(5e12)

    # Samples per chirp: fs * T = 1000
    assert params["samples_per_chirp"] == 1000

    # FFT sizes: next powers of two.
    assert params["range_fft_size"] == 1024
    assert params["doppler_fft_size"] == 128

    # Max unambiguous velocity: lambda / (4T)
    assert params["max_unambiguous_velocity"] == pytest.approx(
        wavelength / (4 * chirp_duration)
    )

    # Velocity resolution: lambda / (2 N T)
    assert params["velocity_resolution"] == pytest.approx(
        wavelength / (2 * num_chirps * chirp_duration)
    )

    # Internal consistency: v_res * N == 2 * v_max_unambiguous.
    assert np.isclose(
        params["velocity_resolution"] * num_chirps,
        2 * params["max_unambiguous_velocity"],
    )
