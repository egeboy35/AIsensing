"""SNR calibration test for AIRadarLib.channel_simulation.add_noise.

add_noise draws from the global numpy RNG, so the test seeds
np.random.seed explicitly.
"""

import numpy as np
import pytest

channel_simulation = pytest.importorskip("AIRadarLib.channel_simulation")


def test_add_noise_snr_calibration():
    num_samples = 200_000
    t = np.arange(num_samples)
    # Unit-power complex exponential.
    signal = np.exp(1j * 2 * np.pi * 0.05 * t)
    assert np.mean(np.abs(signal) ** 2) == pytest.approx(1.0)

    np.random.seed(1234)
    noisy = channel_simulation.add_noise(signal, snr_db=20.0)

    noise = noisy - signal
    measured_snr_db = 10 * np.log10(
        np.mean(np.abs(signal) ** 2) / np.mean(np.abs(noise) ** 2)
    )
    # With 200k samples the empirical SNR concentrates tightly around the
    # requested 20 dB.
    assert abs(measured_snr_db - 20.0) < 0.5
