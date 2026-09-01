"""Waveform generation and spectrum analysis tests for AIRadarLib.

The chirp parameters keep the instantaneous frequency far below fs/2:
generate_linear_chirp returns a *real* cos() signal and derives the
instantaneous frequency by unwrapping/differentiating the phase, which
aliases if the sweep approaches Nyquist.
"""

import numpy as np
import pytest

waveform_utils = pytest.importorskip("AIRadarLib.waveform_utils")
datautil = pytest.importorskip("AIRadarLib.datautil")


def test_linear_chirp_instantaneous_frequency():
    fs = 100e6
    duration = 100e-6
    start_freq = 1e6
    slope = 1e11
    bandwidth = slope * duration  # 10 MHz total sweep

    t = np.arange(int(fs * duration)) / fs
    signal, inst_freq, phase = waveform_utils.generate_linear_chirp(t, start_freq, slope)

    assert len(signal) == len(t)
    assert len(inst_freq) == len(t)
    assert len(phase) == len(t)

    # Mid-sweep instantaneous frequency matches f0 + slope * t within 2%
    # of the total bandwidth.
    mid = len(t) // 2
    expected_mid = start_freq + slope * t[mid]
    assert abs(inst_freq[mid] - expected_mid) < 0.02 * bandwidth

    # Total observed sweep (away from the gradient edge artifacts) covers
    # at least 90% of the design bandwidth.
    sweep = inst_freq[-100] - inst_freq[100]
    assert sweep >= 0.9 * bandwidth


def test_sine_wave_spectrum_peak():
    fs = 1e6
    freq = 123e3
    duration = 1e-3
    n_fft = 1024

    signal = waveform_utils.generate_sine_wave(freq, duration, fs)
    assert len(signal) == 1000

    spectrum = datautil.calculate_spectrum(signal, n_fft)
    assert len(spectrum) == n_fft

    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1 / fs))
    peak_freq, _ = datautil.find_peak_frequency(spectrum, freq_axis)

    # Peak within one FFT bin (fs / n_fft ~ 976 Hz) of the true tone.
    assert abs(peak_freq - freq) <= fs / n_fft
