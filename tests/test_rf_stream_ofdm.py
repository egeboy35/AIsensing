"""802.11a LTF timing-sync properties and OFDM FFT-shift convention.

rf_stream/test_ltf.py is loaded by file path via importlib (under a
non-test module name) so pytest never collects it and no rf_stream entry
is added to sys.path - other rf_stream scripts pull torch/hardware at
import time.

The FFT convention test is self-contained numpy mirroring
rf_stream/test_ofdm_convention.py (a top-level print script), pinning the
one self-consistent TX/RX pairing.
"""

import importlib.util
import pathlib

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_ltf_module():
    path = _REPO / "rf_stream" / "test_ltf.py"
    spec = importlib.util.spec_from_file_location("ltf_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ltf_autocorrelation_peak_dominates_sidelobes():
    ltf = _load_ltf_module()

    ref = np.zeros(64, dtype=complex)
    for k in range(-26, 27):
        ref[(k + 64) % 64] = ltf.get_ltf_val(k)

    # 52 unit-magnitude subcarriers -> zero-lag autocorrelation of 52.
    peak = abs(np.sum(ref * np.conj(ref)))
    assert peak == 52.0

    # Every circular-shift sidelobe stays at or below half the peak
    # (measured worst case: 8), so the LTF is usable for timing sync.
    sidelobes = [
        abs(np.sum(ref * np.conj(np.roll(ref, dk))))
        for dk in range(-15, 16)
        if dk != 0
    ]
    assert max(sidelobes) <= 26.0


def test_ofdm_tx_rx_convention_consistency():
    n_fft, cp_len = 64, 16
    rng = np.random.default_rng(10)

    used_subcarriers = [k for k in range(-26, 27) if k != 0]
    bins = [(k + n_fft) % n_fft for k in used_subcarriers]

    X = np.zeros(n_fft, dtype=complex)
    X[bins] = (
        rng.choice([-1.0, 1.0], size=len(bins))
        + 1j * rng.choice([-1.0, 1.0], size=len(bins))
    ) / np.sqrt(2)

    # TX: x = ifft(ifftshift(X)) * sqrt(N), then cyclic prefix.
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(n_fft)
    x_cp = np.concatenate([x[-cp_len:], x])

    # RX (consistent pairing): strip CP, Y = fftshift(fft(x)) / sqrt(N),
    # read the SAME (k+N)%N bins.  Exact recovery.
    x_no_cp = x_cp[cp_len:]
    y_consistent = np.fft.fftshift(np.fft.fft(x_no_cp)) / np.sqrt(n_fft)
    assert np.max(np.abs(y_consistent[bins] - X[bins])) < 1e-9

    # RX without the fftshift, same bin map: badly wrong (bins move by N/2).
    y_plain = np.fft.fft(x_no_cp) / np.sqrt(n_fft)
    assert np.max(np.abs(y_plain[bins] - X[bins])) > 0.5
