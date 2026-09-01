"""Shared pytest configuration for the AIsensing test suite.

Order of the statements below matters:

1. MPLBACKEND must be set before any repo module gets imported, because
   several modules (AIRadarLib.datautil, rf_image_transfer, ...) import
   matplotlib.pyplot at module top.  Agg prevents any GUI backend from
   being selected on CI or on developer machines.
2. stdout/stderr are reconfigured to UTF-8, because some repo functions
   print Unicode glyphs (e.g. AIRadarLib.datautil prints delta/lambda in
   its summary), which raises UnicodeEncodeError on Windows cp125x
   consoles when pytest runs with -s.
3. sys.path entries make the flat-layout modules importable without any
   packaging changes to the repository.

Heavy imports (sdr_video_comm, AIRadarLib.radar_det, torch) happen inside
the individual test modules behind pytest.importorskip gates, never here,
so test collection stays dependency-free.
"""

import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_REPO = pathlib.Path(__file__).resolve().parents[1]

for _path in (
    _REPO / "sdradi",                  # sdr_video_comm, sdr_ldpc
    _REPO / "sdradi" / "pluto_test",   # rf_image_transfer
    _REPO / "AIRadar",                 # AIRadarLib package
):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: LDPC CPU decode, can take tens of seconds"
    )
