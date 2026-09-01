"""Can the Phaser data files that ship in this repository actually be opened?

`sdradi/phaser/` carries real calibration data and an SDR filter:

    channel_cal_val.pkl   gain_cal_val.pkl   phase_cal_val.pkl   LTE20_MHz.ftr

and every caller of the loaders (`myradar.py:39-40`, `myradar2.py:46-47`,
`myphaser.py:58-60`) invokes them with no argument, so the default path is the
one that matters. On a miss the loaders do not raise: they print a line and
substitute "not calibrated" values, which reads exactly like a genuine first
run.

No board and no `adi` package: `adi` is stubbed with empty base classes so
`mycn0566` can be imported, and nothing under test touches it. A test reports
whether the stub was used, so a stubbed run cannot be mistaken for a real one.

    pytest tests/test_phaser_data_paths.py
"""
import ast
import io
import os
import pickle
import sys
import types
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PHASER = REPO / "sdradi" / "phaser"

# ------------------------------------------------------------------ adi stub
_ADI_STUBBED = False
try:                                              # pragma: no cover - env probe
    import adi  # noqa: F401
except ImportError:
    _ADI_STUBBED = True
    adi = types.ModuleType("adi")
    adar1000 = types.ModuleType("adi.adar1000")
    adf4159 = types.ModuleType("adi.adf4159")

    class _Base:                                  # a usable base class, nothing more
        def __init__(self, *a, **k):
            pass

    adar1000.adar1000_array = type("adar1000_array", (_Base,), {})
    adf4159.adf4159 = type("adf4159", (_Base,), {})
    adi.adar1000, adi.adf4159 = adar1000, adf4159
    sys.modules.update({"adi": adi, "adi.adar1000": adar1000, "adi.adf4159": adf4159})

sys.path.insert(0, str(PHASER))
import mycn0566  # noqa: E402


DATA_FILES = ["channel_cal_val.pkl", "gain_cal_val.pkl",
              "phase_cal_val.pkl", "LTE20_MHz.ftr"]
CAL_PAIRS = [("channel", "channel_cal_val.pkl"),
             ("gain", "gain_cal_val.pkl"),
             ("phase", "phase_cal_val.pkl")]


def test_adi_stub_is_declared():
    """Records in the report whether `adi` was the real package."""
    assert _ADI_STUBBED in (True, False)
    if _ADI_STUBBED:
        print("\n[note] adi was absent and stubbed; nothing under test touches it")


# --------------------------------------------------------- the files are here
@pytest.mark.parametrize("name", DATA_FILES)
def test_the_data_file_ships_with_the_repository(name):
    assert (PHASER / name).is_file()


# ------------------------------------------------------ defaults resolve
# These go through the public loaders rather than the helper this change adds,
# so that running them against the previous code fails on what the loaders do
# rather than on a symbol that is missing. test_the_helper_is_available below
# is the only one that asserts on the new name.
@pytest.mark.parametrize("kind,name", CAL_PAIRS)
def test_the_default_works_from_any_working_directory(kind, name, tmp_path,
                                                      monkeypatch, capsys):
    """The scripts that call these live in sdradi/, not at the repository root."""
    for cwd in (tmp_path, REPO, REPO / "sdradi", PHASER):
        monkeypatch.chdir(cwd)
        obj = mycn0566.CN0566.__new__(mycn0566.CN0566)
        getattr(mycn0566.CN0566, f"load_{kind}_cal")(obj)
        out = capsys.readouterr().out
        assert "file not found" not in out, f"fell back when cwd was {cwd}"


@pytest.mark.parametrize("kind,name", CAL_PAIRS)
def test_an_explicit_filename_still_wins(kind, name, tmp_path, monkeypatch):
    """Callers that pass a path must keep getting exactly that path."""
    payload = [9.0, 8.0]
    mine = tmp_path / "somewhere_else.pkl"
    mine.write_bytes(pickle.dumps(payload))
    monkeypatch.chdir(REPO)
    obj = mycn0566.CN0566.__new__(mycn0566.CN0566)
    getattr(mycn0566.CN0566, f"load_{kind}_cal")(obj, str(mine))
    attr = {"channel": "ccal", "gain": "gcal", "phase": "pcal"}[kind]
    assert getattr(obj, attr) == payload


def test_the_helper_is_available():
    """The one test that names what this change adds."""
    assert mycn0566._default_cal_path(None, "x.pkl") == str(PHASER / "x.pkl")
    assert mycn0566._default_cal_path("/tmp/mine.pkl", "x.pkl") == "/tmp/mine.pkl"


# ------------------------------------------------ save and load round-trip
@pytest.mark.parametrize("kind,name", CAL_PAIRS)
def test_save_then_load_round_trips_on_the_defaults(kind, name, tmp_path, monkeypatch):
    """The property the two defaults have to share, whatever they are."""
    obj = mycn0566.CN0566.__new__(mycn0566.CN0566)
    attr = {"channel": "ccal", "gain": "gcal", "phase": "pcal"}[kind]
    original = (PHASER / name).read_bytes()
    payload = [0.125, 0.25, 0.5]
    setattr(obj, attr, payload)

    monkeypatch.chdir(tmp_path)                   # a cwd with none of these files
    try:
        getattr(mycn0566.CN0566, f"save_{kind}_cal")(obj)
        setattr(obj, attr, None)
        getattr(mycn0566.CN0566, f"load_{kind}_cal")(obj)
        assert getattr(obj, attr) == payload
    finally:
        (PHASER / name).write_bytes(original)     # leave the shipped data intact


@pytest.mark.parametrize("kind,name", CAL_PAIRS)
def test_loading_the_shipped_file_does_not_fall_back(kind, name, tmp_path,
                                                     monkeypatch, capsys):
    """The fallback prints; a successful load must not."""
    obj = mycn0566.CN0566.__new__(mycn0566.CN0566)
    attr = {"channel": "ccal", "gain": "gcal", "phase": "pcal"}[kind]
    monkeypatch.chdir(tmp_path)
    getattr(mycn0566.CN0566, f"load_{kind}_cal")(obj)
    assert "file not found" not in capsys.readouterr().out
    assert getattr(obj, attr) == pickle.loads((PHASER / name).read_bytes())


def test_a_missing_file_still_falls_back_rather_than_raising(tmp_path, capsys):
    """The existing behaviour for a genuinely uncalibrated board is unchanged."""
    obj = mycn0566.CN0566.__new__(mycn0566.CN0566)
    mycn0566.CN0566.load_phase_cal(obj, str(tmp_path / "nothing_here.pkl"))
    assert "file not found" in capsys.readouterr().out
    assert obj.pcal == [0.0] * 8


# ------------------------------------------------------- regression guards
def _py_files():
    return sorted(p for p in PHASER.glob("*.py"))


def test_no_module_carries_a_backslash_separated_relative_path():
    """`"sdradi\\phaser\\x"` is a path only on Windows, and only from the root."""
    offenders = []
    for p in _py_files():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(io.open(p, encoding="utf-8", errors="replace").read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith("sdradi" + chr(92))):
                offenders.append(f"{p.name}:{node.lineno} {node.value!r}")
    assert offenders == []


def test_no_module_emits_a_syntax_warning():
    """Invalid escapes are a SyntaxWarning today and a SyntaxError later."""
    offenders = []
    for p in _py_files():
        src = io.open(p, encoding="utf-8", errors="replace").read()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                compile(src, str(p), "exec")
            except SyntaxError:
                continue
        offenders += [f"{p.name}:{w.lineno} {w.message}" for w in caught
                      if issubclass(w.category, SyntaxWarning)]
    assert offenders == []
