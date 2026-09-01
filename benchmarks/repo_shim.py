"""Import the AIRadar repo's numpy detectors without installing torch.

Why this file exists
--------------------
``AIRadar/AIRadarLib/radar_det.py`` contains three CFAR detectors that are pure
numpy/scipy (``cfar_2d_detect``, ``cfar_2d_numpy``, ``cfar_2d_advanced``) plus one
that is torch-based (``cfar_2d_pytorch``).  Because the module does
``import torch`` / ``import torch.nn.functional as F`` / ``import matplotlib.pyplot``
at the top level, *none* of the numpy detectors can be imported unless torch and
matplotlib are installed.  The same is true for ``AIRadar/AIradar_datasetv8.py``,
whose FMCW signal model is pure numpy but which imports torch, h5py and
matplotlib at module scope.

Rather than patch the repo (this harness is additive only) or install a
multi-gigabyte torch wheel to reach numpy code, we install minimal placeholder
modules for the *absent* heavy dependencies, import the repo modules, then
remove the placeholders from ``sys.modules`` again.  Anything already installed
for real (numpy, scipy, and matplotlib or tqdm if present) is never stubbed.

The functions this harness measures are then checked to make sure they do not
reference the stubbed names at all -- see :func:`assert_no_stub_dependency`.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from importlib import util as importlib_util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIRADAR_DIR = os.path.join(REPO_ROOT, "AIRadar")

def _passthrough_progress_bar(iterable=None, *args, **kwargs):
    """Stand-in for ``tqdm.tqdm``: return the iterable unchanged.

    Only reached if repo code the benchmark does not call decides to wrap an
    iterable; the benchmarked functions are checked not to reference ``tqdm``.
    """
    if iterable is None:
        return ()
    return iterable


# Modules the repo imports at module scope but that the numpy code paths never
# touch.  Each entry is (module name, {attribute: value}) -- the attributes are
# the ones third-party code introspects (e.g. scipy's array-API dispatch looks
# up ``torch.Tensor``) or that the repo dereferences at import time.
_STUB_SPECS: tuple[tuple[str, dict], ...] = (
    ("torch", {"__version__": "0.0.0+aisensing-benchmark-stub"}),
    ("torch.nn", {}),
    ("torch.nn.functional", {}),
    ("torch.utils", {}),
    ("torch.utils.data", {}),
    ("h5py", {}),
    # ``AIradar_datasetv8.py`` line 5 does ``from tqdm import tqdm``, so a fresh
    # numpy/scipy/matplotlib environment without tqdm fails at import.  The
    # placeholder returns the iterable unchanged; the benchmarked code paths never
    # construct a progress bar (only ``generate_dataset`` does, and that is never
    # called), which ``assert_no_stub_dependency`` enforces.
    ("tqdm", {"tqdm": _passthrough_progress_bar}),
    ("matplotlib", {}),
    ("matplotlib.pyplot", {}),
    ("matplotlib.cm", {}),
    ("mpl_toolkits", {}),
    ("mpl_toolkits.mplot3d", {}),
)


class _StubTensor:
    """Placeholder for ``torch.Tensor``.

    scipy's array-API compatibility layer resolves ``torch.Tensor`` whenever a
    module named ``torch`` is importable, so the placeholder must expose it or
    every scipy call raises ``AttributeError``.
    """


class _StubDataset:
    """Placeholder for ``torch.utils.data.Dataset`` (used only as a base class)."""


def _real_module_available(name: str) -> bool:
    root = name.split(".")[0]
    if root in sys.modules and not getattr(sys.modules[root], "__aisensing_stub__", False):
        return True
    try:
        return importlib_util.find_spec(root) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class ShimReport:
    """What the shim had to fake in order to import the repo modules."""

    stubbed: list[str] = field(default_factory=list)
    real: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"stubbed_modules": sorted(self.stubbed), "real_modules": sorted(self.real)}


def _install_stubs() -> tuple[ShimReport, list[str]]:
    report = ShimReport()
    installed: list[str] = []
    for name, attrs in _STUB_SPECS:
        if _real_module_available(name):
            root = name.split(".")[0]
            if root not in report.real:
                report.real.append(root)
            continue
        module = types.ModuleType(name)
        module.__aisensing_stub__ = True
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        installed.append(name)
        if name not in report.stubbed:
            report.stubbed.append(name)
        # Attach the submodule to its parent so ``from a.b import c`` works.
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child, module)

    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "__aisensing_stub__", False):
        torch.Tensor = _StubTensor
    data = sys.modules.get("torch.utils.data")
    if data is not None and getattr(data, "__aisensing_stub__", False):
        data.Dataset = _StubDataset
    mplot3d = sys.modules.get("mpl_toolkits.mplot3d")
    if mplot3d is not None and getattr(mplot3d, "__aisensing_stub__", False):
        mplot3d.Axes3D = object
    return report, installed


def _remove_stubs(installed: list[str]) -> None:
    for name in reversed(installed):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__aisensing_stub__", False):
            del sys.modules[name]


_LOADED: dict = {}


def load_repo_modules() -> dict:
    """Import the repo modules we benchmark and return them plus a shim report.

    Returns a dict with keys ``radar_det``, ``dataset_module``, ``AIRadarDataset``
    and ``shim_report``.  Idempotent.
    """
    if _LOADED:
        return _LOADED

    if AIRADAR_DIR not in sys.path:
        sys.path.insert(0, AIRADAR_DIR)

    report, installed = _install_stubs()
    try:
        import AIradar_datasetv8 as dataset_module
        from AIRadarLib import radar_det
    finally:
        _remove_stubs(installed)

    _LOADED.update(
        radar_det=radar_det,
        dataset_module=dataset_module,
        AIRadarDataset=dataset_module.AIRadarDataset,
        RADAR_CONFIGS=dataset_module.RADAR_CONFIGS,
        shim_report=report,
    )
    return _LOADED


#: Names that must never appear in the bytecode of a benchmarked function,
#: because they are the modules the shim faked.
_FORBIDDEN_GLOBALS = frozenset(
    {"torch", "F", "h5py", "plt", "cm", "Axes3D", "cv2", "tqdm"}
)


def assert_no_stub_dependency(*functions) -> None:
    """Fail loudly if a benchmarked function references a stubbed module.

    Guards against a future repo change quietly making one of these code paths
    torch-dependent, which would turn the shim from "harmless" into "measuring
    something that never really ran".
    """
    for func in functions:
        code = getattr(func, "__code__", None)
        if code is None:  # pragma: no cover - defensive
            raise TypeError(f"{func!r} has no __code__ to inspect")
        used = set(code.co_names)
        for const in code.co_consts:
            if hasattr(const, "co_names"):  # nested code objects (closures, comprehensions)
                used |= set(const.co_names)
        offenders = sorted(used & _FORBIDDEN_GLOBALS)
        if offenders:
            raise AssertionError(
                f"{func.__qualname__} references stubbed module(s) {offenders}; "
                "it cannot be benchmarked without the real dependency installed"
            )
