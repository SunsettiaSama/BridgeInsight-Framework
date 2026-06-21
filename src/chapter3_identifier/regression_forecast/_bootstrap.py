"""Import path shim for the regression forecast module."""
from __future__ import annotations

import sys
import types
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_CHAPTER3_DIR = _MODULE_DIR.parent
_PROJECT_ROOT = _CHAPTER3_DIR.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"


def project_root() -> Path:
    return _PROJECT_ROOT


def ensure_paths() -> Path:
    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    src_pkg = sys.modules.get("src")
    if src_pkg is None:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(_SRC_DIR)]
        sys.modules["src"] = src_pkg

    mounts = {
        "src.training": _CHAPTER3_DIR / "training",
        "src.identifier": _CHAPTER3_DIR / "identifier",
    }
    for full_name, physical in mounts.items():
        if full_name not in sys.modules:
            mod = types.ModuleType(full_name)
            mod.__path__ = [str(physical)]
            sys.modules[full_name] = mod
    return _PROJECT_ROOT


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()

