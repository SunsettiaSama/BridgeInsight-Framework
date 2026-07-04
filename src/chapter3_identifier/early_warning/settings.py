from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from src.chapter3_identifier.early_warning._bootstrap import ensure_paths, resolve_path
from src.chapter3_identifier.regression_forecast.settings import (
    get_round_forecast_path,
    read_json,
)

ensure_paths()

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | None = None) -> dict[str, Any]:
    path = resolve_path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        path = _DEFAULT_CONFIG
    cfg = load_yaml(path)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_python_executable(cfg: dict[str, Any]) -> str:
    configured = cfg.get("python_executable")
    if configured:
        path = resolve_path(str(configured))
        if path.exists():
            return str(path)
        return sys.executable
    return sys.executable


def warning_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    raw = cfg.get("warning_thresholds") or {}
    return {
        "attention": float(raw.get("attention", 0.30)),
        "warning": float(raw.get("warning", 0.50)),
        "severe": float(raw.get("severe", 0.75)),
    }
