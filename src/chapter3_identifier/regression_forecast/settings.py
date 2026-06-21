from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from src.chapter3_identifier.regression_forecast._bootstrap import ensure_paths, resolve_path

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


def get_rounds_root(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["rounds_output_dir"])


def get_round_dir(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_rounds_root(cfg) / f"round_{int(round_idx):02d}"


def get_round_checkpoint_path(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_round_dir(cfg, round_idx) / "best_checkpoint.pth"


def get_round_metrics_path(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_round_dir(cfg, round_idx) / "metrics.json"


def get_round_live_metrics_path(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_round_dir(cfg, round_idx) / "metrics_live.json"


def get_round_forecast_path(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_round_dir(cfg, round_idx) / "forecast.json"


def get_round_schema_path(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_round_dir(cfg, round_idx) / "schema.json"


def get_round_feature_cache_path(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_round_dir(cfg, round_idx) / "feature_cache.jsonl"


def get_round_split_path(cfg: dict[str, Any], round_idx: int) -> Path:
    return get_round_dir(cfg, round_idx) / "split.json"


def ensure_round_dir(cfg: dict[str, Any], round_idx: int) -> Path:
    path = get_round_dir(cfg, round_idx)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

