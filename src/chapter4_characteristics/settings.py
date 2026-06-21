from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml

from src.chapter4_characteristics._bootstrap import ensure_paths, resolve_path

ensure_paths()

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"

CLASS_LABELS = {0: "normal", 1: "viv", 2: "rwiv", 3: "transition"}
CLASS_DIRS = {0: "class_0_normal", 1: "class_1_viv", 2: "class_2_rwiv", 3: "class_3_transition"}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[str] = None) -> dict:
    path = resolve_path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        path = _DEFAULT_CONFIG
    cfg = load_yaml(path)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_python_executable(cfg: dict) -> str:
    configured = cfg.get("python_executable")
    if configured:
        path = resolve_path(str(configured))
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"python_executable 不存在：{path}")
    return sys.executable


def get_chapter4_root(cfg: dict) -> Path:
    return resolve_path(cfg["chapter4_output_dir"])


def get_inference_dir(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "inference"


def get_inference_path(cfg: dict) -> Path:
    return get_inference_dir(cfg) / "inference.json"


def get_predictions_enriched_path(cfg: dict) -> Path:
    return get_inference_dir(cfg) / "predictions_enriched.json"


def get_enriched_dir(cfg: dict) -> Path:
    return resolve_path(cfg["enriched_stats_dir"])


def get_identifier_checkpoint(cfg: dict) -> Path:
    return resolve_path(cfg["identifier_checkpoint_path"])


def get_reference_stats_path(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "reference_stats.json"


def get_reference_psd_path(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "reference_psd.json"


def get_others_index_path(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "others_index.json"


def get_copula_dir(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "copula"


def get_exports_dir(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "exports"
