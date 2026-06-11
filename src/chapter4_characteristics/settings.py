from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


def get_inference_dir(cfg: dict, round_idx: int) -> Path:
    return get_chapter4_root(cfg) / "inference" / f"round_{round_idx:02d}"


def get_inference_path(cfg: dict, round_idx: int) -> Path:
    return get_inference_dir(cfg, round_idx) / "inference.json"


def get_predictions_enriched_path(cfg: dict, round_idx: int) -> Path:
    return get_inference_dir(cfg, round_idx) / "predictions_enriched.json"


def get_enriched_round_dir(cfg: dict, round_idx: int) -> Path:
    return resolve_path(cfg["enriched_stats_dir"]) / f"round_{round_idx:02d}"


def get_augment_checkpoint(cfg: dict, round_idx: int) -> Path:
    return resolve_path(cfg["augment_rounds_dir"]) / f"round_{round_idx:02d}" / "best_checkpoint.pth"


def get_active_round_path(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "active_round.json"


def get_reference_stats_path(cfg: dict, round_idx: int) -> Path:
    return get_chapter4_root(cfg) / f"reference_stats_round_{round_idx:02d}.json"


def get_reference_psd_path(cfg: dict, round_idx: int) -> Path:
    return get_chapter4_root(cfg) / f"reference_psd_round_{round_idx:02d}.json"


def get_others_index_path(cfg: dict, round_idx: int) -> Path:
    return get_chapter4_root(cfg) / f"others_index_round_{round_idx:02d}.json"


def get_copula_dir(cfg: dict, round_idx: int) -> Path:
    return get_chapter4_root(cfg) / "copula" / f"round_{round_idx:02d}"


def get_exports_dir(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "exports"


def read_active_round(cfg: dict) -> Optional[int]:
    path = get_active_round_path(cfg)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("round_idx", 0)) or None


def write_active_round(cfg: dict, round_idx: int) -> None:
    path = get_active_round_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round_idx": round_idx,
        "inference": str(get_inference_path(cfg, round_idx)),
        "enriched": str(get_enriched_round_dir(cfg, round_idx)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def effective_round(cfg: dict, round_idx: Optional[int] = None) -> int:
    if round_idx is not None:
        return int(round_idx)
    active = read_active_round(cfg)
    if active is not None:
        return active
    return 1
