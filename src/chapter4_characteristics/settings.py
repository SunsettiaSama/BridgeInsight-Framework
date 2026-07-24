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
DEFAULT_LABEL_NAMES = ["随机振动", "VIV", "RWIV", "其他"]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_runtime_defaults(cfg: dict) -> dict:
    root = resolve_path(cfg.get("chapter4_output_dir", "results/chapter4_characteristics"))
    cfg["chapter4_output_dir"] = str(root)
    cfg.setdefault("num_classes", len(DEFAULT_LABEL_NAMES))
    cfg.setdefault("label_names", DEFAULT_LABEL_NAMES)
    cfg.setdefault("job_state_path", str(root / "job_state.json"))
    cfg.setdefault("enriched_stats_dir", str(root / "enriched"))
    cfg.setdefault("feature_analysis_config", None)
    cfg.setdefault("webui_host", "127.0.0.1")
    cfg.setdefault("webui_port", 8766)
    cfg.setdefault("reference_feature_keys", [])
    cfg.setdefault("others_queue_page_size", 30)
    cfg.setdefault("others_neighbor_topk", 3)
    cfg.setdefault("context_windows_before", 5)
    cfg.setdefault("context_windows_after", 5)
    cfg.setdefault("copula_n_modes", 24)
    cfg.setdefault("copula_nfft", 128)
    cfg.setdefault("copula_max_samples", 20000)
    cfg.setdefault("copula_joint_max_samples", 5000)
    cfg.setdefault("copula_rng_seed", 42)
    # 窗间时序 Copula（并行流水线；不覆盖静态 copula 结果）
    td_defaults = {
        "energy_top_k": 4,
        "max_samples": 8000,
        "joint_max_pairs": 5000,
        "n_paths": 100,
        "n_steps": 60,
        "rng_seed": 42,
        "nfft": 128,
    }
    td_cfg = dict(td_defaults)
    td_cfg.update(cfg.get("td_copula") or {})
    cfg["td_copula"] = td_cfg
    cfg.setdefault("window_size", 3000)
    cfg.setdefault("compact_enriched_batches", True)
    cfg.setdefault("auto_compact_on_read", True)
    cfg.setdefault("compact_enriched_force", False)

    dataset_config = cfg.get("inference_dataset_config")
    if dataset_config and not cfg.get("wind_metadata_path"):
        dataset_path = resolve_path(str(dataset_config))
        if dataset_path.exists():
            dataset_cfg = load_yaml(dataset_path)
            if dataset_cfg.get("wind_metadata_path"):
                cfg["wind_metadata_path"] = str(resolve_path(str(dataset_cfg["wind_metadata_path"])))
            if dataset_cfg.get("window_size") is not None:
                cfg.setdefault("window_size", dataset_cfg["window_size"])
    return cfg


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path:
        path = resolve_path(config_path)
    elif _DEFAULT_CONFIG.exists():
        path = _DEFAULT_CONFIG
    else:
        from src.chapter3_identifier.augment.settings import load_config as load_augment_config
        from src.chapter3_identifier.augment.workflow_config import load_chapter4_runtime_config

        return _apply_runtime_defaults(load_chapter4_runtime_config(load_augment_config(None)))
    if not path.exists():
        raise FileNotFoundError(f"chapter4 配置不存在：{path}")
    cfg = load_yaml(path)
    cfg["_config_path"] = str(path)
    return _apply_runtime_defaults(cfg)


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


def get_td_copula_dir(cfg: dict) -> Path:
    return get_copula_dir(cfg) / "td"


def get_exports_dir(cfg: dict) -> Path:
    return get_chapter4_root(cfg) / "exports"
