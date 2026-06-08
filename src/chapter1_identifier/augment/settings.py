from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml

from src.chapter1_identifier.augment._bootstrap import ensure_paths, resolve_path

ensure_paths()

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"
_DUAL_STREAM_CONFIG = Path(__file__).resolve().parent / "config" / "dual_stream_res_cnn.yaml"


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


def load_dual_stream_model_config(config_path: Optional[str] = None) -> dict:
    path = resolve_path(config_path) if config_path else _DUAL_STREAM_CONFIG
    if not path.exists():
        path = _DUAL_STREAM_CONFIG
    return load_yaml(path)


def load_best_params(path: str) -> dict:
    p = resolve_path(path)
    if not p.exists():
        return {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "gradient_clip_norm": 0.5,
        }
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_params", data)


def namespace_from_dict(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**d)


def resolve_python_executable(cfg: dict) -> str:
    configured = cfg.get("python_executable")
    if configured:
        path = resolve_path(str(configured))
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"python_executable 不存在：{path}")
    import sys
    return sys.executable


def check_inference_metadata(cfg: dict) -> Path:
    ds_cfg_path = resolve_path(cfg["inference_dataset_config"])
    if not ds_cfg_path.exists():
        raise FileNotFoundError(
            f"推理数据集配置不存在：{ds_cfg_path}\n"
            "请创建 config/datasets/total_staycable_vib_202409.yaml"
        )
    ds_cfg = load_yaml(ds_cfg_path)
    meta_path = resolve_path(ds_cfg["vib_metadata_path"])
    if not meta_path.exists():
        raise FileNotFoundError(
            f"推理池振动元数据不存在：{meta_path}\n"
            "请先将 preprocess.yaml 的 all_vibration_root 指向 data/2024September/SuTong/VIC，"
            "运行预处理后更新 config/datasets/total_staycable_vib_202409.yaml 中的路径。"
        )
    return meta_path


@dataclass
class BranchConfig:
    in_channels: int = 1
    input_size: int = 3000
    res_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_blocks: int = 2
    kernel_size: int = 3
    fc_hidden_dims: List[int] = field(default_factory=lambda: [128])
    dropout_prob: float = 0.5
    num_classes: int = 4


@dataclass
class DualStreamResCNNConfig:
    time_branch: BranchConfig
    spec_branch: BranchConfig
    fusion_type: str = "weighted_sum"
    num_classes: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "DualStreamResCNNConfig":
        num_classes = int(d.get("num_classes", 4))
        dropout = float(d.get("dropout_prob", 0.5))
        fc_hidden = d.get("fc_hidden_dims", [128])
        time_d = d.get("time_branch", {})
        spec_d = d.get("spec_branch", {})
        fusion = d.get("fusion", {})
        return cls(
            time_branch=BranchConfig(
                in_channels=int(time_d.get("in_channels", 1)),
                input_size=int(time_d.get("input_size", 3000)),
                res_channels=list(time_d.get("res_channels", [64, 128, 256])),
                num_blocks=int(time_d.get("num_blocks", 2)),
                kernel_size=int(time_d.get("kernel_size", 3)),
                fc_hidden_dims=list(fc_hidden),
                dropout_prob=dropout,
                num_classes=num_classes,
            ),
            spec_branch=BranchConfig(
                in_channels=int(spec_d.get("in_channels", 1)),
                input_size=int(spec_d.get("input_size", 129)),
                res_channels=list(spec_d.get("res_channels", [32, 64, 128])),
                num_blocks=int(spec_d.get("num_blocks", 2)),
                kernel_size=int(spec_d.get("kernel_size", 3)),
                fc_hidden_dims=list(fc_hidden),
                dropout_prob=dropout,
                num_classes=num_classes,
            ),
            fusion_type=str(fusion.get("type", "weighted_sum")),
            num_classes=num_classes,
        )
