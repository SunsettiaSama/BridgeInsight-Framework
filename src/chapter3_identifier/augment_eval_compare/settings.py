from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from src.chapter3_identifier.augment_eval_compare._bootstrap import ensure_paths, resolve_path
from src.chapter3_identifier.augment.settings import (
    get_round_checkpoint_path,
    get_round_dir,
    get_round_merged_training_pair_key_path,
    load_config as load_augment_config,
)

ensure_paths()

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "default.yaml"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_compare_config(config_path: Optional[str] = None) -> dict[str, Any]:
    path = resolve_path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        path = _DEFAULT_CONFIG
    cfg = load_yaml(path)
    augment_cfg = load_augment_config(cfg.get("augment_config"))
    cfg["_compare_config_path"] = str(path)
    cfg["_augment_cfg"] = augment_cfg
    return cfg


def get_compare_output_dir(cfg: dict, round_idx: int) -> Path:
    root = resolve_path(cfg.get("output_dir", "results/augment_eval_compare/rounds"))
    return root / f"round_{int(round_idx):02d}"


def _round_idx_from_config(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _infer_round_idx_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("round_"):
            suffix = part.removeprefix("round_")
            if suffix.isdigit():
                return int(suffix)
    return None


def _validate_checkpoint_round(role: str, path: Path, checkpoint_round: int | None, eval_round: int, cfg: dict) -> None:
    effective_round = checkpoint_round if checkpoint_round is not None else _infer_round_idx_from_path(path)
    if effective_round is None:
        return
    if effective_round == int(eval_round):
        return
    if bool(cfg.get("allow_cross_round_baseline", False)):
        return
    raise ValueError(
        f"{role} checkpoint 来自 round_{effective_round:02d}，但当前评估 round 为 round_{int(eval_round):02d}。"
        "模型性能对比默认要求同一 round / 同一评估语义；如需跨 round 基线，"
        "请显式设置 allow_cross_round_baseline: true。"
    )


def resolve_decoupled_checkpoint(cfg: dict, round_idx: int) -> Path:
    explicit = cfg.get("decoupled_checkpoint")
    if explicit:
        path = resolve_path(str(explicit))
        if not path.exists():
            raise FileNotFoundError(f"decoupled checkpoint 不存在：{path}")
        _validate_checkpoint_round(
            "decoupled",
            path,
            _round_idx_from_config(cfg.get("decoupled_round_idx")),
            round_idx,
            cfg,
        )
        return path
    decoupled_round = _round_idx_from_config(cfg.get("decoupled_round_idx"))
    if decoupled_round is None:
        raise ValueError(
            "未指定 decoupled_checkpoint 或 decoupled_round_idx。"
            "请提供与当前 round 语义一致的解耦模型 checkpoint；"
            "不要默认使用 round_01 作为 round_09 对比基线。"
        )
    path = get_round_checkpoint_path(cfg["_augment_cfg"], decoupled_round)
    if not path.exists():
        raise FileNotFoundError(
            f"decoupled checkpoint 不存在：{path}（round_{decoupled_round:02d}）"
        )
    _validate_checkpoint_round("decoupled", path, decoupled_round, round_idx, cfg)
    return path


def resolve_joint_checkpoint(cfg: dict, round_idx: int) -> Path:
    explicit = cfg.get("joint_checkpoint")
    if explicit:
        path = resolve_path(str(explicit))
        if not path.exists():
            raise FileNotFoundError(f"joint checkpoint 不存在：{path}")
        _validate_checkpoint_round(
            "joint",
            path,
            _round_idx_from_config(cfg.get("joint_round_idx")),
            round_idx,
            cfg,
        )
        return path
    joint_round = cfg.get("joint_round_idx")
    joint_round_idx = int(joint_round) if joint_round is not None else int(round_idx)
    path = get_round_checkpoint_path(cfg["_augment_cfg"], joint_round_idx)
    if not path.exists():
        raise FileNotFoundError(
            f"joint checkpoint 不存在：{path}（round_{joint_round_idx:02d}）"
        )
    _validate_checkpoint_round("joint", path, joint_round_idx, round_idx, cfg)
    return path


def get_pair_key_dataset_path(cfg: dict, round_idx: int) -> Path:
    return get_round_merged_training_pair_key_path(cfg["_augment_cfg"], round_idx)


def load_train_profile(round_idx: int, augment_cfg: dict) -> dict:
    path = get_round_dir(augment_cfg, round_idx) / "train_profile.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}
