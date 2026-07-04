from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.chapter3_identifier.augment.datasets.dual_stream_dataset import (
    DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
    DEFAULT_CONTEXT_INPUT_SIZE,
    DEFAULT_CONTEXT_TOTAL_SECONDS,
)
from src.chapter3_identifier.augment.features.wind_features import WIND_FEATURE_DIM
from src.chapter3_identifier.augment.settings import (
    load_best_params,
    resolve_path,
)

SCHEMA_VERSION = 2
MODEL_TYPES = (
    "dual_stream_single_head",
    "quad_stream_dual_head",
    "quad_stream_dual_head_context",
    "quad_stream_serial_context_dual_head",
)
CONTEXT_MODES = ("short_only", "short_long")
PREDICTION_FILL_MODES = ("auto", "always", "off")


def _default_model_type(round_idx: int) -> str:
    if int(round_idx) <= 1:
        return "dual_stream_single_head"
    if int(round_idx) == 2:
        return "quad_stream_dual_head"
    return "quad_stream_serial_context_dual_head"


def _uses_long_context_model(model_type: str) -> bool:
    return model_type in {"quad_stream_dual_head_context", "quad_stream_serial_context_dual_head"}


def _best_params(cfg: dict) -> dict:
    path = cfg.get("best_params")
    if not path:
        return {}
    return load_best_params(str(path))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"无法解析布尔值：{value!r}")


def _positive_int(value: Any, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return out


def _nonnegative_float(value: Any, name: str) -> float:
    out = float(value)
    if out < 0:
        raise ValueError(f"{name} 必须为非负数")
    return out


def _positive_float(value: Any, name: str) -> float:
    out = float(value)
    if out <= 0:
        raise ValueError(f"{name} 必须为正数")
    return out


def _ratio_float(value: Any, name: str) -> float:
    out = float(value)
    if not 0.0 < out < 1.0:
        raise ValueError(f"{name} 必须在 0 到 1 之间")
    return out


def _dropout_float(value: Any, name: str) -> float:
    out = float(value)
    if not 0.0 <= out < 1.0:
        raise ValueError(f"{name} 必须在 [0, 1) 之间")
    return out


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是字符串列表")
    return [str(item).strip() for item in value if str(item).strip()]


def default_training_profile(cfg: dict, round_idx: int) -> dict:
    best = _best_params(cfg)
    model_type = _default_model_type(round_idx)
    legacy_lambda = float(cfg.get("round3_legacy_reg_lambda", 0.2 if int(round_idx) >= 3 else 0.0))
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "context_mode": "short_long" if _uses_long_context_model(model_type) else "short_only",
        "enable_long_context": _uses_long_context_model(model_type),
        "context_input_size": int(cfg.get("context_input_size", DEFAULT_CONTEXT_INPUT_SIZE)),
        "context_total_seconds": float(cfg.get("context_total_seconds", DEFAULT_CONTEXT_TOTAL_SECONDS)),
        "context_allow_cross_file": bool(cfg.get("context_allow_cross_file", DEFAULT_CONTEXT_ALLOW_CROSS_FILE)),
        "enable_wind_features": bool(cfg.get("enable_wind_features", False)),
        "wind_feature_dim": int(cfg.get("wind_feature_dim", 0)),
        "train_val_ratio": float(cfg.get("train_val_ratio", 0.8)),
        "enable_sensor_exclusion": bool(cfg.get("enable_sensor_exclusion", False)),
        "exclude_sensor_ids": list(cfg.get("exclude_sensor_ids", [])),
        "require_bidirectional_labels": False,
        "enable_gold_fill": False,
        "prediction_fill_mode": str(cfg.get("prediction_fill_mode", "auto")),
        "min_pairs_without_prediction": int(cfg.get("min_pairs_without_prediction", 1000)),
        "enable_legacy_regularization": False,
        "legacy_reg_lambda": legacy_lambda,
        "legacy_reg_temperature": float(cfg.get("round3_legacy_reg_temperature", 2.0)),
        "legacy_teacher_checkpoint": str(cfg.get("legacy_teacher_checkpoint") or ""),
        "enable_same_structure_regularization": int(round_idx) >= 2,
        "same_structure_reg_lambda": float(cfg.get("same_structure_reg_lambda", 0.1)),
        "same_structure_reg_temperature": float(cfg.get("same_structure_reg_temperature", 2.0)),
        "same_structure_lr_scale": float(cfg.get("same_structure_lr_scale", 0.3)),
        "inplane_loss_weight": float(cfg.get("round2_inplane_loss_weight", 1.0)),
        "outplane_loss_weight": float(cfg.get("round2_outplane_loss_weight", 1.0)),
        "branch_dropout_prob": float(cfg.get("branch_dropout_prob", cfg.get("dropout_prob", 0.2))),
        "fusion_hidden_dim": int(cfg.get("round2_fusion_hidden_dim", 128)),
        "fusion_dropout": float(cfg.get("round2_fusion_dropout", 0.25)),
        "cross_attn_heads": int(cfg.get("round2_cross_attn_heads", 4)),
        "epochs": int(cfg.get("epochs", 25)),
        "batch_size": int(best.get("batch_size", cfg.get("batch_size", 16))),
        "learning_rate": float(best.get("learning_rate", cfg.get("learning_rate", 1e-4))),
        "weight_decay": float(cfg.get("weight_decay", best.get("weight_decay", 1e-5))),
        "gradient_clip_norm": float(best.get("gradient_clip_norm", cfg.get("gradient_clip_norm", 0.5))),
        "focal_gamma": float(cfg.get("focal_gamma", 2.0)),
        "early_stopping_patience": int(cfg.get("early_stopping_patience", 5)),
        "early_stopping_min_delta": float(cfg.get("early_stopping_min_delta", 0.0)),
    }


def normalize_training_profile(payload: dict, cfg: dict, round_idx: int) -> dict:
    base = default_training_profile(cfg, round_idx)
    merged = {**base}
    payload_schema = int(payload.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)
    dataset_strategy_keys = {
        "train_val_ratio",
        "require_bidirectional_labels",
        "enable_gold_fill",
        "prediction_fill_mode",
        "min_pairs_without_prediction",
    }
    for key in base:
        if payload_schema < 2 and key in dataset_strategy_keys:
            continue
        if key in payload:
            merged[key] = payload[key]

    model_type = str(merged["model_type"])
    if model_type not in MODEL_TYPES:
        raise ValueError(f"未知 model_type：{model_type}")
    context_mode = str(merged.get("context_mode", "short_long" if _uses_long_context_model(model_type) else "short_only"))
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"未知 context_mode：{context_mode}")
    prediction_fill_mode = str(merged.get("prediction_fill_mode", "auto"))
    if prediction_fill_mode not in PREDICTION_FILL_MODES:
        raise ValueError(f"未知 prediction_fill_mode：{prediction_fill_mode}")
    if int(round_idx) <= 1 and model_type != "dual_stream_single_head":
        raise ValueError("round1 只能使用 dual_stream_single_head")

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model_type,
        "context_mode": context_mode,
        "enable_long_context": _as_bool(merged["enable_long_context"]),
        "context_input_size": _positive_int(merged["context_input_size"], "context_input_size"),
        "context_total_seconds": _positive_float(merged["context_total_seconds"], "context_total_seconds"),
        "context_allow_cross_file": _as_bool(merged["context_allow_cross_file"]),
        "enable_wind_features": _as_bool(merged["enable_wind_features"]),
        "wind_feature_dim": int(merged["wind_feature_dim"]),
        "train_val_ratio": _ratio_float(merged["train_val_ratio"], "train_val_ratio"),
        "enable_sensor_exclusion": _as_bool(merged["enable_sensor_exclusion"]),
        "exclude_sensor_ids": _string_list(merged.get("exclude_sensor_ids"), "exclude_sensor_ids"),
        "require_bidirectional_labels": _as_bool(merged["require_bidirectional_labels"]),
        "enable_gold_fill": _as_bool(merged["enable_gold_fill"]),
        "prediction_fill_mode": prediction_fill_mode,
        "min_pairs_without_prediction": _positive_int(
            merged["min_pairs_without_prediction"],
            "min_pairs_without_prediction",
        ),
        "enable_legacy_regularization": _as_bool(merged["enable_legacy_regularization"]),
        "legacy_reg_lambda": _nonnegative_float(merged["legacy_reg_lambda"], "legacy_reg_lambda"),
        "legacy_reg_temperature": _positive_float(merged["legacy_reg_temperature"], "legacy_reg_temperature"),
        "legacy_teacher_checkpoint": str(merged.get("legacy_teacher_checkpoint") or ""),
        "enable_same_structure_regularization": _as_bool(merged["enable_same_structure_regularization"]),
        "same_structure_reg_lambda": _nonnegative_float(merged["same_structure_reg_lambda"], "same_structure_reg_lambda"),
        "same_structure_reg_temperature": _positive_float(
            merged["same_structure_reg_temperature"],
            "same_structure_reg_temperature",
        ),
        "same_structure_lr_scale": _positive_float(merged["same_structure_lr_scale"], "same_structure_lr_scale"),
        "inplane_loss_weight": _positive_float(merged["inplane_loss_weight"], "inplane_loss_weight"),
        "outplane_loss_weight": _positive_float(merged["outplane_loss_weight"], "outplane_loss_weight"),
        "branch_dropout_prob": _dropout_float(merged["branch_dropout_prob"], "branch_dropout_prob"),
        "fusion_hidden_dim": _positive_int(merged["fusion_hidden_dim"], "fusion_hidden_dim"),
        "fusion_dropout": _dropout_float(merged["fusion_dropout"], "fusion_dropout"),
        "cross_attn_heads": _positive_int(merged["cross_attn_heads"], "cross_attn_heads"),
        "epochs": _positive_int(merged["epochs"], "epochs"),
        "batch_size": _positive_int(merged["batch_size"], "batch_size"),
        "learning_rate": _positive_float(merged["learning_rate"], "learning_rate"),
        "weight_decay": _nonnegative_float(merged["weight_decay"], "weight_decay"),
        "gradient_clip_norm": _nonnegative_float(merged["gradient_clip_norm"], "gradient_clip_norm"),
        "focal_gamma": _nonnegative_float(merged["focal_gamma"], "focal_gamma"),
        "early_stopping_patience": int(merged["early_stopping_patience"]),
        "early_stopping_min_delta": _nonnegative_float(
            merged["early_stopping_min_delta"],
            "early_stopping_min_delta",
        ),
    }
    if normalized["early_stopping_patience"] < 0:
        raise ValueError("early_stopping_patience 不能小于 0")
    normalized["enable_wind_features"] = False
    normalized["wind_feature_dim"] = 0
    if normalized["wind_feature_dim"] < 0:
        raise ValueError("wind_feature_dim 不能小于 0")
    if normalized["enable_wind_features"] and normalized["wind_feature_dim"] != WIND_FEATURE_DIM:
        raise ValueError(f"当前短窗风特征维度固定为 {WIND_FEATURE_DIM}")
    if normalized["same_structure_lr_scale"] > 1.0:
        raise ValueError("same_structure_lr_scale 不能大于 1")
    if not _uses_long_context_model(model_type):
        normalized["enable_long_context"] = False
        normalized["context_mode"] = "short_only"
    else:
        normalized["enable_long_context"] = normalized["context_mode"] == "short_long"
    if model_type == "dual_stream_single_head":
        normalized["enable_legacy_regularization"] = False
        normalized["enable_same_structure_regularization"] = False
        normalized["require_bidirectional_labels"] = False
        normalized["enable_gold_fill"] = False
        normalized["prediction_fill_mode"] = "off"
        normalized["enable_wind_features"] = False
        normalized["wind_feature_dim"] = 0
    return normalized


def profile_summary(profile: dict) -> dict:
    return {
        "model_type": profile["model_type"],
        "context_mode": profile.get("context_mode", "short_only"),
        "enable_long_context": bool(profile["enable_long_context"]),
        "enable_wind_features": bool(profile["enable_wind_features"]),
        "wind_feature_dim": int(profile["wind_feature_dim"]),
        "train_val_ratio": float(profile["train_val_ratio"]),
        "enable_sensor_exclusion": bool(profile["enable_sensor_exclusion"]),
        "exclude_sensor_ids": list(profile.get("exclude_sensor_ids", [])),
        "require_bidirectional_labels": bool(profile["require_bidirectional_labels"]),
        "enable_gold_fill": bool(profile["enable_gold_fill"]),
        "prediction_fill_mode": str(profile["prediction_fill_mode"]),
        "min_pairs_without_prediction": int(profile["min_pairs_without_prediction"]),
        "enable_legacy_regularization": bool(profile["enable_legacy_regularization"]),
        "legacy_reg_lambda": float(profile["legacy_reg_lambda"]),
        "enable_same_structure_regularization": bool(profile["enable_same_structure_regularization"]),
        "same_structure_reg_lambda": float(profile["same_structure_reg_lambda"]),
        "same_structure_lr_scale": float(profile["same_structure_lr_scale"]),
        "branch_dropout_prob": float(profile["branch_dropout_prob"]),
        "fusion_dropout": float(profile["fusion_dropout"]),
        "weight_decay": float(profile["weight_decay"]),
        "early_stopping_patience": int(profile["early_stopping_patience"]),
        "early_stopping_min_delta": float(profile["early_stopping_min_delta"]),
        "epochs": int(profile["epochs"]),
        "batch_size": int(profile["batch_size"]),
        "learning_rate": float(profile["learning_rate"]),
        "fusion_hidden_dim": int(profile["fusion_hidden_dim"]),
        "cross_attn_heads": int(profile["cross_attn_heads"]),
    }



def load_training_profile(cfg: dict, round_idx: int, profile_path: str | None = None) -> dict:
    from src.chapter3_identifier.augment.workflow_config import (
        ensure_workflow_config,
        resolve_round_workflow,
        save_round_workflow_snapshot,
        training_profile_from_workflow,
    )

    if profile_path:
        path = resolve_path(profile_path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        profile = normalize_training_profile(payload, cfg, round_idx)
        return {
            "profile": profile,
            "metadata": {
                "source": "explicit_profile_path",
                "path": str(path),
                "round_idx": int(round_idx),
                "schema_version": SCHEMA_VERSION,
                "summary": profile_summary(profile),
            },
        }

    ensure_workflow_config(cfg)
    resolved = resolve_round_workflow(cfg, round_idx)
    save_round_workflow_snapshot(cfg, round_idx, resolved)
    return training_profile_from_workflow(resolved, cfg, round_idx)


def save_training_profile(cfg: dict, round_idx: int, payload: dict) -> dict:
    from src.chapter3_identifier.augment.workflow_config import save_round_training_override

    return save_round_training_override(cfg, round_idx, payload)


def reset_training_profile(cfg: dict, round_idx: int) -> dict:
    from src.chapter3_identifier.augment.workflow_config import reset_round_training_override

    return reset_round_training_override(cfg, round_idx)


def training_profile_options() -> dict:
    return {
        "model_types": [
            {"value": "dual_stream_single_head", "label": "DualStream 单头（time+spec）"},
            {"value": "quad_stream_dual_head", "label": "QuadStream 双头（time+spec）"},
            {"value": "quad_stream_dual_head_context", "label": "QuadStream 双头 + 长上下文"},
            {"value": "quad_stream_serial_context_dual_head", "label": "QuadStream 串行融合 + 长上下文"},
        ],
        "context_modes": [
            {"value": "short_only", "label": "仅短窗口"},
            {"value": "short_long", "label": "短窗口 + 长上下文"},
        ],
        "prediction_fill_modes": [
            {"value": "auto", "label": "自动：双向样本足够则不用 prediction"},
            {"value": "always", "label": "始终允许 prediction 补全"},
            {"value": "off", "label": "关闭 prediction 补全"},
        ],
        "ranges": {
            "fusion_dropout": [0.0, 0.95],
            "learning_rate": [1e-8, 1.0],
            "weight_decay": [0.0, 1.0],
            "gradient_clip_norm": [0.0, 100.0],
            "same_structure_lr_scale": [0.01, 1.0],
            "train_val_ratio": [0.05, 0.95],
        },
    }
