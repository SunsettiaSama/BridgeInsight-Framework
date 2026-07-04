from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.settings import (
    get_round_checkpoint_path,
    get_round_dir,
    get_round_train_profile_path,
    get_round_workflow_override_path,
    get_round_workflow_resolved_path,
    get_rounds_root,
    get_workflow_config_path,
    load_config,
    load_yaml,
)
from src.chapter3_identifier.augment.train.profile import (
    SCHEMA_VERSION,
    normalize_training_profile,
    profile_summary,
)

WORKFLOW_CONFIG_VERSION = 1
WORKFLOW_SECTIONS = ("dataset", "training", "inference", "finalize", "chapter4")
_TEMPLATE_PATH = Path(__file__).resolve().parent / "config" / "workflow_default.yaml"

_TRAINING_PROFILE_KEYS = {
    "model_type",
    "context_mode",
    "enable_long_context",
    "context_input_size",
    "context_total_seconds",
    "context_allow_cross_file",
    "enable_wind_features",
    "wind_feature_dim",
    "train_val_ratio",
    "enable_sensor_exclusion",
    "exclude_sensor_ids",
    "require_bidirectional_labels",
    "enable_gold_fill",
    "prediction_fill_mode",
    "min_pairs_without_prediction",
    "enable_legacy_regularization",
    "legacy_reg_lambda",
    "legacy_reg_temperature",
    "legacy_teacher_checkpoint",
    "enable_same_structure_regularization",
    "same_structure_reg_lambda",
    "same_structure_reg_temperature",
    "same_structure_lr_scale",
    "inplane_loss_weight",
    "outplane_loss_weight",
    "branch_dropout_prob",
    "fusion_hidden_dim",
    "fusion_dropout",
    "cross_attn_heads",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "focal_gamma",
    "early_stopping_patience",
    "early_stopping_min_delta",
}

_DATASET_RUNTIME_KEYS = {
    "use_pair_key_training_dataset",
    "pair_key_training_path",
    "random_seed",
    "train_val_ratio",
    "enable_sensor_exclusion",
    "exclude_sensor_ids",
    "enable_wind_features",
    "wind_feature_dim",
    "require_bidirectional_labels",
    "enable_gold_fill",
    "prediction_fill_mode",
    "min_pairs_without_prediction",
}

_INFERENCE_RUNTIME_KEYS = {
    "infer_batch_size",
    "infer_dataloader_workers",
    "infer_psd_workers",
    "infer_cache_max_mb",
    "infer_prefetch_files",
    "infer_prefetch_batches",
    "infer_prefetch_workers",
    "infer_context_workers",
    "infer_context_cache_entries",
    "infer_context_batch_mode",
    "infer_time_block_producer_workers",
    "infer_joint_queue_depth",
    "infer_record_chunk_size",
    "infer_record_workers",
    "prediction_projection_mode",
    "prediction_projection_direction",
    "infer_enable_sensor_exclusion",
    "infer_exclude_sensor_ids",
    "infer_exclude_gold_annotations",
    "infer_exclude_manual_annotations",
}


def get_workflow_template_path() -> Path:
    return _TEMPLATE_PATH


def get_chapter4_config_snapshot_path(cfg: dict, final_root: Path | None = None) -> Path:
    root = final_root if final_root is not None else get_rounds_root(cfg).parent / "final"
    return root / "chapter4_config_snapshot.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _section_defaults(template: dict) -> dict[str, dict]:
    defaults = template.get("workflow_defaults")
    if not isinstance(defaults, dict):
        raise ValueError("workflow_default.yaml 缺少 workflow_defaults")
    out: dict[str, dict] = {}
    for section in WORKFLOW_SECTIONS:
        payload = defaults.get(section, {})
        if not isinstance(payload, dict):
            raise ValueError(f"workflow_defaults.{section} 必须是 dict")
        out[section] = copy.deepcopy(payload)
    return out


def load_workflow_template() -> dict:
    return load_yaml(get_workflow_template_path())


def _normalize_workflow_payload(payload: dict) -> dict:
    version = int(payload.get("workflow_config_version", WORKFLOW_CONFIG_VERSION))
    defaults = payload.get("workflow_defaults")
    if not isinstance(defaults, dict):
        raise ValueError("workflow_config 缺少 workflow_defaults")
    normalized_defaults: dict[str, dict] = {}
    for section in WORKFLOW_SECTIONS:
        section_payload = defaults.get(section, {})
        if not isinstance(section_payload, dict):
            raise ValueError(f"workflow_defaults.{section} 必须是 dict")
        normalized_defaults[section] = copy.deepcopy(section_payload)
    round_overrides = payload.get("round_overrides", {})
    if round_overrides is None:
        round_overrides = {}
    if not isinstance(round_overrides, dict):
        raise ValueError("round_overrides 必须是 dict")
    return {
        "workflow_config_version": version,
        "workflow_defaults": normalized_defaults,
        "round_overrides": copy.deepcopy(round_overrides),
    }


def load_workflow_config(cfg: dict) -> dict:
    path = get_workflow_config_path(cfg)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        normalized = _normalize_workflow_payload(payload)
        normalized["_source"] = "saved"
        normalized["_path"] = str(path)
        return normalized
    migrated = bootstrap_workflow_config(cfg, baseline_round=8, write=True)
    migrated["_source"] = "bootstrap"
    migrated["_path"] = str(path)
    return migrated


def save_workflow_config(cfg: dict, payload: dict) -> dict:
    path = get_workflow_config_path(cfg)
    normalized = _normalize_workflow_payload(payload)
    normalized["workflow_config_version"] = int(
        normalized.get("workflow_config_version", WORKFLOW_CONFIG_VERSION)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "workflow_config_version": normalized["workflow_config_version"],
                "workflow_defaults": normalized["workflow_defaults"],
                "round_overrides": normalized.get("round_overrides", {}),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    saved = load_workflow_config(cfg)
    saved["_source"] = "saved"
    saved["_path"] = str(path)
    return saved


def ensure_workflow_config(cfg: dict) -> dict:
    path = get_workflow_config_path(cfg)
    if path.exists():
        return load_workflow_config(cfg)
    return bootstrap_workflow_config(cfg, baseline_round=8, write=True)


def _load_round_override_file(cfg: dict, round_idx: int) -> tuple[dict, str | None]:
    path = get_round_workflow_override_path(cfg, round_idx)
    if not path.exists():
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return payload, str(path)


def _load_legacy_train_profile(cfg: dict, round_idx: int) -> tuple[dict, str | None]:
    path = get_round_train_profile_path(cfg, round_idx)
    if not path.exists():
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return payload, str(path)


def _training_override_from_legacy(profile: dict) -> dict[str, Any]:
    return {key: profile[key] for key in _TRAINING_PROFILE_KEYS if key in profile}


def _round_override_layers(
    cfg: dict,
    workflow: dict,
    round_idx: int,
) -> tuple[list[dict], list[str]]:
    layers: list[dict] = []
    sources: list[str] = ["global_default"]
    global_key = str(int(round_idx))
    global_override = workflow.get("round_overrides", {}).get(global_key, {})
    if global_override:
        layers.append(copy.deepcopy(global_override))
        sources.append("global_round_override")
    file_override, file_path = _load_round_override_file(cfg, round_idx)
    if file_override:
        layers.append(copy.deepcopy(file_override))
        sources.append(f"round_override_file:{file_path}")
    legacy_profile, legacy_path = _load_legacy_train_profile(cfg, round_idx)
    if legacy_profile:
        legacy_training = _training_override_from_legacy(legacy_profile)
        if legacy_training:
            layers.append({"training": legacy_training})
            sources.append(f"legacy_train_profile:{legacy_path}")
    return layers, sources


def resolve_round_workflow(cfg: dict, round_idx: int, workflow: dict | None = None) -> dict:
    workflow = workflow if workflow is not None else ensure_workflow_config(cfg)
    resolved_sections = _section_defaults(workflow)
    override_layers, override_sources = _round_override_layers(cfg, workflow, round_idx)
    for layer in override_layers:
        resolved_sections = _deep_merge(resolved_sections, layer)
    training = resolved_sections["training"]
    dataset = resolved_sections["dataset"]
    if "train_val_ratio" in dataset and "train_val_ratio" not in training:
        training["train_val_ratio"] = dataset["train_val_ratio"]
    if "enable_sensor_exclusion" in dataset:
        training.setdefault("enable_sensor_exclusion", dataset["enable_sensor_exclusion"])
    if "exclude_sensor_ids" in dataset:
        training.setdefault("exclude_sensor_ids", dataset["exclude_sensor_ids"])
    if "enable_wind_features" in dataset:
        training.setdefault("enable_wind_features", dataset["enable_wind_features"])
        training.setdefault("wind_feature_dim", dataset.get("wind_feature_dim", 0))
    for dataset_key in (
        "require_bidirectional_labels",
        "enable_gold_fill",
        "prediction_fill_mode",
        "min_pairs_without_prediction",
    ):
        if dataset_key in dataset:
            training.setdefault(dataset_key, dataset[dataset_key])
    snapshot_path = get_round_workflow_resolved_path(cfg, round_idx)
    return {
        "workflow_config_version": int(workflow.get("workflow_config_version", WORKFLOW_CONFIG_VERSION)),
        "round_idx": int(round_idx),
        "resolved_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "sections": resolved_sections,
        "metadata": {
            "workflow_config_version": int(workflow.get("workflow_config_version", WORKFLOW_CONFIG_VERSION)),
            "workflow_config_path": str(get_workflow_config_path(cfg)),
            "workflow_source": workflow.get("_source", "saved"),
            "override_sources": override_sources,
            "snapshot_path": str(snapshot_path),
        },
    }


def save_round_workflow_snapshot(cfg: dict, round_idx: int, resolved: dict | None = None) -> Path:
    resolved = resolved if resolved is not None else resolve_round_workflow(cfg, round_idx)
    path = get_round_workflow_resolved_path(cfg, round_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, ensure_ascii=False, indent=2)
    return path


def apply_workflow_to_runtime_cfg(cfg: dict, resolved: dict) -> dict:
    runtime = dict(cfg)
    sections = resolved["sections"]
    dataset = sections.get("dataset", {})
    training = sections.get("training", {})
    inference = sections.get("inference", {})
    for key in _DATASET_RUNTIME_KEYS:
        if key in dataset:
            runtime[key] = copy.deepcopy(dataset[key])
    for key in _TRAINING_PROFILE_KEYS:
        if key in training:
            runtime[key] = copy.deepcopy(training[key])
    for key in _INFERENCE_RUNTIME_KEYS:
        if key in inference:
            runtime[key] = copy.deepcopy(inference[key])
    runtime["workflow_config_version"] = int(resolved.get("workflow_config_version", WORKFLOW_CONFIG_VERSION))
    runtime["workflow_resolved_path"] = resolved["metadata"]["snapshot_path"]
    runtime["workflow_config_path"] = resolved["metadata"]["workflow_config_path"]
    return runtime


def training_profile_from_workflow(resolved: dict, cfg: dict, round_idx: int) -> dict:
    sections = resolved["sections"]
    dataset = sections.get("dataset", {})
    training = sections.get("training", {})
    payload: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for key in _TRAINING_PROFILE_KEYS:
        if key in training:
            payload[key] = copy.deepcopy(training[key])
        elif key in dataset:
            payload[key] = copy.deepcopy(dataset[key])
    profile = normalize_training_profile(payload, cfg, round_idx)
    override_sources = list(resolved["metadata"].get("override_sources", []))
    source = "workflow"
    if any(item.startswith("legacy_train_profile:") for item in override_sources):
        source = "legacy_fallback"
    elif any("round_override" in item for item in override_sources):
        source = "round_override"
    return {
        "profile": profile,
        "metadata": {
            "source": source,
            "path": resolved["metadata"]["snapshot_path"],
            "round_idx": int(round_idx),
            "schema_version": SCHEMA_VERSION,
            "summary": profile_summary(profile),
            "workflow_config_version": int(resolved.get("workflow_config_version", WORKFLOW_CONFIG_VERSION)),
            "workflow_config_path": resolved["metadata"]["workflow_config_path"],
            "override_sources": override_sources,
        },
    }


def save_round_training_override(cfg: dict, round_idx: int, training_payload: dict) -> dict:
    workflow = load_workflow_config(cfg)
    resolved_before = resolve_round_workflow(cfg, round_idx, workflow=workflow)
    baseline = _section_defaults(workflow)
    global_key = str(int(round_idx))
    global_override = workflow.get("round_overrides", {}).get(global_key, {})
    merged_baseline = _deep_merge(baseline, {"training": global_override.get("training", {})})
    normalized = normalize_training_profile(training_payload, cfg, round_idx)
    diff: dict[str, Any] = {}
    baseline_training = merged_baseline["training"]
    for key in _TRAINING_PROFILE_KEYS:
        if key not in normalized:
            continue
        if normalized[key] != baseline_training.get(key):
            diff[key] = normalized[key]
    override_doc: dict[str, Any] = {}
    if diff:
        override_doc["training"] = diff
    path = get_round_workflow_override_path(cfg, round_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    if override_doc:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(override_doc, f, ensure_ascii=False, indent=2)
    elif path.exists():
        path.unlink()
    resolved = resolve_round_workflow(cfg, round_idx)
    save_round_workflow_snapshot(cfg, round_idx, resolved)
    payload = training_profile_from_workflow(resolved, cfg, round_idx)
    train_profile_path = get_round_train_profile_path(cfg, round_idx)
    with open(train_profile_path, "w", encoding="utf-8") as f:
        json.dump(payload["profile"], f, ensure_ascii=False, indent=2)
    return payload


def reset_round_training_override(cfg: dict, round_idx: int) -> dict:
    path = get_round_workflow_override_path(cfg, round_idx)
    if path.exists():
        path.unlink()
    train_profile_path = get_round_train_profile_path(cfg, round_idx)
    if train_profile_path.exists():
        train_profile_path.unlink()
    resolved = resolve_round_workflow(cfg, round_idx)
    save_round_workflow_snapshot(cfg, round_idx, resolved)
    return training_profile_from_workflow(resolved, cfg, round_idx)


def update_workflow_defaults(cfg: dict, section: str, updates: dict) -> dict:
    if section not in WORKFLOW_SECTIONS:
        raise ValueError(f"未知 workflow section：{section}")
    workflow = load_workflow_config(cfg)
    merged_section = _deep_merge(workflow["workflow_defaults"][section], updates)
    workflow["workflow_defaults"][section] = merged_section
    return save_workflow_config(cfg, workflow)


def update_workflow_global_round_override(cfg: dict, round_idx: int, override: dict) -> dict:
    workflow = load_workflow_config(cfg)
    round_overrides = dict(workflow.get("round_overrides", {}))
    key = str(int(round_idx))
    if override:
        round_overrides[key] = _deep_merge(round_overrides.get(key, {}), override)
    elif key in round_overrides:
        round_overrides.pop(key)
    workflow["round_overrides"] = round_overrides
    return save_workflow_config(cfg, workflow)


def bootstrap_workflow_config(cfg: dict, baseline_round: int = 8, write: bool = True) -> dict:
    template = load_workflow_template()
    payload = _normalize_workflow_payload(template)
    defaults = payload["workflow_defaults"]
    baseline_profile_path = get_round_train_profile_path(cfg, baseline_round)
    if baseline_profile_path.exists():
        with open(baseline_profile_path, "r", encoding="utf-8") as f:
            baseline_profile = json.load(f) or {}
        training_updates = _training_override_from_legacy(baseline_profile)
        defaults["training"] = _deep_merge(defaults["training"], training_updates)
        dataset_updates: dict[str, Any] = {}
        for key in (
            "train_val_ratio",
            "enable_sensor_exclusion",
            "exclude_sensor_ids",
            "require_bidirectional_labels",
            "enable_gold_fill",
            "prediction_fill_mode",
            "min_pairs_without_prediction",
        ):
            if key in baseline_profile:
                dataset_updates[key] = baseline_profile[key]
        dataset_updates["use_pair_key_training_dataset"] = bool(cfg.get("use_pair_key_training_dataset", True))
        dataset_updates["pair_key_training_path"] = cfg.get("pair_key_training_path")
        dataset_updates["random_seed"] = int(cfg.get("random_seed", 42))
        dataset_updates["enable_wind_features"] = False
        dataset_updates["wind_feature_dim"] = 0
        if cfg.get("exclude_sensor_ids"):
            dataset_updates["enable_sensor_exclusion"] = bool(cfg.get("enable_sensor_exclusion", True))
            dataset_updates["exclude_sensor_ids"] = list(cfg.get("exclude_sensor_ids", []))
        defaults["dataset"] = _deep_merge(defaults["dataset"], dataset_updates)
    else:
        defaults["dataset"]["use_pair_key_training_dataset"] = bool(cfg.get("use_pair_key_training_dataset", True))
        defaults["dataset"]["random_seed"] = int(cfg.get("random_seed", 42))
        if cfg.get("exclude_sensor_ids"):
            defaults["dataset"]["enable_sensor_exclusion"] = bool(cfg.get("enable_sensor_exclusion", True))
            defaults["dataset"]["exclude_sensor_ids"] = list(cfg.get("exclude_sensor_ids", []))
    defaults["dataset"]["enable_wind_features"] = False
    defaults["dataset"]["wind_feature_dim"] = 0
    if cfg.get("exclude_sensor_ids"):
        defaults["dataset"]["enable_sensor_exclusion"] = bool(cfg.get("enable_sensor_exclusion", True))
        defaults["dataset"]["exclude_sensor_ids"] = list(cfg.get("exclude_sensor_ids", []))
    for dataset_key in (
        "train_val_ratio",
        "enable_sensor_exclusion",
        "exclude_sensor_ids",
        "enable_wind_features",
        "wind_feature_dim",
        "require_bidirectional_labels",
        "enable_gold_fill",
        "prediction_fill_mode",
        "min_pairs_without_prediction",
    ):
        if dataset_key in defaults["dataset"]:
            defaults["training"][dataset_key] = copy.deepcopy(defaults["dataset"][dataset_key])
    inference_updates = {
        key: cfg[key]
        for key in _INFERENCE_RUNTIME_KEYS
        if key in cfg
    }
    if cfg.get("inference_dataset_config"):
        inference_updates["inference_dataset_config"] = str(cfg["inference_dataset_config"])
    if cfg.get("exclude_sensor_ids"):
        inference_updates["infer_enable_sensor_exclusion"] = bool(cfg.get("infer_enable_sensor_exclusion", True))
        infer_exclude_sensor_ids = cfg.get("infer_exclude_sensor_ids") or cfg.get("exclude_sensor_ids", [])
        inference_updates["infer_exclude_sensor_ids"] = list(infer_exclude_sensor_ids)
    defaults["inference"] = _deep_merge(defaults["inference"], inference_updates)
    defaults["finalize"]["canonical_round"] = int(baseline_round)
    chapter4 = defaults["chapter4"]
    chapter4["identifier_checkpoint_round"] = int(baseline_round)
    chapter4["dual_stream_config"] = str(cfg.get("dual_stream_config", chapter4.get("dual_stream_config", "")))
    chapter4["fs"] = float(cfg.get("fs", chapter4.get("fs", 50.0)))
    chapter4["nfft"] = int(cfg.get("nfft", chapter4.get("nfft", 2048)))
    chapter4["freq_max_hz"] = float(cfg.get("freq_max_hz", chapter4.get("freq_max_hz", 25.0)))
    for infer_key in _INFERENCE_RUNTIME_KEYS:
        if infer_key in defaults["inference"]:
            chapter4[infer_key] = defaults["inference"][infer_key]
    payload["workflow_defaults"] = defaults
    payload["workflow_config_version"] = WORKFLOW_CONFIG_VERSION
    if write:
        return save_workflow_config(cfg, payload)
    normalized = _normalize_workflow_payload(payload)
    normalized["_source"] = "bootstrap"
    normalized["_path"] = str(get_workflow_config_path(cfg))
    return normalized


def build_chapter4_config_snapshot(
    cfg: dict,
    final_root: Path | None = None,
    canonical_round: int | None = None,
) -> Path:
    workflow = ensure_workflow_config(cfg)
    effective_round = int(
        canonical_round
        if canonical_round is not None
        else workflow["workflow_defaults"]["finalize"].get("canonical_round", 8)
    )
    resolved = resolve_round_workflow(cfg, effective_round)
    chapter4_section = copy.deepcopy(resolved["sections"]["chapter4"])
    checkpoint_round = int(chapter4_section.pop("identifier_checkpoint_round", effective_round))
    chapter4_section["identifier_checkpoint_path"] = str(get_round_checkpoint_path(cfg, checkpoint_round))
    chapter4_section["workflow_config_version"] = int(resolved.get("workflow_config_version", WORKFLOW_CONFIG_VERSION))
    chapter4_section["workflow_resolved_path"] = str(get_round_workflow_resolved_path(cfg, effective_round))
    chapter4_section["workflow_config_path"] = str(get_workflow_config_path(cfg))
    chapter4_section["canonical_round"] = effective_round
    for key in ("chapter4_output_dir", "inference_dataset_config", "dual_stream_config"):
        if key in chapter4_section and chapter4_section[key]:
            chapter4_section[key] = str(resolve_path(str(chapter4_section[key])))
    snapshot_path = get_chapter4_config_snapshot_path(cfg, final_root=final_root)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(chapter4_section, f, allow_unicode=True, sort_keys=False)
    return snapshot_path


def load_chapter4_runtime_config(cfg: dict, config_path: str | None = None) -> dict:
    if config_path:
        runtime = load_config(config_path)
        runtime["_config_path"] = str(resolve_path(config_path))
        return runtime
    workflow = ensure_workflow_config(cfg)
    snapshot_path = get_chapter4_config_snapshot_path(cfg)
    if snapshot_path.exists():
        runtime = load_yaml(snapshot_path)
        runtime["_config_path"] = str(snapshot_path)
        return runtime
    build_chapter4_config_snapshot(cfg)
    runtime = load_yaml(snapshot_path)
    runtime["_config_path"] = str(snapshot_path)
    return runtime


def _flatten_resolved(resolved: dict) -> dict:
    flat: dict[str, Any] = {}
    for section, payload in resolved.get("sections", {}).items():
        if isinstance(payload, dict):
            for key, value in payload.items():
                flat[f"{section}.{key}"] = value
    return flat


def diff_resolved_workflows(left: dict, right: dict) -> dict:
    left_flat = _flatten_resolved(left)
    right_flat = _flatten_resolved(right)
    keys = sorted(set(left_flat) | set(right_flat))
    differences: list[dict[str, Any]] = []
    for key in keys:
        left_value = left_flat.get(key)
        right_value = right_flat.get(key)
        if left_value != right_value:
            differences.append(
                {
                    "key": key,
                    "left": left_value,
                    "right": right_value,
                }
            )
    return {
        "left_round": int(left.get("round_idx", 0)),
        "right_round": int(right.get("round_idx", 0)),
        "difference_count": len(differences),
        "differences": differences,
        "left_override_sources": list(left.get("metadata", {}).get("override_sources", [])),
        "right_override_sources": list(right.get("metadata", {}).get("override_sources", [])),
    }


def validate_round_reproducibility(cfg: dict, round_idx: int) -> dict:
    snapshot_path = get_round_workflow_resolved_path(cfg, round_idx)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"round {round_idx} 缺少 workflow_resolved.json：{snapshot_path}")
    with open(snapshot_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    recomputed = resolve_round_workflow(cfg, round_idx)
    diff = diff_resolved_workflows(saved, recomputed)
    checks = {
        "snapshot_exists": True,
        "workflow_config_version_match": int(saved.get("workflow_config_version", 0))
        == int(recomputed.get("workflow_config_version", 0)),
        "sections_match": diff["difference_count"] == 0,
        "difference_count": diff["difference_count"],
    }
    checks["reproducible"] = all(
        checks[key]
        for key in ("snapshot_exists", "workflow_config_version_match", "sections_match")
    )
    return {
        "round_idx": int(round_idx),
        "snapshot_path": str(snapshot_path),
        "checks": checks,
        "diff": diff,
        "saved_override_sources": list(saved.get("metadata", {}).get("override_sources", [])),
        "recomputed_override_sources": list(recomputed.get("metadata", {}).get("override_sources", [])),
    }


def workflow_schema() -> dict:
    template = load_workflow_template()
    return {
        "workflow_config_version": WORKFLOW_CONFIG_VERSION,
        "sections": WORKFLOW_SECTIONS,
        "defaults": template.get("workflow_defaults", {}),
        "round_override_example": template.get("round_overrides", {}),
    }


def diff_round_snapshots(cfg: dict, round_a: int, round_b: int) -> dict:
    path_a = get_round_workflow_resolved_path(cfg, round_a)
    path_b = get_round_workflow_resolved_path(cfg, round_b)
    if not path_a.exists():
        resolved_a = resolve_round_workflow(cfg, round_a)
        save_round_workflow_snapshot(cfg, round_a, resolved_a)
    else:
        with open(path_a, "r", encoding="utf-8") as f:
            resolved_a = json.load(f)
    if not path_b.exists():
        resolved_b = resolve_round_workflow(cfg, round_b)
        save_round_workflow_snapshot(cfg, round_b, resolved_b)
    else:
        with open(path_b, "r", encoding="utf-8") as f:
            resolved_b = json.load(f)
    return diff_resolved_workflows(resolved_a, resolved_b)
