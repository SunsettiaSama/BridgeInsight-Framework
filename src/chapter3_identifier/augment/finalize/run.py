from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.settings import (
    get_round_checkpoint_path,
    get_round_dir,
    get_round_inference_path,
    get_round_manifest_path,
    get_round_manual_delta_path,
    get_round_manual_edits_path,
    get_round_manual_history_path,
    get_round_merged_training_path,
    get_round_merged_training_pair_key_path,
    get_round_pair_key_migration_report_path,
    get_round_train_profile_path,
    get_round_workflow_resolved_path,
    get_rounds_root,
    load_config,
)
from src.chapter3_identifier.augment.train.profile import (
    load_training_profile,
    normalize_training_profile,
    profile_summary,
    save_training_profile,
)
from src.chapter3_identifier.augment.workflow_config import (
    build_chapter4_config_snapshot,
    ensure_workflow_config,
    resolve_round_workflow,
)

ensure_paths()

logger = logging.getLogger(__name__)

_MANUAL_ARTIFACTS = (
    "manual_edits.json",
    "manual_edits_delta.jsonl",
    "manual_edits_history.jsonl",
)
_ROUND1_COPY_ARTIFACTS = (
    "best_checkpoint.pth",
    "train_profile.json",
    "merged_training.json",
    "metrics.json",
    "metrics_live.json",
    "inference.json",
    "round_manifest.json",
)


def _json_len(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return len(records)
    return None


def _jsonl_len(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _path_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
    }


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    return payload if isinstance(payload, dict) else {}


def _load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or []
    return payload if isinstance(payload, list) else []


def _latest_metrics_summary(path: Path) -> dict:
    rows = _load_json_list(path)
    if not rows:
        return {"exists": path.exists(), "epoch_count": 0}
    last = rows[-1] if isinstance(rows[-1], dict) else {}
    return {
        "exists": True,
        "epoch_count": len(rows),
        "last_epoch": last.get("epoch"),
        "last_train_loss": last.get("train_loss"),
        "last_val_loss": last.get("val_loss"),
        "last_val_metrics": last.get("val_metrics"),
    }


def _count_confusion_matrices(round_dir: Path) -> dict:
    return {
        "merged": len(list(round_dir.glob("confusion_matrix_epoch_*.png"))),
        "inplane": len(list(round_dir.glob("confusion_matrix_inplane_epoch_*.png"))),
        "outplane": len(list(round_dir.glob("confusion_matrix_outplane_epoch_*.png"))),
    }


def _final_round_artifact_report(final_cfg: dict, source_cfg: dict, round_idx: int, stage: str) -> dict:
    round_dir = get_round_dir(final_cfg, round_idx)
    metrics_path = round_dir / "metrics.json"
    metrics_live_path = round_dir / "metrics_live.json"
    inference_path = get_round_inference_path(final_cfg, round_idx)
    workflow_path = get_round_workflow_resolved_path(final_cfg, round_idx)
    inference_profile_path = round_dir / "inference_profile.json"
    manifest_path = get_round_manifest_path(final_cfg, round_idx)
    train_profile_path = get_round_train_profile_path(final_cfg, round_idx)
    merged_path = get_round_merged_training_path(final_cfg, round_idx)
    pair_key_path = get_round_merged_training_pair_key_path(final_cfg, round_idx)
    pair_report_path = get_round_pair_key_migration_report_path(final_cfg, round_idx)
    inference_manifest = _load_json_dict(manifest_path)
    inference_profile = _load_json_dict(inference_profile_path)
    artifacts = {
        "round_dir": str(round_dir),
        "checkpoint": _path_info(get_round_checkpoint_path(final_cfg, round_idx)),
        "train_profile": _path_info(train_profile_path),
        "merged_training": {**_path_info(merged_path), "rows": _json_len(merged_path)},
        "merged_training_pair_key": {**_path_info(pair_key_path), "rows": _json_len(pair_key_path)},
        "pair_key_migration_report": _path_info(pair_report_path),
        "metrics": {**_path_info(metrics_path), **_latest_metrics_summary(metrics_path)},
        "metrics_live": _path_info(metrics_live_path),
        "inference": {**_path_info(inference_path), "records": _json_len(inference_path)},
        "inference_profile": _path_info(inference_profile_path),
        "workflow_resolved": _path_info(workflow_path),
        "round_manifest": _path_info(manifest_path),
        "manual_edits": {**_path_info(get_round_manual_edits_path(final_cfg, round_idx)), "rows": _json_len(get_round_manual_edits_path(final_cfg, round_idx))},
        "manual_edits_delta": {**_path_info(get_round_manual_delta_path(final_cfg, round_idx)), "rows": _jsonl_len(get_round_manual_delta_path(final_cfg, round_idx))},
        "manual_edits_history": {**_path_info(get_round_manual_history_path(final_cfg, round_idx)), "rows": _jsonl_len(get_round_manual_history_path(final_cfg, round_idx))},
        "confusion_matrices": _count_confusion_matrices(round_dir),
    }
    return {
        "round_idx": int(round_idx),
        "stage": stage,
        "copy_only": stage == "copy_round1",
        "copy_only_note": (
            "round1 为基线复制轮；若源轮次没有 workflow_resolved.json 或 inference_profile.json，final 中也不会补齐。"
            if stage == "copy_round1"
            else None
        ),
        "artifacts": artifacts,
        "inference_manifest_summary": {
            "record_count": inference_manifest.get("record_count"),
            "filters": inference_manifest.get("filters"),
            "projection_mode": inference_manifest.get("projection_mode"),
            "workflow_resolved_path": inference_manifest.get("workflow_resolved_path"),
        },
        "inference_profile_summary": {
            "stages": inference_profile.get("stages"),
            "runner_detail": inference_profile.get("runner_detail"),
            "config": inference_profile.get("config"),
            "throughput_samples_per_s": inference_profile.get("throughput_samples_per_s"),
        },
        "source_artifacts": {
            "round_dir": str(get_round_dir(source_cfg, round_idx)),
            "checkpoint": str(get_round_checkpoint_path(source_cfg, round_idx)),
            "inference": str(get_round_inference_path(source_cfg, round_idx)),
            "train_profile": str(get_round_train_profile_path(source_cfg, round_idx)),
            "merged_training": str(get_round_merged_training_path(source_cfg, round_idx)),
            "manual_edits": str(get_round_manual_edits_path(source_cfg, round_idx)),
        },
    }


def get_final_root(cfg: dict) -> Path:
    source_root = get_rounds_root(cfg)
    return source_root.parent / "final"


def discover_source_round_indices(cfg: dict) -> list[int]:
    root = get_rounds_root(cfg)
    if not root.exists():
        return []
    indices: list[int] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith("round_"):
            continue
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _profile_hash(profile: dict) -> str:
    payload = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _serialize_cfg_for_yaml(cfg: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if str(key).startswith("_"):
            continue
        out[key] = value
    return out


def _write_finalize_config(cfg: dict, final_root: Path) -> Path:
    final_rounds_dir = final_root / "rounds"
    merged = dict(cfg)
    merged["rounds_output_dir"] = str(final_rounds_dir)
    config_path = final_root / "finalize_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_serialize_cfg_for_yaml(merged), f, allow_unicode=True, sort_keys=False)
    return config_path


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _copy_manual_metadata(source_cfg: dict, final_cfg: dict, round_idx: int) -> dict:
    source_dir = get_round_dir(source_cfg, round_idx)
    final_dir = get_round_dir(final_cfg, round_idx)
    final_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    mapping = {
        "manual_edits.json": get_round_manual_edits_path(source_cfg, round_idx),
        "manual_edits_delta.jsonl": get_round_manual_delta_path(source_cfg, round_idx),
        "manual_edits_history.jsonl": get_round_manual_history_path(source_cfg, round_idx),
    }
    for name, source_path in mapping.items():
        target_path = final_dir / name
        if _copy_if_exists(source_path, target_path):
            copied.append(name)
        else:
            missing.append(name)
    return {"round_idx": round_idx, "copied": copied, "missing": missing}


def _copy_round1_baseline(source_cfg: dict, final_cfg: dict) -> dict:
    source_dir = get_round_dir(source_cfg, 1)
    final_dir = get_round_dir(final_cfg, 1)
    final_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for name in _ROUND1_COPY_ARTIFACTS:
        source_path = source_dir / name
        target_path = final_dir / name
        if _copy_if_exists(source_path, target_path):
            copied.append(name)
        else:
            missing.append(name)
    checkpoint = get_round_checkpoint_path(final_cfg, 1)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"终盘整理需要源 round1 checkpoint，但未找到：{get_round_checkpoint_path(source_cfg, 1)}"
        )
    return {"round_idx": 1, "copied": copied, "missing": missing, "checkpoint": str(checkpoint)}


def _count_manual_rows(cfg: dict, round_idx: int) -> int:
    path = get_round_manual_edits_path(cfg, round_idx)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return len(rows) if isinstance(rows, list) else 0


def build_finalize_summary(
    config_path: str | None = None,
    from_round: int = 1,
    to_round: int | None = None,
    canonical_round: int | None = None,
    overwrite_final: bool = False,
) -> dict:
    cfg = load_config(config_path)
    ensure_workflow_config(cfg)
    workflow = ensure_workflow_config(cfg)
    finalize_defaults = workflow["workflow_defaults"].get("finalize", {})
    if canonical_round is None and finalize_defaults.get("canonical_round") is not None:
        canonical_round = int(finalize_defaults["canonical_round"])
    if to_round is None and finalize_defaults.get("to_round") is not None:
        to_round = int(finalize_defaults["to_round"])
    discovered = discover_source_round_indices(cfg)
    if not discovered:
        raise ValueError(f"未找到任何源 round 目录：{get_rounds_root(cfg)}")
    if from_round < 1:
        raise ValueError("from_round 必须 >= 1")
    effective_to = int(to_round if to_round is not None else discovered[-1])
    if effective_to not in discovered:
        raise ValueError(f"to_round={effective_to} 不存在，可用 round：{discovered}")
    if from_round > effective_to:
        raise ValueError("from_round 不能大于 to_round")
    effective_canonical = int(canonical_round if canonical_round is not None else effective_to)
    if effective_canonical not in discovered:
        raise ValueError(f"canonical_round={effective_canonical} 不存在，可用 round：{discovered}")

    canonical_payload = load_training_profile(cfg, effective_canonical)
    canonical_profile = canonical_payload["profile"]
    final_root = get_final_root(cfg)
    final_exists = final_root.exists() and any(final_root.iterdir())
    retrain_rounds = list(range(max(2, from_round), effective_to + 1))

    manual_copy_plan = []
    for round_idx in range(from_round, effective_to + 1):
        manual_copy_plan.append(
            {
                "round_idx": round_idx,
                "manual_rows": _count_manual_rows(cfg, round_idx),
                "source_dir": str(get_round_dir(cfg, round_idx)),
            }
        )

    round1_checkpoint = get_round_checkpoint_path(cfg, 1)
    warnings: list[str] = []
    if not round1_checkpoint.exists():
        warnings.append(f"源 round1 缺少 best_checkpoint.pth：{round1_checkpoint}")
    if effective_canonical <= 1 and retrain_rounds:
        warnings.append("canonical_round=1 时 round2+ 将沿用 round1 单头结构，可能不符合“最后一轮模型结构”预期")
    if final_exists and not overwrite_final:
        warnings.append(f"final 目录已存在：{final_root}，请在前端勾选“覆盖已有 final”后再启动")

    return {
        "from_round": from_round,
        "to_round": effective_to,
        "canonical_round": effective_canonical,
        "source_rounds": discovered,
        "retrain_rounds": retrain_rounds,
        "final_root": str(final_root),
        "final_rounds_dir": str(final_root / "rounds"),
        "final_exists": final_exists,
        "overwrite_final": overwrite_final,
        "canonical_profile": profile_summary(canonical_profile),
        "canonical_profile_hash": _profile_hash(canonical_profile),
        "manual_copy_plan": manual_copy_plan,
        "round1_checkpoint": str(round1_checkpoint),
        "round1_checkpoint_exists": round1_checkpoint.exists(),
        "warnings": warnings,
        "ready": round1_checkpoint.exists() and (not final_exists or overwrite_final),
    }


def run_finalize(
    config_path: str | None = None,
    from_round: int = 1,
    to_round: int | None = None,
    canonical_round: int | None = None,
    overwrite_final: bool = False,
    dry_run: bool = False,
) -> dict:
    summary = build_finalize_summary(
        config_path=config_path,
        from_round=from_round,
        to_round=to_round,
        canonical_round=canonical_round,
        overwrite_final=overwrite_final,
    )
    if dry_run:
        logger.info("终盘整理 dry-run：%s", json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    if not summary["ready"]:
        raise RuntimeError("终盘整理前置检查未通过：" + "；".join(summary["warnings"]))

    from src.chapter3_identifier.augment.datasets.dual_stream_dataset import clear_round2_feature_cache
    from src.chapter3_identifier.augment.infer.run import run_inference
    from src.chapter3_identifier.augment.train.run import run_training

    cfg = load_config(config_path)
    workflow = ensure_workflow_config(cfg)
    clear_round2_feature_cache()
    if bool(cfg.get("enable_preload_cache", True)):
        cfg["preload_num_workers"] = 0
        logger.info("终盘重训启用进程内特征复用：preload_num_workers=0，仅更新每轮标签")
    effective_to = int(summary["to_round"])
    effective_canonical = int(summary["canonical_round"])
    final_root = Path(summary["final_root"])
    if final_root.exists() and overwrite_final:
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    (final_root / "logs").mkdir(parents=True, exist_ok=True)

    finalize_config_path = _write_finalize_config(cfg, final_root)
    chapter4_snapshot_path = build_chapter4_config_snapshot(cfg, final_root=final_root, canonical_round=effective_canonical)
    final_cfg = load_config(str(finalize_config_path))
    canonical_payload = load_training_profile(cfg, effective_canonical)
    canonical_profile = dict(canonical_payload["profile"])
    if bool(final_cfg.get("enable_sensor_exclusion", False)):
        canonical_profile["enable_sensor_exclusion"] = True
        canonical_profile["exclude_sensor_ids"] = list(final_cfg.get("exclude_sensor_ids", []))

    per_round_reports: list[dict] = []

    logger.info("=== 终盘整理：copy_metadata round %s..%s ===", from_round, effective_to)
    for round_idx in range(from_round, effective_to + 1):
        report = _copy_manual_metadata(cfg, final_cfg, round_idx)
        report["stage"] = "copy_metadata"
        per_round_reports.append(report)
        logger.info("copy_metadata round %s copied=%s missing=%s", round_idx, report["copied"], report["missing"])

    if from_round <= 1 <= effective_to:
        logger.info("=== 终盘整理：copy_round1 ===")
        round1_report = _copy_round1_baseline(cfg, final_cfg)
        round1_report["stage"] = "copy_round1"
        per_round_reports.append(round1_report)
        infer_path = get_round_inference_path(final_cfg, 1)
        if not infer_path.exists():
            logger.info("final round1 缺少 inference.json，基于已复制 checkpoint 补跑推理")
            run_inference(round_idx=1, config_path=str(finalize_config_path))

    for round_idx in range(max(2, from_round), effective_to + 1):
        logger.info("=== 终盘整理：train_round_%02d ===", round_idx)
        normalized = normalize_training_profile(canonical_profile, final_cfg, round_idx)
        save_training_profile(final_cfg, round_idx, normalized)
        run_training(
            round_idx=round_idx,
            gold_only=False,
            config_path=str(finalize_config_path),
        )
        per_round_reports.append({"round_idx": round_idx, "stage": f"train_round_{round_idx:02d}"})

        logger.info("=== 终盘整理：infer_round_%02d ===", round_idx)
        run_inference(round_idx=round_idx, config_path=str(finalize_config_path))
        per_round_reports.append({"round_idx": round_idx, "stage": f"infer_round_{round_idx:02d}"})

    logger.info("=== 终盘整理：write_manifest ===")
    round_artifact_reports = [
        _final_round_artifact_report(
            final_cfg,
            cfg,
            round_idx,
            stage="copy_round1" if round_idx == 1 and from_round <= 1 else f"train_infer_round_{round_idx:02d}",
        )
        for round_idx in range(from_round, effective_to + 1)
    ]
    manifest = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "finalized": True,
        "from_round": from_round,
        "to_round": effective_to,
        "canonical_round": effective_canonical,
        "canonical_profile_hash": summary["canonical_profile_hash"],
        "canonical_profile": summary["canonical_profile"],
        "source_rounds_root": str(get_rounds_root(cfg)),
        "final_rounds_root": str(final_root / "rounds"),
        "finalize_config": str(finalize_config_path),
        "chapter4_config_snapshot": str(chapter4_snapshot_path),
        "workflow_config_path": str(workflow.get("_path", "")),
        "workflow_config_version": int(workflow.get("workflow_config_version", 0)),
        "round_reports": per_round_reports,
        "round_artifact_reports": round_artifact_reports,
    }
    manifest_path = final_root / "final_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    artifact_by_round = {int(item["round_idx"]): item for item in round_artifact_reports}
    for round_idx in range(from_round, effective_to + 1):
        round_dir = get_round_dir(final_cfg, round_idx)
        round_manifest_path = round_dir / "round_manifest.json"
        round_manifest: dict[str, Any] = {}
        if round_manifest_path.exists():
            with open(round_manifest_path, "r", encoding="utf-8") as f:
                round_manifest = json.load(f) or {}
        round_manifest.update(
            {
                "finalized": True,
                "source_round": round_idx,
                "canonical_round": effective_canonical,
                "canonical_profile_hash": summary["canonical_profile_hash"],
                "source_rounds_root": str(get_rounds_root(cfg)),
            }
        )
        round_manifest["final_artifact_report"] = artifact_by_round.get(int(round_idx), {})
        with open(round_manifest_path, "w", encoding="utf-8") as f:
            json.dump(round_manifest, f, ensure_ascii=False, indent=2)

    logger.info("终盘整理完成：%s", manifest_path)
    return {
        **summary,
        "manifest_path": str(manifest_path),
        "finalize_config_path": str(finalize_config_path),
        "chapter4_config_snapshot_path": str(chapter4_snapshot_path),
        "round_reports": per_round_reports,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Augment 终盘整理：生成论文用 final rounds")
    parser.add_argument("--from-round", type=int, default=1)
    parser.add_argument("--to-round", type=int, default=None)
    parser.add_argument("--canonical-round", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overwrite-final", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    result = run_finalize(
        config_path=args.config,
        from_round=args.from_round,
        to_round=args.to_round,
        canonical_round=args.canonical_round,
        overwrite_final=args.overwrite_final,
        dry_run=args.dry_run,
    )
    logger.info("终盘整理结果：%s", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
