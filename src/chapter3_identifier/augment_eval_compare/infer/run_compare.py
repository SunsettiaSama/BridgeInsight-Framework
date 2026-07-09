from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.chapter3_identifier.augment.annotation.sample_key import pair_key_from_paths
from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.settings import get_round_inference_path
from src.chapter3_identifier.augment.workflow_config import (
    apply_workflow_to_runtime_cfg,
    ensure_workflow_config,
    resolve_round_workflow,
)
from src.chapter3_identifier.augment_eval_compare.dataset.eval_pairs import EvalPairDataset
from src.chapter3_identifier.augment_eval_compare.settings import (
    get_compare_output_dir,
    resolve_decoupled_checkpoint,
    resolve_joint_checkpoint,
)

logger = logging.getLogger(__name__)


def _proba_to_pred(proba: np.ndarray) -> int:
    return int(np.argmax(proba))


def _effective_num_workers(configured_workers: int) -> int:
    workers = max(0, int(configured_workers))
    if os.name == "nt" and workers > 0:
        logger.warning(
            "Windows eval 推理禁用 DataLoader 多进程：configured_workers=%s",
            workers,
        )
        return 0
    return workers


def _build_runner(identifier: DualStreamIdentifier, augment_cfg: dict) -> DualStreamInferenceRunner:
    return DualStreamInferenceRunner(
        identifier,
        batch_size=int(augment_cfg.get("infer_batch_size", 256)),
        num_workers=_effective_num_workers(int(augment_cfg.get("infer_dataloader_workers", 4))),
        psd_workers=int(augment_cfg.get("infer_psd_workers", 2)),
        fs=float(augment_cfg.get("fs", 50.0)),
        nfft=int(augment_cfg.get("nfft", 2048)),
        freq_max_hz=float(augment_cfg.get("freq_max_hz", 25.0)),
        wind_config=augment_cfg,
        cache_max_mb=float(augment_cfg.get("infer_cache_max_mb", 512)),
        prefetch_files=int(augment_cfg.get("infer_prefetch_files", 4)),
        prefetch_batches=int(augment_cfg.get("infer_prefetch_batches", 2)),
        prefetch_workers=int(augment_cfg.get("infer_prefetch_workers", 2)),
        context_workers=int(augment_cfg.get("infer_context_workers", 2)),
        context_cache_entries=int(augment_cfg.get("infer_context_cache_entries", 64)),
        context_batch_mode=str(augment_cfg.get("infer_context_batch_mode", "time_block")),
        time_block_producer_workers=int(augment_cfg.get("infer_time_block_producer_workers", 1)),
        joint_queue_depth=int(augment_cfg.get("infer_joint_queue_depth", 2)),
    )


def _load_identifier(checkpoint_path: Path, augment_cfg: dict) -> DualStreamIdentifier:
    return DualStreamIdentifier.from_checkpoint(
        str(checkpoint_path),
        model_config_path=augment_cfg.get("dual_stream_config"),
        fs=float(augment_cfg.get("fs", 50.0)),
        nfft=int(augment_cfg.get("nfft", 2048)),
        freq_max_hz=float(augment_cfg.get("freq_max_hz", 25.0)),
        wind_config=augment_cfg,
    )


def run_checkpoint_inference(
    dataset: EvalPairDataset,
    checkpoint_path: Path,
    augment_cfg: dict,
) -> list[dict[str, Any]]:
    identifier = _load_identifier(checkpoint_path, augment_cfg)
    runner = _build_runner(identifier, augment_cfg)
    inplane_p, outplane_p = runner.run_with_proba(dataset)

    rows: list[dict[str, Any]] = []
    for sample_idx, entry in enumerate(dataset.entries):
        in_proba = inplane_p.get(sample_idx)
        out_proba = outplane_p.get(sample_idx)
        if in_proba is None or out_proba is None:
            raise RuntimeError(
                f"样本 {sample_idx} 缺少推理结果：pair_key={entry.pair_key}"
            )
        in_pred = _proba_to_pred(in_proba)
        out_pred = _proba_to_pred(out_proba)
        rows.append(
            {
                "sample_idx": sample_idx,
                "pair_key": list(entry.pair_key),
                "inplane_file_path": entry.inplane_file_path,
                "outplane_file_path": entry.outplane_file_path,
                "window_index": entry.window_index,
                "inplane_annotation": entry.inplane_annotation,
                "outplane_annotation": entry.outplane_annotation,
                "inplane_prediction": in_pred,
                "outplane_prediction": out_pred,
                "inplane_proba": [float(x) for x in in_proba],
                "outplane_proba": [float(x) for x in out_proba],
                "model_type": identifier.model_type,
                "checkpoint": str(checkpoint_path),
            }
        )
    return rows


def _inference_lookup(inference_path: Path) -> dict[tuple, dict]:
    with open(inference_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    lookup: dict[tuple, dict] = {}
    for rec in records:
        in_fp = rec.get("inplane_file_path")
        out_fp = rec.get("outplane_file_path")
        wi = rec.get("window_index")
        if not in_fp or not out_fp or wi is None:
            continue
        key = pair_key_from_paths(str(in_fp), str(out_fp), int(wi))
        lookup[key] = rec
    return lookup


def load_joint_predictions_from_inference(
    dataset: EvalPairDataset,
    inference_path: Path,
    checkpoint_path: Path,
    model_type: str,
    *,
    strict: bool = True,
) -> list[dict[str, Any]] | None:
    lookup = _inference_lookup(inference_path)
    rows: list[dict[str, Any]] = []
    missing: list[tuple] = []
    for sample_idx, entry in enumerate(dataset.entries):
        rec = lookup.get(entry.pair_key)
        if rec is None:
            missing.append(entry.pair_key)
            continue
        in_proba = rec.get("inplane_proba")
        out_proba = rec.get("outplane_proba")
        if not in_proba or not out_proba:
            missing.append(entry.pair_key)
            continue
        rows.append(
            {
                "sample_idx": sample_idx,
                "pair_key": list(entry.pair_key),
                "inplane_file_path": entry.inplane_file_path,
                "outplane_file_path": entry.outplane_file_path,
                "window_index": entry.window_index,
                "inplane_annotation": entry.inplane_annotation,
                "outplane_annotation": entry.outplane_annotation,
                "inplane_prediction": int(rec.get("inplane_prediction", _proba_to_pred(np.asarray(in_proba)))),
                "outplane_prediction": int(rec.get("outplane_prediction", _proba_to_pred(np.asarray(out_proba)))),
                "inplane_proba": [float(x) for x in in_proba],
                "outplane_proba": [float(x) for x in out_proba],
                "model_type": model_type,
                "checkpoint": str(checkpoint_path),
                "source": "inference_json",
            }
        )
    if missing:
        message = (
            f"inference.json 缺少 {len(missing)} 个 eval pair；"
            f"示例 missing={missing[:3]}"
        )
        if strict:
            raise RuntimeError(message)
        logger.warning("%s", message)
        return None
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_compare(
    cfg: dict,
    round_idx: int,
    dataset: EvalPairDataset,
    *,
    skip_decoupled: bool = False,
    skip_joint: bool = False,
) -> dict[str, Path]:
    augment_cfg = dict(cfg["_augment_cfg"])
    ensure_workflow_config(augment_cfg)
    resolved = resolve_round_workflow(augment_cfg, round_idx)
    augment_cfg = apply_workflow_to_runtime_cfg(augment_cfg, resolved)

    output_dir = get_compare_output_dir(cfg, round_idx)
    output_dir.mkdir(parents=True, exist_ok=True)

    decoupled_ckpt = resolve_decoupled_checkpoint(cfg, round_idx)
    joint_ckpt = resolve_joint_checkpoint(cfg, round_idx)

    paths: dict[str, Path] = {
        "output_dir": output_dir,
        "decoupled_predictions": output_dir / "predictions_decoupled.json",
        "joint_predictions": output_dir / "predictions_joint.json",
    }

    if not skip_decoupled:
        logger.info("解耦模型推理：%s", decoupled_ckpt)
        decoupled_rows = run_checkpoint_inference(dataset, decoupled_ckpt, augment_cfg)
        _write_json(paths["decoupled_predictions"], decoupled_rows)

    if not skip_joint:
        reuse = bool(cfg.get("reuse_joint_predictions", True))
        inference_path = get_round_inference_path(augment_cfg, round_idx)
        identifier = _load_identifier(joint_ckpt, augment_cfg)
        joint_rows = None
        if reuse and inference_path.exists():
            logger.info("尝试复用 round_%02d inference.json：%s", round_idx, inference_path)
            joint_rows = load_joint_predictions_from_inference(
                dataset,
                inference_path,
                joint_ckpt,
                identifier.model_type,
                strict=False,
            )
            if joint_rows is None:
                logger.warning(
                    "inference.json 未完整覆盖 eval pair，改为对评估集运行联合模型推理"
                )
        if joint_rows is None:
            logger.info("联合模型推理：%s", joint_ckpt)
            joint_rows = run_checkpoint_inference(dataset, joint_ckpt, augment_cfg)
        _write_json(paths["joint_predictions"], joint_rows)

    manifest = {
        "round_idx": int(round_idx),
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "eval_split": str(cfg.get("eval_split", "val")),
        "eval_pair_count": len(dataset.entries),
        "decoupled_checkpoint": str(decoupled_ckpt),
        "joint_checkpoint": str(joint_ckpt),
        "reuse_joint_predictions": bool(cfg.get("reuse_joint_predictions", True)),
        "label_names": get_label_names(),
    }
    manifest_path = output_dir / "eval_manifest.json"
    _write_json(manifest_path, manifest)
    paths["eval_manifest"] = manifest_path
    return paths


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"predictions 文件必须是列表：{path}")
    return payload
