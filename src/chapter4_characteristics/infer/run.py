from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml

from src.chapter4_characteristics._bootstrap import ensure_paths, resolve_path
from src.chapter4_characteristics.infer.preflight import run_preflight
from src.chapter4_characteristics.infer.result_adapter import records_to_enriched_json
from src.chapter4_characteristics.settings import (
    get_identifier_checkpoint,
    get_inference_dir,
    get_inference_path,
    get_predictions_enriched_path,
)
from src.chapter3_identifier.augment.settings import load_config as load_augment_config
from src.chapter3_identifier.augment.workflow_config import load_chapter4_runtime_config

ensure_paths()

from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.profile_hooks import StageTimer
from src.chapter3_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.data_processer.datasets.StayCable_Vib2023.StayCableVib2023Dataset import StayCableVib2023Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DATASET_DEFAULTS = {
    "wind_metadata_path": None,
    "wind_sensor_ids": None,
    "require_wind_alignment": False,
    "enable_denoise": False,
    "denoise_freq_threshold": None,
    "missing_rate_threshold": 0.05,
    "time_ordered": True,
    "split_ratio": -1,
    "split_by_time": False,
    "split_seed": 42,
    "use_cache": True,
    "cache_path": None,
    "predictions_cache_path": None,
}


def _load_inference_dataset(config_path: str):
    cfg_path = resolve_path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = {**_DATASET_DEFAULTS, **(yaml.safe_load(f) or {})}
    raw["vib_metadata_path"] = str(resolve_path(raw["vib_metadata_path"]))
    if raw.get("wind_metadata_path"):
        raw["wind_metadata_path"] = str(resolve_path(raw["wind_metadata_path"]))
    if raw.get("cache_path"):
        raw["cache_path"] = str(resolve_path(raw["cache_path"]))
    if raw.get("predictions_cache_path"):
        raw["predictions_cache_path"] = str(resolve_path(raw["predictions_cache_path"]))
    return StayCableVib2023Dataset(config=SimpleNamespace(**raw))


def _record_to_json(
    rec,
    sample_idx: int,
    pred: int,
    proba,
    in_pred: int,
    out_pred: int,
    in_proba,
    out_proba,
    projection_mode: str,
) -> dict:
    in_meta = rec.inplane_meta or {}
    out_meta = rec.outplane_meta or {}
    m, d, h = rec.timestamp_key
    in_fp = in_meta.get("file_path", "")
    wi = rec.window_idx
    proba_list = [float(x) for x in proba]
    in_proba_list = [float(x) for x in in_proba]
    out_proba_list = [float(x) for x in out_proba]
    in_conf = float(max(in_proba_list)) if in_proba_list else 0.0
    out_conf = float(max(out_proba_list)) if out_proba_list else 0.0
    in_uncertainty = float(1.0 - in_conf)
    out_uncertainty = float(1.0 - out_conf)
    uncertainty = float(max(in_uncertainty, out_uncertainty))
    if in_pred == pred and out_pred == pred:
        primary_source = "both"
    elif in_pred == pred:
        primary_source = "inplane"
    elif out_pred == pred:
        primary_source = "outplane"
    else:
        primary_source = "merged"
    return {
        "sample_idx": sample_idx,
        "prediction": int(pred),
        "proba": proba_list,
        "uncertainty": uncertainty,
        "inplane_uncertainty": in_uncertainty,
        "outplane_uncertainty": out_uncertainty,
        "primary_prediction_source": primary_source,
        "projection_mode": projection_mode,
        "inplane_prediction": int(in_pred),
        "outplane_prediction": int(out_pred),
        "inplane_proba": in_proba_list,
        "outplane_proba": out_proba_list,
        "inplane_file_path": in_fp,
        "outplane_file_path": out_meta.get("file_path"),
        "window_index": wi,
        "inplane_sensor_id": in_meta.get("sensor_id"),
        "outplane_sensor_id": out_meta.get("sensor_id"),
        "timestamp": [m, d, h],
    }


def _build_chunk_rows(
    sample_indices: Sequence[int],
    samples,
    merged: Dict[int, Tuple[int, np.ndarray]],
    inplane_p: Dict[int, np.ndarray],
    outplane_p: Dict[int, np.ndarray],
    projection_mode: str,
) -> List[dict]:
    rows: List[dict] = []
    for sample_idx in sample_indices:
        rec = samples[sample_idx]
        pred, proba = merged[sample_idx]
        in_proba = inplane_p.get(sample_idx, proba)
        out_proba = outplane_p.get(sample_idx, proba)
        in_pred = int(in_proba.argmax()) if sample_idx in inplane_p else int(pred)
        out_pred = int(out_proba.argmax()) if sample_idx in outplane_p else int(pred)
        rows.append(
            _record_to_json(
                rec,
                sample_idx,
                pred,
                proba,
                in_pred,
                out_pred,
                in_proba,
                out_proba,
                projection_mode,
            )
        )
    return rows


def _assemble_records_parallel(
    merged_indices: List[int],
    samples,
    merged: Dict[int, Tuple[int, np.ndarray]],
    inplane_p: Dict[int, np.ndarray],
    outplane_p: Dict[int, np.ndarray],
    chunk_size: int,
    workers: int,
    projection_mode: str,
) -> List[dict]:
    total = len(merged_indices)
    chunks = [merged_indices[i : i + chunk_size] for i in range(0, total, chunk_size)]
    n_chunks = len(chunks)
    logger.info(
        f"记录组装：{total} 条，分 {n_chunks} 块（chunk_size={chunk_size}，workers={workers}）"
    )
    chunk_results: List[List[dict] | None] = [None] * n_chunks
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _build_chunk_rows,
                chunk,
                samples,
                merged,
                inplane_p,
                outplane_p,
                projection_mode,
            ): chunk_idx
            for chunk_idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            chunk_idx = futures[future]
            chunk_results[chunk_idx] = future.result()
            done += len(chunk_results[chunk_idx])
            pct = 100.0 * done / total
            logger.info(f"记录组装进度：{done}/{total} ({pct:.1f}%)")
    records: List[dict] = []
    for chunk in chunk_results:
        records.extend(chunk)
    return records


def _stream_write_inference_json(
    path: Path,
    header: dict,
    records: List[dict],
    chunk_size: int,
) -> None:
    total = len(records)
    logger.info(f"写入 {path.name}：{total} 条记录（分块流式写入，chunk_size={chunk_size}）")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "generated_at": {json.dumps(header["generated_at"], ensure_ascii=False)},\n')
        f.write(f'  "checkpoint": {json.dumps(header["checkpoint"], ensure_ascii=False)},\n')
        f.write(f'  "projection_mode": {json.dumps(header["projection_mode"], ensure_ascii=False)},\n')
        f.write(f'  "record_count": {total},\n')
        f.write('  "records": [\n')
        for i, record in enumerate(records):
            if i:
                f.write(",\n")
            f.write("    ")
            f.write(json.dumps(record, ensure_ascii=False))
            if chunk_size > 0 and (i + 1) % chunk_size == 0:
                pct = 100.0 * (i + 1) / total
                logger.info(f"JSON 写入进度：{i + 1}/{total} ({pct:.1f}%)")
        f.write("\n  ]\n}")
    logger.info(f"JSON 写入完成：{path}")


def _apply_limit(dataset, limit: int | None):
    if limit is None or limit <= 0:
        return dataset
    dataset._samples = dataset._samples[: int(limit)]
    logger.info(f"冒烟模式：限制样本数 {len(dataset._samples)}")
    return dataset


def _build_runner(
    cfg: dict,
    identifier: DualStreamIdentifier,
    timer: StageTimer | None = None,
) -> DualStreamInferenceRunner:
    return DualStreamInferenceRunner(
        identifier,
        batch_size=int(cfg.get("infer_batch_size", 256)),
        num_workers=int(cfg.get("infer_dataloader_workers", 4)),
        psd_workers=int(cfg.get("infer_psd_workers", 2)),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
        wind_config=cfg,
        cache_max_mb=float(cfg.get("infer_cache_max_mb", 512)),
        prefetch_files=int(cfg.get("infer_prefetch_files", 4)),
        prefetch_batches=int(cfg.get("infer_prefetch_batches", 2)),
        prefetch_workers=int(cfg.get("infer_prefetch_workers", 2)),
        context_workers=int(cfg.get("infer_context_workers", 2)),
        context_cache_entries=int(cfg.get("infer_context_cache_entries", 64)),
        context_batch_mode=str(cfg.get("infer_context_batch_mode", "time_block")),
        time_block_producer_workers=int(cfg.get("infer_time_block_producer_workers", 1)),
        joint_queue_depth=int(cfg.get("infer_joint_queue_depth", 2)),
        profile_stats=timer.timings.prefetch if timer is not None else None,
    )


def run_inference(
    limit: int | None = None,
    config_path: str | None = None,
) -> str:
    timer = StageTimer()
    augment_cfg = load_augment_config(None)
    cfg = load_chapter4_runtime_config(augment_cfg, config_path=config_path)
    if limit is None and int(cfg.get("dev_limit_samples", 0)) > 0:
        limit = int(cfg["dev_limit_samples"])

    with timer.stage("preflight"):
        run_preflight(config_path=config_path, cfg=cfg)

    out_dir = get_inference_dir(cfg)
    ckpt_path = get_identifier_checkpoint(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    with timer.stage("dataset"):
        dataset = _load_inference_dataset(cfg["inference_dataset_config"])
        dataset = _apply_limit(dataset, limit)
        timer.timings.extra["sample_count"] = int(len(getattr(dataset, "_samples", [])))

    chunk_size = int(cfg.get("infer_record_chunk_size", 4096))
    record_workers = int(cfg.get("infer_record_workers", 8))

    with timer.stage("checkpoint"):
        identifier = DualStreamIdentifier.from_checkpoint(
            str(ckpt_path),
            model_config_path=cfg.get("dual_stream_config"),
            fs=float(cfg["fs"]),
            nfft=int(cfg["nfft"]),
            freq_max_hz=float(cfg["freq_max_hz"]),
            wind_config=cfg,
        )
        timer.timings.extra["model_type"] = str(identifier.model_type)
        timer.timings.extra["context_mode"] = str(getattr(identifier, "context_mode", "short_only"))
        timer.timings.extra["wind_feature_dim"] = int(getattr(identifier, "wind_feature_dim", 0))
    runner = _build_runner(cfg, identifier, timer=timer)
    logger.info(
        "Chapter4 全量识别性能配置：batch=%s cache_mb=%s prefetch_files=%s "
        "prefetch_batches=%s prefetch_workers=%s context_workers=%s context_cache_entries=%s "
        "context_batch_mode=%s time_block_producer_workers=%s joint_queue_depth=%s",
        int(cfg.get("infer_batch_size", 256)),
        float(cfg.get("infer_cache_max_mb", 512)),
        int(cfg.get("infer_prefetch_files", 4)),
        int(cfg.get("infer_prefetch_batches", 2)),
        int(cfg.get("infer_prefetch_workers", 2)),
        int(cfg.get("infer_context_workers", 2)),
        int(cfg.get("infer_context_cache_entries", 64)),
        str(cfg.get("infer_context_batch_mode", "time_block")),
        int(cfg.get("infer_time_block_producer_workers", 1)),
        int(cfg.get("infer_joint_queue_depth", 2)),
    )

    with timer.stage("runner"):
        inplane_p, outplane_p = runner.run_with_proba(dataset)
    logger.info("面内/面外推理完成，正在合并结果 …")
    projection_mode = str(cfg.get("prediction_projection_mode", "direction"))
    projection_direction = str(cfg.get("prediction_projection_direction", "outplane"))
    with timer.stage("merge"):
        merged = runner.merge_proba_predictions(
            inplane_p,
            outplane_p,
            projection_mode=projection_mode,
            projection_direction=projection_direction,
        )
    projection_label = (
        f"direction:{projection_direction}"
        if projection_mode == "direction"
        else projection_mode
    )
    logger.info("推理总 prediction 投影模式：%s", projection_label)

    merged_indices = sorted(merged.keys())
    with timer.stage("records"):
        records = _assemble_records_parallel(
            merged_indices=merged_indices,
            samples=dataset._samples,
            merged=merged,
            inplane_p=inplane_p,
            outplane_p=outplane_p,
            chunk_size=chunk_size,
            workers=record_workers,
            projection_mode=projection_label,
        )

    records.sort(
        key=lambda r: (
            0 if r["prediction"] in (1, 2, 3) else 1,
            -r["uncertainty"],
            int(r.get("sample_idx", 0)),
        )
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stable_path = get_inference_path(cfg)
    enriched_path = get_predictions_enriched_path(cfg)
    archive_path = out_dir / f"inference_{ts}.json"

    header = {
        "generated_at": ts,
        "checkpoint": str(ckpt_path),
        "projection_mode": projection_label,
    }
    with timer.stage("inference_json"):
        _stream_write_inference_json(archive_path, header, records, chunk_size)
        _stream_write_inference_json(stable_path, header, records, chunk_size)

    with timer.stage("enriched_json"):
        enriched_payload = records_to_enriched_json(
            records,
            dataset,
            checkpoint_path=str(ckpt_path),
            dataset_config=str(cfg["inference_dataset_config"]),
        )
        with open(enriched_path, "w", encoding="utf-8") as f:
            json.dump(enriched_payload, f, ensure_ascii=False, indent=2)

    manifest = {
        "generated_at": ts,
        "checkpoint": str(ckpt_path),
        "inference": stable_path.name,
        "inference_archive": archive_path.name,
        "predictions_enriched": enriched_path.name,
        "projection_mode": projection_label,
        "record_count": len(records),
        "limit": limit,
        "workflow_config_version": int(cfg.get("workflow_config_version", 0)),
        "workflow_resolved_path": str(cfg.get("workflow_resolved_path", "")),
        "workflow_config_path": str(cfg.get("workflow_config_path", "")),
        "inference_profile": "inference_profile.json",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    profile_report = timer.timings.to_dict()
    profile_report["config"] = {
        "infer_batch_size": int(cfg.get("infer_batch_size", 256)),
        "infer_dataloader_workers": int(cfg.get("infer_dataloader_workers", 4)),
        "infer_psd_workers": int(cfg.get("infer_psd_workers", 2)),
        "infer_cache_max_mb": float(cfg.get("infer_cache_max_mb", 512)),
        "infer_prefetch_files": int(cfg.get("infer_prefetch_files", 4)),
        "infer_prefetch_batches": int(cfg.get("infer_prefetch_batches", 2)),
        "infer_prefetch_workers": int(cfg.get("infer_prefetch_workers", 2)),
        "infer_context_workers": int(cfg.get("infer_context_workers", 2)),
        "infer_context_cache_entries": int(cfg.get("infer_context_cache_entries", 64)),
        "infer_joint_queue_depth": int(cfg.get("infer_joint_queue_depth", 2)),
        "workflow_config_version": int(cfg.get("workflow_config_version", 0)),
        "workflow_resolved_path": str(cfg.get("workflow_resolved_path", "")),
    }
    profile_path = out_dir / "inference_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_report, f, ensure_ascii=False, indent=2)
    logger.info("Chapter4 全量识别性能报告：%s", json.dumps(profile_report, ensure_ascii=False))

    dist = {i: 0 for i in range(4)}
    for r in records:
        dist[int(r["prediction"])] = dist.get(int(r["prediction"]), 0) + 1
    logger.info(f"类别分布：{dist}")
    logger.info(f"推理完成：{len(records)} 条 → {stable_path}")
    return str(stable_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chapter4 2023 全量识别")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    run_inference(limit=args.limit, config_path=args.config)


if __name__ == "__main__":
    main()
