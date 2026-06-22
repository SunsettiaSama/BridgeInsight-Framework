from __future__ import annotations

import argparse
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.annotation.store import (
    AnnotationStore,
    load_cumulative_manual_edits,
)
from src.chapter3_identifier.augment.infer.dataset_loader import ensure_inference_ready, load_staycable_dataset
from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.chapter3_identifier.augment.settings import (
    get_round_checkpoint_path,
    get_round_dir,
    get_round_inference_path,
    load_config,
)

ensure_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _record_to_json(
    rec,
    sample_idx: int,
    pred: int,
    proba,
    in_pred: int,
    out_pred: int,
    in_proba,
    out_proba,
    gold_keys: Set[Tuple[str, int]],
    annotated_keys: Set[Tuple[str, int]],
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
    lookup_key = annotation_key(in_fp, wi) if in_fp else None
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
        "already_annotated": lookup_key in annotated_keys if lookup_key else False,
        "is_gold": lookup_key in gold_keys if lookup_key else False,
    }


def _build_chunk_rows(
    sample_indices: Sequence[int],
    samples,
    merged: Dict[int, Tuple[int, np.ndarray]],
    inplane_p: Dict[int, np.ndarray],
    outplane_p: Dict[int, np.ndarray],
    gold_keys: Set[Tuple[str, int]],
    annotated_keys: Set[Tuple[str, int]],
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
                gold_keys,
                annotated_keys,
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
    gold_keys: Set[Tuple[str, int]],
    annotated_keys: Set[Tuple[str, int]],
    chunk_size: int,
    workers: int,
    projection_mode: str,
) -> List[dict]:
    total = len(merged_indices)
    chunks = [
        merged_indices[i : i + chunk_size]
        for i in range(0, total, chunk_size)
    ]
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
                gold_keys,
                annotated_keys,
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
        f.write(f'  "round_idx": {json.dumps(header["round_idx"])},\n')
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


def run_inference(round_idx: int = 1, config_path: str | None = None) -> str:
    cfg = load_config(config_path)
    ensure_inference_ready(cfg)

    round_dir = get_round_dir(cfg, round_idx)
    ckpt_path = get_round_checkpoint_path(cfg, round_idx)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}，请先完成 round {round_idx} 训练")

    chunk_size = int(cfg.get("infer_record_chunk_size", 4096))
    record_workers = int(cfg.get("infer_record_workers", 8))

    dataset = load_staycable_dataset(cfg["inference_dataset_config"])
    identifier = DualStreamIdentifier.from_checkpoint(
        str(ckpt_path),
        model_config_path=cfg.get("dual_stream_config"),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
        wind_config=cfg,
    )
    runner = DualStreamInferenceRunner(
        identifier,
        batch_size=int(cfg.get("infer_batch_size", 256)),
        num_workers=int(cfg.get("infer_dataloader_workers", 4)),
        psd_workers=int(cfg.get("infer_psd_workers", 2)),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
    )

    store = AnnotationStore(
        gold_path=cfg["gold_annotation_path"],
        manual_edits_path=str(get_round_dir(cfg, round_idx) / "manual_edits.json"),
        merged_output_path=str(get_round_dir(cfg, round_idx) / "merged_training.json"),
    )

    inplane_p, outplane_p = runner.run_with_proba(dataset)
    logger.info("面内/面外推理完成，正在合并结果并写入 inference.json …")
    projection_mode = str(cfg.get("prediction_projection_mode", "direction"))
    projection_direction = str(cfg.get("prediction_projection_direction", "outplane"))
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
    logger.info(f"概率合并完成：{len(merged_indices)} 条有效预测")

    gold_keys, _ = store.build_lookup_keys()
    cumulative_manual = load_cumulative_manual_edits(cfg, round_idx)
    annotated_keys = set(gold_keys)
    for entry in cumulative_manual:
        annotated_keys.add(annotation_key(entry["file_path"], entry.get("window_index", 0)))
    logger.info(
        f"标注索引就绪：gold={len(gold_keys)} manual(<=round{round_idx})={len(cumulative_manual)} "
        f"annotated={len(annotated_keys)}"
    )

    records = _assemble_records_parallel(
        merged_indices=merged_indices,
        samples=dataset._samples,
        merged=merged,
        inplane_p=inplane_p,
        outplane_p=outplane_p,
        gold_keys=gold_keys,
        annotated_keys=annotated_keys,
        chunk_size=chunk_size,
        workers=record_workers,
        projection_mode=projection_label,
    )

    logger.info("记录排序 …")
    records.sort(
        key=lambda r: (
            0 if r["prediction"] in (1, 2, 3) else 1,
            -r["uncertainty"],
            int(r.get("sample_idx", 0)),
        )
    )
    logger.info("记录排序完成")

    round_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    round_tag = f"round_{int(round_idx):02d}"
    archive_path = round_dir / f"inference_{round_tag}_{ts}.json"
    stable_path = get_round_inference_path(cfg, round_idx)
    model_snapshot_path = round_dir / f"model_{round_tag}_{ts}.pth"

    header = {
        "round_idx": round_idx,
        "generated_at": ts,
        "checkpoint": ckpt_path.name,
        "projection_mode": projection_label,
    }
    _stream_write_inference_json(archive_path, header, records, chunk_size)
    shutil.copy2(archive_path, stable_path)
    logger.info(f"复制归档 → {stable_path}")
    shutil.copy2(ckpt_path, model_snapshot_path)
    logger.info(f"模型快照 → {model_snapshot_path}")

    manifest = {
        "round_idx": round_idx,
        "generated_at": ts,
        "checkpoint": str(ckpt_path),
        "model_snapshot": model_snapshot_path.name,
        "inference": stable_path.name,
        "inference_archive": archive_path.name,
        "projection_mode": projection_label,
        "record_count": len(records),
    }
    with open(round_dir / "round_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Round {round_idx} 推理完成：{len(records)} 条 → {stable_path} "
        f"(归档 {archive_path.name}, 模型快照 {model_snapshot_path.name})"
    )
    return str(stable_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Augment 全量识别")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    run_inference(round_idx=args.round, config_path=args.config)


if __name__ == "__main__":
    main()
