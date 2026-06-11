from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from src.chapter4_characteristics._bootstrap import ensure_paths
from src.chapter4_characteristics.infer.preflight import run_preflight
from src.chapter4_characteristics.infer.result_adapter import records_to_enriched_json
from src.chapter4_characteristics.settings import (
    get_augment_checkpoint,
    get_inference_dir,
    get_inference_path,
    get_predictions_enriched_path,
    load_config,
)

ensure_paths()

from src.chapter3_identifier.augment.infer.dataset_loader import load_staycable_dataset
from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.chapter3_identifier.augment.settings import load_dual_stream_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _record_to_json(rec, sample_idx: int, pred: int, proba, in_pred: int, out_pred: int) -> dict:
    in_meta = rec.inplane_meta or {}
    out_meta = rec.outplane_meta or {}
    m, d, h = rec.timestamp_key
    in_fp = in_meta.get("file_path", "")
    wi = rec.window_idx
    proba_list = [float(x) for x in proba]
    uncertainty = float(1.0 - max(proba_list))
    return {
        "sample_idx": sample_idx,
        "prediction": int(pred),
        "proba": proba_list,
        "uncertainty": uncertainty,
        "inplane_prediction": int(in_pred),
        "outplane_prediction": int(out_pred),
        "inplane_file_path": in_fp,
        "outplane_file_path": out_meta.get("file_path"),
        "window_index": wi,
        "inplane_sensor_id": in_meta.get("sensor_id"),
        "outplane_sensor_id": out_meta.get("sensor_id"),
        "timestamp": [m, d, h],
    }


def _apply_limit(dataset, limit: int | None):
    if limit is None or limit <= 0:
        return dataset
    dataset._samples = dataset._samples[: int(limit)]
    logger.info(f"冒烟模式：限制样本数 {len(dataset._samples)}")
    return dataset


def run_inference(
    round_idx: int = 1,
    limit: int | None = None,
    config_path: str | None = None,
) -> str:
    cfg = load_config(config_path)
    if limit is None and int(cfg.get("dev_limit_samples", 0)) > 0:
        limit = int(cfg["dev_limit_samples"])

    run_preflight(round_idx=round_idx, config_path=config_path)

    out_dir = get_inference_dir(cfg, round_idx)
    ckpt_path = get_augment_checkpoint(cfg, round_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_staycable_dataset(cfg["inference_dataset_config"])
    dataset = _apply_limit(dataset, limit)

    identifier = DualStreamIdentifier.from_checkpoint(
        str(ckpt_path),
        model_config_path=cfg.get("dual_stream_config"),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
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

    inplane_p, outplane_p = runner.run_with_proba(dataset)
    merged = runner.merge_proba_predictions(inplane_p, outplane_p)

    records = []
    for sample_idx, rec in enumerate(dataset._samples):
        if sample_idx not in merged:
            continue
        pred, proba = merged[sample_idx]
        in_pred = int(inplane_p[sample_idx].argmax()) if sample_idx in inplane_p else int(pred)
        out_pred = int(outplane_p[sample_idx].argmax()) if sample_idx in outplane_p else int(pred)
        records.append(_record_to_json(rec, sample_idx, pred, proba, in_pred, out_pred))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stable_path = get_inference_path(cfg, round_idx)
    enriched_path = get_predictions_enriched_path(cfg, round_idx)

    inference_payload = {
        "round_idx": round_idx,
        "generated_at": ts,
        "checkpoint": str(ckpt_path),
        "record_count": len(records),
        "records": records,
    }
    with open(stable_path, "w", encoding="utf-8") as f:
        json.dump(inference_payload, f, ensure_ascii=False, indent=2)

    enriched_payload = records_to_enriched_json(
        records,
        dataset,
        round_idx=round_idx,
        checkpoint_path=str(ckpt_path),
        dataset_config=str(cfg["inference_dataset_config"]),
    )
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched_payload, f, ensure_ascii=False, indent=2)

    manifest = {
        "round_idx": round_idx,
        "generated_at": ts,
        "checkpoint": str(ckpt_path),
        "inference": stable_path.name,
        "predictions_enriched": enriched_path.name,
        "record_count": len(records),
        "limit": limit,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    dist = {i: 0 for i in range(4)}
    for r in records:
        dist[int(r["prediction"])] = dist.get(int(r["prediction"]), 0) + 1
    logger.info(f"类别分布：{dist}")
    logger.info(f"Round {round_idx} 推理完成：{len(records)} 条 → {stable_path}")
    return str(stable_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chapter4 2023 全量识别")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    run_inference(round_idx=args.round, limit=args.limit, config_path=args.config)


if __name__ == "__main__":
    main()
