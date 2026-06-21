from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.chapter4_characteristics._bootstrap import ensure_paths, resolve_path
from src.chapter4_characteristics.infer.preflight import run_preflight
from src.chapter4_characteristics.infer.result_adapter import records_to_enriched_json
from src.chapter4_characteristics.settings import (
    get_identifier_checkpoint,
    get_inference_dir,
    get_inference_path,
    get_predictions_enriched_path,
    load_config,
)

ensure_paths()

from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.chapter3_identifier.augment.settings import load_dual_stream_model_config
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
    limit: int | None = None,
    config_path: str | None = None,
) -> str:
    cfg = load_config(config_path)
    if limit is None and int(cfg.get("dev_limit_samples", 0)) > 0:
        limit = int(cfg["dev_limit_samples"])

    run_preflight(config_path=config_path)

    out_dir = get_inference_dir(cfg)
    ckpt_path = get_identifier_checkpoint(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_inference_dataset(cfg["inference_dataset_config"])
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
    stable_path = get_inference_path(cfg)
    enriched_path = get_predictions_enriched_path(cfg)

    inference_payload = {
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
        checkpoint_path=str(ckpt_path),
        dataset_config=str(cfg["inference_dataset_config"]),
    )
    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched_payload, f, ensure_ascii=False, indent=2)

    manifest = {
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
