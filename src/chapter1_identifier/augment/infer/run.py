from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.chapter1_identifier.augment._bootstrap import ensure_paths, resolve_path
from src.chapter1_identifier.augment.annotation.store import AnnotationStore
from src.chapter1_identifier.augment.infer.dataset_loader import ensure_inference_ready, load_staycable_dataset
from src.chapter1_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter1_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.chapter1_identifier.augment.settings import load_config

ensure_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _record_to_json(rec, sample_idx: int, pred: int, proba, store: AnnotationStore) -> dict:
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
        "inplane_file_path": in_fp,
        "outplane_file_path": out_meta.get("file_path"),
        "window_index": wi,
        "inplane_sensor_id": in_meta.get("sensor_id"),
        "outplane_sensor_id": out_meta.get("sensor_id"),
        "timestamp": [m, d, h],
        "already_annotated": store.is_annotated(in_fp, wi) if in_fp else False,
        "is_gold": store.is_gold_member(in_fp, wi) if in_fp else False,
    }


def run_inference(round_idx: int = 1, config_path: str | None = None) -> str:
    cfg = load_config(config_path)
    ensure_inference_ready(cfg)

    ckpt_path = resolve_path(cfg["training_output_dir"]) / f"round_{round_idx:02d}" / "best_checkpoint.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}，请先完成 round {round_idx} 训练")

    dataset = load_staycable_dataset(cfg["inference_dataset_config"])
    identifier = DualStreamIdentifier.from_checkpoint(
        str(ckpt_path),
        model_config_path=cfg.get("dual_stream_config"),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
    )
    runner = DualStreamInferenceRunner(
        identifier,
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
    )

    store = AnnotationStore(
        gold_path=cfg["gold_annotation_path"],
        manual_edits_path=cfg["manual_edits_path"],
        merged_output_path=cfg["merged_training_path"],
    )

    inplane_p, outplane_p = runner.run_with_proba(dataset)
    merged = runner.merge_proba_predictions(inplane_p, outplane_p)

    records = []
    for sample_idx, rec in enumerate(dataset._samples):
        if sample_idx not in merged:
            continue
        pred, proba = merged[sample_idx]
        in_pred = int(inplane_p.get(sample_idx, proba).argmax()) if sample_idx in inplane_p else 0
        out_pred = int(outplane_p.get(sample_idx, proba).argmax()) if sample_idx in outplane_p else 0
        row = _record_to_json(rec, sample_idx, pred, proba, store)
        row["inplane_prediction"] = in_pred
        row["outplane_prediction"] = out_pred
        records.append(row)

    records.sort(
        key=lambda r: (
            r["already_annotated"],
            0 if r["prediction"] in (1, 2, 3) else 1,
            -r["uncertainty"],
        )
    )

    out_dir = resolve_path(cfg["inference_output_dir"]) / f"round_{round_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"full_dataset_{ts}.json"
    latest_path = resolve_path(cfg["inference_results_path"])

    payload = {"round_idx": round_idx, "generated_at": ts, "records": records}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"推理完成：{len(records)} 条 → {out_path}")
    return str(out_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Augment 全量识别")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    run_inference(round_idx=args.round, config_path=args.config)


if __name__ == "__main__":
    main()
