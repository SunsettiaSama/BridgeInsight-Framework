from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
from src.figure_paintings.figs_for_thesis.Chapter4._any_side_prediction import (
    PREDICTION_OVERRIDE_NOTE,
    PREDICTION_OVERRIDE_RULE,
    merge_any_side_prediction,
)

INFERENCE_DIR = ROOT / "results" / "chapter4_characteristics" / "inference"
INFERENCE_PATH = INFERENCE_DIR / "inference.json"
MANIFEST_PATH = INFERENCE_DIR / "manifest.json"
SOURCE_ENRICHED_PATH = INFERENCE_DIR / "predictions_enriched.json"
OUTPUT_PATH = INFERENCE_DIR / "predictions_enriched_any_side_special_exclude_c34_201_202_301.json"
EXCLUDED_SENSOR_IDS = data_config.EXCLUDED_SENSOR_IDS


def _is_excluded_record(record: dict) -> bool:
    return (
        record.get("inplane_sensor_id") in EXCLUDED_SENSOR_IDS
        or record.get("outplane_sensor_id") in EXCLUDED_SENSOR_IDS
    )


def _iter_inference_records(path: Path):
    in_records = False
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if not in_records:
                if stripped.startswith('"records"'):
                    in_records = True
                continue
            if stripped in {"]", "],"}:
                return
            if not stripped.startswith("{"):
                continue
            payload = stripped.rstrip(",")
            yield json.loads(payload)


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_sample_metadata(record: dict) -> dict:
    return {
        "timestamp": list(record.get("timestamp", [])),
        "window_idx": record.get("window_index"),
        "inplane_sensor_id": record.get("inplane_sensor_id"),
        "outplane_sensor_id": record.get("outplane_sensor_id"),
        "inplane_file_path": record.get("inplane_file_path"),
        "outplane_file_path": record.get("outplane_file_path"),
    }


def main() -> None:
    if not INFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"推理结果不存在：{INFERENCE_PATH}\n"
            "请先运行：python -m src.chapter4_characteristics infer"
        )

    manifest = _load_manifest()
    predictions: dict[str, int] = {}
    sample_metadata: dict[str, dict] = {}
    override_count = 0
    class_counts = Counter()
    processed = 0
    removed = 0

    print(f"streaming records from: {INFERENCE_PATH.name}", flush=True)
    for record in _iter_inference_records(INFERENCE_PATH):
        processed += 1
        if _is_excluded_record(record):
            removed += 1
            continue

        sample_idx = int(record["sample_idx"])
        key = str(sample_idx)
        in_pred = int(record["inplane_prediction"])
        out_pred = int(record["outplane_prediction"])
        original_pred = int(record["prediction"])
        merged_pred = merge_any_side_prediction(in_pred, out_pred)

        if merged_pred != original_pred:
            override_count += 1

        predictions[key] = merged_pred
        sample_metadata[key] = _build_sample_metadata(record)
        class_counts[merged_pred] += 1

        if processed % 300000 == 0:
            print(f"processed={processed} kept={len(predictions)}", flush=True)

    metadata = {
        "source": "chapter4_characteristics",
        "source_result": INFERENCE_PATH.name,
        "source_enriched": SOURCE_ENRICHED_PATH.name,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": manifest.get("checkpoint"),
        "num_samples_raw": processed,
        "num_samples": len(predictions),
        "excluded_sensor_ids": sorted(EXCLUDED_SENSOR_IDS),
        "excluded_sample_count": removed,
        "prediction_override_rule": PREDICTION_OVERRIDE_RULE,
        "prediction_override_note": PREDICTION_OVERRIDE_NOTE,
        "original_projection_mode": manifest.get("projection_mode"),
        "override_count": override_count,
        "class_distribution": {str(k): class_counts[k] for k in sorted(class_counts)},
    }

    payload = {
        "metadata": metadata,
        "predictions": predictions,
        "sample_metadata": sample_metadata,
        "by_file": {},
    }

    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    print(
        f"writing {OUTPUT_PATH.name}: kept={len(predictions)} removed={removed} "
        f"override={override_count}",
        flush=True,
    )
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    tmp_path.replace(OUTPUT_PATH)
    print("done", flush=True)


if __name__ == "__main__":
    main()
