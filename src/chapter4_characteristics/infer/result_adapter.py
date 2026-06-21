from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def build_sample_metadata_mapping(dataset) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for idx, rec in enumerate(dataset._samples):
        mapping[idx] = {
            "cable_pair": list(rec.cable_pair),
            "timestamp": list(rec.timestamp_key),
            "window_idx": rec.window_idx,
            "inplane_sensor_id": rec.inplane_meta.get("sensor_id"),
            "outplane_sensor_id": rec.outplane_meta.get("sensor_id"),
            "inplane_file_path": rec.inplane_meta.get("file_path"),
            "outplane_file_path": rec.outplane_meta.get("file_path"),
            "missing_rate_in": rec.inplane_meta.get("missing_rate"),
            "missing_rate_out": rec.outplane_meta.get("missing_rate"),
            "has_wind": rec.wind_meta is not None,
        }
    return mapping


def records_to_enriched_json(
    records: List[dict],
    dataset,
    checkpoint_path: str,
    dataset_config: str,
) -> dict:
    meta_map = build_sample_metadata_mapping(dataset)
    predictions: Dict[str, int] = {}
    sample_metadata: Dict[str, dict] = {}

    for row in records:
        idx = int(row["sample_idx"])
        predictions[str(idx)] = int(row["prediction"])
        if idx in meta_map:
            sample_metadata[str(idx)] = meta_map[idx]

    return {
        "metadata": {
            "checkpoint": checkpoint_path,
            "dataset_config": dataset_config,
            "enriched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(predictions),
            "source": "chapter4_characteristics",
        },
        "predictions": predictions,
        "sample_metadata": sample_metadata,
        "by_file": {},
    }
