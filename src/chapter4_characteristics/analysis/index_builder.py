from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from src.chapter4_characteristics.analysis.data_loader import get_nested, inference_by_idx, load_class_samples
from src.chapter4_characteristics.settings import CLASS_DIRS, get_enriched_round_dir, get_others_index_path


def _scalar_features(sample: dict) -> dict:
    ws0 = (sample.get("wind_stats") or [{}])[0] if sample.get("wind_stats") else {}
    psd_in = sample.get("psd_inplane") or {}
    freqs = psd_in.get("frequencies") or []
    return {
        "rms_in": get_nested(sample, "time_stats_inplane.rms"),
        "rms_out": get_nested(sample, "time_stats_outplane.rms"),
        "dominant_freq_in": float(freqs[0]) if freqs else None,
        "mean_wind_speed": ws0.get("mean_wind_speed"),
    }


def _sort_key(row: dict) -> tuple:
    proba = row.get("proba") or [0, 0, 0, 0]
    viv_p = float(proba[1]) if len(proba) > 1 else 0.0
    rwiv_p = float(proba[2]) if len(proba) > 2 else 0.0
    mismatch = 0 if row.get("inplane_prediction") == row.get("outplane_prediction") else 1
    return (
        -mismatch,
        -float(row.get("uncertainty", 0)),
        float(max(proba)) if proba else 0.0,
        abs(viv_p - rwiv_p),
    )


def build_others_index(cfg: dict, round_idx: int) -> dict:
    samples = load_class_samples(3, cfg, round_idx)
    infer_map = inference_by_idx(cfg, round_idx)
    class_dir = get_enriched_round_dir(cfg, round_idx) / CLASS_DIRS[3]

    rows: List[dict] = []
    for s in samples:
        idx = int(s.get("sample_idx", -1))
        inf = infer_map.get(idx, {})
        proba = inf.get("proba") or [0, 0, 0, 0]
        ordered = sorted(enumerate(proba), key=lambda x: -x[1])
        second_class = int(ordered[1][0]) if len(ordered) > 1 else None
        scalars = _scalar_features(s)
        rows.append({
            "sample_idx": idx,
            "timestamp": s.get("timestamp"),
            "inplane_sensor_id": s.get("inplane_sensor_id"),
            "uncertainty": inf.get("uncertainty"),
            "proba": proba,
            "prediction": inf.get("prediction", 3),
            "second_class": second_class,
            "inplane_prediction": inf.get("inplane_prediction"),
            "outplane_prediction": inf.get("outplane_prediction"),
            "enriched_sensor_file": str(class_dir / f"{s.get('inplane_sensor_id', 'unknown')}.json"),
            **scalars,
        })

    rows.sort(key=_sort_key)

    payload = {
        "round_idx": round_idx,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rows),
        "samples": rows,
    }
    path = get_others_index_path(cfg, round_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
