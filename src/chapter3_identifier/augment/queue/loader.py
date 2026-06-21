from __future__ import annotations

import json
from typing import Dict, List, Optional

from src.chapter3_identifier.augment._bootstrap import resolve_path


def load_inference_records(path: str) -> List[dict]:
    p = resolve_path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    return []


def filter_queue(
    records: List[dict],
    sensor_id: Optional[str] = None,
    only_unannotated: bool = False,
    only_abnormal: bool = False,
    page: int = 0,
    page_size: int = 50,
) -> Dict:
    filtered = filter_records(records, sensor_id, only_unannotated, only_abnormal)
    total = len(filtered)
    start = page * page_size
    end = start + page_size
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": filtered[start:end],
    }


def filter_records(
    records: List[dict],
    sensor_id: Optional[str] = None,
    only_unannotated: bool = False,
    only_abnormal: bool = False,
) -> List[dict]:
    def _is_abnormal(record: dict) -> bool:
        pred = int(record.get("prediction", 0))
        in_pred = int(record.get("inplane_prediction", pred))
        out_pred = int(record.get("outplane_prediction", pred))
        return pred in (1, 2, 3) or in_pred in (1, 2, 3) or out_pred in (1, 2, 3)

    filtered = records
    if sensor_id:
        filtered = [
            r for r in filtered
            if r.get("inplane_sensor_id") == sensor_id or r.get("outplane_sensor_id") == sensor_id
        ]
    if only_unannotated:
        filtered = [r for r in filtered if not r.get("already_annotated")]
    if only_abnormal:
        filtered = [r for r in filtered if _is_abnormal(r)]
    return filtered
