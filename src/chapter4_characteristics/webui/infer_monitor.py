from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional


def build_infer_monitor_payload(
    cfg: dict,
    round_idx: int,
    job_state: dict,
    log_tail: str,
) -> dict:
    from src.chapter4_characteristics.settings import get_inference_path

    dist = {}
    record_count = 0
    infer_path = get_inference_path(cfg, round_idx)
    if infer_path.exists():
        with open(infer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        record_count = len(records)
        c = Counter(int(r.get("prediction", -1)) for r in records)
        dist = {str(k): v for k, v in sorted(c.items())}

    return {
        "round_idx": round_idx,
        "job": job_state,
        "log_tail": log_tail,
        "record_count": record_count,
        "class_distribution": dist,
        "inference_ready": infer_path.exists(),
    }
