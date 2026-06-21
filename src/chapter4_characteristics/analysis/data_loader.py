from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.chapter4_characteristics.settings import (
    CLASS_DIRS,
    get_enriched_dir,
    get_inference_path,
    get_others_index_path,
)


def get_nested(sample: dict, dotted: str) -> Any:
    cur: Any = sample
    for part in dotted.split("."):
        if cur is None:
            return None
        if part.isdigit():
            idx = int(part)
            if isinstance(cur, list) and idx < len(cur):
                cur = cur[idx]
            else:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _iter_class_json_files(class_dir: Path, class_id: int) -> List[Path]:
    if not class_dir.exists():
        flat = class_dir.parent / f"{CLASS_DIRS[class_id]}.json"
        if flat.exists():
            return [flat]
        return []
    return sorted(class_dir.glob("*.json"))


def load_class_samples(class_id: int, cfg: dict) -> List[dict]:
    class_dir = get_enriched_dir(cfg) / CLASS_DIRS[class_id]
    samples: List[dict] = []
    for jf in _iter_class_json_files(class_dir, class_id):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("samples", []):
            samples.append(s)
    return samples


def load_inference_records(cfg: dict) -> List[dict]:
    path = get_inference_path(cfg)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", [])


def inference_by_idx(cfg: dict) -> Dict[int, dict]:
    return {int(r["sample_idx"]): r for r in load_inference_records(cfg)}


def load_others_index(cfg: dict) -> dict:
    path = get_others_index_path(cfg)
    if not path.exists():
        return {"count": 0, "samples": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_data_status(cfg: dict) -> dict:
    counts = {}
    for cid, cdir in CLASS_DIRS.items():
        n = 0
        for jf in _iter_class_json_files(get_enriched_dir(cfg) / cdir, cid):
            with open(jf, "r", encoding="utf-8") as f:
                n += len(json.load(f).get("samples", []))
        counts[cid] = n
    infer_path = get_inference_path(cfg)
    return {
        "inference_ready": infer_path.exists(),
        "enrich_ready": any(counts.values()),
        "class_counts": counts,
        "inference_path": str(infer_path) if infer_path.exists() else None,
        "enriched_dir": str(get_enriched_dir(cfg)),
    }
