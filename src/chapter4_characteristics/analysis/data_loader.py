from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chapter4_characteristics.settings import (
    CLASS_DIRS,
    effective_round,
    get_enriched_round_dir,
    get_inference_path,
    get_others_index_path,
    load_config,
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


def load_class_samples(class_id: int, cfg: dict, round_idx: Optional[int] = None) -> List[dict]:
    rid = effective_round(cfg, round_idx)
    round_dir = get_enriched_round_dir(cfg, rid)
    class_dir = round_dir / CLASS_DIRS[class_id]
    samples: List[dict] = []
    for jf in _iter_class_json_files(class_dir, class_id):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("samples", []):
            samples.append(s)
    return samples


def load_inference_records(cfg: dict, round_idx: Optional[int] = None) -> List[dict]:
    rid = effective_round(cfg, round_idx)
    path = get_inference_path(cfg, rid)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", [])


def inference_by_idx(cfg: dict, round_idx: Optional[int] = None) -> Dict[int, dict]:
    return {int(r["sample_idx"]): r for r in load_inference_records(cfg, round_idx)}


def load_others_index(cfg: dict, round_idx: Optional[int] = None) -> dict:
    rid = effective_round(cfg, round_idx)
    path = get_others_index_path(cfg, rid)
    if not path.exists():
        return {"round_idx": rid, "count": 0, "samples": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_data_status(cfg: dict, round_idx: Optional[int] = None) -> dict:
    rid = effective_round(cfg, round_idx)
    counts = {}
    for cid, cdir in CLASS_DIRS.items():
        n = 0
        for jf in _iter_class_json_files(get_enriched_round_dir(cfg, rid) / cdir, cid):
            with open(jf, "r", encoding="utf-8") as f:
                n += len(json.load(f).get("samples", []))
        counts[cid] = n
    infer_path = get_inference_path(cfg, rid)
    return {
        "round_idx": rid,
        "inference_ready": infer_path.exists(),
        "enrich_ready": any(counts.values()),
        "class_counts": counts,
        "inference_path": str(infer_path) if infer_path.exists() else None,
        "enriched_dir": str(get_enriched_round_dir(cfg, rid)),
    }
