from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.chapter4_characteristics.feature_analysis._compactor import (
    ensure_class_dir_compacted,
    list_batch_json_files,
    list_canonical_json_files,
)
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


def _iter_class_json_files(class_dir: Path, class_id: int, cfg: dict) -> List[Path]:
    if not class_dir.exists():
        flat = class_dir.parent / f"{CLASS_DIRS[class_id]}.json"
        if flat.exists():
            return [flat]
        return []

    excluded = _excluded_sensor_ids(cfg)
    if list_batch_json_files(class_dir):
        ensure_class_dir_compacted(
            class_dir,
            cfg=cfg,
            excluded_sensor_ids=excluded,
        )

    canonical = list_canonical_json_files(class_dir, excluded_sensor_ids=excluded)
    if canonical:
        return canonical

    flat = class_dir.parent / f"{CLASS_DIRS[class_id]}.json"
    if flat.exists():
        return [flat]

    if list_batch_json_files(class_dir):
        raise FileNotFoundError(
            f"类别目录仅有 batch 文件，未生成 canonical JSON：{class_dir}\n"
            "请先运行：python -m src.chapter4_characteristics enrich --compact-only"
        )
    return []


def _excluded_sensor_ids(cfg: dict) -> set[str]:
    ids = cfg.get("infer_exclude_sensor_ids") or cfg.get("exclude_sensor_ids") or []
    return {str(sensor_id) for sensor_id in ids}


def _is_excluded_sample(sample: dict, excluded: set[str]) -> bool:
    return (
        sample.get("inplane_sensor_id") in excluded
        or sample.get("outplane_sensor_id") in excluded
        or sample.get("sensor_id") in excluded
    )


def load_class_samples(class_id: int, cfg: dict) -> List[dict]:
    class_dir = get_enriched_dir(cfg) / CLASS_DIRS[class_id]
    excluded = _excluded_sensor_ids(cfg)
    samples: List[dict] = []
    for jf in _iter_class_json_files(class_dir, class_id, cfg):
        if jf.stem in excluded:
            continue
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("samples", []):
            if not _is_excluded_sample(s, excluded):
                samples.append(s)
    return samples


def load_inference_records(cfg: dict) -> List[dict]:
    path = get_inference_path(cfg)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    excluded = _excluded_sensor_ids(cfg)
    return [
        r for r in data.get("records", [])
        if not _is_excluded_sample(r, excluded)
    ]


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
    excluded = _excluded_sensor_ids(cfg)
    for cid, cdir in CLASS_DIRS.items():
        n = 0
        for jf in _iter_class_json_files(get_enriched_dir(cfg) / cdir, cid, cfg):
            if jf.stem in excluded:
                continue
            with open(jf, "r", encoding="utf-8") as f:
                samples = json.load(f).get("samples", [])
            n += sum(1 for sample in samples if not _is_excluded_sample(sample, excluded))
        counts[cid] = n
    infer_path = get_inference_path(cfg)
    return {
        "inference_ready": infer_path.exists(),
        "enrich_ready": any(counts.values()),
        "class_counts": counts,
        "inference_path": str(infer_path) if infer_path.exists() else None,
        "enriched_dir": str(get_enriched_dir(cfg)),
    }
