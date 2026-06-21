from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.annotation.store import (
    annotation_store_for_round,
    load_cumulative_manual_edits,
)
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.settings import get_round_merged_training_path


def _distribution(counter: Counter, label_names: List[str]) -> List[dict]:
    total = int(sum(counter.values()))
    denom = total if total > 0 else 1
    rows: List[dict] = []
    for class_id, name in enumerate(label_names):
        count = int(counter.get(class_id, 0))
        rows.append(
            {
                "class_id": class_id,
                "name": name,
                "count": count,
                "ratio": count / denom,
            }
        )
    return rows


def _manual_direction_counter(entries: List[dict], direction: str) -> Counter:
    counter: Counter = Counter()
    for entry in entries:
        if direction == "outplane":
            ann = entry.get("outplane_annotation", entry.get("annotation"))
        else:
            ann = entry.get("inplane_annotation", entry.get("annotation"))
        if ann is None:
            continue
        counter[int(ann)] += 1
    return counter


def build_dataset_summary(cfg: dict, round_idx: int) -> dict:
    label_names = get_label_names(cfg)
    merged_path = get_round_merged_training_path(cfg, round_idx)
    training_rows: List[dict] = []
    if merged_path.exists():
        payload = json.loads(merged_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            training_rows = payload

    training_counter: Counter = Counter()
    gold_counter = 0
    manual_counter = 0
    for row in training_rows:
        ann = row.get("annotation", row.get("class_id"))
        if ann is None:
            continue
        training_counter[int(ann)] += 1
        if row.get("is_gold"):
            gold_counter += 1
        if row.get("is_manual"):
            manual_counter += 1

    round_store = annotation_store_for_round(cfg, round_idx)
    round_manual = round_store.load_manual_edits()
    cumulative_manual = load_cumulative_manual_edits(cfg, round_idx)
    prior_manual = load_cumulative_manual_edits(cfg, round_idx - 1) if round_idx > 1 else []
    prior_keys = {
        annotation_key(entry["file_path"], entry.get("window_index", 0))
        for entry in prior_manual
        if entry.get("file_path")
    }

    new_windows: List[dict] = []
    for entry in round_manual:
        fp = entry.get("file_path")
        if not fp:
            continue
        key = annotation_key(fp, entry.get("window_index", 0))
        if key not in prior_keys:
            new_windows.append(entry)

    return {
        "round_idx": int(round_idx),
        "label_names": label_names,
        "merged_training_path": str(merged_path),
        "training": {
            "total": len(training_rows),
            "gold_count": gold_counter,
            "manual_stream_count": manual_counter,
            "distribution": _distribution(training_counter, label_names),
        },
        "round_manual": {
            "window_count": len(round_manual),
            "inplane_distribution": _distribution(
                _manual_direction_counter(round_manual, "inplane"),
                label_names,
            ),
            "outplane_distribution": _distribution(
                _manual_direction_counter(round_manual, "outplane"),
                label_names,
            ),
        },
        "round_new": {
            "window_count": len(new_windows),
            "inplane_distribution": _distribution(
                _manual_direction_counter(new_windows, "inplane"),
                label_names,
            ),
            "outplane_distribution": _distribution(
                _manual_direction_counter(new_windows, "outplane"),
                label_names,
            ),
        },
        "cumulative_manual": {
            "window_count": len(cumulative_manual),
            "inplane_distribution": _distribution(
                _manual_direction_counter(cumulative_manual, "inplane"),
                label_names,
            ),
            "outplane_distribution": _distribution(
                _manual_direction_counter(cumulative_manual, "outplane"),
                label_names,
            ),
        },
    }
