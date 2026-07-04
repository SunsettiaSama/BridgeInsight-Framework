from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Hashable, List, Tuple

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.annotation.sample_key import pair_key_from_list

PATH_KEY_SCHEMA_VERSION = 1
PAIR_KEY_SCHEMA_VERSION = 2


def _entry_pair_key(entry: dict) -> tuple | None:
    pair_key = entry.get("pair_key")
    if pair_key is None:
        return None
    return pair_key_from_list(pair_key)


def _entry_key(entry: dict):
    pair_key = _entry_pair_key(entry)
    if pair_key is not None:
        return pair_key
    file_path = (
        entry.get("file_path")
        or entry.get("inplane_file_path")
        or entry.get("outplane_file_path")
    )
    if not file_path:
        raise ValueError("split entry 缺少 file_path/inplane_file_path/outplane_file_path/pair_key")
    return annotation_key(file_path, entry.get("window_index", 0))


def _uses_pair_key(entries: List[dict]) -> bool:
    return any(_entry_pair_key(entry) is not None for entry in entries)


def _entry_label(entry: dict) -> Hashable:
    if entry.get("inplane_annotation") is not None and entry.get("outplane_annotation") is not None:
        return (int(entry["inplane_annotation"]), int(entry["outplane_annotation"]))
    if entry.get("annotation") is not None:
        return int(entry["annotation"])
    if entry.get("class_id") is not None:
        return int(entry["class_id"])
    if entry.get("inplane_annotation") is not None:
        return int(entry["inplane_annotation"])
    if entry.get("outplane_annotation") is not None:
        return int(entry["outplane_annotation"])
    return 0


def _label_strategy(entries: List[dict]) -> str:
    if any(e.get("inplane_annotation") is not None and e.get("outplane_annotation") is not None for e in entries):
        return "joint_inplane_outplane"
    return "single_label"


def _to_key_pairs(keys: list[tuple[str, int]]) -> list[list[object]]:
    return [[k[0], int(k[1])] for k in keys]


def _to_pair_key_lists(keys: list[tuple]) -> list[list[object]]:
    return [list(k) for k in keys]


def _stratified_split_indices(
    entries: List[dict],
    train_val_ratio: float,
    random_seed: int,
) -> Tuple[List[int], List[int]]:
    if not entries:
        return [], []

    val_ratio = max(0.0, min(1.0, 1.0 - float(train_val_ratio)))
    rng = random.Random(random_seed)
    by_label: dict[Hashable, list[int]] = {}
    for idx, entry in enumerate(entries):
        by_label.setdefault(_entry_label(entry), []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        val_count = int(round(len(label_indices) * val_ratio))
        val_count = max(0, min(len(label_indices), val_count))
        val_idx.extend(label_indices[:val_count])
        train_idx.extend(label_indices[val_count:])

    if not val_idx and len(train_idx) > 1:
        val_idx.append(train_idx.pop())
    if not train_idx and len(val_idx) > 1:
        train_idx.append(val_idx.pop())

    return sorted(train_idx), sorted(val_idx)


def load_saved_split_key_sets(split_path: str) -> tuple[set, set]:
    path = resolve_path(split_path)
    if not path.exists():
        return set(), set()
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    if "train_pair_keys" in saved or "val_pair_keys" in saved:
        train_keys = {tuple(row) for row in saved.get("train_pair_keys", [])}
        val_keys = {tuple(row) for row in saved.get("val_pair_keys", [])}
        return train_keys, val_keys
    train_keys = {annotation_key(k[0], k[1]) for k in saved.get("train_keys", [])}
    val_keys = {annotation_key(k[0], k[1]) for k in saved.get("val_keys", [])}
    return train_keys, val_keys


def load_or_create_split(
    entries: List[dict],
    split_path: str,
    train_val_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    path = resolve_path(split_path)
    keys = [_entry_key(e) for e in entries]

    train_idx, val_idx = _stratified_split_indices(
        entries,
        train_val_ratio=train_val_ratio,
        random_seed=random_seed,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    train_keys = [keys[i] for i in train_idx]
    val_keys = [keys[i] for i in val_idx]
    payload = {
        "random_seed": random_seed,
        "strategy": "stratified_resample",
        "label_strategy": _label_strategy(entries),
    }
    if _uses_pair_key(entries):
        payload["schema_version"] = PAIR_KEY_SCHEMA_VERSION
        payload["train_pair_keys"] = _to_pair_key_lists(train_keys)
        payload["val_pair_keys"] = _to_pair_key_lists(val_keys)
    else:
        payload["schema_version"] = PATH_KEY_SCHEMA_VERSION
        payload["train_keys"] = _to_key_pairs(train_keys)
        payload["val_keys"] = _to_key_pairs(val_keys)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return train_idx, val_idx
