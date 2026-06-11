from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key


def _entry_key(entry: dict):
    file_path = (
        entry.get("file_path")
        or entry.get("inplane_file_path")
        or entry.get("outplane_file_path")
    )
    if not file_path:
        raise ValueError("split entry 缺少 file_path/inplane_file_path/outplane_file_path")
    return annotation_key(file_path, entry.get("window_index", 0))


def load_or_create_split(
    entries: List[dict],
    split_path: str,
    train_val_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    path = resolve_path(split_path)
    keys = [_entry_key(e) for e in entries]

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        train_keys = {
            annotation_key(k[0], k[1]) for k in saved.get("train_keys", [])
        }
        val_keys = {
            annotation_key(k[0], k[1]) for k in saved.get("val_keys", [])
        }
        train_idx, val_idx = [], []
        for i, key in enumerate(keys):
            if key in val_keys:
                val_idx.append(i)
            elif key in train_keys:
                train_idx.append(i)
            else:
                train_idx.append(i)
        return train_idx, val_idx

    indices = list(range(len(entries)))
    rng = random.Random(random_seed)
    rng.shuffle(indices)
    split_at = max(1, int(len(indices) * train_val_ratio))
    train_idx = sorted(indices[:split_at])
    val_idx = sorted(indices[split_at:]) or train_idx[-1:]

    path.parent.mkdir(parents=True, exist_ok=True)
    train_keys = [keys[i] for i in train_idx]
    val_keys = [keys[i] for i in val_idx]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"train_keys": train_keys, "val_keys": val_keys, "random_seed": random_seed},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return train_idx, val_idx
