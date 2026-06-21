from __future__ import annotations

from collections import defaultdict
from typing import Any


def chronological_split_indices(records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[int]]:
    policy = cfg.get("split_policy") or {}
    train_ratio = float(policy.get("train", 0.7))
    val_ratio = float(policy.get("val", 0.15))
    horizons = [int(x) for x in cfg.get("horizons_hours", [])]
    max_horizon = max(horizons) if horizons else 0

    by_cable: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(records):
        by_cable[str(row.get("cable_key", ""))].append((idx, row))

    splits = {"train": [], "val": [], "test": []}
    for _, rows in by_cable.items():
        ordered = sorted(rows, key=lambda item: int(item[1].get("hour_index", 0)))
        n = len(ordered)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_boundary = train_end
        val_boundary = val_end
        for local_idx, (global_idx, _) in enumerate(ordered):
            if local_idx < train_end:
                if local_idx + max_horizon < train_boundary:
                    splits["train"].append(global_idx)
            elif local_idx < val_end:
                if local_idx + max_horizon < val_boundary:
                    splits["val"].append(global_idx)
            else:
                splits["test"].append(global_idx)

    return {name: sorted(values) for name, values in splits.items()}


def leakage_report(records: list[dict[str, Any]], splits: dict[str, list[int]], cfg: dict[str, Any]) -> dict[str, Any]:
    horizons = [int(x) for x in cfg.get("horizons_hours", [])]
    max_horizon = max(horizons) if horizons else 0
    by_idx = {idx: row for idx, row in enumerate(records)}
    train_max = max((int(by_idx[i].get("hour_index", 0)) + max_horizon for i in splits.get("train", [])), default=-1)
    val_min = min((int(by_idx[i].get("hour_index", 0)) for i in splits.get("val", [])), default=10**12)
    val_max = max((int(by_idx[i].get("hour_index", 0)) + max_horizon for i in splits.get("val", [])), default=-1)
    test_min = min((int(by_idx[i].get("hour_index", 0)) for i in splits.get("test", [])), default=10**12)
    return {
        "train_target_max_hour": train_max,
        "val_input_min_hour": val_min if val_min < 10**12 else None,
        "val_target_max_hour": val_max,
        "test_input_min_hour": test_min if test_min < 10**12 else None,
        "train_val_leak": bool(splits.get("val")) and train_max >= val_min,
        "val_test_leak": bool(splits.get("test")) and val_max >= test_min,
    }

