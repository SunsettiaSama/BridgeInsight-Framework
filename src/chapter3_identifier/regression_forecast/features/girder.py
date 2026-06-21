from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from src.chapter3_identifier.regression_forecast._bootstrap import resolve_path


def _timestamp_from_row(row: dict[str, Any], time_field: str) -> tuple[str, int, int, int]:
    value = row.get(time_field)
    dataset_tag = str(row.get("dataset_tag", ""))
    if isinstance(value, dict):
        return dataset_tag, int(value.get("month", 0)), int(value.get("day", 0)), int(value.get("hour", 0))
    if isinstance(value, list):
        return dataset_tag, int(value[0]), int(value[1]), int(value[2])
    return dataset_tag, int(row.get("month", 0)), int(row.get("day", 0)), int(row.get("hour", 0))


class GirderFeatureAdapter:
    def __init__(self, cfg: dict[str, Any]) -> None:
        source = cfg.get("girder_source") or {}
        self.enabled = bool(source.get("enabled", False))
        self.mode = str(source.get("mode", "disabled"))
        self.fields = [str(x) for x in source.get("feature_fields", [])]
        self.time_field = str(source.get("time_field", "timestamp"))
        self._index: dict[tuple[str, int, int, int], dict[str, float]] = {}
        if self.enabled and self.mode != "smoke":
            path_value = source.get("path")
            if not path_value:
                raise ValueError("girder_source.path 缺失，无法接入主梁数据")
            path = resolve_path(str(path_value))
            if not path.exists():
                raise FileNotFoundError(f"主梁数据不存在：{path}")
            self._index = self._load_index(path)

    def _load_index(self, path: Path) -> dict[tuple[str, int, int, int], dict[str, float]]:
        rows: list[dict[str, Any]]
        if path.suffix.lower() == ".jsonl":
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        rows.append(json.loads(text))
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload.get("records", payload if isinstance(payload, list) else [])
        elif path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                rows = [dict(row) for row in csv.DictReader(f)]
        else:
            raise ValueError(f"不支持的主梁数据格式：{path.suffix}")

        index: dict[tuple[str, int, int, int], dict[str, float]] = {}
        for row in rows:
            key = _timestamp_from_row(row, self.time_field)
            index[key] = {field: float(row[field]) for field in self.fields}
        return index

    def features_for(self, row: dict[str, Any], hour_index: int = 0) -> dict[str, float]:
        if not self.enabled:
            return {}
        if self.mode == "smoke":
            return {
                "girder_disp_mean": 0.3 * math.sin(hour_index / 6.0),
                "girder_disp_std": 0.05 + 0.02 * math.cos(hour_index / 5.0),
            }
        timestamp = row.get("timestamp", {})
        lookup = {
            "dataset_tag": row.get("dataset_tag", ""),
            "timestamp": timestamp,
            "month": row.get("month", 0),
            "day": row.get("day", 0),
            "hour": row.get("hour", 0),
        }
        key = _timestamp_from_row(lookup, "timestamp")
        if key not in self._index:
            raise KeyError(f"主梁数据缺少时间戳：{key}")
        return dict(self._index[key])

