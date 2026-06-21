from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MetadataRow:
    dataset_tag: str
    year: int
    month: int
    day: int
    hour: int
    sensor_id: str
    file_path: str
    raw_metadata_json: Optional[Dict[str, Any]] = None


_LEGACY_CORE_KEYS = frozenset({"month", "day", "hour", "sensor_id", "file_path"})
_LEGACY_OPTIONAL_KEYS = frozenset(
    {
        "minute",
        "second",
        "data_type",
        "path",
        "vib_sensor_id",
        "raw_time",
        "extreme_time_ranges",
        "extreme_window_count",
        "out_of_range_windows",
        "actual_length",
        "missing_rate",
        "rms_per_window",
        "extreme_rms_indices",
        "dominant_freq_per_window",
        "extreme_freq_indices",
    }
)


def _coerce_int(value: Any, field: str) -> int:
    if value is None:
        raise ValueError(f"缺少字段 {field}")
    return int(value)


def parse_raw_metadata_json(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        if not raw.strip():
            return None
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("raw_metadata_json 必须是 JSON 对象")
        return parsed
    raise ValueError(f"无法解析 raw_metadata_json: {type(raw)}")


def to_legacy_metadata_dict(row: MetadataRow) -> dict:
    """将 MySQL 行还原为与旧 JSON 元数据兼容的词典。"""
    legacy: dict = {
        "month": _coerce_int(row.month, "month"),
        "day": _coerce_int(row.day, "day"),
        "hour": _coerce_int(row.hour, "hour"),
        "sensor_id": str(row.sensor_id),
        "file_path": str(row.file_path),
    }
    raw = parse_raw_metadata_json(row.raw_metadata_json)
    if raw:
        for key, value in raw.items():
            if key in _LEGACY_CORE_KEYS:
                continue
            if key in _LEGACY_OPTIONAL_KEYS or key not in legacy:
                legacy[key] = value
    return legacy


def metadata_row_from_db_row(row: dict) -> MetadataRow:
    return MetadataRow(
        dataset_tag=str(row["dataset_tag"]),
        year=_coerce_int(row["year"], "year"),
        month=_coerce_int(row["month"], "month"),
        day=_coerce_int(row["day"], "day"),
        hour=_coerce_int(row["hour"], "hour"),
        sensor_id=str(row["sensor_id"]),
        file_path=str(row["file_path"]),
        raw_metadata_json=parse_raw_metadata_json(row.get("raw_metadata_json")),
    )
