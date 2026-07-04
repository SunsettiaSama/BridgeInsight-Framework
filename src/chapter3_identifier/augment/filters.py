from __future__ import annotations

import re
from typing import Iterable

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key

_SENSOR_ID_RE = re.compile(r"(ST-VIC-[A-Z0-9]+-[0-9]+-[0-9]+)")


def _clean_ids(values: Iterable[object] | None) -> list[str]:
    if values is None:
        return []
    return sorted({str(value).strip() for value in values if str(value).strip()})


def sensor_ids_for_entry(entry: dict) -> set[str]:
    sensors: set[str] = set()
    for key in ("sensor_id", "inplane_sensor_id", "outplane_sensor_id"):
        value = entry.get(key)
        if value:
            sensors.add(str(value))
    for key in ("file_path", "inplane_file_path", "outplane_file_path"):
        match = _SENSOR_ID_RE.search(str(entry.get(key) or ""))
        if match:
            sensors.add(match.group(1))
    metadata = entry.get("metadata") or {}
    sensor_id = metadata.get("sensor_id")
    if sensor_id:
        sensors.add(str(sensor_id))
    return sensors


def filter_excluded_sensors(
    entries: list[dict],
    excluded_sensor_ids: Iterable[object] | None,
    enabled: bool = True,
) -> tuple[list[dict], dict]:
    excluded = _clean_ids(excluded_sensor_ids)
    if not excluded or not enabled:
        return entries, {
            "enable_sensor_exclusion": bool(enabled),
            "exclude_sensor_ids": excluded,
            "excluded_sensor_removed_total": 0,
            "excluded_sensor_hit_counts": {},
        }

    excluded_set = set(excluded)
    kept: list[dict] = []
    hit_counts = {sensor_id: 0 for sensor_id in excluded}
    removed_total = 0
    for entry in entries:
        matched = sensor_ids_for_entry(entry) & excluded_set
        if matched:
            removed_total += 1
            for sensor_id in matched:
                hit_counts[sensor_id] += 1
            continue
        kept.append(entry)

    return kept, {
        "enable_sensor_exclusion": True,
        "exclude_sensor_ids": excluded,
        "excluded_sensor_removed_total": int(removed_total),
        "excluded_sensor_hit_counts": {key: value for key, value in hit_counts.items() if value > 0},
    }


def normalize_inference_filter_config(cfg: dict) -> dict:
    exclude_annotated = bool(cfg.get("infer_exclude_annotated", False))
    sensor_ids = cfg.get("infer_exclude_sensor_ids", cfg.get("exclude_sensor_ids", []))
    if isinstance(sensor_ids, str):
        sensor_ids = [sensor_ids]
    return {
        "enable_sensor_exclusion": bool(
            cfg.get("infer_enable_sensor_exclusion", cfg.get("enable_sensor_exclusion", False))
        ),
        "exclude_sensor_ids": _clean_ids(sensor_ids),
        "exclude_gold_annotations": bool(cfg.get("infer_exclude_gold_annotations", exclude_annotated)),
        "exclude_manual_annotations": bool(cfg.get("infer_exclude_manual_annotations", exclude_annotated)),
    }


def _record_key(record: dict) -> tuple[str, int] | None:
    file_path = record.get("file_path") or record.get("inplane_file_path")
    if not file_path:
        return None
    return annotation_key(file_path, int(record.get("window_index", 0)))


def filter_annotated_records(
    records: list[dict],
    gold_keys: set[tuple[str, int]],
    manual_keys: set[tuple[str, int]],
    exclude_gold: bool = False,
    exclude_manual: bool = False,
) -> tuple[list[dict], dict]:
    if not exclude_gold and not exclude_manual:
        return records, {
            "exclude_gold_annotations": False,
            "exclude_manual_annotations": False,
            "excluded_annotation_removed_total": 0,
            "excluded_gold_removed_total": 0,
            "excluded_manual_removed_total": 0,
            "excluded_gold_manual_overlap_total": 0,
        }

    kept: list[dict] = []
    removed_total = 0
    gold_removed = 0
    manual_removed = 0
    overlap_removed = 0
    for record in records:
        key = _record_key(record)
        is_gold = key in gold_keys if key else bool(record.get("is_gold"))
        is_manual = key in manual_keys if key else bool(record.get("already_annotated")) and not is_gold
        remove = (exclude_gold and is_gold) or (exclude_manual and is_manual)
        if remove:
            removed_total += 1
            if is_gold:
                gold_removed += 1
            if is_manual:
                manual_removed += 1
            if is_gold and is_manual:
                overlap_removed += 1
            continue
        kept.append(record)

    return kept, {
        "exclude_gold_annotations": bool(exclude_gold),
        "exclude_manual_annotations": bool(exclude_manual),
        "excluded_annotation_removed_total": int(removed_total),
        "excluded_gold_removed_total": int(gold_removed),
        "excluded_manual_removed_total": int(manual_removed),
        "excluded_gold_manual_overlap_total": int(overlap_removed),
    }
