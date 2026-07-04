from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.annotation.sample_key import (
    PAIR_KEY_FIELD_COUNT,
    derive_outplane_path_from_inplane,
    is_canonical_pair,
    pair_key_from_paths,
    pair_key_to_list,
    pair_type_from_paths,
    sensor_id_from_path,
    sensor_suffix_from_path,
)

LABEL_SOURCE_PRIORITY = {
    "manual": 4,
    "gold": 3,
    "gold_fill": 2,
    "prediction": 1,
}


def _valid_label(label: object, num_classes: int) -> int | None:
    if label is None:
        return None
    value = int(label)
    if 0 <= value < int(num_classes):
        return value
    return None


def _resolve_direction_annotations(entry: dict) -> tuple[int | None, int | None]:
    inplane = _valid_label(entry.get("inplane_annotation"), 4)
    outplane = _valid_label(entry.get("outplane_annotation"), 4)
    if inplane is None and outplane is None:
        fallback = _valid_label(entry.get("annotation", entry.get("class_id")), 4)
        return fallback, fallback
    return inplane, outplane


def _label_source(entry: dict, direction: str, label: int) -> str:
    if bool(entry.get("is_manual")):
        return "manual"
    if bool(entry.get("is_gold")):
        return "gold"
    if direction == "inplane" and _valid_label(entry.get("inplane_annotation"), 4) == label:
        return "gold"
    if direction == "outplane" and _valid_label(entry.get("outplane_annotation"), 4) == label:
        return "gold"
    return "gold"


def _direct_ann_by_key(entries: List[dict], num_classes: int) -> Dict[Tuple[str, int], tuple[int, str]]:
    direct: Dict[Tuple[str, int], tuple[int, str]] = {}
    for entry in entries:
        fp = entry.get("file_path")
        if not fp:
            continue
        wi = int(entry.get("window_index", 0))
        key = annotation_key(str(fp), wi)
        ann = _valid_label(entry.get("annotation"), num_classes)
        if ann is None:
            continue
        if bool(entry.get("is_manual")):
            direct[key] = (ann, "manual")
        elif key not in direct:
            direct[key] = (ann, "gold")
    return direct


def _resolve_pair_labels(
    in_entry: dict,
    out_entry: dict | None,
    in_pair: tuple[int, str] | None,
    out_pair: tuple[int, str] | None,
    enable_gold_fill: bool,
) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
    if in_pair is None and out_pair is not None and out_pair[1] == "gold" and enable_gold_fill:
        in_pair = (out_pair[0], "gold_fill")
    if out_pair is None and in_pair is not None and in_pair[1] == "gold" and enable_gold_fill:
        out_pair = (in_pair[0], "gold_fill")
    return in_pair, out_pair


def _fill_from_prediction(
    hint: dict | None,
    in_pair: tuple[int, str] | None,
    out_pair: tuple[int, str] | None,
    num_classes: int,
) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
    if hint is None:
        return in_pair, out_pair
    if in_pair is None:
        in_label = _valid_label(hint.get("inplane_prediction"), num_classes)
        if in_label is not None:
            in_pair = (in_label, "prediction")
    if out_pair is None:
        out_label = _valid_label(hint.get("outplane_prediction"), num_classes)
        if out_label is not None:
            out_pair = (out_label, "prediction")
    return in_pair, out_pair


def _pair_record(
    in_fp: str,
    out_fp: str,
    wi: int,
    in_pair: tuple[int, str],
    out_pair: tuple[int, str],
    in_entry: dict,
    out_entry: dict | None,
) -> dict:
    pair_key = pair_key_to_list(pair_key_from_paths(in_fp, out_fp, wi))
    in_meta = (in_entry.get("metadata") or {}) if in_entry else {}
    out_meta = (out_entry.get("metadata") or {}) if out_entry else {}
    timestamp = in_entry.get("timestamp")
    if timestamp is None and in_meta.get("month") is not None:
        timestamp = [
            int(in_meta["month"]),
            int(in_meta["day"]),
            int(in_meta["hour"]),
        ]
    return {
        "schema_version": 2,
        "pair_key": pair_key,
        "pair_type": pair_type_from_paths(in_fp, out_fp),
        "inplane_file_path": str(in_fp),
        "outplane_file_path": str(out_fp),
        "window_index": int(wi),
        "timestamp": timestamp,
        "inplane_sensor_id": in_entry.get("inplane_sensor_id")
        or in_entry.get("sensor_id")
        or in_meta.get("sensor_id")
        or sensor_id_from_path(in_fp),
        "outplane_sensor_id": (out_entry or {}).get("outplane_sensor_id")
        or (out_entry or {}).get("sensor_id")
        or out_meta.get("sensor_id")
        or sensor_id_from_path(out_fp),
        "inplane_annotation": int(in_pair[0]),
        "outplane_annotation": int(out_pair[0]),
        "inplane_label_source": in_pair[1],
        "outplane_label_source": out_pair[1],
        "is_manual": bool(in_entry.get("is_manual")) or bool((out_entry or {}).get("is_manual")),
        "is_gold": bool(in_entry.get("is_gold")) or bool((out_entry or {}).get("is_gold")),
    }


def _should_replace(existing: dict, incoming: dict) -> bool:
    existing_score = max(
        LABEL_SOURCE_PRIORITY.get(str(existing.get("inplane_label_source", "gold")), 0),
        LABEL_SOURCE_PRIORITY.get(str(existing.get("outplane_label_source", "gold")), 0),
    )
    incoming_score = max(
        LABEL_SOURCE_PRIORITY.get(str(incoming.get("inplane_label_source", "gold")), 0),
        LABEL_SOURCE_PRIORITY.get(str(incoming.get("outplane_label_source", "gold")), 0),
    )
    if incoming_score != existing_score:
        return incoming_score > existing_score
    if bool(incoming.get("is_manual")) and not bool(existing.get("is_manual")):
        return True
    return False


def build_pair_key_entries(
    entries: List[dict],
    pair_hints: Dict[Tuple[str, int], dict] | None = None,
    num_classes: int = 4,
    enable_prediction_fill: bool = False,
    enable_gold_fill: bool = True,
) -> tuple[List[dict], dict]:
    merged_by_key: Dict[Tuple[str, int], dict] = {}
    outplane_path_by_in_key: Dict[Tuple[str, int], str] = {}
    direct_ann_by_key = _direct_ann_by_key(entries, num_classes=num_classes)

    for entry in entries:
        fp = entry.get("file_path")
        if not fp:
            continue
        wi = int(entry.get("window_index", 0))
        key = annotation_key(str(fp), wi)
        merged_by_key[key] = dict(entry)
        if sensor_suffix_from_path(str(fp)) == "01":
            out_fp = entry.get("outplane_file_path")
            if out_fp:
                outplane_path_by_in_key[key] = str(out_fp)

    hints = pair_hints or {}
    stats = {
        "input_total": int(len(entries)),
        "same_file_pair": 0,
        "outplane_anchor": 0,
        "noncanonical_pair": 0,
        "missing_counterpart": 0,
        "missing_label": 0,
        "label_conflict": 0,
        "duplicate_pair_key": 0,
        "pair_total": 0,
        "canonical_01_02": 0,
    }
    paired_by_key: Dict[tuple, dict] = {}

    def consider_pair(
        in_fp: str,
        out_fp: str,
        wi: int,
        in_entry: dict,
        out_entry: dict | None,
        hint: dict | None,
    ) -> None:
        if str(in_fp) == str(out_fp):
            stats["same_file_pair"] += 1
            return
        if sensor_suffix_from_path(str(in_fp)) != "01":
            stats["outplane_anchor"] += 1
            return
        if not is_canonical_pair(str(in_fp), str(out_fp)):
            stats["noncanonical_pair"] += 1
            return

        in_key = annotation_key(str(in_fp), wi)
        out_key = annotation_key(str(out_fp), wi)
        in_pair = direct_ann_by_key.get(in_key)
        out_pair = direct_ann_by_key.get(out_key)
        in_pair, out_pair = _resolve_pair_labels(in_entry, out_entry, in_pair, out_pair, enable_gold_fill)
        if enable_prediction_fill:
            in_pair, out_pair = _fill_from_prediction(hint, in_pair, out_pair, num_classes)
        if in_pair is None or out_pair is None:
            stats["missing_label"] += 1
            return

        record = _pair_record(
            in_fp=str(in_fp),
            out_fp=str(out_fp),
            wi=wi,
            in_pair=in_pair,
            out_pair=out_pair,
            in_entry=in_entry,
            out_entry=out_entry,
        )
        pair_key = tuple(record["pair_key"])
        existing = paired_by_key.get(pair_key)
        if existing is None:
            paired_by_key[pair_key] = record
            return

        same_labels = (
            int(existing["inplane_annotation"]) == int(record["inplane_annotation"])
            and int(existing["outplane_annotation"]) == int(record["outplane_annotation"])
        )
        if not same_labels:
            stats["label_conflict"] += 1
        if _should_replace(existing, record):
            paired_by_key[pair_key] = record
        stats["duplicate_pair_key"] += 1

    if hints and enable_prediction_fill:
        for in_key, hint in hints.items():
            in_fp = hint.get("inplane_file_path")
            out_fp = hint.get("outplane_file_path")
            wi = int(hint.get("window_index", 0))
            if not in_fp or not out_fp:
                stats["missing_counterpart"] += 1
                continue
            in_entry = merged_by_key.get(in_key, {})
            out_entry = merged_by_key.get(annotation_key(str(out_fp), wi))
            consider_pair(str(in_fp), str(out_fp), wi, in_entry, out_entry, hint)

    for in_key, in_entry in merged_by_key.items():
        in_fp = in_entry.get("file_path")
        wi = int(in_entry.get("window_index", 0))
        if not in_fp:
            continue
        if sensor_suffix_from_path(str(in_fp)) != "01":
            stats["outplane_anchor"] += 1
            continue
        out_fp = in_entry.get("outplane_file_path") or outplane_path_by_in_key.get(in_key)
        if not out_fp:
            out_fp = derive_outplane_path_from_inplane(str(in_fp))
        if not out_fp:
            stats["missing_counterpart"] += 1
            continue
        out_entry = merged_by_key.get(annotation_key(str(out_fp), wi))
        consider_pair(str(in_fp), str(out_fp), wi, in_entry, out_entry, None)

    result = list(paired_by_key.values())
    stats["pair_total"] = int(len(result))
    stats["canonical_01_02"] = int(
        sum(1 for row in result if row.get("pair_type") == "01_02")
    )
    return result, stats


def validate_pair_key_entries(entries: List[dict]) -> dict:
    seen: set[tuple] = set()
    report = {
        "valid": True,
        "pair_total": int(len(entries)),
        "same_file_pair": 0,
        "noncanonical_pair": 0,
        "duplicate_pair_key": 0,
        "invalid_pair_key": 0,
        "pair_type_counts": {},
        "errors": [],
    }
    for entry in entries:
        in_fp = entry.get("inplane_file_path")
        out_fp = entry.get("outplane_file_path")
        if not in_fp or not out_fp:
            report["valid"] = False
            report["errors"].append("missing inplane/outplane file path")
            continue
        if str(in_fp) == str(out_fp):
            report["valid"] = False
            report["same_file_pair"] += 1
            report["errors"].append(f"same_file_pair: {in_fp}")
        pair_type = entry.get("pair_type") or pair_type_from_paths(str(in_fp), str(out_fp))
        report["pair_type_counts"][pair_type] = int(report["pair_type_counts"].get(pair_type, 0)) + 1
        if pair_type != "01_02" or not is_canonical_pair(str(in_fp), str(out_fp)):
            report["valid"] = False
            report["noncanonical_pair"] += 1
            report["errors"].append(f"noncanonical_pair: in={in_fp} out={out_fp}")
        pair_key = entry.get("pair_key")
        if not isinstance(pair_key, list) or len(pair_key) != PAIR_KEY_FIELD_COUNT:
            report["valid"] = False
            report["invalid_pair_key"] += 1
            report["errors"].append(f"invalid pair_key: {pair_key}")
            continue
        key_tuple = tuple(pair_key)
        if key_tuple in seen:
            report["valid"] = False
            report["duplicate_pair_key"] += 1
            report["errors"].append(f"duplicate pair_key: {pair_key}")
        seen.add(key_tuple)
    report["errors"] = report["errors"][:50]
    return report


def _load_json_list(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    for key in ("records", "entries", "annotations"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError(f"{path} 缺少 list/records/entries/annotations")


def _load_pair_hints(inference_path: Path) -> Dict[Tuple[str, int], dict]:
    if not inference_path.exists():
        return {}
    with open(inference_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    hints: Dict[Tuple[str, int], dict] = {}
    for row in payload.get("records", []):
        in_fp = row.get("inplane_file_path")
        out_fp = row.get("outplane_file_path")
        if not in_fp or not out_fp:
            continue
        wi = int(row.get("window_index", 0))
        key = annotation_key(str(in_fp), wi)
        if key in hints:
            continue
        hints[key] = {
            "inplane_file_path": in_fp,
            "outplane_file_path": out_fp,
            "window_index": wi,
            "inplane_prediction": row.get("inplane_prediction", row.get("prediction")),
            "outplane_prediction": row.get("outplane_prediction", row.get("prediction")),
        }
    return hints


def migrate_round_dataset(
    merged_training_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    inference_path: str | Path | None = None,
    num_classes: int = 4,
    enable_prediction_fill: bool = False,
    enable_gold_fill: bool = True,
) -> dict:
    merged_path = resolve_path(str(merged_training_path))
    out_path = resolve_path(str(output_path))
    report_out = resolve_path(str(report_path))
    entries = _load_json_list(merged_path)
    hints = _load_pair_hints(resolve_path(str(inference_path))) if inference_path else {}
    pair_entries, build_stats = build_pair_key_entries(
        entries,
        pair_hints=hints,
        num_classes=num_classes,
        enable_prediction_fill=enable_prediction_fill,
        enable_gold_fill=enable_gold_fill,
    )
    validation = validate_pair_key_entries(pair_entries)
    report = {
        "schema_version": 2,
        "source_merged_training_path": str(merged_path),
        "output_path": str(out_path),
        "build_stats": build_stats,
        "validation": validation,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pair_entries, f, ensure_ascii=False, indent=2)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not validation["valid"]:
        raise ValueError(f"pair_key 数据集校验失败：{report_out}")
    return report
