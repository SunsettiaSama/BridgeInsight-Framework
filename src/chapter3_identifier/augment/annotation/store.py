from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key, build_gold_index


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _resolve_direction_annotations(entry: dict) -> tuple[int | None, int | None]:
    inplane = _to_int_or_none(entry.get("inplane_annotation"))
    outplane = _to_int_or_none(entry.get("outplane_annotation"))
    if inplane is None and outplane is None:
        fallback = _to_int_or_none(entry.get("annotation", entry.get("class_id")))
        return fallback, fallback
    return inplane, outplane


def annotation_store_for_round(cfg: dict, round_idx: int) -> "AnnotationStore":
    from src.chapter3_identifier.augment.settings import (
        get_round_manual_history_path,
        get_round_manual_edits_path,
        get_round_merged_training_path,
    )

    return AnnotationStore(
        gold_path=cfg["gold_annotation_path"],
        manual_edits_path=str(get_round_manual_edits_path(cfg, round_idx)),
        manual_history_path=str(get_round_manual_history_path(cfg, round_idx)),
        merged_output_path=str(get_round_merged_training_path(cfg, round_idx)),
    )


def load_cumulative_manual_edits(cfg: dict, through_round: int) -> List[dict]:
    from src.chapter3_identifier.augment.settings import get_round_manual_edits_path

    merged: Dict[Tuple[str, int], dict] = {}
    for round_idx in range(1, through_round + 1):
        path = get_round_manual_edits_path(cfg, round_idx)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for entry in rows:
            fp = entry.get("file_path")
            if fp is None:
                continue
            key = annotation_key(fp, entry.get("window_index", 0))
            merged[key] = dict(entry)
    return list(merged.values())


def lookup_manual_annotation(
    manual_entries: List[dict],
    file_path: str,
    window_index: int,
    direction: str = "inplane",
) -> Optional[int]:
    key = annotation_key(file_path, window_index)
    for entry in manual_entries:
        if annotation_key(entry["file_path"], entry.get("window_index", 0)) == key:
            inplane_ann, outplane_ann = _resolve_direction_annotations(entry)
            if direction == "outplane":
                return outplane_ann
            return inplane_ann
    return None


class AnnotationStore:
    def __init__(
        self,
        gold_path: str,
        manual_edits_path: str,
        merged_output_path: str,
        manual_history_path: str | None = None,
    ):
        self.gold_path = resolve_path(gold_path)
        self.manual_edits_path = resolve_path(manual_edits_path)
        if manual_history_path:
            self.manual_history_path = resolve_path(manual_history_path)
        else:
            self.manual_history_path = self.manual_edits_path.with_name("manual_edits_history.jsonl")
        self.merged_output_path = resolve_path(merged_output_path)
        self._gold_keys: Optional[set] = None
        self._annotated_keys: Optional[set] = None

    def _invalidate_lookup_cache(self) -> None:
        self._gold_keys = None
        self._annotated_keys = None

    def _refresh_lookup_keys(self) -> None:
        gold_index = build_gold_index(self.load_gold())
        self._gold_keys = set(gold_index.keys())
        self._annotated_keys = set(self._gold_keys)
        for entry in self.load_manual_edits():
            self._annotated_keys.add(
                annotation_key(entry["file_path"], entry.get("window_index", 0))
            )

    def build_lookup_keys(self) -> Tuple[set, set]:
        if self._gold_keys is None or self._annotated_keys is None:
            self._refresh_lookup_keys()
        return self._gold_keys, self._annotated_keys

    def load_gold(self) -> List[dict]:
        if not self.gold_path.exists():
            return []
        with open(self.gold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = []
        for item in data:
            row = dict(item)
            ann = row.get("annotation", row.get("class_id", 0))
            row["annotation"] = int(ann)
            row["is_gold"] = True
            entries.append(row)
        return entries

    def load_manual_edits(self) -> List[dict]:
        if not self.manual_edits_path.exists():
            return []
        with open(self.manual_edits_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_manual_edits(self, entries: List[dict]) -> None:
        self._write_json(self.manual_edits_path, entries)
        self._invalidate_lookup_cache()

    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def _append_manual_history(self, event: dict) -> None:
        self.manual_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manual_history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False))
            f.write("\n")

    def merge(
        self,
        gold_only: bool = False,
        prior_manual: List[dict] | None = None,
        shuffle: bool = True,
        shuffle_seed: int = 42,
    ) -> List[dict]:
        merged: Dict[Tuple[str, int], dict] = {}
        for entry in self.load_gold():
            key = annotation_key(entry["file_path"], entry.get("window_index", 0))
            merged[key] = dict(entry)

        if not gold_only:
            manual_sources: List[dict] = []
            if prior_manual:
                manual_sources.extend(prior_manual)
            manual_sources.extend(self.load_manual_edits())
            for entry in manual_sources:
                inplane_ann, outplane_ann = _resolve_direction_annotations(entry)
                wi = int(entry.get("window_index", 0))

                inplane_row = dict(entry)
                inplane_row["annotation"] = inplane_ann
                in_fp = inplane_row.get("file_path")
                if in_fp and inplane_ann is not None:
                    inplane_row["direction"] = "inplane"
                    key = annotation_key(in_fp, wi)
                    merged[key] = inplane_row

                out_fp = entry.get("outplane_file_path")
                if out_fp and outplane_ann is not None:
                    outplane_row = dict(entry)
                    outplane_row["file_path"] = out_fp
                    outplane_row["annotation"] = outplane_ann
                    outplane_row["direction"] = "outplane"
                    key = annotation_key(out_fp, wi)
                    merged[key] = outplane_row

        result = list(merged.values())
        if shuffle and len(result) > 1:
            rng = random.Random(shuffle_seed)
            rng.shuffle(result)
        self._write_json(self.merged_output_path, result)
        self._invalidate_lookup_cache()
        return result

    def upsert_manual(
        self,
        file_path: str,
        window_index: int,
        inplane_annotation: int,
        outplane_annotation: int,
        outplane_file_path: Optional[str] = None,
        sample_id: Optional[str] = None,
        is_gold: bool = False,
        round_idx: Optional[int] = None,
    ) -> dict:
        entries = self.load_manual_edits()
        key = annotation_key(file_path, window_index)

        row = {
            "sample_id": sample_id or f"manual_{window_index}",
            "annotation": int(inplane_annotation),
            "inplane_annotation": int(inplane_annotation),
            "outplane_annotation": int(outplane_annotation),
            "file_path": file_path,
            "window_index": int(window_index),
            "data_type": "vic",
            "is_gold": bool(is_gold),
            "is_manual": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if outplane_file_path:
            row["outplane_file_path"] = outplane_file_path
        if round_idx is not None:
            row["round_idx"] = int(round_idx)

        replaced = False
        previous_row = None
        for i, existing in enumerate(entries):
            if annotation_key(existing["file_path"], existing.get("window_index", 0)) == key:
                previous_row = dict(existing)
                entries[i] = row
                replaced = True
                break
        if not replaced:
            entries.append(row)
        self.save_manual_edits(entries)

        before_in = (
            int(previous_row.get("inplane_annotation", previous_row.get("annotation", 0)))
            if previous_row is not None
            else None
        )
        before_out = (
            int(previous_row.get("outplane_annotation", previous_row.get("annotation", 0)))
            if previous_row is not None
            else None
        )
        after_in = int(row["inplane_annotation"])
        after_out = int(row["outplane_annotation"])
        history_event = {
            "event_time": row["updated_at"],
            "round_idx": int(round_idx) if round_idx is not None else None,
            "file_path": file_path,
            "window_index": int(window_index),
            "sample_id": row.get("sample_id"),
            "is_gold": bool(is_gold),
            "change_type": "update" if replaced else "insert",
            "before_inplane_annotation": before_in,
            "before_outplane_annotation": before_out,
            "after_inplane_annotation": after_in,
            "after_outplane_annotation": after_out,
            "changed_inplane": before_in != after_in,
            "changed_outplane": before_out != after_out,
            "changed_any": (before_in != after_in) or (before_out != after_out),
        }
        self._append_manual_history(history_event)
        return row

    def is_gold_member(self, file_path: str, window_index: int) -> bool:
        gold_keys, _ = self.build_lookup_keys()
        return annotation_key(file_path, window_index) in gold_keys

    def is_annotated(self, file_path: str, window_index: int) -> bool:
        _, annotated_keys = self.build_lookup_keys()
        return annotation_key(file_path, window_index) in annotated_keys
