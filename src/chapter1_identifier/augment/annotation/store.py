from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.chapter1_identifier.augment._bootstrap import resolve_path
from src.chapter1_identifier.augment.annotation.gold_index import annotation_key, build_gold_index


class AnnotationStore:
    def __init__(
        self,
        gold_path: str,
        manual_edits_path: str,
        merged_output_path: str,
    ):
        self.gold_path = resolve_path(gold_path)
        self.manual_edits_path = resolve_path(manual_edits_path)
        self.merged_output_path = resolve_path(merged_output_path)

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
        self.manual_edits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manual_edits_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def merge(self, gold_only: bool = False) -> List[dict]:
        merged: Dict[Tuple[str, int], dict] = {}
        for entry in self.load_gold():
            key = annotation_key(entry["file_path"], entry.get("window_index", 0))
            merged[key] = dict(entry)

        if not gold_only:
            for entry in self.load_manual_edits():
                key = annotation_key(entry["file_path"], entry.get("window_index", 0))
                merged[key] = dict(entry)

        result = list(merged.values())
        self.merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.merged_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def upsert_manual(
        self,
        file_path: str,
        window_index: int,
        annotation: int,
        outplane_file_path: Optional[str] = None,
        sample_id: Optional[str] = None,
        is_gold: bool = False,
        round_idx: Optional[int] = None,
    ) -> dict:
        entries = self.load_manual_edits()
        key = annotation_key(file_path, window_index)
        gold_index = build_gold_index(self.load_gold())
        gold_entry = gold_index.get(key)

        row = {
            "sample_id": sample_id or (gold_entry or {}).get("sample_id") or f"manual_{window_index}",
            "annotation": int(annotation),
            "file_path": file_path,
            "window_index": int(window_index),
            "data_type": "vic",
            "is_gold": bool(is_gold or gold_entry is not None),
            "is_manual": True,
        }
        if outplane_file_path:
            row["outplane_file_path"] = outplane_file_path
        if round_idx is not None:
            row["round_idx"] = int(round_idx)

        replaced = False
        for i, existing in enumerate(entries):
            if annotation_key(existing["file_path"], existing.get("window_index", 0)) == key:
                entries[i] = row
                replaced = True
                break
        if not replaced:
            entries.append(row)
        self.save_manual_edits(entries)
        return row

    def is_gold_member(self, file_path: str, window_index: int) -> bool:
        gold_index = build_gold_index(self.load_gold())
        return annotation_key(file_path, window_index) in gold_index

    def is_annotated(self, file_path: str, window_index: int) -> bool:
        key = annotation_key(file_path, window_index)
        merged = self.merge(gold_only=False)
        for entry in merged:
            if annotation_key(entry["file_path"], entry.get("window_index", 0)) == key:
                return True
        return False
