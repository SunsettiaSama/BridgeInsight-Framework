from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from src.chapter3_identifier.augment.annotation.store import AnnotationStore
from src.chapter3_identifier.augment.queue.enrich import AnnotationLookupCache, enrich_record
from src.chapter3_identifier.augment.queue.inference_cache import InferenceSnapshotCache
from src.chapter3_identifier.augment.queue.loader import filter_records

SLIM_QUEUE_FIELDS = (
    "sample_idx",
    "window_index",
    "prediction",
    "uncertainty",
    "inplane_sensor_id",
    "is_gold",
    "gold_label",
    "gold_label_name",
    "manual_inplane_annotation",
    "manual_inplane_annotation_name",
    "manual_outplane_annotation",
    "manual_outplane_annotation_name",
    "manual_annotation",
    "manual_annotation_name",
    "suggested_inplane_label",
    "suggested_inplane_label_name",
    "suggested_outplane_label",
    "suggested_outplane_label_name",
    "suggested_label",
    "suggested_label_name",
    "suggested_source",
    "label_origin",
    "label_origin_name",
    "card_kind",
    "card_kind_name",
    "prediction_matches_gold",
    "already_annotated",
)


def slim_queue_row(row: dict) -> dict:
    return {key: row[key] for key in SLIM_QUEUE_FIELDS if key in row}


class AbnormalQueueCache:
    def __init__(
        self,
        inference_cache: InferenceSnapshotCache,
        annotation_lookup: AnnotationLookupCache,
        gold_store: AnnotationStore,
        cfg: dict,
    ) -> None:
        self._inference_cache = inference_cache
        self._annotation_lookup = annotation_lookup
        self._gold_store = gold_store
        self._cfg = cfg
        self._lock = threading.Lock()
        self._key: Optional[Tuple[str, float, int, int]] = None
        self._abnormal_items: List[dict] = []
        self._class_items: Dict[Tuple[int, ...], List[dict]] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._key = None
            self._abnormal_items = []
            self._class_items = {}

    @staticmethod
    def _normalize_classes(predicted_classes: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
        if not predicted_classes:
            return (1, 2, 3)
        classes = sorted({int(c) for c in predicted_classes if 0 <= int(c) <= 3})
        return tuple(classes) if classes else (1, 2, 3)

    def _build_abnormal_items(self, records: List[dict], round_idx: int) -> List[dict]:
        gold_index = self._annotation_lookup.gold_index(self._gold_store.load_gold())
        manual_index = self._annotation_lookup.manual_index(round_idx)
        filtered = filter_records(records, only_abnormal=True)
        items = [
            slim_queue_row(enrich_record(record, gold_index, manual_index, self._cfg))
            for record in filtered
        ]
        items.sort(
            key=lambda row: (
                -float(row.get("uncertainty", 0.0)),
                int(row.get("sample_idx", 0)),
            )
        )
        return items

    def _build_items_for_classes(
        self,
        records: List[dict],
        round_idx: int,
        classes: Tuple[int, ...],
    ) -> List[dict]:
        class_set = set(classes)
        gold_index = self._annotation_lookup.gold_index(self._gold_store.load_gold())
        manual_index = self._annotation_lookup.manual_index(round_idx)
        selected = [record for record in records if int(record.get("prediction", 0)) in class_set]
        items = [
            slim_queue_row(enrich_record(record, gold_index, manual_index, self._cfg))
            for record in selected
        ]
        items.sort(
            key=lambda row: (
                -float(row.get("uncertainty", 0.0)),
                int(row.get("sample_idx", 0)),
            )
        )
        return items

    def get_items(
        self,
        inference_path: str,
        round_idx: int,
        predicted_classes: Optional[Tuple[int, ...]] = None,
    ) -> List[dict]:
        records = self._inference_cache.get_records(inference_path)
        if not records:
            return []

        class_key = self._normalize_classes(predicted_classes)
        mtime = self._inference_cache.get_mtime(inference_path)
        manual_epoch = self._annotation_lookup.manual_epoch(round_idx)
        key = (inference_path, mtime, round_idx, manual_epoch)
        with self._lock:
            if self._key == key and class_key in self._class_items:
                return list(self._class_items[class_key])
            key_changed = self._key != key

        if key_changed:
            abnormal_items = self._build_abnormal_items(records, round_idx)
            class_items: Dict[Tuple[int, ...], List[dict]] = {(1, 2, 3): abnormal_items}
            for cls in (1, 2, 3):
                class_items[(cls,)] = [row for row in abnormal_items if int(row.get("prediction", 0)) == cls]
            with self._lock:
                self._key = key
                self._abnormal_items = abnormal_items
                self._class_items = class_items

        with self._lock:
            if class_key in self._class_items:
                return list(self._class_items[class_key])

        if set(class_key).issubset({1, 2, 3}):
            with self._lock:
                items = [
                    row for row in self._abnormal_items
                    if int(row.get("prediction", 0)) in set(class_key)
                ]
                self._class_items[class_key] = items
                return list(items)

        items = self._build_items_for_classes(records, round_idx, class_key)
        with self._lock:
            self._class_items[class_key] = items
        return list(items)
