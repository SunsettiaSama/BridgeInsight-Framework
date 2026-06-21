from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.annotation.store import AnnotationStore
from src.chapter3_identifier.augment.labels import label_name
from src.chapter3_identifier.augment.queue.enrich import AnnotationLookupCache, enrich_record
from src.chapter3_identifier.augment.queue.inference_cache import InferenceSnapshotCache

SLIM_QUEUE_FIELDS = (
    "sample_idx",
    "window_index",
    "prediction",
    "inplane_prediction",
    "outplane_prediction",
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
    "has_round_trajectory_change",
    "queue_match_meta",
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
        self._gold_index_cache: Optional[dict] = None

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

    def _manual_labels_for_record(self, record: dict, manual_index: dict) -> Tuple[Optional[int], Optional[int]]:
        in_fp = record.get("inplane_file_path") or record.get("file_path")
        wi = int(record.get("window_index", 0))
        if not in_fp:
            return None, None
        key = annotation_key(in_fp, wi)
        entry = manual_index.get(key)
        if not entry:
            return None, None
        in_ann = entry.get("inplane_annotation", entry.get("annotation"))
        out_ann = entry.get("outplane_annotation", entry.get("annotation"))
        return (
            int(in_ann) if in_ann is not None else None,
            int(out_ann) if out_ann is not None else None,
        )

    def _match_reasons(
        self,
        record: dict,
        class_set: set[int],
        manual_in: Optional[int],
        manual_out: Optional[int],
    ) -> List[str]:
        reasons: List[str] = []
        in_pred = int(record.get("inplane_prediction", record.get("prediction", 0)))
        out_pred = int(record.get("outplane_prediction", record.get("prediction", 0)))
        fused_pred = int(record.get("prediction", 0))
        if fused_pred in class_set:
            reasons.append(f"模型融合={label_name(fused_pred, self._cfg)}")
        if in_pred in class_set:
            reasons.append(f"模型面内={label_name(in_pred, self._cfg)}")
        if out_pred in class_set:
            reasons.append(f"模型面外={label_name(out_pred, self._cfg)}")
        if manual_in is not None and int(manual_in) in class_set:
            reasons.append(f"人工面内={label_name(int(manual_in), self._cfg)}")
        if manual_out is not None and int(manual_out) in class_set:
            reasons.append(f"人工面外={label_name(int(manual_out), self._cfg)}")
        return reasons

    def _build_abnormal_items(self, records: List[dict], round_idx: int) -> List[dict]:
        return self._build_items_for_classes(records, round_idx, (1, 2, 3))

    def _build_items_for_classes(
        self,
        records: List[dict],
        round_idx: int,
        classes: Tuple[int, ...],
        limit: Optional[int] = None,
    ) -> List[dict]:
        class_set = set(classes)
        if self._gold_index_cache is None:
            self._gold_index_cache = self._annotation_lookup.gold_index(self._gold_store.load_gold())
        gold_index = self._gold_index_cache
        blind_validation_keys = self._annotation_lookup.blind_validation_keys() & set(gold_index.keys())
        manual_index = self._annotation_lookup.manual_index(round_idx)
        changed_keys = self._annotation_lookup.changed_manual_keys(round_idx)
        items: List[dict] = []
        for record in records:
            manual_in, manual_out = self._manual_labels_for_record(record, manual_index)
            reasons = self._match_reasons(record, class_set, manual_in, manual_out)
            in_fp = record.get("inplane_file_path") or record.get("file_path")
            key = annotation_key(in_fp, int(record.get("window_index", 0))) if in_fp else None
            is_blind_validation = bool(
                key
                and key in blind_validation_keys
                and bool(self._cfg.get("webui_include_blind_validation_in_queue", True))
            )
            if not reasons and not is_blind_validation:
                continue
            row = enrich_record(
                record,
                gold_index,
                manual_index,
                self._cfg,
                changed_keys=changed_keys,
                blind_gold_keys=blind_validation_keys,
            )
            row["queue_match_meta"] = " | ".join(reasons)
            items.append(slim_queue_row(row))
            if limit is not None and len(items) >= int(limit):
                break
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
        max_items: Optional[int] = None,
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
                cached = list(self._class_items[class_key])
                if max_items is not None:
                    return cached[: max(1, int(max_items))]
                return cached
            key_changed = self._key != key

        if max_items is not None:
            limit = max(1, int(max_items))
            return self._build_items_for_classes(records, round_idx, class_key, limit=limit)

        if key_changed:
            abnormal_items = self._build_abnormal_items(records, round_idx)
            class_items: Dict[Tuple[int, ...], List[dict]] = {(1, 2, 3): abnormal_items}
            for cls in (1, 2, 3):
                class_items[(cls,)] = self._build_items_for_classes(records, round_idx, (cls,))
            with self._lock:
                self._key = key
                self._abnormal_items = abnormal_items
                self._class_items = class_items

        with self._lock:
            if class_key in self._class_items:
                return list(self._class_items[class_key])

        items = self._build_items_for_classes(records, round_idx, class_key)
        with self._lock:
            self._class_items[class_key] = items
        return list(items)
