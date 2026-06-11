from __future__ import annotations

import heapq
import re
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.queue.loader import load_inference_records

_PROGRESS_RE = re.compile(
    r"DualStream\s+(inplane|outplane)\s+推理:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)"
)
_PLAIN_INFER_RE = re.compile(
    r"DualStream\s+(inplane|outplane)\s+推理：(\d+)/(\d+)\s+\(([\d.]+)%\)"
)
_PLAIN_VALIDATE_RE = re.compile(
    r"\[(面内|面外)\]\s+预验证文件长度：(\d+)/(\d+)\s+\(([\d.]+)%\)"
)
_DIRECTION_MAP = {"面内": "inplane", "面外": "outplane"}


def _record_brief(record: dict) -> dict:
    proba = record.get("proba") or []
    pred = int(record.get("prediction", 0))
    confidence = float(proba[pred]) if pred < len(proba) else 0.0
    return {
        "sample_idx": int(record.get("sample_idx", -1)),
        "prediction": pred,
        "confidence": confidence,
        "uncertainty": float(record.get("uncertainty", 0.0)),
        "window_index": int(record.get("window_index", 0)),
        "inplane_sensor_id": record.get("inplane_sensor_id"),
        "proba": [float(x) for x in proba],
    }


def compute_class_distribution(records: List[dict], label_names: List[str]) -> List[dict]:
    counts = Counter(int(r.get("prediction", 0)) for r in records)
    total = len(records) or 1
    rows = []
    for class_id, name in enumerate(label_names):
        count = int(counts.get(class_id, 0))
        rows.append(
            {
                "class_id": class_id,
                "name": name,
                "count": count,
                "ratio": count / total,
            }
        )
    return rows


def parse_infer_progress(log_tail: str) -> Optional[dict]:
    if not log_tail:
        return None
    last = None
    last_kind = None
    for line in log_tail.splitlines():
        m = _PROGRESS_RE.search(line)
        if m:
            last = m
            last_kind = "infer"
            continue
        m = _PLAIN_INFER_RE.search(line)
        if m:
            last = m
            last_kind = "infer"
            continue
        m = _PLAIN_VALIDATE_RE.search(line)
        if m:
            last = m
            last_kind = "validate"
    if last is None:
        return None
    if last_kind == "validate":
        direction = _DIRECTION_MAP[last.group(1)]
        current, total = int(last.group(2)), int(last.group(3))
        percent = float(last.group(4))
    else:
        direction = last.group(1)
        current, total = int(last.group(2)), int(last.group(3))
        percent = round(100.0 * current / total, 1) if total else 0.0
    ratio = current / total if total else 0.0
    return {
        "direction": direction,
        "current": current,
        "total": total,
        "ratio": ratio,
        "percent": percent,
        "phase": last_kind or "infer",
    }


def _pick_typical_samples_fast(
    by_idx: Dict[int, dict],
    records: List[dict],
    num_classes: int,
    topk: int,
) -> Dict[int, List[dict]]:
    heaps: Dict[int, List[tuple]] = {i: [] for i in range(num_classes)}
    for record in records:
        brief = _record_brief(record)
        pred = brief["prediction"]
        if pred not in heaps:
            continue
        item = (-brief["confidence"], brief["uncertainty"], int(brief["sample_idx"]))
        heap = heaps[pred]
        if len(heap) < topk:
            heapq.heappush(heap, item)
        elif item < heap[0]:
            heapq.heapreplace(heap, item)

    typical: Dict[int, List[dict]] = {}
    for class_id, heap in heaps.items():
        ranked = sorted(heap)
        typical[class_id] = [
            _record_brief(by_idx[idx]) for _, _, idx in ranked if idx in by_idx
        ]
    return typical


class InferenceSnapshotCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Optional[str] = None
        self._mtime: float = -1.0
        self._records: List[dict] = []
        self._by_idx: Dict[int, dict] = {}
        self._by_key: Dict[tuple[str, int], int] = {}
        self._summary_key: Optional[Tuple[str, float, int, tuple]] = None
        self._distribution: List[dict] = []
        self._typical: Dict[int, List[dict]] = {}

    def _path_mtime(self, path: str) -> Tuple[str, float]:
        p = resolve_path(path)
        if not p.exists():
            return str(p), -1.0
        return str(p.resolve()), p.stat().st_mtime

    def get_records(self, path: str) -> List[dict]:
        resolved, mtime = self._path_mtime(path)
        if mtime < 0:
            with self._lock:
                if self._path == resolved:
                    self._records = []
                    self._by_idx = {}
                    self._by_key = {}
                    self._summary_key = None
                self._path = resolved
                self._mtime = mtime
            return []

        with self._lock:
            if self._path == resolved and self._mtime == mtime and self._records:
                return self._records

        records = load_inference_records(resolved)
        by_idx = {int(r["sample_idx"]): r for r in records}
        by_key: Dict[tuple[str, int], int] = {}
        for record in records:
            in_fp = record.get("inplane_file_path")
            if not in_fp:
                continue
            by_key[annotation_key(in_fp, int(record.get("window_index", 0)))] = int(record["sample_idx"])
        with self._lock:
            self._path = resolved
            self._mtime = mtime
            self._records = records
            self._by_idx = by_idx
            self._by_key = by_key
            self._summary_key = None
        return records

    def get_key_index(self, path: str) -> Dict[tuple[str, int], int]:
        self.get_records(path)
        with self._lock:
            return dict(self._by_key)

    def get_record(self, path: str, sample_idx: int) -> Optional[dict]:
        self.get_records(path)
        with self._lock:
            return self._by_idx.get(int(sample_idx))

    def get_mtime(self, path: str) -> float:
        _, mtime = self._path_mtime(path)
        return mtime

    def get_summary(
        self,
        path: str,
        label_names: List[str],
        typical_topk: int,
    ) -> Tuple[List[dict], Dict[int, List[dict]], bool]:
        records = self.get_records(path)
        if not records:
            return [], {}, False

        resolved, mtime = self._path_mtime(path)
        key = (resolved, mtime, typical_topk, tuple(label_names))
        with self._lock:
            if self._summary_key == key:
                return self._distribution, self._typical, True

        distribution = compute_class_distribution(records, label_names)
        with self._lock:
            by_idx = dict(self._by_idx)
        typical = _pick_typical_samples_fast(by_idx, records, len(label_names), typical_topk)
        with self._lock:
            self._summary_key = key
            self._distribution = distribution
            self._typical = typical
        return distribution, typical, True
