from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional

from sklearn.metrics import cohen_kappa_score

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.queue.inference_cache import parse_infer_progress
from src.chapter3_identifier.augment.annotation.store import load_cumulative_manual_change_events
from src.chapter3_identifier.augment.settings import get_round_manual_edits_path

if TYPE_CHECKING:
    from src.chapter3_identifier.augment.queue.inference_cache import InferenceSnapshotCache

_CONSISTENCY_CACHE_KEY: tuple | None = None
_CONSISTENCY_CACHE_VALUE: dict | None = None


def _safe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def _load_manual_snapshots(cfg: dict, round_idx: int) -> Dict[tuple[str, int], List[dict]]:
    history: Dict[tuple[str, int], List[dict]] = {}
    for idx in range(1, max(round_idx, 1) + 1):
        path = get_round_manual_edits_path(cfg, idx)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            fp = row.get("file_path")
            if not fp:
                continue
            wi = _safe_int(row.get("window_index"), 0)
            in_ann = _safe_int(row.get("inplane_annotation", row.get("annotation")), 0)
            out_ann = _safe_int(row.get("outplane_annotation", row.get("annotation")), 0)
            key = annotation_key(fp, wi)
            history.setdefault(key, []).append(
                {
                    "round_idx": idx,
                    "updated_at": str(row.get("updated_at", "")),
                    "inplane_annotation": in_ann,
                    "outplane_annotation": out_ann,
                    "file_path": fp,
                    "window_index": wi,
                }
            )
    for key in history:
        history[key].sort(key=lambda x: (int(x["round_idx"]), x["updated_at"]))
    return history


def _load_manual_change_events(cfg: dict, round_idx: int) -> List[dict]:
    return load_cumulative_manual_change_events(cfg, max(round_idx, 1))


def _manual_round_signature(cfg: dict, round_idx: int) -> tuple:
    parts = []
    for idx in range(1, max(round_idx, 1) + 1):
        edits_path = get_round_manual_edits_path(cfg, idx)
        history_path = edits_path.with_name("manual_edits_history.jsonl")
        edits_mtime = edits_path.stat().st_mtime if edits_path.exists() else -1.0
        history_mtime = history_path.stat().st_mtime if history_path.exists() else -1.0
        parts.append((idx, edits_mtime, history_mtime))
    return tuple(parts)


def _kappa_payload(first_labels: List[int], latest_labels: List[int]) -> dict:
    if len(first_labels) < 2 or len(latest_labels) < 2:
        return {
            "available": False,
            "reason": "样本不足（至少需要2个可比样本）",
            "kappa": None,
            "agreement_rate": None,
            "sample_count": len(first_labels),
        }
    agree = sum(1 for a, b in zip(first_labels, latest_labels) if a == b)
    agreement_rate = agree / max(len(first_labels), 1)
    kappa = cohen_kappa_score(first_labels, latest_labels)
    return {
        "available": True,
        "reason": "",
        "kappa": float(kappa),
        "agreement_rate": float(agreement_rate),
        "sample_count": len(first_labels),
    }


def build_annotation_consistency_payload(cfg: dict, round_idx: int, label_names: List[str]) -> dict:
    global _CONSISTENCY_CACHE_KEY, _CONSISTENCY_CACHE_VALUE
    cache_key = (
        int(round_idx),
        tuple(label_names),
        _manual_round_signature(cfg, round_idx),
    )
    if _CONSISTENCY_CACHE_KEY == cache_key and _CONSISTENCY_CACHE_VALUE is not None:
        return _CONSISTENCY_CACHE_VALUE

    num_classes = len(label_names)
    history = _load_manual_snapshots(cfg, round_idx)
    change_events = _load_manual_change_events(cfg, round_idx)
    repeated = [(key, events) for key, events in history.items() if len(events) >= 2]
    in_first: List[int] = []
    in_latest: List[int] = []
    out_first: List[int] = []
    out_latest: List[int] = []
    both_same = 0
    compared = 0
    repeated_rows = []

    for key, events in repeated:
        first = events[0]
        latest = events[-1]
        fi = int(first["inplane_annotation"])
        li = int(latest["inplane_annotation"])
        fo = int(first["outplane_annotation"])
        lo = int(latest["outplane_annotation"])
        if 0 <= fi < num_classes and 0 <= li < num_classes:
            in_first.append(fi)
            in_latest.append(li)
        if 0 <= fo < num_classes and 0 <= lo < num_classes:
            out_first.append(fo)
            out_latest.append(lo)
        compared += 1
        if fi == li and fo == lo:
            both_same += 1
        repeated_rows.append(
            {
                "file_path": latest["file_path"],
                "window_index": int(latest["window_index"]),
                "reannotated_times": len(events),
                "first_round": int(first["round_idx"]),
                "latest_round": int(latest["round_idx"]),
                "first_inplane": fi,
                "latest_inplane": li,
                "first_outplane": fo,
                "latest_outplane": lo,
                "first_inplane_name": label_names[fi] if 0 <= fi < num_classes else str(fi),
                "latest_inplane_name": label_names[li] if 0 <= li < num_classes else str(li),
                "first_outplane_name": label_names[fo] if 0 <= fo < num_classes else str(fo),
                "latest_outplane_name": label_names[lo] if 0 <= lo < num_classes else str(lo),
            }
        )

    repeated_rows.sort(
        key=lambda row: (
            -int(row["reannotated_times"]),
            -abs(int(row["latest_round"]) - int(row["first_round"])),
        )
    )
    agreement_rate = float(both_same / compared) if compared > 0 else 0.0
    per_round_change_counts: Dict[str, int] = {}
    for event in change_events:
        key = str(int(event.get("round_idx", 0)))
        if not bool(event.get("changed_any", True)):
            continue
        per_round_change_counts[key] = int(per_round_change_counts.get(key, 0)) + 1

    recent_changes = []
    for event in reversed(change_events):
        if not bool(event.get("changed_any", True)):
            continue
        bi = event.get("before_inplane_annotation")
        bo = event.get("before_outplane_annotation")
        ai = event.get("after_inplane_annotation")
        ao = event.get("after_outplane_annotation")
        recent_changes.append(
            {
                "event_time": str(event.get("event_time", "")),
                "round_idx": int(event.get("round_idx", 0)),
                "window_index": int(event.get("window_index", 0)),
                "before_inplane_name": label_names[int(bi)] if bi is not None and 0 <= int(bi) < num_classes else "-",
                "before_outplane_name": label_names[int(bo)] if bo is not None and 0 <= int(bo) < num_classes else "-",
                "after_inplane_name": label_names[int(ai)] if ai is not None and 0 <= int(ai) < num_classes else "-",
                "after_outplane_name": label_names[int(ao)] if ao is not None and 0 <= int(ao) < num_classes else "-",
            }
        )
        if len(recent_changes) >= 20:
            break

    payload = {
        "total_distinct_samples": len(history),
        "reannotated_sample_count": len(repeated),
        "compared_pairs": compared,
        "both_direction_agreement_rate": agreement_rate,
        "inplane_kappa": _kappa_payload(in_first, in_latest),
        "outplane_kappa": _kappa_payload(out_first, out_latest),
        "change_events_total": len(change_events),
        "per_round_change_counts": per_round_change_counts,
        "recent_changes": recent_changes,
        "top_reannotated": repeated_rows[:20],
    }
    _CONSISTENCY_CACHE_KEY = cache_key
    _CONSISTENCY_CACHE_VALUE = payload
    return payload


def build_infer_monitor_payload(
    cfg: dict,
    round_idx: int,
    job_state: dict,
    log_tail: str,
    inference_path: str,
    typical_topk: int = 2,
    inference_cache: Optional["InferenceSnapshotCache"] = None,
) -> dict:
    label_names = get_label_names(cfg)
    distribution: List[dict] = []
    typical: Dict[int, List[dict]] = {}
    total_records = 0
    inference_ready = False

    if inference_cache is not None and inference_path:
        distribution, typical, inference_ready = inference_cache.get_summary(
            inference_path,
            label_names,
            typical_topk,
        )
        if inference_ready:
            total_records = len(inference_cache.get_records(inference_path))

    progress = parse_infer_progress(log_tail)
    if job_state.get("status") == "running" and progress:
        job_state = dict(job_state)
        job_state["infer_progress"] = progress

    return {
        "round_idx": round_idx,
        "job": job_state,
        "log_tail": log_tail,
        "label_names": label_names,
        "num_classes": len(label_names),
        "total_records": total_records,
        "inference_ready": inference_ready,
        "distribution": distribution,
        "typical_samples": typical,
        "infer_progress": progress,
        "annotation_consistency": build_annotation_consistency_payload(
            cfg,
            round_idx,
            label_names,
        ),
    }
