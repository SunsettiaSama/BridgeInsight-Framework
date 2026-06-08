from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.chapter1_identifier.augment._bootstrap import resolve_path


def load_metrics_history(training_output_dir: str, round_idx: int) -> List[dict]:
    metrics_path = resolve_path(training_output_dir) / f"round_{round_idx:02d}" / "metrics.json"
    if not metrics_path.exists():
        return []
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    return []


def load_latest_metrics(training_output_dir: str, round_idx: int) -> Optional[dict]:
    live_path = resolve_path(training_output_dir) / f"round_{round_idx:02d}" / "metrics_live.json"
    if live_path.exists():
        with open(live_path, "r", encoding="utf-8") as f:
            return json.load(f)
    history = load_metrics_history(training_output_dir, round_idx)
    return history[-1] if history else None


def find_best_epoch(history: List[dict]) -> Optional[dict]:
    best = None
    best_score = -1.0
    for row in history:
        val_metrics = row.get("val_metrics") or {}
        score = float(val_metrics.get("viv_rwiv_mean_f1", -1.0))
        if score > best_score:
            best_score = score
            best = row
    return best


def latest_confusion_path(training_output_dir: str, round_idx: int, epoch: Optional[int] = None) -> Optional[Path]:
    out_dir = resolve_path(training_output_dir) / f"round_{round_idx:02d}"
    if not out_dir.exists():
        return None
    if epoch is not None:
        path = out_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
        return path if path.exists() else None
    candidates = sorted(out_dir.glob("confusion_matrix_epoch_*.png"))
    return candidates[-1] if candidates else None


def build_monitor_payload(
    training_output_dir: str,
    round_idx: int,
    epochs_total: int,
    job_state: dict,
    log_tail: str,
) -> Dict:
    history = load_metrics_history(training_output_dir, round_idx)
    latest = history[-1] if history else None
    best = find_best_epoch(history)
    confusion = latest_confusion_path(training_output_dir, round_idx)
    current_epoch = int(latest["epoch"]) if latest else 0

    return {
        "round_idx": round_idx,
        "epochs_total": epochs_total,
        "current_epoch": current_epoch,
        "job": job_state,
        "history": history,
        "latest": latest,
        "best": best,
        "log_tail": log_tail,
        "confusion_epoch": best.get("epoch") if best else (latest.get("epoch") if latest else None),
        "has_confusion": confusion is not None,
    }
