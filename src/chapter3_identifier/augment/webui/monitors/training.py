from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.settings import get_round_dir


def _round_path(cfg_or_root, round_idx: int) -> Path:
    if isinstance(cfg_or_root, dict):
        return get_round_dir(cfg_or_root, round_idx)
    return resolve_path(cfg_or_root) / f"round_{round_idx:02d}"


def _read_json(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_metrics_history(rounds_root_or_cfg, round_idx: int) -> List[dict]:
    metrics_path = _round_path(rounds_root_or_cfg, round_idx) / "metrics.json"
    live_path = _round_path(rounds_root_or_cfg, round_idx) / "metrics_live.json"

    data = _read_json(metrics_path)
    if data is None:
        data = _read_json(live_path)
        if isinstance(data, dict):
            return [data]
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    if isinstance(data, dict):
        return [data]
    return []


def load_latest_metrics(rounds_root_or_cfg, round_idx: int) -> Optional[dict]:
    live_path = _round_path(rounds_root_or_cfg, round_idx) / "metrics_live.json"
    data = _read_json(live_path)
    if isinstance(data, dict):
        return data
    history = load_metrics_history(rounds_root_or_cfg, round_idx)
    return history[-1] if history else None


def find_best_epoch(history: List[dict]) -> Optional[dict]:
    best = None
    best_score = -1.0
    for row in history:
        val_metrics = row.get("val_metrics") or {}
        score = float(
            val_metrics.get(
                "viv_rwiv_mean_f1",
                val_metrics.get("joint_viv_rwiv_mean_f1", -1.0),
            )
        )
        if score > best_score:
            best_score = score
            best = row
    return best


def latest_confusion_path(
    rounds_root_or_cfg,
    round_idx: int,
    epoch: Optional[int] = None,
    kind: str = "merged",
) -> Optional[Path]:
    out_dir = _round_path(rounds_root_or_cfg, round_idx)
    if not out_dir.exists():
        return None
    stem = "confusion_matrix"
    if kind in ("inplane", "outplane"):
        stem = f"confusion_matrix_{kind}"
    if epoch is not None:
        path = out_dir / f"{stem}_epoch_{epoch:03d}.png"
        return path if path.exists() else None
    candidates = sorted(out_dir.glob(f"{stem}_epoch_*.png"))
    return candidates[-1] if candidates else None


def build_monitor_payload(
    cfg: dict,
    round_idx: int,
    epochs_total: int,
    job_state: dict,
    log_tail: str,
) -> Dict:
    history = load_metrics_history(cfg, round_idx)
    latest = load_latest_metrics(cfg, round_idx) or (history[-1] if history else None)
    best = find_best_epoch(history)
    confusion = latest_confusion_path(cfg, round_idx, kind="merged")
    confusion_in = latest_confusion_path(cfg, round_idx, kind="inplane")
    confusion_out = latest_confusion_path(cfg, round_idx, kind="outplane")
    current_epoch = int(latest["epoch"]) if latest else 0
    latest_info = (latest or {}).get("dataset_info") or {}
    training_profile = job_state.get("training_profile") or latest_info.get("training_profile")

    return {
        "round_idx": round_idx,
        "epochs_total": epochs_total,
        "current_epoch": current_epoch,
        "job": job_state,
        "history": history,
        "latest": latest,
        "best": best,
        "training_profile": training_profile,
        "log_tail": log_tail,
        "confusion_epoch": best.get("epoch") if best else (latest.get("epoch") if latest else None),
        "has_confusion": confusion is not None,
        "has_inplane_confusion": confusion_in is not None,
        "has_outplane_confusion": confusion_out is not None,
    }
