from __future__ import annotations

from typing import Any

from src.chapter3_identifier.regression_forecast.settings import get_round_metrics_path, read_json


def build_training_monitor_payload(cfg: dict[str, Any], round_idx: int, job_state: dict, log_tail: str) -> dict[str, Any]:
    path = get_round_metrics_path(cfg, round_idx)
    payload = read_json(path) if path.exists() else {"history": []}
    history = payload.get("history", [])
    latest = history[-1] if history else None
    return {
        "round_idx": int(round_idx),
        "job": job_state,
        "history": history,
        "latest": latest,
        "best_val_loss": payload.get("best_val_loss"),
        "split_sizes": payload.get("split_sizes", {}),
        "log_tail": log_tail,
    }

