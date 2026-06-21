from __future__ import annotations

import json
from pathlib import Path

from src.chapter3_identifier.regression_forecast._bootstrap import resolve_path


def idle_state() -> dict:
    return {"status": "idle", "phase": None, "round": 0, "pid": None, "log_path": None, "error": None}


def read_job_state(path: str) -> dict:
    p = resolve_path(path)
    if not p.exists():
        return idle_state()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_job_state(path: str, patch: dict) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = read_job_state(path)
    state.update(patch)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

