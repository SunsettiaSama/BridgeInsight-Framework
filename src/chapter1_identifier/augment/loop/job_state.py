from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.chapter1_identifier.augment._bootstrap import resolve_path


def read_job_state(path: str) -> dict:
    p = resolve_path(path)
    if not p.exists():
        return {"status": "idle", "phase": None, "round": 0, "pid": None, "log_path": None, "error": None}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_job_state(path: str, state: dict) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
