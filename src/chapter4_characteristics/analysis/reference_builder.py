from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from src.chapter4_characteristics.analysis.data_loader import get_nested, load_class_samples
from src.chapter4_characteristics.settings import (
    get_reference_psd_path,
    get_reference_stats_path,
    write_active_round,
)


def _collect_scalars(samples: List[dict], keys: List[str]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {k: [] for k in keys}
    for s in samples:
        for k in keys:
            v = get_nested(s, k)
            if v is not None and np.isfinite(float(v)):
                out[k].append(float(v))
    return out


def build_reference_stats(cfg: dict, round_idx: int) -> dict:
    keys = list(cfg.get("reference_feature_keys", []))
    ref: Dict[str, dict] = {}
    for class_id in (0, 1, 2):
        samples = load_class_samples(class_id, cfg, round_idx)
        scalars = _collect_scalars(samples, keys)
        class_ref = {}
        for k, vals in scalars.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            class_ref[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "n": int(len(arr)),
            }
        ref[str(class_id)] = class_ref

    payload = {
        "round_idx": round_idx,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_keys": keys,
        "classes": ref,
    }
    path = get_reference_stats_path(cfg, round_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _mean_psd_curve(samples: List[dict], plane: str, max_n: int, freq_max: float = 25.0) -> Optional[dict]:
    if not samples:
        return None
    rng = np.random.default_rng(42)
    if len(samples) > max_n:
        idx = rng.choice(len(samples), size=max_n, replace=False)
        subset = [samples[i] for i in idx]
    else:
        subset = samples

    grid0 = np.linspace(0.0, freq_max, 128)
    curves: List[np.ndarray] = []
    for s in subset:
        psd = s.get(f"psd_{plane}") or {}
        freqs = psd.get("frequencies") or []
        powers = psd.get("powers") or []
        if len(freqs) < 2 or not powers:
            continue
        f = np.asarray(freqs, dtype=np.float64)
        p = np.asarray(powers, dtype=np.float64)
        curves.append(np.interp(grid0, f, p, left=0.0, right=0.0))

    if not curves:
        return None
    mean_curve = np.mean(np.stack(curves, axis=0), axis=0)
    return {"frequencies": grid0.tolist(), "powers": mean_curve.tolist()}


def build_reference_psd(cfg: dict, round_idx: int) -> dict:
    max_n = int(cfg.get("reference_psd_max_samples", 2000))
    payload = {"round_idx": round_idx, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "classes": {}}
    for class_id in (0, 1, 2):
        samples = load_class_samples(class_id, cfg, round_idx)
        payload["classes"][str(class_id)] = {
            "inplane": _mean_psd_curve(samples, "inplane", max_n),
            "outplane": _mean_psd_curve(samples, "outplane", max_n),
            "n_samples": len(samples),
        }
    path = get_reference_psd_path(cfg, round_idx)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def post_enrich_artifacts(cfg: dict, round_idx: int) -> None:
    build_reference_stats(cfg, round_idx)
    build_reference_psd(cfg, round_idx)
    write_active_round(cfg, round_idx)
