from __future__ import annotations

import math
from typing import Any


def nested_get(row: dict[str, Any], dotted: str, default: float = math.nan) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def class_id(row: dict[str, Any]) -> int:
    value = row.get("class_id", row.get("predicted_class", row.get("prediction", 0)))
    return int(value)


def label_source(row: dict[str, Any]) -> str:
    source = row.get("label_source")
    if source:
        return str(source)
    if row.get("is_manual"):
        return "manual"
    if row.get("is_gold"):
        return "gold"
    if row.get("source"):
        return str(row["source"])
    return "model"


def _dominant_freq(row: dict[str, Any]) -> float:
    freqs = nested_get(row, "psd_inplane.frequencies", [])
    if isinstance(freqs, list) and freqs:
        return float(freqs[0])
    return float(nested_get(row, "spectral_inplane.spectral_centroid_hz", math.nan))


def metric_values(row: dict[str, Any], metric_names: list[str]) -> dict[str, float]:
    rms = float(nested_get(row, "time_stats_inplane.rms", math.nan))
    crest = float(nested_get(row, "time_stats_inplane.crest_factor", math.nan))
    values = {
        "max_abs_amplitude": float(crest * rms) if math.isfinite(crest) and math.isfinite(rms) else math.nan,
        "rms": rms,
        "dominant_freq": _dominant_freq(row),
        "spectral_entropy": float(nested_get(row, "spectral_inplane.spectral_entropy", math.nan)),
        "coupling_corr": float(nested_get(row, "cross_coupling.cross_correlation", math.nan)),
    }
    return {name: float(values.get(name, math.nan)) for name in metric_names}


def aggregate_metric_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[int, dict[str, float]]:
    by_class: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        cls = class_id(row)
        by_class.setdefault(cls, []).append(metric_values(row, metric_names))

    aggregated: dict[int, dict[str, float]] = {}
    for cls, values in by_class.items():
        out: dict[str, float] = {}
        for name in metric_names:
            clean = [float(v[name]) for v in values if name in v and math.isfinite(float(v[name]))]
            if not clean:
                out[name] = math.nan
            elif name == "max_abs_amplitude":
                out[name] = max(clean)
            else:
                out[name] = sum(clean) / len(clean)
        aggregated[int(cls)] = out
    return aggregated


def timestamp_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    dataset_tag = str(row.get("dataset_tag", ""))
    ts = row.get("timestamp", {})
    if isinstance(ts, dict):
        month = int(ts.get("month", row.get("month", 0)))
        day = int(ts.get("day", row.get("day", 0)))
        hour = int(ts.get("hour", row.get("hour", 0)))
    else:
        month, day, hour = [int(x) for x in ts[:3]]
    return dataset_tag, month, day, hour, int(row.get("window_idx", row.get("window_index", 0)))


def sample_key(row: dict[str, Any]) -> str:
    dataset_tag, month, day, hour, window_idx = timestamp_key(row)
    cable_pair = row.get("cable_pair") or [row.get("inplane_sensor_id", ""), row.get("outplane_sensor_id", "")]
    cable_text = "|".join(str(x) for x in cable_pair)
    return f"{dataset_tag}:{cable_text}:{month:02d}-{day:02d}-{hour:02d}:{window_idx}"

