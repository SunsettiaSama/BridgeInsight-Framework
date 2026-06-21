from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.chapter3_identifier.regression_forecast._bootstrap import resolve_path
from src.chapter3_identifier.regression_forecast.features.girder import GirderFeatureAdapter
from src.chapter3_identifier.regression_forecast.features.target_schema import (
    ForecastSchema,
    dominant_class_from_counts,
    event_target_from_class,
    risk_class_from_event,
    risk_score_from_event,
)
from src.chapter3_identifier.regression_forecast.features.vibration_targets import (
    aggregate_metric_rows,
    class_id,
    label_source,
    metric_values,
    sample_key,
    timestamp_key,
)
from src.chapter3_identifier.regression_forecast.features.wind import extract_wind_features, smoke_wind_features
from src.chapter3_identifier.regression_forecast.settings import (
    ensure_round_dir,
    get_round_feature_cache_path,
    get_round_schema_path,
    load_config,
    write_json,
    write_jsonl,
)


def _cable_key(row: dict[str, Any]) -> str:
    cable_pair = row.get("cable_pair") or [row.get("inplane_sensor_id", ""), row.get("outplane_sensor_id", "")]
    return "|".join(str(x) for x in cable_pair)


def _timestamp_payload(row: dict[str, Any]) -> dict[str, int | str]:
    dataset_tag, month, day, hour, _ = timestamp_key(row)
    return {"dataset_tag": dataset_tag, "month": month, "day": day, "hour": hour}


def _load_json_samples(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [dict(x) for x in payload]
    samples = payload.get("samples")
    if isinstance(samples, list):
        class_id_value = payload.get("metadata", {}).get("class_id")
        rows = [dict(x) for x in samples]
        if class_id_value is not None:
            for row in rows:
                row.setdefault("class_id", int(class_id_value))
        return rows
    records = payload.get("records")
    if isinstance(records, list):
        return [dict(x) for x in records]
    return []


def _load_real_samples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve_path(str(cfg["feature_analysis_path"]))
    if not path.exists():
        raise FileNotFoundError(f"feature_analysis_path 不存在：{path}")
    if path.is_file():
        rows = _load_json_samples(path)
    else:
        rows = []
        for json_path in sorted(path.rglob("*.json")):
            rows.extend(_load_json_samples(json_path))
    limit = int(cfg.get("limit_samples", 0) or 0)
    if limit > 0:
        rows = rows[:limit]
    return rows


def _make_smoke_samples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    limit = int(cfg.get("limit_samples", 120))
    rows: list[dict[str, Any]] = []
    cable_pair = ["SMOKE-IN-01", "SMOKE-OUT-01"]
    for idx in range(max(limit, 32)):
        wind = smoke_wind_features(idx)
        viv_event = wind["wind_speed_mean"] > 8.8 and idx % 9 in (3, 4, 5)
        rwiv_event = wind["turbulence_intensity"] > 0.135 and idx % 11 in (6, 7)
        other_event = idx % 31 == 0 and idx > 0
        cls = 1 if viv_event else (2 if rwiv_event else (3 if other_event else 0))
        rms = 0.03 + 0.01 * wind["wind_speed_mean"] + (0.06 if cls == 1 else 0.0) + (0.04 if cls == 2 else 0.0)
        crest = 3.0 + (1.2 if cls in (1, 2) else 0.0)
        rows.append(
            {
                "sample_idx": idx,
                "dataset_tag": str(cfg.get("dataset_tag", "smoke")),
                "cable_pair": cable_pair,
                "timestamp": {"dataset_tag": str(cfg.get("dataset_tag", "smoke")), "month": 1, "day": 1 + idx // 24, "hour": idx % 24},
                "window_idx": 0,
                "inplane_sensor_id": cable_pair[0],
                "outplane_sensor_id": cable_pair[1],
                "class_id": cls,
                "label_source": "smoke",
                "time_stats_inplane": {"rms": rms, "crest_factor": crest},
                "psd_inplane": {"frequencies": [0.65 + 0.08 * cls], "powers": [1.0]},
                "spectral_inplane": {"spectral_entropy": 3.8 - 0.6 * (cls == 1), "spectral_centroid_hz": 0.65 + 0.08 * cls},
                "cross_coupling": {"cross_correlation": 0.15 + 0.25 * (cls == 2)},
                "wind_stats": [
                    {
                        "mean_wind_speed": wind["wind_speed_mean"],
                        "std_wind_speed": wind["wind_speed_max"] - wind["wind_speed_mean"],
                        "mean_wind_direction": math.degrees(math.atan2(wind["wind_direction_sin"], wind["wind_direction_cos"])),
                        "mean_wind_attack_angle": wind["attack_angle_mean"],
                        "turbulence_intensity": wind["turbulence_intensity"],
                    }
                ],
            }
        )
    return rows[:limit]


def _input_features(row: dict[str, Any], cfg: dict[str, Any], girder: GirderFeatureAdapter, hour_index: int) -> dict[str, float]:
    metrics = metric_values(row, [str(x) for x in cfg.get("target_metric_names", [])])
    wind = extract_wind_features(row)
    girder_features = girder.features_for(row, hour_index=hour_index)
    event = event_target_from_class(class_id(row), len(cfg.get("class_names", [])))
    out: dict[str, float] = {}
    for name, value in wind.items():
        out[f"wind.{name}"] = float(value)
    for name, value in girder_features.items():
        out[f"girder.{name}"] = float(value)
    for name, value in metrics.items():
        out[f"vibration.{name}"] = float(value)
    for idx, value in enumerate(event):
        out[f"event.{cfg['class_names'][idx]}"] = float(value)
    return out


def _build_records(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    metric_names = [str(x) for x in cfg.get("target_metric_names", [])]
    class_names = [str(x) for x in cfg.get("class_names", [])]
    horizons = [int(x) for x in cfg.get("horizons_hours", [])]
    girder = GirderFeatureAdapter(cfg)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_cable_key(row), []).append(row)

    records: list[dict[str, Any]] = []
    input_names: list[str] = []
    for cable, group_rows in groups.items():
        ordered = sorted(group_rows, key=timestamp_key)
        for hour_index, row in enumerate(ordered):
            input_features = _input_features(row, cfg, girder, hour_index)
            if not input_names:
                input_names = sorted(input_features.keys())
            future_by_horizon: dict[str, dict[str, Any]] = {}
            for horizon in horizons:
                future_rows = ordered[hour_index + 1 : hour_index + horizon + 1]
                counts = [0] * len(class_names)
                event = [0.0] * len(class_names)
                for future in future_rows:
                    cls = class_id(future)
                    if 0 <= cls < len(class_names):
                        counts[cls] += 1
                        event[cls] = 1.0
                metrics_by_class = aggregate_metric_rows(future_rows, metric_names) if future_rows else {}
                future_by_horizon[str(horizon)] = {
                    "horizon_hours": horizon,
                    "event_target": event,
                    "dominant_class": dominant_class_from_counts(counts, class_names),
                    "risk_class": risk_class_from_event(event, class_names),
                    "risk_score": risk_score_from_event(event),
                    "metrics_by_class": {class_names[int(k)]: v for k, v in metrics_by_class.items() if int(k) < len(class_names)},
                    "mask": 1 if future_rows else 0,
                    "label_source": label_source(row),
                }
            records.append(
                {
                    "sample_key": sample_key(row),
                    "sample_idx": int(row.get("sample_idx", len(records))),
                    "cable_pair": row.get("cable_pair") or [row.get("inplane_sensor_id", ""), row.get("outplane_sensor_id", "")],
                    "timestamp": _timestamp_payload(row),
                    "window_idx": int(row.get("window_idx", row.get("window_index", 0))),
                    "hour_index": hour_index,
                    "cable_key": cable,
                    "input_features": input_features,
                    "current_class": class_id(row),
                    "label_source": label_source(row),
                    "future_targets_by_horizon": future_by_horizon,
                }
            )
    return records, input_names


def build_feature_cache(round_idx: int = 1, config_path: str | None = None) -> str:
    cfg = load_config(config_path)
    ensure_round_dir(cfg, round_idx)
    mode = str(cfg.get("feature_cache_mode", "real"))
    source_rows = _make_smoke_samples(cfg) if mode == "smoke" else _load_real_samples(cfg)
    if not source_rows:
        raise ValueError("没有可用于构建 feature cache 的样本")
    records, input_names = _build_records(source_rows, cfg)
    schema = ForecastSchema.from_config(cfg, input_feature_names=input_names)
    write_json(get_round_schema_path(cfg, round_idx), schema.to_dict())
    path = get_round_feature_cache_path(cfg, round_idx)
    write_jsonl(path, records)
    return str(path)


def check_data(round_idx: int = 1, config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    mode = str(cfg.get("feature_cache_mode", "real"))
    if mode == "smoke":
        rows = _make_smoke_samples(cfg)
    else:
        rows = _load_real_samples(cfg)
        GirderFeatureAdapter(cfg)
    horizons = [int(x) for x in cfg.get("horizons_hours", [])]
    return {
        "ok": bool(rows),
        "mode": mode,
        "source_samples": len(rows),
        "horizons_hours": horizons,
        "round_idx": int(round_idx),
        "feature_cache_path": str(get_round_feature_cache_path(cfg, round_idx)),
    }

