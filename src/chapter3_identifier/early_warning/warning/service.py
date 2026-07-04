from __future__ import annotations

from typing import Any

from src.chapter3_identifier.early_warning.warning.policy import WarningPolicy


def _record_best_horizon(record: dict[str, Any], horizon_filter: int | None = None) -> dict[str, Any]:
    horizons = record.get("horizons", [])
    selected = [
        row
        for row in horizons
        if horizon_filter is None or int(row.get("horizon_hours", -1)) == int(horizon_filter)
    ]
    if not selected:
        selected = horizons
    if not selected:
        return {"risk_score": 0.0, "risk_class": "Normal", "horizon_hours": None, "event_proba": {}, "metrics_by_class": {}}
    return max(selected, key=lambda row: float(row.get("risk_score", 0.0)))


def _feature_group(features: dict[str, float], prefix: str, display_order: list[str]) -> dict[str, float]:
    group: dict[str, float] = {}
    for name in display_order:
        key = f"{prefix}.{name}"
        if key in features:
            group[name] = float(features[key])
    for key, value in sorted(features.items()):
        if key.startswith(f"{prefix}.") and key.split(".", 1)[1] not in group:
            group[key.split(".", 1)[1]] = float(value)
    return group


def summarize_wind(features: dict[str, float], display_order: list[str]) -> dict[str, float]:
    return _feature_group(features, "wind", display_order)


def summarize_vibration(features: dict[str, float], display_order: list[str]) -> dict[str, float]:
    return _feature_group(features, "vibration", display_order)


def summarize_girder(features: dict[str, float], display_order: list[str]) -> dict[str, float]:
    return _feature_group(features, "girder", display_order)


def summarize_event(features: dict[str, float], class_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in class_names:
        key = f"event.{name}"
        if key in features:
            out[name] = float(features[key])
    return out


def build_horizon_matrix(
    record: dict[str, Any],
    class_names: list[str],
    monitor_classes: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in record.get("horizons", []):
        event_proba = horizon.get("event_proba", {})
        cells = {name: float(event_proba.get(name, 0.0)) for name in monitor_classes}
        rows.append(
            {
                "horizon_hours": int(horizon.get("horizon_hours", 0)),
                "event_proba": cells,
                "risk_score": float(horizon.get("risk_score", 0.0)),
                "risk_class": str(horizon.get("risk_class", "Normal")),
                "dominant_class": str(horizon.get("dominant_class", "Normal")),
            }
        )
    rows.sort(key=lambda row: int(row["horizon_hours"]))
    return rows


def record_to_warning_summary(
    record: dict[str, Any],
    policy: WarningPolicy,
    class_names: list[str],
    horizon_filter: int | None = None,
    wind_order: list[str] | None = None,
    vibration_order: list[str] | None = None,
    girder_order: list[str] | None = None,
) -> dict[str, Any]:
    best = _record_best_horizon(record, horizon_filter)
    risk_score = float(best.get("risk_score", 0.0))
    warning_level = policy.level_from_score(risk_score)
    features = dict((record.get("input_summary") or {}).get("features") or {})
    input_summary = record.get("input_summary") or {}
    return {
        "sample_idx": record.get("sample_idx"),
        "timestamp": record.get("timestamp"),
        "cable_pair": record.get("cable_pair", []),
        "risk_class": best.get("risk_class"),
        "risk_score": risk_score,
        "warning_level": warning_level,
        "warning_level_label": policy.level_label(warning_level),
        "horizon_hours": best.get("horizon_hours"),
        "dominant_class": best.get("dominant_class"),
        "current_class": input_summary.get("current_class"),
        "wind_summary": summarize_wind(features, wind_order or []),
        "vibration_summary": summarize_vibration(features, vibration_order or []),
        "girder_summary": summarize_girder(features, girder_order or []),
        "event_summary": summarize_event(features, class_names),
        "horizon_matrix": build_horizon_matrix(record, class_names, policy.monitor_classes),
    }


def record_to_warning_detail(
    record: dict[str, Any],
    policy: WarningPolicy,
    class_names: list[str],
    wind_order: list[str] | None = None,
    vibration_order: list[str] | None = None,
    girder_order: list[str] | None = None,
) -> dict[str, Any]:
    summary = record_to_warning_summary(
        record,
        policy,
        class_names,
        wind_order=wind_order,
        vibration_order=vibration_order,
        girder_order=girder_order,
    )
    features = dict((record.get("input_summary") or {}).get("features") or {})
    grouped_features = {
        "wind": summarize_wind(features, wind_order or []),
        "girder": summarize_girder(features, girder_order or []),
        "vibration": summarize_vibration(features, vibration_order or []),
        "event": summarize_event(features, class_names),
    }
    return {
        **summary,
        "feature_groups": grouped_features,
        "input_summary": record.get("input_summary") or {},
        "horizons": record.get("horizons", []),
        "forecast_record": record,
    }


def list_warnings(
    records: list[dict[str, Any]],
    policy: WarningPolicy,
    class_names: list[str],
    *,
    horizon_hours: int | None = None,
    warning_level: str | None = None,
    risk_class: str | None = None,
    min_risk: float = 0.0,
    cable_pair: str | None = None,
    limit: int = 200,
    wind_order: list[str] | None = None,
    vibration_order: list[str] | None = None,
    girder_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for record in records:
        summary = record_to_warning_summary(
            record,
            policy,
            class_names,
            horizon_filter=horizon_hours,
            wind_order=wind_order,
            vibration_order=vibration_order,
            girder_order=girder_order,
        )
        if not policy.show_normal_in_queue and summary["warning_level"] == "normal":
            continue
        if warning_level and summary["warning_level"] != warning_level:
            continue
        if risk_class and str(summary.get("risk_class")) != risk_class:
            continue
        if float(summary.get("risk_score", 0.0)) < float(min_risk):
            continue
        cable_text = "|".join(str(x) for x in summary.get("cable_pair", []))
        if cable_pair and cable_pair not in cable_text:
            continue
        items.append(summary)
    items.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
    max_items = max(1, int(limit))
    return items[:max_items]
