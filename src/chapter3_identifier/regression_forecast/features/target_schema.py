from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForecastSchema:
    schema_version: str
    class_names: list[str]
    horizons_hours: list[int]
    metric_names: list[str]
    input_feature_names: list[str]
    normalization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict, input_feature_names: list[str] | None = None) -> "ForecastSchema":
        return cls(
            schema_version=str(cfg.get("schema_version", "regression_forecast_v1")),
            class_names=[str(x) for x in cfg.get("class_names", ["Normal", "VIV", "RWIV", "Others"])],
            horizons_hours=[int(x) for x in cfg.get("horizons_hours", [1, 3, 6, 12, 24])],
            metric_names=[str(x) for x in cfg.get("target_metric_names", [])],
            input_feature_names=list(input_feature_names or []),
            normalization={},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def event_target_from_class(class_id: int, num_classes: int) -> list[float]:
    target = [0.0] * int(num_classes)
    if 0 <= int(class_id) < int(num_classes):
        target[int(class_id)] = 1.0
    return target


def dominant_class_from_counts(counts: list[int], class_names: list[str]) -> str:
    if not counts:
        return class_names[0]
    best_idx = max(range(len(counts)), key=lambda idx: (counts[idx], idx != 0))
    if 0 <= best_idx < len(class_names):
        return class_names[best_idx]
    return class_names[0]


def risk_class_from_event(event_values: list[float], class_names: list[str]) -> str:
    risk_order = [3, 2, 1]
    for idx in risk_order:
        if idx < len(event_values) and float(event_values[idx]) > 0.5:
            return class_names[idx]
    return class_names[0]


def risk_score_from_event(event_values: list[float]) -> float:
    abnormal = [float(x) for x in event_values[1:]]
    return max(abnormal) if abnormal else 0.0

