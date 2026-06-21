from __future__ import annotations

import math
from typing import Any


def _angle_features(deg: float) -> tuple[float, float]:
    rad = math.radians(float(deg))
    return math.sin(rad), math.cos(rad)


def extract_wind_features(row: dict[str, Any]) -> dict[str, float]:
    wind_stats = row.get("wind_stats") or []
    if not wind_stats:
        return {
            "wind_speed_mean": math.nan,
            "wind_speed_max": math.nan,
            "wind_direction_sin": math.nan,
            "wind_direction_cos": math.nan,
            "attack_angle_mean": math.nan,
            "turbulence_intensity": math.nan,
        }
    first = wind_stats[0]
    speed = float(first.get("mean_wind_speed", math.nan))
    speed_std = float(first.get("std_wind_speed", 0.0))
    direction = float(first.get("mean_wind_direction", math.nan))
    direction_sin, direction_cos = _angle_features(direction) if math.isfinite(direction) else (math.nan, math.nan)
    return {
        "wind_speed_mean": speed,
        "wind_speed_max": speed + speed_std if math.isfinite(speed) else math.nan,
        "wind_direction_sin": direction_sin,
        "wind_direction_cos": direction_cos,
        "attack_angle_mean": float(first.get("mean_wind_attack_angle", math.nan)),
        "turbulence_intensity": float(first.get("turbulence_intensity", math.nan)),
    }


def smoke_wind_features(hour_index: int) -> dict[str, float]:
    direction = 180.0 + (hour_index % 12) * 7.0
    direction_sin, direction_cos = _angle_features(direction)
    speed = 7.5 + 2.0 * math.sin(hour_index / 4.0)
    return {
        "wind_speed_mean": speed,
        "wind_speed_max": speed + 1.2,
        "wind_direction_sin": direction_sin,
        "wind_direction_cos": direction_cos,
        "attack_angle_mean": -2.0 + 0.5 * math.cos(hour_index / 3.0),
        "turbulence_intensity": 0.12 + 0.03 * math.sin(hour_index / 5.0),
    }

