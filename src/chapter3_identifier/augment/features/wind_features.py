from __future__ import annotations

import re
from typing import Optional

import numpy as np

from src.chapter3_identifier.augment.features.wind_index import resolve_wind_meta
from src.config.sensor_config import WIND_DIR_CORRECTION, WIND_FS, WIND_TIME_WINDOW, WIND_VALID_THRESHOLD
from src.data_processer.preprocess.get_data_wind import parse_single_metadata_to_wind_data

WIND_FEATURE_NAMES = (
    "avg_wind_speed_scaled",
    "turbulence_scaled",
    "wind_direction_sin",
    "wind_direction_cos",
    "window_coverage",
)
WIND_FEATURE_DIM = len(WIND_FEATURE_NAMES)

_VIC_PATH_RE = re.compile(r"[\\/](\d{2})[\\/](\d{2})[\\/][^\\/]+_(\d{2})\d{4}\.VIC$", re.IGNORECASE)
_VIC_SENSOR_RE = re.compile(r"([^\\/]+)_\d{6}\.VIC$", re.IGNORECASE)


def zero_wind_features() -> np.ndarray:
    return np.zeros(WIND_FEATURE_DIM, dtype=np.float32)


def _timestamp_from_path(file_path: str) -> Optional[list[int]]:
    match = _VIC_PATH_RE.search(str(file_path))
    if match is None:
        return None
    return [int(match.group(1)), int(match.group(2)), int(match.group(3))]


def _sensor_from_path(file_path: str) -> str:
    match = _VIC_SENSOR_RE.search(str(file_path))
    if match is None:
        return ""
    return str(match.group(1))


def _record_for_wind_lookup(record: dict) -> dict:
    in_meta = record.get("inplane_meta") or {}
    metadata = record.get("metadata") or in_meta
    inplane_file_path = (
        record.get("inplane_file_path")
        or record.get("file_path")
        or in_meta.get("file_path")
        or metadata.get("file_path")
        or metadata.get("path")
        or ""
    )
    timestamp = record.get("timestamp")
    if timestamp is None:
        month = metadata.get("month")
        day = metadata.get("day")
        hour = metadata.get("hour")
        if month is not None and day is not None and hour is not None:
            timestamp = [int(month), int(day), int(hour)]
    if timestamp is None and inplane_file_path:
        timestamp = _timestamp_from_path(str(inplane_file_path))

    sensor_id = (
        record.get("inplane_sensor_id")
        or record.get("sensor_id")
        or in_meta.get("sensor_id")
        or metadata.get("sensor_id")
        or _sensor_from_path(str(inplane_file_path))
        or ""
    )
    return {
        "timestamp": timestamp,
        "window_index": int(record.get("window_index", record.get("window_idx", 0))),
        "inplane_sensor_id": str(sensor_id),
        "inplane_file_path": str(inplane_file_path),
    }


def _mean_wind_direction(direction_deg: np.ndarray) -> Optional[float]:
    if direction_deg.size == 0:
        return None
    radians = np.deg2rad(direction_deg)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return None
    return float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0)


def _window_arrays(
    wind_speed: np.ndarray,
    wind_direction: np.ndarray,
    window_index: int,
    max_missing_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    window_samples = int(WIND_TIME_WINDOW * WIND_FS)
    start_idx = int(window_index) * window_samples
    end_idx = start_idx + window_samples
    data_len = int(wind_speed.size)
    if data_len <= 0 or window_samples <= 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), 0.0
    clipped_start = max(0, min(start_idx, data_len))
    clipped_end = max(0, min(end_idx, data_len))
    available = max(0, clipped_end - clipped_start)
    coverage = float(available / max(window_samples, 1))
    if coverage < 1.0 - float(max_missing_ratio):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), coverage
    speed = np.asarray(wind_speed[clipped_start:clipped_end], dtype=np.float32)
    valid = speed > WIND_VALID_THRESHOLD
    speed_valid = speed[valid]
    direction_valid = np.asarray([], dtype=np.float32)
    if wind_direction.size >= clipped_end:
        direction_valid = np.asarray(wind_direction[clipped_start:clipped_end], dtype=np.float32)[valid]
    return speed_valid, direction_valid, coverage


def build_short_wind_features(record: dict, cfg: Optional[dict]) -> np.ndarray:
    if not cfg:
        return zero_wind_features()
    lookup_record = _record_for_wind_lookup(record)
    if not lookup_record.get("timestamp"):
        return zero_wind_features()
    wind_meta = resolve_wind_meta(lookup_record, cfg)
    if wind_meta is None:
        return zero_wind_features()

    parsed = parse_single_metadata_to_wind_data(wind_meta, enable_denoise=False)
    data = parsed.get("data") or {}
    wind_speed = np.asarray(data.get("wind_speed", []), dtype=np.float32)
    wind_direction = np.asarray(data.get("wind_direction", []), dtype=np.float32)
    speed_valid, direction_valid, coverage = _window_arrays(
        wind_speed,
        wind_direction,
        int(lookup_record.get("window_index", 0)),
        max_missing_ratio=float(cfg.get("wind_window_max_missing_ratio", 0.3)),
    )
    if speed_valid.size == 0:
        return zero_wind_features()

    avg_speed = float(np.mean(speed_valid))
    speed_std = float(np.std(speed_valid, ddof=1)) if speed_valid.size >= 2 else 0.0
    turbulence = speed_std / max(avg_speed, 1e-6)
    direction_sin = 0.0
    direction_cos = 0.0
    if direction_valid.size:
        sensor_id = str(lookup_record.get("inplane_sensor_id", ""))
        correction = float(WIND_DIR_CORRECTION.get(sensor_id, 360.0))
        corrected = np.mod(correction - direction_valid, 360.0)
        avg_direction = _mean_wind_direction(corrected)
        if avg_direction is not None:
            theta = np.deg2rad(avg_direction)
            direction_sin = float(np.sin(theta))
            direction_cos = float(np.cos(theta))

    return np.asarray(
        [
            avg_speed / 50.0,
            turbulence,
            direction_sin,
            direction_cos,
            float(coverage),
        ],
        dtype=np.float32,
    )
