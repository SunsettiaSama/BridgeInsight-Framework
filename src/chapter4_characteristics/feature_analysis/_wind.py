import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.data_processer.preprocess.get_data_wind import parse_single_metadata_to_wind_data

logger = logging.getLogger(__name__)

_TimestampKey = Tuple[int, int, int]


def load_wind_metadata(wind_metadata_path: str) -> List[Dict]:
    with open(wind_metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_wind_lookup(
    wind_metadata: List[Dict],
) -> Dict[_TimestampKey, List[Dict]]:
    """
    将风元数据按 (month, day, hour) 分组，保留该时间戳下所有传感器记录。
    返回 {(month, day, hour): [meta_record, ...]}
    """
    lookup: Dict[_TimestampKey, List[Dict]] = {}
    for rec in wind_metadata:
        m = rec.get("month")
        d = rec.get("day")
        h = rec.get("hour")
        if m is None or d is None or h is None:
            continue
        key = (int(m), int(d), int(h))
        lookup.setdefault(key, []).append(rec)
    return lookup


def _compute_turbulence_intensity(wind_speed: np.ndarray) -> Optional[float]:
    """
    紊流度 Iu = σ_u / Ū
    其中 σ_u 为风速标准差，Ū 为平均风速。
    平均风速接近零时返回 None。
    """
    U_mean = float(np.mean(wind_speed))
    if U_mean < 1e-6:
        return None
    U_std = float(np.std(wind_speed))
    return U_std / U_mean


def compute_wind_stats_for_sensor(wind_meta: Dict) -> Optional[Dict]:
    """
    对单个传感器的风元数据记录加载并计算统计量。
    返回含各维度均值/标准差及紊流度的字典，失败则返回 None。
    """
    result = parse_single_metadata_to_wind_data(wind_meta, enable_denoise=False)
    data = result.get("data")
    if data is None:
        return None

    U   = data["wind_speed"]
    D   = data["wind_direction"]
    A   = data["wind_attack_angle"]

    return {
        "sensor_id":              wind_meta.get("sensor_id", ""),
        "mean_wind_speed":        float(np.mean(U)),
        "std_wind_speed":         float(np.std(U)),
        "mean_wind_direction":    float(np.mean(D)),
        "std_wind_direction":     float(np.std(D)),
        "mean_wind_attack_angle": float(np.mean(A)),
        "std_wind_attack_angle":  float(np.std(A)),
        "turbulence_intensity":   _compute_turbulence_intensity(U),
    }


def compute_wind_stats_by_timestamp(
    wind_lookup: Dict[_TimestampKey, List[Dict]],
) -> Dict[_TimestampKey, List[Dict]]:
    """
    预先为所有时间戳计算各传感器风统计量。
    返回 {(month, day, hour): [sensor_stats_dict, ...]}
    """
    result: Dict[_TimestampKey, List[Dict]] = {}

    for ts_key, sensor_records in tqdm(
        wind_lookup.items(),
        total=len(wind_lookup),
        desc="风统计量计算",
        unit="时间戳",
        dynamic_ncols=True,
    ):
        stats_list = []
        for rec in sensor_records:
            stats = compute_wind_stats_for_sensor(rec)
            if stats is not None:
                stats_list.append(stats)
        result[ts_key] = stats_list

    return result


def _serialize_wind_stats(stats_by_ts: Dict[_TimestampKey, List[Dict]]) -> list[dict]:
    return [
        {"timestamp": list(ts_key), "stats": stats}
        for ts_key, stats in sorted(stats_by_ts.items())
    ]


def _deserialize_wind_stats(rows: list[dict]) -> Dict[_TimestampKey, List[Dict]]:
    out: Dict[_TimestampKey, List[Dict]] = {}
    for row in rows:
        ts = row.get("timestamp") or []
        if len(ts) != 3:
            continue
        out[(int(ts[0]), int(ts[1]), int(ts[2]))] = list(row.get("stats") or [])
    return out


def load_or_compute_wind_stats_by_timestamp(
    wind_metadata_path: str,
    cache_path: str,
) -> Dict[_TimestampKey, List[Dict]]:
    meta_path = Path(wind_metadata_path)
    cache = Path(cache_path)
    source_stat = meta_path.stat()

    if cache.exists():
        with open(cache, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            payload.get("source_path") == str(meta_path.resolve())
            and payload.get("source_mtime") == source_stat.st_mtime
            and payload.get("source_size") == source_stat.st_size
        ):
            logger.info(f"风统计缓存命中：{cache}")
            return _deserialize_wind_stats(payload.get("items") or [])

    wind_meta_list = load_wind_metadata(str(meta_path))
    logger.info(f"风元数据共 {len(wind_meta_list)} 条记录")
    wind_lookup = build_wind_lookup(wind_meta_list)
    logger.info(f"去重后共 {len(wind_lookup)} 个时间戳")
    stats_by_ts = compute_wind_stats_by_timestamp(wind_lookup)

    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(meta_path.resolve()),
        "source_mtime": source_stat.st_mtime,
        "source_size": source_stat.st_size,
        "items": _serialize_wind_stats(stats_by_ts),
    }
    tmp_path = cache.with_name(f"{cache.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp_path.replace(cache)
    logger.info(f"风统计缓存已写入：{cache}")
    return stats_by_ts


def get_wind_stats_for_sample(
    timestamp: List[int],
    wind_stats_by_ts: Dict[_TimestampKey, List[Dict]],
) -> List[Dict]:
    """
    获取某样本时间戳对应的所有传感器风统计量列表。
    """
    ts_key = tuple(timestamp[:3])
    return wind_stats_by_ts.get(ts_key, [])


def compute_reduced_velocity(
    mean_wind_speed: float,
    dominant_freq_hz: float,
    cable_diameter: float,
) -> Optional[float]:
    """
    折减风速 Vr = U / (f₁ × D)

    Parameters
    ----------
    mean_wind_speed  : 平均风速 U（m/s）
    dominant_freq_hz : 振动主频 f₁（Hz）
    cable_diameter   : 拉索外径 D（m）

    Returns
    -------
    float | None
        折减风速，当 f₁ 或 D 过小时返回 None。
    VIV 通常发生在 Vr ≈ 5~12 区间。
    """
    if dominant_freq_hz < 1e-6 or cable_diameter < 1e-6:
        return None
    return float(mean_wind_speed / (dominant_freq_hz * cable_diameter))
