from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from src.chapter3_identifier.augment._bootstrap import resolve_path
from src.chapter3_identifier.augment.settings import load_yaml
from src.config.sensor_config import DEFAULT_WIND_SENSORS, VIB_TO_WIND_SENSOR_MAP
from src.data_infra.config import load_mysql_config
from src.data_infra.mysql_client import MySQLClient
from src.data_infra.repository import MetadataRepository

_TimestampKey = Tuple[int, int, int]
_TimestampSensorKey = Tuple[int, int, int, str]
_MIDSPAN_WIND_SENSOR = "ST-UAN-G04-001-01"
_MIDSPAN_WIND_FALLBACK = "ST-UAN-G04-002-01"
_DATASET_YEAR = {"2023": 2023, "202409": 2024}
_REPOSITORY_CACHE: Dict[str, MetadataRepository] = {}
_FILE_INDEX_CACHE: Dict[str, WindMetadataIndex] = {}
_DEGRADE_MODE_CACHE: Dict[str, bool] = {}


class WindMetadataIndex:
    """文件索引（仅用于本地 smoke/单测）；生产查询走 MySQL。"""

    def __init__(self, metadata_path: str, sensor_ids: Optional[Sequence[str]] = None) -> None:
        path = resolve_path(metadata_path)
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"风元数据格式错误，期望 list: {path}")

        priority: Dict[str, int] = {}
        if sensor_ids is not None:
            priority = {sid: i for i, sid in enumerate(sensor_ids)}

        by_ts_sid: Dict[_TimestampSensorKey, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = _timestamp_key(row)
            if key is None:
                continue
            sid = str(row.get("sensor_id", ""))
            if priority and sid not in priority:
                continue
            by_ts_sid[(key[0], key[1], key[2], sid)] = row

        self._by_ts_sid = by_ts_sid
        self._metadata_path = str(path)

    @property
    def metadata_path(self) -> str:
        return self._metadata_path

    def lookup(self, month: int, day: int, hour: int, sensor_id: str) -> Optional[dict]:
        return self._by_ts_sid.get((int(month), int(day), int(hour), str(sensor_id)))


class MySQLWindMetadataIndex:
    def __init__(
        self,
        repository: MetadataRepository,
        dataset_tag: str,
        year: int,
        sensor_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self._repository = repository
        self._dataset_tag = dataset_tag
        self._year = int(year)
        self._sensor_ids = [str(sid) for sid in sensor_ids] if sensor_ids else None
        self._metadata_path = f"mysql://{dataset_tag}"

    @property
    def metadata_path(self) -> str:
        return self._metadata_path

    def lookup(self, month: int, day: int, hour: int, sensor_id: str) -> Optional[dict]:
        return self._repository.get_wind_meta_legacy(
            self._dataset_tag,
            self._year,
            int(month),
            int(day),
            int(hour),
            str(sensor_id),
        )


def resolve_wind_sensor_id(record: dict) -> str:
    inplane_id = str(record.get("inplane_sensor_id", ""))
    mapped = VIB_TO_WIND_SENSOR_MAP.get(inplane_id)
    if mapped:
        return str(mapped[0])
    return _MIDSPAN_WIND_SENSOR


def _timestamp_key(row: dict) -> Optional[_TimestampKey]:
    month = row.get("month")
    day = row.get("day")
    hour = row.get("hour")
    if month is None or day is None or hour is None:
        return None
    return int(month), int(day), int(hour)


def _dataset_wind_config(cfg: dict) -> tuple[Optional[str], Optional[List[str]], str, int]:
    ds_cfg_path = cfg.get("inference_dataset_config")
    dataset_tag = str(cfg.get("wind_dataset_tag", "202409"))
    if dataset_tag != "202409":
        raise ValueError(f"chapter3_identifier 仅允许 wind_dataset_tag=202409，当前为: {dataset_tag}")
    year = int(_DATASET_YEAR.get(dataset_tag, 2024))
    if not ds_cfg_path:
        return None, None, dataset_tag, year
    ds_path = resolve_path(str(ds_cfg_path))
    if not str(ds_path).replace("\\", "/").lower().endswith("total_staycable_vib_202409.yaml"):
        raise ValueError(
            f"chapter3_identifier 仅允许 2024-09 推理数据集配置，当前为: {ds_path}"
        )
    if not ds_path.exists():
        return None, None, dataset_tag, year
    ds_cfg = load_yaml(ds_path)
    wind_path = ds_cfg.get("wind_metadata_path")
    sensor_ids = ds_cfg.get("wind_sensor_ids")
    if wind_path:
        wind_path = str(resolve_path(str(wind_path)))
    if sensor_ids is not None:
        sensor_ids = [str(sid) for sid in sensor_ids]
    return wind_path, sensor_ids, dataset_tag, year


def _degrade_cache_key(cfg: dict) -> str:
    cfg_path = str(cfg.get("_config_path", "default"))
    mysql_cfg = cfg.get("mysql") or {}
    host = str(mysql_cfg.get("host", "127.0.0.1"))
    port = str(mysql_cfg.get("port", "3306"))
    database = str(mysql_cfg.get("database", "vibration_data"))
    return f"{cfg_path}|{host}:{port}/{database}"


def should_degrade_to_dict_mode(cfg: dict) -> bool:
    if bool(cfg.get("wind_force_dict_mode", False)):
        return True
    return bool(_DEGRADE_MODE_CACHE.get(_degrade_cache_key(cfg), False))


def _set_degrade_to_dict_mode(cfg: dict) -> None:
    _DEGRADE_MODE_CACHE[_degrade_cache_key(cfg)] = True


def get_metadata_repository(cfg: dict) -> MetadataRepository:
    mysql_cfg = load_mysql_config(cfg)
    cache_key = f"{mysql_cfg.host}:{mysql_cfg.port}/{mysql_cfg.database}"
    cached = _REPOSITORY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    client = MySQLClient(mysql_cfg)
    client.ping()
    repository = MetadataRepository(client)
    _REPOSITORY_CACHE[cache_key] = repository
    return repository


def _file_sensor_candidates(record: dict, wind_sensor_ids: Optional[Sequence[str]]) -> List[str]:
    mapped = [str(sid) for sid in VIB_TO_WIND_SENSOR_MAP.get(str(record.get("inplane_sensor_id", "")), [])]
    if not mapped:
        mapped = [resolve_wind_sensor_id(record), _MIDSPAN_WIND_FALLBACK]
    ordered: List[str] = []
    seen = set()
    for sid in mapped:
        if sid not in seen:
            ordered.append(sid)
            seen.add(sid)
    if wind_sensor_ids:
        allowed = [str(sid) for sid in wind_sensor_ids]
        preferred = [sid for sid in ordered if sid in allowed]
        if preferred:
            ordered = preferred
        else:
            for sid in allowed:
                if sid not in seen:
                    ordered.append(sid)
                    seen.add(sid)
    return ordered


def get_wind_index(cfg: dict) -> Optional[MySQLWindMetadataIndex | WindMetadataIndex]:
    wind_path, sensor_ids, dataset_tag, year = _dataset_wind_config(cfg)
    if should_degrade_to_dict_mode(cfg):
        if not wind_path:
            return None
        return _get_file_wind_index(wind_path, sensor_ids)
    try:
        repository = get_metadata_repository(cfg)
    except Exception:
        _set_degrade_to_dict_mode(cfg)
        if not wind_path:
            return None
        return _get_file_wind_index(wind_path, sensor_ids)
    return MySQLWindMetadataIndex(repository, dataset_tag, year, sensor_ids)


def _get_file_wind_index(wind_path: str, sensor_ids: Optional[Sequence[str]]) -> WindMetadataIndex:
    key = f"{wind_path}|{','.join(sensor_ids or [])}"
    cached = _FILE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    index = WindMetadataIndex(wind_path, sensor_ids=sensor_ids)
    _FILE_INDEX_CACHE[key] = index
    return index


def _wind_sensor_candidates(
    record: dict,
    repository: MetadataRepository,
    wind_sensor_ids: Optional[Sequence[str]],
) -> List[str]:
    inplane_id = str(record.get("inplane_sensor_id", ""))
    mapped = [str(sid) for sid in VIB_TO_WIND_SENSOR_MAP.get(inplane_id, [])]
    if not mapped:
        mapped = repository.get_wind_sensor_candidates(inplane_id)
    if not mapped:
        mapped = [resolve_wind_sensor_id(record), _MIDSPAN_WIND_FALLBACK]

    ordered: List[str] = []
    seen = set()
    for sid in mapped:
        if sid not in seen:
            ordered.append(sid)
            seen.add(sid)

    if wind_sensor_ids:
        allowed = [str(sid) for sid in wind_sensor_ids]
        preferred = [sid for sid in ordered if sid in allowed]
        if preferred:
            ordered = preferred
        else:
            for sid in allowed:
                if sid not in seen:
                    ordered.append(sid)
                    seen.add(sid)
    return ordered


def resolve_wind_meta(record: dict, cfg: dict) -> Optional[dict]:
    timestamp = record.get("timestamp")
    if not timestamp or len(timestamp) < 3:
        return None
    wind_path, wind_sensor_ids, dataset_tag, year = _dataset_wind_config(cfg)
    month, day, hour = int(timestamp[0]), int(timestamp[1]), int(timestamp[2])
    if should_degrade_to_dict_mode(cfg):
        if not wind_path:
            return None
        file_index = _get_file_wind_index(wind_path, wind_sensor_ids)
        candidates = _file_sensor_candidates(record, wind_sensor_ids)
        for sid in candidates:
            matched = file_index.lookup(month, day, hour, sid)
            if matched is not None:
                return matched
        return None

    try:
        repository = get_metadata_repository(cfg)
    except Exception:
        _set_degrade_to_dict_mode(cfg)
        if not wind_path:
            return None
        file_index = _get_file_wind_index(wind_path, wind_sensor_ids)
        candidates = _file_sensor_candidates(record, wind_sensor_ids)
        for sid in candidates:
            matched = file_index.lookup(month, day, hour, sid)
            if matched is not None:
                return matched
        return None
    candidates = _wind_sensor_candidates(record, repository, wind_sensor_ids)
    resolved = repository.resolve_wind_meta_with_priority(
        dataset_tag,
        year,
        month,
        day,
        hour,
        candidates,
    )
    if resolved is not None:
        return resolved

    all_rows = repository.list_wind_meta_at_timestamp(dataset_tag, year, month, day, hour, None)
    if not all_rows:
        if not wind_path:
            return None
        file_index = _get_file_wind_index(wind_path, wind_sensor_ids)
        for sid in candidates:
            matched = file_index.lookup(month, day, hour, sid)
            if matched is not None:
                return matched
        return None

    available = {row.sensor_id: row for row in all_rows}
    fallback_order: List[str] = []
    for sid in candidates:
        if sid not in fallback_order:
            fallback_order.append(sid)
    if wind_sensor_ids:
        for sid in wind_sensor_ids:
            ss = str(sid)
            if ss not in fallback_order:
                fallback_order.append(ss)
    for sid in DEFAULT_WIND_SENSORS:
        ss = str(sid)
        if ss not in fallback_order:
            fallback_order.append(ss)
    for sid in sorted(available.keys()):
        if sid not in fallback_order:
            fallback_order.append(sid)

    for sid in fallback_order:
        row = available.get(sid)
        if row is not None:
            return repository.to_legacy_wind_dict(row)
    return repository.to_legacy_wind_dict(all_rows[0])
