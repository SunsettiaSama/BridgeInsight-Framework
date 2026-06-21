from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.config.sensor_config import VIB_TO_WIND_SENSOR_MAP
from src.data_infra.config import DATASET_REGISTRY, load_dataset_yaml, load_mysql_config
from src.data_infra.models import MetadataRow
from src.data_infra.mysql_client import MySQLClient
from src.data_infra.repository import MetadataRepository

_CORE_KEYS = frozenset({"month", "day", "hour", "sensor_id", "file_path"})
_HOUR_PATTERN = re.compile(r"_(\d{2})\d{2}\d{2}")


def _require_int(row: dict, key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ValueError(f"元数据缺少字段 {key}: {row}")
    return int(value)


def _extract_hour_from_file_path(row: dict) -> int:
    file_path = str(row.get("file_path") or row.get("path") or "")
    name = Path(file_path).name
    matched = _HOUR_PATTERN.search(name)
    if not matched:
        raise ValueError(f"无法从文件名解析小时字段: {file_path}")
    return int(matched.group(1))


def _split_metadata_row(row: dict, dataset_tag: str, year: int) -> MetadataRow:
    if not isinstance(row, dict):
        raise ValueError(f"元数据行必须是 dict: {row}")
    month = _require_int(row, "month")
    day = _require_int(row, "day")
    hour_value = row.get("hour")
    hour = _extract_hour_from_file_path(row) if hour_value is None else int(hour_value)
    sensor_id = row.get("sensor_id")
    file_path = row.get("file_path")
    if not sensor_id:
        raise ValueError(f"元数据缺少 sensor_id: {row}")
    if not file_path:
        raise ValueError(f"元数据缺少 file_path: {row}")
    raw_payload = {k: v for k, v in row.items() if k not in _CORE_KEYS}
    return MetadataRow(
        dataset_tag=dataset_tag,
        year=int(year),
        month=month,
        day=day,
        hour=hour,
        sensor_id=str(sensor_id),
        file_path=str(file_path),
        raw_metadata_json=raw_payload or None,
    )


def _load_json_rows(path: str | Path) -> List[dict]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"元数据文件不存在: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"元数据格式错误，期望 list: {json_path}")
    return payload


def _chunked(items: Sequence[MetadataRow], size: int) -> Iterable[Sequence[MetadataRow]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ingest_dataset(
    repository: MetadataRepository,
    dataset_key: str,
    batch_size: int = 1000,
) -> dict[str, int]:
    spec = DATASET_REGISTRY[dataset_key]
    ds_cfg = load_dataset_yaml(spec["yaml_path"])
    dataset_tag = str(spec["dataset_tag"])
    year = int(spec["year"])

    vib_path = ds_cfg.get("vib_metadata_path")
    wind_path = ds_cfg.get("wind_metadata_path")
    if not vib_path or not wind_path:
        raise ValueError(f"数据集 {dataset_key} 缺少 vib_metadata_path 或 wind_metadata_path")

    vib_rows = [_split_metadata_row(row, dataset_tag, year) for row in _load_json_rows(vib_path)]
    wind_rows = [_split_metadata_row(row, dataset_tag, year) for row in _load_json_rows(wind_path)]

    vib_written = 0
    for chunk in _chunked(vib_rows, batch_size):
        vib_written += repository.upsert_vib_rows(chunk)

    wind_written = 0
    for chunk in _chunked(wind_rows, batch_size):
        wind_written += repository.upsert_wind_rows(chunk)

    mapping_written = repository.seed_vib_to_wind_mapping(VIB_TO_WIND_SENSOR_MAP)

    return {
        "dataset_tag": dataset_tag,
        "vib_rows": len(vib_rows),
        "wind_rows": len(wind_rows),
        "vib_written": vib_written,
        "wind_written": wind_written,
        "mapping_written": mapping_written,
        "vib_count": repository.count_rows("vib_metadata_hourly", dataset_tag),
        "wind_count": repository.count_rows("wind_metadata_hourly", dataset_tag),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Import vibration/wind metadata into MySQL")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_REGISTRY.keys()),
        action="append",
        help="Dataset key to ingest (2023 or 202409). Repeatable.",
    )
    parser.add_argument("--all", action="store_true", help="Ingest all registered datasets")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args(argv)

    dataset_keys = sorted(DATASET_REGISTRY.keys()) if args.all else (args.dataset or [])
    if not dataset_keys:
        raise SystemExit("请指定 --dataset 或 --all")

    mysql_cfg = load_mysql_config()
    client = MySQLClient(mysql_cfg)
    client.ping()
    repository = MetadataRepository(client)

    for key in dataset_keys:
        result = ingest_dataset(repository, key, batch_size=args.batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
