from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

PairKey = Tuple[str, str, int, int, int, int]

_SENSOR_ID_RE = re.compile(r"(ST-VIC-[A-Z0-9]+-[0-9]+)-([0-9]+)")
_PAIR_KEY_SEP = "|"
PAIR_KEY_FIELD_COUNT = 6


def sensor_id_from_path(file_path: str) -> str:
    match = _SENSOR_ID_RE.search(str(file_path))
    if match is None:
        raise ValueError(f"无法从路径解析 sensor_id：{file_path}")
    return f"{match.group(1)}-{match.group(2)}"


def sensor_suffix_from_path(file_path: str) -> str:
    match = _SENSOR_ID_RE.search(str(file_path))
    if match is None:
        raise ValueError(f"无法从路径解析 sensor suffix：{file_path}")
    return match.group(2)


def cable_base_from_sensor_id(sensor_id: str) -> str:
    parts = str(sensor_id).rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(f"无法解析 cable base：{sensor_id}")
    return parts[0]


def time_axis_from_path(file_path: str) -> tuple[int, int, int]:
    path = Path(file_path)
    parts = path.parts
    if len(parts) < 2:
        raise ValueError(f"无法从路径解析时间轴：{file_path}")
    month = int(parts[-3])
    day = int(parts[-2])
    stem = path.stem
    time_token = stem.rsplit("_", 1)[-1]
    hour = int(time_token[0:2]) if len(time_token) >= 2 else 0
    return month, day, hour


def is_canonical_pair(inplane_file_path: str, outplane_file_path: str) -> bool:
    in_suffix = sensor_suffix_from_path(inplane_file_path)
    out_suffix = sensor_suffix_from_path(outplane_file_path)
    if in_suffix != "01" or out_suffix != "02":
        return False
    in_base = cable_base_from_sensor_id(sensor_id_from_path(inplane_file_path))
    out_base = cable_base_from_sensor_id(sensor_id_from_path(outplane_file_path))
    return in_base == out_base


def pair_key_from_paths(
    inplane_file_path: str,
    outplane_file_path: str,
    window_index: int,
) -> PairKey:
    in_sensor_id = sensor_id_from_path(inplane_file_path)
    out_sensor_id = sensor_id_from_path(outplane_file_path)
    month, day, hour = time_axis_from_path(inplane_file_path)
    out_month, out_day, out_hour = time_axis_from_path(outplane_file_path)
    if (month, day, hour) != (out_month, out_day, out_hour):
        raise ValueError(
            "面内/面外文件时间轴不一致："
            f"in={inplane_file_path} out={outplane_file_path}"
        )
    return (
        in_sensor_id,
        out_sensor_id,
        int(month),
        int(day),
        int(hour),
        int(window_index),
    )


def pair_key_from_entry(entry: dict) -> PairKey:
    pair_key = entry.get("pair_key")
    if pair_key is not None:
        return pair_key_from_list(pair_key)
    in_fp = entry.get("inplane_file_path") or entry.get("file_path")
    out_fp = entry.get("outplane_file_path")
    wi = int(entry.get("window_index", 0))
    if not in_fp or not out_fp:
        raise ValueError("entry 缺少 inplane/outplane file path，无法构造 pair_key")
    return pair_key_from_paths(str(in_fp), str(out_fp), wi)


def pair_key_to_list(key: PairKey) -> list[object]:
    return [
        str(key[0]),
        str(key[1]),
        int(key[2]),
        int(key[3]),
        int(key[4]),
        int(key[5]),
    ]


def pair_key_from_list(values: list[object] | tuple[object, ...]) -> PairKey:
    if len(values) != PAIR_KEY_FIELD_COUNT:
        raise ValueError(f"pair_key 长度必须为 {PAIR_KEY_FIELD_COUNT}，实际 {len(values)}")
    return (
        str(values[0]),
        str(values[1]),
        int(values[2]),
        int(values[3]),
        int(values[4]),
        int(values[5]),
    )


def pair_key_to_str(key: PairKey) -> str:
    return _PAIR_KEY_SEP.join(
        [
            str(key[0]),
            str(key[1]),
            str(key[2]),
            str(key[3]),
            str(key[4]),
            str(key[5]),
        ]
    )


def pair_key_from_str(text: str) -> PairKey:
    parts = str(text).split(_PAIR_KEY_SEP)
    if len(parts) != PAIR_KEY_FIELD_COUNT:
        raise ValueError(f"无法解析 pair_key 字符串：{text}")
    return pair_key_from_list(parts)


def pair_type_from_paths(inplane_file_path: str, outplane_file_path: str) -> str:
    in_suffix = sensor_suffix_from_path(inplane_file_path)
    out_suffix = sensor_suffix_from_path(outplane_file_path)
    return f"{in_suffix}_{out_suffix}"


def derive_outplane_path_from_inplane(inplane_file_path: str) -> str:
    path = Path(inplane_file_path)
    name = path.name.replace("-01_", "-02_")
    if name == path.name:
        raise ValueError(f"无法从面内路径推导面外路径：{inplane_file_path}")
    return str(path.with_name(name))
