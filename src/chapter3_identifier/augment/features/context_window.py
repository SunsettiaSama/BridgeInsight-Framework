from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_processer.preprocess.get_data_vib import VICWindowExtractor


@dataclass
class ContextWindowResult:
    signal: np.ndarray
    start_window_idx: int
    end_window_idx: int
    current_window_idx: int
    window_size: int
    fs: float
    t0_offset_s: float
    current_start_s: float
    current_end_s: float
    discontinuity_note: Optional[str] = None


_SENSOR_FILE_CACHE: Dict[Tuple[str, str], List[Path]] = {}
_FILE_DURATION_CACHE: Dict[str, float] = {}


def _sensor_prefix(file_path: str) -> str:
    stem = Path(file_path).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def _sensor_search_root(file_path: str) -> Path:
    p = Path(file_path).resolve()
    if len(p.parents) >= 3:
        return p.parents[2]
    return p.parent


def _parse_vic_start_datetime(file_path: str) -> datetime:
    path = Path(file_path)
    parts = path.parts
    month = int(parts[-3]) if len(parts) >= 3 else 1
    day = int(parts[-2]) if len(parts) >= 2 else 1
    stem = path.stem
    time_token = stem.rsplit("_", 1)[-1]
    hour = int(time_token[0:2]) if len(time_token) >= 2 else 0
    minute = int(time_token[2:4]) if len(time_token) >= 4 else 0
    second = int(time_token[4:6]) if len(time_token) >= 6 else 0
    return datetime(2024, month, day, hour, minute, second)


def _file_duration_seconds(file_path: str, sample_count: int, fs: float) -> float:
    return float(sample_count) / fs if fs > 0 else 0.0


def _files_are_time_adjacent(
    prev_path: str,
    prev_sample_count: int,
    next_path: str,
    fs: float,
    tolerance_s: float = 2.0,
) -> bool:
    prev_start = _parse_vic_start_datetime(prev_path)
    prev_end = prev_start + timedelta(seconds=_file_duration_seconds(prev_path, prev_sample_count, fs))
    next_start = _parse_vic_start_datetime(next_path)
    gap_s = abs((next_start - prev_end).total_seconds())
    return gap_s <= tolerance_s


def _sensor_files(file_path: str) -> List[Path]:
    sensor = _sensor_prefix(file_path)
    root = _sensor_search_root(file_path)
    cache_key = (str(root), sensor)
    cached = _SENSOR_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    files = list(root.rglob(f"{sensor}_*.VIC"))
    files.sort(key=lambda p: _parse_vic_start_datetime(str(p)))
    _SENSOR_FILE_CACHE[cache_key] = files
    return files


def extract_context_window(
    file_path: str,
    window_index: int,
    window_size: int = 3000,
    fs: float = 50.0,
    before: int = 3,
    after: int = 3,
    extractor: VICWindowExtractor | None = None,
    allow_cross_file: bool = True,
) -> ContextWindowResult:
    if extractor is None:
        extractor = VICWindowExtractor(enable_denoise=False)
    vic_data = np.asarray(extractor.load_file(file_path), dtype=np.float32)
    total_len = int(len(vic_data))
    max_window_idx = max(0, total_len // window_size - 1)

    center_win = int(window_index)
    start_win = max(0, center_win - before)
    end_win = min(max_window_idx, center_win + after)

    center_start = max(0, center_win * window_size)
    center_end = min(total_len, center_start + window_size)
    center_len = max(0, center_end - center_start)
    left_need = max(0, before * window_size)
    right_need = max(0, after * window_size)

    left_current_start = max(0, center_start - left_need)
    left_current = vic_data[left_current_start:center_start]
    left_parts = [left_current]
    left_missing = left_need - int(len(left_current))

    right_current_end = min(total_len, center_end + right_need)
    center_and_right = vic_data[center_start:right_current_end]
    right_missing = right_need - max(0, right_current_end - center_end)
    right_parts: List[np.ndarray] = [center_and_right]

    discontinuity_notes: List[str] = []

    if allow_cross_file and (left_missing > 0 or right_missing > 0):
        files = _sensor_files(file_path)
        current = Path(file_path).resolve()
        if current in files:
            idx = files.index(current)
            if left_missing > 0 and idx > 0:
                prev_path = files[idx - 1]
                try:
                    prev_data = np.asarray(extractor.load_file(str(prev_path)), dtype=np.float32)
                except RuntimeError as exc:
                    discontinuity_notes.append(f"左侧跨文件读取失败，已停止拼接：{exc}")
                else:
                    if _files_are_time_adjacent(
                        str(prev_path),
                        int(len(prev_data)),
                        str(current),
                        fs,
                    ):
                        take = min(left_missing, int(len(prev_data)))
                        if take > 0:
                            left_parts.insert(0, prev_data[-take:])
                            left_missing -= take
                    else:
                        discontinuity_notes.append("左侧跨文件时间不连续，已停止拼接")
            elif left_missing > 0:
                discontinuity_notes.append("左侧上下文不足")

            if right_missing > 0 and idx + 1 < len(files):
                next_path = files[idx + 1]
                try:
                    next_data = np.asarray(extractor.load_file(str(next_path)), dtype=np.float32)
                except RuntimeError as exc:
                    discontinuity_notes.append(f"右侧跨文件读取失败，已停止拼接：{exc}")
                else:
                    if _files_are_time_adjacent(
                        str(current),
                        total_len,
                        str(next_path),
                        fs,
                    ):
                        take = min(right_missing, int(len(next_data)))
                        if take > 0:
                            right_parts.append(next_data[:take])
                            right_missing -= take
                    else:
                        discontinuity_notes.append("右侧跨文件时间不连续，已停止拼接")
            elif right_missing > 0:
                discontinuity_notes.append("右侧上下文不足")

    long_signal = np.concatenate(left_parts + right_parts).astype(np.float32, copy=False)
    current_start_s = sum(int(len(part)) for part in left_parts) / fs
    current_end_s = current_start_s + center_len / fs
    note = "；".join(discontinuity_notes) if discontinuity_notes else None

    return ContextWindowResult(
        signal=long_signal,
        start_window_idx=start_win,
        end_window_idx=end_win,
        current_window_idx=center_win,
        window_size=window_size,
        fs=fs,
        t0_offset_s=-current_start_s,
        current_start_s=current_start_s,
        current_end_s=current_end_s,
        discontinuity_note=note,
    )
