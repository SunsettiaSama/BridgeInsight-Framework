from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


_SENSOR_FILE_CACHE: Dict[Tuple[str, str], List[Path]] = {}


def _sensor_prefix(file_path: str) -> str:
    stem = Path(file_path).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def _sensor_search_root(file_path: str) -> Path:
    p = Path(file_path).resolve()
    if len(p.parents) >= 3:
        return p.parents[2]
    return p.parent


def _sensor_files(file_path: str) -> List[Path]:
    sensor = _sensor_prefix(file_path)
    root = _sensor_search_root(file_path)
    cache_key = (str(root), sensor)
    cached = _SENSOR_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    files = sorted(root.rglob(f"{sensor}_*.VIC"), key=lambda x: str(x).replace("\\", "/"))
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

    if allow_cross_file and (left_missing > 0 or right_missing > 0):
        files = _sensor_files(file_path)
        current = Path(file_path).resolve()
        if current in files:
            idx = files.index(current)
            if left_missing > 0:
                for prev_path in reversed(files[:idx]):
                    prev_data = np.asarray(extractor.load_file(str(prev_path)), dtype=np.float32)
                    take = min(left_missing, int(len(prev_data)))
                    if take <= 0:
                        continue
                    left_parts.insert(0, prev_data[-take:])
                    left_missing -= take
                    if left_missing <= 0:
                        break
            if right_missing > 0:
                for next_path in files[idx + 1:]:
                    next_data = np.asarray(extractor.load_file(str(next_path)), dtype=np.float32)
                    take = min(right_missing, int(len(next_data)))
                    if take <= 0:
                        continue
                    right_parts.append(next_data[:take])
                    right_missing -= take
                    if right_missing <= 0:
                        break

    long_signal = np.concatenate(left_parts + right_parts).astype(np.float32, copy=False)
    current_start_s = sum(int(len(part)) for part in left_parts) / fs
    current_end_s = current_start_s + window_size / fs

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
    )
