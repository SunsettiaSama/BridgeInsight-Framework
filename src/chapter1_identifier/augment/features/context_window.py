from __future__ import annotations

from dataclasses import dataclass

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


def extract_context_window(
    file_path: str,
    window_index: int,
    window_size: int = 3000,
    fs: float = 50.0,
    before: int = 3,
    after: int = 3,
    extractor: VICWindowExtractor | None = None,
) -> ContextWindowResult:
    if extractor is None:
        extractor = VICWindowExtractor(enable_denoise=False)
    vic_data = extractor.load_file(file_path)
    total_len = len(vic_data)
    max_window_idx = max(0, total_len // window_size - 1)

    start_win = max(0, int(window_index) - before)
    end_win = min(max_window_idx, int(window_index) + after)

    start_sample = start_win * window_size
    end_sample = min(total_len, (end_win + 1) * window_size)
    long_signal = np.asarray(vic_data[start_sample:end_sample], dtype=np.float32)

    current_start_s = (int(window_index) - start_win) * window_size / fs
    current_end_s = current_start_s + window_size / fs

    return ContextWindowResult(
        signal=long_signal,
        start_window_idx=start_win,
        end_window_idx=end_win,
        current_window_idx=int(window_index),
        window_size=window_size,
        fs=fs,
        t0_offset_s=start_sample / fs,
        current_start_s=current_start_s,
        current_end_s=current_end_s,
    )
