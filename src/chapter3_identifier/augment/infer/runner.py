from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.chapter3_identifier.augment.datasets.dual_stream_dataset import (
    context_side_windows,
    load_context_array,
)
from src.chapter3_identifier.augment.features.context_window import (
    _files_are_time_adjacent,
    _sensor_files,
)
from src.chapter3_identifier.augment.features.wind_features import build_short_wind_features
from src.chapter3_identifier.augment.infer.batch_ops import (
    AsyncJointBatchPipeline,
    AsyncPsdPipeline,
    PreparedJointBatch,
    compute_psd_batch,
    normalize_time_batch,
)
from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.prefetch import (
    BoundedVicFileCache,
    PrefetchScheduler,
    PrefetchStats,
    upcoming_batch_file_paths,
)
from src.chapter3_identifier.augment.infer.profile_hooks import track_gpu_forward
from src.chapter3_identifier.identifier.dl.progress import iter_progress
from src.identifier.dl.runner import FullDatasetRunner

logger = logging.getLogger(__name__)

_NORMAL_LABEL = 0
_DUAL_HEAD_MODEL_TYPES = {
    "quad_stream_dual_head",
    "quad_stream_dual_head_context",
    "quad_stream_serial_context_dual_head",
}
_LONG_CONTEXT_MODEL_TYPES = {
    "quad_stream_dual_head_context",
    "quad_stream_serial_context_dual_head",
}


def _effective_joint_num_workers(configured_workers: int, sample_count: int) -> int:
    workers = max(0, int(configured_workers))
    if os.name == "nt" and workers > 0:
        logger.warning(
            "Windows 联合双头推理禁用 DataLoader 多进程：configured_workers=%s, "
            "paired_samples=%s。原因：spawn 会复制大体量 sample metadata，容易触发 MemoryError。",
            workers,
            sample_count,
        )
        return 0
    return workers


def _collate_windows(batch):
    times, idxs = zip(*batch)
    return torch.stack(times), list(idxs)


class _DirectionWindowDataset(Dataset):
    def __init__(
        self,
        records,
        direction: str,
        window_size: int,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
        file_cache: Optional[BoundedVicFileCache] = None,
        profile_stats: Optional[PrefetchStats] = None,
    ):
        self._records = records
        self._direction = direction
        self._window_size = window_size
        self._enable_denoise = enable_denoise
        self._original_indices = original_indices or list(range(len(records)))
        self._file_cache = file_cache
        self._profile_stats = profile_stats
        self._extractor = None
        self._cache_path = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self._records)

    def _load_file(self, file_path: str):
        if self._file_cache is not None:
            return self._file_cache.get_or_load(file_path, self._extractor.load_file)
        if file_path != self._cache_path:
            self._cache_data = self._extractor.load_file(file_path)
            self._cache_path = file_path
        return self._cache_data

    def __getitem__(self, idx: int):
        if self._extractor is None:
            from src.identifier.dl.runner import _make_extractor

            self._extractor = _make_extractor(enable_denoise=self._enable_denoise)

        rec = self._records[idx]
        meta_key = f"{self._direction}_meta"
        meta = getattr(rec, meta_key) or {}
        file_path = meta.get("file_path")
        t0 = time.perf_counter()
        vic_data = self._load_file(file_path)
        signal = self._extractor.extract_window_from_data(
            vic_data,
            rec.window_idx,
            self._window_size,
            metadata=meta,
            file_path=file_path,
        )
        if self._profile_stats is not None:
            self._profile_stats.window_slice_time_s += time.perf_counter() - t0
        time_sig = torch.from_numpy(np.asarray(signal, dtype=np.float32).reshape(-1)).float()
        return time_sig.unsqueeze(-1), self._original_indices[idx]


class _CachedContextExtractor:
    def __init__(self, extractor, file_cache: BoundedVicFileCache):
        self._extractor = extractor
        self._file_cache = file_cache

    def load_file(self, file_path: str):
        return self._file_cache.get_or_load(file_path, self._extractor.load_file)


class _ContextFileBufferCache:
    """LRU cache for file-level long-context buffers used by full inference."""

    def __init__(
        self,
        max_entries: int,
        file_cache: Optional[BoundedVicFileCache] = None,
    ):
        self._max_entries = max(1, int(max_entries))
        self._file_cache = file_cache
        self._cache: OrderedDict[tuple, tuple[np.ndarray, int, int]] = OrderedDict()
        self._build_locks: Dict[tuple, threading.Lock] = {}
        self._resample_cache: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    def _load_file(self, file_path: str, extractor) -> np.ndarray:
        if self._file_cache is not None:
            data = self._file_cache.get_or_load(file_path, extractor.load_file)
        else:
            data = extractor.load_file(file_path)
        return np.asarray(data, dtype=np.float32).reshape(-1)

    def _cache_key(
        self,
        file_path: str,
        window_size: int,
        fs: float,
        before: int,
        after: int,
        allow_cross_file: bool,
    ) -> tuple:
        return (
            str(Path(file_path).resolve()),
            int(window_size),
            round(float(fs), 6),
            int(before),
            int(after),
            bool(allow_cross_file),
        )

    def _build_buffer(
        self,
        file_path: str,
        extractor,
        window_size: int,
        fs: float,
        before: int,
        after: int,
        allow_cross_file: bool,
    ) -> tuple[np.ndarray, int, int]:
        current = Path(file_path).resolve()
        current_data = self._load_file(str(current), extractor)
        total_len = int(current_data.size)
        left_need = max(0, int(before) * int(window_size))
        right_need = max(0, int(after) * int(window_size))
        left = np.empty(0, dtype=np.float32)
        right = np.empty(0, dtype=np.float32)

        if allow_cross_file and (left_need > 0 or right_need > 0):
            files = _sensor_files(str(current))
            if current in files:
                idx = files.index(current)
                if left_need > 0 and idx > 0:
                    prev_path = files[idx - 1]
                    try:
                        prev_data = self._load_file(str(prev_path), extractor)
                    except RuntimeError as exc:
                        logger.warning("左侧跨文件上下文读取失败，跳过邻接文件：%s (%s)", prev_path, exc)
                    else:
                        if _files_are_time_adjacent(
                            str(prev_path),
                            int(prev_data.size),
                            str(current),
                            fs,
                        ):
                            take = min(left_need, int(prev_data.size))
                            left = prev_data[-take:] if take > 0 else left
                if right_need > 0 and idx + 1 < len(files):
                    next_path = files[idx + 1]
                    try:
                        next_data = self._load_file(str(next_path), extractor)
                    except RuntimeError as exc:
                        logger.warning("右侧跨文件上下文读取失败，跳过邻接文件：%s (%s)", next_path, exc)
                    else:
                        if _files_are_time_adjacent(
                            str(current),
                            total_len,
                            str(next_path),
                            fs,
                        ):
                            take = min(right_need, int(next_data.size))
                            right = next_data[:take] if take > 0 else right

        if left.size or right.size:
            buffer = np.concatenate([left, current_data, right]).astype(np.float32, copy=False)
        else:
            buffer = current_data
        return buffer, int(left.size), total_len

    def get_context(
        self,
        file_path: str,
        window_index: int,
        extractor,
        window_size: int,
        fs: float,
        context_input_size: int,
        context_total_seconds: float,
        allow_cross_file: bool,
    ) -> np.ndarray:
        before = context_side_windows(context_total_seconds, window_size, fs)
        after = before
        key = self._cache_key(file_path, window_size, fs, before, after, allow_cross_file)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is None:
            with self._lock:
                build_lock = self._build_locks.get(key)
                if build_lock is None:
                    build_lock = threading.Lock()
                    self._build_locks[key] = build_lock
            with build_lock:
                with self._lock:
                    cached = self._cache.get(key)
                    if cached is not None:
                        self._cache.move_to_end(key)
                if cached is None:
                    cached = self._build_buffer(
                        file_path=file_path,
                        extractor=extractor,
                        window_size=window_size,
                        fs=fs,
                        before=before,
                        after=after,
                        allow_cross_file=allow_cross_file,
                    )
                    with self._lock:
                        self._cache[key] = cached
                        while len(self._cache) > self._max_entries:
                            evicted_key, _ = self._cache.popitem(last=False)
                            self._build_locks.pop(evicted_key, None)

        buffer, offset, total_len = cached
        center_win = int(window_index)
        center_start = max(0, center_win * int(window_size))
        center_end = min(int(total_len), center_start + int(window_size))
        left_need = max(0, int(before) * int(window_size))
        right_need = max(0, int(after) * int(window_size))
        start = max(0, int(offset) + center_start - left_need)
        end = min(int(buffer.size), int(offset) + center_end + right_need)
        return self._resample_or_pad(buffer[start:end], int(context_input_size))

    def _resample_or_pad(self, signal: np.ndarray, target_size: int) -> np.ndarray:
        target = max(1, int(target_size))
        arr = np.asarray(signal, dtype=np.float32).reshape(-1)
        n = int(arr.size)
        if n == target:
            return arr.astype(np.float32, copy=False)
        if n == 0:
            return np.zeros(target, dtype=np.float32)
        if n < target:
            out = np.zeros(target, dtype=np.float32)
            out[:n] = arr
            return out

        key = (n, target)
        with self._lock:
            cached = self._resample_cache.get(key)
        if cached is None:
            x = np.linspace(0, n - 1, target, dtype=np.float64)
            left_idx = np.floor(x).astype(np.int64)
            right_idx = np.minimum(left_idx + 1, n - 1).astype(np.int64)
            weight = (x - left_idx).astype(np.float32)
            cached = (left_idx, right_idx, weight)
            with self._lock:
                self._resample_cache[key] = cached
        left_idx, right_idx, weight = cached
        return (arr[left_idx] * (1.0 - weight) + arr[right_idx] * weight).astype(np.float32, copy=False)


class _PairedWindowDataset(Dataset):
    def __init__(
        self,
        records,
        window_size: int,
        fs: float = 50.0,
        enable_denoise: bool = False,
        original_indices: Optional[List[int]] = None,
        enable_long_context: bool = False,
        context_input_size: int = 3000,
        context_total_seconds: float = 1500.0,
        context_allow_cross_file: bool = True,
        enable_wind_features: bool = False,
        wind_config: Optional[dict] = None,
        file_cache: Optional[BoundedVicFileCache] = None,
        profile_stats: Optional[PrefetchStats] = None,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
        context_workers: int = 1,
        context_cache_entries: int = 32,
    ):
        self._records = records
        self._window_size = window_size
        self._fs = float(fs)
        self._nfft = int(nfft)
        self._freq_max_hz = float(freq_max_hz)
        self._enable_denoise = enable_denoise
        self._original_indices = original_indices or list(range(len(records)))
        self._enable_long_context = bool(enable_long_context)
        self._context_input_size = int(context_input_size)
        self._context_total_seconds = float(context_total_seconds)
        self._context_allow_cross_file = bool(context_allow_cross_file)
        self._enable_wind_features = bool(enable_wind_features)
        self._wind_config = dict(wind_config or {})
        self._file_cache = file_cache
        self._profile_stats = profile_stats
        self._context_workers = max(1, int(context_workers))
        self._context_cache = (
            _ContextFileBufferCache(
                max_entries=int(context_cache_entries),
                file_cache=file_cache,
            )
            if self._enable_long_context and int(context_cache_entries) > 0
            else None
        )
        self._extractor = None
        self._thread_local = threading.local()
        self._in_cache_path = None
        self._in_cache_data = None
        self._out_cache_path = None
        self._out_cache_data = None

    def __len__(self) -> int:
        return len(self._records)

    def _get_extractor(self):
        extractor = getattr(self._thread_local, "extractor", None)
        if extractor is None:
            from src.identifier.dl.runner import _make_extractor

            extractor = _make_extractor(enable_denoise=self._enable_denoise)
            self._thread_local.extractor = extractor
        return extractor

    def _load_file(self, file_path: str, side: str, extractor):
        if self._file_cache is not None:
            return self._file_cache.get_or_load(file_path, extractor.load_file)
        cache_path_attr = f"_{side}_cache_path"
        cache_data_attr = f"_{side}_cache_data"
        cache_path = getattr(self, cache_path_attr)
        if file_path != cache_path:
            data = extractor.load_file(file_path)
            setattr(self, cache_path_attr, file_path)
            setattr(self, cache_data_attr, data)
        return getattr(self, cache_data_attr)

    def _load_signal(
        self,
        file_path: str,
        window_idx: int,
        metadata: dict,
        side: str,
        extractor,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        vic_data = self._load_file(file_path, side, extractor)
        signal = extractor.extract_window_from_data(
            vic_data,
            window_idx,
            self._window_size,
            metadata=metadata,
            file_path=file_path,
        )
        if self._profile_stats is not None:
            self._profile_stats.window_slice_time_s += time.perf_counter() - t0
        if signal is None:
            raise ValueError(f"无法提取窗口：{file_path} idx={window_idx}")
        arr = np.asarray(signal, dtype=np.float32).reshape(-1)
        return torch.from_numpy(arr).float().unsqueeze(-1)

    def _load_context_from_cache(
        self,
        file_path: str,
        window_idx: int,
        extractor,
        context_cache: _ContextFileBufferCache,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        arr = context_cache.get_context(
            file_path=file_path,
            window_index=window_idx,
            extractor=extractor,
            window_size=self._window_size,
            fs=self._fs,
            context_input_size=self._context_input_size,
            context_total_seconds=self._context_total_seconds,
            allow_cross_file=self._context_allow_cross_file,
        )
        if self._profile_stats is not None:
            self._profile_stats.context_load_time_s += time.perf_counter() - t0
        return torch.from_numpy(arr).float().unsqueeze(-1)

    def _load_context(
        self,
        file_path: str,
        window_idx: int,
        extractor,
        context_cache: _ContextFileBufferCache | None = None,
    ) -> torch.Tensor:
        if context_cache is not None:
            return self._load_context_from_cache(file_path, window_idx, extractor, context_cache)
        if self._context_cache is not None:
            return self._load_context_from_cache(file_path, window_idx, extractor, self._context_cache)

        t0 = time.perf_counter()
        context_extractor = extractor
        if self._file_cache is not None:
            context_extractor = _CachedContextExtractor(extractor, self._file_cache)
        arr = load_context_array(
            file_path=file_path,
            window_index=window_idx,
            window_size=self._window_size,
            fs=self._fs,
            context_input_size=self._context_input_size,
            context_total_seconds=self._context_total_seconds,
            allow_cross_file=self._context_allow_cross_file,
            extractor=context_extractor,
        )
        if self._profile_stats is not None:
            self._profile_stats.context_load_time_s += time.perf_counter() - t0
        return torch.from_numpy(arr).float().unsqueeze(-1)

    def __getitem__(self, idx: int):
        extractor = self._get_extractor()
        rec = self._records[idx]
        in_meta = rec.inplane_meta or {}
        out_meta = rec.outplane_meta or {}
        in_fp = in_meta.get("file_path")
        out_fp = out_meta.get("file_path")
        in_time = self._load_signal(in_fp, rec.window_idx, in_meta, "in", extractor)
        out_time = self._load_signal(out_fp, rec.window_idx, out_meta, "out", extractor)
        wind_features = None
        if self._enable_wind_features:
            m, d, h = rec.timestamp_key
            wind_features = torch.from_numpy(
                build_short_wind_features(
                    {
                        "timestamp": [m, d, h],
                        "window_index": rec.window_idx,
                        "inplane_sensor_id": in_meta.get("sensor_id"),
                        "outplane_sensor_id": out_meta.get("sensor_id"),
                        "inplane_file_path": in_fp,
                        "outplane_file_path": out_fp,
                    },
                    self._wind_config,
                )
            ).float()
        if self._enable_long_context:
            in_context = self._load_context(in_fp, rec.window_idx, extractor)
            out_context = self._load_context(out_fp, rec.window_idx, extractor)
            if self._enable_wind_features:
                return in_time, in_context, out_time, out_context, wind_features, self._original_indices[idx]
            return in_time, in_context, out_time, out_context, self._original_indices[idx]
        if self._enable_wind_features:
            return in_time, out_time, wind_features, self._original_indices[idx]
        return in_time, out_time, self._original_indices[idx]

    def _load_one_sample(
        self,
        ds_idx: int,
        context_cache: _ContextFileBufferCache | None = None,
    ):
        extractor = self._get_extractor()
        rec = self._records[ds_idx]
        in_meta = rec.inplane_meta or {}
        out_meta = rec.outplane_meta or {}
        in_fp = in_meta.get("file_path")
        out_fp = out_meta.get("file_path")
        in_time = self._load_signal(in_fp, rec.window_idx, in_meta, "in", extractor)
        out_time = self._load_signal(out_fp, rec.window_idx, out_meta, "out", extractor)
        in_context = None
        out_context = None
        wind_features = None
        if self._enable_long_context:
            in_context = self._load_context(in_fp, rec.window_idx, extractor, context_cache=context_cache)
            out_context = self._load_context(out_fp, rec.window_idx, extractor, context_cache=context_cache)
        if self._enable_wind_features:
            m, d, h = rec.timestamp_key
            wind_features = torch.from_numpy(
                build_short_wind_features(
                    {
                        "timestamp": [m, d, h],
                        "window_index": rec.window_idx,
                        "inplane_sensor_id": in_meta.get("sensor_id"),
                        "outplane_sensor_id": out_meta.get("sensor_id"),
                        "inplane_file_path": in_fp,
                        "outplane_file_path": out_fp,
                    },
                    self._wind_config,
                )
            ).float()
        return (
            in_time,
            out_time,
            in_context,
            out_context,
            wind_features,
            self._original_indices[ds_idx],
        )

    def _time_block_sort_key(self, ds_idx: int) -> tuple:
        rec = self._records[ds_idx]
        in_meta = rec.inplane_meta or {}
        out_meta = rec.outplane_meta or {}
        return (
            str(in_meta.get("file_path") or ""),
            str(out_meta.get("file_path") or ""),
            int(rec.window_idx),
            int(self._original_indices[ds_idx]),
        )

    def _batch_context_cache(self, ds_indices: List[int]) -> _ContextFileBufferCache:
        file_paths: set[str] = set()
        for ds_idx in ds_indices:
            rec = self._records[ds_idx]
            in_fp = (rec.inplane_meta or {}).get("file_path")
            out_fp = (rec.outplane_meta or {}).get("file_path")
            if in_fp:
                file_paths.add(str(in_fp))
            if out_fp:
                file_paths.add(str(out_fp))
        return _ContextFileBufferCache(max_entries=max(1, len(file_paths)), file_cache=self._file_cache)

    def _rows_to_prepared_batch(self, rows) -> PreparedJointBatch:
        in_times = [row[0] for row in rows]
        out_times = [row[1] for row in rows]
        in_contexts = [row[2] for row in rows if row[2] is not None]
        out_contexts = [row[3] for row in rows if row[3] is not None]
        winds = [row[4] for row in rows if row[4] is not None]
        idx_batch = [row[5] for row in rows]

        in_time_batch = normalize_time_batch(torch.stack(in_times))
        out_time_batch = normalize_time_batch(torch.stack(out_times))
        psd_start = time.perf_counter()
        in_psd_batch = compute_psd_batch(
            in_time_batch, fs=self._fs, nfft=self._nfft, freq_max_hz=self._freq_max_hz
        )
        out_psd_batch = compute_psd_batch(
            out_time_batch, fs=self._fs, nfft=self._nfft, freq_max_hz=self._freq_max_hz
        )
        psd_time_s = time.perf_counter() - psd_start

        return PreparedJointBatch(
            idx_batch=idx_batch,
            in_time_batch=in_time_batch,
            out_time_batch=out_time_batch,
            in_psd_batch=in_psd_batch,
            out_psd_batch=out_psd_batch,
            in_context_batch=(
                normalize_time_batch(torch.stack(in_contexts)) if self._enable_long_context else None
            ),
            out_context_batch=(
                normalize_time_batch(torch.stack(out_contexts)) if self._enable_long_context else None
            ),
            wind_features_batch=torch.stack(winds) if self._enable_wind_features else None,
            psd_time_s=psd_time_s,
        )

    def load_time_block_batch(self, ds_indices: List[int]) -> PreparedJointBatch:
        ordered_indices = sorted(ds_indices, key=self._time_block_sort_key)
        context_cache = self._batch_context_cache(ordered_indices) if self._enable_long_context else None
        rows = [self._load_one_sample(ds_idx, context_cache=context_cache) for ds_idx in ordered_indices]
        return self._rows_to_prepared_batch(rows)

    def load_batch(self, ds_indices: List[int]) -> PreparedJointBatch:
        if self._context_workers > 1 and len(ds_indices) > 1:
            rows = [None] * len(ds_indices)
            with ThreadPoolExecutor(max_workers=self._context_workers) as executor:
                futures = {
                    executor.submit(self._load_one_sample, ds_idx): pos
                    for pos, ds_idx in enumerate(ds_indices)
                }
                for future in as_completed(futures):
                    pos = futures[future]
                    rows[pos] = future.result()
        else:
            rows = [self._load_one_sample(ds_idx) for ds_idx in ds_indices]
        return self._rows_to_prepared_batch(rows)


class DualStreamInferenceRunner:
    def __init__(
        self,
        identifier: DualStreamIdentifier,
        batch_size: int = 256,
        num_workers: int = 4,
        psd_workers: int = 2,
        fs: float = 50.0,
        nfft: int = 2048,
        freq_max_hz: float = 25.0,
        wind_config: Optional[dict] = None,
        cache_max_mb: float = 512.0,
        prefetch_files: int = 4,
        prefetch_batches: int = 2,
        prefetch_workers: int = 2,
        context_workers: int = 2,
        context_cache_entries: int = 64,
        context_batch_mode: str = "time_block",
        time_block_producer_workers: int = 1,
        joint_queue_depth: int = 2,
        profile_stats: Optional[PrefetchStats] = None,
    ):
        self.identifier = identifier
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.psd_workers = psd_workers
        self.fs = fs
        self.nfft = nfft
        self.freq_max_hz = freq_max_hz
        self.wind_config = dict(wind_config or {})
        self.cache_max_mb = float(cache_max_mb)
        self.prefetch_files = int(prefetch_files)
        self.prefetch_batches = int(prefetch_batches)
        self.prefetch_workers = int(prefetch_workers)
        self.context_workers = int(context_workers)
        self.context_cache_entries = int(context_cache_entries)
        self.context_batch_mode = str(context_batch_mode or "time_block").strip().lower()
        self.time_block_producer_workers = max(1, int(time_block_producer_workers))
        self.joint_queue_depth = max(1, int(joint_queue_depth))
        self.profile_stats = profile_stats or PrefetchStats()

    def _make_file_cache(self) -> BoundedVicFileCache:
        return BoundedVicFileCache(max_mb=self.cache_max_mb, stats=self.profile_stats)

    def _run_direction_proba(self, dataset, direction: str) -> Dict[int, np.ndarray]:
        from src.identifier.dl.runner import _make_extractor

        window_size = dataset.config.window_size
        enable_denoise = dataset.config.enable_denoise
        extractor = _make_extractor(enable_denoise=enable_denoise)

        if direction == "inplane":
            recs, idxs = FullDatasetRunner._validate_records(
                dataset._samples, window_size, extractor=extractor
            )
        else:
            recs, idxs = FullDatasetRunner._validate_outplane_records(
                dataset._samples, window_size, extractor=extractor
            )
        recs, idxs = FullDatasetRunner._sort_by_meta_path(recs, idxs, direction)

        file_cache = self._make_file_cache()
        ds = _DirectionWindowDataset(
            records=recs,
            direction=direction,
            window_size=window_size,
            enable_denoise=enable_denoise,
            original_indices=idxs,
            file_cache=file_cache,
            profile_stats=self.profile_stats,
        )

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_windows,
            pin_memory=torch.cuda.is_available(),
        )

        probas: Dict[int, np.ndarray] = {}
        pipeline = AsyncPsdPipeline(
            fs=self.fs,
            nfft=self.nfft,
            freq_max_hz=self.freq_max_hz,
            psd_workers=self.psd_workers,
        )

        predict_fn = self.identifier.predict_proba_batch
        use_cuda = torch.cuda.is_available()

        batch_iter = iter_progress(loader, total=len(loader), desc=f"DualStream {direction} 推理")
        for batch_idx, (time_batch, idx_batch) in enumerate(batch_iter):
            wait_start = time.perf_counter()
            time_batch = normalize_time_batch(time_batch)
            self.profile_stats.batch_wait_time_s += time.perf_counter() - wait_start
            if pipeline.has_pending():
                with track_gpu_forward(self.profile_stats, use_cuda):
                    pipeline.consume(predict_fn, probas)
            pipeline.submit(time_batch, idx_batch)

        if pipeline.has_pending():
            with track_gpu_forward(self.profile_stats, use_cuda):
                pipeline.consume(predict_fn, probas)

        pipeline.close()
        return probas

    def run_with_proba(self, dataset) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        if self.identifier.model_type in _DUAL_HEAD_MODEL_TYPES:
            return self._run_joint_proba(dataset)
        inplane = self._run_direction_proba(dataset, "inplane")
        outplane = self._run_direction_proba(dataset, "outplane")
        return inplane, outplane

    def _run_joint_proba(self, dataset) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        from src.identifier.dl.runner import _make_extractor

        window_size = dataset.config.window_size
        enable_denoise = dataset.config.enable_denoise
        enable_long_context = (
            self.identifier.model_type in _LONG_CONTEXT_MODEL_TYPES
            and self.identifier.context_mode == "short_long"
        )
        enable_wind_features = int(getattr(self.identifier, "wind_feature_dim", 0)) > 0
        extractor = _make_extractor(enable_denoise=enable_denoise)
        _, in_idxs = FullDatasetRunner._validate_records(dataset._samples, window_size, extractor=extractor)
        _, out_idxs = FullDatasetRunner._validate_outplane_records(
            dataset._samples, window_size, extractor=extractor
        )
        valid_idx = sorted(set(in_idxs) & set(out_idxs))
        valid_records = [dataset._samples[i] for i in valid_idx]
        valid_records, valid_idx = FullDatasetRunner._sort_by_meta_path(
            valid_records, valid_idx, "inplane"
        )
        effective_workers = _effective_joint_num_workers(self.num_workers, len(valid_records))
        logger.info(
            "配对双头联合推理样本：valid=%s, batch=%s, dataloader_workers=%s, "
            "cache_max_mb=%s, prefetch_files=%s, prefetch_batches=%s",
            len(valid_records),
            self.batch_size,
            effective_workers,
            self.cache_max_mb,
            self.prefetch_files,
            self.prefetch_batches,
        )
        file_cache = self._make_file_cache()
        ds = _PairedWindowDataset(
            records=valid_records,
            window_size=window_size,
            fs=self.fs,
            enable_denoise=enable_denoise,
            original_indices=valid_idx,
            enable_long_context=enable_long_context,
            context_input_size=self.identifier.context_input_size,
            context_total_seconds=self.identifier.context_total_seconds,
            context_allow_cross_file=self.identifier.context_allow_cross_file,
            enable_wind_features=enable_wind_features,
            wind_config=self.wind_config,
            file_cache=file_cache,
            profile_stats=self.profile_stats,
            nfft=self.nfft,
            freq_max_hz=self.freq_max_hz,
            context_workers=self.context_workers,
            context_cache_entries=self.context_cache_entries,
        )
        prefetch_scheduler = PrefetchScheduler(
            cache=file_cache,
            loader=extractor.load_file,
            prefetch_files=self.prefetch_files,
            prefetch_workers=self.prefetch_workers,
        )
        use_cuda = torch.cuda.is_available()

        in_probas: Dict[int, np.ndarray] = {}
        out_probas: Dict[int, np.ndarray] = {}

        if enable_long_context:
            batch_ranges = [
                list(range(i, min(i + self.batch_size, len(valid_records))))
                for i in range(0, len(valid_records), self.batch_size)
            ]
            if self.context_batch_mode == "legacy":
                prepare_fn = ds.load_batch
                producer_workers = self.context_workers
            elif self.context_batch_mode == "time_block":
                prepare_fn = ds.load_time_block_batch
                producer_workers = self.time_block_producer_workers
            else:
                raise ValueError(f"未知 context_batch_mode：{self.context_batch_mode}")
            logger.info(
                "启用 AsyncJointBatchPipeline：mode=%s, batches=%s, producer_workers=%s, "
                "context_workers=%s, queue_depth=%s",
                self.context_batch_mode,
                len(batch_ranges),
                producer_workers,
                self.context_workers,
                self.joint_queue_depth,
            )
            pipeline = AsyncJointBatchPipeline(
                prepare_fn,
                workers=producer_workers,
                queue_depth=self.joint_queue_depth,
            )
            if batch_ranges:
                upcoming = upcoming_batch_file_paths(
                    valid_records, self.batch_size, -1, self.prefetch_batches
                )
                prefetch_scheduler.schedule_for_batch(upcoming)
                for submit_idx in range(min(self.joint_queue_depth, len(batch_ranges))):
                    pipeline.submit(batch_ranges[submit_idx])

            batch_iter = iter_progress(
                enumerate(batch_ranges), total=len(batch_ranges), desc="配对双头联合推理"
            )
            for batch_idx, ds_indices in batch_iter:
                wait_start = time.perf_counter()
                prepared = pipeline.consume()
                wait_elapsed = time.perf_counter() - wait_start
                self.profile_stats.batch_wait_time_s += wait_elapsed
                if wait_elapsed > 0.05:
                    self.profile_stats.queue_empty_count += 1
                self.profile_stats.psd_time_s += prepared.psd_time_s

                next_submit_idx = batch_idx + self.joint_queue_depth
                if next_submit_idx < len(batch_ranges):
                    upcoming = upcoming_batch_file_paths(
                        valid_records,
                        self.batch_size,
                        next_submit_idx - 1,
                        self.prefetch_batches,
                    )
                    prefetch_scheduler.schedule_for_batch(upcoming)
                    pipeline.submit(batch_ranges[next_submit_idx])

                with track_gpu_forward(self.profile_stats, use_cuda):
                    in_batch_proba, out_batch_proba = self.identifier.predict_dual_proba_batch(
                        prepared.in_time_batch,
                        prepared.in_psd_batch,
                        prepared.out_time_batch,
                        prepared.out_psd_batch,
                        in_context=prepared.in_context_batch,
                        out_context=prepared.out_context_batch,
                        wind_features=prepared.wind_features_batch,
                    )
                for idx, in_p, out_p in zip(prepared.idx_batch, in_batch_proba, out_batch_proba):
                    in_probas[int(idx)] = in_p
                    out_probas[int(idx)] = out_p
            pipeline.close()
            prefetch_scheduler.close()
            return in_probas, out_probas

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=effective_workers,
            pin_memory=torch.cuda.is_available(),
        )
        batch_iter = iter_progress(loader, total=len(loader), desc="配对双头联合推理")
        for batch_idx, batch in enumerate(batch_iter):
            upcoming = upcoming_batch_file_paths(
                valid_records,
                self.batch_size,
                batch_idx,
                self.prefetch_batches,
            )
            prefetch_scheduler.schedule_for_batch(upcoming)

            wait_start = time.perf_counter()
            wind_features_batch = None
            if enable_long_context:
                if enable_wind_features:
                    in_time_batch, in_context_batch, out_time_batch, out_context_batch, wind_features_batch, idx_batch = batch
                else:
                    in_time_batch, in_context_batch, out_time_batch, out_context_batch, idx_batch = batch
            else:
                if enable_wind_features:
                    in_time_batch, out_time_batch, wind_features_batch, idx_batch = batch
                else:
                    in_time_batch, out_time_batch, idx_batch = batch
            wait_elapsed = time.perf_counter() - wait_start
            self.profile_stats.batch_wait_time_s += wait_elapsed
            if wait_elapsed > 0.05:
                self.profile_stats.queue_empty_count += 1

            in_time_batch = normalize_time_batch(in_time_batch)
            out_time_batch = normalize_time_batch(out_time_batch)

            psd_start = time.perf_counter()
            in_psd_batch = compute_psd_batch(
                in_time_batch, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz
            )
            out_psd_batch = compute_psd_batch(
                out_time_batch, fs=self.fs, nfft=self.nfft, freq_max_hz=self.freq_max_hz
            )
            self.profile_stats.psd_time_s += time.perf_counter() - psd_start

            with track_gpu_forward(self.profile_stats, use_cuda):
                if enable_long_context:
                    in_context_batch = normalize_time_batch(in_context_batch)
                    out_context_batch = normalize_time_batch(out_context_batch)
                    in_batch_proba, out_batch_proba = self.identifier.predict_dual_proba_batch(
                        in_time_batch,
                        in_psd_batch,
                        out_time_batch,
                        out_psd_batch,
                        in_context=in_context_batch,
                        out_context=out_context_batch,
                        wind_features=wind_features_batch,
                    )
                else:
                    in_batch_proba, out_batch_proba = self.identifier.predict_dual_proba_batch(
                        in_time_batch,
                        in_psd_batch,
                        out_time_batch,
                        out_psd_batch,
                        wind_features=wind_features_batch,
                    )
            for idx, in_p, out_p in zip(idx_batch, in_batch_proba, out_batch_proba):
                in_probas[int(idx)] = in_p
                out_probas[int(idx)] = out_p

        prefetch_scheduler.close()
        return in_probas, out_probas

    @staticmethod
    def merge_proba_predictions(
        inplane: Dict[int, np.ndarray],
        outplane: Dict[int, np.ndarray],
        projection_mode: str = "direction",
        projection_direction: str = "outplane",
    ) -> Dict[int, Tuple[int, np.ndarray]]:
        merged: Dict[int, Tuple[int, np.ndarray]] = {}
        mode = str(projection_mode or "direction")
        direction = str(projection_direction or "outplane")
        if mode not in {"direction", "abnormal_priority"}:
            raise ValueError(f"未知 prediction_projection_mode: {mode}")
        if direction not in {"inplane", "outplane"}:
            raise ValueError(f"未知 prediction_projection_direction: {direction}")
        all_idx = set(inplane.keys()) | set(outplane.keys())
        for idx in all_idx:
            in_p = inplane.get(idx)
            out_p = outplane.get(idx)
            if in_p is None and out_p is None:
                continue
            if in_p is None:
                pred = int(out_p.argmax())
                merged[idx] = (pred, out_p)
                continue
            if out_p is None:
                pred = int(in_p.argmax())
                merged[idx] = (pred, in_p)
                continue
            if mode == "direction":
                proba = out_p if direction == "outplane" else in_p
                pred = int(proba.argmax())
                merged[idx] = (pred, proba)
                continue
            in_pred = int(in_p.argmax())
            out_pred = int(out_p.argmax())
            in_conf = float(np.max(in_p))
            out_conf = float(np.max(out_p))
            in_abnormal = in_pred != _NORMAL_LABEL
            out_abnormal = out_pred != _NORMAL_LABEL

            if in_abnormal or out_abnormal:
                candidates = []
                if in_abnormal:
                    candidates.append((in_conf, in_pred, in_p))
                if out_abnormal:
                    candidates.append((out_conf, out_pred, out_p))
                _, pred, proba = max(candidates, key=lambda x: x[0])
                merged[idx] = (int(pred), proba)
                continue

            merged[idx] = (_NORMAL_LABEL, (in_p + out_p) / 2.0)
        return merged
