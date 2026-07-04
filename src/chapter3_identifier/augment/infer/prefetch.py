from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrefetchStats:
    cache_hits: int = 0
    cache_misses: int = 0
    file_load_count: int = 0
    file_load_time_s: float = 0.0
    window_slice_time_s: float = 0.0
    context_load_time_s: float = 0.0
    psd_time_s: float = 0.0
    gpu_forward_time_s: float = 0.0
    batch_wait_time_s: float = 0.0
    queue_empty_count: int = 0
    queue_full_wait_time_s: float = 0.0
    prefetch_submit_count: int = 0
    cache_evict_count: int = 0
    cache_bytes_peak: int = 0

    def to_dict(self) -> dict:
        io_psd = (
            self.file_load_time_s
            + self.window_slice_time_s
            + self.context_load_time_s
            + self.psd_time_s
        )
        total = io_psd + self.gpu_forward_time_s + self.batch_wait_time_s
        gpu_idle_ratio = (
            self.batch_wait_time_s / total if total > 0 else 0.0
        )
        gap = io_psd - self.gpu_forward_time_s
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "file_load_count": self.file_load_count,
            "file_load_time_s": round(self.file_load_time_s, 4),
            "window_slice_time_s": round(self.window_slice_time_s, 4),
            "context_load_time_s": round(self.context_load_time_s, 4),
            "psd_time_s": round(self.psd_time_s, 4),
            "gpu_forward_time_s": round(self.gpu_forward_time_s, 4),
            "batch_wait_time_s": round(self.batch_wait_time_s, 4),
            "io_psd_time_s": round(io_psd, 4),
            "io_psd_minus_gpu_gap_s": round(gap, 4),
            "gpu_idle_wait_ratio": round(gpu_idle_ratio, 4),
            "queue_empty_count": self.queue_empty_count,
            "queue_full_wait_time_s": round(self.queue_full_wait_time_s, 4),
            "prefetch_submit_count": self.prefetch_submit_count,
            "cache_evict_count": self.cache_evict_count,
            "cache_bytes_peak": self.cache_bytes_peak,
        }


class BoundedVicFileCache:
    """有界 LRU 文件级 VIC 缓存，按字节预算淘汰。"""

    def __init__(self, max_mb: float, stats: Optional[PrefetchStats] = None):
        self._max_bytes = max(1, int(float(max_mb) * 1024 * 1024))
        self._current_bytes = 0
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._load_locks: Dict[str, threading.Lock] = {}
        self.stats = stats or PrefetchStats()

    def get(self, file_path: str) -> Optional[np.ndarray]:
        with self._lock:
            arr = self._cache.get(file_path)
            if arr is None:
                return None
            self._cache.move_to_end(file_path)
            self.stats.cache_hits += 1
            return arr

    def get_or_load(self, file_path: str, loader: Callable[[str], np.ndarray]) -> np.ndarray:
        cached = self.get(file_path)
        if cached is not None:
            return cached
        with self._lock:
            load_lock = self._load_locks.get(file_path)
            if load_lock is None:
                load_lock = threading.Lock()
                self._load_locks[file_path] = load_lock
        with load_lock:
            cached = self.get(file_path)
            if cached is not None:
                return cached
            self.stats.cache_misses += 1
            t0 = time.perf_counter()
            data = loader(file_path)
            self.stats.file_load_time_s += time.perf_counter() - t0
            self.stats.file_load_count += 1
            self.put(file_path, data)
            return data

    def put(self, file_path: str, data: np.ndarray) -> None:
        nbytes = int(data.nbytes)
        with self._lock:
            if file_path in self._cache:
                old = self._cache.pop(file_path)
                self._current_bytes -= int(old.nbytes)
            while self._current_bytes + nbytes > self._max_bytes and self._cache:
                evicted_path, evicted = self._cache.popitem(last=False)
                self._current_bytes -= int(evicted.nbytes)
                self.stats.cache_evict_count += 1
                self._load_locks.pop(evicted_path, None)
            self._cache[file_path] = data
            self._current_bytes += nbytes
            peak = self._current_bytes
        if peak > self.stats.cache_bytes_peak:
            self.stats.cache_bytes_peak = peak

    def prefetch(self, file_paths: List[str], loader: Callable[[str], np.ndarray]) -> None:
        for fp in file_paths:
            if fp and self.get(fp) is None:
                self.stats.prefetch_submit_count += 1
                self.get_or_load(fp, loader)


class PrefetchScheduler:
    """
    按即将访问的 batch 预取 VIC 文件，有界线程池 + 回压。
    """

    def __init__(
        self,
        cache: BoundedVicFileCache,
        loader: Callable[[str], np.ndarray],
        prefetch_files: int = 4,
        prefetch_workers: int = 2,
        max_inflight: int = 8,
    ):
        self._cache = cache
        self._loader = loader
        self._prefetch_files = max(0, int(prefetch_files))
        self._executor = (
            ThreadPoolExecutor(max_workers=max(1, prefetch_workers), thread_name_prefix="vic-prefetch")
            if prefetch_workers > 0 and prefetch_files > 0
            else None
        )
        self._max_inflight = max(1, int(max_inflight))
        self._futures: List[Future] = []
        self._scheduled: Set[str] = set()
        self._lock = threading.Lock()

    def schedule_for_batch(self, file_paths: List[str]) -> None:
        if self._executor is None or self._prefetch_files <= 0:
            return
        unique = []
        seen: Set[str] = set()
        for fp in file_paths:
            if not fp or fp in seen:
                continue
            seen.add(fp)
            if self._cache.get(fp) is not None:
                continue
            with self._lock:
                if fp in self._scheduled:
                    continue
                self._scheduled.add(fp)
            unique.append(fp)
            if len(unique) >= self._prefetch_files:
                break
        for fp in unique:
            self._cache.stats.prefetch_submit_count += 1
            fut = self._executor.submit(self._cache.get_or_load, fp, self._loader)
            self._futures.append(fut)
        self._drain_done()

    def _drain_done(self) -> None:
        alive: List[Future] = []
        for fut in self._futures:
            if fut.done():
                fp_key = None
                with self._lock:
                    pass
                fut.result()
            else:
                alive.append(fut)
        while len(alive) > self._max_inflight:
            wait_start = time.perf_counter()
            oldest = alive.pop(0)
            oldest.result()
            self._cache.stats.queue_full_wait_time_s += time.perf_counter() - wait_start
        self._futures = alive

    def close(self) -> None:
        if self._executor is None:
            return
        for fut in self._futures:
            fut.result()
        self._futures.clear()
        with self._lock:
            self._scheduled.clear()
        self._executor.shutdown(wait=True)


def collect_file_paths_from_records(records, batch_indices: List[int]) -> List[str]:
    paths: List[str] = []
    for idx in batch_indices:
        rec = records[idx]
        in_fp = (rec.inplane_meta or {}).get("file_path")
        out_fp = (rec.outplane_meta or {}).get("file_path")
        if in_fp:
            paths.append(in_fp)
        if out_fp:
            paths.append(out_fp)
    return paths


def upcoming_batch_file_paths(
    records,
    batch_size: int,
    current_batch: int,
    prefetch_batches: int,
) -> List[str]:
    start = (current_batch + 1) * batch_size
    end = min(len(records), start + batch_size * max(1, prefetch_batches))
    return _paths_for_records(records, list(range(start, end)))


def _paths_for_records(records, record_indices: List[int]) -> List[str]:
    paths: List[str] = []
    for rec_idx in record_indices:
        if rec_idx < 0 or rec_idx >= len(records):
            continue
        rec = records[rec_idx]
        in_fp = (rec.inplane_meta or {}).get("file_path")
        out_fp = (rec.outplane_meta or {}).get("file_path")
        if in_fp:
            paths.append(in_fp)
        if out_fp:
            paths.append(out_fp)
    return paths
