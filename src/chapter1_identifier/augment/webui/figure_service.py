from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

from src.chapter1_identifier.augment.webui.context_figures import render_context_figures
from src.chapter1_identifier.augment.webui.figures import render_sample_figures

SAMPLE_FIGURE_NAMES = (
    "in_timeseries",
    "out_timeseries",
    "in_spectrum",
    "out_spectrum",
    "trajectory",
    "prediction",
)


class FigureRenderService:
    def __init__(self, max_workers: int = 2, cache_size: int = 48):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aug-fig")
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._pending: Dict[int, Future] = {}
        self._pending_ctx: Dict[str, Future] = {}

    def _touch(self, key: str, value: bytes) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _sample_key(self, sample_idx: int, figure_name: str) -> str:
        return f"s:{sample_idx}:{figure_name}"

    def _context_key(self, sample_idx: int, direction: str, part: str) -> str:
        return f"c:{sample_idx}:{direction}:{part}"

    def _sample_cached(self, sample_idx: int) -> bool:
        return all(self._sample_key(sample_idx, name) in self._cache for name in SAMPLE_FIGURE_NAMES)

    def _store_sample_bundle(self, record: dict) -> None:
        sample_idx = int(record["sample_idx"])
        figs = render_sample_figures(record)
        with self._lock:
            for name, png in figs.items():
                self._touch(self._sample_key(sample_idx, name), png)

    def _store_context_bundle(
        self,
        record: dict,
        direction: str,
        before: int,
        after: int,
        segment_s: float,
        cache_size: int,
    ) -> None:
        sample_idx = int(record["sample_idx"])
        if direction == "inplane":
            fp = record.get("inplane_file_path")
            sid = record.get("inplane_sensor_id", "in")
        else:
            fp = record.get("outplane_file_path")
            sid = record.get("outplane_sensor_id", "out")
        if not fp:
            return
        ts_png, sp_png = render_context_figures(
            fp,
            int(record.get("window_index", 0)),
            direction,
            sid or direction,
            before,
            after,
            segment_s,
            cache_size,
        )
        with self._lock:
            self._touch(self._context_key(sample_idx, direction, "timeseries"), ts_png)
            self._touch(self._context_key(sample_idx, direction, "spectrogram"), sp_png)

    def _run_exclusive(
        self,
        pending: Dict,
        key,
        fn: Callable[[], None],
    ) -> None:
        with self._lock:
            if key in pending:
                fut = pending[key]
            else:
                fut = self._executor.submit(fn)
                pending[key] = fut

                def _done(_f: Future) -> None:
                    with self._lock:
                        pending.pop(key, None)

                fut.add_done_callback(_done)
        fut.result()

    def ensure_sample(self, record: dict) -> None:
        sample_idx = int(record["sample_idx"])
        if self._sample_cached(sample_idx):
            return
        self._run_exclusive(self._pending, sample_idx, lambda: self._store_sample_bundle(record))

    def preload_sample(self, record: dict) -> None:
        sample_idx = int(record["sample_idx"])
        with self._lock:
            if self._sample_cached(sample_idx) or sample_idx in self._pending:
                return
            fut = self._executor.submit(self._store_sample_bundle, record)
            self._pending[sample_idx] = fut

            def _done(_f: Future) -> None:
                with self._lock:
                    self._pending.pop(sample_idx, None)

            fut.add_done_callback(_done)

    def preload_context(
        self,
        record: dict,
        direction: str,
        before: int,
        after: int,
        segment_s: float,
        cache_size: int,
    ) -> None:
        sample_idx = int(record["sample_idx"])
        ctx_key = f"{sample_idx}:{direction}"
        with self._lock:
            ts_key = self._context_key(sample_idx, direction, "timeseries")
            sp_key = self._context_key(sample_idx, direction, "spectrogram")
            if ts_key in self._cache and sp_key in self._cache:
                return
            if ctx_key in self._pending_ctx:
                return
            fut = self._executor.submit(
                self._store_context_bundle,
                record,
                direction,
                before,
                after,
                segment_s,
                cache_size,
            )
            self._pending_ctx[ctx_key] = fut

            def _done(_f: Future) -> None:
                with self._lock:
                    self._pending_ctx.pop(ctx_key, None)

            fut.add_done_callback(_done)

    def get_sample_figure(self, record: dict, figure_name: str) -> bytes:
        self.ensure_sample(record)
        sample_idx = int(record["sample_idx"])
        key = self._sample_key(sample_idx, figure_name)
        with self._lock:
            png = self._cache.get(key)
            if png is not None:
                self._cache.move_to_end(key)
                return png
        raise KeyError(f"figure not found: {figure_name}")

    def get_context_figure(
        self,
        record: dict,
        direction: str,
        part: str,
        before: int,
        after: int,
        segment_s: float,
        cache_size: int,
    ) -> bytes:
        sample_idx = int(record["sample_idx"])
        key = self._context_key(sample_idx, direction, part)
        with self._lock:
            png = self._cache.get(key)
            if png is not None:
                self._cache.move_to_end(key)
                return png

        self._run_exclusive(
            self._pending_ctx,
            f"{sample_idx}:{direction}",
            lambda: self._store_context_bundle(
                record, direction, before, after, segment_s, cache_size
            ),
        )
        with self._lock:
            png = self._cache.get(key)
            if png is not None:
                self._cache.move_to_end(key)
                return png
        raise KeyError(f"context figure not found: {part}")

    def preload_records(
        self,
        records: List[dict],
        direction: str,
        before: int,
        after: int,
        segment_s: float,
        cache_size: int,
    ) -> int:
        queued = 0
        for record in records:
            self.preload_sample(record)
            self.preload_context(record, direction, before, after, segment_s, cache_size)
            queued += 1
        return queued
