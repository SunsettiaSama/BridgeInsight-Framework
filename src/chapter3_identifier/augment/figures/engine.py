from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Optional

from src.chapter3_identifier.augment.figures.cache import PngCache
from src.chapter3_identifier.augment.figures.render.context import (
    build_context_render_data,
    render_context_part_from_data,
)
from src.chapter3_identifier.augment.figures.render.placeholders import render_placeholder_figure
from src.chapter3_identifier.augment.figures.render.sample import (
    build_sample_render_data,
    render_prediction_figure,
    render_sample_figure_from_data,
)
from src.chapter3_identifier.augment.figures.types import ContextParams, SAMPLE_FIGURE_NAMES


class FigureRenderEngine:
    """线程池渲染引擎：只负责异步渲染与内存缓存，不处理 HTTP。"""

    def __init__(self, cache: PngCache, max_workers: int = 2, image_workers: int = 6) -> None:
        self._cache = cache
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aug-fig-worker")
        self._image_executor = ThreadPoolExecutor(
            max_workers=image_workers, thread_name_prefix="aug-fig-image"
        )
        self._lock = threading.Lock()
        self._pending: Dict[str, Future] = {}
        self._pending_ctx: Dict[str, Future] = {}

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._image_executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _sample_key(
        sample_idx: int,
        figure_name: str,
        layout_profile: str,
        prediction_direction: str = "inplane",
    ) -> str:
        if figure_name == "prediction":
            direction = "outplane" if prediction_direction == "outplane" else "inplane"
            return f"s:{sample_idx}:{layout_profile}:{figure_name}:{direction}"
        return f"s:{sample_idx}:{layout_profile}:{figure_name}"

    @staticmethod
    def _context_key(sample_idx: int, direction: str, part: str, layout_profile: str) -> str:
        return f"c:{sample_idx}:{layout_profile}:{direction}:{part}"

    @staticmethod
    def _context_file_path(record: dict, direction: str) -> Optional[str]:
        if direction == "inplane":
            return record.get("inplane_file_path")
        return record.get("outplane_file_path")

    def _sample_keys(self, sample_idx: int, layout_profile: str) -> list[str]:
        keys: list[str] = []
        for name in SAMPLE_FIGURE_NAMES:
            if name == "prediction":
                keys.append(self._sample_key(sample_idx, name, layout_profile, "inplane"))
                keys.append(self._sample_key(sample_idx, name, layout_profile, "outplane"))
            else:
                keys.append(self._sample_key(sample_idx, name, layout_profile))
        return keys

    def sample_ready(self, sample_idx: int, layout_profile: str = "wide_fill_v1") -> bool:
        return self._cache.has_all(self._sample_keys(sample_idx, layout_profile))

    def context_ready(self, sample_idx: int, direction: str, layout_profile: str = "wide_fill_v1") -> bool:
        return self._cache.has_all(
            [
                self._context_key(sample_idx, direction, "timeseries", layout_profile),
                self._context_key(sample_idx, direction, "spectrogram", layout_profile),
            ]
        )

    def bundle_ready(
        self,
        sample_idx: int,
        direction: str,
        record: Optional[dict] = None,
        layout_profile: str = "wide_fill_v1",
    ) -> bool:
        if not self.sample_ready(sample_idx, layout_profile):
            return False
        if record is not None and not self._context_file_path(record, direction):
            return True
        return self.context_ready(sample_idx, direction, layout_profile)

    def _store_placeholder_context(
        self,
        sample_idx: int,
        direction: str,
        message: str,
        layout_profile: str = "wide_fill_v1",
    ) -> None:
        label = "面外" if direction == "outplane" else "面内"
        png = render_placeholder_figure(f"{label}上下文不可用\n{message}")
        self._cache.put(self._context_key(sample_idx, direction, "timeseries", layout_profile), png)
        self._cache.put(self._context_key(sample_idx, direction, "spectrogram", layout_profile), png)

    def _store_sample_bundle(self, record: dict, layout_profile: str = "wide_fill_v1") -> None:
        sample_idx = int(record["sample_idx"])
        if self.sample_ready(sample_idx, layout_profile):
            return
        try:
            render_data = build_sample_render_data(record)
        except Exception as exc:
            msg = render_placeholder_figure(f"样本图渲染失败\nsample={sample_idx}\n{exc}")
            for name in SAMPLE_FIGURE_NAMES:
                if name == "prediction":
                    self._cache.put(
                        self._sample_key(sample_idx, name, layout_profile, "inplane"),
                        msg,
                    )
                    self._cache.put(
                        self._sample_key(sample_idx, name, layout_profile, "outplane"),
                        msg,
                    )
                else:
                    self._cache.put(self._sample_key(sample_idx, name, layout_profile), msg)
            return

        image_plan = [
            ("in_timeseries", "inplane"),
            ("out_timeseries", "inplane"),
            ("in_spectrum", "inplane"),
            ("out_spectrum", "inplane"),
            ("trajectory", "inplane"),
            ("prediction", "inplane"),
            ("prediction", "outplane"),
        ]
        futures: Dict[Future, tuple[str, str]] = {}
        for fig_name, pred_dir in image_plan:
            fut = self._image_executor.submit(
                render_sample_figure_from_data,
                render_data,
                fig_name,
                layout_profile,
                pred_dir,
            )
            futures[fut] = (fig_name, pred_dir)
        for fut in as_completed(futures):
            fig_name, pred_dir = futures[fut]
            try:
                png = fut.result()
            except Exception as exc:
                png = render_placeholder_figure(f"样本图渲染失败\nsample={sample_idx}\n{exc}")
            self._cache.put(
                self._sample_key(sample_idx, fig_name, layout_profile, pred_dir),
                png,
            )

    def _store_context_bundle(self, record: dict, direction: str, ctx: ContextParams) -> None:
        sample_idx = int(record["sample_idx"])
        fp = self._context_file_path(record, direction)
        if not fp:
            self._store_placeholder_context(
                sample_idx,
                direction,
                "缺少对应方向文件路径",
                layout_profile=ctx.layout_profile,
            )
            return
        ts_key = self._context_key(sample_idx, direction, "timeseries", ctx.layout_profile)
        sp_key = self._context_key(sample_idx, direction, "spectrogram", ctx.layout_profile)
        if self._cache.has_all([ts_key, sp_key]):
            return
        if direction == "inplane":
            sid = record.get("inplane_sensor_id", "in")
        else:
            sid = record.get("outplane_sensor_id", "out")
        try:
            render_data = build_context_render_data(
                file_path=fp,
                window_index=int(record.get("window_index", 0)),
                direction=direction,
                sensor_id=sid or direction,
                before=ctx.windows_before,
                after=ctx.windows_after,
            )
        except Exception as exc:
            png = render_placeholder_figure(f"上下文渲染失败\nsample={sample_idx}\n{exc}")
            self._cache.put(ts_key, png)
            self._cache.put(sp_key, png)
            return

        ts_future = self._image_executor.submit(
            render_context_part_from_data,
            render_data,
            "timeseries",
            ctx.spectrogram_segment_s,
            ctx.layout_profile,
        )
        sp_future = self._image_executor.submit(
            render_context_part_from_data,
            render_data,
            "spectrogram",
            ctx.spectrogram_segment_s,
            ctx.layout_profile,
        )
        try:
            ts_png = ts_future.result()
        except Exception as exc:
            ts_png = render_placeholder_figure(f"上下文渲染失败\nsample={sample_idx}\n{exc}")
        try:
            sp_png = sp_future.result()
        except Exception as exc:
            sp_png = render_placeholder_figure(f"上下文渲染失败\nsample={sample_idx}\n{exc}")
        self._cache.put(ts_key, ts_png)
        self._cache.put(sp_key, sp_png)

    def _submit(self, pending: Dict, key, fn: Callable[[], None]) -> None:
        with self._lock:
            if key in pending:
                return
            fut = self._executor.submit(fn)
            pending[key] = fut

            def _done(_f: Future) -> None:
                with self._lock:
                    pending.pop(key, None)

            fut.add_done_callback(_done)

    def preload_sample(self, record: dict, layout_profile: str = "wide_fill_v1") -> None:
        sample_idx = int(record["sample_idx"])
        if self.sample_ready(sample_idx, layout_profile):
            return
        sample_key = f"{sample_idx}:{layout_profile}"
        with self._lock:
            if sample_key in self._pending:
                return
        self._submit(
            self._pending,
            sample_key,
            lambda: self._store_sample_bundle(record, layout_profile=layout_profile),
        )

    def preload_context(self, record: dict, ctx: ContextParams) -> None:
        sample_idx = int(record["sample_idx"])
        direction = ctx.direction
        ctx_key = f"{sample_idx}:{direction}:{ctx.layout_profile}"
        ts_key = self._context_key(sample_idx, direction, "timeseries", ctx.layout_profile)
        sp_key = self._context_key(sample_idx, direction, "spectrogram", ctx.layout_profile)
        if self._cache.has_all([ts_key, sp_key]):
            return
        with self._lock:
            if ctx_key in self._pending_ctx:
                return
        self._submit(
            self._pending_ctx,
            ctx_key,
            lambda: self._store_context_bundle(record, direction, ctx),
        )

    def preload_bundle(self, record: dict, ctx: ContextParams) -> None:
        self.preload_sample(record, layout_profile=ctx.layout_profile)
        self.preload_context(record, ctx)

    def get_sample_png(
        self,
        record: dict,
        figure_name: str,
        layout_profile: str = "wide_fill_v1",
        prediction_direction: str = "inplane",
    ) -> Optional[bytes]:
        sample_idx = int(record["sample_idx"])
        png = self._cache.get(
            self._sample_key(sample_idx, figure_name, layout_profile, prediction_direction)
        )
        if png is not None:
            return png
        if figure_name == "prediction":
            png = render_prediction_figure(
                record,
                prediction_direction=prediction_direction,
                layout_profile=layout_profile,
            )
            self._cache.put(
                self._sample_key(sample_idx, figure_name, layout_profile, prediction_direction),
                png,
            )
            return png
        self.preload_sample(record, layout_profile=layout_profile)
        return None

    def get_context_png(
        self,
        record: dict,
        part: str,
        ctx: ContextParams,
    ) -> Optional[bytes]:
        sample_idx = int(record["sample_idx"])
        direction = ctx.direction
        key = self._context_key(sample_idx, direction, part, ctx.layout_profile)
        png = self._cache.get(key)
        if png is not None:
            return png
        if not self._context_file_path(record, direction):
            self._store_placeholder_context(
                sample_idx,
                direction,
                "缺少对应方向文件路径",
                layout_profile=ctx.layout_profile,
            )
            return self._cache.get(key)
        self.preload_context(record, ctx)
        return None
