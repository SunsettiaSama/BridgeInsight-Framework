from __future__ import annotations

from typing import Callable, List, Optional, Set

from src.chapter3_identifier.augment.figures.cache import PngCache
from src.chapter3_identifier.augment.figures.dispatch import RenderDispatchCenter
from src.chapter3_identifier.augment.figures.engine import FigureRenderEngine
from src.chapter3_identifier.augment.figures.scheduler import FigureScheduler
from src.chapter3_identifier.augment.figures.types import ContextParams, FigureNotReadyError


class FigureService:
    """图像服务 Facade：调度线程 + 渲染线程池 + 内存 PNG 缓存。"""

    def __init__(self, max_workers: int = 2, image_workers: int = 6, cache_size: int = 256) -> None:
        self._cache = PngCache(max_size=cache_size)
        self._dispatch = RenderDispatchCenter(self._cache)
        self._engine = FigureRenderEngine(
            self._cache, max_workers=max_workers, image_workers=image_workers
        )
        self._scheduler = FigureScheduler(self._engine)

    def start(self) -> None:
        self._scheduler.start()

    def shutdown(self) -> None:
        self._scheduler.shutdown()
        self._engine.shutdown()

    def stats(self) -> dict:
        row = self._scheduler.stats()
        row["cache_entries"] = self._cache.size()
        row["dispatch_epoch"] = self._dispatch.epoch()
        return row

    def schedule_preload(
        self,
        records: List[dict],
        ctx: ContextParams,
        *,
        priority_samples: Optional[Set[int]] = None,
        replace: bool = False,
    ) -> int:
        return self._scheduler.schedule_records(
            records,
            ctx,
            replace=replace,
            priority_samples=priority_samples,
        )

    def schedule_by_indices(
        self,
        lookup: Callable[[int], Optional[dict]],
        sample_indices: List[int],
        ctx: ContextParams,
        *,
        priority_samples: Optional[Set[int]] = None,
        replace: bool = False,
    ) -> int:
        return self._scheduler.schedule_by_indices(
            lookup,
            sample_indices,
            ctx,
            priority_samples=priority_samples,
            replace=replace,
        )

    def get_sample_png(
        self,
        record: dict,
        figure_name: str,
        layout_profile: str = "wide_fill_v3",
        prediction_direction: str = "inplane",
        round_idx: int = 1,
    ) -> bytes:
        png = self._engine.get_sample_png(
            record,
            figure_name,
            layout_profile=layout_profile,
            prediction_direction=prediction_direction,
            round_idx=round_idx,
        )
        if png is not None:
            return png
        sample_idx = int(record["sample_idx"])
        raise FigureNotReadyError(
            f"sample figure rendering: {figure_name} sample={sample_idx} layout={layout_profile}"
        )

    def wait_sample_png(
        self,
        record: dict,
        figure_name: str,
        layout_profile: str = "wide_fill_v3",
        prediction_direction: str = "inplane",
        wait_ms: int = 0,
        round_idx: int = 1,
    ) -> bytes:
        png = self._engine.get_sample_png(
            record,
            figure_name,
            layout_profile=layout_profile,
            prediction_direction=prediction_direction,
            round_idx=round_idx,
        )
        if png is not None:
            return png
        if self._engine.is_wind_figure(figure_name):
            sample_idx = int(record["sample_idx"])
            raise FigureNotReadyError(
                f"wind figure rendering: {figure_name} sample={sample_idx} layout={layout_profile}"
            )
        if wait_ms > 0:
            sample_idx = int(record["sample_idx"])
            ctx = ContextParams(
                direction="inplane",
                round_idx=round_idx,
                layout_profile=layout_profile,
            )
            self._scheduler.schedule_records(
                [record],
                ctx,
                priority_samples={sample_idx},
                replace=False,
            )
            key = self._engine.sample_cache_key(
                round_idx,
                sample_idx,
                figure_name,
                layout_profile=layout_profile,
                prediction_direction=prediction_direction,
            )
            expected_epoch = self._dispatch.epoch()
            self._dispatch.wait_for_keys([key], wait_ms, expected_epoch=expected_epoch)
            png = self._engine.get_sample_png(
                record,
                figure_name,
                layout_profile=layout_profile,
                prediction_direction=prediction_direction,
                round_idx=round_idx,
            )
            if png is not None:
                return png
        sample_idx = int(record["sample_idx"])
        raise FigureNotReadyError(
            f"sample figure rendering: {figure_name} sample={sample_idx} layout={layout_profile}"
        )

    def get_context_png(self, record: dict, part: str, ctx: ContextParams) -> bytes:
        png = self._engine.get_context_png(record, part, ctx)
        if png is not None:
            return png
        sample_idx = int(record["sample_idx"])
        raise FigureNotReadyError(
            f"context figure rendering: {part} sample={sample_idx} direction={ctx.direction}"
        )

    def wait_context_png(self, record: dict, part: str, ctx: ContextParams, wait_ms: int = 0) -> bytes:
        png = self._engine.get_context_png(record, part, ctx)
        if png is not None:
            return png
        if wait_ms > 0:
            sample_idx = int(record["sample_idx"])
            self._scheduler.schedule_records(
                [record],
                ctx,
                priority_samples={sample_idx},
                replace=False,
            )
            key = self._engine.context_cache_key(
                int(ctx.round_idx),
                sample_idx,
                ctx.direction,
                part,
                layout_profile=ctx.layout_profile,
            )
            expected_epoch = self._dispatch.epoch()
            self._dispatch.wait_for_keys([key], wait_ms, expected_epoch=expected_epoch)
            png = self._engine.get_context_png(record, part, ctx)
            if png is not None:
                return png
        sample_idx = int(record["sample_idx"])
        raise FigureNotReadyError(
            f"context figure rendering: {part} sample={sample_idx} direction={ctx.direction}"
        )

    def sample_ready(
        self,
        sample_idx: int,
        *,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
    ) -> bool:
        return self._engine.sample_ready(round_idx, sample_idx, layout_profile=layout_profile)

    def context_ready(
        self,
        sample_idx: int,
        direction: str,
        *,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
    ) -> bool:
        return self._engine.context_ready(round_idx, sample_idx, direction, layout_profile=layout_profile)

    def bundle_ready(
        self,
        sample_idx: int,
        direction: str,
        record: dict | None = None,
        *,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
    ) -> bool:
        return self._engine.bundle_ready(
            round_idx,
            sample_idx,
            direction,
            record=record,
            layout_profile=layout_profile,
        )

    def get_wind_stats(self, record: dict, round_idx: int = 1) -> dict:
        return self._engine.get_wind_stats(record, round_idx=round_idx)

    def wait_bundle_ready(
        self,
        sample_idx: int,
        direction: str,
        record: dict | None = None,
        *,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
        wait_ms: int = 0,
        ctx: ContextParams | None = None,
    ) -> bool:
        if self.bundle_ready(
            sample_idx,
            direction,
            record=record,
            round_idx=round_idx,
            layout_profile=layout_profile,
        ):
            return True
        if wait_ms <= 0:
            return False
        if record is not None:
            if ctx is not None:
                self._scheduler.schedule_records(
                    [record],
                    ctx,
                    priority_samples={sample_idx},
                    replace=False,
                )
            else:
                self._engine.preload_sample(record, layout_profile=layout_profile, round_idx=round_idx)
        keys = self._engine.sample_bundle_keys(round_idx, sample_idx, layout_profile) + [
            self._engine.context_cache_key(round_idx, sample_idx, direction, "timeseries", layout_profile),
            self._engine.context_cache_key(round_idx, sample_idx, direction, "spectrogram", layout_profile),
        ]
        expected_epoch = self._dispatch.epoch()
        self._dispatch.wait_for_keys(keys, wait_ms, expected_epoch=expected_epoch)
        return self.bundle_ready(
            sample_idx,
            direction,
            record=record,
            round_idx=round_idx,
            layout_profile=layout_profile,
        )

    def on_user_jump_reset(self) -> int:
        self._engine.cancel_pending()
        self._engine.clear_wind_stats()
        self._cache.clear()
        self._scheduler.reset_generation()
        return self._dispatch.mark_jump_reset()
