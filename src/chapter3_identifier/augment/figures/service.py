from __future__ import annotations

from typing import Callable, List, Optional, Set

from src.chapter3_identifier.augment.figures.cache import PngCache
from src.chapter3_identifier.augment.figures.engine import FigureRenderEngine
from src.chapter3_identifier.augment.figures.scheduler import FigureScheduler
from src.chapter3_identifier.augment.figures.types import ContextParams, FigureNotReadyError


class FigureService:
    """图像服务 Facade：调度线程 + 渲染线程池 + 内存 PNG 缓存。"""

    def __init__(self, max_workers: int = 2, image_workers: int = 6, cache_size: int = 256) -> None:
        self._cache = PngCache(max_size=cache_size)
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
    ) -> int:
        return self._scheduler.schedule_by_indices(
            lookup,
            sample_indices,
            ctx,
            priority_samples=priority_samples,
        )

    def get_sample_png(
        self,
        record: dict,
        figure_name: str,
        layout_profile: str = "wide_fill_v1",
        prediction_direction: str = "inplane",
    ) -> bytes:
        png = self._engine.get_sample_png(
            record,
            figure_name,
            layout_profile=layout_profile,
            prediction_direction=prediction_direction,
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
