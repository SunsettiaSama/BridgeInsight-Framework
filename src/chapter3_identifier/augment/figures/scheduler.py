from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.chapter3_identifier.augment.figures.engine import FigureRenderEngine
from src.chapter3_identifier.augment.figures.types import ContextParams


@dataclass(order=True)
class _PreloadJob:
    sort_key: Tuple[float, int, int]
    sample_idx: int = field(compare=False)
    record: dict = field(compare=False)
    ctx: ContextParams = field(compare=False)
    generation: int = field(compare=False)
    queue_key: Tuple[int, int, str, str] = field(compare=False)


class FigureScheduler:
    """专用调度线程 aug-fig-scheduler：按优先级排队，投递给渲染引擎。"""

    def __init__(self, engine: FigureRenderEngine) -> None:
        self._engine = engine
        self._cv = threading.Condition()
        self._heap: List[_PreloadJob] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="aug-fig-scheduler", daemon=True)
        self._seq = 0
        self._generation = 0
        self._queued: Dict[Tuple[int, int, str, str], int] = {}
        self._index_by_key: Dict[Tuple[int, int, str, str], int] = {}
        self._ordered_keys: List[Tuple[int, int, str, str]] = []
        self._completed = 0
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._started:
            self._thread.join(timeout=5.0)

    def stats(self) -> dict:
        with self._cv:
            return {
                "pending": len(self._heap),
                "queued_samples": len(self._queued),
                "completed": self._completed,
                "generation": self._generation,
                "ordered_queue": [
                    {
                        "round_idx": key[0],
                        "sample_idx": key[1],
                        "direction": key[2],
                        "layout_profile": key[3],
                        "order": self._index_by_key.get(key),
                    }
                    for key in self._ordered_keys[:20]
                    if key in self._queued
                ],
            }

    def reset_generation(self) -> int:
        with self._cv:
            self._generation += 1
            self._heap.clear()
            self._queued.clear()
            self._index_by_key.clear()
            self._ordered_keys.clear()
            self._cv.notify_all()
            return self._generation

    def schedule_records(
        self,
        records: List[dict],
        ctx: ContextParams,
        *,
        replace: bool = False,
        priority_samples: Optional[Set[int]] = None,
    ) -> int:
        priority_samples = priority_samples or set()
        candidates: List[tuple[int, int, dict, Tuple[int, int, str, str]]] = []
        for rank, record in enumerate(records):
            sample_idx = int(record["sample_idx"])
            if self._engine.bundle_ready(
                int(ctx.round_idx),
                sample_idx,
                ctx.direction,
                record,
                layout_profile=ctx.layout_profile,
            ):
                continue
            queue_key = (int(ctx.round_idx), sample_idx, ctx.direction, ctx.layout_profile)
            candidates.append((rank, sample_idx, record, queue_key))

        with self._cv:
            if replace:
                generation = self._generation + 1
                self._generation = generation
                self._heap.clear()
                self._queued.clear()
                generation = self._generation
                self._index_by_key.clear()
                self._ordered_keys.clear()
            else:
                generation = self._generation
            queued = 0
            for rank, sample_idx, record, queue_key in candidates:
                self._seq += 1
                priority_rank = 0 if sample_idx in priority_samples else 1
                job = _PreloadJob(
                    sort_key=(priority_rank, rank, self._seq),
                    sample_idx=sample_idx,
                    record=record,
                    ctx=ctx,
                    generation=generation,
                    queue_key=queue_key,
                )
                if self._queued.get(job.queue_key) == job.generation:
                    continue
                heapq.heappush(self._heap, job)
                self._queued[job.queue_key] = job.generation
                self._index_by_key[job.queue_key] = int(rank)
                self._ordered_keys.append(job.queue_key)
                queued += 1
            if queued:
                self._cv.notify_all()
            return queued

    def schedule_by_indices(
        self,
        lookup: Callable[[int], Optional[dict]],
        sample_indices: List[int],
        ctx: ContextParams,
        *,
        priority_samples: Optional[Set[int]] = None,
        replace: bool = False,
    ) -> int:
        records: List[dict] = []
        for sample_idx in sample_indices:
            record = lookup(int(sample_idx))
            if record is not None:
                records.append(record)
        return self.schedule_records(
            records,
            ctx,
            priority_samples=priority_samples,
            replace=replace,
        )

    def _worker(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._heap and not self._stop.is_set():
                    self._cv.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                job = heapq.heappop(self._heap)

            if job.generation != self._generation:
                with self._cv:
                    if self._queued.get(job.queue_key) == job.generation:
                        self._queued.pop(job.queue_key, None)
                        self._index_by_key.pop(job.queue_key, None)
                continue

            self._engine.preload_bundle(job.record, job.ctx)

            with self._cv:
                if self._queued.get(job.queue_key) == job.generation:
                    self._queued.pop(job.queue_key, None)
                    self._index_by_key.pop(job.queue_key, None)
                self._completed += 1
