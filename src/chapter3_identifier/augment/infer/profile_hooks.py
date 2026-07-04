from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from src.chapter3_identifier.augment.infer.prefetch import PrefetchStats


@dataclass
class InferStageTimings:
    preflight_s: float = 0.0
    dataset_s: float = 0.0
    checkpoint_s: float = 0.0
    runner_s: float = 0.0
    merge_s: float = 0.0
    records_s: float = 0.0
    inference_json_s: float = 0.0
    enriched_json_s: float = 0.0
    prefetch: PrefetchStats = field(default_factory=PrefetchStats)
    extra: Dict[str, Any] = field(default_factory=dict)

    def total_s(self) -> float:
        return (
            self.preflight_s
            + self.dataset_s
            + self.checkpoint_s
            + self.runner_s
            + self.merge_s
            + self.records_s
            + self.inference_json_s
            + self.enriched_json_s
        )

    def to_dict(self) -> dict:
        runner_detail = self.prefetch.to_dict()
        total = self.total_s()
        samples = int(self.extra.get("sample_count", 0))
        throughput = samples / self.runner_s if self.runner_s > 0 and samples > 0 else 0.0
        return {
            "stages": {
                "preflight_s": round(self.preflight_s, 4),
                "dataset_s": round(self.dataset_s, 4),
                "checkpoint_s": round(self.checkpoint_s, 4),
                "runner_s": round(self.runner_s, 4),
                "merge_s": round(self.merge_s, 4),
                "records_s": round(self.records_s, 4),
                "inference_json_s": round(self.inference_json_s, 4),
                "enriched_json_s": round(self.enriched_json_s, 4),
                "total_s": round(total, 4),
            },
            "runner_detail": runner_detail,
            "throughput_samples_per_s": round(throughput, 2),
            "extra": dict(self.extra),
        }


class StageTimer:
    def __init__(self):
        self.timings = InferStageTimings()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        setattr(self.timings, f"{name}_s", getattr(self.timings, f"{name}_s") + elapsed)


def gpu_sync_elapsed(start_event: torch.cuda.Event, end_event: torch.cuda.Event) -> float:
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0


@contextmanager
def track_gpu_forward(stats: PrefetchStats, use_cuda: bool):
    if use_cuda and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        stats.gpu_forward_time_s += gpu_sync_elapsed(start, end)
    else:
        t0 = time.perf_counter()
        yield
        stats.gpu_forward_time_s += time.perf_counter() - t0
