from __future__ import annotations

from dataclasses import dataclass


SAMPLE_FIGURE_NAMES = (
    "in_timeseries",
    "out_timeseries",
    "in_spectrum",
    "out_spectrum",
    "trajectory",
    "prediction",
)


class FigureNotReadyError(Exception):
    pass


@dataclass(frozen=True)
class ContextParams:
    direction: str = "inplane"
    layout_profile: str = "wide_fill_v1"
    windows_before: int = 3
    windows_after: int = 3
    spectrogram_segment_s: float = 2.0
    figure_cache_size: int = 30
