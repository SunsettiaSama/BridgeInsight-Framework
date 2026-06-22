from __future__ import annotations

from dataclasses import dataclass


SAMPLE_FIGURE_NAMES = (
    "in_timeseries",
    "out_timeseries",
    "in_spectrum",
    "out_spectrum",
    "trajectory",
    "prediction",
    "wind_direction",
    "wind_speed_timeseries",
)


class FigureNotReadyError(Exception):
    pass


@dataclass(frozen=True)
class ContextParams:
    direction: str = "inplane"
    round_idx: int = 1
    layout_profile: str = "wide_fill_v3"
    windows_before: int = 3
    windows_after: int = 3
    spectrogram_segment_s: float = 2.0
    figure_cache_size: int = 30
