from __future__ import annotations

from typing import Callable, Dict, List

PLOT_REGISTRY: List[dict] = [
    {"id": "rms_histogram", "title": "RMS 主体直方图", "classes": [0, 1, 2, 3]},
    {"id": "rms_scatter", "title": "面内外 RMS 散点", "classes": [0, 1, 2, 3]},
    {"id": "kurtosis_histogram", "title": "峭度分布", "classes": [0, 1, 2, 3]},
    {"id": "freq_histogram", "title": "主频分布", "classes": [0, 1, 2, 3]},
    {"id": "freq_energy_scatter", "title": "主频-能量散点", "classes": [0, 1, 2, 3]},
    {"id": "energy_histogram", "title": "能量占比分布", "classes": [0, 1, 2, 3]},
    {"id": "energy_cumsum", "title": "前10阶累积能量", "classes": [0, 1, 2, 3]},
    {"id": "trajectory_cloud", "title": "轨迹云图", "classes": [0, 1, 2, 3]},
    {"id": "ellipticity_hist", "title": "椭圆率分布", "classes": [0, 3]},
    {"id": "wind_rms_scatter", "title": "风速-RMS 关系", "classes": [0, 1, 2, 3]},
    {"id": "class_distribution", "title": "类别分布", "classes": [3]},
]


def registry_for_class(class_id: int) -> List[dict]:
    return [p for p in PLOT_REGISTRY if class_id in p["classes"]]


def get_plot_ids(class_id: int) -> List[str]:
    return [p["id"] for p in registry_for_class(class_id)]
