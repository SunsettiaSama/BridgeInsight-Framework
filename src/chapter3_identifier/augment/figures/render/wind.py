from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from src.chapter3_identifier.augment.features.wind_index import resolve_wind_meta
from src.chapter3_identifier.augment.figures.layout import sample_fig_size
from src.chapter3_identifier.augment.settings import load_config
from src.config.sensor_config import (
    AXIS_OF_BRIDGE,
    WIND_DIR_CORRECTION,
    WIND_FS,
    WIND_SENSOR_NAMES,
    WIND_TIME_WINDOW,
    WIND_VALID_THRESHOLD,
)
from src.data_processer.preprocess.get_data_wind import parse_single_metadata_to_wind_data
from src.figure_paintings.figs_for_thesis.config import get_red_color_map, get_viridis_color_map

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_CFG = load_config()
_WIND_FIG_DPI = int(_CFG.get("figure_sample_dpi", 96))
_USE_TIGHT_BBOX = bool(_CFG.get("figure_export_tight_bbox", False))
_INTERVAL_NUMS = int(_CFG.get("wind_rose_bins", 36))
_TI_CMAP_MIN = float(_CFG.get("wind_ti_cmap_min", 0.0))
_TI_CMAP_MAX = float(_CFG.get("wind_ti_cmap_max", 15.0))


@dataclass
class WindRenderPayload:
    wind_speed: np.ndarray
    wind_direction: np.ndarray
    avg_wind_speed: float
    avg_turbulence: float
    sensor_id: str
    title: str
    theta: np.ndarray
    speed_bin_values: List[float]
    ti_bin_values: List[float]
    count_bins: List[int]
    bin_step: int
    window_coverage: float = 1.0


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    if _USE_TIGHT_BBOX:
        fig.savefig(buf, format="png", dpi=_WIND_FIG_DPI, bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", dpi=_WIND_FIG_DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def calculate_turbulence_intensity(wind_speed_group: np.ndarray) -> float:
    if len(wind_speed_group) < 2:
        return float("nan")
    u_mean = float(np.mean(wind_speed_group))
    if u_mean <= 1e-6:
        return float("nan")
    u_std = float(np.std(wind_speed_group, ddof=1))
    return round(u_std / u_mean * 100.0, 2)


def correct_wind_direction(wind_directions: np.ndarray, correction_val: float) -> np.ndarray:
    corrected = correction_val - wind_directions
    return np.mod(corrected, 360.0)


def _extract_window_arrays(
    wind_speed: np.ndarray,
    wind_direction: np.ndarray,
    window_index: int,
    max_missing_ratio: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float]:
    window_samples = int(WIND_TIME_WINDOW * WIND_FS)
    start_idx = int(window_index * window_samples)
    end_idx = start_idx + window_samples
    data_len = min(int(len(wind_speed)), int(len(wind_direction)))
    expected_hour_samples = int(3600 * WIND_FS)
    file_coverage = data_len / max(expected_hour_samples, 1)
    if data_len < window_samples:
        raise ValueError(
            f"风数据文件过短: coverage={file_coverage:.1%}, len={data_len}, "
            f"window_samples={window_samples}"
        )
    clipped_start = max(0, min(start_idx, data_len))
    clipped_end = max(0, min(end_idx, data_len))
    available = max(0, clipped_end - clipped_start)
    coverage = available / max(window_samples, 1)
    min_coverage = 1.0 - float(max_missing_ratio)
    if coverage < min_coverage and file_coverage >= min_coverage:
        if start_idx >= data_len:
            clipped_start = max(0, data_len - window_samples)
            clipped_end = data_len
        elif end_idx > data_len:
            clipped_end = data_len
            clipped_start = max(0, clipped_end - window_samples)
        elif start_idx < 0:
            clipped_start = 0
            clipped_end = min(data_len, window_samples)
        available = max(0, clipped_end - clipped_start)
        coverage = available / max(window_samples, 1)
    if coverage < min_coverage:
        raise ValueError(
            f"风数据窗口越界过多: window_index={window_index}, "
            f"coverage={coverage:.1%}, file_coverage={file_coverage:.1%}, len={data_len}"
        )
    speed = np.asarray(wind_speed[clipped_start:clipped_end], dtype=np.float32)
    direction = np.asarray(wind_direction[clipped_start:clipped_end], dtype=np.float32)
    valid = speed > WIND_VALID_THRESHOLD
    speed_valid = speed[valid]
    direction_valid = direction[valid]
    if speed_valid.size < 2:
        raise ValueError("当前窗口有效风样本不足")
    return speed_valid, direction_valid, coverage


def _bin_by_direction(
    wind_speed: np.ndarray,
    wind_direction: np.ndarray,
    interval_nums: int = _INTERVAL_NUMS,
) -> tuple[np.ndarray, List[float], List[float], List[int], int]:
    bin_step = int(360 / interval_nums)
    bins = np.arange(0, 360 + bin_step, bin_step)
    digitized = np.digitize(wind_direction, bins)
    speed_bins: List[float] = []
    ti_bins: List[float] = []
    count_bins: List[int] = []
    for i in range(1, len(bins)):
        mask = digitized == i
        count = int(np.sum(mask))
        count_bins.append(count)
        if count <= 0:
            speed_bins.append(0.0)
            ti_bins.append(0.0)
            continue
        group_speed = wind_speed[mask]
        speed_bins.append(float(np.mean(group_speed)))
        ti = calculate_turbulence_intensity(group_speed)
        ti_bins.append(0.0 if np.isnan(ti) else float(ti))
    theta = np.deg2rad(bins[:-1])
    return theta, speed_bins, ti_bins, count_bins, bin_step


def build_wind_render_payload(record: dict, cfg: Optional[dict] = None) -> WindRenderPayload:
    cfg = cfg or _CFG
    wind_meta = resolve_wind_meta(record, cfg)
    if wind_meta is None:
        raise ValueError("未找到与样本时间戳对齐的风元数据")

    parsed = parse_single_metadata_to_wind_data(wind_meta, enable_denoise=False)
    data = parsed.get("data") or {}
    wind_speed = np.asarray(data.get("wind_speed", []), dtype=np.float32)
    wind_direction = np.asarray(data.get("wind_direction", []), dtype=np.float32)
    if wind_speed.size == 0 or wind_direction.size == 0:
        raise ValueError("风数据文件为空")

    window_index = int(record.get("window_index", 0))
    speed_valid, direction_valid, coverage = _extract_window_arrays(
        wind_speed,
        wind_direction,
        window_index,
        max_missing_ratio=float(cfg.get("wind_window_max_missing_ratio", 0.3)),
    )
    sensor_id = str(wind_meta.get("sensor_id", "WIND"))
    correction = float(WIND_DIR_CORRECTION.get(sensor_id, 360))
    direction_valid = correct_wind_direction(direction_valid, correction)

    avg_speed = float(np.mean(speed_valid))
    avg_ti = calculate_turbulence_intensity(speed_valid)
    if np.isnan(avg_ti):
        avg_ti = 0.0

    theta, speed_bins, ti_bins, count_bins, bin_step = _bin_by_direction(speed_valid, direction_valid)
    ts = record.get("timestamp") or ["--", "--", "--"]
    sensor_name = WIND_SENSOR_NAMES.get(sensor_id, sensor_id)
    coverage_note = "" if coverage >= 0.999 else f" · 覆盖率 {coverage:.0%}"
    title = (
        f"{sensor_name} @ {int(ts[0]):02d}/{int(ts[1]):02d} "
        f"{int(ts[2]):02d}:00 (Window {window_index}){coverage_note}"
    )
    return WindRenderPayload(
        wind_speed=speed_valid,
        wind_direction=direction_valid,
        avg_wind_speed=avg_speed,
        avg_turbulence=float(avg_ti),
        sensor_id=sensor_id,
        title=title,
        theta=theta,
        speed_bin_values=speed_bins,
        ti_bin_values=ti_bins,
        count_bins=count_bins,
        bin_step=bin_step,
        window_coverage=float(coverage),
    )


def _plot_polar_rose_base(
    theta: np.ndarray, bin_step: int, title: str, layout_profile: str = "wide_fill_v3"
) -> tuple:
    fig, ax = plt.subplots(
        figsize=sample_fig_size(layout_profile, "wind_rose"),
        subplot_kw={"projection": "polar"},
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    bridge_theta1 = np.deg2rad(AXIS_OF_BRIDGE)
    bridge_theta2 = np.deg2rad(AXIS_OF_BRIDGE + 180)
    ax.plot([bridge_theta1, bridge_theta1], [0, 1.0], color="red", linestyle="--", linewidth=1.2)
    ax.plot([bridge_theta2, bridge_theta2], [0, 1.0], color="red", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=8, pad=12)
    return fig, ax


def plot_wind_speed_rose(payload: WindRenderPayload, layout_profile: str = "wide_fill_v3") -> bytes:
    values = np.array(payload.speed_bin_values, dtype=np.float64)
    y_max = float(np.max(values)) if values.size else 1.0
    if y_max <= 0:
        y_max = 1.0
    fig, ax = _plot_polar_rose_base(
        payload.theta,
        payload.bin_step,
        f"风速玫瑰图 · {payload.title}",
        layout_profile=layout_profile,
    )
    bars = ax.bar(
        payload.theta,
        values,
        width=np.deg2rad(payload.bin_step),
        bottom=0.0,
        alpha=0.85,
        align="edge",
        color="#4f8ef7",
    )
    for _ in bars:
        pass
    ax.set_ylim(0, y_max * 1.15)
    y_tick_interval = max(0.5, round(y_max / 5, 1))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    ax.set_rlabel_position(270)
    ax.text(
        0.5,
        -0.12,
        f"平均风速: {payload.avg_wind_speed:.2f} m/s",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_wind_turbulence_rose(
    payload: WindRenderPayload, layout_profile: str = "wide_fill_v3"
) -> bytes:
    counts = np.array(payload.count_bins, dtype=np.float64)
    total = float(np.sum(counts))
    if total > 0:
        heights = counts / total * 100.0
    else:
        heights = counts
    y_max = float(np.max(heights)) if heights.size else 1.0
    if y_max <= 0:
        y_max = 1.0

    cmap = get_red_color_map(style="gradient")
    ti_values = np.array(payload.ti_bin_values, dtype=np.float64)
    ti_max = max(_TI_CMAP_MAX, float(payload.avg_turbulence), float(np.max(ti_values)) if ti_values.size else 0.0)
    ti_max = max(_TI_CMAP_MAX, float(np.ceil(ti_max / 5.0) * 5.0))
    norm = plt.Normalize(_TI_CMAP_MIN, ti_max)
    fig, ax = _plot_polar_rose_base(
        payload.theta,
        payload.bin_step,
        f"紊流度玫瑰图 · {payload.title}",
        layout_profile=layout_profile,
    )
    bars = ax.bar(
        payload.theta,
        heights,
        width=np.deg2rad(payload.bin_step),
        bottom=0.0,
        alpha=0.85,
        align="edge",
    )
    for bar, ti in zip(bars, payload.ti_bin_values):
        bar.set_facecolor(cmap(norm(ti)))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.08, shrink=0.82)
    cbar.set_label("紊流度 (%)", fontsize=7)
    ax.set_ylim(0, y_max * 1.15)
    y_tick_interval = max(2, round(y_max / 5))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    ax.set_rlabel_position(270)
    ax.text(
        0.5,
        -0.12,
        f"整窗紊流度: {payload.avg_turbulence:.2f}%",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    return _fig_to_bytes(fig)


def wind_stats_to_dict(payload: WindRenderPayload) -> dict:
    sensor_name = WIND_SENSOR_NAMES.get(payload.sensor_id, payload.sensor_id)
    return {
        "ready": True,
        "avg_wind_speed": round(float(payload.avg_wind_speed), 2),
        "avg_turbulence": round(float(payload.avg_turbulence), 2),
        "sensor_id": payload.sensor_id,
        "sensor_name": sensor_name,
        "title": payload.title,
        "window_coverage": round(float(payload.window_coverage), 3),
    }


def render_wind_figure_from_payload(
    payload: WindRenderPayload, figure_name: str, layout_profile: str = "wide_fill_v3"
) -> bytes:
    if figure_name == "wind_speed_rose":
        return plot_wind_speed_rose(payload, layout_profile=layout_profile)
    if figure_name == "wind_turbulence_rose":
        return plot_wind_turbulence_rose(payload, layout_profile=layout_profile)
    raise ValueError(f"未知风特征图类型: {figure_name}")
