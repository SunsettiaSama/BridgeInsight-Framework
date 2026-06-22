from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from src.chapter3_identifier.augment.features.wind_index import resolve_wind_meta
from src.chapter3_identifier.augment.figures.layout import sample_fig_size
from src.chapter3_identifier.augment.settings import load_config
from src.config.sensor_config import WIND_DIR_CORRECTION, WIND_FS, WIND_SENSOR_NAMES, WIND_TIME_WINDOW, WIND_VALID_THRESHOLD
from src.data_processer.preprocess.get_data_wind import parse_single_metadata_to_wind_data

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_CFG = load_config()
_WIND_FIG_DPI = int(_CFG.get("figure_sample_dpi", 96))
_USE_TIGHT_BBOX = bool(_CFG.get("figure_export_tight_bbox", False))
@dataclass
class WindRenderPayload:
    wind_speed: np.ndarray
    wind_time_s: np.ndarray
    window_start_s: float
    window_end_s: float
    avg_wind_speed: float
    avg_turbulence: float
    avg_wind_direction: float | None
    sensor_id: str
    title: str
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


def calculate_mean_wind_direction(wind_direction_group: np.ndarray) -> float | None:
    if wind_direction_group.size == 0:
        return None
    radians = np.deg2rad(wind_direction_group)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if abs(sin_mean) <= 1e-12 and abs(cos_mean) <= 1e-12:
        return None
    return float(np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0))


def _extract_window_arrays(
    wind_speed: np.ndarray,
    wind_direction: np.ndarray,
    window_index: int,
    max_missing_ratio: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    window_samples = int(WIND_TIME_WINDOW * WIND_FS)
    start_idx = int(window_index * window_samples)
    end_idx = start_idx + window_samples
    data_len = int(len(wind_speed))
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
    valid = speed > WIND_VALID_THRESHOLD
    speed_valid = speed[valid]
    if speed_valid.size < 2:
        raise ValueError("当前窗口有效风样本不足")
    direction_valid = np.asarray([], dtype=np.float32)
    if wind_direction.size >= clipped_end:
        direction = np.asarray(wind_direction[clipped_start:clipped_end], dtype=np.float32)
        direction_valid = direction[valid]
    return speed_valid, direction_valid, coverage, clipped_start, clipped_end


def _extract_context_speed(
    wind_speed: np.ndarray,
    clipped_start: int,
    clipped_end: int,
    before: int = 3,
    after: int = 3,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    window_samples = int(WIND_TIME_WINDOW * WIND_FS)
    left_need = max(0, int(before) * window_samples)
    right_need = max(0, int(after) * window_samples)
    data_len = int(len(wind_speed))
    context_start = max(0, int(clipped_start) - left_need)
    context_end = min(data_len, int(clipped_end) + right_need)
    context_speed = np.asarray(wind_speed[context_start:context_end], dtype=np.float32)
    context_time_s = np.arange(int(context_speed.size), dtype=np.float32) / max(float(WIND_FS), 1e-6)
    current_start_s = float(int(clipped_start) - context_start) / max(float(WIND_FS), 1e-6)
    current_end_s = float(int(clipped_end) - context_start) / max(float(WIND_FS), 1e-6)
    return context_speed, context_time_s, current_start_s, current_end_s


def build_wind_render_payload(record: dict, cfg: Optional[dict] = None) -> WindRenderPayload:
    cfg = cfg or _CFG
    wind_meta = resolve_wind_meta(record, cfg)
    if wind_meta is None:
        raise ValueError("未找到与样本时间戳对齐的风元数据")

    parsed = parse_single_metadata_to_wind_data(wind_meta, enable_denoise=False)
    data = parsed.get("data") or {}
    wind_speed = np.asarray(data.get("wind_speed", []), dtype=np.float32)
    wind_direction = np.asarray(data.get("wind_direction", []), dtype=np.float32)
    if wind_speed.size == 0:
        raise ValueError("风数据文件为空")

    window_index = int(record.get("window_index", 0))
    speed_valid, direction_valid, coverage, clipped_start, clipped_end = _extract_window_arrays(
        wind_speed,
        wind_direction,
        window_index,
        max_missing_ratio=float(cfg.get("wind_window_max_missing_ratio", 0.3)),
    )
    sensor_id = str(wind_meta.get("sensor_id", "WIND"))
    if direction_valid.size:
        correction = float(WIND_DIR_CORRECTION.get(sensor_id, 360))
        direction_valid = correct_wind_direction(direction_valid, correction)

    avg_speed = float(np.mean(speed_valid))
    avg_ti = calculate_turbulence_intensity(speed_valid)
    if np.isnan(avg_ti):
        avg_ti = 0.0
    avg_direction = calculate_mean_wind_direction(direction_valid)

    ts = record.get("timestamp") or ["--", "--", "--"]
    sensor_name = WIND_SENSOR_NAMES.get(sensor_id, sensor_id)
    coverage_note = "" if coverage >= 0.999 else f" · 覆盖率 {coverage:.0%}"
    title = (
        f"{sensor_name} @ {int(ts[0]):02d}/{int(ts[1]):02d} "
        f"{int(ts[2]):02d}:00 (Window {window_index}){coverage_note}"
    )
    context_speed, context_time_s, window_start_s, window_end_s = _extract_context_speed(
        wind_speed,
        clipped_start,
        clipped_end,
        before=int(cfg.get("context_windows_before", 3)),
        after=int(cfg.get("context_windows_after", 3)),
    )
    return WindRenderPayload(
        wind_speed=context_speed,
        wind_time_s=context_time_s,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        avg_wind_speed=avg_speed,
        avg_turbulence=float(avg_ti),
        avg_wind_direction=avg_direction,
        sensor_id=sensor_id,
        title=title,
        window_coverage=float(coverage),
    )


def _rolling_nanmean(values: np.ndarray, window_size: int) -> np.ndarray:
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    window = np.ones(max(1, int(window_size)), dtype=np.float64)
    sums = np.convolve(filled, window, mode="same")
    counts = np.convolve(valid.astype(np.float64), window, mode="same")
    return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float64), where=counts > 0)


def plot_wind_speed_timeseries(payload: WindRenderPayload, layout_profile: str = "wide_fill_v3") -> bytes:
    fig, ax = plt.subplots(figsize=sample_fig_size(layout_profile, "wind_timeseries"))
    time_s = payload.wind_time_s
    speed = np.asarray(payload.wind_speed, dtype=np.float32)
    valid = speed > WIND_VALID_THRESHOLD
    plot_speed = np.where(valid, speed, np.nan)
    mean_window = max(1, int(round(float(WIND_TIME_WINDOW) * float(WIND_FS))))
    mean_trend = _rolling_nanmean(plot_speed, mean_window)
    ax.plot(time_s, plot_speed, color="#2563eb", linewidth=0.7, alpha=0.78, label="原始风速 u")
    ax.fill_between(
        time_s,
        plot_speed,
        mean_trend,
        where=np.isfinite(plot_speed) & np.isfinite(mean_trend),
        color="#93c5fd",
        alpha=0.22,
        label="脉动分量 u'",
    )
    ax.plot(time_s, mean_trend, color="#dc2626", linewidth=1.3, label="平均风速趋势 U")
    ax.axvspan(payload.window_start_s, payload.window_end_s, color="#93c5fd", alpha=0.28, label="当前窗口")
    ax.set_title(f"风速时程 · {payload.title}", fontsize=8)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("风速 (m/s)", fontsize=8)
    ax.grid(True, alpha=0.25)
    y_max = float(np.nanmax(plot_speed)) if np.any(np.isfinite(plot_speed)) else 1.0
    ax.set_ylim(0.0, max(1.0, y_max * 1.15))
    if time_s.size:
        ax.set_xlim(0.0, float(time_s[-1]))
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_wind_direction(payload: WindRenderPayload, layout_profile: str = "wide_fill_v3") -> bytes:
    fig, ax = plt.subplots(
        figsize=sample_fig_size(layout_profile, "square"),
        subplot_kw={"projection": "polar"},
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)
    ax.set_title("平均风向", fontsize=8, pad=10)
    if payload.avg_wind_direction is None:
        ax.text(0.5, 0.5, "风向不可用", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        fig.tight_layout()
        return _fig_to_bytes(fig)
    theta = np.deg2rad(payload.avg_wind_direction)
    ax.annotate(
        "",
        xy=(theta, 0.08),
        xytext=(theta, 0.92),
        arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": "#dc2626"},
    )
    ax.scatter([0.0], [0.0], s=20, color="#111827", zorder=5)
    ax.text(
        0.5,
        -0.10,
        f"{payload.avg_wind_direction:.0f}°",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#dc2626",
        fontweight="bold",
    )
    fig.tight_layout()
    return _fig_to_bytes(fig)


def wind_stats_to_dict(payload: WindRenderPayload) -> dict:
    sensor_name = WIND_SENSOR_NAMES.get(payload.sensor_id, payload.sensor_id)
    return {
        "ready": True,
        "avg_wind_speed": round(float(payload.avg_wind_speed), 2),
        "avg_turbulence": round(float(payload.avg_turbulence), 2),
        "avg_wind_direction": None
        if payload.avg_wind_direction is None
        else round(float(payload.avg_wind_direction), 1),
        "sensor_id": payload.sensor_id,
        "sensor_name": sensor_name,
        "title": payload.title,
        "window_coverage": round(float(payload.window_coverage), 3),
    }


def render_wind_figure_from_payload(
    payload: WindRenderPayload, figure_name: str, layout_profile: str = "wide_fill_v3"
) -> bytes:
    if figure_name == "wind_direction":
        return plot_wind_direction(payload, layout_profile=layout_profile)
    if figure_name == "wind_speed_timeseries":
        return plot_wind_speed_timeseries(payload, layout_profile=layout_profile)
    raise ValueError(f"未知风特征图类型: {figure_name}")
