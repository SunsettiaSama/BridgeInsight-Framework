from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.chapter3_identifier.augment.features.spectrum import welch_psd
from src.chapter3_identifier.augment.figures.render.titles import format_sample_title
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.settings import load_config
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_WINDOW_SIZE = 3000
_FS = 50.0
_NFFT = 2048
_FREQ_MAX = 25.0
_LABELS = get_label_names(load_config())
_SAMPLE_LAYOUT_PRESETS = {
    "wide_fill_v1": {
        "timeseries": (6.8, 1.9),
        "spectrum": (4.0, 2.0),
        "trajectory": (4.0, 4.0),
        "prediction": (8.2, 1.7),
    },
    "default": {
        "timeseries": (4.0, 2.0),
        "spectrum": (4.0, 2.0),
        "trajectory": (4.0, 4.0),
        "prediction": (4.0, 2.5),
    },
}


def _sample_fig_size(layout_profile: str, kind: str) -> tuple[float, float]:
    preset = _SAMPLE_LAYOUT_PRESETS.get(layout_profile, _SAMPLE_LAYOUT_PRESETS["wide_fill_v1"])
    return preset[kind]


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _load_window(file_path: str, window_index: int, extractor: VICWindowExtractor) -> np.ndarray:
    data = extractor.load_file(file_path)
    sig = extractor.extract_window_from_data(
        data, window_index, _WINDOW_SIZE, metadata={"window_index": window_index}, file_path=file_path
    )
    return np.asarray(sig, dtype=np.float32).reshape(-1)


def plot_timeseries(data: np.ndarray, title: str, layout_profile: str = "wide_fill_v1") -> bytes:
    t = np.arange(len(data)) / _FS
    fig, ax = plt.subplots(figsize=_sample_fig_size(layout_profile, "timeseries"))
    ax.plot(t, data, color="#333333", linewidth=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel(r"加速度 ($m/s^2$)")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_spectrum(data: np.ndarray, title: str, layout_profile: str = "wide_fill_v1") -> bytes:
    f, psd = welch_psd(data, fs=_FS, nfft=_NFFT, freq_max_hz=_FREQ_MAX)
    fig, ax = plt.subplots(figsize=_sample_fig_size(layout_profile, "spectrum"))
    ax.plot(f, psd, color="#333333", linewidth=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_trajectory(
    in_data: np.ndarray,
    out_data: np.ndarray,
    title: str,
    in_id: str = "",
    out_id: str = "",
    layout_profile: str = "wide_fill_v1",
) -> bytes:
    if in_id or out_id:
        title = f"面内: {in_id or '-'}  |  面外: {out_id or '-'}\n{title}"
    all_vals = np.concatenate([in_data, out_data])
    g_min, g_max = float(all_vals.min()), float(all_vals.max())
    margin = (g_max - g_min) * 0.05 if g_max > g_min else 0.05

    fig, ax = plt.subplots(figsize=_sample_fig_size(layout_profile, "trajectory"))
    ax.scatter(out_data, in_data, s=8, alpha=0.35, linewidths=0, color="#4f8ef7")
    ax.set_xlim(g_min - margin, g_max + margin)
    ax.set_ylim(g_min - margin, g_max + margin)
    ax.set_xlabel(r"面外加速度 ($m/s^2$)")
    ax.set_ylabel(r"面内加速度 ($m/s^2$)")
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_prediction_bar(
    proba: List[float],
    prediction: int,
    title: str,
    layout_profile: str = "wide_fill_v1",
) -> bytes:
    fig, ax = plt.subplots(figsize=_sample_fig_size(layout_profile, "prediction"))
    x = np.arange(len(proba))
    colors = ["#888888"] * len(proba)
    if 0 <= prediction < len(proba):
        colors[prediction] = "#4f8ef7"
    ax.bar(x, proba, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS, rotation=12)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.margins(x=0.06)
    fig.tight_layout()
    return _fig_to_bytes(fig)


@dataclass
class SampleRenderData:
    in_data: np.ndarray
    out_data: np.ndarray
    in_title: str
    out_title: str
    in_pred_proba: list[float]
    in_pred_cls: int
    out_pred_proba: list[float]
    out_pred_cls: int


def _direction_prediction(record: dict, direction: str) -> tuple[list[float], int]:
    dir_key = "outplane" if direction == "outplane" else "inplane"
    dir_proba = record.get(f"{dir_key}_proba")
    dir_pred = record.get(f"{dir_key}_prediction")
    if dir_proba is None:
        dir_proba = record.get("proba", [0.25, 0.25, 0.25, 0.25])
    if dir_pred is None:
        dir_pred = record.get("prediction", 0)
    return [float(x) for x in dir_proba], int(dir_pred)


def render_prediction_figure(
    record: dict,
    prediction_direction: str = "inplane",
    layout_profile: str = "wide_fill_v1",
) -> bytes:
    proba, pred = _direction_prediction(record, prediction_direction)
    label = "面外" if prediction_direction == "outplane" else "面内"
    return plot_prediction_bar(proba, pred, f"模型预测（{label}）", layout_profile=layout_profile)


def build_sample_render_data(record: dict) -> SampleRenderData:
    extractor = VICWindowExtractor(enable_denoise=False)
    in_fp = record.get("inplane_file_path")
    out_fp = record.get("outplane_file_path")
    wi = int(record.get("window_index", 0))
    in_id = record.get("inplane_sensor_id", "in")
    out_id = record.get("outplane_sensor_id", "out")
    ts = record.get("timestamp")

    in_data = _load_window(in_fp, wi, extractor) if in_fp else np.zeros(_WINDOW_SIZE, dtype=np.float32)
    out_data = _load_window(out_fp, wi, extractor) if out_fp else np.zeros(_WINDOW_SIZE, dtype=np.float32)
    in_title = format_sample_title("面内", in_id, in_fp, wi, ts)
    out_title = format_sample_title("面外", out_id, out_fp, wi, ts)
    in_pred_proba, in_pred_cls = _direction_prediction(record, "inplane")
    out_pred_proba, out_pred_cls = _direction_prediction(record, "outplane")
    return SampleRenderData(
        in_data=in_data,
        out_data=out_data,
        in_title=in_title,
        out_title=out_title,
        in_pred_proba=in_pred_proba,
        in_pred_cls=in_pred_cls,
        out_pred_proba=out_pred_proba,
        out_pred_cls=out_pred_cls,
    )


def render_sample_figure_from_data(
    sample_data: SampleRenderData,
    figure_name: str,
    layout_profile: str = "wide_fill_v1",
    prediction_direction: str = "inplane",
) -> bytes:
    if figure_name == "in_timeseries":
        return plot_timeseries(sample_data.in_data, f"{sample_data.in_title} 时程", layout_profile=layout_profile)
    if figure_name == "out_timeseries":
        return plot_timeseries(sample_data.out_data, f"{sample_data.out_title} 时程", layout_profile=layout_profile)
    if figure_name == "in_spectrum":
        return plot_spectrum(sample_data.in_data, f"{sample_data.in_title} 频谱", layout_profile=layout_profile)
    if figure_name == "out_spectrum":
        return plot_spectrum(sample_data.out_data, f"{sample_data.out_title} 频谱", layout_profile=layout_profile)
    if figure_name == "trajectory":
        return plot_trajectory(
            sample_data.in_data,
            sample_data.out_data,
            f"{sample_data.in_title}\n{sample_data.out_title}",
            layout_profile=layout_profile,
        )
    if figure_name == "prediction":
        if prediction_direction == "outplane":
            return plot_prediction_bar(
                sample_data.out_pred_proba,
                sample_data.out_pred_cls,
                "模型预测（面外）",
                layout_profile=layout_profile,
            )
        return plot_prediction_bar(
            sample_data.in_pred_proba,
            sample_data.in_pred_cls,
            "模型预测（面内）",
            layout_profile=layout_profile,
        )
    raise ValueError(f"未知样本图类型: {figure_name}")


def render_sample_figures(record: dict, layout_profile: str = "wide_fill_v1") -> dict:
    sample_data = build_sample_render_data(record)
    return {
        name: render_sample_figure_from_data(
            sample_data,
            name,
            layout_profile=layout_profile,
            prediction_direction="inplane",
        )
        for name in ("in_timeseries", "out_timeseries", "in_spectrum", "out_spectrum", "trajectory", "prediction")
    }
