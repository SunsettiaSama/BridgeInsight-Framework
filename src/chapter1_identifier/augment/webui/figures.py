from __future__ import annotations

import io
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from src.chapter1_identifier.augment.features.spectrum import welch_psd
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_WINDOW_SIZE = 3000
_FS = 50.0
_NFFT = 2048
_FREQ_MAX = 25.0
_LABELS = ["Normal", "VIV", "RWIV", "Transition"]


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


def plot_timeseries(data: np.ndarray, title: str) -> bytes:
    t = np.arange(len(data)) / _FS
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(t, data, color="#333333", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("加速度 (m/s²)")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_spectrum(data: np.ndarray, title: str) -> bytes:
    f, psd = welch_psd(data, fs=_FS, nfft=_NFFT, freq_max_hz=_FREQ_MAX)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(f, psd, color="#333333", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_trajectory(in_data: np.ndarray, out_data: np.ndarray, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(out_data, in_data, s=8, alpha=0.35, linewidths=0, color="#4f8ef7")
    ax.set_xlabel("面外 (m/s²)")
    ax.set_ylabel("面内 (m/s²)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_prediction_bar(proba: List[float], prediction: int, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    x = np.arange(len(proba))
    colors = ["#888888"] * len(proba)
    if 0 <= prediction < len(proba):
        colors[prediction] = "#4f8ef7"
    ax.bar(x, proba, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(_LABELS, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def render_sample_figures(record: dict) -> dict:
    extractor = VICWindowExtractor(enable_denoise=False)
    in_fp = record.get("inplane_file_path")
    out_fp = record.get("outplane_file_path")
    wi = int(record.get("window_index", 0))
    in_id = record.get("inplane_sensor_id", "in")
    out_id = record.get("outplane_sensor_id", "out")

    in_data = _load_window(in_fp, wi, extractor) if in_fp else np.zeros(_WINDOW_SIZE, dtype=np.float32)
    out_data = _load_window(out_fp, wi, extractor) if out_fp else np.zeros(_WINDOW_SIZE, dtype=np.float32)

    proba = record.get("proba", [0.25, 0.25, 0.25, 0.25])
    pred = int(record.get("prediction", 0))

    return {
        "in_timeseries": plot_timeseries(in_data, f"面内时程 {in_id}"),
        "out_timeseries": plot_timeseries(out_data, f"面外时程 {out_id}"),
        "in_spectrum": plot_spectrum(in_data, f"面内频谱 {in_id}"),
        "out_spectrum": plot_spectrum(out_data, f"面外频谱 {out_id}"),
        "trajectory": plot_trajectory(in_data, out_data, "面内-面外轨迹"),
        "prediction": plot_prediction_bar(proba, pred, "模型预测"),
    }
