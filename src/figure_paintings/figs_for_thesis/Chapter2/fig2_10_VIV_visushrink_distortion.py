import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import signal

_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import DEFAULT_PORT as DASHBOARD_DEFAULT_PORT
from src.visualize_tools.web_dashboard import push as web_push

_paths_cfg = yaml.safe_load((_project_root / "config" / "io" / "paths.yaml").read_text(encoding="utf-8"))
WEBUI_PAGE = "fig2_10 VisuShrink VIV 失真"
FIGURE_SPECS = (
    ("fig2_10a_VIV_visushrink_timeseries.png", "图2.10(a) VisuShrink 前后 VIV 时域波形对比"),
    ("fig2_10b_VIV_visushrink_psd.png", "图2.10(b) VisuShrink 前后 VIV 频域能量对比"),
    ("fig2_10c_VIV_visushrink_residual.png", "图2.10(c) VisuShrink 移除成分的时域残差"),
)


class Config:
    FS = 50.0
    WINDOW_SIZE = 3000
    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 12
    MAX_CANDIDATES = 80

    WAVELET_TYPE = "db4"
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = "soft"
    THRESHOLD_METHOD = "sqtwolog"
    LAYER_WISE_THRESHOLD = True

    NPERSEG = 1024
    FREQ_MAX_PLOT = FS / 2

    FIG_SIZE = REC_FIG_SIZE
    TITLE_FONTSIZE = FONT_SIZE
    AXIS_FONTSIZE = FONT_SIZE
    TICK_FONTSIZE = FONT_SIZE
    LEGEND_FONTSIZE = FONT_SIZE - 4
    LABELPAD = 10

    COLOR_ORIGINAL = VIV_INPLANE_COLOR
    COLOR_DENOISED = VIV_OUTPLANE_COLOR
    COLOR_RESIDUAL = "#333333"
    COLOR_ZERO_LINE = "#202020"

    LINEWIDTH_SIGNAL = 1.0
    LINEWIDTH_PSD = 1.2
    LINEWIDTH_RESIDUAL = 1.0
    LINEWIDTH_ZERO = 0.8
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = "--"

    OUTPUT_DIR = _project_root / "results" / "webui" / "figs_for_thesis" / "Chapter2"
    LEGACY_OUTPUT_NAME = "fig2_10_VIV_visushrink_distortion.png"
    WEBUI_PORT = DASHBOARD_DEFAULT_PORT


def load_viv_windows():
    annotation_file = _project_root / _paths_cfg["annotation"]["viv_sample"]
    if not annotation_file.exists():
        raise ValueError(f"找不到标注文件: {annotation_file}")

    annotation_data = json.loads(annotation_file.read_text(encoding="utf-8"))
    viv_records = [item for item in annotation_data if item.get("annotation") == "1"]
    if not viv_records:
        raise ValueError("无 VIV 标注的记录，无法生成图2.10")

    unpacker = UNPACK(init_path=False)
    windows = []
    for record in viv_records[: Config.MAX_CANDIDATES]:
        metadata = record["metadata"]
        file_path = metadata["file_path"]
        window_idx = record["window_index"]
        vibration_data = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=float)

        start_sample = window_idx * Config.WINDOW_SIZE
        end_sample = (window_idx + 1) * Config.WINDOW_SIZE
        if end_sample <= len(vibration_data):
            windows.append(
                {
                    "data": vibration_data[start_sample:end_sample],
                    "sensor_id": record["sensor_id"],
                    "time": record["time"],
                    "window_index": window_idx,
                    "file_path": file_path,
                }
            )

    if not windows:
        raise ValueError("VIV 标注存在，但没有可截取的完整 60 s 窗口")

    return windows


def apply_visushrink(data):
    denoised, info = denoise(
        signal=data,
        wavelet=Config.WAVELET_TYPE,
        level=Config.WAVELET_LEVEL,
        threshold_type=Config.THRESHOLD_TYPE,
        threshold_method=Config.THRESHOLD_METHOD,
        layer_wise_threshold=Config.LAYER_WISE_THRESHOLD,
    )
    return np.asarray(denoised, dtype=float), info


def compute_psd(data):
    nperseg = min(Config.NPERSEG, len(data))
    freqs, psd = signal.welch(data, fs=Config.FS, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def distortion_score(original, denoised):
    residual = original - denoised
    rms_original = np.sqrt(np.mean(original**2))
    rms_residual = np.sqrt(np.mean(residual**2))

    freqs, psd_original = compute_psd(original)
    _, psd_denoised = compute_psd(denoised)
    viv_band = (freqs >= 0.1) & (freqs <= 5.0)
    band_loss = 1 - np.trapz(psd_denoised[viv_band], freqs[viv_band]) / np.trapz(psd_original[viv_band], freqs[viv_band])

    return rms_residual / rms_original + max(float(band_loss), 0.0)


def select_most_distorted_window(windows):
    evaluated = []
    for window in windows:
        original = np.asarray(window["data"], dtype=float)
        denoised, info = apply_visushrink(original)
        evaluated.append((distortion_score(original, denoised), window, denoised))

    evaluated.sort(key=lambda item: item[0], reverse=True)
    return evaluated[0][1], evaluated[0][2]


def _format_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontproperties=CN_FONT, fontsize=Config.AXIS_FONTSIZE, labelpad=Config.LABELPAD)
    ax.set_ylabel(ylabel, fontproperties=CN_FONT, fontsize=Config.AXIS_FONTSIZE, labelpad=Config.LABELPAD)
    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linewidth=Config.GRID_LINEWIDTH,
        linestyle=Config.GRID_LINESTYLE,
    )
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONTSIZE)


def plot_visushrink_distortion(window_info, denoised):
    original_full = np.asarray(window_info["data"], dtype=float)
    trim_start = int(Config.TRIM_START_SECOND * Config.FS)
    trim_end = int(Config.TRIM_END_SECOND * Config.FS)
    original = original_full[trim_start:trim_end]
    denoised_trimmed = denoised[trim_start:trim_end]
    residual = original - denoised_trimmed

    time_axis = np.arange(len(original)) / Config.FS + Config.TRIM_START_SECOND
    freqs, psd_original = compute_psd(original_full)
    _, psd_denoised = compute_psd(denoised)
    freq_mask = freqs <= Config.FREQ_MAX_PLOT

    fig_ts, ax_ts = plt.subplots(figsize=Config.FIG_SIZE)

    ax_ts.plot(time_axis, original, color=Config.COLOR_ORIGINAL, linewidth=Config.LINEWIDTH_SIGNAL, label="Original VIV")
    ax_ts.plot(
        time_axis,
        denoised_trimmed,
        color=Config.COLOR_DENOISED,
        linewidth=Config.LINEWIDTH_SIGNAL,
        label="After VisuShrink",
    )
    _format_axis(ax_ts, "时间 (s)", r"加速度 ($m/s^2$)")
    ax_ts.legend(prop=ENG_FONT, fontsize=Config.LEGEND_FONTSIZE, loc="upper right", framealpha=0.8)
    fig_ts.tight_layout()

    fig_psd, ax_fft = plt.subplots(figsize=Config.FIG_SIZE)

    ax_fft.semilogy(
        freqs[freq_mask],
        psd_original[freq_mask],
        color=Config.COLOR_ORIGINAL,
        linewidth=Config.LINEWIDTH_PSD,
        label="Original VIV",
    )
    ax_fft.semilogy(
        freqs[freq_mask],
        psd_denoised[freq_mask],
        color=Config.COLOR_DENOISED,
        linewidth=Config.LINEWIDTH_PSD,
        label="After VisuShrink",
    )
    ax_fft.set_xlim(0, Config.FREQ_MAX_PLOT)
    _format_axis(ax_fft, "频率 (Hz)", "PSD")
    ax_fft.legend(prop=ENG_FONT, fontsize=Config.LEGEND_FONTSIZE, loc="upper right", framealpha=0.8)
    fig_psd.tight_layout()

    fig_residual, ax_residual = plt.subplots(figsize=Config.FIG_SIZE)

    ax_residual.plot(time_axis, residual, color=Config.COLOR_RESIDUAL, linewidth=Config.LINEWIDTH_RESIDUAL)
    ax_residual.axhline(0, color=Config.COLOR_ZERO_LINE, linewidth=Config.LINEWIDTH_ZERO)
    _format_axis(ax_residual, "时间 (s)", r"残差 ($m/s^2$)")
    fig_residual.tight_layout()

    return (fig_ts, fig_psd, fig_residual)


def main(port=Config.WEBUI_PORT, push_web=True):
    windows = load_viv_windows()
    window_info, denoised = select_most_distorted_window(windows)
    figures = plot_visushrink_distortion(window_info, denoised)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (Config.OUTPUT_DIR / Config.LEGACY_OUTPUT_NAME).unlink(missing_ok=True)
    output_paths = []
    for slot, (fig, (output_name, title)) in enumerate(zip(figures, FIGURE_SPECS)):
        output_path = Config.OUTPUT_DIR / output_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        output_paths.append(output_path)
        print(f"{title} 已保存至: {output_path}")
        if push_web:
            web_push(fig, page=WEBUI_PAGE, slot=slot, title=title, port=port, page_cols=1)
    if push_web:
        print(f"图2.10 已推送到 VibDash: page={WEBUI_PAGE}, port={port}")
    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成图2.10并推送到 VibDash")
    parser.add_argument("--port", type=int, default=Config.WEBUI_PORT, help="VibDash 服务端口")
    parser.add_argument("--no-web", action="store_true", help="只保存图像，不推送到 VibDash")
    args = parser.parse_args()
    main(port=args.port, push_web=not args.no_web)
