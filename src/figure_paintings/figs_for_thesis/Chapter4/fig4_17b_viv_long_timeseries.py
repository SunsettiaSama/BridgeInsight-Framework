"""图4-17（续）：涡激共振长时程波形（1500 s）。

取 fig4_x_viv_timeseries 同 seed 抽样结果的前 8 个样本，从源 VIC 文件（约 3600 s）
以识别窗口为中心截取 1500 s 连续段。
图幅与 fig4_x_viv_timeseries 一致 (18, 11)，布局 4 行 × 2 列（每行 2 张）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import (
    get_viv_samples as _pipeline_get_viv_samples,
    load_dl_result,
)
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_x_viv_timeseries import (
    Config as SharedConfig,
    _format_title,
    _wavelet_denoise,
    random_sample,
)
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    FONT_SIZE,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    FS = SharedConfig.FS
    WINDOW_SIZE = SharedConfig.WINDOW_SIZE  # 识别窗 60 s
    RANDOM_SEED = SharedConfig.RANDOM_SEED
    NUM_SAMPLES_SOURCE = SharedConfig.NUM_SAMPLES_TO_PLOT
    NUM_SAMPLES_TO_PLOT = 10

    # 从源文件截取的长时程长度（秒）
    LONG_SECONDS = 1500.0

    FIG_SIZE = SharedConfig.FIG_SIZE  # (18, 11)
    N_ROWS = 5
    N_COLS = 2
    INPLANE_COLOR = SharedConfig.INPLANE_COLOR
    OUTPLANE_COLOR = SharedConfig.OUTPLANE_COLOR
    LINEWIDTH = SharedConfig.LINEWIDTH
    GRID_COLOR = SharedConfig.GRID_COLOR
    GRID_ALPHA = SharedConfig.GRID_ALPHA
    GRID_LINEWIDTH = SharedConfig.GRID_LINEWIDTH
    GRID_LINESTYLE = SharedConfig.GRID_LINESTYLE

    # 识别窗高亮
    WINDOW_SHADE_COLOR = "#F0C284"
    WINDOW_SHADE_ALPHA = 0.28

    APPLY_DENOISE = SharedConfig.APPLY_DENOISE
    WEB_PAGE = "fig4_17 VIV长时程"


def _load_raw(file_path: str, unpacker: UNPACK) -> np.ndarray:
    return np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float64)


def _long_segment_bounds(n_raw: int, window_idx: int) -> tuple[int, int, int, int]:
    """以识别窗中心为锚，截取 LONG_SECONDS；贴边时整体平移，尽量保长。"""
    n_long = int(Config.LONG_SECONDS * Config.FS)
    win_start = int(window_idx) * Config.WINDOW_SIZE
    win_end = win_start + Config.WINDOW_SIZE
    if win_end > n_raw:
        raise ValueError(
            f"识别窗越界：window_idx={window_idx}, n_raw={n_raw}, "
            f"need_end={win_end}"
        )

    center = (win_start + win_end) // 2
    start = center - n_long // 2
    end = start + n_long
    if start < 0:
        start = 0
        end = min(n_long, n_raw)
    if end > n_raw:
        end = n_raw
        start = max(0, end - n_long)
    return start, end, win_start, win_end


def load_long_segment(
    file_path: str,
    window_idx: int,
    unpacker: UNPACK,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """返回 (信号, 时间轴[s,相对源文件起点], 识别窗起/止时刻[s])。"""
    raw = _load_raw(file_path, unpacker)
    start, end, win_start, win_end = _long_segment_bounds(len(raw), window_idx)
    seg = raw[start:end]
    if Config.APPLY_DENOISE:
        seg = _wavelet_denoise(seg)
    time_axis = np.arange(start, end, dtype=np.float64) / Config.FS
    return seg, time_axis, win_start / Config.FS, win_end / Config.FS


def take_first_n_from_fig416(all_samples: list) -> list:
    prev_n = SharedConfig.NUM_SAMPLES_TO_PLOT
    SharedConfig.NUM_SAMPLES_TO_PLOT = Config.NUM_SAMPLES_SOURCE
    sampled20 = random_sample(all_samples)
    SharedConfig.NUM_SAMPLES_TO_PLOT = prev_n

    selected = sampled20[: Config.NUM_SAMPLES_TO_PLOT]
    print(
        f"  取 fig4_x_viv_timeseries 同批前 {len(selected)} 个样本（seed={Config.RANDOM_SEED}）"
    )
    return selected


def plot_long_timeseries_grid(samples: list, unpacker: UNPACK) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.N_ROWS,
        Config.N_COLS,
        figsize=Config.FIG_SIZE,
        sharex=False,
    )
    axes_flat = axes.ravel()

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        print(
            f"  [样本 {i + 1}] sensor="
            f"{sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}  "
            f"win={sample['window_idx']}"
        )

        in_seg, t_axis, win_t0, win_t1 = load_long_segment(
            sample["inplane_file_path"], sample["window_idx"], unpacker
        )
        out_seg, t_out, _, _ = load_long_segment(
            sample["outplane_file_path"], sample["window_idx"], unpacker
        )
        if len(t_out) != len(t_axis):
            n = min(len(t_axis), len(t_out), len(in_seg), len(out_seg))
            t_axis = t_axis[:n]
            in_seg = in_seg[:n]
            out_seg = out_seg[:n]

        ax.axvspan(
            win_t0,
            win_t1,
            color=Config.WINDOW_SHADE_COLOR,
            alpha=Config.WINDOW_SHADE_ALPHA,
            zorder=0,
            label="识别窗 60 s" if i == 0 else None,
        )
        ax.plot(
            t_axis,
            in_seg,
            color=Config.INPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            label="面内" if i == 0 else None,
            zorder=2,
        )
        ax.plot(
            t_axis,
            out_seg,
            color=Config.OUTPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            alpha=0.85,
            label="面外" if i == 0 else None,
            zorder=2,
        )

        ax.set_title(
            _format_title(sample, i),
            fontproperties=ENG_FONT,
            fontsize=FONT_SIZE - 12,
            pad=3,
        )
        ax.grid(
            True,
            color=Config.GRID_COLOR,
            alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH,
            linestyle=Config.GRID_LINESTYLE,
        )
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 12)
        ax.set_xlim(float(t_axis[0]), float(t_axis[-1]))
        print(
            f"    源文件截取 {t_axis[0]:.1f}-{t_axis[-1]:.1f} s "
            f"（约 {t_axis[-1] - t_axis[0]:.0f} s），识别窗 {win_t0:.1f}-{win_t1:.1f} s"
        )

    for ax in axes_flat[len(samples) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("时间 (s，相对源文件起点)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        prop=CN_FONT,
        fontsize=FONT_SIZE - 8,
    )
    fig.text(
        0.99,
        0.01,
        f"取自 fig4_x_viv_timeseries 前 {Config.NUM_SAMPLES_TO_PLOT} 样本；"
        f"源文件截取 {Config.LONG_SECONDS:g} s（锚定识别窗）；"
        f"seed={Config.RANDOM_SEED}",
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 12,
        color="#404040",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.91, hspace=0.42, wspace=0.22)
    return fig


def main() -> None:
    print("=" * 80)
    print(
        f"图4-17 涡激共振长时程波形"
        f"（fig4_x_viv_timeseries 前 {Config.NUM_SAMPLES_TO_PLOT} 样本，源文件 {Config.LONG_SECONDS:g} s）"
    )
    print("=" * 80)

    print("\n[步骤1] 加载 DL 识别结果...")
    dl_result = load_dl_result()
    all_samples = _pipeline_get_viv_samples(dl_result)
    print(f"[OK] DL VIV 样本：{len(all_samples)} 个")

    print("\n[步骤2] 复现 fig4_x_viv_timeseries 抽样并取前 8 个...")
    samples = take_first_n_from_fig416(all_samples)

    print(
        f"\n[步骤3] 从源 VIC 截取 {Config.LONG_SECONDS:g} s 并绘图"
        f"（{len(samples)} 个样本，{Config.N_ROWS}×{Config.N_COLS}）..."
    )
    unpacker = UNPACK(init_path=False)
    figure = plot_long_timeseries_grid(samples, unpacker)

    web_push(
        figure,
        page=Config.WEB_PAGE,
        slot=0,
        title=f"DL-VIV 长时程波形（{Config.LONG_SECONDS:g} s）",
        page_cols=1,
    )
    plt.close(figure)

    print("\n" + "=" * 80)
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
