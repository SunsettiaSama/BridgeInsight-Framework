"""图4-31：风雨振频谱图 20 样本总览。

样式对齐 VIV 探索脚本频谱图。样本池与 fig4_29 共用（同 seed / 同数据源开关）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4._rwiv_pipeline import (
    RWIV_SAMPLE_COPY_PATH,
    USE_MERGED_DATASET,
    add_dataset_switch_args,
    load_rwiv_samples_for_figures,
    resolve_use_merged,
)
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_29_rwiv_timeseries import (
    Config as SharedConfig,
    _format_title,
    _load_window,
    random_sample,
)
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    FS = SharedConfig.FS
    WINDOW_SIZE = SharedConfig.WINDOW_SIZE
    NUM_SAMPLES_TO_PLOT = SharedConfig.NUM_SAMPLES_TO_PLOT
    RANDOM_SEED = SharedConfig.RANDOM_SEED

    FIG_SIZE = SharedConfig.FIG_SIZE
    N_ROWS = SharedConfig.N_ROWS
    N_COLS = SharedConfig.N_COLS

    NFFT = 2048
    FREQ_LIMIT = 25.0

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    LINEWIDTH = 0.75
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.25
    GRID_LINESTYLE = "--"
    GRID_LINEWIDTH = 0.5

    WEB_PAGE = "fig4_31 风雨振频谱"


def load_sample_pair(sample: dict, unpacker: UNPACK) -> tuple[np.ndarray, np.ndarray]:
    in_data = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
    out_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
    return in_data, out_data


def _apply_grid(ax) -> None:
    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linewidth=Config.GRID_LINEWIDTH,
        linestyle=Config.GRID_LINESTYLE,
    )
    ax.set_axisbelow(True)


def _welch(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, psd = scipy_signal.welch(
        data,
        fs=Config.FS,
        nperseg=Config.NFFT // 2,
        noverlap=Config.NFFT // 4,
        nfft=Config.NFFT,
        scaling="density",
    )
    mask = f <= Config.FREQ_LIMIT
    return f[mask], psd[mask]


def plot_spectrum_grid(samples: list, unpacker: UNPACK) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.N_ROWS,
        Config.N_COLS,
        figsize=Config.FIG_SIZE,
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.ravel()

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        in_data, out_data = load_sample_pair(sample, unpacker)
        print(
            f"  [频谱 {i + 1}] sensor="
            f"{sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}"
        )

        f_in, p_in = _welch(in_data)
        f_out, p_out = _welch(out_data)
        ax.plot(
            f_in,
            p_in,
            color=Config.INPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            label="面内" if i == 0 else None,
        )
        ax.plot(
            f_out,
            p_out,
            color=Config.OUTPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            alpha=0.85,
            label="面外" if i == 0 else None,
        )
        ax.set_xlim(0, Config.FREQ_LIMIT)
        ax.set_title(
            _format_title(sample, i),
            fontproperties=ENG_FONT,
            fontsize=FONT_SIZE - 12,
            pad=3,
        )
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 12)
        _apply_grid(ax)

    for ax in axes_flat[len(samples) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD", fontproperties=ENG_FONT, fontsize=FONT_SIZE - 8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        prop=CN_FONT,
        fontsize=FONT_SIZE - 8,
    )
    fig.text(
        0.99,
        0.01,
        f"Welch PSD；nfft={Config.NFFT}；f≤{Config.FREQ_LIMIT:g} Hz；seed={Config.RANDOM_SEED}",
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 12,
        color="#404040",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.91, hspace=0.42, wspace=0.25)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-31 风雨振频谱图")
    add_dataset_switch_args(parser)
    args = parser.parse_args()
    use_merged = resolve_use_merged(args.use_merged)

    print("=" * 80)
    print("风雨振频谱图 20 样本总览")
    print(f"  默认开关 USE_MERGED_DATASET={USE_MERGED_DATASET}  → 本次 use_merged={use_merged}")
    print("=" * 80)

    print("\n[步骤1] 加载风雨振样本...")
    if use_merged:
        print(f"  副本路径：{RWIV_SAMPLE_COPY_PATH}")
    all_samples = load_rwiv_samples_for_figures(
        use_merged=use_merged,
        force_refresh=args.refresh_sample_copy,
    )
    print(f"✓ 风雨振配对样本：{len(all_samples)} 个")

    print("\n[步骤2] 随机抽取样本（与 fig4_29 同 seed）...")
    samples = random_sample(all_samples)

    print(f"\n[步骤3] 绘制频谱图（{len(samples)} 个样本）...")
    unpacker = UNPACK(init_path=False)
    figure = plot_spectrum_grid(samples, unpacker)
    web_push(
        figure,
        page=Config.WEB_PAGE,
        slot=0,
        title="风雨振 频谱图 20 样本总览",
        page_cols=1,
    )
    plt.close(figure)

    print("\n" + "=" * 80)
    print(f"✓ 已推送到 WebUI：{Config.WEB_PAGE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
