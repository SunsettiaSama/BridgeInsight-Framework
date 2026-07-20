"""图4-25b：风雨振大振幅样本长时程（1500 s）。

复现 fig4_25 同 seed 的 20 样本抽样顺序，筛出识别窗内振幅 > 5 m/s² 者，
从源 VIC 以识别窗为中心截取 1500 s 绘制。序号不打乱。
"""

from __future__ import annotations

import argparse
import math
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
from src.figure_paintings.figs_for_thesis.Chapter4._rwiv_pipeline import (
    RWIV_SAMPLE_COPY_PATH,
    USE_MERGED_DATASET,
    add_dataset_switch_args,
    load_rwiv_samples_for_figures,
    resolve_use_merged,
)
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_25_rwiv_timeseries import (
    Config as SharedConfig,
    _format_title,
    _load_window,
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
    WINDOW_SIZE = SharedConfig.WINDOW_SIZE
    RANDOM_SEED = SharedConfig.RANDOM_SEED
    NUM_SAMPLES_SOURCE = SharedConfig.NUM_SAMPLES_TO_PLOT

    AMP_THRESHOLD = 5.0  # m/s²
    LONG_SECONDS = 1500.0
    MAX_DISPLAY = 10  # 超过则按原序号裁到前 10 个，避免子图过密

    FIG_SIZE = SharedConfig.FIG_SIZE  # (18, 11)
    N_COLS = 2
    INPLANE_COLOR = SharedConfig.INPLANE_COLOR
    OUTPLANE_COLOR = SharedConfig.OUTPLANE_COLOR
    LINEWIDTH = SharedConfig.LINEWIDTH
    GRID_COLOR = SharedConfig.GRID_COLOR
    GRID_ALPHA = SharedConfig.GRID_ALPHA
    GRID_LINEWIDTH = SharedConfig.GRID_LINEWIDTH
    GRID_LINESTYLE = SharedConfig.GRID_LINESTYLE

    WINDOW_SHADE_COLOR = "#F0C284"
    WINDOW_SHADE_ALPHA = 0.28

    APPLY_DENOISE = SharedConfig.APPLY_DENOISE
    WEB_PAGE = "fig4_25b 风雨振长时程"


def _load_raw(file_path: str, unpacker: UNPACK) -> np.ndarray:
    return np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float64)


def _long_segment_bounds(n_raw: int, window_idx: int) -> tuple[int, int, int, int]:
    n_long = int(Config.LONG_SECONDS * Config.FS)
    win_start = int(window_idx) * Config.WINDOW_SIZE
    win_end = win_start + Config.WINDOW_SIZE
    if win_end > n_raw:
        raise ValueError(
            f"识别窗越界：window_idx={window_idx}, n_raw={n_raw}, need_end={win_end}"
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
    raw = _load_raw(file_path, unpacker)
    start, end, win_start, win_end = _long_segment_bounds(len(raw), window_idx)
    seg = raw[start:end]
    if Config.APPLY_DENOISE:
        seg = _wavelet_denoise(seg)
    time_axis = np.arange(start, end, dtype=np.float64) / Config.FS
    return seg, time_axis, win_start / Config.FS, win_end / Config.FS


def _window_peak_amp(sample: dict, unpacker: UNPACK) -> float:
    """识别窗 60 s 内，面内/面外面峰值绝对值。"""
    in_win = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
    out_win = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
    if Config.APPLY_DENOISE:
        in_win = _wavelet_denoise(in_win)
        out_win = _wavelet_denoise(out_win)
    return float(max(np.max(np.abs(in_win)), np.max(np.abs(out_win))))


def filter_high_amp_keep_order(
    samples_fig425: list,
    unpacker: UNPACK,
) -> list[tuple[int, dict, float]]:
    """按 fig4_25 抽样顺序筛振幅 > 阈值，保留原序号。

    返回 [(原序号1-based, sample, peak_amp), ...]
    """
    kept: list[tuple[int, dict, float]] = []
    print(
        f"\n[筛选] 在 fig4_25 的 {len(samples_fig425)} 个样本中，"
        f"识别窗峰值振幅 > {Config.AMP_THRESHOLD:g} m/s²："
    )
    for i, sample in enumerate(samples_fig425):
        peak = _window_peak_amp(sample, unpacker)
        keep = peak > Config.AMP_THRESHOLD
        tag = "KEEP" if keep else "skip"
        print(
            f"  [{tag}] 序号={i + 1:02d}  peak={peak:7.3f} m/s²  "
            f"win={sample['window_idx']}  "
            f"{sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}  "
            f"idx={sample.get('idx')}  split={sample.get('split', '')}  "
            f"time={sample.get('time_str') or sample.get('timestamp')}"
        )
        if keep:
            kept.append((i + 1, sample, peak))

    print(f"\n[筛选结果] 振幅条件命中 {len(kept)} 个（顺序与 fig4_25 一致）：")
    for ord_idx, sample, peak in kept:
        print(
            f"  -> 原序号={ord_idx:02d}  peak={peak:.3f} m/s²  "
            f"win={sample['window_idx']}  "
            f"{sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}"
        )

    if len(kept) > Config.MAX_DISPLAY:
        trimmed = kept[: Config.MAX_DISPLAY]
        dropped = kept[Config.MAX_DISPLAY :]
        print(
            f"\n[裁剪显示] 命中 {len(kept)} > {Config.MAX_DISPLAY}，"
            f"按原序号保留前 {Config.MAX_DISPLAY} 个用于绘图："
        )
        for ord_idx, sample, peak in trimmed:
            print(
                f"  [PLOT] 原序号={ord_idx:02d}  peak={peak:.3f} m/s²  "
                f"{sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}"
            )
        print(f"  未绘制（超出上限）：{[ord_idx for ord_idx, _, _ in dropped]}")
        return trimmed

    return kept


def plot_long_timeseries_grid(
    kept: list[tuple[int, dict, float]],
    unpacker: UNPACK,
) -> plt.Figure:
    n = len(kept)
    if n == 0:
        raise ValueError(
            f"fig4_25 抽样中无识别窗振幅 > {Config.AMP_THRESHOLD:g} m/s² 的样本"
        )

    n_cols = Config.N_COLS
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=Config.FIG_SIZE,
        sharex=False,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for plot_i, (ord_idx, sample, peak) in enumerate(kept):
        ax = axes_flat[plot_i]
        print(
            f"  [绘图 {plot_i + 1}/{n}] 原序号={ord_idx}  peak={peak:.3f}  "
            f"sensor={sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}"
        )

        in_seg, t_axis, win_t0, win_t1 = load_long_segment(
            sample["inplane_file_path"], sample["window_idx"], unpacker
        )
        out_seg, t_out, _, _ = load_long_segment(
            sample["outplane_file_path"], sample["window_idx"], unpacker
        )
        if len(t_out) != len(t_axis):
            m = min(len(t_axis), len(t_out), len(in_seg), len(out_seg))
            t_axis = t_axis[:m]
            in_seg = in_seg[:m]
            out_seg = out_seg[:m]

        ax.axvspan(
            win_t0,
            win_t1,
            color=Config.WINDOW_SHADE_COLOR,
            alpha=Config.WINDOW_SHADE_ALPHA,
            zorder=0,
            label="识别窗 60 s" if plot_i == 0 else None,
        )
        ax.plot(
            t_axis,
            in_seg,
            color=Config.INPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            label="面内" if plot_i == 0 else None,
            zorder=2,
        )
        ax.plot(
            t_axis,
            out_seg,
            color=Config.OUTPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            alpha=0.85,
            label="面外" if plot_i == 0 else None,
            zorder=2,
        )

        # 标题保留 fig4_25 原序号
        base_title = _format_title(sample, ord_idx - 1)
        # _format_title 以 sample_idx+1 开头，这里已是原序号；再标注峰值
        title = f"{base_title}  |A|max={peak:.2f}"
        ax.set_title(title, fontproperties=ENG_FONT, fontsize=FONT_SIZE - 12, pad=3)
        ax.grid(
            True,
            color=Config.GRID_COLOR,
            alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH,
            linestyle=Config.GRID_LINESTYLE,
        )
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 12)
        ax.set_xlim(float(t_axis[0]), float(t_axis[-1]))

    for ax in axes_flat[n:]:
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
        f"源自 fig4_25（seed={Config.RANDOM_SEED}）；"
        f"识别窗 |A|>{Config.AMP_THRESHOLD:g} m/s²；"
        f"源文件截取 {Config.LONG_SECONDS:g} s；"
        f"显示上限 {Config.MAX_DISPLAY}；n={n}",
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 12,
        color="#404040",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.91, hspace=0.42, wspace=0.22)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-25b 风雨振大振幅长时程")
    add_dataset_switch_args(parser)
    args = parser.parse_args()
    use_merged = resolve_use_merged(args.use_merged)

    print("=" * 80)
    print(
        f"图4-25b 风雨振长时程"
        f"（fig4_25 同批中 |A|>{Config.AMP_THRESHOLD:g} m/s²，源文件 {Config.LONG_SECONDS:g} s）"
    )
    print(f"  默认开关 USE_MERGED_DATASET={USE_MERGED_DATASET}  → 本次 use_merged={use_merged}")
    print("=" * 80)

    print("\n[步骤1] 加载风雨振样本（与 fig4_25 一致）...")
    if use_merged:
        print(f"  副本路径：{RWIV_SAMPLE_COPY_PATH}")
    all_samples = load_rwiv_samples_for_figures(
        use_merged=use_merged,
        force_refresh=args.refresh_sample_copy,
    )
    print(f"[OK] 风雨振配对样本：{len(all_samples)} 个")

    print("\n[步骤2] 复现 fig4_25 抽样（保持顺序）...")
    prev_n = SharedConfig.NUM_SAMPLES_TO_PLOT
    SharedConfig.NUM_SAMPLES_TO_PLOT = Config.NUM_SAMPLES_SOURCE
    samples_fig425 = random_sample(all_samples)
    SharedConfig.NUM_SAMPLES_TO_PLOT = prev_n

    print("\n[步骤3] 按振幅阈值筛选...")
    unpacker = UNPACK(init_path=False)
    kept = filter_high_amp_keep_order(samples_fig425, unpacker)

    print(f"\n[步骤4] 绘制长时程总图（{len(kept)} 个样本）...")
    figure = plot_long_timeseries_grid(kept, unpacker)
    web_push(
        figure,
        page=Config.WEB_PAGE,
        slot=0,
        title=f"风雨振长时程（|A|>{Config.AMP_THRESHOLD:g}，{Config.LONG_SECONDS:g} s）",
        page_cols=1,
    )
    plt.close(figure)

    print("\n" + "=" * 80)
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
