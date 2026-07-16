import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.web_dashboard import push as web_push
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_8_normal_vib_timeseries import (
    Config as SharedSampleConfig,
    get_normal_vib_samples,
    load_filtered_dl_result,
    random_sample,
)
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, ENG_FONT, FONT_SIZE,
)


# ==================== 常量配置 ====================
class Config:
    FS = 50.0
    WINDOW_SIZE = 3000

    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'

    NORMAL_VIB_CLASS_ID = 0

    NUM_SAMPLES_TO_PLOT = SharedSampleConfig.NUM_SAMPLES_TO_PLOT
    RANDOM_SEED = SharedSampleConfig.RANDOM_SEED

    FIG_SIZE = SharedSampleConfig.FIG_SIZE
    N_ROWS = SharedSampleConfig.N_ROWS
    N_COLS = SharedSampleConfig.N_COLS
    SCATTER_COLOR = "#202020"
    SCATTER_SIZE = 3
    SCATTER_ALPHA = 0.28
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.25
    GRID_LINESTYLE = '--'


# ==================== 数据加载 ====================
def _load_window(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw = unpacker.VIC_DATA_Unpack(file_path)
    raw = np.array(raw)
    start = window_idx * Config.WINDOW_SIZE
    end   = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, data_len={len(raw)}")
    return raw[start:end]


def _wavelet_denoise(data: np.ndarray) -> np.ndarray:
    denoised, _ = denoise(
        signal=data,
        wavelet=Config.WAVELET_TYPE,
        level=Config.WAVELET_LEVEL,
        threshold_type=Config.THRESHOLD_TYPE,
        threshold_method=Config.THRESHOLD_METHOD,
    )
    return denoised

def load_sample_pair(sample: dict, unpacker: UNPACK):
    in_raw  = _load_window(sample["inplane_file_path"],  sample["window_idx"], unpacker)
    out_raw = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
    return _wavelet_denoise(in_raw), _wavelet_denoise(out_raw)


# ==================== 绘图函数 ====================
def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _plot_single_trajectory(ax, sample: dict, sample_idx: int, total: int, unpacker: UNPACK) -> None:
    in_data, out_data = load_sample_pair(sample, unpacker)

    ts = sample.get("timestamp", [])
    in_id  = sample.get("inplane_sensor_id",  "未知")
    if ts and len(ts) >= 3:
        time_line = f"{int(ts[0]):02d}-{int(ts[1]):02d} {int(ts[2]):02d}h"
    elif ts and len(ts) >= 2:
        time_line = f"{int(ts[0]):02d}-{int(ts[1]):02d}"
    else:
        time_line = f"样本{sample_idx + 1}"
    cable_id = in_id.replace("ST-VIC-", "").rsplit("-", 1)[0]
    title = f"{sample_idx + 1}. {cable_id}  {time_line}  win={sample['window_idx']}"

    ax.scatter(
        out_data, in_data,
        s=Config.SCATTER_SIZE,
        color=Config.SCATTER_COLOR,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
    )

    all_vals = np.concatenate([in_data, out_data])
    global_min, global_max = float(all_vals.min()), float(all_vals.max())
    margin = (global_max - global_min) * 0.05
    ax.set_xlim(global_min - margin, global_max + margin)
    ax.set_ylim(global_min - margin, global_max + margin)

    ax.set_title(title, fontproperties=ENG_FONT, fontsize=FONT_SIZE - 12, pad=3)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 12)

    _apply_grid(ax)
    print(f"  ✓ 样本 {sample_idx + 1}/{total} 已绘制  sensor={sample['inplane_sensor_id']}")


def plot_trajectory_cloud_grid(samples: list, unpacker: UNPACK) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.N_ROWS,
        Config.N_COLS,
        figsize=Config.FIG_SIZE,
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.ravel()
    for i, sample in enumerate(samples):
        _plot_single_trajectory(axes_flat[i], sample, i, len(samples), unpacker)

    for ax in axes_flat[len(samples):]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"面外加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"面内加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)

    fig.text(
        0.99,
        0.01,
        f"已剔除 C34-201/202/301；窗口 {Config.WINDOW_SIZE / Config.FS:.0f} s；seed={Config.RANDOM_SEED}",
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 12,
        color="#404040",
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.91, hspace=0.42, wspace=0.28)
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 随机振动轨迹云图（面内 vs 面外散点）")
    print("=" * 80)

    print("\n[步骤1] 加载识别结果（已剔除版本，与 fig4_8 一致）...")
    result = load_filtered_dl_result()

    print("\n[步骤2] 筛选随机振动（class 0）样本...")
    all_samples = get_normal_vib_samples(result)
    print(f"✓ 共筛选到 {len(all_samples)} 个随机振动样本")

    print("\n[步骤3] 随机抽取样本...")
    samples = random_sample(all_samples)

    print(f"\n[步骤4] 加载数据并绘制轨迹云图（{len(samples)} 个样本）...")
    unpacker = UNPACK(init_path=False)
    figure = plot_trajectory_cloud_grid(samples, unpacker)

    print("\n" + "=" * 80)
    print(f"✓ 轨迹云图绘制完成，总图 1 张，子图 {len(samples)} 个")
    print("=" * 80)

    page = "fig4_9 随机振动轨迹"
    web_push(
        figure,
        page=page,
        slot=0,
        title="随机振动轨迹 20 样本总览",
        page_cols=1,
    )
    plt.close(figure)
    print(f"✓ 已推送到 WebUI：{page}")


if __name__ == "__main__":
    main()
