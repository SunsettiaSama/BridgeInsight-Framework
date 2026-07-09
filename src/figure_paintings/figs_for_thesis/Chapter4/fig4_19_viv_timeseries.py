import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import (
    get_viv_samples as _pipeline_get_viv_samples,
    load_dl_result,
)


# ==================== 常量配置 ====================
class Config:
    FS = 50.0
    WINDOW_SIZE = 3000          # 60s @ 50Hz

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20        # 展示前20s，涡激共振周期性更明显

    NUM_SAMPLES_TO_PLOT = 20
    RANDOM_SEED = 7

    FIG_SIZE = (18, 11)
    N_ROWS = 4
    N_COLS = 5
    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    LINEWIDTH = 0.7
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.25
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    APPLY_DENOISE = False       # True: 先去噪再绘图；False: 直接用原始信号
    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'


# ==================== 数据获取 ====================
def random_sample(samples: list) -> list:
    n = len(samples)
    k = min(Config.NUM_SAMPLES_TO_PLOT, n)
    rng = np.random.default_rng(Config.RANDOM_SEED)
    chosen_indices = rng.choice(n, size=k, replace=False)
    chosen_indices_sorted = sorted(chosen_indices.tolist())
    print(f"  随机抽取索引：{chosen_indices_sorted}（共 {n} 个样本中选 {k} 个，seed={Config.RANDOM_SEED}）")
    return [samples[i] for i in chosen_indices_sorted]


# ==================== 数据预处理 ====================
def _wavelet_denoise(data: np.ndarray) -> np.ndarray:
    denoised, _ = denoise(
        signal=data,
        wavelet=Config.WAVELET_TYPE,
        level=Config.WAVELET_LEVEL,
        threshold_type=Config.THRESHOLD_TYPE,
        threshold_method=Config.THRESHOLD_METHOD,
    )
    return denoised


def _load_window(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw = unpacker.VIC_DATA_Unpack(file_path)
    raw = np.array(raw)
    start = window_idx * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, data_len={len(raw)}")
    return raw[start:end]


def _prepare_plot_window(data: np.ndarray) -> np.ndarray:
    data_src = _wavelet_denoise(data) if Config.APPLY_DENOISE else data
    trim_start = int(Config.TRIM_START_SECOND * Config.FS)
    trim_end = int(Config.TRIM_END_SECOND * Config.FS)
    return data_src[trim_start:trim_end]


def _format_title(sample: dict, sample_idx: int) -> str:
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(t) for t in timestamp) if timestamp else ""
    sensor_id = sample.get("inplane_sensor_id", "")
    cable_id = sensor_id.replace("ST-VIC-", "").rsplit("-", 1)[0]
    return f"{sample_idx + 1}. {cable_id}  {time_str}  win={sample['window_idx']}"


# ==================== 绘图函数 ====================
def plot_viv_timeseries_grid(samples: list, unpacker: UNPACK) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.N_ROWS,
        Config.N_COLS,
        figsize=Config.FIG_SIZE,
        sharex=True,
    )
    axes_flat = axes.ravel()

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        print(f"  [样本 {i + 1}] sensor={sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}")

        inplane_data = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
        outplane_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
        inplane_plot = _prepare_plot_window(inplane_data)
        outplane_plot = _prepare_plot_window(outplane_data)
        time_axis = np.arange(len(inplane_plot)) / Config.FS + Config.TRIM_START_SECOND

        ax.plot(
            time_axis,
            inplane_plot,
            color=Config.INPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            label="面内" if i == 0 else None,
        )
        ax.plot(
            time_axis,
            outplane_plot,
            color=Config.OUTPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            alpha=0.85,
            label="面外" if i == 0 else None,
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

    for ax in axes_flat[len(samples):]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)

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
        f"窗口 {Config.WINDOW_SIZE / Config.FS:.0f} s；展示 {Config.TRIM_START_SECOND:g}-{Config.TRIM_END_SECOND:g} s；seed={Config.RANDOM_SEED}",
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 12,
        color="#404040",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.91, hspace=0.42, wspace=0.22)
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振时域波形 20 样本总览（DL，面内 & 面外）")
    print("=" * 80)

    print("\n[步骤1] 加载 DL 识别结果...")
    dl_result = load_dl_result()
    all_samples = _pipeline_get_viv_samples(dl_result)
    print(f"✓ DL VIV 样本：{len(all_samples)} 个")

    print("\n[步骤2] 随机抽取样本...")
    samples = random_sample(all_samples)

    print(f"\n[步骤3] 加载数据并绘制时程总图（{len(samples)} 个样本）...")
    unpacker = UNPACK(init_path=False)
    figure = plot_viv_timeseries_grid(samples, unpacker)

    print("\n" + "=" * 80)
    print(f"✓ 时程图绘制完成，总图 1 张，子图 {len(samples)} 个")
    print("=" * 80)

    page = "fig4_19 VIV时程"
    web_push(
        figure,
        page=page,
        slot=0,
        title="DL-VIV 时域波形 20 样本总览",
        page_cols=1,
    )
    plt.close(figure)
    print(f"✓ 已推送到 WebUI：{page}")


if __name__ == "__main__":
    main()
