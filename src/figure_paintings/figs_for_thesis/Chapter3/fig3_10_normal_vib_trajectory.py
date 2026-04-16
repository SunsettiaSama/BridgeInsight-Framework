import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.Chapter3.fig3_2_all_data_display import load_identification_result
from src.figure_paintings.figs_for_thesis.Chapter3.fig3_3_normal_vib_timeseries import get_normal_vib_samples
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_blue_color_map, get_full_color_map,
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

    NUM_SAMPLES_TO_PLOT = 2
    RANDOM_SEED = 50

    FIG_SIZE = SQUARE_FIG_SIZE
    SCATTER_SIZE = 10
    SCATTER_ALPHA = 0.35
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'

    SAMPLE_COLORS: list = []   # 延迟填充，见类定义后


_full_palette = get_full_color_map(style='discrete').colors
Config.SAMPLE_COLORS = [
    _full_palette[i]
    for i in np.linspace(0, len(_full_palette) - 1, Config.NUM_SAMPLES_TO_PLOT, dtype=int).tolist()
]


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


def random_sample(samples: list) -> list:
    n = len(samples)
    k = min(Config.NUM_SAMPLES_TO_PLOT, n)
    rng = np.random.default_rng(Config.RANDOM_SEED)
    chosen = sorted(rng.choice(n, size=k, replace=False).tolist())
    print(f"  随机抽取索引：{chosen}（共 {n} 个样本中选 {k} 个，seed={Config.RANDOM_SEED}）")
    return [samples[i] for i in chosen]


def load_sample_pair(sample: dict, unpacker: UNPACK):
    in_raw  = _load_window(sample["inplane_file_path"],  sample["window_idx"], unpacker)
    out_raw = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
    return _wavelet_denoise(in_raw), _wavelet_denoise(out_raw)


# ==================== 绘图函数 ====================
def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _plot_single_trajectory(sample: dict, color, sample_idx: int, total: int, unpacker: UNPACK) -> plt.Figure:
    in_data, out_data = load_sample_pair(sample, unpacker)

    ts = sample.get("timestamp", [])
    in_id  = sample.get("inplane_sensor_id",  "未知")
    out_id = sample.get("outplane_sensor_id", "未知")
    sensor_line = f"面内: {in_id}  |  面外: {out_id}"
    if ts and len(ts) >= 3:
        time_line = f"{int(ts[0]):02d}月{int(ts[1]):02d}日  {int(ts[2]):02d}时"
    elif ts and len(ts) >= 2:
        time_line = f"{int(ts[0]):02d}月{int(ts[1]):02d}日"
    else:
        time_line = f"样本{sample_idx + 1}"
    title = f"{sensor_line}\n{time_line}"

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.scatter(
        out_data, in_data,
        s=Config.SCATTER_SIZE,
        color=color,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
    )

    all_vals = np.concatenate([in_data, out_data])
    global_min, global_max = float(all_vals.min()), float(all_vals.max())
    margin = (global_max - global_min) * 0.05
    ax.set_xlim(global_min - margin, global_max + margin)
    ax.set_ylim(global_min - margin, global_max + margin)

    ax.set_xlabel(r'面外加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'面内加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)

    _apply_grid(ax)
    plt.tight_layout()
    print(f"  ✓ 样本 {sample_idx + 1}/{total} 已绘制  sensor={sample['inplane_sensor_id']}")
    return fig


def plot_trajectory_cloud(samples: list, unpacker: UNPACK) -> list:
    figs = []
    for i, sample in enumerate(samples):
        color = Config.SAMPLE_COLORS[i % len(Config.SAMPLE_COLORS)]
        fig = _plot_single_trajectory(sample, color, i, len(samples), unpacker)
        figs.append(fig)
    return figs


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 随机振动轨迹云图（面内 vs 面外散点）")
    print("=" * 80)

    result_dir = project_root / "results" / "identification_result"
    if not result_dir.exists():
        raise FileNotFoundError(f"识别结果目录不存在：{result_dir}")

    result_files = sorted(result_dir.glob("res_cnn_full_dataset_*.json"))
    if not result_files:
        raise FileNotFoundError("未找到识别结果文件 res_cnn_full_dataset_*.json")

    result_path = result_files[-1]
    print(f"\n[步骤1] 加载识别结果：{result_path.name}")
    result = load_identification_result(str(result_path))

    print("\n[步骤2] 筛选随机振动（class 0）样本...")
    all_samples = get_normal_vib_samples(result)
    print(f"✓ 共筛选到 {len(all_samples)} 个随机振动样本")

    print("\n[步骤3] 随机抽取样本...")
    samples = random_sample(all_samples)

    print(f"\n[步骤4] 加载数据并绘制轨迹云图（{len(samples)} 个样本）...")
    unpacker = UNPACK(init_path=False)
    figs = plot_trajectory_cloud(samples, unpacker)

    print("\n" + "=" * 80)
    print(f"✓ 轨迹云图绘制完成，共 {len(figs)} 张")
    print("=" * 80)

    ploter = PlotLib()
    ploter.figs.extend(figs)
    ploter.show()


if __name__ == "__main__":
    main()
