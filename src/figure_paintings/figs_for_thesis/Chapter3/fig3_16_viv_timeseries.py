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
from src.figure_paintings.figs_for_thesis.config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, VIV_VIB_COLOR,
)


# ==================== 常量配置 ====================
class Config:
    FS = 50.0
    WINDOW_SIZE = 3000          # 60s @ 50Hz

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20        # 展示前20s，涡激共振周期性更明显

    NUM_SAMPLES_TO_PLOT = 2
    RANDOM_SEED = 7

    FIG_SIZE = SQUARE_FIG_SIZE
    WAVEFORM_COLOR = VIV_VIB_COLOR
    LINEWIDTH = 1.0
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'

    VIV_CLASS_ID = 1            # 涡激共振对应的类别编号


# ==================== 数据获取 ====================
def get_viv_samples(result: dict) -> list:
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    viv_samples = []
    for idx, pred_label in predictions.items():
        if pred_label != Config.VIV_CLASS_ID:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        inplane_path = meta.get("inplane_file_path")
        outplane_path = meta.get("outplane_file_path")
        if not inplane_path or not outplane_path:
            continue
        viv_samples.append({
            "idx": idx,
            "window_idx": meta["window_idx"],
            "inplane_sensor_id": meta.get("inplane_sensor_id", ""),
            "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
            "inplane_file_path": inplane_path,
            "outplane_file_path": outplane_path,
            "timestamp": meta.get("timestamp", []),
        })

    return viv_samples


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


# ==================== 绘图函数 ====================
def _plot_single(data: np.ndarray, sensor_id: str, timestamp: list) -> plt.Figure:
    denoised = _wavelet_denoise(data)

    trim_start = int(Config.TRIM_START_SECOND * Config.FS)
    trim_end = int(Config.TRIM_END_SECOND * Config.FS)
    data_plot = denoised[trim_start:trim_end]

    time_axis = np.arange(len(data_plot)) / Config.FS + Config.TRIM_START_SECOND

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.plot(time_axis, data_plot, color=Config.WAVEFORM_COLOR, linewidth=Config.LINEWIDTH)

    time_str = "-".join(str(t) for t in timestamp) if timestamp else ""
    ax.set_title(f"{sensor_id} @ {time_str}", fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)

    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linewidth=Config.GRID_LINEWIDTH,
        linestyle=Config.GRID_LINESTYLE,
    )
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    return fig


def plot_viv_timeseries(samples: list, unpacker: UNPACK):
    inplane_figs = []
    outplane_figs = []

    for i, sample in enumerate(samples, 1):
        if len(inplane_figs) >= Config.NUM_SAMPLES_TO_PLOT and len(outplane_figs) >= Config.NUM_SAMPLES_TO_PLOT:
            break

        print(f"  [样本 {i}] sensor={sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}")

        if len(inplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            inplane_data = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
            fig_in = _plot_single(inplane_data, sample["inplane_sensor_id"], sample["timestamp"])
            inplane_figs.append(fig_in)
            print(f"    ✓ 面内图已生成（共 {len(inplane_figs)}）")

        if len(outplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            outplane_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
            fig_out = _plot_single(outplane_data, sample["outplane_sensor_id"], sample["timestamp"])
            outplane_figs.append(fig_out)
            print(f"    ✓ 面外图已生成（共 {len(outplane_figs)}）")

    return inplane_figs, outplane_figs


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 涡激共振时域波形绘制（面内 & 面外）")
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

    print("\n[步骤2] 筛选涡激共振（class 1）样本...")
    samples = get_viv_samples(result)
    print(f"✓ 共筛选到 {len(samples)} 个涡激共振样本")

    print("\n[步骤3] 随机抽取样本...")
    samples = random_sample(samples)

    print("\n[步骤4] 加载原始数据并绘图...")
    unpacker = UNPACK(init_path=False)
    inplane_figs, outplane_figs = plot_viv_timeseries(samples, unpacker)

    print("\n" + "=" * 80)
    print(f"✓ 面内图：{len(inplane_figs)} 张  |  面外图：{len(outplane_figs)} 张")
    print("=" * 80)

    ploter = PlotLib()
    for fig in inplane_figs + outplane_figs:
        ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
