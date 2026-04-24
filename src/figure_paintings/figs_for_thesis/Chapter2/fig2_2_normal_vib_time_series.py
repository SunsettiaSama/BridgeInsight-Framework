import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE

_paths_cfg = yaml.safe_load((_project_root / "config" / "io" / "paths.yaml").read_text(encoding='utf-8'))


# ==================== 常量配置 ====================
class Config:
    FS = 50.0

    WINDOW_SIZE = 3000
    NUM_SAMPLES_TO_PLOT = 4
    RANDOM_SEED = 42

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 10

    FIG_SIZE = SQUARE_FIG_SIZE
    WAVEFORM_COLOR = '#333333'
    LINEWIDTH = 1.0
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    APPLY_DENOISE = False
    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'


# ==================== 数据获取函数 ====================
def get_extreme_windows_from_metadata():
    annotation_file = _project_root / _paths_cfg['annotation']['rv_sample']

    if not annotation_file.exists():
        raise ValueError(f"找不到标注文件: {annotation_file}")

    print("[获取数据] 读取标注数据文件...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)

    print(f"✓ 读取到 {len(annotation_data)} 条标注记录")

    normal_records = [item for item in annotation_data if item.get('annotation') == '0']
    print(f"✓ 其中 Normal_Vib (annotation=='0') 的记录：{len(normal_records)} 条")

    if not normal_records:
        raise ValueError("无 Normal_Vib 标注的记录")

    unpacker = UNPACK(init_path=False)
    all_extreme_windows = []

    print("\n[加载数据] 正在加载极端窗口数据...")
    for record in normal_records:
        metadata = record['metadata']
        file_path = metadata['file_path']
        sensor_id = record['sensor_id']
        time_str = record['time']
        window_idx = record['window_index']

        vibration_data = np.array(unpacker.VIC_DATA_Unpack(file_path))

        start_sample = window_idx * Config.WINDOW_SIZE
        end_sample = (window_idx + 1) * Config.WINDOW_SIZE

        if end_sample <= len(vibration_data):
            window_data = vibration_data[start_sample:end_sample]
            all_extreme_windows.append({
                'data': window_data,
                'sensor_id': sensor_id,
                'time': time_str,
                'window_index': window_idx,
                'file_path': file_path,
            })

    print(f"✓ 成功加载 {len(all_extreme_windows)} 个极端窗口")
    return all_extreme_windows


# ==================== 数据预处理函数 ====================
def _wavelet_denoise(data: np.ndarray) -> np.ndarray:
    denoised, _ = denoise(
        signal=data,
        wavelet=Config.WAVELET_TYPE,
        level=Config.WAVELET_LEVEL,
        threshold_type=Config.THRESHOLD_TYPE,
        threshold_method=Config.THRESHOLD_METHOD,
    )
    return denoised


# ==================== 绘图函数 ====================
def plot_extreme_window(window_info, fs=Config.FS):
    data = window_info['data']

    if len(data) == 0:
        raise ValueError("窗口数据为空")

    data_src = _wavelet_denoise(data) if Config.APPLY_DENOISE else data

    trim_start_idx = int(Config.TRIM_START_SECOND * fs)
    trim_end_idx = int(Config.TRIM_END_SECOND * fs) if Config.TRIM_END_SECOND is not None else len(data_src)
    data_plot = data_src[trim_start_idx:trim_end_idx]

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)

    time_axis = np.arange(len(data_plot)) / fs + Config.TRIM_START_SECOND

    ax.plot(time_axis, data_plot, color=Config.WAVEFORM_COLOR, linewidth=Config.LINEWIDTH)

    title = f"{window_info['sensor_id']} @ {window_info['time']} (窗口 {window_info['window_index']})"
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH, linestyle=Config.GRID_LINESTYLE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    return fig, ax


def main():
    print("=" * 80)
    print("Normal_Vib 时域波形绘制")
    print("=" * 80)

    print("\n[步骤1] 获取元数据和极端窗口数据...")
    all_windows = get_extreme_windows_from_metadata()

    rng = np.random.default_rng(Config.RANDOM_SEED)
    n = min(Config.NUM_SAMPLES_TO_PLOT, len(all_windows))
    indices = sorted(rng.choice(len(all_windows), size=n, replace=False))
    sampled = [all_windows[i] for i in indices]
    print(f"✓ 随机抽取 {len(sampled)} 个窗口（seed={Config.RANDOM_SEED}）")

    print("\n[步骤2] 生成绘图...")
    ploter = PlotLib()

    for i, window_info in enumerate(sampled, 1):
        fig, ax = plot_extreme_window(window_info)
        ploter.figs.append(fig)
        print(f"  ✓ 已生成图表 {i}")

    print("\n" + "=" * 80)
    print(f"✓ 成功生成 {len(ploter.figs)} 个图表")
    print("=" * 80 + "\n")

    ploter.show()


if __name__ == "__main__":
    main()
