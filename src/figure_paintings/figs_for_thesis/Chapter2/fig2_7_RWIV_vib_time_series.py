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
    NUM_IN_PLANE = 2    # 面内抽样数量
    NUM_OUT_PLANE = 2   # 面外抽样数量
    RANDOM_SEED = 40

    IN_PLANE_SUFFIX = '-01'    # 面内传感器通道后缀
    OUT_PLANE_SUFFIX = '-02'   # 面外传感器通道后缀

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 3

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
def get_rwiv_windows():
    annotation_file = _project_root / _paths_cfg['annotation']['rwiv_sample']

    if not annotation_file.exists():
        raise ValueError(f"找不到标注文件: {annotation_file}")

    print("[获取数据] 读取标注数据文件...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)

    print(f"✓ 读取到 {len(annotation_data)} 条标注记录")

    rwiv_records = [item for item in annotation_data if item.get('annotation') == '2']
    print(f"✓ 其中 RWIV (annotation=='2') 的记录：{len(rwiv_records)} 条")

    if not rwiv_records:
        raise ValueError("无 RWIV 标注记录")

    unpacker = UNPACK(init_path=False)
    in_plane_windows = []
    out_plane_windows = []

    print("\n[加载数据] 正在加载窗口数据...")
    for record in rwiv_records:
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
            entry = {
                'data': window_data,
                'sensor_id': sensor_id,
                'time': time_str,
                'window_index': window_idx,
                'file_path': file_path,
            }
            if sensor_id.endswith(Config.IN_PLANE_SUFFIX):
                in_plane_windows.append(entry)
            elif sensor_id.endswith(Config.OUT_PLANE_SUFFIX):
                out_plane_windows.append(entry)

    print(f"✓ 成功加载 面内 {len(in_plane_windows)} 个 / 面外 {len(out_plane_windows)} 个窗口")
    return in_plane_windows, out_plane_windows


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


def _sample(pool: list, n: int, rng: np.random.Generator) -> list:
    k = min(n, len(pool))
    indices = sorted(rng.choice(len(pool), size=k, replace=False))
    return [pool[i] for i in indices]


def main():
    print("=" * 80)
    print("RWIV 时域波形绘制（面内 + 面外）")
    print("=" * 80)

    print("\n[步骤1] 获取元数据和窗口数据...")
    in_plane_windows, out_plane_windows = get_rwiv_windows()

    rng = np.random.default_rng(Config.RANDOM_SEED)
    sampled_in = _sample(in_plane_windows, Config.NUM_IN_PLANE, rng)
    sampled_out = _sample(out_plane_windows, Config.NUM_OUT_PLANE, rng)
    print(f"✓ 面内抽取 {len(sampled_in)} 个 / 面外抽取 {len(sampled_out)} 个（seed={Config.RANDOM_SEED}）")

    sampled = sampled_in + sampled_out

    print("\n[步骤2] 生成绘图...")
    ploter = PlotLib()

    for i, window_info in enumerate(sampled, 1):
        fig, ax = plot_extreme_window(window_info)
        ploter.figs.append(fig)
        print(f"  ✓ 已生成图表 {i}")

    print("\n" + "=" * 80)
    print(f"✓ 成功生成 {len(ploter.figs)} 个图表（面内 {len(sampled_in)} + 面外 {len(sampled_out)}）")
    print("=" * 80 + "\n")

    ploter.show()


if __name__ == "__main__":
    main()
