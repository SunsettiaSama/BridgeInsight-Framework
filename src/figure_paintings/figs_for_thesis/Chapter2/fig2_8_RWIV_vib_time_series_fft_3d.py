import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_gray_to_red_color_map,
)

_paths_cfg = yaml.safe_load((_project_root / "config" / "io" / "paths.yaml").read_text(encoding='utf-8'))
cmap_func = get_gray_to_red_color_map

RSR = 10
CSTR = 10


# ==================== 常量配置 ====================
class Config:
    FS = 50.0

    WINDOW_SIZE = 3000
    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 120

    NFFT = 1024
    WINDOW_DURATION = 60
    STEP_DURATION = 1

    NFFT_LOW = 1024
    NFFT_HIGH = 256
    NFFT_TOTAL = 256

    FIG_SIZE = SQUARE_FIG_SIZE

    ELEV = 30
    AZIM = 150

    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    FREQ_BAND_LOW_MIN = 0
    FREQ_BAND_LOW_MAX = 2
    FREQ_BAND_HIGH_MIN = 2
    FREQ_BAND_HIGH_MAX = 25

    NUM_IN_PLANE = 2    # 面内抽样数量
    NUM_OUT_PLANE = 2   # 面外抽样数量
    RANDOM_SEED = 42

    IN_PLANE_SUFFIX = '-01'    # 面内传感器通道后缀
    OUT_PLANE_SUFFIX = '-02'   # 面外传感器通道后缀

    SMOOTHING_SIGMA = 1.5
    INTERPOLATION_FACTOR = 20

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
        raise ValueError("无 RWIV 标注的记录")

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

            trim_start_idx = int(Config.TRIM_START_SECOND * Config.FS)
            trim_end_idx = int(Config.TRIM_END_SECOND * Config.FS) if Config.TRIM_END_SECOND is not None else len(window_data)
            window_data = window_data[trim_start_idx:trim_end_idx]

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
def plot_3d_vibration_psd_frequency_band(data, freq_min, freq_max, fs=Config.FS, nfft=None):
    if nfft is None:
        nfft = Config.NFFT

    continuous_data = data.flatten() if data.ndim == 2 else data

    samples_per_window = int(Config.WINDOW_DURATION * fs)
    samples_per_step = int(Config.STEP_DURATION * fs)

    if len(continuous_data) < samples_per_window:
        raise ValueError(f"数据长度不足，需要至少 {samples_per_window} 个采样点")

    nfft_half = int(nfft / 2)
    window_center_start = nfft_half
    window_center_end = len(continuous_data) - nfft_half

    freqs = None
    psd_matrix = []
    time_points = []

    center_idx = window_center_start
    while center_idx < window_center_end:
        start_idx = center_idx - nfft_half
        end_idx = start_idx + samples_per_window

        f, p = signal.welch(
            continuous_data[start_idx:end_idx],
            fs=fs,
            nperseg=int(nfft / 2),
            noverlap=int(nfft / 4),
            nfft=nfft,
        )

        if freqs is None:
            freqs = f

        psd_matrix.append(p)
        time_points.append(center_idx / fs)
        center_idx += samples_per_step

    psd_matrix = np.array(psd_matrix)
    psd_matrix_smoothed = gaussian_filter(psd_matrix, sigma=Config.SMOOTHING_SIGMA)
    time_axis = np.array(time_points) - time_points[0]

    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_filtered = freqs[freq_mask]
    psd_matrix_filtered = psd_matrix_smoothed[:, freq_mask]

    num_times_interp = (len(time_axis) - 1) * Config.INTERPOLATION_FACTOR + 1
    num_freqs_interp = (len(freqs_filtered) - 1) * Config.INTERPOLATION_FACTOR + 1

    time_axis_interp = np.linspace(time_axis[0], time_axis[-1], num_times_interp)
    freqs_filtered_interp = np.linspace(freqs_filtered[0], freqs_filtered[-1], num_freqs_interp)

    spline = RectBivariateSpline(time_axis, freqs_filtered, psd_matrix_filtered, kx=3, ky=3)
    psd_matrix_interp = spline(time_axis_interp, freqs_filtered_interp)

    T, F = np.meshgrid(time_axis_interp, freqs_filtered_interp)

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')

    cmap = cmap_func(style='gradient')

    surf = ax.plot_surface(
        T, F, psd_matrix_interp.T,
        cmap=cmap, linewidth=0, antialiased=True, alpha=0.9, shade=True,
        rstride=RSR, cstride=CSTR,
    )

    cbar = fig.colorbar(surf, pad=0.1)
    cbar.set_label('PSD', rotation=270, labelpad=20, fontproperties=ENG_FONT)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    ax.grid(True)
    ax.tick_params(
        axis='both', which='major',
        grid_color=Config.GRID_COLOR, grid_alpha=Config.GRID_ALPHA,
        grid_linewidth=Config.GRID_LINEWIDTH, grid_linestyle=Config.GRID_LINESTYLE,
        labelsize=FONT_SIZE,
    )
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.set_xlabel('时间 (秒)', labelpad=10, fontproperties=CN_FONT)
    ax.set_ylabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT)
    ax.set_zlabel('PSD', labelpad=16, fontproperties=ENG_FONT)
    ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.view_init(elev=Config.ELEV, azim=Config.AZIM)

    return fig, ax


def _sample(pool: list, n: int, rng: np.random.Generator) -> list:
    k = min(n, len(pool))
    indices = sorted(rng.choice(len(pool), size=k, replace=False))
    return [pool[i] for i in indices]


def main():
    print("=" * 80)
    print("RWIV 频域3D绘制（面内 + 面外）")
    print("=" * 80)

    print("\n[步骤1] 获取 RWIV 窗口数据...")
    in_plane_windows, out_plane_windows = get_rwiv_windows()

    rng = np.random.default_rng(Config.RANDOM_SEED)
    sampled_in = _sample(in_plane_windows, Config.NUM_IN_PLANE, rng)
    sampled_out = _sample(out_plane_windows, Config.NUM_OUT_PLANE, rng)
    sampled = sampled_in + sampled_out
    print(f"✓ 面内抽取 {len(sampled_in)} 个 / 面外抽取 {len(sampled_out)} 个（seed={Config.RANDOM_SEED}）")

    print("\n[步骤2] 生成绘图...")
    ploter = PlotLib()

    for i, window_info in enumerate(sampled, 1):
        data = window_info['data']
        data_src = _wavelet_denoise(data) if Config.APPLY_DENOISE else data
        data_2d = data_src.reshape(-1, 1)

        sensor_id = window_info['sensor_id']
        time_str = window_info['time']
        window_idx = window_info['window_index']

        print(f"  ✓ 样本 {i}: 生成完整频率范围 {Config.FREQ_BAND_LOW_MIN}~{Config.FREQ_BAND_HIGH_MAX}Hz 图表...")
        fig_full, _ = plot_3d_vibration_psd_frequency_band(
            data_2d, Config.FREQ_BAND_LOW_MIN, Config.FREQ_BAND_HIGH_MAX, nfft=Config.NFFT_TOTAL,
        )
        fig_full.suptitle(
            f"{sensor_id} @ {time_str} (窗口 {window_idx})",
            fontproperties=CN_FONT, fontsize=FONT_SIZE,
        )
        ploter.figs.append(fig_full)
        print(f"  ✓ 样本 {i} 已生成图表（共 {len(ploter.figs)} 个）")

    print("\n" + "=" * 80)
    print(f"✓ 成功生成 {len(ploter.figs)} 个图表（面内 {len(sampled_in)} + 面外 {len(sampled_out)}）")
    print("=" * 80 + "\n")

    ploter.show()
    return ploter


if __name__ == '__main__':
    main()
