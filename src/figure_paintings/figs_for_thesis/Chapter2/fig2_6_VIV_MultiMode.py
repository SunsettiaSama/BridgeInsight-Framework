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

_project_root = Path(__file__).parent.parent.parent.parent
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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

RSR = 10
CSTR = 10


# ==================== 常量配置 ====================
class Config:
    FS = 50.0

    WINDOW_SIZE = 3000
    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 120

    NFFT = 2048
    WINDOW_DURATION = 60
    STEP_DURATION = 5

    FIG_SIZE = SQUARE_FIG_SIZE

    ELEV = 30
    AZIM = 150

    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    CURVE_ALPHA = 0.5
    CURVE_LINEWIDTH = 1.5

    MAIN_PLOT_WIDTH_RATIO = 4
    SUB_PLOT_WIDTH_RATIO = 1
    PLOT_SPACING_RATIO = 0.1
    SUBPLOT_WSPACE = 0.2

    SUB_PLOT_FREQ_RATIO = 0.1

    USE_LOG_SCALE_Y = False

    NUM_SAMPLES_TO_PLOT = 20

    SMOOTHING_SIGMA = 1.5
    INTERPOLATION_FACTOR = 20

    APPLY_DENOISE = False
    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'


# ==================== 数据获取函数 ====================
def get_viv_windows():
    annotation_file = _project_root / _paths_cfg['annotation']['viv_sample']

    if not annotation_file.exists():
        raise ValueError(f"找不到标注文件: {annotation_file}")

    print("[获取数据] 读取标注数据文件...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)

    print(f"✓ 读取到 {len(annotation_data)} 条标注记录")

    viv_records = [item for item in annotation_data if item.get('annotation') == '1']
    print(f"✓ 其中 VIV (annotation=='1') 的记录：{len(viv_records)} 条")

    if not viv_records:
        raise ValueError("无 VIV 标注的记录")

    unpacker = UNPACK(init_path=False)
    all_windows = []

    print("\n[加载数据] 正在加载极端窗口数据...")
    for record in viv_records:
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

            all_windows.append({
                'data': window_data,
                'sensor_id': sensor_id,
                'time': time_str,
                'window_index': window_idx,
                'file_path': file_path,
            })

    print(f"✓ 成功加载 {len(all_windows)} 个极端窗口")
    return all_windows


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
def plot_multi_time_window_psd(data, fs=Config.FS, nfft=None):
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
    psd_data_list = []

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

        psd_data_list.append(p)
        center_idx += samples_per_step

    psd_matrix = np.array(psd_data_list)
    psd_matrix_smoothed = gaussian_filter(psd_matrix, sigma=Config.SMOOTHING_SIGMA)

    freq_mask = (freqs >= 0) & (freqs <= fs / 2)
    freqs_filtered = freqs[freq_mask]
    psd_matrix_filtered = psd_matrix_smoothed[:, freq_mask]

    psd_curve = np.mean(psd_matrix_filtered, axis=0)

    max_psd_overall = np.max(psd_curve)
    max_psd_freq_idx = np.argmax(psd_curve)
    max_freq = freqs_filtered[max_psd_freq_idx]

    main_freq_range = freqs_filtered[-1] - freqs_filtered[0]
    sub_freq_range = main_freq_range * Config.SUB_PLOT_FREQ_RATIO
    sub_freq_min = max(freqs_filtered[0], max_freq - sub_freq_range / 2)
    sub_freq_max = min(freqs_filtered[-1], max_freq + sub_freq_range / 2)

    fig = plt.figure(figsize=(14, 7))
    width_ratios = [
        Config.MAIN_PLOT_WIDTH_RATIO,
        Config.PLOT_SPACING_RATIO,
        Config.SUB_PLOT_WIDTH_RATIO,
        Config.PLOT_SPACING_RATIO,
    ]
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=Config.SUBPLOT_WSPACE)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[2])

    ax_main.plot(freqs_filtered, psd_curve, color='steelblue', linewidth=Config.CURVE_LINEWIDTH)

    sub_freq_mask = (freqs_filtered >= sub_freq_min) & (freqs_filtered <= sub_freq_max)
    ax_sub.plot(freqs_filtered[sub_freq_mask], psd_curve[sub_freq_mask],
                color='steelblue', linewidth=Config.CURVE_LINEWIDTH)

    ax_main.set_xlim(freqs_filtered[0], freqs_filtered[-1])
    if Config.USE_LOG_SCALE_Y:
        ax_main.set_yscale('log')
        ax_main.set_ylim(max_psd_overall / 1e3, max_psd_overall * 10)
    else:
        ax_main.set_ylim(0, max_psd_overall * 1.15)
    ax_main.set_xlabel('频率 (Hz)', fontsize=FONT_SIZE, fontproperties=CN_FONT)
    ylabel_main = '功率谱密度 (对数)' if Config.USE_LOG_SCALE_Y else '功率谱密度'
    ax_main.set_ylabel(ylabel_main, fontsize=FONT_SIZE, fontproperties=CN_FONT)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.tick_params(labelsize=FONT_SIZE)

    ax_sub.set_xlim(sub_freq_min, sub_freq_max)
    sub_psd_values = psd_curve[sub_freq_mask]
    if Config.USE_LOG_SCALE_Y:
        ax_sub.set_yscale('log')
        sub_psd_max = np.max(sub_psd_values)
        ax_sub.set_ylim(sub_psd_max / 1e3, sub_psd_max * 10)
    else:
        ax_sub.set_ylim(0, np.max(sub_psd_values) * 1.15)
    ax_sub.set_xlabel('频率 (Hz)', fontsize=FONT_SIZE - 1, fontproperties=CN_FONT)
    ax_sub.set_ylabel('功率谱密度', fontsize=FONT_SIZE - 1, fontproperties=CN_FONT)
    ax_sub.grid(True, alpha=0.3, linestyle='--')
    ax_sub.tick_params(labelsize=FONT_SIZE)

    return fig, (ax_main, ax_sub)


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


def main():
    print("=" * 80)
    print("VIV Frequency Domain Multi-Window PSD Visualization")
    print("=" * 80)

    print("\n[Step 1] Loading VIV window data...")
    windows = get_viv_windows()

    print("\n[Step 2] Generating multi-window PSD plots with peak energy zoom...")
    ploter = PlotLib()

    for i, window_info in enumerate(windows[:Config.NUM_SAMPLES_TO_PLOT], 1):
        data = window_info['data']
        data_src = _wavelet_denoise(data) if Config.APPLY_DENOISE else data
        data_2d = data_src.reshape(-1, 1)

        sensor_id = window_info['sensor_id']
        time_str = window_info['time']
        window_idx = window_info['window_index']

        fig, (ax_main, ax_sub) = plot_multi_time_window_psd(data_2d, fs=Config.FS, nfft=Config.NFFT)
        title = (f"{sensor_id} @ {time_str} (窗口 {window_idx})\n"
                 f"多时间窗口频域PSD分析 | NFFT: {Config.NFFT}")
        fig.suptitle(title, fontsize=FONT_SIZE, fontproperties=CN_FONT)
        ploter.figs.append(fig)
        print(f"  ✓ Sample {i} completed")

    print("\n" + "=" * 80)
    print(f"✓ Successfully generated {len(ploter.figs)} plots ({Config.NUM_SAMPLES_TO_PLOT} samples)")
    print("=" * 80 + "\n")

    ploter.show()
    return ploter


if __name__ == '__main__':
    main()
