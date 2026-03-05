import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
import matplotlib.ticker as ticker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.utils import PlotLib
from src.config.wavelet.config import WaveletDenoisingConfig
from .config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_gray_to_red_color_map
)
cmap_func = get_gray_to_red_color_map

plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

RSR = 10
CSTR = 10

# ==================== 常量配置 ====================
class Config:
    # 采样频率
    FS = 50.0

    # 极端窗口配置
    WINDOW_SIZE = 3000  # 60秒窗口 @ 50Hz
    WINDOW_DURATION_SECONDS = 60
    
    # 数据截取配置
    TRIM_START_SECOND = 0  # 从第几秒开始截取（秒）
    TRIM_END_SECOND = 120   # 截取到第几秒（秒），设为None表示不截取
    
    # FFT和滑动窗口配置
    NFFT = 2048
    WINDOW_DURATION = 60    # 秒
    STEP_DURATION = 5       # 秒
    
    # 绘图配置 - 从 config.py 继承
    FIG_SIZE = SQUARE_FIG_SIZE

    # 3D绘图视角
    ELEV = 30
    AZIM = 150
    
    # 网格线配置
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'
    
    # 曲线绘图配置
    CURVE_ALPHA = 0.5              # 曲线透明度（浅色效果）
    CURVE_LINEWIDTH = 1.5           # 曲线宽度
    
    # 窗口比例配置
    MAIN_PLOT_WIDTH_RATIO = 4       # 主图宽度比例
    SUB_PLOT_WIDTH_RATIO = 1        # 子图宽度比例
    PLOT_SPACING_RATIO = 0.1        # 主图和子图之间的间隔比例
    SUBPLOT_WSPACE = 0.2            # 子图间的水平间距
    
    # 子图频率范围配置
    SUB_PLOT_FREQ_RATIO = 0.1       # 子图频率范围相对于主图的比例（例：0.1表示子图范围为主图的10%）
    
    # Y轴坐标系配置
    USE_LOG_SCALE_Y = False         # 是否使用Y轴对数坐标（True: 对数, False: 线性）
    
    # 绘图样本数配置
    NUM_SAMPLES_TO_PLOT = 20    # 要绘制的样本数
    
    # 3D表面平滑配置
    SMOOTHING_SIGMA = 1.5      # 高斯平滑的标准差（值越大平滑越强）
    
    # 数据加密配置（插值）
    INTERPOLATION_FACTOR = 20   # 插值系数，数据点会增加此倍数

    # 小波去噪配置（使用自定义阈值）
    ENABLE_WAVELET_DENOISE = False                           # 是否启用小波去噪
    WAVELET_TYPE = WaveletDenoisingConfig.WAVELET_TYPE      # 小波基类型
    WAVELET_LEVEL = WaveletDenoisingConfig.WAVELET_LEVEL    # 分解层数
    THRESHOLD_TYPE = WaveletDenoisingConfig.THRESHOLD_TYPE  # 阈值类型
    CUSTOM_THRESHOLD = WaveletDenoisingConfig.get_threshold('custom_1')  # 使用自定义阈值常量


# ==================== 数据获取函数 ====================
def get_viv_windows():
    """
    从 VIVSample/annotation_results.json 中读取元数据，
    筛选出标注为 VIV (annotation=="1") 的窗口数据
    
    返回：
        list: 极端窗口数据列表，每项为包含窗口数据和元数据的字典
    """
    annotation_file = os.path.join(
        project_root, 
        "results", 
        "figs", 
        "figs_for_thesis", 
        "VIVSample", 
        "annotation_results.json"
    )
    
    if not os.path.exists(annotation_file):
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
    for i, record in enumerate(viv_records):
        metadata = record['metadata']
        file_path = metadata['file_path']
        sensor_id = record['sensor_id']
        time_str = record['time']
        window_idx = record['window_index']
        
        try:
            vibration_data = unpacker.VIC_DATA_Unpack(file_path)
            vibration_data = np.array(vibration_data)
            
            start_sample = window_idx * Config.WINDOW_SIZE
            end_sample = (window_idx + 1) * Config.WINDOW_SIZE
            
            if end_sample <= len(vibration_data):
                window_data = vibration_data[start_sample:end_sample]
                
                trim_start_idx = int(Config.TRIM_START_SECOND * Config.FS)
                trim_end_idx = int(Config.TRIM_END_SECOND * Config.FS) if Config.TRIM_END_SECOND is not None else len(window_data)
                window_data = window_data[trim_start_idx:trim_end_idx]
                
                window_info = {
                    'data': window_data,
                    'sensor_id': sensor_id,
                    'time': time_str,
                    'window_index': window_idx,
                    'file_path': file_path
                }
                all_windows.append(window_info)
        except Exception as e:
            print(f"  ⚠ 加载失败 {sensor_id} {time_str}: {e}")
    
    print(f"✓ 成功加载 {len(all_windows)} 个极端窗口")
    
    return all_windows


# ==================== 数据预处理函数 ====================
def preprocess_data_with_wavelet_denoise(data):
    """
    使用小波去噪对数据进行预处理
    
    参数：
        data: 原始振动数据（一维数组）
    
    返回：
        denoised_data: 去噪后的数据
        denoise_info: 去噪信息字典
    """
    if not Config.ENABLE_WAVELET_DENOISE:
        return data, {}
    
    try:
        denoised_data, denoise_info = denoise(
            signal=data,
            wavelet=Config.WAVELET_TYPE,
            level=Config.WAVELET_LEVEL,
            threshold_type=Config.THRESHOLD_TYPE,
            threshold_method=Config.THRESHOLD_METHOD
        )
        print(f"    ✓ 小波去噪成功 (小波基: {Config.WAVELET_TYPE}, 层数: {Config.WAVELET_LEVEL}, 阈值: {denoise_info['threshold']:.4f})")
        return denoised_data, denoise_info
    except Exception as e:
        print(f"    ⚠ 小波去噪失败: {e}，使用原始数据")
        return data, {}


# ==================== 绘图函数 ====================
def plot_multi_time_window_psd(data, fs=Config.FS, nfft=None, num_curves=5):
    """
    绘制多时间窗口叠加PSD曲线（主图 + 高能量子图）
    
    参数：
        data: 一维或二维振动数据
        fs: 采样频率
        nfft: FFT大小（可选，默认使用Config.NFFT）
        num_curves: 要绘制的曲线数量（默认5条）
    
    返回：
        (fig, (ax_main, ax_sub)): matplotlib 图表对象
    """
    if nfft is None:
        nfft = Config.NFFT
    
    if data.ndim == 2:
        continuous_data = data.flatten()
    else:
        continuous_data = data
    
    window_duration = Config.WINDOW_DURATION
    step_duration = Config.STEP_DURATION
    
    samples_per_window = int(window_duration * fs)
    samples_per_step = int(step_duration * fs)
    
    if len(continuous_data) < samples_per_window:
        raise ValueError(f"数据长度不足，需要至少 {samples_per_window} 个采样点")
    
    nfft_half = int(nfft / 2)
    window_center_start = nfft_half
    window_center_end = len(continuous_data) - nfft_half
    
    freqs = None
    psd_data_list = []
    time_points = []
    
    center_idx = window_center_start
    while center_idx < window_center_end:
        start_idx = center_idx - nfft_half
        end_idx = start_idx + samples_per_window
        
        window_data = continuous_data[start_idx:end_idx]
        
        f, p = signal.welch(
            window_data, 
            fs=fs, 
            nperseg=int(nfft / 2), 
            noverlap=int(nfft / 4), 
            nfft=nfft
        )
        
        if freqs is None:
            freqs = f
        
        psd_data_list.append(p)
        time_points.append(center_idx / fs)
        
        center_idx += samples_per_step
    
    psd_matrix = np.array(psd_data_list)
    psd_matrix_smoothed = gaussian_filter(psd_matrix, sigma=Config.SMOOTHING_SIGMA)
    time_axis = np.array(time_points) - time_points[0]
    
    freq_mask = (freqs >= 0) & (freqs <= fs / 2)
    freqs_filtered = freqs[freq_mask]
    psd_matrix_filtered = psd_matrix_smoothed[:, freq_mask]
    
    fig = plt.figure(figsize=(14, 7))
    width_ratios = [
        Config.MAIN_PLOT_WIDTH_RATIO,
        Config.PLOT_SPACING_RATIO,
        Config.SUB_PLOT_WIDTH_RATIO,
        Config.PLOT_SPACING_RATIO
    ]
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=Config.SUBPLOT_WSPACE)
    ax_main = fig.add_subplot(gs[0])
    ax_sub = fig.add_subplot(gs[2])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(psd_matrix_filtered)))
    
    step = max(1, len(psd_matrix_filtered) // num_curves)
    indices = list(range(0, len(psd_matrix_filtered), step))[-num_curves:]
    
    max_psd_overall = np.max(psd_matrix_filtered)
    max_psd_freq_idx = np.argmax(np.mean(psd_matrix_filtered, axis=0))
    max_freq = freqs_filtered[max_psd_freq_idx]
    
    main_freq_range = freqs_filtered[-1] - freqs_filtered[0]
    sub_freq_range = main_freq_range * Config.SUB_PLOT_FREQ_RATIO
    sub_freq_min = max(freqs_filtered[0], max_freq - sub_freq_range / 2)
    sub_freq_max = min(freqs_filtered[-1], max_freq + sub_freq_range / 2)
    
    for idx, frame_idx in enumerate(indices):
        psd_curve = psd_matrix_filtered[frame_idx, :]
        time_start = time_axis[frame_idx]
        time_end = time_start + window_duration
        
        label = f'{time_start:.1f}~{time_end:.1f}s'
        ax_main.plot(freqs_filtered, psd_curve, color=colors[frame_idx], 
                     linewidth=Config.CURVE_LINEWIDTH, label=label, 
                     alpha=Config.CURVE_ALPHA)
        
        sub_freq_mask = (freqs_filtered >= sub_freq_min) & (freqs_filtered <= sub_freq_max)
        ax_sub.plot(freqs_filtered[sub_freq_mask], psd_curve[sub_freq_mask], 
                   color=colors[frame_idx], linewidth=Config.CURVE_LINEWIDTH, 
                   alpha=Config.CURVE_ALPHA)
    
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
    ax_main.legend(loc='upper right', fontsize=FONT_SIZE-2,
                   title_fontsize=FONT_SIZE-1, prop=CN_FONT)
    
    ax_sub.set_xlim(sub_freq_min, sub_freq_max)
    sub_psd_filtered = psd_matrix_filtered[:, (freqs_filtered >= sub_freq_min) & (freqs_filtered <= sub_freq_max)]
    if Config.USE_LOG_SCALE_Y:
        ax_sub.set_yscale('log')
        sub_psd_max = np.max(sub_psd_filtered)
        ax_sub.set_ylim(sub_psd_max / 1e3, sub_psd_max * 10)
    else:
        ax_sub.set_ylim(0, np.max(sub_psd_filtered) * 1.15)
    ax_sub.set_xlabel('频率 (Hz)', fontsize=FONT_SIZE-1, fontproperties=CN_FONT)
    ax_sub.set_ylabel('功率谱密度', fontsize=FONT_SIZE-1, fontproperties=CN_FONT)
    ax_sub.grid(True, alpha=0.3, linestyle='--')
    ax_sub.tick_params(labelsize=FONT_SIZE)

    return fig, (ax_main, ax_sub)


def plot_3d_vibration_psd_frequency_band(data, freq_min, freq_max, fs=Config.FS, nfft=None):
    """
    绘制3D振动PSD可视化（指定频率范围）
    
    参数：
        data: 一维或二维振动数据
        freq_min: 最小频率（Hz）
        freq_max: 最大频率（Hz）
        fs: 采样频率
        nfft: FFT大小（可选，默认使用Config.NFFT）
    
    返回：
        (fig, ax): matplotlib 图表对象
    """
    if nfft is None:
        nfft = Config.NFFT
    
    if data.ndim == 2:
        continuous_data = data.flatten()
    else:
        continuous_data = data
    
    window_duration = Config.WINDOW_DURATION
    step_duration = Config.STEP_DURATION
    
    samples_per_window = int(window_duration * fs)
    samples_per_step = int(step_duration * fs)
    
    if len(continuous_data) < samples_per_window:
        raise ValueError(f"数据长度不足，需要至少 {samples_per_window} 个采样点")
    
    # 计算有效的窗口中心位置范围
    nfft_half = int(nfft / 2)
    window_center_start = nfft_half
    window_center_end = len(continuous_data) - nfft_half
    
    freqs = None
    psd_matrix = []
    time_points = []
    
    # 从起始位置逐步加算，步长为STEP_DURATION
    center_idx = window_center_start
    while center_idx < window_center_end:
        # 根据窗口中心位置计算窗口的起始和结束索引
        start_idx = center_idx - nfft_half
        end_idx = start_idx + samples_per_window
        
        window_data = continuous_data[start_idx:end_idx]
        
        f, p = signal.welch(
            window_data, 
            fs=fs, 
            nperseg=int(nfft / 2), 
            noverlap=int(nfft / 4), 
            nfft=nfft
        )
        
        if freqs is None:
            freqs = f
        
        psd_matrix.append(p)
        time_points.append(center_idx / fs)
        
        # 按步长STEP_DURATION加算移动窗口
        center_idx += samples_per_step
    
    psd_matrix = np.array(psd_matrix)
    
    # 对PSD矩阵进行高斯平滑处理
    psd_matrix_smoothed = gaussian_filter(psd_matrix, sigma=Config.SMOOTHING_SIGMA)
    
    # 时间轴基于采样点位置，单位为秒，归零化从0开始
    time_axis = np.array(time_points) - time_points[0]
    
    # 筛选频率范围
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_filtered = freqs[freq_mask]
    psd_matrix_filtered = psd_matrix_smoothed[:, freq_mask]
    
    # 数据加密（插值加密）
    num_times_orig = len(time_axis)
    num_freqs_orig = len(freqs_filtered)
    
    num_times_interp = (num_times_orig - 1) * Config.INTERPOLATION_FACTOR + 1
    num_freqs_interp = (num_freqs_orig - 1) * Config.INTERPOLATION_FACTOR + 1
    
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
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        shade=True, 
        rstride=RSR,
        cstride=CSTR
    )
    
    cbar = fig.colorbar(surf, pad=0.1)
    cbar.set_label('PSD', rotation=270, labelpad=20, fontproperties=ENG_FONT)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    
    ax.grid(True)
    ax.tick_params(
        axis='both',
        which='major',
        grid_color=Config.GRID_COLOR,
        grid_alpha=Config.GRID_ALPHA,
        grid_linewidth=Config.GRID_LINEWIDTH,
        grid_linestyle=Config.GRID_LINESTYLE,
        labelsize=FONT_SIZE
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
    """
    VIV Frequency Domain Multi-Window PSD Visualization
    Generate overlaid PSD curves with zoom inset for peak energy region
    """
    print("="*80)
    print("VIV Frequency Domain Multi-Window PSD Visualization")
    print("="*80)
    
    print("\n[Step 1] Loading VIV window data...")
    windows = get_viv_windows()
    
    print("\n[Step 2] Generating multi-window PSD plots with peak energy zoom...")
    ploter = PlotLib()
    
    for i, window_info in enumerate(windows[:Config.NUM_SAMPLES_TO_PLOT], 1):
        data = window_info['data']
        
        print(f"  ✓ Sample {i}: Performing wavelet denoising...")
        data_denoised, denoise_info = preprocess_data_with_wavelet_denoise(data)
        data_2d = data_denoised.reshape(-1, 1)
        
        sensor_id = window_info['sensor_id']
        time_str = window_info['time']
        window_idx = window_info['window_index']
        
        fig, (ax_main, ax_sub) = plot_multi_time_window_psd(
            data_2d, 
            fs=Config.FS,
            nfft=Config.NFFT,
            num_curves=5
        )
        title = f"{sensor_id} @ {time_str} (窗口 {window_idx})\n" \
                f"多时间窗口频域PSD分析 | NFFT: {Config.NFFT}"
        fig.suptitle(title, fontsize=FONT_SIZE, fontproperties=CN_FONT)
        ploter.figs.append(fig)
        print(f"  ✓ Sample {i}: Generated full-spectrum plot with peak energy zoom...")
        
        print(f"  ✓ Sample {i} completed")
    
    print("\n" + "="*80)
    print(f"✓ Successfully generated {len(ploter.figs)} plots ({Config.NUM_SAMPLES_TO_PLOT} samples)")
    print("="*80 + "\n")
    
    ploter.show()
    return ploter


if __name__ == '__main__':
    main()
