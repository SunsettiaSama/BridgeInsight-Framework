import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
import matplotlib.ticker as ticker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelet import wavelet_denoise
from src.visualize_tools.utils import PlotLib
from src.config.wavelet.config import WaveletDenoisingConfig
from .config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_gray_to_red_color_map
)
cmap_func = get_gray_to_red_color_map

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
    NFFT = 1024
    WINDOW_DURATION = 60    # 秒
    STEP_DURATION = 1       # 秒
    
    # 频率范围特定的NFFT配置（用于不同频率范围的细粒度控制）
    NFFT_LOW = 1024         # 低频范围 (0~2Hz) 的NFFT - 获得更好频率分辨率
    NFFT_HIGH = 256        # 高频范围 (2~25Hz) 的NFFT - 平衡时频分辨率
    NFFT_TOTAL = 256       # 全频范围 (0~25Hz) 的NFFT - 平衡时频分辨率
    
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
    
    # 频率范围配置
    FREQ_BAND_LOW_MIN = 0      # 低频范围最小值 (Hz)
    FREQ_BAND_LOW_MAX = 2      # 低频范围最大值 (Hz)
    FREQ_BAND_HIGH_MIN = 2     # 高频范围最小值 (Hz)
    FREQ_BAND_HIGH_MAX = 25    # 高频范围最大值 (Hz)
    
    # 绘图样本数配置
    NUM_SAMPLES_TO_PLOT = 3    # 要绘制的样本数
    
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
def get_rwiv_windows():
    """
    从 RWIVSample/annotation_results.json 中读取元数据，
    筛选出标注为 RWIV (annotation=="2") 的窗口数据
    
    返回：
        list: 窗口数据列表，每项为包含窗口数据和元数据的字典
    """
    annotation_file = os.path.join(
        project_root, 
        "results", 
        "figs", 
        "figs_for_thesis", 
        "RWIVSample", 
        "annotation_results.json"
    )
    
    if not os.path.exists(annotation_file):
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
    all_windows = []
    
    print("\n[加载数据] 正在加载窗口数据...")
    for i, record in enumerate(rwiv_records):
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
    
    print(f"✓ 成功加载 {len(all_windows)} 个窗口")
    
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
        denoised_data, denoise_info = wavelet_denoise(
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
    风雨联合诱发振动 (RWIV) 时域和频域3D绘制主函数
    为每个样本生成三个频率范围的PSD图（0~2Hz、2~25Hz 和 0~25Hz）
    """
    print("="*80)
    print("RWIV 频域3D绘制（三个频率范围）")
    print("="*80)
    
    print("\n[步骤1] 获取 RWIV 窗口数据...")
    windows = get_rwiv_windows()
    
    print("\n[步骤2] 生成绘图...")
    ploter = PlotLib()
    
    for i, window_info in enumerate(windows[:Config.NUM_SAMPLES_TO_PLOT], 1):
        data = window_info['data']
        
        print(f"  ✓ 样本 {i}: 正在进行小波去噪预处理...")
        data_denoised, denoise_info = preprocess_data_with_wavelet_denoise(data)
        data_2d = data_denoised.reshape(-1, 1)
        
        sensor_id = window_info['sensor_id']
        time_str = window_info['time']
        window_idx = window_info['window_index']
        
        # 生成低频范围图 (0~2Hz) - 使用NFFT_LOW获得更好的频率分辨率
        print(f"  ✓ 样本 {i}: 生成低频范围 {Config.FREQ_BAND_LOW_MIN}~{Config.FREQ_BAND_LOW_MAX}Hz 图表 (NFFT={Config.NFFT_LOW})...")
        fig_low, ax_low = plot_3d_vibration_psd_frequency_band(
            data_2d, 
            Config.FREQ_BAND_LOW_MIN, 
            Config.FREQ_BAND_LOW_MAX,
            nfft=Config.NFFT_LOW
        )
        fig_low.suptitle(
            f"[图1-低频] {sensor_id} @ {time_str} (窗口 {window_idx})\n"
            f"频率范围: {Config.FREQ_BAND_LOW_MIN}~{Config.FREQ_BAND_LOW_MAX}Hz | NFFT: {Config.NFFT_LOW}",
            fontproperties=CN_FONT, fontsize=FONT_SIZE
        )
        ploter.figs.append(fig_low)
        
        # 生成高频范围图 (2~25Hz) - 使用NFFT_HIGH平衡时频分辨率
        print(f"  ✓ 样本 {i}: 生成高频范围 {Config.FREQ_BAND_HIGH_MIN}~{Config.FREQ_BAND_HIGH_MAX}Hz 图表 (NFFT={Config.NFFT_HIGH})...")
        fig_high, ax_high = plot_3d_vibration_psd_frequency_band(
            data_2d, 
            Config.FREQ_BAND_HIGH_MIN, 
            Config.FREQ_BAND_HIGH_MAX,
            nfft=Config.NFFT_HIGH
        )
        fig_high.suptitle(
            f"[图2-高频] {sensor_id} @ {time_str} (窗口 {window_idx})\n"
            f"频率范围: {Config.FREQ_BAND_HIGH_MIN}~{Config.FREQ_BAND_HIGH_MAX}Hz | NFFT: {Config.NFFT_HIGH}",
            fontproperties=CN_FONT, fontsize=FONT_SIZE
        )
        ploter.figs.append(fig_high)
        
        # 生成完整频率范围图 (0~25Hz) - 不分区间，全频谱展示
        print(f"  ✓ 样本 {i}: 生成完整频率范围 {Config.FREQ_BAND_LOW_MIN}~{Config.FREQ_BAND_HIGH_MAX}Hz 图表 (NFFT={Config.NFFT_TOTAL})...")
        fig_full, ax_full = plot_3d_vibration_psd_frequency_band(
            data_2d, 
            Config.FREQ_BAND_LOW_MIN, 
            Config.FREQ_BAND_HIGH_MAX,
            nfft=Config.NFFT_TOTAL
        )
        fig_full.suptitle(
            f"[图3-完整] {sensor_id} @ {time_str} (窗口 {window_idx})\n"
            f"频率范围: {Config.FREQ_BAND_LOW_MIN}~{Config.FREQ_BAND_HIGH_MAX}Hz | NFFT: {Config.NFFT_HIGH}",
            fontproperties=CN_FONT, fontsize=FONT_SIZE
        )
        ploter.figs.append(fig_full)
        
        print(f"  ✓ 样本 {i} 已生成 3 个图表（共 {len(ploter.figs)} 个）")
    
    print("\n" + "="*80)
    print(f"✓ 成功生成 {len(ploter.figs)} 个图表（{Config.NUM_SAMPLES_TO_PLOT} 个样本 × 3个频率范围）")
    print("="*80 + "\n")
    
    ploter.show()
    return ploter


if __name__ == '__main__':
    main()
