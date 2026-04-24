import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.utils import PlotLib
from ..config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_gray_to_red_color_map
)
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow

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
    TRIM_END_SECOND = 10   # 截取到第几秒（秒），设为None表示不截取
    
    # FFT和滑动窗口配置
    NFFT = 256              # 全频谱NFFT配置
    WINDOW_DURATION = 60    # 秒
    STEP_DURATION = 1       # 秒
    
    # 频率范围配置
    FREQ_MIN = 0            # 最小频率 (Hz)
    FREQ_MAX = 25           # 最大频率 (Hz)
    
    # 绘图配置 - 使用config导入的正方形配置
    FIG_SIZE = SQUARE_FIG_SIZE     # 独立图像采用正方形配置
    
    # 3D绘图视角
    ELEV = 30
    AZIM = 150
    
    # 网格线配置
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'
    
    # 时域绘图配置
    WAVEFORM_COLOR = '#333333'
    LINEWIDTH = 1.0
    
    # 数据加密配置（插值）
    INTERPOLATION_FACTOR = 20   # 插值系数
    
    # 3D表面平滑配置
    SMOOTHING_SIGMA = 1.5      # 高斯平滑的标准差
    
    # 随机抽样配置
    NUM_SAMPLES_TO_PLOT = 10    # 要抽样的记录数
    NUM_WINDOWS_TO_PLOT = 10    # 要绘制的窗口数（最终图表生成的窗口个数）
    RANDOM_SEED = 42            # 随机种子，保证结果可复现
    
    # 小波去噪配置
    ENABLE_WAVELET_DENOISE = True   # 是否启用小波去噪
    WAVELET_TYPE = 'db3'           # 小波基类型
    WAVELET_LEVEL = 5                # 分解层数
    THRESHOLD_TYPE = 'soft'          # 阈值类型（软阈值）
    THRESHOLD_METHOD = 'heursure'    # 阈值计算方法


# ==================== 数据获取函数 ====================
def get_extreme_samples_from_workflow():
    """
    从振动工作流中获取元数据，执行两层随机抽样：
    第一层：随机抽样10个包含极端窗口的记录
    第二层：从这些记录的所有极端窗口中随机抽样10个窗口
    
    返回：
        list: 最终选中的窗口数据列表，每项为包含窗口数据和元数据的字典
    """
    print("[获取数据] 运行振动工作流获取元数据...")
    metadata = run_vib_workflow(use_cache=True, force_recompute=False)
    
    print(f"✓ 从工作流获取 {len(metadata)} 条元数据记录")
    
    # 第一层筛选：筛选包含极端窗口的元数据
    records_with_extreme = [
        item for item in metadata 
        if len(item.get('extreme_rms_indices', [])) > 0
    ]
    print(f"✓ 其中包含极端窗口的记录：{len(records_with_extreme)} 条")
    
    if not records_with_extreme:
        raise ValueError("无包含极端窗口的记录")
    
    # 第一层抽样：固定随机种子并随机抽样N个记录
    np.random.seed(Config.RANDOM_SEED)
    num_records_to_sample = min(Config.NUM_SAMPLES_TO_PLOT, len(records_with_extreme))
    sampled_record_indices = np.random.choice(len(records_with_extreme), num_records_to_sample, replace=False)
    sampled_records = [records_with_extreme[i] for i in sampled_record_indices]
    
    print(f"✓ 第一层抽样：从 {len(records_with_extreme)} 条记录中随机抽样 {num_records_to_sample} 条")
    print(f"  (种子: {Config.RANDOM_SEED})")
    
    # 收集所有极端窗口信息
    unpacker = UNPACK(init_path=False)
    all_extreme_windows = []
    
    print("\n[加载数据] 从抽样的记录中收集极端窗口...")
    for i, record in enumerate(sampled_records, 1):
        file_path = record['file_path']
        sensor_id = record['sensor_id']
        time_str = f"{record.get('month', 'N/A')}/{record.get('day', 'N/A')} {record.get('hour', 'N/A')}:00"
        extreme_indices = record.get('extreme_rms_indices', [])
        
        try:
            vibration_data = unpacker.VIC_DATA_Unpack(file_path)
            vibration_data = np.array(vibration_data)
            
            # 对每个极端窗口进行处理
            for window_idx in extreme_indices:
                start_sample = window_idx * Config.WINDOW_SIZE
                end_sample = (window_idx + 1) * Config.WINDOW_SIZE
                
                if end_sample <= len(vibration_data):
                    window_data = vibration_data[start_sample:end_sample]
                    window_info = {
                        'data': window_data,
                        'sensor_id': sensor_id,
                        'time': time_str,
                        'window_index': window_idx,
                        'file_path': file_path
                    }
                    all_extreme_windows.append(window_info)
        except Exception as e:
            print(f"  ⚠ 加载失败 {sensor_id} {time_str}: {e}")
    
    print(f"✓ 从 {num_records_to_sample} 条记录中收集到 {len(all_extreme_windows)} 个极端窗口")
    
    if not all_extreme_windows:
        raise ValueError("未能加载任何极端窗口数据")
    
    # 第二层抽样：从所有极端窗口中随机抽样指定数量的窗口
    num_windows_to_sample = min(Config.NUM_WINDOWS_TO_PLOT, len(all_extreme_windows))
    sampled_window_indices = np.random.choice(len(all_extreme_windows), num_windows_to_sample, replace=False)
    final_windows = [all_extreme_windows[i] for i in sampled_window_indices]
    
    print(f"\n✓ 第二层抽样：从 {len(all_extreme_windows)} 个窗口中随机抽样 {num_windows_to_sample} 个")
    print(f"  (种子: {Config.RANDOM_SEED})")
    print(f"\n✓ 最终加载 {len(final_windows)} 个极端窗口用于绘制")
    
    return final_windows


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
def plot_time_series_figure(data, window_info, denoise_state="original", fs=Config.FS):
    """
    绘制时域波形为独立图像
    
    参数：
        data: 一维振动数据
        window_info: 包含窗口信息的字典
        denoise_state: "original" 或 "denoised"
        fs: 采样频率
    
    返回：
        fig: matplotlib figure对象
    """
    trim_start_idx = int(Config.TRIM_START_SECOND * fs)
    trim_end_idx = int(Config.TRIM_END_SECOND * fs) if Config.TRIM_END_SECOND is not None else len(data)
    data_plot = data[trim_start_idx:trim_end_idx]
    
    time_axis = np.arange(len(data_plot)) / fs + Config.TRIM_START_SECOND
    
    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)
    
    ax.plot(
        time_axis, 
        data_plot, 
        color=Config.WAVEFORM_COLOR,
        linewidth=Config.LINEWIDTH
    )
    
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, 
            linewidth=Config.GRID_LINEWIDTH, linestyle=Config.GRID_LINESTYLE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    sensor_id = window_info['sensor_id']
    time_str = window_info['time']
    window_idx = window_info['window_index']
    denoise_label = "去噪前" if denoise_state == "original" else "去噪后"
    
    fig.suptitle(
        f"{sensor_id} @ {time_str} (窗口 {window_idx}) - {denoise_label} 时程",
        fontproperties=CN_FONT, fontsize=FONT_SIZE, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def plot_3d_timefrequency_spectrum(data, window_info, denoise_state="original", fs=Config.FS, freq_min=Config.FREQ_MIN, freq_max=Config.FREQ_MAX):
    """
    绘制3D时频谱为独立图像
    
    参数：
        data: 一维振动数据
        window_info: 包含窗口信息的字典
        denoise_state: "original" 或 "denoised"
        fs: 采样频率
        freq_min: 最小频率 (Hz)
        freq_max: 最大频率 (Hz)
    
    返回：
        fig: matplotlib figure对象
    """
    if data.ndim == 2:
        continuous_data = data.flatten()
    else:
        continuous_data = data
    
    window_duration = Config.WINDOW_DURATION
    step_duration = Config.STEP_DURATION
    nfft = Config.NFFT
    
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
    
    sensor_id = window_info['sensor_id']
    time_str = window_info['time']
    window_idx = window_info['window_index']
    denoise_label = "去噪前" if denoise_state == "original" else "去噪后"
    
    fig.suptitle(
        f"{sensor_id} @ {time_str} (窗口 {window_idx}) - {denoise_label} 时频谱\n"
        f"频率范围: {freq_min}~{freq_max}Hz | NFFT: {Config.NFFT}",
        fontproperties=CN_FONT, fontsize=FONT_SIZE, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def generate_sample_figures(window_info):
    """
    为一个样本生成4个独立图像：
    - 去噪前时程
    - 去噪前时频谱
    - 去噪后时程
    - 去噪后时频谱
    
    参数：
        window_info: 包含窗口数据和元数据的字典
    
    返回：
        (fig_original_time, fig_original_freq, fig_denoised_time, fig_denoised_freq): 四个figure对象
    """
    data_original = window_info['data']
    data_denoised, denoise_info = preprocess_data_with_wavelet_denoise(data_original)
    
    # 去噪前时程
    fig_original_time = plot_time_series_figure(data_original, window_info, denoise_state="original")
    
    # 去噪前时频谱
    fig_original_freq = plot_3d_timefrequency_spectrum(data_original, window_info, denoise_state="original")
    
    # 去噪后时程
    fig_denoised_time = plot_time_series_figure(data_denoised, window_info, denoise_state="denoised")
    
    # 去噪后时频谱
    fig_denoised_freq = plot_3d_timefrequency_spectrum(data_denoised, window_info, denoise_state="denoised")
    
    return fig_original_time, fig_original_freq, fig_denoised_time, fig_denoised_freq


def main():
    """
    绘图主函数：生成综合的去噪前后效果对比图
    执行两层随机抽样：
    - 第一层：从工作流元数据中随机抽样 NUM_SAMPLES_TO_PLOT 个包含极端窗口的记录
    - 第二层：从这些记录的所有极端窗口中随机抽样 NUM_WINDOWS_TO_PLOT 个窗口
    为每个窗口生成4个独立图像
    """
    try:
        print("="*80)
        print("小波去噪效果展示 - 时程与时频谱对比")
        print("="*80)
        
        print("\n[步骤1] 从工作流获取数据并执行两层随机抽样...")
        windows = get_extreme_samples_from_workflow()
        
        print("\n[步骤2] 生成绘图...")
        ploter = PlotLib()
        figs = []
        
        for i, window_info in enumerate(windows, 1):
            print(f"  ✓ 窗口 {i}/{len(windows)}: 生成时程和时频谱图像...")
            fig_original_time, fig_original_freq, fig_denoised_time, fig_denoised_freq = generate_sample_figures(window_info)
            
            figs.append(fig_original_time)
            figs.append(fig_original_freq)
            figs.append(fig_denoised_time)
            figs.append(fig_denoised_freq)
            
            ploter.figs.append(fig_original_time)
            ploter.figs.append(fig_original_freq)
            ploter.figs.append(fig_denoised_time)
            ploter.figs.append(fig_denoised_freq)
            
            print(f"    ✓ 已生成 {len(figs)} 个图表")
        
        print("\n" + "="*80)
        print(f"✓ 成功生成 {len(figs)} 个图表")
        print(f"  - 抽样记录数: {Config.NUM_SAMPLES_TO_PLOT}")
        print(f"  - 抽样窗口数: {Config.NUM_WINDOWS_TO_PLOT}")
        print(f"  - 每个窗口4张图: {Config.NUM_WINDOWS_TO_PLOT} × 4 = {len(figs)}")
        print("="*80 + "\n")
        
        ploter.show()
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        raise


if __name__ == "__main__":
    main()
