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
from src.data_processer.singals.wavelet import wavelet_denoise
from src.visualize_tools.utils import PlotLib
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
    NFFT = 512
    WINDOW_DURATION = 60    # 秒
    STEP_DURATION = 1       # 秒
    
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
    
    # 频率范围配置（全频段）
    FREQ_BAND_MIN = 0           # 最小频率 (Hz)
    FREQ_BAND_MAX = 25          # 最大频率 (Hz)
    
    # 绘图样本数配置
    NUM_SAMPLES_TO_PLOT = 3    # 要绘制的样本数
    
    # 3D表面平滑配置
    SMOOTHING_SIGMA = 1.5      # 高斯平滑的标准差（值越大平滑越强）
    
    # 数据加密配置（插值）
    INTERPOLATION_FACTOR = 20   # 插值系数，数据点会增加此倍数

    # 小波去噪配置
    ENABLE_WAVELET_DENOISE = True   # 是否启用小波去噪
    WAVELET_TYPE = 'db4'            # 小波基类型（Daubechies 4）
    WAVELET_LEVEL = 3               # 分解层数
    THRESHOLD_TYPE = 'soft'         # 阈值类型（软阈值）
    THRESHOLD_METHOD = 'sqtwolog'   # 阈值计算方法（平方根双对数法）


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
def plot_3d_vibration_psd_full_spectrum(data_original, data_denoised, fs=Config.FS, nfft=None):
    """
    绘制对比图：去噪前和去噪后的3D振动PSD可视化（全频谱，不分区间）
    
    参数：
        data_original: 原始（未去噪）振动数据
        data_denoised: 去噪后的振动数据
        fs: 采样频率
        nfft: FFT大小（可选，默认使用Config.NFFT）
    
    返回：
        (fig_original, ax_original, fig_denoised, ax_denoised): matplotlib 图表对象对
    """
    if nfft is None:
        nfft = Config.NFFT
    
    def compute_psd_matrix(data):
        """计算PSD矩阵的内部函数"""
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
        time_axis = np.array(time_points) - time_points[0]
        
        return freqs, psd_matrix, time_axis
    
    # 计算原始数据和去噪数据的PSD矩阵
    freqs_orig, psd_matrix_orig, time_axis_orig = compute_psd_matrix(data_original)
    freqs_denoise, psd_matrix_denoise, time_axis_denoise = compute_psd_matrix(data_denoised)
    
    # 对PSD矩阵进行高斯平滑处理
    psd_matrix_orig_smoothed = gaussian_filter(psd_matrix_orig, sigma=Config.SMOOTHING_SIGMA)
    psd_matrix_denoise_smoothed = gaussian_filter(psd_matrix_denoise, sigma=Config.SMOOTHING_SIGMA)
    
    # 筛选频率范围（全频谱）
    freq_mask_orig = (freqs_orig >= Config.FREQ_BAND_MIN) & (freqs_orig <= Config.FREQ_BAND_MAX)
    freqs_orig_filtered = freqs_orig[freq_mask_orig]
    psd_matrix_orig_filtered = psd_matrix_orig_smoothed[:, freq_mask_orig]
    
    freq_mask_denoise = (freqs_denoise >= Config.FREQ_BAND_MIN) & (freqs_denoise <= Config.FREQ_BAND_MAX)
    freqs_denoise_filtered = freqs_denoise[freq_mask_denoise]
    psd_matrix_denoise_filtered = psd_matrix_denoise_smoothed[:, freq_mask_denoise]
    
    def create_3d_plot(time_axis, freqs_filtered, psd_matrix_filtered, title_prefix):
        """创建3D表面图的内部函数"""
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
    
    # 创建两个3D图
    fig_orig, ax_orig = create_3d_plot(time_axis_orig, freqs_orig_filtered, psd_matrix_orig_filtered, "去噪前")
    fig_denoise, ax_denoise = create_3d_plot(time_axis_denoise, freqs_denoise_filtered, psd_matrix_denoise_filtered, "去噪后")
    
    return fig_orig, ax_orig, fig_denoise, ax_denoise


def main():
    """
    VIV 频域对比绘制主函数
    为每个样本生成两张图：去噪前和去噪后的全频谱PSD 3D可视化
    """
    print("="*80)
    print("VIV 频域对比绘制（去噪前 vs 去噪后）")
    print("="*80)
    
    print("\n[步骤1] 获取 VIV 窗口数据...")
    windows = get_viv_windows()
    
    print("\n[步骤2] 生成对比绘图...")
    ploter = PlotLib()
    
    for i, window_info in enumerate(windows[:Config.NUM_SAMPLES_TO_PLOT], 1):
        data_original = window_info['data']
        
        print(f"  ✓ 样本 {i}: 正在进行小波去噪预处理...")
        data_denoised, denoise_info = preprocess_data_with_wavelet_denoise(data_original)
        
        sensor_id = window_info['sensor_id']
        time_str = window_info['time']
        window_idx = window_info['window_index']
        
        # 生成对比图
        print(f"  ✓ 样本 {i}: 生成去噪前后对比频域图表...")
        fig_orig, ax_orig, fig_denoise, ax_denoise = plot_3d_vibration_psd_full_spectrum(
            data_original.reshape(-1, 1),
            data_denoised.reshape(-1, 1)
        )
        
        # 设置去噪前图表标题
        fig_orig.suptitle(
            f"[去噪前] {sensor_id} @ {time_str} (窗口 {window_idx})\n"
            f"频率范围: {Config.FREQ_BAND_MIN}~{Config.FREQ_BAND_MAX}Hz | NFFT: {Config.NFFT}",
            fontproperties=CN_FONT, fontsize=FONT_SIZE
        )
        ploter.figs.append(fig_orig)
        
        # 设置去噪后图表标题
        fig_denoise.suptitle(
            f"[去噪后] {sensor_id} @ {time_str} (窗口 {window_idx})\n"
            f"频率范围: {Config.FREQ_BAND_MIN}~{Config.FREQ_BAND_MAX}Hz | NFFT: {Config.NFFT}",
            fontproperties=CN_FONT, fontsize=FONT_SIZE
        )
        ploter.figs.append(fig_denoise)
        
        print(f"  ✓ 样本 {i} 已生成 2 个对比图表（共 {len(ploter.figs)} 个）")
    
    print("\n" + "="*80)
    print(f"✓ 成功生成 {len(ploter.figs)} 个图表（{Config.NUM_SAMPLES_TO_PLOT} 个样本 × 2个对比图）")
    print("="*80 + "\n")
    
    ploter.show()
    return ploter


if __name__ == '__main__':
    main()
