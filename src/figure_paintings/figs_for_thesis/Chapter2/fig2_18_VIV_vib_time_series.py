import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.utils import PlotLib
from .config import ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE


# ==================== 常量配置 ====================
class Config:
    # 采样频率
    FS = 50.0
    
    # 极端窗口配置
    WINDOW_SIZE = 3000  # 60秒窗口 @ 50Hz
    WINDOW_DURATION_SECONDS = 60  # 时间窗口
    NUM_SAMPLES_TO_PLOT = 50  # 随机选取50个极端窗口
    
    # 数据截取配置
    TRIM_START_SECOND = 0  # 从第几秒开始截取（秒）
    TRIM_END_SECOND = 3  # 截取到第几秒（秒），设为None表示不截取
    
    # 绘图配置
    FIG_SIZE = SQUARE_FIG_SIZE
    WAVEFORM_COLOR = '#333333'
    LINEWIDTH = 1.0
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'
    
    # 小波去噪配置
    ENABLE_WAVELET_DENOISE = True   # 是否启用小波去噪
    WAVELET_TYPE = 'coif2'            # 小波基类型（coif2）
    WAVELET_LEVEL = 5              # 分解层数
    THRESHOLD_TYPE = 'soft'         # 阈值类型（软阈值）
    THRESHOLD_METHOD = 'sqtwolog'   # 阈值计算方法（平方根双对数法）


# ==================== 数据获取函数 ====================
def get_extreme_windows_from_metadata():
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
    all_extreme_windows = []
    
    print("\n[加载数据] 正在加载极端窗口数据...")
    for i, record in enumerate(viv_records):
        metadata = record['metadata']
        file_path = metadata['file_path']
        extreme_indices = metadata['extreme_rms_indices']
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
    
    print(f"✓ 成功加载 {len(all_extreme_windows)} 个极端窗口")
    
    return all_extreme_windows


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
def plot_extreme_window(window_info, fs=Config.FS):
    """
    绘制单个极端窗口的时域波形
    
    参数：
        window_info: 包含窗口数据和元数据的字典
        fs: 采样频率
    
    返回：
        (fig, ax): matplotlib 图表对象
    """
    data = window_info['data']
    
    if len(data) == 0:
        print("警告：窗口数据为空")
        return plt.figure(), None
    
    data_denoised, denoise_info = preprocess_data_with_wavelet_denoise(data)
    
    trim_start_idx = int(Config.TRIM_START_SECOND * fs)
    trim_end_idx = int(Config.TRIM_END_SECOND * fs) if Config.TRIM_END_SECOND is not None else len(data_denoised)
    data_plot = data_denoised[trim_start_idx:trim_end_idx]

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)
    
    time_axis = np.arange(len(data_plot)) / fs + Config.TRIM_START_SECOND
    
    ax.plot(
        time_axis, 
        data_plot, 
        color=Config.WAVEFORM_COLOR,
        linewidth=Config.LINEWIDTH
    )
    
    sensor_id = window_info['sensor_id']
    time_str = window_info['time']
    window_idx = window_info['window_index']
    
    title = f"{sensor_id} @ {time_str} (窗口 {window_idx})"
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, 
            linewidth=Config.GRID_LINEWIDTH, linestyle=Config.GRID_LINESTYLE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    return fig, ax


def main():
    """
    绘图主函数：通过 vibration_io_process 工作流获取极端窗口数据并绘制
    """
    try:
        print("="*80)
        print("极端振动窗口时域波形绘制")
        print("="*80)
        
        print("\n[步骤1] 获取元数据和极端窗口数据...")
        extreme_windows = get_extreme_windows_from_metadata()
        
        print("\n[步骤2] 生成绘图...")
        figs = []
        ploter = PlotLib()
        
        for i, window_info in enumerate(extreme_windows, 1):
            print(f"  ✓ 样本 {i}: 正在进行小波去噪预处理...")
            fig, ax = plot_extreme_window(window_info)
            if fig is not None and ax is not None:
                figs.append(fig)
                ploter.figs.append(fig)
                print(f"  ✓ 已生成图表 {len(figs)}")
                
                if len(figs) >= Config.NUM_SAMPLES_TO_PLOT:
                    print(f"  ✓ 已达到目标数量 {Config.NUM_SAMPLES_TO_PLOT}，停止生成")
                    break
        
        print("\n" + "="*80)
        print(f"✓ 成功生成 {len(figs)} 个图表")
        print("="*80 + "\n")
        
        ploter.show()
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        raise


if __name__ == "__main__":
    main()
