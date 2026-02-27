import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.ticker as ticker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.visualize_tools.utils import PlotLib
from .config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_blue_color_map,
    get_viridis_color_map
)


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
    
    # 绘图配置 - 从 config.py 继承
    FIG_SIZE = SQUARE_FIG_SIZE
    START_MAP_INDEX = 1
    END_MAP_INDEX = 4

    # 3D绘图视角
    ELEV = 30
    AZIM = -150
    
    # 网格线配置
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'


# ==================== 数据获取函数 ====================
def get_normal_vib_windows():
    """
    从 RVSample/annotation_results.json 中读取元数据，
    筛选出标注为 Normal_Vib (annotation=="0") 的窗口数据
    
    返回：
        list: 极端窗口数据列表，每项为包含窗口数据和元数据的字典
    """
    annotation_file = os.path.join(
        project_root, 
        "results", 
        "figs", 
        "figs_for_thesis", 
        "RVSample", 
        "annotation_results.json"
    )
    
    if not os.path.exists(annotation_file):
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
    all_windows = []
    
    print("\n[加载数据] 正在加载极端窗口数据...")
    for i, record in enumerate(normal_records):
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


# ==================== 绘图函数 ====================
def plot_3d_vibration_psd(data, fs=Config.FS):
    """
    绘制3D振动PSD可视化
    
    参数：
        data: 一维或二维振动数据
        fs: 采样频率
    
    返回：
        (fig, ax): matplotlib 图表对象
    """
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
    # nfft代表分辨率，它的中间值才是当前算出来的值
    nfft_half = int(Config.NFFT / 2)
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
            nperseg=int(Config.NFFT / 2), 
            noverlap=int(Config.NFFT / 4), 
            nfft=Config.NFFT
        )
        
        if freqs is None:
            freqs = f
        
        psd_matrix.append(p)
        time_points.append(center_idx / fs)
        
        # 按步长STEP_DURATION加算移动窗口
        center_idx += samples_per_step
    
    psd_matrix = np.array(psd_matrix)
    
    # 时间轴基于采样点位置，单位为秒，归零化从0开始
    time_axis = np.array(time_points) - time_points[0]
    
    T, F = np.meshgrid(time_axis, freqs)
    
    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = get_blue_color_map(style = 'gradient', start_map_index = Config.START_MAP_INDEX, end_map_index = Config.END_MAP_INDEX)
    
    surf = ax.plot_surface(
        T, F, psd_matrix.T,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
        shade=True
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
    Normal_Vib 时域和频域3D绘制主函数
    """
    print("="*80)
    print("Normal_Vib 时域和频域3D绘制")
    print("="*80)
    
    print("\n[步骤1] 获取 Normal_Vib 窗口数据...")
    windows = get_normal_vib_windows()
    
    print("\n[步骤2] 生成绘图...")
    ploter = PlotLib()
    
    for i, window_info in enumerate(windows[:5], 1):
        data = window_info['data']
        data_2d = data.reshape(-1, 1)
        
        fig, ax = plot_3d_vibration_psd(data_2d)
        
        sensor_id = window_info['sensor_id']
        time_str = window_info['time']
        window_idx = window_info['window_index']
        
        fig.suptitle(f"{sensor_id} @ {time_str} (窗口 {window_idx})", 
                    fontproperties=CN_FONT, fontsize=FONT_SIZE)
        
        ploter.figs.append(fig)
        print(f"  ✓ 已生成图表 {i}")
    
    print("\n" + "="*80)
    print(f"✓ 成功生成 {len(ploter.figs)} 个图表")
    print("="*80 + "\n")
    
    ploter.show()
    return ploter


if __name__ == '__main__':
    main()
