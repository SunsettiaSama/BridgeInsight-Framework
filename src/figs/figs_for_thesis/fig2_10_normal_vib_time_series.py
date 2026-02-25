import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.visualize_tools.utils import PlotLib
from .config import ENG_FONT, CN_FONT, FONT_SIZE, REC_FIG_SIZE


# ==================== 常量配置 ====================
class Config:
    # 数据路径
    NORMAL_SAMPLE_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\01\ST-VIC-C18-101-01_010000.VIC"
    VIV_SAMPLE_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\11\ST-VIC-C18-102-01_130000.VIC"
    
    # 绘图配置
    FIG_SIZE = REC_FIG_SIZE
    WAVEFORM_COLOR = '#333333'
    LINEWIDTH = 1.0
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'
    
    # 数据切分配置
    INTERVAL = (45, 46)
    SECOND_INTERVAL = (10, 15)
    
    # 采样频率
    FS = 50.0
    
    # 数据解包参数
    TIME_INTERVAL = 1


# ==================== 绘图函数 ====================
def plot_time_domain_waveform(data, fs=Config.FS):
    """
    绘制时域波形
    
    参数：
        data: 振动数据数组
        fs: 采样频率
    
    返回：
        (fig, ax): matplotlib 图表对象
    """
    data_segment = data[Config.INTERVAL[0]: Config.INTERVAL[1]]
    continuous_data = data_segment.flatten()
    
    start_sample = int(Config.SECOND_INTERVAL[0] * fs)
    end_sample = int(Config.SECOND_INTERVAL[1] * fs)
    
    final_data = continuous_data[start_sample:end_sample]
    
    if len(final_data) == 0:
        print("警告：根据当前的分钟和秒级切分设置，没有数据被选中。请检查 INTERVAL 和 SECOND_INTERVAL 的值。")
        return plt.figure(), None

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)
    
    time_axis = np.arange(len(final_data)) / fs
    
    ax.plot(
        time_axis, 
        final_data, 
        color=Config.WAVEFORM_COLOR,
        linewidth=Config.LINEWIDTH
    )
    
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT)
    
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
    
    return fig, ax


def main():
    """
    绘图主函数：加载数据并生成图表
    """
    unpacker = UNPACK(init_path=False)
    ploter = PlotLib()
    
    data = unpacker.File_Detach_Data(Config.NORMAL_SAMPLE_PATH, time_interval=Config.TIME_INTERVAL)
    fig, ax = plot_time_domain_waveform(data)
    if fig:
        ploter.figs.append(fig)

    data = unpacker.File_Detach_Data(Config.VIV_SAMPLE_PATH, time_interval=Config.TIME_INTERVAL)
    fig, ax = plot_time_domain_waveform(data)
    if fig:
        ploter.figs.append(fig)
    
    ploter.show()
