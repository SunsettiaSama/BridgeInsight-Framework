# Autor@ 猫毛
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import signal
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker

# 导入项目内部模块 (改为绝对导入以支持直接运行)
from src.data_processer.io_unpacker import DataManager
from src.visualize_tools.utils import PlotLib
from src.utils import UNPACK
# 导入统一配置
from .config import (
    FONT_SIZE, 
    ENG_FONT, 
    CN_FONT, 
    LABEL_FONT_SIZE,
    FIG_SIZE,
    NFFT,
    DEFAULT_COLOR, 
)


# --- 时间切分区间配置 ---
# 定义要分析的分钟范围 (起始分钟, 结束分钟)，左闭右开
INTERVAL = (45, 46)
# 定义在上述分钟范围内，进一步切分的秒级范围 (起始秒, 结束秒)，左闭右开
SECOND_INTERVAL = (10, 15)

WINDOW_DURATION = 1 # min
STEP_DURATION = 1 # s

def create_custom_gray_cmap(start_gray=0.2):
    start_gray = np.clip(start_gray, 0.0, 1.0)
    original_cmap = plt.cm.gist_yarg
    colors = original_cmap(np.linspace(start_gray, 1.0, 256))
    return ListedColormap(colors)

CMAP = create_custom_gray_cmap(start_gray=0.2)

def plot_time_domain_waveform(data, fs=50.0):
    """
    绘制时域波形函数
    """
    # 1. 数据准备：先进行分钟级切分
    data_segment = data[INTERVAL[0]: INTERVAL[1]]
    # 将切分出的分钟数据压平成一个连续的时间序列
    continuous_data = data_segment.flatten()
    
    # 2. 进行秒级切分
    start_sample = int(SECOND_INTERVAL[0] * fs)
    end_sample = int(SECOND_INTERVAL[1] * fs)
    
    # 从压平后的连续数据中切分出最终需要分析的数据
    final_data = continuous_data[start_sample:end_sample]
    
    if len(final_data) == 0:
        print("警告：根据当前的分钟和秒级切分设置，没有数据被选中。请检查 INTERVAL 和 SECOND_INTERVAL 的值。")
        return None, None

    # 3. 创建图形
    fig = plt.figure(figsize = FIG_SIZE)
    ax = fig.add_subplot(111)
    
    # 4. 构建时间轴
    time_axis = np.arange(len(final_data)) / fs
    
    # 5. 绘制波形
    ax.plot(
        time_axis, 
        final_data, 
        color=DEFAULT_COLOR,  # 使用配置中的深灰色
        linewidth=1.0
    )
    
    # 6. 设置坐标轴和样式
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT)
    
    ax.grid(True)
    ax.tick_params(
        axis='both',
        which='major',
        grid_color='gray',
        grid_alpha=0.4,
        grid_linewidth=0.5,
        grid_linestyle='--',
        labelsize=FONT_SIZE
    )
    # 设置刻度字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ENG_FONT)
        label.set_size(FONT_SIZE)
        
    return fig, ax

def Vibration_Below_Threshold_Analysis():
    """
    主分析函数：替代原有的 Fig5.plot
    """
    normal_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\01\ST-VIC-C18-101-01_010000.VIC"
    VIV_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\11\ST-VIC-C18-102-01_130000.VIC"
    
    ploter = PlotLib()
    unpacker = UNPACK(init_path = False)

    # 处理正常样本
    print(f"正在处理样本: {normal_sample_path}")
    data_normal = unpacker.File_Detach_Data(normal_sample_path, time_interval = 1)
    fig1, _ = plot_time_domain_waveform(data_normal)
    if fig1: ploter.figs.append(fig1)

    # 处理VIV样本 (如果需要)
    print(f"正在处理样本: {VIV_sample_path}")
    data_viv = unpacker.File_Detach_Data(VIV_sample_path, time_interval = 1)
    fig2, _ = plot_time_domain_waveform(data_viv)
    if fig2: ploter.figs.append(fig2)
    
    ploter.show()

if __name__ == '__main__':
    Vibration_Below_Threshold_Analysis()
