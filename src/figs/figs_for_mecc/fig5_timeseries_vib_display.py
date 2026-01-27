# Autor@ 猫毛
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import signal
from ..data_processer.io_unpacker import DataManager
from ..visualize_tools.utils import PlotLib
from matplotlib.colors import ListedColormap
# --- 修改点 1: 导入 ticker 模块 ---
import matplotlib.ticker as ticker


# --------------- 全局绘图配置（统一设置，优化字体配置）---------------
plt.style.use('default')
font_size = 16
label_font_size = 20
FIG_SIZE = (12, 4)
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.unicode_minus'] = False

ENG_FONT = FontProperties(family='Times New Roman', size=label_font_size)
CN_FONT = FontProperties(family='SimHei', size=label_font_size)
# fft分辨率
NFFT = 2048

# --- 修改点 2: 新增秒级切分区间 ---
# 定义要分析的分钟范围 (起始分钟, 结束分钟)，左闭右开
INTERVAL = (45, 46)
# 定义在上述分钟范围内，进一步切分的秒级范围 (起始秒, 结束秒)，左闭右开
# 例如，(10, 25) 表示分析第45分钟的第10秒到第25秒的数据
SECOND_INTERVAL = (10, 15)

WINDOW_DURATION = 1 # min
STEP_DURATION = 1 # s

def create_custom_gray_cmap(start_gray=0.2):
    start_gray = np.clip(start_gray, 0.0, 1.0)
    original_cmap = plt.cm.gist_yarg
    colors = original_cmap(np.linspace(start_gray, 1.0, 256))
    return ListedColormap(colors)

CMAP = create_custom_gray_cmap(start_gray=0.2)

class Fig5: 
    def __init__(self):
        self.normal_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\01\ST-VIC-C18-101-01_010000.VIC"
        self.VIV_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\11\ST-VIC-C18-102-01_130000.VIC"
        self.ploter = PlotLib()
        return 
    
    # --- 修改点 3: 函数重构以支持秒级切分 ---
    def plot_time_domain_waveform(self, data, fs=50.0):
        # 1. 数据准备：先进行分钟级切分
        data_segment = data[INTERVAL[0]: INTERVAL[1]]
        # 将切分出的分钟数据压平成一个连续的时间序列
        # 例如，从第45分钟到第46分钟的所有数据
        continuous_data = data_segment.flatten()
        
        # 2. 进行秒级切分
        # 计算每个秒级切分范围对应的样本点数
        start_sample = int(SECOND_INTERVAL[0] * fs)
        end_sample = int(SECOND_INTERVAL[1] * fs)
        
        # 从压平后的连续数据中，根据样本点索引切分出最终需要分析的数据
        final_data = continuous_data[start_sample:end_sample]
        
        # 如果切分出的数据为空，则打印提示并返回
        if len(final_data) == 0:
            print("警告：根据当前的分钟和秒级切分设置，没有数据被选中。请检查 INTERVAL 和 SECOND_INTERVAL 的值。")
            return plt.figure(), None # 返回一个空的fig和ax以避免错误

        # 3. 创建2D图形
        fig = plt.figure(figsize = FIG_SIZE)
        ax = fig.add_subplot(111)
        
        # 4. 构建新的时间轴，从0开始
        # 时间轴的长度与最终切分出的数据长度一致
        time_axis = np.arange(len(final_data)) / fs
        
        # 5. 绘制单一的时域波形，使用灰度
        ax.plot(
            time_axis, 
            final_data, 
            color='#333333',  # 使用深灰色
            linewidth=1.0
        )
        
        # 6. 设置坐标轴和样式
        ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT)
        ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT)
        
        # 7. 显示配置（完全保留）
        ax.grid(True)
        ax.tick_params(
            axis='both',
            which='major',
            grid_color='gray',
            grid_alpha=0.4,
            grid_linewidth=0.5,
            grid_linestyle='--',
            labelsize=label_font_size
        )
        
        return fig, ax
    
    def plot(self):
        from src.utils import UNPACK
        unpacker = UNPACK(init_path = False)

        # --- 修改点 4: 调用新的绘图函数 ---
        data = unpacker.File_Detach_Data(self.normal_sample_path, time_interval = 1)
        fig, ax = self.plot_time_domain_waveform(data)
        if fig: self.ploter.figs.append(fig)

        data = unpacker.File_Detach_Data(self.VIV_sample_path, time_interval = 1)
        fig, ax = self.plot_time_domain_waveform(data)
        if fig: self.ploter.figs.append(fig)
        
        self.ploter.show()
        return

if __name__ == '__main__':
    fig5 = Fig5()
    fig5.plot()