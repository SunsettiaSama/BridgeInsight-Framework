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
font_size = 12
label_font_size = 20

plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.unicode_minus'] = False

ENG_FONT = FontProperties(family='Times New Roman', size=label_font_size)
CN_FONT = FontProperties(family='SimHei', size=label_font_size)
# fft分辨率
NFFT = 2048

INTERVAL = (45, 50)
WINDOW_DURATION = 1 # min
STEP_DURATION = 1 # s

def create_custom_gray_cmap(start_gray=0.2):
    start_gray = np.clip(start_gray, 0.0, 1.0)
    original_cmap = plt.cm.gist_yarg
    colors = original_cmap(np.linspace(start_gray, 1.0, 256))
    return ListedColormap(colors)

CMAP = create_custom_gray_cmap(start_gray=0.2)

class Fig6: 
    def __init__(self):
        self.normal_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\01\ST-VIC-C18-101-01_010000.VIC"
        self.VIV_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\11\ST-VIC-C18-102-01_130000.VIC"
        self.ploter = PlotLib()
        return 
    
    def plot_3d_vibration_psd(self, data, fs=50.0):
        data_segment = data[INTERVAL[0]: INTERVAL[1]]
        # 将截取的数据从 (分钟数, 每分钟点数) 重构为一个连续的一维时间序列
        continuous_data = data_segment.flatten()
        
        # 2. 定义滑动窗口参数
        window_duration = WINDOW_DURATION  # 窗口长度（分钟）
        step_duration = STEP_DURATION   # 滑动步长（秒）

        # 计算窗口和步长对应的样本点数
        samples_per_window = int(window_duration * 60 * fs)
        samples_per_step = int(step_duration * fs)

        # 3. 执行滑动窗口计算
        num_windows = (len(continuous_data) - samples_per_window) // samples_per_step + 1
        
        freqs = None
        psd_matrix = []
        for i in range(num_windows):
            # 计算当前窗口的起始和结束索引
            start_idx = i * samples_per_step
            end_idx = start_idx + samples_per_window
            window_data = continuous_data[start_idx:end_idx]
            
            # 对窗口内的数据执行Welch分析
            f, p = signal.welch(window_data, fs=fs, nperseg=int(NFFT / 2), noverlap=int(NFFT / 4), nfft=NFFT)
            
            if i == 0:
                freqs = f
            psd_matrix.append(p)
        
        psd_matrix = np.array(psd_matrix)
        
        # 4. 构建新的、加密后的时间轴
        # 时间轴的单位仍为“分钟”，但刻度更密集，精确到10秒
        # 第一个窗口的中心时间点是 INTERVAL[0] + 0.5 分钟
        # 后续每个窗口向后移动 step_duration/60 分钟
        time_axis = np.arange(num_windows) * (step_duration / 60.0) # INTERVAL[0] + 0.5 +
        
        T, F = np.meshgrid(time_axis, freqs)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(
            T, F, psd_matrix.T,
            cmap=CMAP,
            linewidth=0,
            antialiased=True,
            alpha=0.9,
            shade=True
        )
        
        cbar = fig.colorbar(surf, pad=0.1)
        cbar.set_label('PSD', rotation=270, labelpad=20, fontproperties=ENG_FONT)
        cbar.ax.tick_params(labelsize=label_font_size)

        # --- 修改点 2: 简化并统一设置网格线 ---
        # 开启所有轴的网格线
        ax.grid(True)
        # 使用 tick_params 一次性设置所有轴的网格线样式
        ax.tick_params(
            axis='both',
            which='major',
            grid_color='gray',
            grid_alpha=0.4,  # 建议降低透明度，避免网格线喧宾夺主
            grid_linewidth=0.5,
            grid_linestyle='--',
            labelsize=label_font_size
        )
        
        # 让坐标平面背景透明，视觉效果更好
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # 隐藏坐标平面的边框线
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.set_xlabel('时间 (分)', labelpad=10, fontproperties=CN_FONT)
        ax.set_ylabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT)
        ax.set_zlabel('PSD', labelpad=16, fontproperties=ENG_FONT)
        
        ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
        ax.view_init(elev=30, azim=-150)
        
        return fig, ax

    def plot(self):
        from src.utils import UNPACK
        unpacker = UNPACK(init_path = False)

        data = unpacker.File_Detach_Data(self.normal_sample_path, time_interval = 1)
        fig, ax = self.plot_3d_vibration_psd(data)
        self.ploter.figs.append(fig)

        data = unpacker.File_Detach_Data(self.VIV_sample_path, time_interval = 1)
        fig, ax = self.plot_3d_vibration_psd(data)

        self.ploter.figs.append(fig)
        self.ploter.show()
        return 







if __name__ == '__main__':
    fig6 = Fig6()
    fig6.plot()