# Autor@ 猫毛

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src.visualize_tools.utils import PlotLib


plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['font.size'] = 22

class Fig5:

    
    def __init__(self):
        self.normal_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\01\ST-VIC-C18-101-01_010000.VIC"
        self.VIV_sample_path = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\11\ST-VIC-C18-102-01_130000.VIC"
        self.ploter = PlotLib()
        return 
    
    def plot_3d_vibration_psd(self, data, start_time = None, end_time = None, fs=50.0):
        """
        生成三维PSD面图（分钟级频谱）- 优化视觉效果（柔和颜色+无网格线）
        
        参数:
        data (List[List]): 60分钟数据列表，每分钟3000点
        fs (float): 采样频率 (Hz)
        title (str): 图表标题
        
        返回:
        fig: Matplotlib figure对象
        ax: 3D Axes对象
        """
        if start_time == None:
            start_time = 0
        if end_time == None: 
            end_time = len(data)

        data = data[start_time: end_time]
        # 1. 按分钟分割数据
        minutes = len(data)
        
        # 2. 为每分钟计算PSD
        freqs = None
        psd_matrix = []

        for i, minute_data in enumerate(data):
            minute_data = np.array(minute_data, dtype=np.float32)
            f, p = signal.welch(
                minute_data,
                fs=fs,
                nperseg=128,
                noverlap=64,
                nfft=256
            )
            
            if i == 0:
                freqs = f  # 保存频率轴
            
            psd_matrix.append(p)
        
        psd_matrix = np.array(psd_matrix)  # (60, len(freqs))
        
        # 3. 创建时间-频率网格
        time_axis = np.arange(minutes)
        T, F = np.meshgrid(time_axis, freqs)
        
        # 4. 创建3D图形
        fig = plt.figure(figsize=(12, 8))  # 适当增大画布尺寸
        ax = fig.add_subplot(111, projection='3d')
        
        # 5. 绘制优化的曲面（关键：使用柔和的viridis颜色映射 + 无网格线）
        surf = ax.plot_surface(
            T, F, psd_matrix.T,
            cmap='viridis',  # 使用柔和的渐变色（蓝-绿-黄）
            linewidth=0,
            antialiased=True,
            alpha=0.9,
            shade=True
        )
        
        # 6. 添加颜色条
        cbar = fig.colorbar(surf, pad=0.1)
        # ========== 修改颜色条标签（标注对数单位） ==========
        cbar.set_label('PSD', rotation=270, labelpad=20, fontsize=20)
        cbar.ax.tick_params(labelsize=20)

        # 7. 设置坐标轴标签
        ax.set_xlabel('时间 (Minutes)', labelpad=10, fontsize=20)
        ax.set_ylabel('频率 (Hz)', labelpad=10, fontsize=20)
        ax.set_zlabel('PSD', labelpad=16, fontsize=20)
        
        # 8. 移除所有网格线（关键修改）
        ax.grid(False)  # 移除背景网格线

        # 9. 优化视角和标签
        ax.view_init(elev=30, azim=-150)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        return fig, ax

    def plot(self, start_time = None, end_time = None):
        from src.utils import UNPACK
        unpacker = UNPACK(init_path = False)

        data = unpacker.File_Detach_Data(self.normal_sample_path, time_interval = 1)
        fig, ax = self.plot_3d_vibration_psd(data, start_time, end_time)
        self.ploter.figs.append(fig)

        data = unpacker.File_Detach_Data(self.VIV_sample_path, time_interval = 1)
        fig, ax = self.plot_3d_vibration_psd(data, start_time, end_time)

        self.ploter.figs.append(fig)
        self.ploter.show()
        return 

if __name__ == '__main__':
    fig5 = Fig5()
    fig5.plot()