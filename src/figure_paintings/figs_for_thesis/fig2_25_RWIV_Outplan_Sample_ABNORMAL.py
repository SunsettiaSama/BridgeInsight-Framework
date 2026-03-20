import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, Optional, Tuple

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualize_tools.utils import PlotLib, ChartApp
from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.config import (
    get_viridis_color_map, 
    ENG_FONT, 
    CN_FONT, 
    FONT_SIZE, 
    LABEL_FONT_SIZE,
    REC_FIG_SIZE,
    SQUARE_FIG_SIZE,
    REC_FONT_SIZE,
    SQUARE_FONT_SIZE
)

# ==================== 常量配置 ====================
# 数据文件路径占位符
DATA_FILE_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\09\16\ST-VIC-C34-101-02_140000.VIC"

# 图形大小配置：使用矩形配置（适合论文插图）
FIG_SIZE = REC_FIG_SIZE  # (12, 8)

# 绘图参数
NFFT = 2048
FS = 50.0
WINDOW_SIZE = 3000
WINDOW_INDEX = 36  # min

# 从数据文件路径中提取元数据，生成标题
def _extract_metadata_from_path(file_path: str):
    """
    从数据文件路径中提取元数据
    路径格式: .../data/YYYY{Month}/{Location}/VIC/{MM}/{DD}/{SensorID}_{HHMMSS}.VIC
    
    Returns:
        (sensor_id, date_str, time_str) 元组
    """
    try:
        path_parts = file_path.replace('\\', '/').split('/')
        filename = path_parts[-1]  # "ST-VIC-C18-101-01_120000.VIC"
        
        # 提取传感器ID和时间戳
        filename_without_ext = os.path.splitext(filename)[0]  # "ST-VIC-C18-101-01_120000"
        sensor_time_parts = filename_without_ext.rsplit('_', 1)
        
        if len(sensor_time_parts) == 2:
            sensor_id = sensor_time_parts[0]  # "ST-VIC-C18-101-01"
            time_stamp = sensor_time_parts[1]  # "120000"
        else:
            sensor_id = filename_without_ext
            time_stamp = "000000"
        
        # 提取月和日
        if len(path_parts) >= 3:
            month = path_parts[-3]  # "09"
            day = path_parts[-2]    # "16"
        else:
            month = "01"
            day = "01"
        
        # 格式化时间戳: "120000" -> "12:00:00"
        if len(time_stamp) >= 6:
            time_str = f"{time_stamp[0:2]}:{time_stamp[2:4]}:{time_stamp[4:6]}"
        else:
            time_str = "00:00:00"
        
        date_str = f"{month}/{day}"
        
        return sensor_id, date_str, time_str
    except:
        return "Unknown", "01/01", "00:00:00"



# 生成标题（对齐annotation.py的格式）
_sensor_id, _date_str, _time_str = _extract_metadata_from_path(DATA_FILE_PATH)
FIG_TITLE = f"{_sensor_id} @ {_date_str} {_time_str} (Window {WINDOW_INDEX})"


# ==================== 图像生成器 ====================
class VibrationFigureGenerator:
    """单个振动数据窗口的图像生成器"""
    
    def __init__(self, fs: float = 50.0):
        self.fs = fs
    
    def generate_figure(self, window_data: np.ndarray, 
                       sensor_id: str = "ST-VIC-C18-101-01",
                       time_str: str = "01/01 00:00",
                       window_idx: int = 0) -> Tuple[Optional[plt.Figure], Optional[str]]:
        """
        生成单个窗口的图像：左侧时域+频域，右侧频域变化
        
        Args:
            window_data: 振动数据数组
            sensor_id: 传感器ID
            time_str: 时间字符串
            window_idx: 窗口索引
            
        Returns:
            (fig, error_msg) 元组
        """
        try:
            if len(window_data) == 0:
                return None, "窗口数据为空"
            
            fig = plt.figure(figsize=FIG_SIZE)
            
            fig.suptitle(FIG_TITLE, fontproperties=ENG_FONT, fontsize=REC_FONT_SIZE, 
                        fontweight='bold', y=0.98)
            
            # 调整gridspec：增加垂直间距(hspace)，留出底部空间(bottom)显示x轴标签
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], wspace=0.3, 
                                 hspace=0.4, top=0.93, bottom=0.12)
            ax_time = fig.add_subplot(gs[0, 0])
            ax_freq = fig.add_subplot(gs[1, 0])
            ax_freq_change = fig.add_subplot(gs[:, 1])
            
            self._plot_time_domain(ax_time, window_data, sensor_id, time_str, window_idx)
            self._plot_frequency_domain(ax_freq, window_data, sensor_id, time_str, window_idx)
            self._plot_frequency_evolution(ax_freq_change, window_data, sensor_id, time_str, window_idx)
            
            return fig, None
            
        except Exception as e:
            return None, f"生成图像失败: {str(e)}"
    
    def _plot_time_domain(self, ax, data: np.ndarray, sensor_id: str, 
                          time_str: str, window_idx: int):
        """绘制时域波形图"""
        time_axis = np.arange(len(data)) / self.fs
        
        ax.plot(
            time_axis, 
            data, 
            color='#333333',
            linewidth=1.0
        )

        ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=REC_FONT_SIZE)
        ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=REC_FONT_SIZE)
        
        ax.grid(True, color='gray', alpha=0.4, linewidth=0.5, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=REC_FONT_SIZE)
    
    def _plot_frequency_domain(self, ax, data: np.ndarray, sensor_id: str, 
                               time_str: str, window_idx: int):
        """绘制频域谱图"""
        f, psd = signal.welch(data, fs=self.fs, nperseg=int(NFFT/2), 
                              noverlap=int(NFFT/4), nfft=NFFT)
        
        freq_limit = 25
        mask = f <= freq_limit
        f_limited = f[mask]
        psd_limited = psd[mask]
        
        ax.plot(
            f_limited,
            psd_limited,
            color='#333333',
            linewidth=1.0
        )
        
        ax.set_xlabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=REC_FONT_SIZE)
        ax.set_ylabel('PSD $(m/s^2)^2/Hz$', labelpad=10, fontproperties=ENG_FONT, fontsize=REC_FONT_SIZE)
        
        ax.grid(True, color='gray', alpha=0.4, linewidth=0.5, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=REC_FONT_SIZE)
        
        ax.set_xlim(0, freq_limit)
    
    def _plot_frequency_evolution(self, ax, data: np.ndarray, sensor_id: str, 
                                  time_str: str, window_idx: int):
        """绘制频域随时间变化的频带堆积图"""
        freq_limit = 25
        fs_int = int(self.fs)
        window_total = int(len(data) // fs_int)
        
        psd_list = []
        time_labels = []
        
        for i in range(window_total):
            start_idx = i * fs_int
            end_idx = (i + 1) * fs_int
            
            if end_idx <= len(data):
                segment = data[start_idx:end_idx]
                f, psd = signal.welch(segment, fs=self.fs, nperseg=int(self.fs * 0.8),
                                     noverlap=int(self.fs * 0.4), nfft=NFFT)
                
                mask = f <= freq_limit
                psd_limited = psd[mask]
                psd_list.append(psd_limited)
                time_labels.append(f"{i}s")
        
        if not psd_list:
            return
        
        spec_array = np.array(psd_list)
        f_limited = f[f <= freq_limit]
        
        cmap_gray = get_viridis_color_map(start_gray=0.2)

        im = ax.imshow(spec_array, aspect='auto', origin='lower', cmap=cmap_gray, 
                       extent=[0, freq_limit, 0, window_total],
                       interpolation='bilinear')
        
        ax.set_xlabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=REC_FONT_SIZE)
        ax.set_ylabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=REC_FONT_SIZE)
        
        ax.set_yticks(np.arange(0, window_total + 1, max(1, window_total // 10)))
        ax.tick_params(axis='both', which='major', labelsize=REC_FONT_SIZE)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PSD $(m/s^2)^2/Hz$', fontproperties=ENG_FONT, fontsize=REC_FONT_SIZE)
        cbar.ax.tick_params(labelsize=REC_FONT_SIZE)


# ==================== 主程序 ====================
def main():
    """生成并显示图像"""
    
    if not DATA_FILE_PATH or DATA_FILE_PATH == "":
        raise ValueError("请先设置 DATA_FILE_PATH 常量")
    
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(f"数据文件不存在: {DATA_FILE_PATH}")
    
    print(f"[步骤1] 加载数据文件: {DATA_FILE_PATH}")
    unpacker = UNPACK(init_path=False)
    vibration_data = unpacker.VIC_DATA_Unpack(DATA_FILE_PATH)
    vibration_data = np.array(vibration_data)
    print(f"✓ 加载完成，总采样点数: {len(vibration_data)}")
    
    print(f"[步骤2] 提取窗口数据 (窗口索引: {WINDOW_INDEX})")
    start_sample = WINDOW_INDEX * WINDOW_SIZE
    end_sample = (WINDOW_INDEX + 1) * WINDOW_SIZE
    
    if end_sample > len(vibration_data):
        raise ValueError(f"窗口范围超出数据长度: [{start_sample}, {end_sample}] 超过 {len(vibration_data)}")
    
    window_data = vibration_data[start_sample:end_sample]
    print(f"✓ 窗口数据提取完成，长度: {len(window_data)} 个采样点")
    
    print(f"[步骤3] 生成图像")
    generator = VibrationFigureGenerator(fs=FS)
    fig, error_msg = generator.generate_figure(
        window_data, 
        sensor_id="ST-VIC-C18-101-01",
        time_str="01/01 00:00",
        window_idx=WINDOW_INDEX
    )
    
    if fig is None:
        raise RuntimeError(f"图像生成失败: {error_msg}")
    
    print(f"✓ 图像生成成功")
    
    print(f"[步骤4] 使用GUI显示图像")
    plotlib = PlotLib()
    plotlib.figs.append(fig)
    plotlib.show()


