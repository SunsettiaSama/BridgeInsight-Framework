from ..visualize_tools.utils import ChartApp, PlotLib
from ..data_processer.io_unpacker import UNPACK, DataManager
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 抑制无关警告

# ======================== 全局配置（独立抽离）========================
# 基础显示配置（保持原有设置，统一管理）
plt.style.use('default')
FONT_SIZE = 12
CMAP = plt.cm.gist_yarg  # 颜色映射
VALID_WIND_SPEED_THRESHOLD = 0.1  # 有效风速阈值
WIND_SPEED_BINS = 20  # 风速分箱数量
NORMALIZE = False  # 不做归一化

# 字体配置（统一封装）
def init_font_config():
    """初始化全局字体配置，解决中文/负号显示问题"""
    # 全局默认字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = FONT_SIZE
    
    # 中英文字体对象
    eng_font = FontProperties(family='Times New Roman', size=FONT_SIZE)
    cn_font = FontProperties(
        family='SimHei', 
        size=FONT_SIZE,
        # Linux/Mac需手动指定SimHei路径，示例：
        # fname='/usr/share/fonts/truetype/simhei/SimHei.ttf'
    )
    return eng_font, cn_font

# 初始化字体
ENG_FONT, CN_FONT = init_font_config()

# ======================== 数据获取逻辑（完全分离）========================
class WindDataFetcher:
    """风速数据获取类，封装所有数据读取/清洗逻辑"""
    def __init__(self):
        self.unpacker = UNPACK()
        self.data_manager = DataManager()

    def get_viv_wind_velocity(self, viv_excel_path, sensor_id, time_interval=1, fs=1):
        """
        从VIV Excel文件获取指定传感器的有效风速数据
        :param viv_excel_path: VIV数据表格路径
        :param sensor_id: 目标传感器ID
        :param time_interval: 截取VIV发生时段的时长（分钟）
        :param fs: 采样频率（Hz）
        :return: 有效风速数组（仅保留>0.1m/s的数据）
        """
        # 读取VIV表格
        try:
            df_viv = pd.read_excel(viv_excel_path)
        except Exception as e:
            print(f"读取VIV Excel失败：{e}")
            return np.array([])
        
        # 存储有效风速
        valid_wind_vel = np.array([])
        
        # 遍历VIV记录匹配风数据
        for row in df_viv.values:
            viv_path, viv_time, plane = row
            # 匹配对应风传感器路径
            wind_paths = self.unpacker.VIC_Path_2_WindPath(
                VICpath=viv_path, 
                wind_sensor_ids=[sensor_id]
            )
            
            if len(wind_paths) != 1:
                continue  # 仅处理匹配到唯一路径的情况
            
            # 解析风数据（仅保留风速）
            wind_velocity, _, _ = self.unpacker.Wind_Data_Unpack(wind_paths[0])
            wind_velocity = np.array(wind_velocity)
            
            # 截取指定时段数据（防止索引越界）
            start_idx = viv_time
            end_idx = viv_time + time_interval * 60 * fs
            end_idx = min(end_idx, len(wind_velocity))
            if start_idx >= end_idx:
                continue
            
            # 过滤无效风速
            vel_valid = np.mean(wind_velocity[start_idx:end_idx])

            valid_wind_vel = np.hstack((valid_wind_vel, vel_valid))
        
        return valid_wind_vel

    def get_folder_wind_velocity(self, wind_dir, sensor_id):
        """
        从文件夹读取指定传感器的有效风速数据
        :param wind_dir: 风数据根目录
        :param sensor_id: 目标传感器名称（如"跨中桥面上游"）
        :return: 有效风速数组（仅保留>0.1m/s的数据）
        """
        # 读取文件夹中风数据
        try:
            wind_velocities, _, _ = self.data_manager.get_wind_data_from_root(
                wind_dir, 
                mode="interval", 
                sensor_id=sensor_id, 
                bin_nums = 60, 
            )
        except Exception as e:
            print(f"读取文件夹风数据失败：{e}")
            return np.array([])
        
        # 转换为数组并过滤无效风速
        wind_velocities = np.array(wind_velocities)
        mask = wind_velocities > VALID_WIND_SPEED_THRESHOLD
        valid_wind_vel = wind_velocities[mask]
        
        return valid_wind_vel

# ======================== 画图逻辑（完全分离）========================
def plot_wind_speed_histogram(wind_vel_data, sensor_name, bins=WIND_SPEED_BINS):
    """
    绘制风速直方图（核心画图逻辑，独立封装）
    :param wind_vel_data: 风速数据数组
    :param sensor_name: 传感器名称（用于标题）
    :param bins: 风速分箱数量
    :return: 绘制好的figure对象（无数据时返回None）
    """
    # 空数据处理
    if len(wind_vel_data) == 0:
        print(f"警告：{sensor_name} 无有效风速数据，跳过画图")
        return None
    
    # 创建画布（非极坐标）
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 风速分箱计算
    bins_edges = np.linspace(0, np.max(wind_vel_data), bins + 1)
    counts, bin_edges = np.histogram(wind_vel_data, bins=bins_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 区间中点（用于颜色映射）
    
    # 绘制直方图
    bar_width = np.diff(bins_edges)[0]
    bars = ax.bar(
        bin_centers, counts, 
        width=bar_width, alpha=0.8, edgecolor='black'
    )
    
    # 颜色映射（保持原有逻辑）
    norm = Normalize(min(bin_centers), max(bin_centers))
    for bar, center in zip(bars, bin_centers):
        bar.set_facecolor(CMAP(norm(center)))
    
    # 添加颜色条
    sm = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='风速 (m/s)', pad=0.1)
    cbar.set_label('风速 (m/s)', fontproperties=CN_FONT)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    
    # 坐标轴/标题配置（保持原有显示风格）
    ax.set_xlabel('风速 (m/s)', fontproperties=CN_FONT)
    ax.set_ylabel('样本数量（粒度：分钟）', fontproperties=CN_FONT)
    ax.tick_params(axis='x', labelsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    
    # 优化布局（防止标签重叠）
    plt.tight_layout()
    
    return fig

# ======================== 主函数（协调数据获取+画图）========================
def Wind_Speed_Histogram():
    """主函数：整合数据获取和画图逻辑，展示最终结果"""
    # 1. 初始化工具
    data_fetcher = WindDataFetcher()
    ploter = PlotLib()
    figs = []  # 存储所有生成的图表
    
    # 2. 配置参数（可集中修改）
    # VIV数据相关
    VIV_EXCEL_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\Img\VIV.xlsx'
    target_sensors = [
        # 'ST-UAN-T01-003-01',  # 北索塔塔顶
        'ST-UAN-G04-001-01'     # 跨中桥面上游
    ]
    sensor_names = [
        # 'VIV Wind Distribution (Top Of North Pylon)',
        'VIV Wind Distribution (Upstream of Mid-Span)'
    ]
    
    # 文件夹数据相关
    WIND_DIRS = [r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September"]
    FOLDER_SENSOR_IDS = ["跨中桥面上游"]
    
    # 3. 处理VIV数据并画图
    for idx, sensor_id in enumerate(target_sensors):
        # 获取VIV风速数据
        wind_vel_data = data_fetcher.get_viv_wind_velocity(
            viv_excel_path=VIV_EXCEL_PATH,
            sensor_id=sensor_id
        )
        
        # 绘制直方图
        fig = plot_wind_speed_histogram(
            wind_vel_data=wind_vel_data,
            sensor_name=sensor_names[idx]
        )
        
        if fig:
            figs.append(fig)
    
    # 4. 处理文件夹数据并画图
    for wind_dir in WIND_DIRS:
        for sensor_id in FOLDER_SENSOR_IDS:
            # 获取文件夹风速数据
            wind_vel_data = data_fetcher.get_folder_wind_velocity(
                wind_dir=wind_dir,
                sensor_id=sensor_id
            )
            
            # 绘制直方图（传感器名称简化）
            fig = plot_wind_speed_histogram(
                wind_vel_data=wind_vel_data,
                sensor_name=f"Wind Distribution ({sensor_id})"
            )
            
            if fig:
                figs.append(fig)
    
    # 5. 展示所有图表
    ploter.figs.extend(figs)
    plt.close('all')  # 释放内存
    ploter.show()

# 执行主函数（如需外部调用，可注释此行）
if __name__ == "__main__":
    Wind_Speed_Histogram()