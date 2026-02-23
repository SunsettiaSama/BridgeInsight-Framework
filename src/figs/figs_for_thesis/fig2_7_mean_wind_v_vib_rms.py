"""
图2.7: 平均风速与振动RMS的关系分析
绘制平均风速（横坐标）与振动加速度RMS（纵坐标）的散点图
显示面内和面外两个方向的传感器数据

研究目的：
- 对比同一根拉索在面内和面外两个方向的风-振动响应特性
- 分析风速与不同方向振动的相关性
- 揭示拉索运动的方向特异性

选定拉索：
1. 北塔边跨1/4跨上游拉索：
   - 面内传感器 (In-plane): ST-VIC-C18-101-01
   - 面外传感器 (Out-of-plane): ST-VIC-C18-101-02

2. 北塔跨中1/4跨上游拉索：
   - 面内传感器 (In-plane): ST-VIC-C18-201-01
   - 面外传感器 (Out-of-plane): ST-VIC-C18-201-02

数据处理流程：
1. 从工作流获取极端振动对应的风数据和振动文件路径
2. 使用滑动窗口计算振动RMS
3. 使用滑动窗口计算风速平均值
4. 绘制散点图，显示面内/面外两种方向的数据对比
"""

# --------------- 模块导入 ---------------
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据处理工作流
from src.data_processer.statistics.wind_data_io_process.workflow import run as run_wind_workflow
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.io_unpacker import UNPACK

# 导入可视化工具
from src.visualize_tools.utils import PlotLib

# 导入传感器配置
from src.config.sensor_config import (
    WIND_SENSOR_NAMES,
    WIND_FS,
    WIND_VALID_THRESHOLD,
    VIB_TO_WIND_SENSOR_MAP,
)

# 导入绘图配置
from .config import ENG_FONT, CN_FONT, FONT_SIZE, REC_FIG_SIZE

# --------------- 全局配置 ---------------
plt.style.use('default')
plt.rcParams['font.size'] = FONT_SIZE

# 选定的传感器组合
# 选择同一根拉索的面内和面外传感器对，对比不同方向的振动特性
# 北塔边跨1/4跨上游：ST-VIC-C18-101-01（面内）vs ST-VIC-C18-101-02（面外）
# 北塔跨中1/4跨上游：ST-VIC-C18-201-01（面内）vs ST-VIC-C18-201-02（面外）
# 参考配置：src.config.sensor_config.VIBRATION_SENSOR_MAP
SELECTED_VIB_SENSORS = {
    'edge_span_inplane': 'ST-VIC-C18-101-01',      # 北索塔边跨1/4跨 面内上游
    'edge_span_outplane': 'ST-VIC-C18-101-02',     # 北索塔边跨1/4跨 面外上游
    'mid_span_inplane': 'ST-VIC-C18-201-01',       # 北索塔跨中1/4跨 面内上游
    'mid_span_outplane': 'ST-VIC-C18-201-02',      # 北索塔跨中1/4跨 面外上游
}

# 对应的风传感器
SELECTED_WIND_SENSOR = 'ST-UAN-G04-001-01'  # 跨中桥面上游

# 采样频率配置
VIB_FS = 50  # 振动信号采样频率 (Hz)
WIND_FS_CONFIG = 1  # 风速采样频率 (Hz)

# 时间窗口配置
VIB_TIME_WINDOW = 60.0  # 计算振动RMS的时间窗口（秒）
WIND_TIME_WINDOW = 60.0  # 计算风速平均值的时间窗口（秒）

# 计算窗口大小
VIB_WINDOW_SIZE = int(VIB_FS * VIB_TIME_WINDOW)  # 振动窗口大小（采样点）
WIND_WINDOW_SIZE = int(WIND_FS_CONFIG * WIND_TIME_WINDOW)  # 风速窗口大小（采样点）


# --------------- 数据计算函数 ---------------
def process_vibration_file(file_path, sensor_id, window_size=VIB_WINDOW_SIZE):
    """
    处理单个振动文件，计算RMS时间序列
    
    参数:
        file_path: 文件路径
        sensor_id: 传感器ID
        window_size: 滑动窗口大小（采样点）
    
    返回:
        rms_list: RMS值列表
    """
    try:
        unpacker = UNPACK(init_path=False)
        vibration_data = unpacker.VIC_DATA_Unpack(file_path)
        vibration_data = np.array(vibration_data)
        
        if len(vibration_data) == 0:
            return []
        
        rms_list = []
        
        if len(vibration_data) >= window_size:
            for i in range(0, len(vibration_data) - window_size + 1, window_size):
                window_data = vibration_data[i:i+window_size]
                rms_val = np.sqrt(np.mean(np.square(window_data)))
                if rms_val > 0:
                    rms_list.append(rms_val)
        else:
            rms_val = np.sqrt(np.mean(np.square(vibration_data)))
            if rms_val > 0:
                rms_list.append(rms_val)
        
        return rms_list
    
    except Exception as e:
        return []


def process_wind_file(file_path, sensor_id, window_size=WIND_WINDOW_SIZE):
    """
    处理单个风速文件，计算平均风速时间序列
    
    参数:
        file_path: 文件路径
        sensor_id: 传感器ID
        window_size: 滑动窗口大小（采样点）
    
    返回:
        mean_wind_speeds: 平均风速值列表
    """
    try:
        unpacker = UNPACK(init_path=False)
        wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(file_path)
        wind_velocity = np.array(wind_velocity)
        
        if len(wind_velocity) == 0:
            return []
        
        mean_wind_speeds = []
        
        if len(wind_velocity) >= window_size:
            for i in range(0, len(wind_velocity) - window_size + 1, window_size):
                window_vel = wind_velocity[i:i+window_size]
                # 过滤无效风速
                valid_mask = window_vel > WIND_VALID_THRESHOLD
                window_vel_valid = window_vel[valid_mask]
                
                if len(window_vel_valid) > 0:
                    mean_vel = np.mean(window_vel_valid)
                    mean_wind_speeds.append(mean_vel)
        else:
            valid_mask = wind_velocity > WIND_VALID_THRESHOLD
            wind_velocity_valid = wind_velocity[valid_mask]
            if len(wind_velocity_valid) > 0:
                mean_vel = np.mean(wind_velocity_valid)
                mean_wind_speeds.append(mean_vel)
        
        return mean_wind_speeds
    
    except Exception as e:
        return []


def load_vibration_rms_by_sensor(vib_metadata, sensor_id, use_multiprocess=True):
    """
    从振动文件中加载并计算指定传感器的RMS数据
    
    参数:
        vib_metadata: 振动元数据列表
        sensor_id: 传感器ID
        use_multiprocess: 是否使用多进程
    
    返回:
        rms_values: RMS数据列表
    """
    # 筛选该传感器的文件
    sensor_files = [item for item in vib_metadata if item.get('sensor_id') == sensor_id]
    
    if len(sensor_files) == 0:
        return np.array([])
    
    # 提取文件路径
    file_paths = [item.get('file_path') for item in sensor_files if item.get('file_path')]
    
    all_rms_values = []
    
    if use_multiprocess:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_vibration_file, fp, sensor_id): fp 
                      for fp in file_paths}
            
            for future in tqdm(as_completed(futures), 
                             total=len(file_paths),
                             desc=f"处理振动文件 {sensor_id}"):
                try:
                    rms_list = future.result()
                    if len(rms_list) > 0:
                        all_rms_values.extend(rms_list)
                except Exception as e:
                    pass
    else:
        for file_path in tqdm(file_paths, desc=f"处理振动文件 {sensor_id}"):
            try:
                rms_list = process_vibration_file(file_path, sensor_id)
                if len(rms_list) > 0:
                    all_rms_values.extend(rms_list)
            except Exception as e:
                pass
    
    return np.array(all_rms_values)


def load_wind_speed_by_sensor(wind_metadata, sensor_id, use_multiprocess=True):
    """
    从风速文件中加载并计算指定传感器的平均风速数据
    
    参数:
        wind_metadata: 风数据元数据列表
        sensor_id: 风传感器ID
        use_multiprocess: 是否使用多进程
    
    返回:
        wind_speeds: 平均风速数据列表
    """
    # 筛选该传感器的文件
    sensor_files = [item for item in wind_metadata if item.get('sensor_id') == sensor_id]
    
    if len(sensor_files) == 0:
        return np.array([])
    
    # 提取文件路径
    file_paths = [item.get('file_path') for item in sensor_files if item.get('file_path')]
    
    all_wind_speeds = []
    
    if use_multiprocess:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_wind_file, fp, sensor_id): fp 
                      for fp in file_paths}
            
            for future in tqdm(as_completed(futures), 
                             total=len(file_paths),
                             desc=f"处理风速文件 {sensor_id}"):
                try:
                    wind_list = future.result()
                    if len(wind_list) > 0:
                        all_wind_speeds.extend(wind_list)
                except Exception as e:
                    pass
    else:
        for file_path in tqdm(file_paths, desc=f"处理风速文件 {sensor_id}"):
            try:
                wind_list = process_wind_file(file_path, sensor_id)
                if len(wind_list) > 0:
                    all_wind_speeds.extend(wind_list)
            except Exception as e:
                pass
    
    return np.array(all_wind_speeds)


def plot_wind_vs_vibration(wind_speeds, vib_inplane, vib_outplane, 
                           location_name='', output_path=None):
    """
    绘制风速与振动RMS的关系图
    
    参数:
        wind_speeds: 风速数据
        vib_inplane: 面内振动RMS
        vib_outplane: 面外振动RMS
        location_name: 位置名称（用于标题）
        output_path: 输出文件路径（可选）
    
    返回:
        fig, ax: matplotlib figure和axis对象
    """
    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)
    
    # 确保数据长度一致
    min_len = min(len(wind_speeds), len(vib_inplane), len(vib_outplane))
    wind_speeds = wind_speeds[:min_len]
    vib_inplane = vib_inplane[:min_len]
    vib_outplane = vib_outplane[:min_len]
    
    # 绘制散点图
    ax.scatter(wind_speeds, vib_inplane, 
              color='#1f77b4', s=80, alpha=0.6, 
              edgecolors='#1f77b4', linewidth=1.5,
              label='面内方向（In-plane）', marker='o')
    
    ax.scatter(wind_speeds, vib_outplane, 
              color='#ff7f0e', s=80, alpha=0.6, 
              edgecolors='#ff7f0e', linewidth=1.5,
              label='面外方向（Out-of-plane）', marker='s')
    
    # 设置标签
    ax.set_xlabel('平均风速 (Mean Wind Speed) [m/s]', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('加速度RMS (Acceleration RMS) [m/s²]', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    # 设置标题
    title = f'风速与振动RMS关系 - {location_name}\n(Wind Speed vs Vibration RMS - {location_name})'
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=20)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置图例
    ax.legend(loc='upper left', fontsize=FONT_SIZE-2, framealpha=0.95,
             prop=CN_FONT if location_name else ENG_FONT)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（可选）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存至: {output_path}")
    
    return fig, ax


# --------------- 主函数 ---------------
def main():
    """
    主函数：绘制平均风速与振动RMS的关系图
    """
    print("="*80)
    print(" "*15 + "图2.7: 平均风速与振动RMS的关系分析")
    print("="*80)
    
    # ============================================================
    # Step 1: 运行振动数据工作流（获取振动元数据）
    # ============================================================
    print("\n[Step 1] 运行振动数据工作流...")
    print("-"*80)
    vib_metadata = run_vib_workflow(use_cache=True)
    print(f"✓ 获取振动数据元数据: {len(vib_metadata)} 条")
    
    # ============================================================
    # Step 2: 运行风数据工作流（筛选极端振动对应的风数据）
    # ============================================================
    print("\n[Step 2] 运行风数据工作流（筛选极端振动时段）...")
    print("-"*80)
    wind_metadata = run_wind_workflow(
        vib_metadata=vib_metadata,
        use_cache=True,
        force_recompute=False,
        extreme_only=True
    )
    print(f"✓ 获取极端振动对应的风数据元数据: {len(wind_metadata)} 条")
    
    # ============================================================
    # Step 3: 加载风速数据
    # ============================================================
    print("\n[Step 3] 加载和计算风速数据...")
    print("-"*80)
    print(f"风传感器: {SELECTED_WIND_SENSOR} ({WIND_SENSOR_NAMES.get(SELECTED_WIND_SENSOR, '未知')})")
    print(f"时间窗口: {WIND_TIME_WINDOW}秒, 窗口大小: {WIND_WINDOW_SIZE}个采样点")
    
    wind_speeds = load_wind_speed_by_sensor(wind_metadata, SELECTED_WIND_SENSOR, use_multiprocess=True)
    print(f"✓ 计算得到 {len(wind_speeds)} 个风速样本")
    if len(wind_speeds) > 0:
        print(f"  风速范围: {np.min(wind_speeds):.2f} ~ {np.max(wind_speeds):.2f} m/s")
        print(f"  平均风速: {np.mean(wind_speeds):.2f} m/s")
    
    # 初始化 PlotLib
    ploter = PlotLib()
    
    # ============================================================
    # Step 4: 绘制边跨拉索数据（面内vs面外）
    # ============================================================
    print("\n[Step 4] 计算北塔边跨拉索RMS数据（In-plane vs Out-of-plane）...")
    print("-"*80)
    
    print(f"面内传感器: {SELECTED_VIB_SENSORS['edge_span_inplane']}")
    print(f"时间窗口: {VIB_TIME_WINDOW}秒, 窗口大小: {VIB_WINDOW_SIZE}个采样点")
    edge_span_inplane = load_vibration_rms_by_sensor(vib_metadata, SELECTED_VIB_SENSORS['edge_span_inplane'], use_multiprocess=True)
    print(f"✓ 计算得到 {len(edge_span_inplane)} 个RMS样本")
    if len(edge_span_inplane) > 0:
        print(f"  RMS范围: {np.min(edge_span_inplane):.4f} ~ {np.max(edge_span_inplane):.4f} m/s²")
    
    print(f"\n面外传感器: {SELECTED_VIB_SENSORS['edge_span_outplane']}")
    edge_span_outplane = load_vibration_rms_by_sensor(vib_metadata, SELECTED_VIB_SENSORS['edge_span_outplane'], use_multiprocess=True)
    print(f"✓ 计算得到 {len(edge_span_outplane)} 个RMS样本")
    if len(edge_span_outplane) > 0:
        print(f"  RMS范围: {np.min(edge_span_outplane):.4f} ~ {np.max(edge_span_outplane):.4f} m/s²")
    
    fig1, ax1 = plot_wind_vs_vibration(
        wind_speeds, edge_span_inplane, edge_span_outplane,
        location_name='北索塔边跨1/4跨（North Tower Edge-span）'
    )
    ploter.figs.append(fig1)
    print(f"✓ 边跨拉索图表绘制完成")
    
    # ============================================================
    # Step 5: 绘制跨中拉索数据（面内vs面外）
    # ============================================================
    print("\n[Step 5] 计算北塔跨中拉索RMS数据（In-plane vs Out-of-plane）...")
    print("-"*80)
    
    print(f"面内传感器: {SELECTED_VIB_SENSORS['mid_span_inplane']}")
    mid_span_inplane = load_vibration_rms_by_sensor(vib_metadata, SELECTED_VIB_SENSORS['mid_span_inplane'], use_multiprocess=True)
    print(f"✓ 计算得到 {len(mid_span_inplane)} 个RMS样本")
    if len(mid_span_inplane) > 0:
        print(f"  RMS范围: {np.min(mid_span_inplane):.4f} ~ {np.max(mid_span_inplane):.4f} m/s²")
    
    print(f"\n面外传感器: {SELECTED_VIB_SENSORS['mid_span_outplane']}")
    mid_span_outplane = load_vibration_rms_by_sensor(vib_metadata, SELECTED_VIB_SENSORS['mid_span_outplane'], use_multiprocess=True)
    print(f"✓ 计算得到 {len(mid_span_outplane)} 个RMS样本")
    if len(mid_span_outplane) > 0:
        print(f"  RMS范围: {np.min(mid_span_outplane):.4f} ~ {np.max(mid_span_outplane):.4f} m/s²")
    
    fig2, ax2 = plot_wind_vs_vibration(
        wind_speeds, mid_span_inplane, mid_span_outplane,
        location_name='北索塔跨中1/4跨（North Tower Mid-span）'
    )
    ploter.figs.append(fig2)
    print(f"✓ 跨中拉索图表绘制完成")
    
    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "="*80)
    print(" "*20 + "所有图表绘制完成")
    print("="*80)
    print(f"✓ 共生成 {len(ploter.figs)} 个图表")
    print(f"✓ 风传感器: {SELECTED_WIND_SENSOR} ({WIND_SENSOR_NAMES.get(SELECTED_WIND_SENSOR, '未知')})")
    print(f"✓ 选定拉索:")
    print(f"  1. 北塔边跨1/4跨上游: 面内({SELECTED_VIB_SENSORS['edge_span_inplane']}) vs 面外({SELECTED_VIB_SENSORS['edge_span_outplane']})")
    print(f"  2. 北塔跨中1/4跨上游: 面内({SELECTED_VIB_SENSORS['mid_span_inplane']}) vs 面外({SELECTED_VIB_SENSORS['mid_span_outplane']})")
    print(f"✓ 时间窗口配置: 振动{VIB_TIME_WINDOW}秒, 风速{WIND_TIME_WINDOW}秒")
    print(f"✓ 采样频率: 振动{VIB_FS}Hz, 风速{WIND_FS_CONFIG}Hz")
    print(f"✓ 使用 PlotLib 展示图表...")
    
    # 使用 PlotLib 展示所有图表
    plt.close('all')
    ploter.show()


if __name__ == "__main__":
    main()
