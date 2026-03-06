"""
图2.6: 风数据紊流度风玫瑰图（极端振动时段） - 改进版
基于风数据工作流接口，绘制风速和紊流度的极坐标统计图

功能特点：
1. 仅使用极端振动对应的风荷载数据（通过extreme_only=True模式）
2. 根据extreme_time_ranges时间窗口截取原始风数据
3. 针对每个极端时间窗口计算统计量，而非处理全部原始数据点
4. 计算内容：平均风速、平均风向（圆形统计）、紊流度
5. 支持多进程并行处理数据
6. 每个风速传感器独立绘图
7. 使用 PlotLib 统一管理图表
8. 颜色映射统一为0-100%范围

改进的数据处理流程：
- Step 1: 获取振动数据元数据（包含极端振动索引）
- Step 2: 运行风数据工作流，筛选极端振动对应的风数据
- Step 3: 根据extreme_time_ranges截取原始风数据的极端时间窗口
- Step 4: 对每个时间窗口计算：
    * 平均风速（有效数据点的平均）
    * 平均风向（使用向量平均法的圆形统计）
    * 紊流度（该窗口数据的风速标准差/平均值）
- Step 5: 按平均风向进行角度分箱
- Step 6: 计算每个分箱内的平均紊流度
- Step 7: 绘制风玫瑰图
"""

# --------------- 模块导入 ---------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据处理工作流
from src.data_processer.preprocess.wind_data_io_process.workflow import run as run_wind_workflow
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.io_unpacker import UNPACK

# 导入可视化工具
from src.visualize_tools.utils import PlotLib

# 导入绘图配置
from .config import ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_full_color_map

# 导入传感器配置（从 sensor_config 导入，不是从绘图配置导入）
from src.config.sensor_config import (
    WIND_SENSOR_NAMES,
    WIND_DIR_CORRECTION,
    AXIS_OF_BRIDGE,
    WIND_FS,
    WIND_VALID_THRESHOLD,
)

# --------------- 全局绘图配置 ---------------
plt.style.use('default')
plt.rcParams['font.size'] = FONT_SIZE

# 颜色映射
CMAP = plt.cm.viridis
MIN_CMAP = 0
MAX_CMAP = 50

# 图片大小
FIG_SIZE = SQUARE_FIG_SIZE

# 多进程
USE_MULTIPROCESS = True


# --------------- 工具函数 ---------------
def calculate_turbulence_intensity(wind_speed_group):
    """
    计算紊流度：TI = (风速样本标准差 / 平均风速) × 100%
    
    计算说明：
    1. 使用样本标准差（ddof=1）符合工程样本统计规范
    2. 返回 np.nan 表示无效数据，便于区分「无效TI」和「有效低TI」
    3. 不对紊流度进行上限截断，保留原始计算结果
    
    参数:
        wind_speed_group: 某一分箱内的风速样本（np.array）
    
    返回:
        ti_value: 紊流度值（百分比），异常情况返回 np.nan
    """
    # 1. 样本量不足，返回 nan
    if len(wind_speed_group) < 2:
        return np.nan
    
    # 2. 计算平均风速
    u_mean = np.mean(wind_speed_group)
    
    # 3. 避免除以零，返回 nan
    if u_mean <= 1e-6:
        return np.nan
    
    # 4. 计算样本标准差（ddof=1）
    u_std = np.std(wind_speed_group, ddof=1)
    
    # 5. 计算紊流度（不截断）
    ti_value = (u_std / u_mean) * 100  # 转换为百分比

    # 6. 保留两位小数，返回计算值
    return round(ti_value, 2)


def correct_wind_direction(wind_directions, correction_val):
    """
    风向修正：统一修正逻辑，归一化到0-360度
    
    参数:
        wind_directions: 原始风向数据（np.array）
        correction_val: 修正值（如360、180）
    
    返回:
        修正后的风向数据
    """
    wind_directions = correction_val - wind_directions
    wind_directions = np.mod(wind_directions, 360)
    return wind_directions


def plot_wind_rose(theta, counts, ti_values, axis_of_bridge, bin_step, cmap, title=None):
    """
    绘制风玫瑰图（紊流度颜色映射，count归一化表示占比）
    
    参数:
        theta: 分箱起始角度（弧度）
        counts: 每个分箱的原始样本数
        ti_values: 每个分箱的紊流度（用于颜色映射）
        axis_of_bridge: 桥轴线角度（度）
        bin_step: 分箱步长（度）
        cmap: 颜色映射
        title: 图表标题（可选）
    
    返回:
        fig, ax: 绘图对象
    """
    # 创建极坐标图
    fig, ax = plt.subplots(figsize=FIG_SIZE, subplot_kw={'projection': 'polar'})
    
    # count归一化，转为样本占比百分比
    counts = np.array(counts)
    total_count = counts.sum()
    if total_count > 0:
        counts_normalized = (counts / total_count) * 100
    else:
        counts_normalized = counts
    
    # 绘制柱状图
    bars = ax.bar(
        theta, counts_normalized,
        width=np.deg2rad(bin_step),
        bottom=0.0,
        alpha=0.85,
        align='edge'
    )
    
    # 设置极坐标轴方向（0度朝北，顺时针）
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # 紊流度颜色映射（统一范围：0-100%，确保不同传感器可比）
    norm = plt.Normalize(MIN_CMAP, MAX_CMAP)
    
    for bar, ti in zip(bars, ti_values):
        bar.set_facecolor(cmap(norm(ti)))
    
    # 颜色条配置
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax,
        orientation='vertical',
        label='紊流度 (%)',
        pad=0.08,
        shrink=0.85
    )
    cbar.set_label('紊流度 (%)', fontproperties=CN_FONT)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    
    # 地理坐标标签
    x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticklabels(x_ticks, fontproperties=ENG_FONT)
    
    # 桥轴线标注
    y_max = np.max(counts_normalized) if len(counts_normalized) > 0 else 1
    bridge_theta1 = np.deg2rad(axis_of_bridge)
    bridge_theta2 = np.deg2rad(axis_of_bridge + 180)
    ax.plot([bridge_theta1, bridge_theta1], [0, y_max * 1.1], 
            color='red', linestyle='--', linewidth=2, label='Bridge Axis')
    ax.plot([bridge_theta2, bridge_theta2], [0, y_max * 1.1], 
            color='red', linestyle='--', linewidth=2)
    
    # y轴刻度配置（百分比）
    y_tick_interval = max(2, round(y_max / 5))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
    ax.set_ylim(0, y_max * 1.1)
    ax.set_yticklabels([f"{int(num)}%" for num in ax.get_yticks()], 
                       fontproperties=ENG_FONT)
    ax.set_rlabel_position(270)  # 左侧显示刻度
    
    # 设置标题
    if title:
        ax.set_title(title, fontproperties=CN_FONT, pad=20)
    
    plt.tight_layout()
    return fig, ax


# --------------- 多进程数据加载函数 ---------------
def extract_extreme_time_windows(wind_velocity, wind_direction, extreme_time_ranges, fs=WIND_FS):
    """
    根据极端时间窗口截取风数据，每个窗口独立处理
    
    参数:
        wind_velocity: 完整的风速数组
        wind_direction: 完整的风向数组
        extreme_time_ranges: 极端时间窗口列表 [(start_sec, end_sec), ...]
        fs: 采样频率（Hz），默认为配置的风数据采样频率
    
    返回:
        (extracted_velocities, extracted_directions): 每个窗口独立处理后的风速和风向数组
    """
    if len(extreme_time_ranges) == 0:
        return np.array([]), np.array([])
    
    wind_velocity = np.array(wind_velocity)
    wind_direction = np.array(wind_direction)
    
    extracted_velocities = []
    extracted_directions = []
    
    for start_sec, end_sec in extreme_time_ranges:
        # 计算对应的采样点索引（1Hz采样，索引即为秒数）
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(wind_velocity), end_idx)
        
        if start_idx < end_idx:
            # 截取该时间窗口的数据
            window_vel = wind_velocity[start_idx:end_idx]
            window_dir = wind_direction[start_idx:end_idx]
            
            # 数据清洗：过滤无效风速
            valid_mask = window_vel > WIND_VALID_THRESHOLD
            window_vel_valid = window_vel[valid_mask]
            window_dir_valid = window_dir[valid_mask]
            
            # 只有当窗口内有有效数据时才添加
            if len(window_vel_valid) > 0:
                extracted_velocities.append(window_vel_valid)
                extracted_directions.append(window_dir_valid)
    
    # 如果没有任何有效窗口，返回空数组
    if len(extracted_velocities) == 0:
        return np.array([]), np.array([])
    
    # 使用hstack合并所有窗口的数据（保持每个窗口独立处理）
    return np.hstack(extracted_velocities), np.hstack(extracted_directions)


def process_single_wind_file(file_path, extreme_time_ranges=None, valid_threshold=WIND_VALID_THRESHOLD):
    """
    单文件风数据加载工作函数，用于多进程
    
    参数:
        file_path: 文件路径
        extreme_time_ranges: 极端时间窗口列表 [(start_sec, end_sec), ...]，如果为None则使用全部数据
        valid_threshold: 风速有效阈值（m/s），从配置导入
    
    返回:
        (wind_velocities, wind_directions): 风速和风向数组，失败返回 (None, None)
    """
    try:
        from src.data_processer.io_unpacker import UNPACK
        unpacker = UNPACK(init_path=False)
        
        # 解析风数据
        wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(file_path)
        
        # 转换为numpy数组
        wind_velocity = np.array(wind_velocity)
        wind_direction = np.array(wind_direction)
        
        # 如果提供了极端时间窗口，进行时间截取
        if extreme_time_ranges is not None and len(extreme_time_ranges) > 0:
            wind_velocity, wind_direction = extract_extreme_time_windows(
                wind_velocity, wind_direction, extreme_time_ranges
            )
        else:
            # 如果没有提供极端时间窗口，使用全部数据并过滤无效风速
            if len(wind_velocity) > 0:
                valid_mask = wind_velocity > valid_threshold
                wind_velocity = wind_velocity[valid_mask]
                wind_direction = wind_direction[valid_mask]
        
        # 返回有效数据
        if len(wind_velocity) > 0:
            return (wind_velocity, wind_direction)
        else:
            return (None, None)
    
    except Exception as e:
        return (None, None)


def compute_window_statistics(wind_velocity, wind_direction, extreme_time_ranges, fs=WIND_FS):
    """
    计算每个极端时间窗口的统计量（平均风速、平均风向、紊流度）
    
    参数:
        wind_velocity: 完整的风速数组
        wind_direction: 完整的风向数组
        extreme_time_ranges: 极端时间窗口列表 [(start_sec, end_sec), ...]
        fs: 采样频率（Hz），默认为配置的风数据采样频率
    
    返回:
        (mean_velocities, mean_directions, ti_values): 每个窗口的平均风速、平均风向、紊流度列表
    """
    if len(extreme_time_ranges) == 0:
        return [], [], []
    
    wind_velocity = np.array(wind_velocity)
    wind_direction = np.array(wind_direction)
    
    mean_velocities = []
    mean_directions = []
    ti_values = []
    
    for start_sec, end_sec in extreme_time_ranges:
        # 计算对应的采样点索引
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(wind_velocity), end_idx)
        
        if start_idx < end_idx:
            # 截取该时间窗口的数据
            window_vel = wind_velocity[start_idx:end_idx]
            window_dir = wind_direction[start_idx:end_idx]
            
            # 数据清洗：过滤无效风速
            valid_mask = window_vel > WIND_VALID_THRESHOLD
            window_vel_valid = window_vel[valid_mask]
            window_dir_valid = window_dir[valid_mask]
            
            # 只有当窗口内有有效数据时才计算统计量
            if len(window_vel_valid) > 0:
                # 计算平均风速
                mean_vel = np.mean(window_vel_valid)
                
                # 计算平均风向（需要考虑圆形统计）
                # 使用向量平均法计算平均风向
                angles_rad = np.deg2rad(window_dir_valid)
                sin_sum = np.sum(np.sin(angles_rad))
                cos_sum = np.sum(np.cos(angles_rad))
                mean_dir = np.rad2deg(np.arctan2(sin_sum, cos_sum))
                mean_dir = mean_dir % 360  # 确保在0-360范围内
                
                # 计算紊流度
                ti = calculate_turbulence_intensity(window_vel_valid)
                
                mean_velocities.append(mean_vel)
                mean_directions.append(mean_dir)
                ti_values.append(ti)
    
    return mean_velocities, mean_directions, ti_values


def process_single_wind_file_statistics(file_path, extreme_time_ranges=None, valid_threshold=WIND_VALID_THRESHOLD):
    """
    单文件风数据统计处理工作函数，用于多进程
    计算每个极端时间窗口的统计量而非返回原始数据
    
    参数:
        file_path: 文件路径
        extreme_time_ranges: 极端时间窗口列表 [(start_sec, end_sec), ...]，如果为None则返回空
        valid_threshold: 风速有效阈值（m/s），从配置导入
    
    返回:
        (mean_velocities, mean_directions, ti_values): 平均风速、平均风向、紊流度列表，失败返回 ([], [], [])
    """
    try:
        from src.data_processer.io_unpacker import UNPACK
        unpacker = UNPACK(init_path=False)
        
        # 解析风数据
        wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(file_path)
        
        # 转换为numpy数组
        wind_velocity = np.array(wind_velocity)
        wind_direction = np.array(wind_direction)
        
        # 计算时间窗口统计量
        if extreme_time_ranges is not None and len(extreme_time_ranges) > 0:
            mean_velocities, mean_directions, ti_values = compute_window_statistics(
                wind_velocity, wind_direction, extreme_time_ranges
            )
            return (mean_velocities, mean_directions, ti_values)
        else:
            return ([], [], [])
    
    except Exception as e:
        return ([], [], [])


def load_wind_data_by_sensor(wind_metadata, sensor_id, use_multiprocess=True, max_workers=None):
    """
    按传感器加载风数据统计量（支持多进程），仅加载极端振动时间窗口的数据
    改进的逻辑：
    1. 获取路径解析的风速数据文件
    2. 通过极端振动索引，截取时间窗口
    3. 求取这段时间内数据的平均值，包括风速风向的平均值、以及对应的该窗口的紊流度
    
    参数:
        wind_metadata: 风数据元数据列表（包含extreme_time_ranges信息）
        sensor_id: 目标传感器ID
        use_multiprocess: 是否使用多进程（默认 True）
        max_workers: 最大进程数（None表示自动）
    
    返回:
        (mean_velocities, mean_directions, ti_values): 该传感器的所有极端窗口的平均风速、平均风向、紊流度列表
    """
    # 筛选该传感器的文件
    sensor_files = [item for item in wind_metadata if item.get('sensor_id') == sensor_id]
    
    if len(sensor_files) == 0:
        return [], [], []
    
    # 提取文件路径和对应的极端时间窗口
    file_info_list = []
    for item in sensor_files:
        file_path = item.get('file_path')
        extreme_time_ranges = item.get('extreme_time_ranges', [])
        file_info_list.append((file_path, extreme_time_ranges))
    
    all_mean_velocities = []
    all_mean_directions = []
    all_ti_values = []
    
    if use_multiprocess:
        # 多进程并行加载和处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_wind_file_statistics, fp, etr): (fp, etr) 
                      for fp, etr in file_info_list}
            
            for future in tqdm(as_completed(futures), 
                             total=len(file_info_list),
                             desc=f"处理 {sensor_id} (极端窗口统计)"):
                try:
                    mean_vels, mean_dirs, tis = future.result()
                    if len(mean_vels) > 0:
                        all_mean_velocities.extend(mean_vels)
                        all_mean_directions.extend(mean_dirs)
                        all_ti_values.extend(tis)
                except Exception as e:
                    pass
    else:
        # 单进程顺序加载和处理
        for file_path, extreme_time_ranges in tqdm(file_info_list, desc=f"处理 {sensor_id} (极端窗口统计)"):
            try:
                mean_vels, mean_dirs, tis = process_single_wind_file_statistics(file_path, extreme_time_ranges)
                if len(mean_vels) > 0:
                    all_mean_velocities.extend(mean_vels)
                    all_mean_directions.extend(mean_dirs)
                    all_ti_values.extend(tis)
            except Exception as e:
                pass
    
    return all_mean_velocities, all_mean_directions, all_ti_values


def process_sensor_data(sensor_id, wind_metadata, interval_nums=36, 
                       use_multiprocess=True):
    """
    处理单个传感器的数据并绘图
    改进的逻辑：针对每个极端时间窗口计算统计量，然后按风向分箱绘制风玫瑰图
    
    参数:
        sensor_id: 传感器ID
        wind_metadata: 风数据元数据列表
        interval_nums: 角度分箱数量
        use_multiprocess: 是否使用多进程加载数据
    
    返回:
        fig: matplotlib figure对象，如果无数据返回 None
    """
    print(f"\n{'='*70}")
    print(f"处理传感器: {sensor_id}")
    print(f"传感器名称: {WIND_SENSOR_NAMES.get(sensor_id, '未知')}")
    print(f"{'='*70}")
    
    # 统计该传感器的极端时间窗口信息
    sensor_files = [item for item in wind_metadata if item.get('sensor_id') == sensor_id]
    total_windows = sum(len(item.get('extreme_time_ranges', [])) for item in sensor_files)
    print(f"极端数据文件数: {len(sensor_files)}")
    print(f"极端时间窗口数: {total_windows}")
    
    # 加载该传感器的数据（计算每个极端时间窗口的统计量）
    mean_velocities, mean_directions, ti_values = load_wind_data_by_sensor(
        wind_metadata, sensor_id, use_multiprocess=use_multiprocess
    )
    
    if len(mean_velocities) == 0:
        print(f"警告：传感器 {sensor_id} 无有效极端振动风数据，跳过")
        return None
    
    print(f"✓ 极端窗口统计完成")
    print(f"  极端窗口数: {len(mean_velocities)}")
    print(f"  风速范围: {min(mean_velocities):.2f} ~ {max(mean_velocities):.2f} m/s")
    print(f"  平均风速: {np.mean(mean_velocities):.2f} m/s")
    
    # 风向修正（应用于平均风向）
    correction_val = WIND_DIR_CORRECTION.get(sensor_id, 360)
    mean_directions = np.array(mean_directions)
    mean_directions = correct_wind_direction(mean_directions, correction_val)
    print(f"✓ 风向修正完成（修正值: {correction_val}度）")
    
    # 角度分箱（基于修正后的平均风向）
    bin_step = int(360 / interval_nums)
    bins = np.arange(0, 360 + bin_step, bin_step)
    digitized = np.digitize(mean_directions, bins)
    
    # 按分箱统计
    counts = []
    ti_values_binned = []
    
    for i in range(1, len(bins)):
        mask = digitized == i
        count = np.sum(mask)
        counts.append(count)
        
        # 计算该分箱内的平均紊流度
        if count > 0:
            binned_ti_values = [ti_values[j] for j in range(len(ti_values)) if mask[j] and ti_values[j] is not None]
            if len(binned_ti_values) > 0:
                # 过滤有效TI值（非nan）
                valid_ti_in_bin = [ti for ti in binned_ti_values if not np.isnan(ti)]
                if len(valid_ti_in_bin) > 0:
                    avg_ti = np.mean(valid_ti_in_bin)
                    ti_values_binned.append(avg_ti)
                else:
                    ti_values_binned.append(0.0)
            else:
                ti_values_binned.append(0.0)
        else:
            ti_values_binned.append(0.0)
    
    # 处理 nan 值用于可视化
    ti_values_for_plot = [0.0 if np.isnan(ti) else ti for ti in ti_values_binned]
    
    # 统计信息
    valid_ti = [ti for ti in ti_values_binned if not np.isnan(ti) and ti > 0]
    
    print(f"✓ 统计计算完成")
    print(f"  分箱数: {interval_nums}")
    print(f"  有效数据分箱数: {len([c for c in counts if c > 0])}/{interval_nums}")
    print(f"  有效紊流度分箱数: {len(valid_ti)}/{interval_nums}")
    if len(valid_ti) > 0:
        print(f"  平均紊流度: {np.mean(valid_ti):.2f}%")
        print(f"  紊流度范围: {np.min(valid_ti):.2f}% ~ {np.max(valid_ti):.2f}%")
        print(f"  紊流度中位数: {np.median(valid_ti):.2f}%")
    else:
        print(f"  警告: 无有效紊流度数据")
    
    # 绘制风玫瑰图
    theta = np.deg2rad(bins[:-1])
    sensor_name = WIND_SENSOR_NAMES.get(sensor_id, sensor_id)
    fig, ax = plot_wind_rose(
        theta=theta,
        counts=counts,
        ti_values=ti_values_for_plot,
        axis_of_bridge=AXIS_OF_BRIDGE,
        bin_step=bin_step,
        cmap=CMAP,
    )
    
    print(f"✓ 绘图完成")
    
    return fig


# --------------- 主函数 ---------------
def main():
    """
    主函数：调用风数据工作流，为每个传感器独立绘制风玫瑰图
    仅使用极端振动对应的风荷载数据
    """
    print("="*80)
    print(" "*20 + "图2.6: 风数据紊流度分析（极端振动时段）")
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
    
    # 统计极端时间窗口总数
    total_extreme_windows = sum(len(item.get('extreme_time_ranges', [])) for item in wind_metadata)
    print(f"✓ 极端时间窗口总数: {total_extreme_windows}")
    
    # 统计各传感器的文件数
    sensor_counts = defaultdict(int)
    for item in wind_metadata:
        sensor_counts[item.get('sensor_id')] += 1
    
    print(f"\n各传感器文件数统计：")
    for sensor_id, count in sorted(sensor_counts.items()):
        sensor_name = WIND_SENSOR_NAMES.get(sensor_id, sensor_id)
        print(f"  {sensor_name} ({sensor_id}): {count} 个文件")
    
    # ============================================================
    # Step 3: 为每个传感器独立处理数据并绘图
    # ============================================================
    print("\n[Step 3] 为每个传感器处理数据并绘图...")
    print("="*80)
    
    # 获取所有传感器ID
    sensor_ids = sorted(list(sensor_counts.keys()))
    
    # 初始化 PlotLib
    ploter = PlotLib()
    
    # 处理每个传感器
    for sensor_id in sensor_ids:
        fig = process_sensor_data(
            sensor_id=sensor_id,
            wind_metadata=wind_metadata,
            interval_nums=36,
            use_multiprocess=USE_MULTIPROCESS, 
        )
        
        if fig is not None:
            ploter.figs.append(fig)
    
    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "="*80)
    print(" "*25 + "所有绘图完成（极端振动时段）")
    print("="*80)
    print(f"✓ 共生成 {len(ploter.figs)} 个图表")
    print(f"✓ 数据来源：仅极端振动对应的风荷载数据（1Hz采样）")
    print(f"✓ 极端时间窗口总数: {total_extreme_windows}")
    print(f"✓ 使用 PlotLib 展示图表...")
    
    # 使用 PlotLib 展示所有图表
    plt.close('all')  # 释放内存
    ploter.show()


if __name__ == "__main__":
    main()
