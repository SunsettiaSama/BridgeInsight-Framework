"""
图2.6: 高能量振动对应的越界风速时程曲线
基于极端振动样本对应的风速数据，找出其中风速超出[0, 70]范围的位置并绘制

功能特点：
1. 获取高能量振动对应的风速数据（通过extreme_only=True模式）
2. 找出其中风速越过[0, 70]范围的样本位置（风速<0或>70）
3. 绘制这些样本的风速时程曲线
4. 仅保留前5张图，超过5张则不保存
5. 支持多传感器独立处理
6. 使用 PlotLib 统一管理图表

越界范围：
- 下界: 风速 < 0 m/s
- 上界: 风速 > 70 m/s
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.statistics.wind_data_io_process.workflow import run as run_wind_workflow
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.io_unpacker import UNPACK

from src.visualize_tools.utils import PlotLib

from .config import ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE

from src.config.sensor_config import (
    WIND_SENSOR_NAMES,
    WIND_FS,
    WIND_VALID_THRESHOLD,
)

plt.style.use('default')
plt.rcParams['font.size'] = FONT_SIZE

FIG_SIZE = SQUARE_FIG_SIZE


def check_window_out_of_range(window_vel, min_vel=0, max_vel=70):
    """
    检查时间窗口内是否存在越界风速，返回完整窗口数据和越界掩码
    
    参数:
        window_vel: 窗口内的风速数据
        min_vel: 风速下界（m/s）
        max_vel: 风速上界（m/s）
    
    返回:
        (has_out_of_range, out_of_range_mask): 
        是否有越界数据（布尔值），越界掩码（布尔数组）
    """
    out_of_range_mask = (window_vel < min_vel) | (window_vel > max_vel)
    has_out_of_range = np.any(out_of_range_mask)
    return has_out_of_range, out_of_range_mask


def plot_window_with_out_of_range_highlight(window_vel, out_of_range_mask, sensor_id, 
                                             min_vel=0, max_vel=70, title=None, fs=WIND_FS):
    """
    绘制完整时间窗口的风速时程曲线，并高亮显示越界点
    
    参数:
        window_vel: 时间窗口内的完整风速数据
        out_of_range_mask: 布尔掩码，标记越界位置
        sensor_id: 传感器ID
        min_vel: 风速下界（m/s）
        max_vel: 风速上界（m/s）
        title: 图表标题（可选）
        fs: 采样频率（Hz）
    
    返回:
        fig, ax: matplotlib 图表对象
    """
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    
    # 计算时间轴（单位：s，风速采样频率为1Hz，所以每个数据点对应1秒）
    time_array = np.arange(len(window_vel), dtype=int)
    
    # 绘制完整的时程曲线（所有点）
    ax.plot(
        time_array,
        window_vel,
        color='#666666',
        linewidth=1.2,
        alpha=0.6
    )
    
    # 分别绘制正常点和越界点
    normal_mask = ~out_of_range_mask
    
    if np.any(normal_mask):
        ax.scatter(
            time_array[normal_mask],
            window_vel[normal_mask],
            color='#3399FF',
            s=20,
            alpha=0.5,
            label='有效测量点',
            zorder=2
        )
    
    if np.any(out_of_range_mask):
        ax.scatter(
            time_array[out_of_range_mask],
            window_vel[out_of_range_mask],
            color='#FF3333',
            s=60,
            alpha=0.9,
            marker='o',
            edgecolors='darkred',
            linewidth=1.5,
            label='越界点',
            zorder=3
        )
    
    # 添加范围边界线
    ax.axhline(y=min_vel, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label=f'下界 ({min_vel} m/s)')
    ax.axhline(y=max_vel, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label=f'上界 ({max_vel} m/s)')
    
    # 用阴影区域标示正常范围
    ax.axhspan(min_vel, max_vel, alpha=0.08, color='green')
    
    # 标题
    out_count = np.sum(out_of_range_mask)
    if title is None:
        title = f"{sensor_id} - 时间窗口风速分析 (越界: {out_count}/{len(window_vel)})"
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE - 2, pad=15)
    
    # 轴标签
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('风速 (m/s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    # 网格
    ax.grid(True, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    # 刻度设置（整数秒标记）
    if len(time_array) > 0:
        time_max = time_array[-1]
        if time_max > 0:
            # 计算合适的刻度间隔（保证显示整数秒）
            if time_max <= 10:
                tick_interval = 1
            elif time_max <= 60:
                tick_interval = max(1, time_max // 10)
            else:
                tick_interval = max(1, time_max // 10)
            ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
    
    # 添加图例（中文显示）
    legend = ax.legend(loc='best', fontsize=FONT_SIZE - 2, framealpha=0.9)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(CN_FONT)
    
    plt.tight_layout()
    return fig, ax


def load_wind_data_raw(file_path):
    """
    加载原始风数据
    
    参数:
        file_path: 风数据文件路径
    
    返回:
        (wind_velocity, wind_direction): 风速和风向数组，失败返回 (None, None)
    """
    try:
        unpacker = UNPACK(init_path=False)
        wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(file_path)
        return np.array(wind_velocity), np.array(wind_direction)
    except Exception as e:
        return None, None


def process_sensor_data(sensor_id, wind_metadata, min_vel=0, max_vel=70):
    """
    处理单个传感器的数据，为每个有越界风速的极端时间窗口绘图
    
    参数:
        sensor_id: 传感器ID
        wind_metadata: 风数据元数据列表（包含extreme_time_ranges）
        min_vel: 风速下界（m/s）
        max_vel: 风速上界（m/s）
    
    返回:
        window_figures: 列表，每项为 (fig, count, window_info) 元组
    """
    print(f"\n{'='*70}")
    print(f"处理传感器: {sensor_id}")
    print(f"传感器名称: {WIND_SENSOR_NAMES.get(sensor_id, '未知')}")
    print(f"风速范围: [{min_vel}, {max_vel}] m/s")
    print(f"{'='*70}")
    
    # 筛选该传感器的文件
    sensor_files = [item for item in wind_metadata if item.get('sensor_id') == sensor_id]
    
    if len(sensor_files) == 0:
        print(f"警告：传感器 {sensor_id} 无数据")
        return []
    
    print(f"✓ 找到 {len(sensor_files)} 个极端振动对应的风数据文件")
    
    # 存储所有有越界数据的窗口信息
    window_figures = []
    total_out_of_range_samples = 0
    window_count = 0
    
    for file_idx, file_info in enumerate(sensor_files):
        file_path = file_info.get('file_path')
        extreme_time_ranges = file_info.get('extreme_time_ranges', [])
        
        # 加载原始风数据
        wind_velocity, wind_direction = load_wind_data_raw(file_path)
        
        if wind_velocity is None or len(wind_velocity) == 0:
            print(f"  ⚠ 文件 {file_idx+1}/{len(sensor_files)} 加载失败")
            continue
        
        wind_velocity = np.array(wind_velocity)
        wind_direction = np.array(wind_direction)
        
        # 对每个极端时间窗口单独处理
        for window_idx, (start_sec, end_sec) in enumerate(extreme_time_ranges):
            start_idx = int(start_sec * WIND_FS)
            end_idx = int(end_sec * WIND_FS)
            
            start_idx = max(0, start_idx)
            end_idx = min(len(wind_velocity), end_idx)
            
            if start_idx < end_idx:
                window_vel = wind_velocity[start_idx:end_idx]
                window_dir = wind_direction[start_idx:end_idx]
                
                # 检查该窗口是否有越界风速
                has_out_of_range, out_of_range_mask = check_window_out_of_range(
                    window_vel, min_vel, max_vel
                )
                
                if has_out_of_range:
                    window_count += 1
                    out_count = np.sum(out_of_range_mask)
                    total_out_of_range_samples += out_count
                    
                    print(f"  ✓ 文件 {file_idx+1}/{len(sensor_files)} 窗口 {window_idx+1}/{len(extreme_time_ranges)} 找到 {out_count} 个越界样本")
                    
                    # 绘制完整窗口数据，突出显示越界点
                    fig, ax = plot_window_with_out_of_range_highlight(
                        window_vel,
                        out_of_range_mask,
                        sensor_id,
                        min_vel,
                        max_vel,
                        title=f"{sensor_id} - 窗口 {window_count} (文件{file_idx+1}窗口{window_idx+1}) [越界: {out_count}个]"
                    )
                    
                    window_info = {
                        'sensor_id': sensor_id,
                        'file_idx': file_idx,
                        'window_idx': window_idx,
                        'start_time': start_sec,
                        'end_time': end_sec,
                        'out_of_range_count': out_count
                    }
                    window_figures.append((fig, out_count, window_info))
    
    if len(window_figures) == 0:
        print(f"警告：传感器 {sensor_id} 无越界风速样本，跳过")
        return []
    
    print(f"\n✓ 数据统计完成")
    print(f"  找到有越界数据的窗口数: {len(window_figures)}")
    print(f"  总越界风速样本数: {total_out_of_range_samples}")
    
    return window_figures


def main():
    """
    主函数：调用工作流，为每个传感器绘制越界风速时程曲线
    仅保留前5张图，如果生成图表数 >= 5，则不保存
    """
    print("="*80)
    print(" "*15 + "图2.6: 高能量振动对应的越界风速时程曲线")
    print("="*80)
    
    # 配置参数
    MIN_VEL = 0
    MAX_VEL = 70
    MAX_FIGS = 5
    
    print(f"\n风速有效范围: [{MIN_VEL}, {MAX_VEL}] m/s")
    print(f"最多保留图表数: {MAX_FIGS} 张")
    
    # Step 1: 运行振动数据工作流
    print("\n[Step 1] 运行振动数据工作流...")
    print("-"*80)
    vib_metadata = run_vib_workflow(use_cache=True)
    print(f"✓ 获取振动数据元数据: {len(vib_metadata)} 条")
    
    # Step 2: 运行风数据工作流（获取极端振动对应的风数据）
    print("\n[Step 2] 运行风数据工作流（获取极端振动对应的风数据）...")
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
    
    # Step 3: 为每个传感器处理数据并绘图
    print("\n[Step 3] 为每个传感器处理数据并绘图...")
    print("="*80)
    
    # 获取所有传感器ID
    sensor_ids = sorted(set(item.get('sensor_id') for item in wind_metadata))
    
    # 初始化 PlotLib
    ploter = PlotLib()
    
    # 用于跟踪已保存的图表数和越界样本总数
    saved_figs_count = 0
    total_out_of_range_samples = 0
    all_window_figs = []
    
    # 处理每个传感器，收集所有有越界数据的窗口图表
    for sensor_id in sensor_ids:
        window_figures = process_sensor_data(sensor_id, wind_metadata, MIN_VEL, MAX_VEL)
        
        if len(window_figures) > 0:
            all_window_figs.extend(window_figures)
            for fig, count, window_info in window_figures:
                total_out_of_range_samples += count
    
    # 保留前 MAX_FIGS 张图表
    print(f"\n{'='*70}")
    print(f"图表筛选与保存")
    print(f"{'='*70}")
    print(f"✓ 共找到 {len(all_window_figs)} 个有越界数据的时间窗口")
    print(f"✓ 最多保留图表数: {MAX_FIGS} 张")
    
    if len(all_window_figs) > MAX_FIGS:
        print(f"\n⚠ 窗口数量超过限制，仅保留前 {MAX_FIGS} 张")
        # 保留前 MAX_FIGS 张，关闭其余的
        for i, (fig, count, window_info) in enumerate(all_window_figs):
            if i < MAX_FIGS:
                ploter.figs.append(fig)
                saved_figs_count += 1
                print(f"  ✓ 保存图表 {i+1}/{MAX_FIGS} - {window_info['sensor_id']} 窗口 {window_info['window_idx']+1}")
            else:
                plt.close(fig)
                print(f"  ✗ 丢弃图表 - {window_info['sensor_id']} 窗口 {window_info['window_idx']+1}")
    else:
        # 保存所有图表
        for i, (fig, count, window_info) in enumerate(all_window_figs):
            ploter.figs.append(fig)
            saved_figs_count += 1
            print(f"  ✓ 保存图表 {i+1}/{len(all_window_figs)} - {window_info['sensor_id']} 窗口 {window_info['window_idx']+1}")
    
    # 完成
    print("\n" + "="*80)
    print(" "*20 + "绘图处理完成")
    print("="*80)
    print(f"✓ 总越界时间窗口数: {len(all_window_figs)}")
    print(f"✓ 已保存图表数: {saved_figs_count}/{len(all_window_figs)}")
    print(f"✓ 总越界样本数: {total_out_of_range_samples}")
    print(f"✓ 数据来源：高能量振动对应的风速数据（1Hz采样）")
    print(f"✓ 极端时间窗口总数: {total_extreme_windows}")
    
    if saved_figs_count == 0:
        print(f"\n⚠ 未发现任何越界风速数据，无图表生成")
        return
    
    print(f"\n✓ 使用 PlotLib 展示图表...")
    
    plt.close('all')
    ploter.show()


if __name__ == "__main__":
    main()
