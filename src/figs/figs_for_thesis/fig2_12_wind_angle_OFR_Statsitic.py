"""
图2.9: 高能量振动对应的越界风攻角数据统计直方图
基于极端振动样本对应的风攻角数据，通过step3的越界点查询，统计每个传感器的越界窗口占比

功能特点：
1. 使用 step3 工作流结果作为数据来源
2. 统计每个传感器的越界数据窗口占比（正常窗口占比 = 1 - 越界窗口数/总窗口数）
3. 绘制直方图展示数据分布
4. 支持多传感器独立统计和展示
5. 计算全局统计指标（平均占比、最大值、最小值等）

越界范围：
- 风攻角: < -45 ° 或 > 45 °
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.statistics.wind_data_io_process.workflow import run as run_wind_workflow
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow
from src.visualize_tools.utils import PlotLib

from .config import ENG_FONT, CN_FONT, REC_FONT_SIZE, REC_FIG_SIZE, get_blue_color_map

from src.config.sensor_config import WIND_SENSOR_NAMES

plt.style.use('default')
plt.rcParams['font.size'] = REC_FONT_SIZE

FONT_SIZE = REC_FONT_SIZE

# 重构风速数据与否
FORCE_RECOMPUTE = False

def calculate_sensor_statistics(metadata):
    """
    计算每个传感器的风攻角越界窗口占比统计
    
    参数:
        metadata: step3 处理后的元数据，包含out_of_range_windows信息
    
    返回:
        sensor_stats: 字典，每个传感器包含：
            {
                'sensor_id': str,
                'sensor_name': str,
                'total_windows': int,
                'angle_out_of_range_windows': int,
                'normal_windows': int,
                'normal_ratio': float,  # 正常窗口占比
                'angle_out_of_range_ratio': float  # 风攻角越界窗口占比
            }
    """
    # 按传感器统计窗口数
    sensor_window_stats = defaultdict(lambda: {'total': 0, 'angle_out_of_range': 0})
    
    for item in metadata:
        sensor_id = item.get('sensor_id')
        extreme_time_ranges = item.get('extreme_time_ranges', [])
        out_of_range_windows = item.get('out_of_range_windows', [])
        
        # 统计该文件的极端时间窗口总数
        total_windows = len(extreme_time_ranges)
        sensor_window_stats[sensor_id]['total'] += total_windows
        
        # 仅统计有风攻角越界点的窗口数
        angle_out_of_range_count = 0
        for window in out_of_range_windows:
            if window.get('ang_out_of_range', {}).get('count', 0) > 0:
                angle_out_of_range_count += 1
        
        sensor_window_stats[sensor_id]['angle_out_of_range'] += angle_out_of_range_count
    
    # 计算每个传感器的占比
    sensor_stats = {}
    for sensor_id, stats in sensor_window_stats.items():
        total = stats['total']
        angle_out_of_range = stats['angle_out_of_range']
        normal = total - angle_out_of_range
        
        sensor_stats[sensor_id] = {
            'sensor_id': sensor_id,
            'sensor_name': WIND_SENSOR_NAMES.get(sensor_id, '未知'),
            'total_windows': total,
            'angle_out_of_range_windows': angle_out_of_range,
            'normal_windows': normal,
            'normal_ratio': normal / total if total > 0 else 0.0,
            'angle_out_of_range_ratio': angle_out_of_range / total if total > 0 else 0.0
        }
    
    return sensor_stats


def plot_sensor_statistics_histogram(sensor_stats):
    """
    绘制传感器风攻角越界窗口占比的直方图
    
    参数:
        sensor_stats: 传感器统计字典
    
    返回:
        fig, ax: matplotlib 图表对象
    """
    # 按传感器ID排序
    sorted_sensors = sorted(sensor_stats.items(), key=lambda x: x[0])
    
    sensor_ids = [item[0] for item in sorted_sensors]
    sensor_names = [item[1]['sensor_name'] for item in sorted_sensors]
    normal_ratios = [item[1]['normal_ratio'] * 100 for item in sorted_sensors]  # 转换为百分比
    
    # 创建图表
    fig = plt.figure(figsize=REC_FIG_SIZE)
    ax = fig.add_subplot(111)
    
    # 从蓝色色图中提取颜色（使用更深的颜色）
    blue_cmap = get_blue_color_map(style='discrete')
    bar_color = blue_cmap(3)  # 选择最深的蓝色（最黑）
    
    # 绘制直方图
    x_pos = np.arange(len(sensor_ids))
    bars = ax.bar(
        x_pos,
        normal_ratios,
        color=bar_color,
        alpha=0.8,
        edgecolor='#403040',
        linewidth=1.5,
        # label='正常窗口占比'
    )
    
    # 在每个柱子上添加数值标签（使用英文字体）
    for i, (bar, ratio) in enumerate(zip(bars, normal_ratios)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f'{ratio:.1f}%',
            ha='center',
            va='bottom',
            fontproperties=ENG_FONT,
            fontsize=FONT_SIZE - 2
        )
    
    # 设置轴标签和标题（不包含标题）
    # ax.set_xlabel('传感器', fontproperties=CN_FONT, fontsize=FONT_SIZE, labelpad=10)
    ax.set_ylabel('正常窗口占比 (%)', fontproperties=CN_FONT, fontsize=FONT_SIZE, labelpad=10)
    
    # 设置 x 轴标签（使用传感器名称，检查字符长度并添加换行）
    formatted_sensor_names = []
    for name in sensor_names:
        if len(name) > 6:
            formatted_sensor_names.append(name[:6] + '\n' + name[6:])
        else:
            formatted_sensor_names.append(name)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(formatted_sensor_names, fontproperties=CN_FONT, fontsize=FONT_SIZE - 2, rotation=45, ha='right')
    
    # 设置 y 轴范围（0-110%，允许 legend 正确显示）
    ax.set_ylim(0, 110)
    
    # 添加网格
    ax.grid(True, axis='y', color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加参考线（如50%、75%等）
    ax.axhline(y=50, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='50%')
    ax.axhline(y=75, color='red', linestyle=':', linewidth=1, alpha=0.5, label='75%')
    
    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    
    # 设置Y轴标签为英文字体
    for label in ax.get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    
    # 添加图例
    legend = ax.legend(loc='lower right', fontsize=FONT_SIZE - 2, framealpha=0.9)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(ENG_FONT)
    
    plt.tight_layout()
    return fig, ax


def print_statistics_summary(sensor_stats):
    """
    打印统计摘要信息
    
    参数:
        sensor_stats: 传感器统计字典
    """
    print(f"\n{'='*80}")
    print(f"{'传感器风攻角越界窗口占比统计摘要':^80}")
    print(f"{'='*80}")
    
    # 创建表格头
    print(f"{'传感器ID':<12} {'传感器名称':<15} {'总窗口数':>10} {'风攻角越界窗口':>12} {'正常窗口':>10} {'正常占比':>12}")
    print(f"{'-'*80}")
    
    # 统计全局数据
    total_all_windows = 0
    total_angle_out_of_range_windows = 0
    
    # 按传感器ID排序打印
    for sensor_id in sorted(sensor_stats.keys()):
        stats = sensor_stats[sensor_id]
        print(
            f"{stats['sensor_id']:<12} {stats['sensor_name']:<15} "
            f"{stats['total_windows']:>10} {stats['angle_out_of_range_windows']:>12} "
            f"{stats['normal_windows']:>10} {stats['normal_ratio']*100:>11.2f}%"
        )
        
        total_all_windows += stats['total_windows']
        total_angle_out_of_range_windows += stats['angle_out_of_range_windows']
    
    # 全局统计
    print(f"{'-'*80}")
    
    if len(sensor_stats) > 0:
        normal_ratios = [s['normal_ratio'] for s in sensor_stats.values()]
        global_normal_ratio = (total_all_windows - total_angle_out_of_range_windows) / total_all_windows if total_all_windows > 0 else 0.0
        
        print(f"\n全局统计:")
        print(f"  总窗口数: {total_all_windows}")
        print(f"  风攻角越界窗口数: {total_angle_out_of_range_windows}")
        print(f"  正常窗口数: {total_all_windows - total_angle_out_of_range_windows}")
        print(f"  全局正常占比: {global_normal_ratio*100:.2f}%")
        print(f"  平均正常占比: {np.mean(normal_ratios)*100:.2f}%")
        print(f"  最高正常占比: {np.max(normal_ratios)*100:.2f}%")
        print(f"  最低正常占比: {np.min(normal_ratios)*100:.2f}%")
        print(f"  标准差: {np.std(normal_ratios)*100:.2f}%")
    
    print(f"{'='*80}\n")


def main():
    """
    主函数：通过 step3 的结果统计每个传感器的越界窗口占比并绘制直方图
    """
    print("="*80)
    print(" "*15 + "图2.9: 高能量振动对应的越界风攻角数据统计直方图")
    print("="*80)
    
    # Step 1: 运行振动数据工作流
    print("\n[Step 1] 运行振动数据工作流...")
    print("-"*80)
    vib_metadata = run_vib_workflow(use_cache=True)
    print(f"✓ 获取振动数据元数据: {len(vib_metadata)} 条")
    
    # Step 2: 运行风数据工作流（包含 step3 的越界点查询）
    print("\n[Step 2] 运行风数据工作流（包含越界点查询）...")
    print("-"*80)
    wind_metadata = run_wind_workflow(
        vib_metadata=vib_metadata,
        use_cache=True,
        force_recompute=FORCE_RECOMPUTE,
        extreme_only=True
    )
    print(f"✓ 获取极端振动对应的风数据元数据（含越界信息）: {len(wind_metadata)} 条")
    
    # 验证 step3 数据
    has_out_of_range = any(item.get('out_of_range_windows') for item in wind_metadata)
    if not has_out_of_range:
        print(f"\n⚠ 警告：未找到越界点信息，请确保工作流包含 step3")
        return
    
    print(f"✓ 数据包含 step3 越界点查询结果")
    
    # Step 3: 统计每个传感器的越界窗口占比
    print("\n[Step 3] 统计每个传感器的越界窗口占比...")
    print("-"*80)
    sensor_stats = calculate_sensor_statistics(wind_metadata)
    
    if len(sensor_stats) == 0:
        print(f"⚠ 未找到传感器数据")
        return
    
    print(f"✓ 统计完成: {len(sensor_stats)} 个传感器")
    
    # 打印统计摘要
    print_statistics_summary(sensor_stats)
    
    # Step 4: 绘制直方图
    print("[Step 4] 绘制直方图...")
    print("-"*80)
    
    fig, ax = plot_sensor_statistics_histogram(sensor_stats)
    
    print(f"✓ 直方图绘制完成")
    
    # Step 5: 显示图表
    print("\n" + "="*80)
    print(" "*25 + "绘图处理完成")
    print("="*80)
    print(f"✓ 已生成统计直方图")
    print(f"✓ 传感器数量: {len(sensor_stats)}")
    print(f"✓ 数据来源: 极端振动对应的风攻角数据（通过 step3 越界点查询）")
    
    # 使用 PlotLib 展示
    ploter = PlotLib()
    ploter.figs.append(fig)
    
    plt.close('all')
    ploter.show()


if __name__ == "__main__":
    main()
