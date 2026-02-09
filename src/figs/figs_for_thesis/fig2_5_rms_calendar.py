"""
================================================================================
文件依赖说明 (File Dependencies)
================================================================================
本文件依赖以下数据处理模块：

1. 数据来源:
   - 振动数据处理工作流缓存文件
     路径: F:\Research\Vibration Characteristics In Cable Vibration\results\vib\workflow_cache.json
   
2. 数据生成工作流:
   - src.data_processer.statistics.vibration_io_process.workflow
     └─> Step 0: 获取所有振动文件
     └─> Step 1: 缺失率筛选
     └─> Step 2: RMS 统计分析与极端振动识别
     └─> 输出: metadata（包含 extreme_rms_indices）

3. 配置文件:
   - src.figs.figs_for_thesis.config
     └─> 定义字体、颜色、尺寸等绘图配置

================================================================================
数据格式说明 (Data Format)
================================================================================

输入数据格式 (从 workflow_cache.json 读取):
{
    "metadata": [
        {
            "sensor_id": "传感器ID",
            "month": "09",
            "day": "01",
            "hour": "12",
            "file_path": "文件路径",
            "actual_length": 180000,
            "missing_rate": 0.02,
            "extreme_rms_indices": [5, 12, 23, ...]  // 极端振动的窗口索引列表
        },
        ...
    ],
    "process_params": {
        "rms_threshold_95": 0.1234,  // RMS 95%分位值阈值
        ...
    }
}

数据说明:
- metadata: 每个条目表示一个小时的振动数据文件元数据
- extreme_rms_indices: 该小时内超过 95% 分位值阈值的时间窗口索引列表
- 日历图根据每日累计的 extreme_rms_indices 数量进行着色

================================================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import calendar
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.colorbar import ColorbarBase

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. 从 config.py 导入配置
try:
    from src.figs.figs_for_thesis.config import (
        FONT_SIZE, ENG_FONT, CN_FONT, SQUARE_FIG_SIZE, ANNOTATION_COLOR, 
        get_blue_color_map
    )
    from src.visualize_tools.utils import PlotLib
except ImportError:
    # 备选配置（以防导入失败）
    FONT_SIZE = 14
    ENG_FONT = FontProperties(family='Times New Roman', size=FONT_SIZE)
    CN_FONT = FontProperties(family='SimSun', size=FONT_SIZE)
    ANNOTATION_COLOR = '#404040'
    def get_blue_color_map(style='gradient'):
        return plt.cm.viridis
    print("警告：无法导入配置，使用默认设置。")

# 常量定义 - 从 workflow 获取数据
WORKFLOW_CACHE_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\results\vib\workflow_cache.json'


def load_rms_daily_counts_from_workflow(workflow_cache_path=WORKFLOW_CACHE_PATH, month_str="09", rms_threshold_percentile=95):
    """
    从 workflow 缓存中读取元数据并汇总每日极端振动发生频次
    
    参数:
        workflow_cache_path: workflow 缓存文件路径
        month_str: 目标月份（字符串格式，如 "09"）
        rms_threshold_percentile: RMS 阈值分位数（95 表示取95%以上的作为极端振动）
    
    返回:
        daily_counts: 每日极端振动次数字典 {day: count}
    """
    if not os.path.exists(workflow_cache_path):
        print(f"错误：未找到 workflow 缓存文件 {workflow_cache_path}")
        return {}

    with open(workflow_cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    # 获取元数据和处理参数
    metadata = cache_data.get('metadata', [])
    process_params = cache_data.get('process_params', {})
    
    # 获取 RMS 阈值（如果使用 95%，则统计所有有极端振动索引的样本）
    # rms_threshold_95 = process_params.get('rms_threshold_95', None)
    
    # 统计逻辑：遍历所有元数据，对符合月份的日期，累加极端振动窗口数量
    daily_counts = {}
    for entry in metadata:
        if entry.get("month") == month_str:
            day = int(entry.get("day"))
            # 统计频次：该小时内发生极端振动的窗口数量
            extreme_indices = entry.get("extreme_rms_indices", [])
            count = len(extreme_indices)
            daily_counts[day] = daily_counts.get(day, 0) + count
            
    return daily_counts

def _plot_calendar_core(daily_counts):
    """
    核心绘图逻辑：根据每日频次生成日历 Figure
    """
    # 获取颜色映射 (从 config.py)
    cmap = get_blue_color_map(style='gradient')
    
    # 补全所有日期（1-30日），未出现的日期设置为0
    for day in range(1, 31):
        if day not in daily_counts:
            daily_counts[day] = 0
    
    # 统计最大值和最小值（线性归一化）
    max_val = max(daily_counts.values()) if daily_counts else 1
    min_val = 0
    
    # 使用线性归一化
    norm = Normalize(vmin=min_val, vmax=max_val)
    
    # 创建图形和主轴
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 生成日历网格
    year, month = 2024, 9
    cal_matrix = calendar.monthcalendar(year, month)
    while len(cal_matrix) < 6:
        cal_matrix.append([0]*7)
    
    # 绘制日历热力块 + 日期 + 频次标注
    grid_color = "#dddddd"
    text_color = "#333333"
    
    for week_idx, week in enumerate(cal_matrix):
        for day_idx, day in enumerate(week):
            x = day_idx + 0.5
            y = week_idx + 0.5
            
            # 获取当日频次
            count = daily_counts.get(day, 0) if day != 0 else 0
            
            # 确定颜色
            if day != 0:
                color = cmap(norm(count))
            else:
                color = "#ffffff"
            
            # 绘制方块
            rect = mpatches.Rectangle(
                (day_idx, week_idx), 1, 1,
                facecolor=color,
                edgecolor=grid_color,
                linewidth=1
            )
            ax.add_patch(rect)
            
            # 绘制日期数字
            if day != 0:
                ax.text(
                    x, y + 0.15, str(day),
                    ha="center", va="center",
                    color=text_color,
                    fontproperties=ENG_FONT,
                    fontsize=FONT_SIZE
                )
                
                # 绘制频次标注
                if count >= 0:
                    ax.text(
                        x + 0.45, y + 0.45, " " * 4 + r"$(\times$" + f"{count}" + r"$)$",
                        ha="right", va="bottom",
                        color=ANNOTATION_COLOR,
                        fontproperties=ENG_FONT,
                        fontsize=FONT_SIZE - 2,
                        weight='bold'
                    )
    
    # 绘制星期标签
    weekday_labels = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]
    label_y = 6.3
    for idx, label in enumerate(weekday_labels):
        ax.text(
            idx + 0.5, label_y, label,
            ha="center", va="center",
            color="#333333",
            weight="bold",
            fontproperties=CN_FONT,
            fontsize=FONT_SIZE
        )
    
    # 设置轴范围
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6.8)
    ax.invert_yaxis()
    ax.axis("off")
    
    # 添加连续颜色条
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.5)
    
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('频次', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    for label in cax.get_yticklabels():
        label.set_fontproperties(ENG_FONT)
        label.set_fontsize(FONT_SIZE - 2)
    
    plt.tight_layout()
    return fig

def plot_rms_95_calendar(ploter=None, workflow_cache_path=WORKFLOW_CACHE_PATH, month_str="09"):
    """
    绘制 95% 振动发生频次日历（从 workflow 缓存读取）
    
    参数:
        ploter: PlotLib 实例
        workflow_cache_path: workflow 缓存文件路径
        month_str: 目标月份
    """
    daily_counts = load_rms_daily_counts_from_workflow(
        workflow_cache_path=workflow_cache_path,
        month_str=month_str,
        rms_threshold_percentile=95
    )
    if not daily_counts:
        print("未发现振动统计数据。")
        return None
    
    fig = _plot_calendar_core(daily_counts)
    if ploter:
        ploter.figs.append(fig)
        plt.close(fig)
    return fig

def plot_rms_extreme_calendar(ploter=None, workflow_cache_path=WORKFLOW_CACHE_PATH, month_str="09"):
    """
    绘制极端振动发生频次日历（从 workflow 缓存读取）
    
    注意：此函数与 plot_rms_95_calendar 使用相同的数据源（workflow metadata 中的 extreme_rms_indices）
    如需区分不同阈值，需在 workflow 中添加更多统计信息
    
    参数:
        ploter: PlotLib 实例
        workflow_cache_path: workflow 缓存文件路径
        month_str: 目标月份
    """
    # 当前使用相同的数据源（95%分位值以上的极端振动）
    daily_counts = load_rms_daily_counts_from_workflow(
        workflow_cache_path=workflow_cache_path,
        month_str=month_str,
        rms_threshold_percentile=95
    )
    if not daily_counts:
        print("未发现极端振动统计数据。")
        return None
    
    fig = _plot_calendar_core(daily_counts)
    if ploter:
        ploter.figs.append(fig)
        plt.close(fig)
    return fig

def plot_vibration_calendar_results(ploter = None):
    """
    主接口：绘制 95% 振动和 Top 0.25% 极端振动的日历图，并整合到 ploter 对象中
    
    参数:
        ploter: PlotLib 实例。如果为 None，则内部创建一个新的。
    返回:
        figs: 包含生成的所有 Figure 对象的列表
    """
    if ploter is None:
        ploter = PlotLib()
        
    figs = []
    
    # 1. 绘制 95% 振动日历
    fig_95 = plot_rms_95_calendar(ploter)
    if fig_95:
        figs.append(fig_95)
        
    # 2. 绘制 Top 0.25% 极端振动日历
    fig_extreme = plot_rms_extreme_calendar(ploter)
    if fig_extreme:
        figs.append(fig_extreme)
    
    ploter.figs.extend(figs)
    ploter.show()
    return figs

