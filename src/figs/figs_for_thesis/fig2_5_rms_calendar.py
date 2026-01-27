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
        FONT_SIZE, ENG_FONT, CN_FONT, FIG_SIZE, ANNOTATION_COLOR, 
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

# 常量定义
RMS_95_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\results\statistics\rms_statistics.json'
RMS_EXTREME_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\results\statistics\rms_statistics_extreme.json'


def load_rms_daily_counts(json_path, month_str="09"):
    """
    读取统计结果并汇总每日发生频次
    """
    if not os.path.exists(json_path):
        print(f"错误：未找到统计结果文件 {json_path}")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计逻辑：遍历所有条目，对符合月份的日期，累加 indices 长度
    daily_counts = {}
    for entry in data:
        if entry.get("month") == month_str:
            day = int(entry.get("day"))
            # 统计频次：该小时内发生 extreme 振动的窗口数量
            count = len(entry.get("indices", []))
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

def plot_rms_95_calendar(ploter=None):
    """
    绘制 95% 振动发生频次日历
    """
    daily_counts = load_rms_daily_counts(RMS_95_PATH, month_str="09")
    if not daily_counts:
        print("未发现 95% 振动统计数据。")
        return None
    
    fig = _plot_calendar_core(daily_counts)
    if ploter:
        ploter.figs.append(fig)
        plt.close(fig)
    return fig

def plot_rms_extreme_calendar(ploter=None):
    """
    绘制 Top 0.25% 极端振动发生频次日历
    """
    daily_counts = load_rms_daily_counts(RMS_EXTREME_PATH, month_str="09")
    if not daily_counts:
        print("未发现 Top 0.25% 极端振动统计数据。")
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

