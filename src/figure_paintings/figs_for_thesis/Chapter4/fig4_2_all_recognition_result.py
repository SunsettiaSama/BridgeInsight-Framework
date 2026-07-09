"""
识别结果可视化：2023年各月份各类振动占比分布

数据源：Chapter4 全量 DL 识别结果（已排除 C34-201/202），
见 data_config.CHAPTER4["predictions_enriched"]。

主要功能：
- 加载识别结果 JSON
- 按月份聚合各类振动的窗口计数
- 生成堆积柱状图展示全年分布
- 推送到 WebUI 查看
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import json
import logging
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from typing import Dict

from src.visualize_tools.web_dashboard import push as web_push
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import load_dl_result

# 已排除 C34-201/202 的 DL 全量识别结果
DL_RESULT_PATH = data_config.PROJECT_ROOT / data_config.CHAPTER4["predictions_enriched"]
PAGE_NAME = "fig3_1 月份分布"

# 从统一配置模块导入图像配置（字体、尺寸、配色）
from src.figure_paintings.figs_for_thesis.config import ENG_FONT, CN_FONT, FONT_SIZE, REC_FIG_SIZE, get_blue_color_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 再次应用全局样式，避免被其它模块导入副作用覆盖
plt.style.use("default")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = FONT_SIZE

# 类别标签映射
CLASS_LABELS = {
    0: "随机振动",
    1: "涡激共振", 
    2: "风雨振 ",
    3: "其他振动",
}
_class_palette = get_blue_color_map(style="discrete", start_map_index=1, end_map_index=5).colors
CLASS_COLORS = {cls_id: _class_palette[cls_id] for cls_id in range(4)}


def aggregate_by_month(
    result: Dict,
) -> Dict[int, Dict[int, int]]:
    """
    按月份聚合各类振动的窗口计数。
    
    Parameters
    ----------
    result : Dict
        识别结果字典
    
    Returns
    -------
    Dict[int, Dict[int, int]]
        格式: {month: {class_id: count}}
        例: {1: {0: 5000, 1: 3000, 2: 1000, 3: 100}, ...}
    """
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = {int(k): v for k, v in result["sample_metadata"].items()}
    
    monthly_counts = defaultdict(lambda: defaultdict(int))
    
    for idx, pred_label in predictions.items():
        if idx not in sample_metadata:
            logger.warning(f"样本 {idx} 缺少元数据")
            continue
        
        meta = sample_metadata[idx]
        month = meta["timestamp"][0]  # timestamp = (month, day, hour)
        
        monthly_counts[month][pred_label] += 1
    
    return dict(monthly_counts)


def plot_monthly_distribution(
    monthly_counts: Dict[int, Dict[int, int]],
):
    """
    绘制 2023 年各月份各类振动窗口数分布（分组柱状图 + 纵轴对数坐标）。

    Parameters
    ----------
    monthly_counts : Dict[int, Dict[int, int]]
        按月份聚合的各类窗口计数
    """
    months = sorted(monthly_counts.keys())

    class_counts = {cls_id: [] for cls_id in range(4)}

    for month in months:
        for cls_id in range(4):
            count = monthly_counts[month].get(cls_id, 0)
            class_counts[cls_id].append(count)

    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)

    x = np.arange(len(months))
    n_cls = 4
    group_w = 0.72
    bar_w = group_w / n_cls
    offsets = (np.arange(n_cls) - (n_cls - 1) / 2) * bar_w

    for cls_id in range(4):
        vals = np.array(class_counts[cls_id], dtype=float)
        vals_plot = np.where(vals > 0, vals, np.nan)
        ax.bar(
            x + offsets[cls_id],
            vals_plot,
            bar_w,
            label=CLASS_LABELS[cls_id],
            color=CLASS_COLORS[cls_id],
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)

    ax.set_xlabel("月份", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("窗口数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(
        "2023 年拉索振动分类结果月份分布",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE,
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}月" for m in months], fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)

    leg = ax.legend(
        loc="upper right",
        fontsize=FONT_SIZE - 2,
        framealpha=0.95,
        prop=CN_FONT,
    )
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))

    plt.tight_layout()

    return fig, ax


def plot_monthly_percentage(
    monthly_counts: Dict[int, Dict[int, int]],
):
    """
    绘制 2023 年各月份各类振动百分比堆积柱状图（纵轴为线性 0–100%，便于读占比）。

    Parameters
    ----------
    monthly_counts : Dict[int, Dict[int, int]]
        按月份聚合的各类窗口计数
    """
    months = sorted(monthly_counts.keys())

    class_percentages = {cls_id: [] for cls_id in range(4)}

    for month in months:
        total = sum(monthly_counts[month].values())
        for cls_id in range(4):
            count = monthly_counts[month].get(cls_id, 0)
            percentage = (count / total * 100) if total > 0 else 0
            class_percentages[cls_id].append(percentage)

    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)

    x = np.arange(len(months))
    width = 0.6

    bottom = np.zeros(len(months))
    for cls_id in range(4):
        percentages = np.array(class_percentages[cls_id])
        ax.bar(
            x,
            percentages,
            width,
            label=CLASS_LABELS[cls_id],
            color=CLASS_COLORS[cls_id],
            bottom=bottom,
        )
        bottom += percentages

    ax.set_xlabel("月份", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("占比 (%)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(
        "2023 年拉索振动分类结果月份占比分布",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE,
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}月" for m in months], fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)
    ax.set_ylim([0, 100])

    leg = ax.legend(
        loc="upper right",
        fontsize=FONT_SIZE - 2,
        framealpha=0.95,
        prop=CN_FONT,
    )
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    return fig, ax


def plot_monthly_stacked_count(
    monthly_counts: Dict[int, Dict[int, int]],
):
    """
    绘制 2023 年各月份各类振动窗口数堆叠柱状图（线性 y 轴，含全部 4 类）。
    """
    months = sorted(monthly_counts.keys())

    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)

    x = np.arange(len(months))
    width = 0.6

    bottom = np.zeros(len(months))
    for cls_id in range(4):
        counts = np.array([monthly_counts[month].get(cls_id, 0) for month in months], dtype=float)
        ax.bar(
            x,
            counts,
            width,
            label=CLASS_LABELS[cls_id],
            color=CLASS_COLORS[cls_id],
            bottom=bottom,
        )
        bottom += counts

    ax.set_xlabel("月份", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("窗口数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(
        "2023 年拉索振动分类结果月份分布（堆叠）",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE,
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}月" for m in months], fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)

    leg = ax.legend(loc="upper right", fontsize=FONT_SIZE - 2, framealpha=0.95, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    return fig, ax


def plot_monthly_stacked_count_viv_rain(
    monthly_counts: Dict[int, Dict[int, int]],
):
    """
    绘制 2023 年各月份涡激共振（Class 1）与风雨振（Class 2）窗口数堆叠柱状图
    （线性 y 轴，不含随机振动 Class 0 和其他振动 Class 3）。
    """
    TARGET_CLASSES = [1, 2]
    months = sorted(monthly_counts.keys())

    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)

    x = np.arange(len(months))
    width = 0.6

    bottom = np.zeros(len(months))
    for cls_id in TARGET_CLASSES:
        counts = np.array([monthly_counts[month].get(cls_id, 0) for month in months], dtype=float)
        ax.bar(
            x,
            counts,
            width,
            label=CLASS_LABELS[cls_id],
            color=CLASS_COLORS[cls_id],
            bottom=bottom,
        )
        bottom += counts

    ax.set_xlabel("月份", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("窗口数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(
        "2023 年涡激共振与风雨振月份分布（堆叠）",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE,
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}月" for m in months], fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)

    leg = ax.legend(loc="upper right", fontsize=FONT_SIZE - 2, framealpha=0.95, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()

    return fig, ax


def print_monthly_statistics(monthly_counts: Dict[int, Dict[int, int]]) -> None:
    """打印月份统计信息。"""
    print("\n" + "=" * 80)
    print("2023 年各月份振动分类统计".center(80))
    print("=" * 80)
    
    for month in sorted(monthly_counts.keys()):
        total = sum(monthly_counts[month].values())
        print(f"\n【{month:2d} 月】总窗口数: {total:8d}")
        
        for cls_id in range(4):
            count = monthly_counts[month].get(cls_id, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {CLASS_LABELS[cls_id]:20s}: {count:8d} ({percentage:6.2f}%)")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    if not DL_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"DL 识别结果不存在：{DL_RESULT_PATH}\n"
            "请先运行：python scripts/filter_chapter4_predictions.py"
        )

    logger.info("加载 DL 识别结果（已排除 C34-201/202）：%s", DL_RESULT_PATH.name)
    result = load_dl_result()
    
    # 按月份聚合
    monthly_counts = aggregate_by_month(result)
    
    # 打印统计信息
    print_monthly_statistics(monthly_counts)
    
    # 生成图表并推送到 WebUI
    fig_count, _ = plot_monthly_distribution(monthly_counts)
    fig_pct, _ = plot_monthly_percentage(monthly_counts)
    fig_stacked, _ = plot_monthly_stacked_count(monthly_counts)
    fig_viv_rain, _ = plot_monthly_stacked_count_viv_rain(monthly_counts)

    figures = [
        (fig_count, "月份分布（对数）"),
        (fig_pct, "月份占比"),
        (fig_stacked, "月份分布（堆叠）"),
        (fig_viv_rain, "涡激+风雨振（堆叠）"),
    ]
    for slot, (fig, title) in enumerate(figures):
        web_push(
            fig,
            page=PAGE_NAME,
            slot=slot,
            title=title,
            page_cols=2 if slot == 0 else None,
        )

    logger.info("图表已推送到 WebUI：%s", PAGE_NAME)


if __name__ == "__main__":
    main()
