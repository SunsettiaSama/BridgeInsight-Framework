"""
识别结果可视化：2023年各月份各类振动占比分布

主要功能：
- 加载识别结果 JSON
- 按月份聚合各类振动的窗口计数
- 生成堆积柱状图展示全年分布
- 支持按拉索对、传感器等维度筛选统计
- 使用 PlotLib GUI 交互式查看和保存
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from src.visualize_tools.utils import PlotLib

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
    0: "随机振动 (Class 0)",
    1: "涡激共振 (Class 1)", 
    2: "风雨振 (Class 2)",
    3: "其他振动 (Class 3)",
}
_class_palette = get_blue_color_map(style="discrete", start_map_index=1, end_map_index=5).colors
CLASS_COLORS = {cls_id: _class_palette[cls_id] for cls_id in range(4)}


def load_identification_result(result_path: str) -> Dict:
    """
    加载识别结果 JSON 文件。
    
    支持两种格式：
    1. 新格式（含 sample_metadata）
    2. 旧格式（不含 sample_metadata）- 将自动补全
    
    Parameters
    ----------
    result_path : str
        识别结果 JSON 文件路径
    
    Returns
    -------
    Dict
        包含 predictions、sample_metadata、by_file 等字段的字典
    """
    result_p = Path(result_path)
    enriched_path = result_p.with_name(result_p.stem + "_enriched" + result_p.suffix)

    # 优先加载已经补全过的文件，避免重复耗时补全
    if enriched_path.exists():
        logger.info(f"检测到已补全文件，直接加载：{enriched_path}")
        with open(enriched_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"加载识别结果：{result_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    logger.info(f"  - 样本总数: {result['metadata']['num_samples']}")
    logger.info(f"  - 类别数: {result['metadata']['num_classes']}")
    logger.info(f"  - 生成时间: {result['metadata']['created_at']}")

    # 检查是否需要补全 sample_metadata
    if "sample_metadata" not in result or not result["sample_metadata"]:
        logger.warning("⚠ 识别结果缺少 sample_metadata，将自动补全...")
        result = _enrich_result_with_dataset_metadata(result)
        with open(enriched_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 补全结果已保存至：{enriched_path}")

    return result


def _enrich_result_with_dataset_metadata(result: Dict) -> Dict:
    """
    从数据集补全识别结果中的 sample_metadata。
    
    Parameters
    ----------
    result : Dict
        不含 sample_metadata 的识别结果
    
    Returns
    -------
    Dict
        补全后的识别结果
    """
    from pathlib import Path
    import yaml
    from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (
        StayCableVib2023Config,
    )
    from src.data_processer.datasets.data_factory import get_dataset
    
    logger.info("加载数据集以补全元数据...")
    
    # fig3_1_all_recognition_result.py 位于 src/figure_paintings/figs_for_thesis/Chapter3/
    # 需要 5 个 parent 到达项目根目录
    project_root = Path(__file__).parent.parent.parent.parent.parent
    dataset_config_path = project_root / "config" / "train" / "datasets" / "total_staycable_vib.yaml"
    
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config_dict = yaml.safe_load(f)
    
    dataset_config = StayCableVib2023Config(**dataset_config_dict)
    dataset = get_dataset(dataset_config)
    
    logger.info(f"数据集加载完成（{len(dataset)} 个样本）")
    
    # 计算数据集指纹并验证
    dataset_fingerprint = dataset._compute_fingerprint()
    dataset_fingerprint_hash = dataset._fingerprint_hash(dataset_fingerprint)
    logger.info(f"当前数据集指纹: {dataset_fingerprint_hash}")
    
    # 如果原结果中有指纹，验证是否匹配
    original_hash = result.get("metadata", {}).get("dataset_fingerprint_hash")
    if original_hash and original_hash != dataset_fingerprint_hash:
        logger.warning(
            f"⚠ 数据集指纹不匹配！\n"
            f"  识别时指纹: {original_hash}\n"
            f"  当前指纹:   {dataset_fingerprint_hash}\n"
            f"  可能导致样本索引对应错误，请检查数据集配置是否变化"
        )
    
    # 转换 predictions 键为整数
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    
    # 只为有预测结果的样本构建 sample_metadata
    sample_metadata = {}
    missing_count = 0
    for idx in predictions.keys():
        if idx < len(dataset._samples):
            rec = dataset._samples[idx]
            sample_metadata[str(idx)] = {
                "cable_pair":          list(rec.cable_pair),
                "timestamp":           list(rec.timestamp_key),
                "window_idx":          rec.window_idx,
                "inplane_sensor_id":   rec.inplane_meta.get("sensor_id"),
                "outplane_sensor_id":  rec.outplane_meta.get("sensor_id"),
                "inplane_file_path":   rec.inplane_meta.get("file_path"),
                "outplane_file_path":  rec.outplane_meta.get("file_path"),
                "missing_rate_in":     rec.inplane_meta.get("missing_rate"),
                "missing_rate_out":    rec.outplane_meta.get("missing_rate"),
                "has_wind":            rec.wind_meta is not None,
            }
        else:
            logger.warning(f"样本索引 {idx} 超出数据集范围（数据集共 {len(dataset._samples)} 个样本）")
            missing_count += 1
    
    if missing_count > 0:
        logger.warning(f"共 {missing_count} 个样本索引超出范围")
    
    result["sample_metadata"] = sample_metadata
    result["metadata"]["enrichment_note"] = "Sample metadata自动补全"
    result["metadata"]["dataset_fingerprint_hash"] = dataset_fingerprint_hash
    
    logger.info(f"✓ 补全完成，共 {len(sample_metadata)} 个样本")
    
    return result


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
    output_path: Optional[str] = None,
):
    """
    绘制 2023 年各月份各类振动窗口数分布（分组柱状图 + 纵轴对数坐标）。

    Parameters
    ----------
    monthly_counts : Dict[int, Dict[int, int]]
        按月份聚合的各类窗口计数
    output_path : str, optional
        图片保存路径
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

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"图表已保存：{output_path}")
    else:
        plt.show()

    return fig, ax


def plot_monthly_percentage(
    monthly_counts: Dict[int, Dict[int, int]],
    output_path: Optional[str] = None,
):
    """
    绘制 2023 年各月份各类振动百分比堆积柱状图（纵轴为线性 0–100%，便于读占比）。

    Parameters
    ----------
    monthly_counts : Dict[int, Dict[int, int]]
        按月份聚合的各类窗口计数
    output_path : str, optional
        图片保存路径
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

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"图表已保存：{output_path}")
    else:
        plt.show()

    return fig, ax


def plot_monthly_stacked_count(
    monthly_counts: Dict[int, Dict[int, int]],
    output_path: Optional[str] = None,
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

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"图表已保存：{output_path}")
    else:
        plt.show()

    return fig, ax


def plot_monthly_stacked_count_viv_rain(
    monthly_counts: Dict[int, Dict[int, int]],
    output_path: Optional[str] = None,
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

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"图表已保存：{output_path}")
    else:
        plt.show()

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
    # 识别结果路径（使用最新的结果文件）
    # fig3_1_all_recognition_result.py 位于 src/figure_paintings/figs_for_thesis/Chapter3/
    project_root = Path(__file__).parent.parent.parent.parent.parent
    
    # 查找最新的识别结果文件
    result_dir = project_root / "results" / "identification_result"
    if not result_dir.exists():
        logger.error(f"识别结果目录不存在：{result_dir}")
        return
    
    result_files = sorted(result_dir.glob("res_cnn_full_dataset_*.json"))
    if not result_files:
        logger.error("未找到识别结果文件")
        return
    
    result_path = result_files[-1]  # 选择最新的结果文件
    logger.info(f"使用结果文件：{result_path}")
    
    # 加载识别结果
    result = load_identification_result(str(result_path))
    
    # 按月份聚合
    monthly_counts = aggregate_by_month(result)
    
    # 打印统计信息
    print_monthly_statistics(monthly_counts)
    
    # 生成图表
    output_dir = project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_lib = PlotLib()
    
    # 绝对数量分布图（对数坐标）
    count_fig_path = output_dir / "fig3_1_monthly_distribution_count.png"
    fig_count, _ = plot_monthly_distribution(monthly_counts, str(count_fig_path))
    plot_lib.figs.append(fig_count)
    
    # 百分比堆积柱状图
    percentage_fig_path = output_dir / "fig3_1_monthly_distribution_percentage.png"
    fig_pct, _ = plot_monthly_percentage(monthly_counts, str(percentage_fig_path))
    plot_lib.figs.append(fig_pct)

    # 全类别线性堆叠图
    stacked_fig_path = output_dir / "fig3_1_monthly_stacked_count.png"
    fig_stacked, _ = plot_monthly_stacked_count(monthly_counts, str(stacked_fig_path))
    plot_lib.figs.append(fig_stacked)

    # 仅涡激共振 + 风雨振线性堆叠图
    viv_rain_fig_path = output_dir / "fig3_1_monthly_stacked_viv_rain.png"
    fig_viv_rain, _ = plot_monthly_stacked_count_viv_rain(monthly_counts, str(viv_rain_fig_path))
    plot_lib.figs.append(fig_viv_rain)

    logger.info("图表生成完成！")
    plot_lib.show()


if __name__ == "__main__":
    main()
