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
import matplotlib.pyplot as plt
from typing import Dict, Optional

from src.visualize_tools.utils import PlotLib

# 从统一配置模块导入图像配置（字体、尺寸、配色）
from src.figure_paintings.figs_for_thesis.config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, get_blue_color_map,
)

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
    2: "风雨振",
    3: "其他振动",
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


def aggregate_total_counts(result: Dict) -> Dict[int, int]:
    """
    对全年所有月份的预测结果求和，得到每类的总窗口数。

    Parameters
    ----------
    result : Dict
        识别结果字典（含 predictions 字段）

    Returns
    -------
    Dict[int, int]
        格式: {class_id: total_count}
    """
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    total: Dict[int, int] = {cls_id: 0 for cls_id in range(4)}
    for pred_label in predictions.values():
        if pred_label in total:
            total[pred_label] += 1
    return total


def plot_class_distribution_pie(
    total_counts: Dict[int, int],
    output_path: Optional[str] = None,
):
    """
    绘制全年四类振动窗口数占比饼状图。
    - 饼内无文字标注
    - 图例横排置于饼图正下方，含中文标签、窗口数、占比
    """
    total_samples = sum(total_counts.values())
    cls_ids = list(range(4))
    counts = [total_counts.get(cls_id, 0) for cls_id in cls_ids]
    colors = [CLASS_COLORS[cls_id] for cls_id in cls_ids]

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    wedges, _ = ax.pie(
        counts,
        colors=colors,
        explode=[0.05] * 4,
        startangle=45,
        shadow=True,
    )

    legend_labels = [
        f"{CLASS_LABELS[cls_id]}  {counts[i]:,}（{counts[i] / total_samples * 100:.2f}%）"
        for i, cls_id in enumerate(cls_ids)
    ]
    legend = ax.legend(
        wedges,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=FONT_SIZE - 4,
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=1.0,
        handlelength=1.2,
    )
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)
        text.set_fontsize(FONT_SIZE - 4)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"图表已保存：{output_path}")
    else:
        plt.show()

    return fig, ax


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

    # 生成图表
    output_dir = project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_lib = PlotLib()

    # 全年四类占比饼状图
    total_counts = aggregate_total_counts(result)
    pie_fig_path = output_dir / "fig3_2_class_distribution_pie.png"
    fig_pie, _ = plot_class_distribution_pie(total_counts, str(pie_fig_path))
    plot_lib.figs.append(fig_pie)

    logger.info("图表生成完成！")
    plot_lib.show()


if __name__ == "__main__":
    main()
