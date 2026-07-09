"""
识别结果可视化：全年四类振动窗口占比饼状图

数据源：Chapter4 DL 识别结果（已排除 C34-201/202），推送到 WebUI。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict

from src.visualize_tools.web_dashboard import push as web_push
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
from src.chapter4_characteristics.settings import (
    get_inference_path,
    load_config as load_chapter4_config,
)

PAGE_NAME = "fig4_3 全量识别类别占比"
SPECIAL_CLASSES = (1, 2, 3)

# 从统一配置模块导入图像配置（字体、尺寸、配色）
from src.figure_paintings.figs_for_thesis.config import (
    ENG_FONT,
    CN_FONT,
    FONT_SIZE,
    SQUARE_FIG_SIZE,
    get_blue_color_map,
    get_red_color_map,
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
_minor_palette = get_red_color_map(style="discrete").colors
MINOR_CLASS_COLORS = {
    1: _minor_palette[3],
    2: _minor_palette[5],
    3: _minor_palette[1],
}


def _load_chapter4_cfg() -> dict:
    runtime_path = data_config.CHAPTER4.get("runtime_config_path")
    return load_chapter4_config(str(runtime_path) if runtime_path else None)


def _is_excluded_record(record: Dict) -> bool:
    excluded = data_config.EXCLUDED_SENSOR_IDS
    return (
        record.get("inplane_sensor_id") in excluded
        or record.get("outplane_sensor_id") in excluded
    )


def _sensitive_prediction(record: Dict) -> int:
    in_pred = int(record.get("inplane_prediction", record.get("prediction", 0)))
    out_pred = int(record.get("outplane_prediction", record.get("prediction", 0)))
    for cls_id in SPECIAL_CLASSES:
        if in_pred == cls_id or out_pred == cls_id:
            return cls_id
    return 0


def load_inference_records() -> list[Dict]:
    cfg = _load_chapter4_cfg()
    inference_path = get_inference_path(cfg)
    if not inference_path.exists():
        raise FileNotFoundError(
            f"单侧推理结果不存在：{inference_path}\n"
            "图4-3需要 inference.json 中的 inplane_prediction / outplane_prediction 字段，"
            "请先运行：python -m src.chapter4_characteristics infer"
        )

    logger.info("加载单侧推理结果：%s", inference_path.name)
    with open(inference_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    return [
        record for record in records
        if not _is_excluded_record(record)
    ]


def aggregate_total_counts(records: list[Dict]) -> Dict[int, int]:
    """
    对全年所有月份的预测结果求和，得到每类的总窗口数。

    面内或面外任一方向命中 1/2/3 时，即将该窗口计入对应特殊振动类别；
    两个方向均为随机振动时才计入 0。
    """
    total: Dict[int, int] = {cls_id: 0 for cls_id in range(4)}
    for record in records:
        pred_label = _sensitive_prediction(record)
        if pred_label in total:
            total[pred_label] += 1
    return total


def plot_class_distribution_pie(
    total_counts: Dict[int, int],
):
    """
    绘制全年四类振动窗口数占比饼状图。
    - 环形饼图弱化极端占比造成的视觉压迫
    - 图例横排置于饼图正下方，含中文标签、窗口数、占比
    """
    total_samples = sum(total_counts.values())
    cls_ids = list(range(4))
    counts = [total_counts.get(cls_id, 0) for cls_id in cls_ids]
    colors = [CLASS_COLORS[cls_id] for cls_id in cls_ids]

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE, constrained_layout=True)

    wedges, _ = ax.pie(
        counts,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=0.92,
        wedgeprops={
            "width": 0.38,
            "edgecolor": "white",
            "linewidth": 1.4,
        },
    )
    ax.text(
        0,
        0.04,
        "总窗口数",
        ha="center",
        va="center",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 5,
        color="#4a4a4a",
    )
    ax.text(
        0,
        -0.08,
        f"{total_samples:,}",
        ha="center",
        va="center",
        fontproperties=ENG_FONT,
        fontsize=FONT_SIZE - 3,
        color="#303030",
    )

    def _percentage_text(count: int) -> str:
        pct = count / total_samples * 100 if total_samples else 0.0
        return f"{pct:.3f}%" if 0 < pct < 0.01 else f"{pct:.2f}%"

    legend_labels = []
    for i, cls_id in enumerate(cls_ids):
        legend_labels.append(
            f"{CLASS_LABELS[cls_id]}  {counts[i]:,}（{_percentage_text(counts[i])}）"
        )
    legend = ax.legend(
        wedges,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=FONT_SIZE - 5,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#d0d0d0",
        columnspacing=1.5,
        handlelength=1.0,
        handletextpad=0.7,
        borderpad=0.5,
    )
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)
        text.set_fontsize(FONT_SIZE - 5)

    ax.set(aspect="equal")

    return fig, ax


def plot_minor_class_comparison(
    total_counts: Dict[int, int],
):
    """单独比较涡激共振、风雨振和其他振动。"""
    cls_ids = [1, 2, 3]
    labels = [CLASS_LABELS[cls_id] for cls_id in cls_ids]
    counts = [total_counts.get(cls_id, 0) for cls_id in cls_ids]
    minor_total = sum(counts)
    colors = [MINOR_CLASS_COLORS[cls_id] for cls_id in cls_ids]

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE, constrained_layout=True)
    y = list(range(len(cls_ids)))
    bars = ax.barh(
        y,
        counts,
        color=colors,
        height=0.52,
        edgecolor="white",
        linewidth=1.2,
    )

    max_count = max(counts) if counts else 0
    ax.set_xlim(0, max_count * 1.18 if max_count else 1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=CN_FONT, fontsize=FONT_SIZE - 3)
    ax.invert_yaxis()
    ax.set_xlabel("窗口数（个）", fontproperties=CN_FONT, fontsize=FONT_SIZE - 4)
    ax.set_title(
        "非随机振动类别对比",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 2,
        pad=12,
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for bar, count in zip(bars, counts):
        pct = count / minor_total * 100 if minor_total else 0.0
        ax.text(
            bar.get_width() + max_count * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}（{pct:.2f}%）",
            va="center",
            ha="left",
            fontproperties=ENG_FONT,
            fontsize=FONT_SIZE - 5,
            color="#303030",
        )

    ax.text(
        0.98,
        0.05,
        f"小类别总数：{minor_total:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 6,
        color="#666666",
    )

    return fig, ax


def main():
    """主函数"""
    records = load_inference_records()
    logger.info("用于敏感统计的推理记录：%d 条", len(records))

    total_counts = aggregate_total_counts(records)
    fig_pie, _ = plot_class_distribution_pie(total_counts)
    fig_minor, _ = plot_minor_class_comparison(total_counts)

    figures = [
        (fig_pie, "全年四类振动窗口占比"),
        (fig_minor, "非随机振动类别对比"),
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
