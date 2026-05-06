import sys
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE,
    VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR, ABOVE_THRESHOLD_COLOR,
)

_chapter4_dir = str(Path(__file__).parent)
if _chapter4_dir not in sys.path:
    sys.path.insert(0, _chapter4_dir)
from _viv_pipeline import load_latest_result


# ==================== 常量配置 ====================
_VIV_CLASS = 1

class Config:
    FIG_SIZE = SQUARE_FIG_SIZE

    COLOR_INPLANE  = VIV_INPLANE_COLOR     # '#8074C8' 深紫：仅面内
    COLOR_OUTPLANE = VIV_OUTPLANE_COLOR   # '#E3625D' 珊瑚红：仅面外
    COLOR_BOTH     = ABOVE_THRESHOLD_COLOR  # '#7895C1' 钢蓝：同时振动

    DL_RESULT_GLOB   = project_root / "results" / "identification_result"         / "res_cnn_full_dataset_*.json"
    MECC_RESULT_GLOB = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"


# ==================== 分类统计 ====================
def classify_from_result(result: dict) -> dict:
    pred_in  = {int(k): int(v) for k, v in result.get("predictions_inplane",  {}).items()}
    pred_out = {int(k): int(v) for k, v in result.get("predictions_outplane", {}).items()}

    all_indices = set(pred_in.keys()) | set(pred_out.keys())

    n_inplane  = 0
    n_outplane = 0
    n_both     = 0

    for idx in all_indices:
        is_in  = pred_in.get(idx,  0) == _VIV_CLASS
        is_out = pred_out.get(idx, 0) == _VIV_CLASS

        if is_in and is_out:
            n_both += 1
        elif is_in:
            n_inplane += 1
        elif is_out:
            n_outplane += 1

    total = n_inplane + n_outplane + n_both
    print(f"  面内 VIV={n_inplane}，面外 VIV={n_outplane}，同时={n_both}，合计={total}")
    return {
        "inplane":  n_inplane,
        "outplane": n_outplane,
        "both":     n_both,
        "total":    total,
    }


# ==================== 绘图 ====================
def plot_pie(counts: dict, title: str) -> plt.Figure:
    n_in  = counts["inplane"]
    n_out = counts["outplane"]
    n_b   = counts["both"]
    total = counts["total"]

    if total == 0:
        raise ValueError("无有效 VIV 样本，无法绘制饼图")

    all_slices = [
        (n_in,  '面内涡激共振', Config.COLOR_INPLANE),
        (n_out, '面外涡激共振', Config.COLOR_OUTPLANE),
        (n_b,   '同时振动',     Config.COLOR_BOTH),
    ]
    non_zero = [(s, l, c) for s, l, c in all_slices if s > 0]
    sizes, labels_raw, colors = zip(*non_zero)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        explode=[0.05] * len(sizes),
        startangle=45,
        shadow=True,
    )

    legend_labels = [
        f"{lbl}  {sz:,}（{sz / total * 100:.2f}%）"
        for lbl, sz in zip(labels_raw, sizes)
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

    ax.set_title(
        f"{title}\n（共 {total} 个 VIV 窗口）",
        fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=16,
    )
    plt.tight_layout()
    return fig


# ==================== 主绘图入口 ====================
def _classify_and_push(result: dict, page_name: str, pie_title: str, slot: int):
    counts = classify_from_result(result)
    fig    = plot_pie(counts, pie_title)
    web_push(fig, page=page_name, slot=slot, title=pie_title,
             page_cols=2 if slot == 0 else None)
    print(f"  ✓ 饼图已推送到页面「{page_name}」slot={slot}")


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振面内/面外/同时振动占比统计（DL vs MECC）")
    print("=" * 80)

    page = "fig4_x VIV面内外占比"

    print("\n[步骤1] 加载 DL 识别结果并统计...")
    dl_result = load_latest_result(Config.DL_RESULT_GLOB)
    _classify_and_push(dl_result, page, "DL 识别 VIV 振动方向分布", slot=0)

    print("\n[步骤2] 加载 MECC 识别结果并统计...")
    mecc_result = load_latest_result(Config.MECC_RESULT_GLOB)
    _classify_and_push(mecc_result, page, "MECC 识别 VIV 振动方向分布", slot=1)

    print("\n" + "=" * 80)
    print("饼图已推送完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
