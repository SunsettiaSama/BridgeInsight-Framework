import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, ENG_FONT, CN_FONT,
    get_viridis_color_map, get_blue_color_map
)

ECC_RESULT_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\results\ecc_results\ecc_search_results.json"
SIGMA    = 0.1
CMAP     = get_blue_color_map()
CMAP_CURVE = get_blue_color_map(style='gradient')

LABELS = ['Normal Vibration', 'VIV']


# ===================== 数据加载 =====================
def load_ecc_results(result_path: str) -> dict:
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ===================== F1 曲线图（threshold 一维）=====================
def plot_ecc_f1_curve(search_results: list, best_params: dict, ploter: PlotLib):
    colors = CMAP_CURVE(np.linspace(0, 1, 256))
    curve_color = colors[255]

    threshold_values = [r['params']['threshold'] for r in search_results]
    f1_scores = np.array([r['f1'] for r in search_results])

    x_positions = np.arange(len(threshold_values))
    param_str_values = [str(t) for t in threshold_values]

    best_t  = best_params['threshold']
    best_idx = threshold_values.index(best_t)
    best_f1  = f1_scores[best_idx]

    print(f"\n【ECC 最优参数】")
    print(f"  threshold = {best_t}")
    print(f"  sigma     = {SIGMA}  (固定)")
    print(f"  F1 Score  = {best_f1:.4f}")

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    ax.plot(x_positions, f1_scores, color=curve_color, linewidth=3, marker='o', markersize=8)
    ax.fill_between(x_positions, f1_scores, alpha=0.3, color=curve_color)

    ax.set_xlabel(r"$C'_{ECC}$ (threshold)", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('F1 Score', labelpad=10, fontproperties=ENG_FONT)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(param_str_values, rotation=45, ha='right', fontproperties=ENG_FONT)
    ax.set_ylim((0, 1))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelsize=SQUARE_FONT_SIZE - 4)

    plt.tight_layout()
    ploter.figs.append(fig)


# ===================== 混淆矩阵 =====================
def plot_ecc_confusion_matrix(best_params: dict, best_metrics: dict, ploter: PlotLib):
    cm = np.array(best_metrics['confusion_matrix'])

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=CMAP,
        xticklabels=LABELS, yticklabels=LABELS,
        ax=ax, cbar=False, annot_kws={'size': 20}
    )
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('True Label',      labelpad=10, fontproperties=ENG_FONT)
    ax.set_title(
        f"threshold={best_params['threshold']}, sigma={SIGMA}\n"
        f"Weighted F1: {best_metrics['weighted']['F1']:.3f}",
        pad=20, fontproperties=ENG_FONT
    )
    plt.tight_layout()
    ploter.figs.append(fig)


# ===================== 主函数 =====================
def main():
    print("=" * 80)
    print("ECC 参数搜索结果可视化（从缓存 JSON 加载）")
    print("=" * 80)

    if not os.path.exists(ECC_RESULT_PATH):
        print(f"结果文件不存在：{ECC_RESULT_PATH}")
        return

    data           = load_ecc_results(ECC_RESULT_PATH)
    best_params    = data['best_params']
    search_results = data['search_results']
    param_metrics  = data['param_metrics']

    print(f"\n共 {len(search_results)} 个参数组合")
    print(f"最优参数：threshold={best_params['threshold']}, sigma={SIGMA}（硬编码）")

    best_key     = str(best_params['threshold'])
    best_metrics = param_metrics[best_key]

    print(f"加权 Precision : {best_metrics['weighted']['Precision']:.4f}")
    print(f"加权 Recall    : {best_metrics['weighted']['Recall']:.4f}")
    print(f"加权 F1 Score  : {best_metrics['weighted']['F1']:.4f}")

    ploter = PlotLib()
    plot_ecc_f1_curve(search_results, best_params, ploter)
    plot_ecc_confusion_matrix(best_params, best_metrics, ploter)

    print(f"\n生成了 {len(ploter.figs)} 张图")
    ploter.show()

    print("\n" + "=" * 80)
    print("ECC 结果绘图完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
