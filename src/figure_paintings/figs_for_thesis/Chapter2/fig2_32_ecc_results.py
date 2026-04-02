import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, ENG_FONT, get_blue_color_map
)

ECC_RESULT_PATH = (
    r"F:\Research\Vibration Characteristics In Cable Vibration"
    r"\results\ecc_results\ecc_search_results.json"
)
SAVE_PATH = r""

SIGMA  = 0.1
LABELS = ['Normal Vibration', 'VIV']

CMAP       = get_blue_color_map()
CMAP_CURVE = get_blue_color_map(style='gradient')


def load_results(result_path: str) -> dict:
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_f1_curve(search_results: list, best_params: dict, ploter: PlotLib):
    colors      = CMAP_CURVE(np.linspace(0, 1, 256))
    curve_color = colors[200]

    threshold_values = [r['params']['threshold'] for r in search_results]
    f1_scores        = np.array([r['f1'] for r in search_results])
    x_positions      = np.arange(len(threshold_values))
    x_labels         = [str(t) for t in threshold_values]

    best_t   = best_params['threshold']
    best_idx = threshold_values.index(best_t)
    best_f1  = f1_scores[best_idx]

    print(f"\n【ECC 最优参数】")
    print(f"  threshold = {best_t}")
    print(f"  sigma     = {SIGMA}  (固定)")
    print(f"  F1 Score  = {best_f1:.4f}")

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    ax.plot(x_positions, f1_scores, color=curve_color, linewidth=3,
            marker='o', markersize=8, zorder=3)
    ax.fill_between(x_positions, f1_scores, alpha=0.25, color=curve_color)
    ax.axvline(x=best_idx, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Best threshold={best_t}')

    ax.set_xlabel(r"$C'_{ECC}$ (threshold)", labelpad=10, fontproperties=ENG_FONT,
                  fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel('F1 Score', labelpad=10, fontproperties=ENG_FONT,
                  fontsize=SQUARE_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=SQUARE_FONT_SIZE - 4)
    ax.set_ylim((0, 1.05))
    ax.tick_params(axis='y', labelsize=SQUARE_FONT_SIZE - 4)
    ax.legend(prop=ENG_FONT, fontsize=SQUARE_FONT_SIZE - 4, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    ploter.figs.append(fig)


def plot_confusion_matrix(best_params: dict, best_metrics: dict, ploter: PlotLib):
    cm = np.array(best_metrics['confusion_matrix'])

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=CMAP,
        xticklabels=LABELS, yticklabels=LABELS,
        ax=ax, cbar=False, annot_kws={'size': SQUARE_FONT_SIZE}
    )
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT,
                  fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel('True Label', labelpad=10, fontproperties=ENG_FONT,
                  fontsize=SQUARE_FONT_SIZE)
    ax.set_title(
        f"threshold={best_params['threshold']},  sigma={SIGMA}\n"
        f"Weighted F1: {best_metrics['weighted']['F1']:.4f}",
        pad=16, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE - 2
    )
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)
    plt.tight_layout()
    ploter.figs.append(fig)


def main():
    print("=" * 80)
    print("ECC 参数搜索结果可视化")
    print("=" * 80)

    if not os.path.exists(ECC_RESULT_PATH):
        raise FileNotFoundError(f"结果文件不存在：{ECC_RESULT_PATH}")

    data           = load_results(ECC_RESULT_PATH)
    best_params    = data['best_params']
    search_results = data['search_results']
    param_metrics  = data['param_metrics']

    print(f"共 {len(search_results)} 个参数组合")
    print(f"最优参数：threshold={best_params['threshold']}, sigma={SIGMA}（固定）")

    best_key     = str(best_params['threshold'])
    best_metrics = param_metrics[best_key]
    print(f"加权 Precision : {best_metrics['weighted']['Precision']:.4f}")
    print(f"加权 Recall    : {best_metrics['weighted']['Recall']:.4f}")
    print(f"加权 F1 Score  : {best_metrics['weighted']['F1']:.4f}")

    ploter = PlotLib()
    plot_f1_curve(search_results, best_params, ploter)
    plot_confusion_matrix(best_params, best_metrics, ploter)

    print(f"\n生成了 {len(ploter.figs)} 张图")

    if SAVE_PATH:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        for i, fig in enumerate(ploter.figs):
            base, ext = os.path.splitext(SAVE_PATH)
            save_name = f"{base}_{i}{ext}" if len(ploter.figs) > 1 else SAVE_PATH
            fig.savefig(save_name, dpi=300, bbox_inches='tight')
            print(f"图像已保存至：{save_name}")

    ploter.show()

    print("\n" + "=" * 80)
    print("ECC 结果绘图完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
