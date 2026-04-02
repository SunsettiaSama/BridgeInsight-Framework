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

MECC_RESULT_PATH = (
    r"F:\Research\Vibration Characteristics In Cable Vibration"
    r"\results\mecc_results\mecc_search_results.json"
)
SAVE_PATH = r""

LABELS = ['Normal', 'VIV']

CMAP         = get_blue_color_map()
CMAP_CONTOUR = get_blue_color_map(style='gradient')


def load_results(result_path: str) -> dict:
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_f1_contour(search_results: list, best_params: dict, ploter: PlotLib):
    k_viv_vals = sorted(set(r['params']['k_viv'] for r in search_results))
    C_viv_vals = sorted(set(r['params']['C_viv'] for r in search_results))

    f1_matrix = np.full((len(k_viv_vals), len(C_viv_vals)), np.nan)
    for r in search_results:
        ki = k_viv_vals.index(r['params']['k_viv'])
        ci = C_viv_vals.index(r['params']['C_viv'])
        f1_matrix[ki, ci] = r['f1']

    X, Y = np.meshgrid(C_viv_vals, k_viv_vals)

    best_k   = best_params['k_viv']
    best_C   = best_params['C_viv']
    best_ki  = k_viv_vals.index(best_k)
    best_ci  = C_viv_vals.index(best_C)
    best_f1  = f1_matrix[best_ki, best_ci]

    print(f"\n【MECC 最优参数】")
    print(f"  k_viv   = {best_k}")
    print(f"  C_viv   = {best_C}")
    print(f"  Best Weighted F1 = {best_f1:.4f}")

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    cf = ax.contourf(X, Y, f1_matrix, cmap=CMAP_CONTOUR, levels=20, alpha=1.0)
    ax.contour(X, Y, f1_matrix, levels=20, colors='black', linewidths=0.5, alpha=0.3)

    ax.scatter([best_C], [best_k], color='red', s=120, zorder=5, marker='*',
               label=f'Best: $k$={best_k}, $C$={best_C}\nF1={best_f1:.4f}')
    ax.legend(prop=ENG_FONT, fontsize=SQUARE_FONT_SIZE - 4, framealpha=0.9, loc='upper right')

    ax.set_xlabel(r"$C_{MECC}$", labelpad=10, fontproperties=ENG_FONT,
                  fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel(r"$k_{MECC}$", labelpad=10, fontproperties=ENG_FONT,
                  fontsize=SQUARE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label('Weighted F1 Score', fontproperties=ENG_FONT,
                   fontsize=SQUARE_FONT_SIZE - 2)
    cbar.ax.tick_params(labelsize=SQUARE_FONT_SIZE - 6)

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
        f"$k_{{MECC}}$={best_params['k_viv']},  $C_{{MECC}}$={best_params['C_viv']}\n"
        f"Weighted F1: {best_metrics['weighted']['F1']:.4f}",
        pad=16, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE - 2
    )
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)
    plt.tight_layout()
    ploter.figs.append(fig)


def main():
    print("=" * 80)
    print("MECC 参数搜索结果可视化")
    print("=" * 80)

    if not os.path.exists(MECC_RESULT_PATH):
        raise FileNotFoundError(f"结果文件不存在：{MECC_RESULT_PATH}")

    data           = load_results(MECC_RESULT_PATH)
    best_params    = data['best_params']
    search_results = data['search_results']
    param_metrics  = data['param_metrics']

    print(f"共 {len(search_results)} 个参数组合")
    print(f"最优参数：k_viv={best_params['k_viv']}, C_viv={best_params['C_viv']}")

    best_key     = f"{best_params['k_viv']}_{best_params['C_viv']}"
    best_metrics = param_metrics[best_key]
    print(f"加权 Precision : {best_metrics['weighted']['Precision']:.4f}")
    print(f"加权 Recall    : {best_metrics['weighted']['Recall']:.4f}")
    print(f"加权 F1 Score  : {best_metrics['weighted']['F1']:.4f}")

    ploter = PlotLib()
    plot_f1_contour(search_results, best_params, ploter)
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
    print("MECC 结果绘图完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
