import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
import sys
from itertools import product

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    REC_FIG_SIZE, REC_FONT_SIZE, ENG_FONT, get_blue_color_map
)

MLP_SEARCH_RESULT_PATH = (
    r"F:\Research\Vibration Characteristics In Cable Vibration"
    r"\results\training_result\deep_learning_module"
    r"\search_best_hyperparams\mlp_search_result.json"
)

SAVE_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\文字工作\Paper\大论文\Img\Chapter2\Fig 2.39 深度学习最优超参搜索\1.png"

PARAM_GRID = {
    'batch_size':        [8, 16, 32],
    'learning_rate':     [1e-4, 1e-3, 5e-3],
    'weight_decay':      [1e-5, 1e-4],
    'gradient_clip_norm':[0.5, 1.0],
    'label_smoothing':   [0.0, 0.05, 0.1, 0.15],
}

_PARAM_KEYS = list(PARAM_GRID.keys())

LABEL_PADDING_SIZE = 12
TICK_FONT_SIZE  = REC_FONT_SIZE + LABEL_PADDING_SIZE
TITLE_FONT_SIZE = REC_FONT_SIZE + LABEL_PADDING_SIZE + 2
LABEL_FONT_SIZE = REC_FONT_SIZE + LABEL_PADDING_SIZE + 4


def load_results(result_path):
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['search_results']


def build_lookup(search_results):
    lookup = {}
    for result in search_results:
        p = result['params']
        key = tuple(p[k] for k in _PARAM_KEYS)
        lookup[key] = result['best_metric_value']
    return lookup


def plot_hyperp_contour(search_results, ploter):
    cmap = get_blue_color_map(style='gradient')

    bs_values  = PARAM_GRID['batch_size']
    lr_values  = PARAM_GRID['learning_rate']
    wd_values  = PARAM_GRID['weight_decay']
    gc_values  = PARAM_GRID['gradient_clip_norm']
    ls_values  = PARAM_GRID['label_smoothing']

    lookup = build_lookup(search_results)

    vmin = min(lookup.values())
    vmax = max(lookup.values())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 4 cols = (weight_decay, gradient_clip_norm) combos; 3 rows = batch_size
    fixed_combos = list(product(wd_values, gc_values))

    lr_idx = np.arange(len(lr_values))
    ls_idx = np.arange(len(ls_values))

    # 全局最优参数，用于定位唯一红星
    global_best = max(search_results, key=lambda r: r['best_metric_value'])
    gbp = global_best['params']
    gb_row = fixed_combos.index((gbp['weight_decay'], gbp['gradient_clip_norm']))
    gb_col = bs_values.index(gbp['batch_size'])
    gb_x   = lr_values.index(gbp['learning_rate'])
    gb_y   = ls_values.index(gbp['label_smoothing'])

    # 4 行 × 3 列：行 = (wd, gc) 组合；列 = batch_size
    figsize = (REC_FIG_SIZE[0] * 2, REC_FIG_SIZE[1] * 3)
    fig, axes = plt.subplots(4, 3, figsize=figsize)

    lr_labels = ['1e-4', '1e-3', '5e-3']
    ls_labels = ['0.00', '0.05', '0.10', '0.15']

    for row_idx, (wd, gc) in enumerate(fixed_combos):
        for col_idx, bs in enumerate(bs_values):
            ax = axes[row_idx, col_idx]

            # rows=label_smoothing, cols=learning_rate
            f1_mat = np.full((len(ls_values), len(lr_values)), np.nan)
            for i, ls in enumerate(ls_values):
                for j, lr in enumerate(lr_values):
                    key = (bs, lr, wd, gc, ls)
                    f1_mat[i, j] = lookup.get(key, np.nan)

            X, Y = np.meshgrid(lr_idx, ls_idx)

            ax.contourf(X, Y, f1_mat, cmap=cmap, levels=10, norm=norm, alpha=1.0)
            ax.contour(X, Y, f1_mat, levels=10, colors='black', linewidths=0.5, alpha=0.3)

            # 仅在全局最优所在子图绘制红星
            if row_idx == gb_row and col_idx == gb_col:
                ax.plot(gb_x, gb_y, 'r*', markersize=16, zorder=5)

            ax.set_xticks(lr_idx)
            ax.set_yticks(ls_idx)
            ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

            # 仅最底行显示 x 轴刻度标签
            if row_idx == len(fixed_combos) - 1:
                ax.set_xticklabels(lr_labels, fontsize=TICK_FONT_SIZE)
            else:
                ax.set_xticklabels([])

            # 仅最左列显示 y 轴刻度标签
            if col_idx == 0:
                ax.set_yticklabels(ls_labels, fontsize=TICK_FONT_SIZE)
            else:
                ax.set_yticklabels([])

            wd_str = f'{wd:.0e}'
            ax.set_title(
                f'BS={bs}  WD={wd_str}  GC={gc}',
                fontproperties=ENG_FONT, fontsize=TITLE_FONT_SIZE, pad=8
            )

    # 共享轴标签
    fig.supxlabel('Learning Rate', fontproperties=ENG_FONT, fontsize=LABEL_FONT_SIZE, y=0.01)
    fig.supylabel('Label Smoothing', fontproperties=ENG_FONT, fontsize=LABEL_FONT_SIZE, x=0.01)

    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Best Val F1 Score', fontproperties=ENG_FONT, fontsize=LABEL_FONT_SIZE - 2)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.suptitle(
        'MLP Hyperparameter Search: Learning Rate vs Label Smoothing',
        fontproperties=ENG_FONT, fontsize=LABEL_FONT_SIZE, y=1.01
    )
    plt.tight_layout(rect=[0.03, 0.04, 0.91, 1.0])

    ploter.figs.append(fig)


def main():
    print("=" * 80)
    print("开始绘制MLP超参数搜索等高线图")
    print("=" * 80)

    if not os.path.exists(MLP_SEARCH_RESULT_PATH):
        raise FileNotFoundError(f"结果文件不存在：{MLP_SEARCH_RESULT_PATH}")

    search_results = load_results(MLP_SEARCH_RESULT_PATH)
    print(f"加载了 {len(search_results)} 条搜索结果")

    ploter = PlotLib()
    plot_hyperp_contour(search_results, ploter)

    print(f"生成了 {len(ploter.figs)} 张图")

    if SAVE_PATH:
        ploter.figs[0].savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
        print(f"图像已保存至：{SAVE_PATH}")

    ploter.show()

    print("\n" + "=" * 80)
    print("MLP超参数等高线图绘制完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
