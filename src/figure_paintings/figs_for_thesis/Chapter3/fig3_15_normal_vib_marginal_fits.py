import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, ENG_FONT, FONT_SIZE,
    get_blue_color_map,
)
from src.statistics.run import load_mode_data, _build_var_names
from src.config.statistics.config import load_config


# ==================== 常量配置 ====================
class Config:
    N_MODES     = 8
    N_ROWS      = 4
    N_COLS      = 8
    N_HIST_BINS = 38
    P_CLIP      = (1.0, 99.0)       # 直方图 x 范围百分位裁剪

    FIG_W       = 20.0
    FIG_H       = 9.0
    L_MARGIN    = 0.11              # 为行标签留白
    R_MARGIN    = 0.995
    T_MARGIN    = 0.975
    B_MARGIN    = 0.07
    HSPACE      = 0.38
    WSPACE      = 0.26

    TICK_SIZE   = 5.5
    PDF_LW      = 1.6

    _pal        = get_blue_color_map(
        style='discrete', start_map_index=1, end_map_index=5
    ).colors
    HIST_COLOR  = _pal[1]           # 浅蓝：直方图
    PDF_COLOR   = _pal[3]           # 深蓝/紫：拟合曲线

    RESULTS_FILE = (
        project_root / "results" / "statistics" / "normal_vib_mode_analysis.json"
    )

    ROW_LABELS = [
        '面内主频（Hz）',
        '面内能量占比',
        '面外主频（Hz）',
        '面外能量占比',
    ]


# ==================== 数据加载 ====================
def load_data() -> tuple[np.ndarray, list[str], dict]:
    if not Config.RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"拟合结果文件不存在，请先运行 src/statistics/run.py：\n{Config.RESULTS_FILE}"
        )

    cfg        = load_config()
    matrix, _ = load_mode_data(cfg)
    var_names  = _build_var_names(cfg.n_modes)

    with open(Config.RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    return matrix, var_names, results["marginals"]


# ==================== 单子图绘制 ====================
def _plot_one(ax: plt.Axes, col_data: np.ndarray, marginal: dict | None) -> None:
    valid = col_data[np.isfinite(col_data) & (col_data > 0)]
    if len(valid) < 10:
        ax.set_visible(False)
        return

    p_lo, p_hi = np.percentile(valid, Config.P_CLIP)
    clipped    = valid[(valid >= p_lo) & (valid <= p_hi)]

    ax.hist(
        clipped, bins=Config.N_HIST_BINS, density=True,
        color=Config.HIST_COLOR, alpha=0.70, edgecolor='none',
    )

    if marginal is not None:
        best   = marginal["best"]
        x_pdf  = np.linspace(p_lo, p_hi, 300)

        if best["form"].startswith("gmm"):
            p       = best["params"]
            weights = p["weights"]
            means   = p["means"]
            stds    = [v ** 0.5 for v in p["variances"]]
            pdf = sum(
                w * scipy_stats.norm.pdf(x_pdf, mu, sig)
                for w, mu, sig in zip(weights, means, stds)
            )
        else:
            dist   = getattr(scipy_stats, best["form"])
            params = list(best["params"].values())
            pdf    = dist.pdf(x_pdf, *params)

        ax.plot(x_pdf, pdf, color=Config.PDF_COLOR,
                linewidth=Config.PDF_LW, zorder=3)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(2, prune='both'))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(2, prune='both'))
    ax.tick_params(axis='both', labelsize=Config.TICK_SIZE,
                   pad=1.5, length=2.5, width=0.6)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


# ==================== 主图布局 ====================
def plot_marginal_fits(
    matrix: np.ndarray,
    var_names: list[str],
    marginals: dict,
) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.N_ROWS, Config.N_COLS,
        figsize=(Config.FIG_W, Config.FIG_H),
    )
    plt.subplots_adjust(
        left=Config.L_MARGIN,   right=Config.R_MARGIN,
        top=Config.T_MARGIN,    bottom=Config.B_MARGIN,
        hspace=Config.HSPACE,   wspace=Config.WSPACE,
    )

    for j, name in enumerate(var_names):
        row = j // Config.N_COLS
        col = j % Config.N_COLS
        _plot_one(axes[row, col], matrix[:, j], marginals.get(name))

    # ── 行标签（左侧，旋转 90°）──────────────────────────────────
    for r, label in enumerate(Config.ROW_LABELS):
        pos      = axes[r, 0].get_position()
        y_center = pos.y0 + pos.height / 2
        fig.text(
            Config.L_MARGIN * 0.48, y_center,
            label,
            ha='center', va='center', rotation=90,
            fontproperties=CN_FONT, fontsize=FONT_SIZE - 4,
        )

    # ── 全局 y 轴标注（最左侧）────────────────────────────────────
    fig.text(
        Config.L_MARGIN * 0.08, 0.5,
        '概率密度',
        ha='center', va='center', rotation=90,
        fontproperties=CN_FONT, fontsize=FONT_SIZE - 2,
    )

    # ── 全局 x 轴标注（底部居中）──────────────────────────────────
    fig.text(
        (Config.L_MARGIN + Config.R_MARGIN) / 2, Config.B_MARGIN * 0.28,
        '取值',
        ha='center', va='center',
        fontproperties=CN_FONT, fontsize=FONT_SIZE - 2,
    )

    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 图3.14 随机振动前8阶主频/能量边缘分布拟合（4×8 网格）")
    print("=" * 80)

    print("\n[步骤 1/2] 加载原始数据与拟合结果...")
    matrix, var_names, marginals = load_data()
    n_fitted = sum(1 for v in marginals.values() if v is not None)
    print(f"  ✓ 数据矩阵：{matrix.shape}，有效拟合变量：{n_fitted} / {len(var_names)}")

    print("\n[步骤 2/2] 绘制 4×8 子图...")
    fig = plot_marginal_fits(matrix, var_names, marginals)
    print("  ✓ 图像生成完成")

    print("=" * 80)
    ploter = PlotLib()
    ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
