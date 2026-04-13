import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.statistics.fitting import fit_distribution, fit_gmm
from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, ENG_FONT,
    REC_FIG_SIZE, REC_FONT_SIZE,
    NORMAL_VIB_COLOR, NORMAL_EDGE_COLOR,
)

FONT_SIZE = REC_FONT_SIZE


# ==================== 常量配置 ====================
class Config:
    RANDOM_SEED = 42
    N_SAMPLES   = 2000
    N_BINS      = 45

    # 单峰模拟：随机振动 RMS（单一 Gamma，明显右偏）
    UNIMODAL_SHAPE = 3.5
    UNIMODAL_SCALE = 0.13   # 均值 ≈ 0.455，峰值在 0.3 附近

    # 双峰模拟：低幅背景振动 + 中幅阵风激励
    # 第一峰：Gamma(5, 0.06) → 均值 ≈ 0.30
    BIMODAL_SHAPE1 = 5.0
    BIMODAL_SCALE1 = 0.06
    BIMODAL_SHIFT1 = 0.0
    BIMODAL_RATIO1 = 0.55
    # 第二峰：Gamma(5, 0.06) + 偏移 → 均值 ≈ 0.72
    BIMODAL_SHAPE2 = 5.0
    BIMODAL_SCALE2 = 0.06
    BIMODAL_SHIFT2 = 0.42
    BIMODAL_RATIO2 = 0.45

    DISTRIBUTIONS = [
        ("norm",        "正态分布",     '#8074C8'),
        ("lognorm",     "对数正态分布", '#E3625D'),
        ("gamma",       "伽马分布",     '#7895C1'),
        ("weibull_min", "威布尔分布",   '#B54764'),
    ]
    GMM_N_COMPONENTS = 2
    GMM_COLOR        = '#303030'
    GMM_LABEL        = f'高斯混合（{GMM_N_COMPONENTS} 分量）'

    LINEWIDTH  = 2.0
    HIST_ALPHA = 0.55
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LW    = 0.5
    GRID_LS    = '--'

    FIG_SIZE = REC_FIG_SIZE


# ==================== 数据模拟 ====================
def simulate_unimodal(seed: int = Config.RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.gamma(Config.UNIMODAL_SHAPE, Config.UNIMODAL_SCALE, Config.N_SAMPLES)


def simulate_bimodal(seed: int = Config.RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n1  = int(Config.N_SAMPLES * Config.BIMODAL_RATIO1)
    n2  = Config.N_SAMPLES - n1
    pop1 = rng.gamma(Config.BIMODAL_SHAPE1, Config.BIMODAL_SCALE1, n1) + Config.BIMODAL_SHIFT1
    pop2 = rng.gamma(Config.BIMODAL_SHAPE2, Config.BIMODAL_SCALE2, n2) + Config.BIMODAL_SHIFT2
    return np.concatenate([pop1, pop2])


# ==================== GMM PDF ====================
def gmm_pdf(x: np.ndarray, result) -> np.ndarray:
    weights   = result.params["weights"]
    means     = result.params["means"]
    variances = result.params["variances"]
    pdf = np.zeros_like(x, dtype=float)
    for w, mu, var in zip(weights, means, variances):
        pdf += w * stats.norm.pdf(x, loc=mu, scale=np.sqrt(var))
    return pdf


# ==================== 网格辅助 ====================
def _ax_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LW, linestyle=Config.GRID_LS)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 4)


# ==================== 通用全量拟合绘图 ====================
def _plot_all_fits(data: np.ndarray, legend_loc: str = 'upper right') -> plt.Figure:
    """对任意一维正值数据绘制长方形全量拟合图（直方图 + 5 种分布 + GMM）。"""
    x_min = max(data.min() - 0.02, 0.0)
    x_max = data.max() + 0.05
    x     = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    ax.hist(
        data,
        bins=Config.N_BINS,
        density=True,
        color=NORMAL_VIB_COLOR,
        edgecolor=NORMAL_EDGE_COLOR,
        linewidth=0.5,
        alpha=Config.HIST_ALPHA,
        zorder=1,
    )

    legend_handles = [
        mpatches.Patch(
            facecolor=NORMAL_VIB_COLOR, edgecolor=NORMAL_EDGE_COLOR,
            alpha=Config.HIST_ALPHA, label='模拟 RMS 样本',
        )
    ]

    for dist_name, cn_label, color in Config.DISTRIBUTIONS:
        result       = fit_distribution(data, distribution=dist_name)
        dist_obj     = getattr(stats, dist_name)
        fitted_params = list(result.params.values())
        pdf_vals     = dist_obj.pdf(x, *fitted_params)
        aic_str      = f"{result.aic:.0f}" if result.aic is not None else "-"
        full_label   = f"{cn_label}  (AIC={aic_str})"

        ax.plot(x, pdf_vals, color=color, linewidth=Config.LINEWIDTH,
                zorder=3)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=Config.LINEWIDTH,
                       label=full_label)
        )

    gmm_result     = fit_gmm(data, n_components=Config.GMM_N_COMPONENTS)
    gmm_vals       = gmm_pdf(x, gmm_result)
    aic_gmm        = f"{gmm_result.aic:.0f}" if gmm_result.aic is not None else "-"
    gmm_full_label = f"{Config.GMM_LABEL}  (AIC={aic_gmm})"

    ax.plot(x, gmm_vals, color=Config.GMM_COLOR, linewidth=Config.LINEWIDTH,
            linestyle='--', zorder=4)
    legend_handles.append(
        plt.Line2D([0], [0], color=Config.GMM_COLOR, linewidth=Config.LINEWIDTH,
                   linestyle='--', label=gmm_full_label)
    )

    ax.set_xlabel(r'加速度 RMS ($\mathrm{m/s^2}$)', labelpad=8,
                  fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('概率密度', labelpad=8, fontproperties=CN_FONT, fontsize=FONT_SIZE)

    legend = ax.legend(handles=legend_handles, framealpha=0.85,
                       loc=legend_loc, fontsize=FONT_SIZE - 2)
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)
        text.set_fontsize(FONT_SIZE - 2)

    _ax_grid(ax)
    fig.tight_layout()
    return fig


def plot_unimodal_fitting() -> plt.Figure:
    data = simulate_unimodal()
    return _plot_all_fits(data, legend_loc='upper right')


def plot_bimodal_fitting() -> plt.Figure:
    data = simulate_bimodal()
    return _plot_all_fits(data, legend_loc='best')


# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("Fig 3-5  边缘概率密度拟合对比（单峰 & 双峰模拟振动 RMS）")
    print("=" * 70)

    uni_data = simulate_unimodal()
    bi_data  = simulate_bimodal()

    print(f"\n单峰样本：均值={uni_data.mean():.4f}  σ={uni_data.std():.4f}")
    print(f"双峰样本：均值={bi_data.mean():.4f}  σ={bi_data.std():.4f}")

    print("\n[图 A] 单峰分布全量拟合...")
    fig_a = plot_unimodal_fitting()

    print("\n[图 B] 双峰分布全量拟合...")
    fig_b = plot_bimodal_fitting()

    print("\n共生成 2 张长方形拟合图")
    print("=" * 70)

    ploter = PlotLib()
    ploter.figs.append(fig_a)
    ploter.figs.append(fig_b)
    ploter.show()


if __name__ == "__main__":
    main()
