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

from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, ENG_FONT, FONT_SIZE,
    get_blue_color_map,
)
from src.statistics.run import load_mode_data, _build_var_names
from src.config.statistics.config import load_config


# ==================== 常量配置 ====================
class Config:
    N_HIST_BINS = 38
    P_CLIP      = (1.0, 99.0)       # 直方图 x 范围百分位裁剪

    FIG_SIZE    = (6.0, 4.6)

    TICK_SIZE   = FONT_SIZE - 3
    PDF_LW      = 1.8

    _pal        = get_blue_color_map(
        style='discrete', start_map_index=1, end_map_index=5
    ).colors
    HIST_COLOR  = _pal[1]           # 浅蓝：直方图
    PDF_COLOR   = _pal[3]           # 深蓝/紫：拟合曲线

    RESULTS_FILE = (
        project_root / "results" / "statistics" / "normal_vib_mode_analysis.json"
    )

    TARGETS = [
        ("freq_in_1", "面内 F1"),
        ("freq_out_1", "面外 F1"),
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
    ax.set_xlabel('频率 (Hz)', fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)
    ax.set_ylabel('概率密度', fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)
    ax.grid(True, color='gray', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


# ==================== 单图布局 ====================
def plot_f1_marginal(
    matrix: np.ndarray,
    var_names: list[str],
    marginals: dict,
    var_name: str,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    col_idx = var_names.index(var_name)
    _plot_one(ax, matrix[:, col_idx], marginals.get(var_name))
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=10)
    fig.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 图3.14 随机振动 F1 边缘分布拟合（面内 / 面外）")
    print("=" * 80)

    print("\n[步骤 1/2] 加载原始数据与拟合结果...")
    matrix, var_names, marginals = load_data()
    n_fitted = sum(1 for v in marginals.values() if v is not None)
    print(f"  ✓ 数据矩阵：{matrix.shape}，有效拟合变量：{n_fitted} / {len(var_names)}")

    print("\n[步骤 2/2] 绘制 F1 面内/面外独立图...")
    figures = [
        (plot_f1_marginal(matrix, var_names, marginals, var_name, title), title)
        for var_name, title in Config.TARGETS
    ]
    print(f"  ✓ 图像生成完成：{len(figures)} 张")

    print("=" * 80)
    for slot, (fig, title) in enumerate(figures):
        web_push(
            fig,
            page="fig3_17 F1边缘分布",
            slot=slot,
            title=title,
            page_cols=2 if slot == 0 else None,
        )
    print("✓ 已推送到 WebUI")


if __name__ == "__main__":
    main()
