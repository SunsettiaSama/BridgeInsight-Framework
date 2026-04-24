import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, REC_FIG_SIZE,
    VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR,
)


# ==================== 常量配置 ====================
class Config:
    VIV_CLASS_ID = 1

    N_BINS      = 100
    N_BINS_TAIL = 60

    HIST_X_PERCENTILE = 95.0
    HIST_LOG_SCALE    = False

    SCATTER_MAX_POINTS = 100_000
    SCATTER_SEED = 42

    # VIV 振幅通常大于随机振动，适当扩大 zoom 范围
    SCATTER_ZOOM_XLIM = (0.0, 2.0)
    SCATTER_ZOOM_YLIM = (0.0, 5.0)
    SCATTER_ZOOM_FIGSIZE = (5, 5)

    TAIL_X_MAX = max(SCATTER_ZOOM_XLIM[1], SCATTER_ZOOM_YLIM[1])  # 5.0

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'

    INPLANE_COLOR  = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    BAR_ALPHA  = 0.72
    SCATTER_SIZE  = 6
    SCATTER_ALPHA = 0.35

    OUTLIER_SIZE  = 25
    OUTLIER_ALPHA = 0.55
    OUTLIER_COLOR = plt.cm.YlOrRd(0.72)

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_1_viv"
    )


# ==================== 数据加载 ====================
def load_rms_from_enriched_stats() -> dict:
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    inplane_rms:  list[float] = []
    outplane_rms: list[float] = []

    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            ts_in  = sample.get("time_stats_inplane")  or {}
            ts_out = sample.get("time_stats_outplane") or {}
            rms_in  = ts_in.get("rms")
            rms_out = ts_out.get("rms")
            if rms_in is not None and rms_out is not None:
                inplane_rms.append(rms_in)
                outplane_rms.append(rms_out)

    return {
        "inplane_rms":  np.array(inplane_rms,  dtype=np.float64),
        "outplane_rms": np.array(outplane_rms, dtype=np.float64),
    }


# ==================== 工具 ====================
def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _grouped_bars(ax, centers, counts_in, counts_out, width):
    half = width * 0.46
    ax.bar(centers - half / 2, counts_in,  width=half,
           color=Config.INPLANE_COLOR,  alpha=Config.BAR_ALPHA,
           label='面内', edgecolor='none')
    ax.bar(centers + half / 2, counts_out, width=half,
           color=Config.OUTPLANE_COLOR, alpha=Config.BAR_ALPHA,
           label='面外', edgecolor='none')


def _add_legend(ax):
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for t in leg.get_texts():
        t.set_fontproperties(CN_FONT)


# ==================== 图1a：主体分布直方图 ====================
def plot_rms_histogram(stats: dict, x_split: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    in_rms  = stats["inplane_rms"]
    out_rms = stats["outplane_rms"]

    bins    = np.linspace(0, x_split, Config.N_BINS + 1)
    width   = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2

    counts_in,  _ = np.histogram(in_rms,  bins=bins)
    counts_out, _ = np.histogram(out_rms, bins=bins)

    _grouped_bars(ax, centers, counts_in, counts_out, width)

    if Config.HIST_LOG_SCALE:
        ax.set_yscale('log')
        ylabel = '样本数（个，对数坐标）'
    else:
        ylabel = '样本数（个）'

    ax.set_xlim(0, x_split)
    ax.set_xlabel(r'RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振加速度 RMS 主体分布', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 图1b：尾部分布直方图 ====================
def plot_rms_tail_histogram(stats: dict, x_split: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    in_rms  = stats["inplane_rms"]
    out_rms = stats["outplane_rms"]

    x_tail_max = Config.TAIL_X_MAX
    in_tail  = in_rms[(in_rms   > x_split) & (in_rms   <= x_tail_max)]
    out_tail = out_rms[(out_rms > x_split) & (out_rms  <= x_tail_max)]

    bins    = np.linspace(x_split, x_tail_max, Config.N_BINS_TAIL + 1)
    width   = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2

    counts_in,  _ = np.histogram(in_tail,  bins=bins)
    counts_out, _ = np.histogram(out_tail, bins=bins)

    _grouped_bars(ax, centers, counts_in, counts_out, width)

    ax.set_xlim(x_split, x_tail_max)
    ax.set_xlabel(r'RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('样本数（个）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振加速度 RMS 尾部分布', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 图2：全量散点（面内 vs 面外 RMS） ====================
def plot_rms_scatter(stats: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    in_rms  = stats["inplane_rms"]
    out_rms = stats["outplane_rms"]

    n = len(in_rms)
    if n > Config.SCATTER_MAX_POINTS:
        rng = np.random.default_rng(Config.SCATTER_SEED)
        idx = rng.choice(n, size=Config.SCATTER_MAX_POINTS, replace=False)
        in_rms  = in_rms[idx]
        out_rms = out_rms[idx]

    x_lo, x_hi = Config.SCATTER_ZOOM_XLIM
    y_lo, y_hi = Config.SCATTER_ZOOM_YLIM

    mask_normal  = (
        (out_rms >= x_lo) & (out_rms <= x_hi) &
        (in_rms  >= y_lo) & (in_rms  <= y_hi)
    )
    mask_outlier = ~mask_normal

    ax.scatter(
        out_rms[mask_normal], in_rms[mask_normal],
        s=Config.SCATTER_SIZE,
        color=Config.INPLANE_COLOR,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
        zorder=1,
    )
    ax.scatter(
        out_rms[mask_outlier], in_rms[mask_outlier],
        s=Config.OUTLIER_SIZE,
        color=Config.OUTLIER_COLOR,
        alpha=Config.OUTLIER_ALPHA,
        linewidths=0,
        zorder=2,
        label='离群点',
    )

    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for t in leg.get_texts():
        t.set_fontproperties(CN_FONT)

    ax.set_xlabel(r'面外 RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'面内 RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振 RMS 面内-面外散点图', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 图3：放大散点（正方形） ====================
def plot_rms_scatter_zoomed(stats: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=Config.SCATTER_ZOOM_FIGSIZE)

    in_rms  = stats["inplane_rms"]
    out_rms = stats["outplane_rms"]

    x_lo, x_hi = Config.SCATTER_ZOOM_XLIM
    y_lo, y_hi = Config.SCATTER_ZOOM_YLIM

    mask = (
        (out_rms >= x_lo) & (out_rms <= x_hi) &
        (in_rms  >= y_lo) & (in_rms  <= y_hi)
    )
    x_plot = out_rms[mask]
    y_plot = in_rms[mask]

    if len(x_plot) > Config.SCATTER_MAX_POINTS:
        rng = np.random.default_rng(Config.SCATTER_SEED)
        idx = rng.choice(len(x_plot), size=Config.SCATTER_MAX_POINTS, replace=False)
        x_plot = x_plot[idx]
        y_plot = y_plot[idx]

    ax.scatter(
        x_plot, y_plot,
        s=Config.SCATTER_SIZE,
        color=Config.INPLANE_COLOR,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(r'面外 RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'面内 RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)

    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振 RMS 分布（主体 / 尾部 / 全量散点 / 放大散点）")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载 RMS 数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    stats = load_rms_from_enriched_stats()

    n = len(stats["inplane_rms"])
    combined = np.concatenate([stats["inplane_rms"], stats["outplane_rms"]])
    x_split  = float(np.percentile(combined, Config.HIST_X_PERCENTILE))

    print(f"✓ 共加载 {n} 个配对样本")
    print(f"  {Config.HIST_X_PERCENTILE}pct 分界值：{x_split:.4f} m/s²")
    print(f"  面内 RMS：min={stats['inplane_rms'].min():.4f}  "
          f"max={stats['inplane_rms'].max():.4f}  "
          f"mean={stats['inplane_rms'].mean():.4f}")
    print(f"  面外 RMS：min={stats['outplane_rms'].min():.4f}  "
          f"max={stats['outplane_rms'].max():.4f}  "
          f"mean={stats['outplane_rms'].mean():.4f}")

    in_tail_n  = int(np.sum(stats["inplane_rms"]  > x_split))
    out_tail_n = int(np.sum(stats["outplane_rms"] > x_split))
    print(f"  尾部（> {x_split:.4f}）：面内 {in_tail_n} 条，面外 {out_tail_n} 条")

    print("\n[步骤2] 绘制图像...")
    fig_body    = plot_rms_histogram(stats, x_split)
    fig_tail    = plot_rms_tail_histogram(stats, x_split)
    fig_scatter = plot_rms_scatter(stats)
    fig_zoom    = plot_rms_scatter_zoomed(stats)
    print("✓ 四幅图像生成完成")
    print("=" * 80)

    ploter = PlotLib()
    for fig in [fig_body, fig_tail, fig_scatter, fig_zoom]:
        ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
