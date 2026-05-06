import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, REC_FIG_SIZE,
    VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR,
)

_chapter4_dir = str(Path(__file__).parent)
if _chapter4_dir not in sys.path:
    sys.path.insert(0, _chapter4_dir)
from _viv_pipeline import (
    load_latest_result, get_viv_samples, compute_signal_stats, load_enriched_stats,
    MECC_INPLANE_COLOR, MECC_OUTPLANE_COLOR,
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

    DL_RESULT_GLOB   = project_root / "results" / "identification_result"        / "res_cnn_full_dataset_*.json"
    MECC_RESULT_GLOB = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"
    MAX_SAMPLES      = 5000


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


def _step_hist(ax, data: np.ndarray, bins, color: str, label: str):
    """绘制阶梯轮廓直方图（用于 MECC 叠加，不遮挡 DL 柱）。"""
    counts, edges = np.histogram(data, bins=bins)
    ax.step(edges[:-1], counts, where='post', color=color,
            linewidth=1.6, alpha=0.85, label=label)


# ==================== 图1a：主体分布直方图 ====================
def plot_rms_histogram(stats: dict, x_split: float, mecc_stats: dict | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    in_rms  = stats["inplane_rms"]
    out_rms = stats["outplane_rms"]

    bins    = np.linspace(0, x_split, Config.N_BINS + 1)
    width   = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2

    counts_in,  _ = np.histogram(in_rms,  bins=bins)
    counts_out, _ = np.histogram(out_rms, bins=bins)

    _grouped_bars(ax, centers, counts_in, counts_out, width)

    if mecc_stats is not None:
        m_in  = mecc_stats["inplane_rms"]
        m_out = mecc_stats["outplane_rms"]
        _step_hist(ax, m_in[m_in   <= x_split], bins, MECC_INPLANE_COLOR,  'MECC 面内')
        _step_hist(ax, m_out[m_out <= x_split], bins, MECC_OUTPLANE_COLOR, 'MECC 面外')

    if Config.HIST_LOG_SCALE:
        ax.set_yscale('log')
        ylabel = '样本数（个，对数坐标）'
    else:
        ylabel = '样本数（个）'

    ax.set_xlim(0, x_split)
    ax.set_xlabel(r'RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振加速度 RMS 主体分布（DL vs MECC）', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 图1b：尾部分布直方图 ====================
def plot_rms_tail_histogram(stats: dict, x_split: float, mecc_stats: dict | None = None) -> plt.Figure:
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

    if mecc_stats is not None:
        m_in  = mecc_stats["inplane_rms"]
        m_out = mecc_stats["outplane_rms"]
        _step_hist(ax, m_in[(m_in   > x_split) & (m_in   <= x_tail_max)], bins, MECC_INPLANE_COLOR,  'MECC 面内')
        _step_hist(ax, m_out[(m_out > x_split) & (m_out  <= x_tail_max)], bins, MECC_OUTPLANE_COLOR, 'MECC 面外')

    ax.set_xlim(x_split, x_tail_max)
    ax.set_xlabel(r'RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('样本数（个）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振加速度 RMS 尾部分布（DL vs MECC）', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 图2：全量散点（面内 vs 面外 RMS） ====================
def plot_rms_scatter(stats: dict, mecc_stats: dict | None = None) -> plt.Figure:
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
        label='DL 离群点',
    )

    if mecc_stats is not None:
        m_in  = mecc_stats["inplane_rms"]
        m_out = mecc_stats["outplane_rms"]
        n_m = len(m_in)
        if n_m > Config.SCATTER_MAX_POINTS:
            rng_m = np.random.default_rng(Config.SCATTER_SEED + 10)
            idx_m = rng_m.choice(n_m, size=Config.SCATTER_MAX_POINTS, replace=False)
            m_in  = m_in[idx_m]
            m_out = m_out[idx_m]
        ax.scatter(m_out, m_in, s=Config.SCATTER_SIZE,
                   color=MECC_INPLANE_COLOR, alpha=Config.SCATTER_ALPHA,
                   linewidths=0, marker='D', zorder=3, label='MECC')

    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for t in leg.get_texts():
        t.set_fontproperties(CN_FONT)

    ax.set_xlabel(r'面外 RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'面内 RMS ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振 RMS 面内-面外散点图（DL vs MECC）', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 图3：放大散点（正方形） ====================
def plot_rms_scatter_zoomed(stats: dict, mecc_stats: dict | None = None) -> plt.Figure:
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
        label='DL',
    )

    if mecc_stats is not None:
        m_in  = mecc_stats["inplane_rms"]
        m_out = mecc_stats["outplane_rms"]
        m_mask = (
            (m_out >= x_lo) & (m_out <= x_hi) &
            (m_in  >= y_lo) & (m_in  <= y_hi)
        )
        mx_plot = m_out[m_mask]
        my_plot = m_in[m_mask]
        if len(mx_plot) > Config.SCATTER_MAX_POINTS:
            rng_m = np.random.default_rng(Config.SCATTER_SEED + 10)
            idx_m = rng_m.choice(len(mx_plot), size=Config.SCATTER_MAX_POINTS, replace=False)
            mx_plot = mx_plot[idx_m]
            my_plot = my_plot[idx_m]
        ax.scatter(mx_plot, my_plot, s=Config.SCATTER_SIZE,
                   color=MECC_INPLANE_COLOR, alpha=Config.SCATTER_ALPHA,
                   linewidths=0, marker='D', label='MECC')
        leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
        for t in leg.get_texts():
            t.set_fontproperties(CN_FONT)

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
    print("涡激共振 RMS 分布（DL vs MECC）")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载 DL RMS 数据...")
    dl_stats = load_enriched_stats(Config.ENRICHED_STATS_DIR)
    n = len(dl_stats["inplane_rms"])
    print(f"✓ DL 配对样本：{n}")

    print(f"\n[步骤2] 加载 MECC 识别结果并计算统计量...")
    mecc_result  = load_latest_result(Config.MECC_RESULT_GLOB)
    mecc_samples = get_viv_samples(mecc_result, max_n=Config.MAX_SAMPLES)
    print(f"  MECC VIV 样本：{len(mecc_samples)} 个，开始计算原始统计量...")
    mecc_stats = compute_signal_stats(mecc_samples, source='MECC')
    print(f"✓ MECC 配对样本：{len(mecc_stats['inplane_rms'])}")

    combined = np.concatenate([dl_stats["inplane_rms"], dl_stats["outplane_rms"]])
    x_split  = float(np.percentile(combined, Config.HIST_X_PERCENTILE))
    print(f"\n  分界值（DL {Config.HIST_X_PERCENTILE}pct）：{x_split:.4f} m/s²")

    print("\n[步骤3] 绘制四幅图像...")
    fig_body    = plot_rms_histogram(dl_stats, x_split, mecc_stats)
    fig_tail    = plot_rms_tail_histogram(dl_stats, x_split, mecc_stats)
    fig_scatter = plot_rms_scatter(dl_stats, mecc_stats)
    fig_zoom    = plot_rms_scatter_zoomed(dl_stats, mecc_stats)
    print("✓ 四幅图像生成完成")

    print("\n[步骤4] 推送到 WebUI...")
    PAGE   = 'fig4_20-21 VIV RMS DL vs MECC'
    TITLES = ['RMS 主体分布', 'RMS 尾部分布', 'RMS 面内-面外散点', 'RMS 放大散点']
    for slot, (fig, title) in enumerate(zip([fig_body, fig_tail, fig_scatter, fig_zoom], TITLES)):
        web_push(fig, page=PAGE, slot=slot, title=title,
                 page_cols=2 if slot == 0 else None)
    print("✓ 推送完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
