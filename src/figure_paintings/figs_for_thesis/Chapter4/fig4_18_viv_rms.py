import json
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import get_enriched_class_dir, iter_enriched_json_files
from src.figure_paintings.figs_for_thesis.config import (
    ANNOTATION_COLOR,
    CN_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    VIV_CLASS_ID = 1
    FEATURE_BATCH_SIZE = 512

    N_BINS = 80
    RMS_X_PERCENTILE = 99.0
    # 图4-9 随机振动 RMS 直方图右边界（目视读取，单位 m/s²）
    NORMAL_TAIL_RMS = 0.16
    SCATTER_AXIS_PERCENTILE = 99.5
    SCATTER_AXIS_PAD = 1.08
    SCATTER_MAX_POINTS = 120_000
    SCATTER_SEED = 42

    FIG_SIZE = SQUARE_FIG_SIZE
    LABEL_FONT_SIZE = SQUARE_FONT_SIZE
    TICK_FONT_SIZE = SQUARE_FONT_SIZE - 4
    LEGEND_FONT_SIZE = SQUARE_FONT_SIZE - 4

    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    SCATTER_COLOR = VIV_INPLANE_COLOR
    BAR_ALPHA = 0.72
    SCATTER_SIZE = 5
    SCATTER_ALPHA = 0.35

    TAIL_LINE_COLOR = ANNOTATION_COLOR
    TAIL_LINEWIDTH = 1.8
    TAIL_LINESTYLE = "-."

    ENRICHED_STATS_DIR = get_enriched_class_dir(1)
    WEB_DASHBOARD_PORT = 15678


def load_rms_data() -> dict:
    ensure_enriched_for_figures(class_id=Config.VIV_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    stats_dir = Config.ENRICHED_STATS_DIR
    json_files = iter_enriched_json_files(stats_dir)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    inplane_rms: list[float] = []
    outplane_rms: list[float] = []
    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            ts_in = sample.get("time_stats_inplane") or {}
            ts_out = sample.get("time_stats_outplane") or {}
            rms_in = ts_in.get("rms")
            rms_out = ts_out.get("rms")
            if rms_in is not None and rms_out is not None:
                inplane_rms.append(float(rms_in))
                outplane_rms.append(float(rms_out))

    return {
        "inplane_rms": np.asarray(inplane_rms, dtype=np.float64),
        "outplane_rms": np.asarray(outplane_rms, dtype=np.float64),
    }


def _mark_normal_tail_hist(ax, x_tail: float) -> None:
    ax.axvline(
        x_tail,
        color=Config.TAIL_LINE_COLOR,
        linewidth=Config.TAIL_LINEWIDTH,
        linestyle=Config.TAIL_LINESTYLE,
        zorder=4,
        label="随机振动尾部",
    )


def _mark_normal_tail_scatter(ax, x_tail: float) -> None:
    ax.axvline(
        x_tail,
        color=Config.TAIL_LINE_COLOR,
        linewidth=Config.TAIL_LINEWIDTH,
        linestyle=Config.TAIL_LINESTYLE,
        zorder=4,
        label="随机振动尾部",
    )
    ax.axhline(
        x_tail,
        color=Config.TAIL_LINE_COLOR,
        linewidth=Config.TAIL_LINEWIDTH,
        linestyle=Config.TAIL_LINESTYLE,
        zorder=4,
    )


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax) -> None:
    leg = ax.legend(fontsize=Config.LEGEND_FONT_SIZE, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)


def _is_web_dashboard_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _calc_rms_histogram(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    in_rms = data["inplane_rms"]
    out_rms = data["outplane_rms"]
    combined = np.concatenate([in_rms, out_rms])
    x_max = float(np.percentile(combined, Config.RMS_X_PERCENTILE))
    x_max = max(x_max, 1e-6)

    bins = np.linspace(0, x_max, Config.N_BINS + 1)
    width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    counts_in, _ = np.histogram(in_rms[in_rms <= x_max], bins=bins)
    counts_out, _ = np.histogram(out_rms[out_rms <= x_max], bins=bins)
    return centers, counts_in, counts_out, width, x_max


def plot_rms_histogram(data: dict, normal_tail: float) -> plt.Figure:
    centers, counts_in, counts_out, width, x_max = _calc_rms_histogram(data)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    half = width * 0.46
    ax.bar(centers - half / 2, counts_in, width=half, color=Config.INPLANE_COLOR, alpha=Config.BAR_ALPHA, label="面内")
    ax.bar(centers + half / 2, counts_out, width=half, color=Config.OUTPLANE_COLOR, alpha=Config.BAR_ALPHA, label="面外")
    if 0 < normal_tail < x_max:
        _mark_normal_tail_hist(ax, normal_tail)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r"RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel("样本数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    _add_legend(ax)
    _apply_grid(ax)
    fig.tight_layout()
    return fig


def plot_rms_scatter(data: dict, normal_tail: float) -> plt.Figure:
    in_rms = data["inplane_rms"]
    out_rms = data["outplane_rms"]
    x_plot = out_rms
    y_plot = in_rms
    if len(x_plot) > Config.SCATTER_MAX_POINTS:
        rng = np.random.default_rng(Config.SCATTER_SEED)
        idx = rng.choice(len(x_plot), size=Config.SCATTER_MAX_POINTS, replace=False)
        x_plot = x_plot[idx]
        y_plot = y_plot[idx]

    combined_full = np.concatenate([out_rms, in_rms])
    xy_max = float(np.percentile(combined_full, Config.SCATTER_AXIS_PERCENTILE))
    xy_max = max(xy_max * Config.SCATTER_AXIS_PAD, 1e-6)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.scatter(
        x_plot,
        y_plot,
        s=Config.SCATTER_SIZE,
        color=Config.SCATTER_COLOR,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
    )
    if 0 < normal_tail < xy_max:
        _mark_normal_tail_scatter(ax, normal_tail)
        _add_legend(ax)
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)
    ax.set_xlabel(r"面外 RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel(r"面内 RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    _apply_grid(ax)
    fig.tight_layout()
    return fig


def push_figures(figures: list[tuple[plt.Figure, str]]) -> None:
    if not _is_web_dashboard_available(Config.WEB_DASHBOARD_PORT):
        print(
            "  未检测到 VibDash 服务，跳过 WebUI 推送；"
            "如需预览请先运行：python -m src.visualize_tools.web_dashboard"
        )
        return

    page = "fig4_18 涡激共振 RMS"
    for slot, (fig, title) in enumerate(figures):
        web_push(fig, page=page, slot=slot, title=title, page_cols=2 if slot == 0 else None)
    print(f"[OK] 已推送到 WebUI：{page}")


def main() -> None:
    print("=" * 80)
    print("图4-18 涡激共振 RMS 分布")
    print("=" * 80)
    print(f"\n[步骤1] 加载 VIV RMS enriched 数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    data = load_rms_data()
    print(f"[OK] 有效配对样本：{len(data['inplane_rms'])}")
    print(f"  面内 RMS median={float(np.median(data['inplane_rms'])):.4f}")
    print(f"  面外 RMS median={float(np.median(data['outplane_rms'])):.4f}")

    normal_tail = Config.NORMAL_TAIL_RMS
    print(f"\n[步骤2] 随机振动尾部标记（图4-9 右边界）：{normal_tail:.2f} m/s^2")

    print("\n[步骤3] 绘制图像...")
    fig_hist = plot_rms_histogram(data, normal_tail)
    fig_scatter = plot_rms_scatter(data, normal_tail)
    print("[OK] 已生成 2 张独立图像")

    push_figures([
        (fig_hist, "涡激共振 RMS 直方图"),
        (fig_scatter, "面内-面外 RMS 散点图"),
    ])
    plt.close(fig_hist)
    plt.close(fig_scatter)


if __name__ == "__main__":
    main()
