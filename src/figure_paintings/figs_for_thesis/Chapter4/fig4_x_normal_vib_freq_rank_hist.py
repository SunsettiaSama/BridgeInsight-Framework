import json
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import get_enriched_class_dir, iter_enriched_json_files
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    NORMAL_VIB_CLASS_ID = 0
    FEATURE_BATCH_SIZE = 512

    FREQ_MAX_HZ = 25.0
    BIN_WIDTH_HZ = 0.5
    RANKS = (1, 2, 3)
    RANK_LABELS = ("主频", "次频", "第三阶")

    # 簇内三柱合计占 bin 宽度的比例，剩余为簇间可见缝隙
    CLUSTER_FILL = 0.72
    # 全色谱跨越取色：紫 → 橙 → 珊瑚红（跳过浅黄白）
    _full = get_full_color_map(style="discrete").colors
    RANK_COLORS = (_full[0], _full[7], _full[9])

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = "#d8d8d8"
    GRID_ALPHA = 0.55
    GRID_LINESTYLE = "-"
    BAR_EDGE_COLOR = "white"
    BAR_EDGE_WIDTH = 0.5

    ENRICHED_STATS_DIR = get_enriched_class_dir(0)
    WEB_DASHBOARD_PORT = 15678
    WEB_PAGE = "fig4_x 主频频数分布"


def extract_rank_frequency(freqs, powers, rank: int) -> float | None:
    if not freqs or not powers:
        return None
    f_arr = np.asarray(freqs, dtype=np.float64)
    p_arr = np.asarray(powers, dtype=np.float64)
    if len(f_arr) != len(p_arr) or len(f_arr) == 0:
        return None
    order = np.argsort(p_arr)[::-1]
    idx = rank - 1
    if idx >= len(order):
        return None
    return float(f_arr[order[idx]])


def load_rank_frequency_data() -> dict[int, np.ndarray]:
    ensure_enriched_for_figures(class_id=Config.NORMAL_VIB_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    stats_dir = Config.ENRICHED_STATS_DIR
    json_files = iter_enriched_json_files(stats_dir)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    buckets: dict[int, list[float]] = {rank: [] for rank in Config.RANKS}
    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            for plane_key in ("psd_inplane", "psd_outplane"):
                psd = sample.get(plane_key) or {}
                freqs = psd.get("frequencies")
                powers = psd.get("powers")
                for rank in Config.RANKS:
                    freq = extract_rank_frequency(freqs, powers, rank)
                    if freq is not None and 0.0 <= freq <= Config.FREQ_MAX_HZ:
                        buckets[rank].append(freq)

    return {rank: np.asarray(vals, dtype=np.float64) for rank, vals in buckets.items()}


def _calc_histograms(rank_data: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    bin_edges = np.arange(0.0, Config.FREQ_MAX_HZ + Config.BIN_WIDTH_HZ, Config.BIN_WIDTH_HZ)
    centers = bin_edges[:-1] + Config.BIN_WIDTH_HZ / 2.0
    counts = {
        rank: np.histogram(values, bins=bin_edges)[0].astype(np.float64)
        for rank, values in rank_data.items()
    }
    return centers, counts


def _bar_geometry() -> tuple[float, np.ndarray]:
    n = len(Config.RANKS)
    cluster_width = Config.BIN_WIDTH_HZ * Config.CLUSTER_FILL
    bar_width = cluster_width / n
    offsets = (np.arange(n) - (n - 1) / 2.0) * bar_width
    return bar_width, offsets


def plot_rank_frequency_histogram(rank_data: dict[int, np.ndarray]) -> plt.Figure:
    centers, counts = _calc_histograms(rank_data)
    bar_width, offsets = _bar_geometry()

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    for rank_i, rank in enumerate(Config.RANKS):
        n_samples = len(rank_data[rank])
        ax.bar(
            centers + offsets[rank_i],
            counts[rank],
            width=bar_width,
            color=Config.RANK_COLORS[rank_i],
            edgecolor=Config.BAR_EDGE_COLOR,
            linewidth=Config.BAR_EDGE_WIDTH,
            align="center",
            label=f"{Config.RANK_LABELS[rank_i]}（n={n_samples:,}）",
            zorder=3,
        )

    all_max = max(float(np.max(c)) for c in counts.values())
    ax.set_xlim(-0.05, Config.FREQ_MAX_HZ + 0.05)
    ax.set_ylim(0, all_max * 1.08)
    ax.set_xlabel("频率 (Hz)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("频数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(
        "随机振动前三阶主频频数分布（0.5 Hz 分组）",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE,
        pad=14,
    )

    ax.set_xticks(np.arange(0.0, Config.FREQ_MAX_HZ + 1e-9, 2.0))
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{int(x):,}"))

    ax.grid(True, axis="y", color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(fontsize=FONT_SIZE - 3, framealpha=0.9, prop=CN_FONT, loc="upper right")
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)

    fig.tight_layout()
    return fig


def _is_web_dashboard_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def push_figure(fig: plt.Figure, slot: int) -> None:
    if not _is_web_dashboard_available(Config.WEB_DASHBOARD_PORT):
        print(
            "  未检测到 VibDash 服务，跳过 WebUI 推送；"
            "如需预览请先运行：python -m src.visualize_tools.web_dashboard"
        )
        return
    web_push(
        fig,
        page=Config.WEB_PAGE,
        slot=slot,
        title="前三阶主频频数分布",
        page_cols=2,
    )
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE} / slot={slot}")


def main() -> None:
    print("=" * 80)
    print("图4-x 随机振动前三阶主频频数分布")
    print("=" * 80)
    print("\n[步骤1] 加载 enriched PSD 数据（本地不存在则自动小 batch 生成）...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    rank_data = load_rank_frequency_data()
    for rank, label in zip(Config.RANKS, Config.RANK_LABELS):
        print(f"  {label}有效频点：{len(rank_data[rank]):,}")

    print("\n[步骤2] 绘制图像...")
    fig = plot_rank_frequency_histogram(rank_data)
    push_figure(fig, slot=2)
    plt.close(fig)


if __name__ == "__main__":
    main()
