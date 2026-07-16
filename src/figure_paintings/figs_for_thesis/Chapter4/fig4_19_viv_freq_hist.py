import json
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
    CN_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    VIV_CLASS_ID = 1
    FEATURE_BATCH_SIZE = 512
    N_BINS = 80
    FREQ_X_PERCENTILE = 100.0

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    BAR_ALPHA = 0.72

    ENRICHED_STATS_DIR = get_enriched_class_dir(1)
    WEB_PAGE = "fig4_22 VIV主频分布"


def load_dominant_freq_data() -> dict:
    ensure_enriched_for_figures(class_id=Config.VIV_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = iter_enriched_json_files(stats_dir)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    dom_freq_in: list[float] = []
    dom_freq_out: list[float] = []
    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            psd_in = sample.get("psd_inplane") or {}
            psd_out = sample.get("psd_outplane") or {}
            freqs_in = psd_in.get("frequencies")
            powers_in = psd_in.get("powers")
            freqs_out = psd_out.get("frequencies")
            powers_out = psd_out.get("powers")

            if freqs_in and powers_in:
                dom_idx = int(np.argmax(powers_in))
                dom_freq_in.append(float(freqs_in[dom_idx]))
            if freqs_out and powers_out:
                dom_idx = int(np.argmax(powers_out))
                dom_freq_out.append(float(freqs_out[dom_idx]))

    return {
        "dom_freq_in": np.asarray(dom_freq_in, dtype=np.float64),
        "dom_freq_out": np.asarray(dom_freq_out, dtype=np.float64),
    }


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax) -> None:
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)


def plot_dominant_freq_histogram(data: dict) -> plt.Figure:
    freq_in = data["dom_freq_in"]
    freq_out = data["dom_freq_out"]
    combined = np.concatenate([freq_in, freq_out])
    x_max = float(np.percentile(combined, Config.FREQ_X_PERCENTILE))
    x_max = max(x_max, 1e-6)

    bins = np.linspace(0, x_max, Config.N_BINS + 1)
    width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    counts_in, _ = np.histogram(freq_in[freq_in <= x_max], bins=bins)
    counts_out, _ = np.histogram(freq_out[freq_out <= x_max], bins=bins)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    half = width * 0.46
    ax.bar(centers - half / 2, counts_in, width=half, color=Config.INPLANE_COLOR, alpha=Config.BAR_ALPHA, label="面内")
    ax.bar(centers + half / 2, counts_out, width=half, color=Config.OUTPLANE_COLOR, alpha=Config.BAR_ALPHA, label="面外")

    ax.set_xlim(0, x_max)
    ax.set_xlabel("主频 (Hz)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("样本数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.96)
    return fig


def main() -> None:
    print("=" * 80)
    print("图4-22 涡激共振主频分布直方图")
    print("=" * 80)
    print("\n[步骤1] 加载 VIV 主频 enriched 数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    data = load_dominant_freq_data()
    print(f"[OK] 面内有效样本：{len(data['dom_freq_in'])}，面外有效样本：{len(data['dom_freq_out'])}")
    print(f"  面内主频 median={float(np.median(data['dom_freq_in'])):.4f} Hz")
    print(f"  面外主频 median={float(np.median(data['dom_freq_out'])):.4f} Hz")

    print("\n[步骤2] 绘制图像...")
    fig = plot_dominant_freq_histogram(data)
    web_push(fig, page=Config.WEB_PAGE, slot=0, title="涡激共振主频分布", page_cols=1)
    plt.close(fig)
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}")


if __name__ == "__main__":
    main()
