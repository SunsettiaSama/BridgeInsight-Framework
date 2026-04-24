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
    N_BINS = 80
    ENERGY_X_PERCENTILE = 100.0

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'

    INPLANE_COLOR  = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    BAR_ALPHA = 0.72

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_1_viv"
    )


# ==================== 数据加载 ====================
def load_energy_data() -> dict:
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    dom_energy_in:  list[float] = []
    dom_energy_out: list[float] = []

    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            spec_in  = sample.get("spectral_inplane")  or {}
            spec_out = sample.get("spectral_outplane") or {}

            energy_in  = spec_in.get("dominant_mode_energy_ratio")
            energy_out = spec_out.get("dominant_mode_energy_ratio")

            if energy_in is not None:
                dom_energy_in.append(energy_in)
            if energy_out is not None:
                dom_energy_out.append(energy_out)

    return {
        "dom_energy_in":  np.array(dom_energy_in,  dtype=np.float64),
        "dom_energy_out": np.array(dom_energy_out, dtype=np.float64),
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


# ==================== 绘图 ====================
def plot_dominant_energy_histogram(data: dict) -> plt.Figure:
    energy_in  = data["dom_energy_in"]
    energy_out = data["dom_energy_out"]

    combined = np.concatenate([energy_in, energy_out])
    x_max = float(np.percentile(combined, Config.ENERGY_X_PERCENTILE))
    x_max = min(x_max, 1.0)

    bins    = np.linspace(0, x_max, Config.N_BINS + 1)
    width   = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2

    counts_in,  _ = np.histogram(energy_in[energy_in   <= x_max], bins=bins)
    counts_out, _ = np.histogram(energy_out[energy_out <= x_max], bins=bins)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    _grouped_bars(ax, centers, counts_in, counts_out, width)

    ax.set_xlim(0, x_max)
    ax.set_xlabel('主频能量占比', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('样本数（个）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振主频能量占比分布', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振主频能量占比分布直方图")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载主频能量数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    data = load_energy_data()

    n_in  = len(data["dom_energy_in"])
    n_out = len(data["dom_energy_out"])
    print(f"✓ 面内有效样本：{n_in}，面外有效样本：{n_out}")
    print(f"  面内主频能量：mean={data['dom_energy_in'].mean():.4f}  "
          f"median={float(np.median(data['dom_energy_in'])):.4f}")
    print(f"  面外主频能量：mean={data['dom_energy_out'].mean():.4f}  "
          f"median={float(np.median(data['dom_energy_out'])):.4f}")

    print("\n[步骤2] 绘制图像...")
    fig = plot_dominant_energy_histogram(data)
    print("✓ 图像生成完成")
    print("=" * 80)

    ploter = PlotLib()
    ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
