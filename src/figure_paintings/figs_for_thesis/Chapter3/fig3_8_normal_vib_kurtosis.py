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
    get_blue_color_map,
)


# ==================== 常量配置 ====================
class Config:
    N_BINS = 80
    X_LOW_PERCENTILE  = 1.0
    X_HIGH_PERCENTILE = 95.0

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'

    _palette = get_blue_color_map(style='discrete', start_map_index=1, end_map_index=5).colors
    INPLANE_COLOR  = _palette[2]
    OUTPLANE_COLOR = _palette[3]
    BAR_ALPHA = 0.72

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_0_normal"
    )


# ==================== 数据加载 ====================
def load_kurtosis_data() -> dict:
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    kurtosis_in:  list[float] = []
    kurtosis_out: list[float] = []

    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            ts_in  = sample.get("time_stats_inplane")  or {}
            ts_out = sample.get("time_stats_outplane") or {}

            val_in  = ts_in.get("kurtosis")
            val_out = ts_out.get("kurtosis")

            if val_in is not None:
                kurtosis_in.append(val_in)
            if val_out is not None:
                kurtosis_out.append(val_out)

    return {
        "kurtosis_in":  np.array(kurtosis_in,  dtype=np.float64),
        "kurtosis_out": np.array(kurtosis_out, dtype=np.float64),
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
def plot_kurtosis_histogram(data: dict) -> plt.Figure:
    k_in  = data["kurtosis_in"]
    k_out = data["kurtosis_out"]

    combined = np.concatenate([k_in, k_out])
    x_min = float(np.percentile(combined, Config.X_LOW_PERCENTILE))
    x_max = float(np.percentile(combined, Config.X_HIGH_PERCENTILE))

    bins    = np.linspace(x_min, x_max, Config.N_BINS + 1)
    width   = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2

    mask_in  = (k_in  >= x_min) & (k_in  <= x_max)
    mask_out = (k_out >= x_min) & (k_out <= x_max)
    counts_in,  _ = np.histogram(k_in[mask_in],   bins=bins)
    counts_out, _ = np.histogram(k_out[mask_out],  bins=bins)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    _grouped_bars(ax, centers, counts_in, counts_out, width)

    ax.axvline(x=0, color='gray', linewidth=1.0, linestyle='-', alpha=0.8, zorder=3)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('峭度（Fisher 定义）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('样本数（个）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('随机振动峭度分布', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 图3.8 随机振动峭度分布直方图")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载峭度数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    data = load_kurtosis_data()

    n_in  = len(data["kurtosis_in"])
    n_out = len(data["kurtosis_out"])
    print(f"✓ 面内有效样本：{n_in}，面外有效样本：{n_out}")
    print(f"  面内峭度：min={data['kurtosis_in'].min():.4f}  "
          f"max={data['kurtosis_in'].max():.4f}  "
          f"mean={data['kurtosis_in'].mean():.4f}  "
          f"median={float(np.median(data['kurtosis_in'])):.4f}")
    print(f"  面外峭度：min={data['kurtosis_out'].min():.4f}  "
          f"max={data['kurtosis_out'].max():.4f}  "
          f"mean={data['kurtosis_out'].mean():.4f}  "
          f"median={float(np.median(data['kurtosis_out'])):.4f}")

    print("\n[步骤2] 绘制图像...")
    fig = plot_kurtosis_histogram(data)
    print("✓ 图像生成完成")
    print("=" * 80)

    ploter = PlotLib()
    ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
