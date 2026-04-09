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

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'

    _palette = get_blue_color_map(style='discrete', start_map_index=1, end_map_index=5).colors
    BAR_COLOR = _palette[2]
    BAR_ALPHA = 0.72

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_0_normal"
    )


# ==================== 数据加载 ====================
def load_ellipticity_data() -> np.ndarray:
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    ellipticity: list[float] = []

    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            coupling = sample.get("cross_coupling") or {}
            val = coupling.get("ellipticity")
            if val is not None:
                ellipticity.append(val)

    return np.array(ellipticity, dtype=np.float64)


# ==================== 工具 ====================
def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


# ==================== 绘图 ====================
def plot_ellipticity_histogram(ellipticity: np.ndarray) -> plt.Figure:
    bins    = np.linspace(0.0, 1.0, Config.N_BINS + 1)
    width   = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2

    counts, _ = np.histogram(ellipticity, bins=bins)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.bar(centers, counts, width=width,
           color=Config.BAR_COLOR, alpha=Config.BAR_ALPHA, edgecolor='none')

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('椭圆率', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('样本数（个）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('随机振动轨迹椭圆率分布', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 图3.8 随机振动轨迹椭圆率分布直方图")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载椭圆率数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    ellipticity = load_ellipticity_data()

    print(f"✓ 有效样本数：{len(ellipticity)}")
    print(f"  椭圆率：min={ellipticity.min():.4f}  "
          f"max={ellipticity.max():.4f}  "
          f"mean={ellipticity.mean():.4f}  "
          f"median={float(np.median(ellipticity)):.4f}")

    print("\n[步骤2] 绘制图像...")
    fig = plot_ellipticity_histogram(ellipticity)
    print("✓ 图像生成完成")
    print("=" * 80)

    ploter = PlotLib()
    ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
