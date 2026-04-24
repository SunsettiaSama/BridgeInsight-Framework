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
    N_MODES = 10

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'
    SHADE_ALPHA = 0.18
    LINE_WIDTH = 2.2

    INPLANE_COLOR  = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_1_viv"
    )


# ==================== 数据加载 ====================
def load_psd_modes() -> dict:
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    cumsum_in:  list[np.ndarray] = []
    cumsum_out: list[np.ndarray] = []

    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            psd_in  = sample.get("psd_inplane")  or {}
            psd_out = sample.get("psd_outplane") or {}

            powers_in  = psd_in.get("powers")
            powers_out = psd_out.get("powers")

            if powers_in and len(powers_in) > 0:
                arr = np.array(powers_in, dtype=np.float64)
                arr_sorted = np.sort(arr)[::-1]
                total = arr_sorted.sum()
                if total > 0:
                    cumsum_in.append(np.cumsum(arr_sorted) / total)

            if powers_out and len(powers_out) > 0:
                arr = np.array(powers_out, dtype=np.float64)
                arr_sorted = np.sort(arr)[::-1]
                total = arr_sorted.sum()
                if total > 0:
                    cumsum_out.append(np.cumsum(arr_sorted) / total)

    return {
        "cumsum_in":  cumsum_in,
        "cumsum_out": cumsum_out,
    }


# ==================== 统计聚合 ====================
def _aggregate(curves: list[np.ndarray], n_modes: int) -> dict:
    mat = np.full((len(curves), n_modes), np.nan)
    for i, c in enumerate(curves):
        length = min(len(c), n_modes)
        mat[i, :length] = c[:length]
        if length < n_modes:
            mat[i, length:] = c[-1]

    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)
    return {"mean": mean, "std": std, "n": len(curves)}


# ==================== 工具 ====================
def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax):
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for t in leg.get_texts():
        t.set_fontproperties(CN_FONT)


# ==================== 绘图 ====================
def plot_energy_cumsum(data: dict) -> plt.Figure:
    n = Config.N_MODES
    stats_in  = _aggregate(data["cumsum_in"],  n)
    stats_out = _aggregate(data["cumsum_out"], n)

    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    mean_in = stats_in["mean"]
    std_in  = stats_in["std"]
    ax.plot(x, mean_in, color=Config.INPLANE_COLOR,
            linewidth=Config.LINE_WIDTH, marker='o', markersize=6,
            label='面内')
    ax.fill_between(x,
                    np.clip(mean_in - std_in, 0, 1),
                    np.clip(mean_in + std_in, 0, 1),
                    color=Config.INPLANE_COLOR, alpha=Config.SHADE_ALPHA)

    mean_out = stats_out["mean"]
    std_out  = stats_out["std"]
    ax.plot(x, mean_out, color=Config.OUTPLANE_COLOR,
            linewidth=Config.LINE_WIDTH, marker='s', markersize=6,
            label='面外')
    ax.fill_between(x,
                    np.clip(mean_out - std_out, 0, 1),
                    np.clip(mean_out + std_out, 0, 1),
                    color=Config.OUTPLANE_COLOR, alpha=Config.SHADE_ALPHA)

    ax.axhline(y=1.0, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xlabel('主频阶序', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('累积能量占比', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('前10阶主频累积能量分布（涡激共振）', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振前10阶主频累积能量分布")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载 PSD 模态数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    data = load_psd_modes()

    n_in  = len(data["cumsum_in"])
    n_out = len(data["cumsum_out"])
    print(f"✓ 面内有效样本：{n_in}，面外有效样本：{n_out}")

    stats_in  = _aggregate(data["cumsum_in"],  Config.N_MODES)
    stats_out = _aggregate(data["cumsum_out"], Config.N_MODES)
    print(f"  面内  第1阶占比均值：{stats_in['mean'][0]:.4f}  "
          f"前10阶累积均值：{stats_in['mean'][-1]:.4f}")
    print(f"  面外  第1阶占比均值：{stats_out['mean'][0]:.4f}  "
          f"前10阶累积均值：{stats_out['mean'][-1]:.4f}")

    print("\n[步骤2] 绘制图像...")
    fig = plot_energy_cumsum(data)
    print("✓ 图像生成完成")
    print("=" * 80)

    ploter = PlotLib()
    ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
