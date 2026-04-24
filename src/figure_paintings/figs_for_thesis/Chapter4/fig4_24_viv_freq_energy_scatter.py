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
    SCATTER_MAX_POINTS = 100_000
    SCATTER_SEED = 42

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'

    INPLANE_COLOR  = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    SCATTER_SIZE  = 6
    SCATTER_ALPHA = 0.30

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_1_viv"
    )


# ==================== 数据加载 ====================
def load_freq_energy_data() -> dict:
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    dom_freq_in:    list[float] = []
    dom_freq_out:   list[float] = []
    dom_energy_in:  list[float] = []
    dom_energy_out: list[float] = []

    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            psd_in  = sample.get("psd_inplane")  or {}
            psd_out = sample.get("psd_outplane") or {}
            spec_in  = sample.get("spectral_inplane")  or {}
            spec_out = sample.get("spectral_outplane") or {}

            freqs_in  = psd_in.get("frequencies")
            powers_in = psd_in.get("powers")
            freqs_out  = psd_out.get("frequencies")
            powers_out = psd_out.get("powers")
            energy_in  = spec_in.get("dominant_mode_energy_ratio")
            energy_out = spec_out.get("dominant_mode_energy_ratio")

            if freqs_in and powers_in and energy_in is not None:
                dom_idx = int(np.argmax(powers_in))
                dom_freq_in.append(freqs_in[dom_idx])
                dom_energy_in.append(energy_in)

            if freqs_out and powers_out and energy_out is not None:
                dom_idx = int(np.argmax(powers_out))
                dom_freq_out.append(freqs_out[dom_idx])
                dom_energy_out.append(energy_out)

    return {
        "dom_freq_in":    np.array(dom_freq_in,    dtype=np.float64),
        "dom_freq_out":   np.array(dom_freq_out,   dtype=np.float64),
        "dom_energy_in":  np.array(dom_energy_in,  dtype=np.float64),
        "dom_energy_out": np.array(dom_energy_out, dtype=np.float64),
    }


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
def plot_freq_energy_scatter(data: dict) -> plt.Figure:
    freq_in   = data["dom_freq_in"]
    freq_out  = data["dom_freq_out"]
    energy_in  = data["dom_energy_in"]
    energy_out = data["dom_energy_out"]

    def _downsample(x, y, seed, max_pts):
        n = len(x)
        if n <= max_pts:
            return x, y
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_pts, replace=False)
        return x[idx], y[idx]

    fi, ei = _downsample(freq_in,  energy_in,  Config.SCATTER_SEED,     Config.SCATTER_MAX_POINTS)
    fo, eo = _downsample(freq_out, energy_out, Config.SCATTER_SEED + 1, Config.SCATTER_MAX_POINTS)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    ax.scatter(
        fi, ei,
        s=Config.SCATTER_SIZE,
        color=Config.INPLANE_COLOR,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
        label='面内',
        zorder=1,
    )
    ax.scatter(
        fo, eo,
        s=Config.SCATTER_SIZE,
        color=Config.OUTPLANE_COLOR,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
        label='面外',
        zorder=2,
    )

    ax.set_xlabel('主频 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('主频能量占比', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振主频与能量占比散点图', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振主频与能量占比散点图")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载主频及能量数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    data = load_freq_energy_data()

    n_in  = len(data["dom_freq_in"])
    n_out = len(data["dom_freq_out"])
    print(f"✓ 面内有效样本：{n_in}，面外有效样本：{n_out}")
    print(f"  面内主频能量：mean={data['dom_energy_in'].mean():.4f}  "
          f"median={float(np.median(data['dom_energy_in'])):.4f}")
    print(f"  面外主频能量：mean={data['dom_energy_out'].mean():.4f}  "
          f"median={float(np.median(data['dom_energy_out'])):.4f}")

    print("\n[步骤2] 绘制图像...")
    fig = plot_freq_energy_scatter(data)
    print("✓ 图像生成完成")
    print("=" * 80)

    ploter = PlotLib()
    ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
