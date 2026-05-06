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

    DL_RESULT_GLOB   = project_root / "results" / "identification_result"        / "res_cnn_full_dataset_*.json"
    MECC_RESULT_GLOB = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"
    MAX_SAMPLES      = 5000


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
def plot_freq_energy_scatter(dl_data: dict, mecc_data: dict | None = None) -> plt.Figure:
    def _downsample(x, y, seed, max_pts):
        n = len(x)
        if n <= max_pts:
            return x, y
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_pts, replace=False)
        return x[idx], y[idx]

    freq_in   = dl_data["dom_freq_in"]
    freq_out  = dl_data["dom_freq_out"]
    energy_in  = dl_data["dom_energy_in"]
    energy_out = dl_data["dom_energy_out"]

    fi, ei = _downsample(freq_in,  energy_in,  Config.SCATTER_SEED,     Config.SCATTER_MAX_POINTS)
    fo, eo = _downsample(freq_out, energy_out, Config.SCATTER_SEED + 1, Config.SCATTER_MAX_POINTS)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    ax.scatter(fi, ei, s=Config.SCATTER_SIZE, color=Config.INPLANE_COLOR,
               alpha=Config.SCATTER_ALPHA, linewidths=0, label='DL 面内', zorder=1)
    ax.scatter(fo, eo, s=Config.SCATTER_SIZE, color=Config.OUTPLANE_COLOR,
               alpha=Config.SCATTER_ALPHA, linewidths=0, label='DL 面外', zorder=2)

    if mecc_data is not None:
        mfi, mei = _downsample(
            mecc_data["dom_freq_in"], mecc_data["dom_energy_in"],
            Config.SCATTER_SEED + 2, Config.SCATTER_MAX_POINTS,
        )
        mfo, meo = _downsample(
            mecc_data["dom_freq_out"], mecc_data["dom_energy_out"],
            Config.SCATTER_SEED + 3, Config.SCATTER_MAX_POINTS,
        )
        ax.scatter(mfi, mei, s=Config.SCATTER_SIZE, color=MECC_INPLANE_COLOR,
                   alpha=Config.SCATTER_ALPHA, linewidths=0, marker='D',
                   label='MECC 面内', zorder=3)
        ax.scatter(mfo, meo, s=Config.SCATTER_SIZE, color=MECC_OUTPLANE_COLOR,
                   alpha=Config.SCATTER_ALPHA, linewidths=0, marker='D',
                   label='MECC 面外', zorder=4)

    ax.set_xlabel('主频 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('主频能量占比', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振主频与能量占比散点图（DL vs MECC）', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振主频与能量占比散点图（DL vs MECC）")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载 DL 主频能量数据...")
    dl_stats = load_enriched_stats(Config.ENRICHED_STATS_DIR)
    print(f"✓ DL 面内：{len(dl_stats['dom_freq_in'])}，面外：{len(dl_stats['dom_freq_out'])}")

    print(f"\n[步骤2] 加载 MECC 识别结果并计算统计量...")
    mecc_result  = load_latest_result(Config.MECC_RESULT_GLOB)
    mecc_samples = get_viv_samples(mecc_result, max_n=Config.MAX_SAMPLES)
    print(f"  MECC VIV 样本：{len(mecc_samples)} 个")
    mecc_stats = compute_signal_stats(mecc_samples, source='MECC')
    print(f"✓ MECC 面内：{len(mecc_stats['dom_freq_in'])}，面外：{len(mecc_stats['dom_freq_out'])}")

    print("\n[步骤3] 绘制图像...")
    fig = plot_freq_energy_scatter(dl_stats, mecc_stats)
    print("✓ 图像生成完成")

    print("\n[步骤4] 推送到 WebUI...")
    web_push(fig, page='fig4_24 主频-能量散点 DL vs MECC', slot=0,
             title='主频-能量散点对比', page_cols=1)
    print("✓ 推送完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
