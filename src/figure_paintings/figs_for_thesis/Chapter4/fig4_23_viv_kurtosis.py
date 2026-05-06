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
    N_BINS = 80
    X_LOW_PERCENTILE  = 1.0
    X_HIGH_PERCENTILE = 95.0

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

    DL_RESULT_GLOB   = project_root / "results" / "identification_result"        / "res_cnn_full_dataset_*.json"
    MECC_RESULT_GLOB = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"
    MAX_SAMPLES      = 5000


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
def plot_kurtosis_histogram(dl_data: dict, mecc_data: dict | None = None) -> plt.Figure:
    k_in  = dl_data["kurtosis_in"]
    k_out = dl_data["kurtosis_out"]

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

    if mecc_data is not None:
        mk_in  = mecc_data["kurtosis_in"]
        mk_out = mecc_data["kurtosis_out"]
        mk_in_clipped  = mk_in[(mk_in  >= x_min) & (mk_in  <= x_max)]
        mk_out_clipped = mk_out[(mk_out >= x_min) & (mk_out <= x_max)]
        if len(mk_in_clipped):
            ax.step(bins[:-1], np.histogram(mk_in_clipped,  bins=bins)[0],
                    where='post', color=MECC_INPLANE_COLOR,  linewidth=1.6, alpha=0.85, label='MECC 面内')
        if len(mk_out_clipped):
            ax.step(bins[:-1], np.histogram(mk_out_clipped, bins=bins)[0],
                    where='post', color=MECC_OUTPLANE_COLOR, linewidth=1.6, alpha=0.85, label='MECC 面外')

    ax.axvline(x=0, color='gray', linewidth=1.0, linestyle='-', alpha=0.8, zorder=3)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('峭度（Fisher 定义）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('样本数（个）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('涡激共振峭度分布（DL vs MECC）', fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振峭度分布直方图（DL vs MECC）")
    print("=" * 80)

    print(f"\n[步骤1] 从 enriched_stats 加载 DL 峭度数据...")
    dl_stats = load_enriched_stats(Config.ENRICHED_STATS_DIR)
    print(f"✓ DL 面内：{len(dl_stats['kurtosis_in'])}，面外：{len(dl_stats['kurtosis_out'])}")

    print(f"\n[步骤2] 加载 MECC 识别结果并计算统计量...")
    mecc_result  = load_latest_result(Config.MECC_RESULT_GLOB)
    mecc_samples = get_viv_samples(mecc_result, max_n=Config.MAX_SAMPLES)
    print(f"  MECC VIV 样本：{len(mecc_samples)} 个")
    mecc_stats = compute_signal_stats(mecc_samples, source='MECC')
    print(f"✓ MECC 面内：{len(mecc_stats['kurtosis_in'])}，面外：{len(mecc_stats['kurtosis_out'])}")

    print("\n[步骤3] 绘制图像...")
    fig = plot_kurtosis_histogram(dl_stats, mecc_stats)
    print("✓ 图像生成完成")

    print("\n[步骤4] 推送到 WebUI...")
    web_push(fig, page='fig4_23 VIV峭度 DL vs MECC', slot=0,
             title='峭度分布对比', page_cols=1)
    print("✓ 推送完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
