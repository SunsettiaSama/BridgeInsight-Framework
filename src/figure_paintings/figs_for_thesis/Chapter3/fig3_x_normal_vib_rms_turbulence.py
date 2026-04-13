import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, ENG_FONT, FONT_SIZE, REC_FIG_SIZE,
    get_blue_color_map,
)


# ==================== 常量配置 ====================
class Config:
    FIG_SIZE      = REC_FIG_SIZE
    SCATTER_ALPHA = 0.30
    SCATTER_SIZE  = 14
    GRID_COLOR    = 'gray'
    GRID_ALPHA    = 0.28
    GRID_LS       = '--'

    _pal           = get_blue_color_map(style='discrete', start_map_index=1, end_map_index=5).colors
    INPLANE_COLOR  = _pal[2]
    OUTPLANE_COLOR = _pal[3]

    # 面内/面外使用不同标记形状，提升黑白打印可读性
    INPLANE_MARKER  = 'o'
    OUTPLANE_MARKER = 's'

    ENRICHED_STATS_DIR = project_root / "results" / "enriched_stats" / "class_0_normal"

    SENSOR_GROUPS = {
        'C18 边跨': 'ST-VIC-C18-101-01.json',
        'C34 边跨': 'ST-VIC-C34-101-01.json',
        'C34 跨中': 'ST-VIC-C34-201-01.json',
    }

    # 紊流度 x 轴上限（百分比，固定范围）
    TI_X_MAX_PCT = 60.0


# ==================== 数据加载 ====================
def _load_single(json_file: Path) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tis:      list[float] = []
    rms_in:   list[float] = []
    rms_out:  list[float] = []
    speeds:   list[float] = []

    for sample in raw["samples"]:
        wl      = sample.get("wind_stats") or []
        ts_in   = sample.get("time_stats_inplane")  or {}
        ts_out  = sample.get("time_stats_outplane") or {}

        ri = ts_in.get("rms")
        ro = ts_out.get("rms")
        if ri is None or ro is None:
            continue

        ti_list  = [w["turbulence_intensity"] for w in wl
                    if w.get("turbulence_intensity") is not None]
        spd_list = [w["mean_wind_speed"] for w in wl
                    if w.get("mean_wind_speed") is not None]

        if not ti_list:
            continue

        ti_mean  = float(np.nanmean(ti_list))
        spd_mean = float(np.nanmean(spd_list)) if spd_list else np.nan

        if not np.isfinite(ti_mean):
            continue

        tis.append(ti_mean)
        rms_in.append(float(ri))
        rms_out.append(float(ro))
        speeds.append(spd_mean)

    return {
        "ti":      np.array(tis,     dtype=np.float64),
        "rms_in":  np.array(rms_in,  dtype=np.float64),
        "rms_out": np.array(rms_out, dtype=np.float64),
        "speeds":  np.array(speeds,  dtype=np.float64),
    }


def load_all_data() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label, fname in Config.SENSOR_GROUPS.items():
        jf = Config.ENRICHED_STATS_DIR / fname
        if not jf.exists():
            raise FileNotFoundError(f"数据文件不存在：{jf}")
        print(f"  读取 [{label}] {fname} ...")
        out[label] = _load_single(jf)
    return out


# ==================== 辅助 ====================
def _apply_grid(ax: plt.Axes) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LS)
    ax.set_axisbelow(True)


def _format_axes(ax: plt.Axes, x_max: float) -> None:
    ax.set_xlim(left=0, right=x_max * 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='both'))
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
    ax.set_xlabel('紊流度（%）', labelpad=10,
                  fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('加速度 RMS（m/s²）', labelpad=10,
                  fontproperties=CN_FONT, fontsize=FONT_SIZE)


def _add_legend(ax: plt.Axes) -> None:
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for t in leg.get_texts():
        t.set_fontproperties(CN_FONT)


# ==================== 绘图（单位置）====================
def plot_ti_vs_rms(data: dict, label: str) -> plt.Figure:
    ti      = data["ti"] * 100.0   # 转为百分比显示
    rms_in  = data["rms_in"]
    rms_out = data["rms_out"]

    x_max = Config.TI_X_MAX_PCT
    mask  = ti <= x_max

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    ax.scatter(
        ti[mask], rms_in[mask],
        color=Config.INPLANE_COLOR,
        marker=Config.INPLANE_MARKER,
        s=Config.SCATTER_SIZE,
        alpha=Config.SCATTER_ALPHA,
        edgecolors='none',
        label='面内',
    )
    ax.scatter(
        ti[mask], rms_out[mask],
        color=Config.OUTPLANE_COLOR,
        marker=Config.OUTPLANE_MARKER,
        s=Config.SCATTER_SIZE,
        alpha=Config.SCATTER_ALPHA,
        edgecolors='none',
        label='面外',
    )

    ax.set_title(
        f'随机振动紊流度–振动RMS关系（{label}）',
        fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14,
    )
    _format_axes(ax, x_max)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 图3.15 随机振动紊流度–振动RMS散点图")
    print("=" * 80)

    print("\n[步骤 1/2] 加载各传感器数据...")
    sensor_data = load_all_data()

    for label, d in sensor_data.items():
        n = len(d["ti"])
        print(f"  {label}：有效样本 {n}，"
              f"TI mean={d['ti'].mean():.3f}，"
              f"面内RMS mean={d['rms_in'].mean():.4f}，"
              f"面外RMS mean={d['rms_out'].mean():.4f}")

    print("\n[步骤 2/2] 生成散点图...")
    ploter = PlotLib()

    for label, d in sensor_data.items():
        fig = plot_ti_vs_rms(d, label)
        ploter.figs.append(fig)
        print(f"  ✓ {label}")

    print(f"\n共生成 {len(ploter.figs)} 张图像")
    print("=" * 80)
    ploter.show()


if __name__ == "__main__":
    main()
