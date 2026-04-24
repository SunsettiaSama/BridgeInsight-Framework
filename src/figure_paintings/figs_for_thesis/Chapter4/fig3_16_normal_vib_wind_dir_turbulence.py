import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, ENG_FONT, SQUARE_FONT_SIZE, get_red_color_map, SQUARE_FIG_SIZE
)

FONT_SIZE = SQUARE_FONT_SIZE
# ==================== 配色：高对比，去掉粉色区间 ====================
# 全量色板取深紫→中蓝→橙黄→橙→珊瑚红→深红，跳过所有淡色
# _HC_COLORS = [
#     '#8074C8',   # 深紫（低风速）
#     '#7895C1',   # 中蓝
#     '#F0C284',   # 橙黄（跳过所有浅蓝/浅黄过渡区）
#     '#EF8B67',   # 橙
#     '#E3625D',   # 珊瑚红
#     '#B54764',   # 深玫红
#     '#992224',   # 深红（高风速）
# ]
_CMAP = get_red_color_map(style='gradient')


# ==================== 常量配置 ====================
class Config:
    INTERVAL_NUMS   = 36       # 风向分箱数（每 10°）
    AXIS_OF_BRIDGE  = 10.6    # 桥轴线角度（°）
    MIN_BIN_SAMPLES = 3       # 分箱最少样本数

    # 正方形图：极区占左侧 ~8/10，colorbar 占右侧
    FIG_W           = SQUARE_FIG_SIZE[0]
    FIG_H           = SQUARE_FIG_SIZE[1]
    GRID_ALPHA      = 0.22
    BAR_ALPHA       = 0.90
    CMAP            = _CMAP

    ENRICHED_STATS_DIR = project_root / "results" / "enriched_stats" / "class_0_normal"

    # 3 个位置 × 2 方向 = 6 张独立图
    SENSOR_GROUPS = {
        'C18 边跨': 'ST-VIC-C18-101-01.json',
        'C34 跨中': 'ST-VIC-C34-201-01.json',
    }


# ==================== 数据加载 ====================
def _load_single(json_file: Path) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dirs, tis, r_in, r_out = [], [], [], []

    for sample in raw["samples"]:
        wl    = sample.get("wind_stats") or []
        ts_in  = sample.get("time_stats_inplane")  or {}
        ts_out = sample.get("time_stats_outplane") or {}

        ri = ts_in.get("rms")
        ro = ts_out.get("rms")

        d_list  = [w["mean_wind_direction"] for w in wl
                   if w.get("mean_wind_direction") is not None]
        ti_list = [w["turbulence_intensity"] for w in wl
                   if w.get("turbulence_intensity") is not None]

        if not d_list or ri is None or ro is None:
            continue

        corrected = (360.0 - float(np.mean(d_list))) % 360.0
        dirs.append(corrected)
        ti_mean = float(np.nanmean(ti_list)) if ti_list else np.nan
        tis.append(ti_mean)
        r_in.append(float(ri))
        r_out.append(float(ro))

    return {
        "wind_dirs":   np.array(dirs),
        "turb_intens": np.array(tis),
        "rms_in":      np.array(r_in),
        "rms_out":     np.array(r_out),
    }


def load_sensor_data() -> dict[str, dict]:
    stats_dir = Config.ENRICHED_STATS_DIR
    out: dict[str, dict] = {}
    for label, fname in Config.SENSOR_GROUPS.items():
        jf = stats_dir / fname
        if not jf.exists():
            raise FileNotFoundError(f"找不到数据文件：{jf}")
        print(f"  读取 [{label}] {fname} ...")
        out[label] = _load_single(jf)
    return out


# ==================== 分箱聚合 ====================
def _bin_by_direction(
    wind_dirs: np.ndarray,
    values: np.ndarray,
    colorvar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n     = Config.INTERVAL_NUMS
    step  = 360.0 / n
    bins  = np.arange(0, 360 + step, step)
    digi  = np.digitize(wind_dirs, bins)

    mean_rms   = np.full(n, np.nan)
    mean_color = np.full(n, np.nan)

    for i in range(1, len(bins)):
        mask = digi == i
        if mask.sum() >= Config.MIN_BIN_SAMPLES:
            mean_rms[i - 1]   = float(np.mean(values[mask]))
            cv = colorvar[mask]
            cv_fin = cv[np.isfinite(cv)]
            if len(cv_fin):
                mean_color[i - 1] = float(np.mean(cv_fin))

    return mean_rms, mean_color, bins


# ==================== 全局 vmin / vmax ====================
def compute_global_ti_range(sensor_data: dict[str, dict]) -> tuple[float, float]:
    all_ti: list[np.ndarray] = []
    for d in sensor_data.values():
        ti = d["turb_intens"]
        all_ti.append(ti[np.isfinite(ti)])
    combined = np.concatenate(all_ti)
    return float(combined.min()), float(combined.max())


# ==================== 绘制单张独立图 ====================
def _draw_polar(ax, mean_rms, mean_ti, bins, title, vmin, vmax):
    cmap     = Config.CMAP
    n        = Config.INTERVAL_NUMS
    step     = 360.0 / n
    theta    = np.deg2rad(bins[:-1])
    width    = np.deg2rad(step)
    heights  = np.where(np.isfinite(mean_rms), mean_rms, 0.0)
    norm     = plt.Normalize(vmin=vmin, vmax=vmax)

    bars = ax.bar(theta, heights, width=width, bottom=0.0,
                  align='edge', alpha=Config.BAR_ALPHA, edgecolor='none')

    for bar, ti in zip(bars, mean_ti):
        if np.isfinite(ti):
            bar.set_facecolor(cmap(norm(ti)))
        else:
            bar.set_alpha(0.0)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                       fontproperties=ENG_FONT, fontsize=FONT_SIZE - 3)

    y_max = float(np.nanmax(heights)) if np.any(heights > 0) else 1e-6
    ax.set_ylim(0, y_max * 1.18)

    for ang in [Config.AXIS_OF_BRIDGE, Config.AXIS_OF_BRIDGE + 180]:
        ax.plot([np.deg2rad(ang)] * 2, [0, y_max * 1.12],
                color='#CC3333', lw=1.2, ls='--', zorder=5)
    ax.annotate('桥轴线',
                xy=(np.deg2rad(Config.AXIS_OF_BRIDGE), y_max * 0.95),
                ha='center', va='bottom',
                fontproperties=CN_FONT, fontsize=FONT_SIZE - 5, color='#CC3333')

    r_ticks = np.linspace(0, y_max, 5)[1:]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f'{v:.3f}' for v in r_ticks],
                       fontproperties=ENG_FONT, fontsize=FONT_SIZE - 6)
    ax.set_rlabel_position(255)
    ax.yaxis.grid(True, alpha=Config.GRID_ALPHA, ls='--')
    ax.xaxis.grid(True, alpha=Config.GRID_ALPHA, ls='--')
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE - 1, pad=16)


def make_single_figure(
    mean_rms: np.ndarray,
    mean_spd: np.ndarray,
    bins: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(Config.FIG_W, Config.FIG_H))

    # 极坐标区：顶部留 0.10 给 title，底部留 0.03，右侧留给颜色条
    ax = fig.add_axes([0.03, 0.03, 0.81, 0.84], projection='polar')

    # 颜色条：纵向范围与极坐标区对齐
    cax = fig.add_axes([0.87, 0.10, 0.025, 0.65])

    _draw_polar(ax, mean_rms, mean_spd, bins, title, vmin, vmax)

    sm = plt.cm.ScalarMappable(
        cmap=Config.CMAP,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('紊流度（%）', fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x * 100:.0f}')
    )
    cbar.ax.tick_params(labelsize=FONT_SIZE - 4)

    return fig


# ==================== 合并所有传感器 ====================
def merge_all_sensors(sensor_data: dict[str, dict]) -> dict:
    dirs    = np.concatenate([d["wind_dirs"]   for d in sensor_data.values()])
    tis     = np.concatenate([d["turb_intens"] for d in sensor_data.values()])
    rms_in  = np.concatenate([d["rms_in"]      for d in sensor_data.values()])
    rms_out = np.concatenate([d["rms_out"]     for d in sensor_data.values()])
    return {
        "wind_dirs":   dirs,
        "turb_intens": tis,
        "rms_in":      rms_in,
        "rms_out":     rms_out,
    }


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 图3.13 随机振动 风向–RMS 极坐标图（6 张独立图）")
    print("=" * 80)

    print("\n[步骤 1/3] 加载各传感器数据...")
    sensor_data = load_sensor_data()

    for label, d in sensor_data.items():
        n = len(d["wind_dirs"])
        print(f"  {label}：有效样本 {n}，"
              f"面内RMS mean={d['rms_in'].mean():.4f}，"
              f"面外RMS mean={d['rms_out'].mean():.4f}")

    print("\n[步骤 2/3] 设定全局紊流度映射范围（vmin / vmax）...")
    vmin, vmax = 0.0, 0.6
    print(f"  全局紊流度范围：{vmin:.4f} ~ {vmax:.4f}（所有图像共享，显示为 0%~60%）")

    print("\n[步骤 3/3] 生成独立图像（各传感器 + 整体汇总）...")
    n_bins = Config.INTERVAL_NUMS
    step   = 360.0 / n_bins
    bins   = np.arange(0, 360 + step, step)

    ploter = PlotLib()

    for label, d in sensor_data.items():
        for direction, rms_key in [('面内', 'rms_in'), ('面外', 'rms_out')]:
            mean_rms, mean_ti, _ = _bin_by_direction(
                d["wind_dirs"], d[rms_key], d["turb_intens"]
            )
            title = f'{label}  {direction}振动 RMS（m/s²）'
            fig   = make_single_figure(mean_rms, mean_ti, bins, title, vmin, vmax)
            ploter.figs.append(fig)
            print(f"  ✓ {title}")

    print("\n[补充] 生成整体汇总图（所有传感器合并）...")
    all_data = merge_all_sensors(sensor_data)
    for direction, rms_key in [('面内', 'rms_in'), ('面外', 'rms_out')]:
        mean_rms, mean_ti, _ = _bin_by_direction(
            all_data["wind_dirs"], all_data[rms_key], all_data["turb_intens"]
        )
        title = f'整体汇总  {direction}振动 RMS（m/s²）'
        fig   = make_single_figure(mean_rms, mean_ti, bins, title, vmin, vmax)
        ploter.figs.append(fig)
        print(f"  ✓ {title}")

    print(f"\n共生成 {len(ploter.figs)} 张图像")
    print("=" * 80)
    ploter.show()


if __name__ == "__main__":
    main()
