import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, ENG_FONT, SQUARE_FONT_SIZE, get_red_color_map, SQUARE_FIG_SIZE
)

_chapter4_dir = str(Path(__file__).parent)
if _chapter4_dir not in sys.path:
    sys.path.insert(0, _chapter4_dir)
from _viv_pipeline import (
    load_latest_result, get_viv_samples,
    build_enriched_lookup, load_mecc_wind_dir_by_sensor,
)

FONT_SIZE = SQUARE_FONT_SIZE
_CMAP = get_red_color_map(style='gradient')


# ==================== 常量配置 ====================
class Config:
    INTERVAL_NUMS   = 36
    AXIS_OF_BRIDGE  = 10.6
    MIN_BIN_SAMPLES = 3

    FIG_W           = SQUARE_FIG_SIZE[0]
    FIG_H           = SQUARE_FIG_SIZE[1]
    GRID_ALPHA      = 0.22
    BAR_ALPHA       = 0.90
    CMAP            = _CMAP

    ENRICHED_STATS_DIR = project_root / "results" / "enriched_stats" / "class_1_viv"

    SENSOR_GROUPS = {
        'C18 边跨': 'ST-VIC-C18-101-01.json',
        'C34 边跨': 'ST-VIC-C34-101-01.json',
        'C34 跨中': 'ST-VIC-C34-201-01.json',
        'C34 辅跨': 'ST-VIC-C34-301-01.json',
    }

    MECC_RESULT_GLOB = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"


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

    ax  = fig.add_axes([0.03, 0.03, 0.81, 0.84], projection='polar')
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
    print("涡激共振 风向–RMS 极坐标图（DL vs MECC）")
    print("=" * 80)

    print("\n[步骤1] 加载 DL 各传感器数据...")
    dl_sensor_data = load_sensor_data()
    for label, d in dl_sensor_data.items():
        n = len(d["wind_dirs"])
        print(f"  {label}：DL 有效样本 {n}")

    print("\n[步骤2] 加载 MECC 识别结果并匹配风数据...")
    mecc_result  = load_latest_result(Config.MECC_RESULT_GLOB)
    mecc_samples = get_viv_samples(mecc_result)
    print(f"  MECC VIV 样本：{len(mecc_samples)} 个")
    wind_lookup      = build_enriched_lookup(Config.ENRICHED_STATS_DIR)
    mecc_sensor_data = load_mecc_wind_dir_by_sensor(
        mecc_samples, wind_lookup, Config.SENSOR_GROUPS,
    )
    for label, d in mecc_sensor_data.items():
        print(f"  {label}：MECC 匹配 {len(d['wind_dirs'])} 条")

    print("\n[步骤3] 设定全局紊流度范围 0%~60%...")
    vmin, vmax = 0.0, 0.6
    n_bins = Config.INTERVAL_NUMS
    step   = 360.0 / n_bins
    bins   = np.arange(0, 360 + step, step)

    print("\n[步骤4] 生成独立图像并推送 WebUI...")
    PAGE = 'fig4_28 风向极坐标 DL vs MECC'
    slot = 0

    for label, dl_d in dl_sensor_data.items():
        for direction, rms_key in [('面内', 'rms_in'), ('面外', 'rms_out')]:
            # DL polar figure
            mean_rms, mean_ti, _ = _bin_by_direction(
                dl_d["wind_dirs"], dl_d[rms_key], dl_d["turb_intens"]
            )
            title_dl = f'{label} DL {direction}振动 RMS（m/s²）'
            fig_dl   = make_single_figure(mean_rms, mean_ti, bins, title_dl, vmin, vmax)
            web_push(fig_dl, page=PAGE, slot=slot, title=title_dl,
                     page_cols=3 if slot == 0 else None)
            slot += 1

            # MECC polar figure (if data available)
            mecc_d = mecc_sensor_data.get(label)
            if mecc_d is not None and len(mecc_d["wind_dirs"]) >= Config.MIN_BIN_SAMPLES:
                mean_rms_m, mean_ti_m, _ = _bin_by_direction(
                    mecc_d["wind_dirs"], mecc_d[rms_key], mecc_d["turb_intens"]
                )
                title_mecc = f'{label} MECC {direction}振动 RMS（m/s²）'
                fig_mecc   = make_single_figure(mean_rms_m, mean_ti_m, bins, title_mecc, vmin, vmax)
                web_push(fig_mecc, page=PAGE, slot=slot, title=title_mecc)
                slot += 1
                print(f"  ✓ {label} {direction}：DL + MECC 各1张")
            else:
                print(f"  ✓ {label} {direction}：DL 1张（MECC 无数据）")

    print("\n[步骤5] 生成整体汇总图...")
    all_dl = merge_all_sensors(dl_sensor_data)
    for direction, rms_key in [('面内', 'rms_in'), ('面外', 'rms_out')]:
        mean_rms, mean_ti, _ = _bin_by_direction(
            all_dl["wind_dirs"], all_dl[rms_key], all_dl["turb_intens"]
        )
        title = f'DL 整体汇总 {direction}振动 RMS（m/s²）'
        fig   = make_single_figure(mean_rms, mean_ti, bins, title, vmin, vmax)
        web_push(fig, page=PAGE, slot=slot, title=title)
        slot += 1
        print(f"  ✓ {title}")

    print(f"\n共推送 {slot} 张图像")
    print("=" * 80)


if __name__ == "__main__":
    main()
