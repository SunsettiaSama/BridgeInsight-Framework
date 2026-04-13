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
    CN_FONT, ENG_FONT, FONT_SIZE, REC_FIG_SIZE,
    get_blue_color_map,
)


# ==================== 常量配置 ====================
class Config:
    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'
    SCATTER_ALPHA = 0.35
    SCATTER_SIZE  = 18

    _palette = get_blue_color_map(style='discrete', start_map_index=1, end_map_index=5).colors
    INPLANE_COLOR  = _palette[2]
    OUTPLANE_COLOR = _palette[3]

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_0_normal"
    )

    # 每个位置对应的面内传感器 JSON 文件名（文件按面内传感器命名）
    SENSOR_GROUPS = {
        '边跨': 'ST-VIC-C34-101-01.json',
        '跨中': 'ST-VIC-C34-201-01.json',
    }


# ==================== 数据加载 ====================
def load_wind_vib_data(json_file: Path) -> dict:
    if not json_file.exists():
        raise FileNotFoundError(f"数据文件不存在：{json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    wind_speeds:  list[float] = []
    rms_inplane:  list[float] = []
    rms_outplane: list[float] = []

    for sample in data["samples"]:
        wind_list = sample.get("wind_stats") or []
        ts_in     = sample.get("time_stats_inplane")  or {}
        ts_out    = sample.get("time_stats_outplane") or {}

        rms_in  = ts_in.get("rms")
        rms_out = ts_out.get("rms")

        if not wind_list or rms_in is None or rms_out is None:
            continue

        # 多个风传感器时取均值
        speeds = [w["mean_wind_speed"] for w in wind_list
                  if w.get("mean_wind_speed") is not None]
        if not speeds:
            continue

        wind_speeds.append(float(np.mean(speeds)))
        rms_inplane.append(float(rms_in))
        rms_outplane.append(float(rms_out))

    return {
        "wind_speeds":  np.array(wind_speeds,  dtype=np.float64),
        "rms_inplane":  np.array(rms_inplane,  dtype=np.float64),
        "rms_outplane": np.array(rms_outplane, dtype=np.float64),
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
def plot_wind_vs_vibration(data: dict, location_name: str = '') -> plt.Figure:
    wind  = data["wind_speeds"]
    r_in  = data["rms_inplane"]
    r_out = data["rms_outplane"]

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    ax.scatter(wind, r_in,
               color=Config.INPLANE_COLOR,  s=Config.SCATTER_SIZE,
               alpha=Config.SCATTER_ALPHA,  edgecolors='none',
               label='面内')
    ax.scatter(wind, r_out,
               color=Config.OUTPLANE_COLOR, s=Config.SCATTER_SIZE,
               alpha=Config.SCATTER_ALPHA,  edgecolors='none',
               label='面外')

    ax.set_xlabel('平均风速（m/s）', labelpad=10,
                  fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('加速度RMS（m/s²）', labelpad=10,
                  fontproperties=CN_FONT, fontsize=FONT_SIZE)
    if location_name:
        ax.set_title(f'随机振动风速–振动RMS关系（{location_name}）',
                     fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 图3.12 随机振动平均风速–振动RMS散点图")
    print("=" * 80)

    stats_dir = Config.ENRICHED_STATS_DIR
    ploter = PlotLib()

    for location_name, filename in Config.SENSOR_GROUPS.items():
        json_file = stats_dir / filename
        print(f"\n[加载] {location_name}：{filename}")

        data = load_wind_vib_data(json_file)

        n = len(data["wind_speeds"])
        print(f"  ✓ 有效样本：{n}")
        print(f"  风速范围：{data['wind_speeds'].min():.2f} ~ "
              f"{data['wind_speeds'].max():.2f} m/s")
        print(f"  面内RMS：mean={data['rms_inplane'].mean():.4f}  "
              f"max={data['rms_inplane'].max():.4f}")
        print(f"  面外RMS：mean={data['rms_outplane'].mean():.4f}  "
              f"max={data['rms_outplane'].max():.4f}")

        fig = plot_wind_vs_vibration(data, location_name=location_name)
        ploter.figs.append(fig)
        print(f"  ✓ 图像生成完成")

    print("\n" + "=" * 80)
    print(f"共生成 {len(ploter.figs)} 张图表")
    print("=" * 80)
    ploter.show()


if __name__ == "__main__":
    main()
