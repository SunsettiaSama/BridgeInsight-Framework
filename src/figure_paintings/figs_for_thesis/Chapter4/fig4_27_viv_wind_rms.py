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
    load_latest_result, get_viv_samples,
    build_enriched_lookup, load_mecc_wind_by_sensor,
    MECC_INPLANE_COLOR, MECC_OUTPLANE_COLOR,
)


# ==================== 常量配置 ====================
class Config:
    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'
    SCATTER_ALPHA = 0.35
    SCATTER_SIZE  = 18

    INPLANE_COLOR  = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR

    ENRICHED_STATS_DIR = (
        project_root / "results" / "enriched_stats" / "class_1_viv"
    )

    SENSOR_GROUPS = {
        'C18 边跨': 'ST-VIC-C18-101-01.json',
        'C34 边跨': 'ST-VIC-C34-101-01.json',
        'C34 跨中': 'ST-VIC-C34-201-01.json',
        'C34 辅跨': 'ST-VIC-C34-301-01.json',
    }

    MECC_RESULT_GLOB = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"


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
def plot_wind_vs_vibration(
    dl_data: dict,
    location_name: str = '',
    mecc_data: dict | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    wind  = dl_data["wind_speeds"]
    r_in  = dl_data["rms_inplane"]
    r_out = dl_data["rms_outplane"]
    ax.scatter(wind, r_in,  color=Config.INPLANE_COLOR,  s=Config.SCATTER_SIZE,
               alpha=Config.SCATTER_ALPHA, edgecolors='none', label='DL 面内')
    ax.scatter(wind, r_out, color=Config.OUTPLANE_COLOR, s=Config.SCATTER_SIZE,
               alpha=Config.SCATTER_ALPHA, edgecolors='none', label='DL 面外')

    if mecc_data is not None and len(mecc_data.get("wind_speeds", [])) > 0:
        mw   = mecc_data["wind_speeds"]
        mr_i = mecc_data["rms_inplane"]
        mr_o = mecc_data["rms_outplane"]
        ax.scatter(mw, mr_i, color=MECC_INPLANE_COLOR,  s=Config.SCATTER_SIZE,
                   alpha=Config.SCATTER_ALPHA, edgecolors='none', marker='D', label='MECC 面内')
        ax.scatter(mw, mr_o, color=MECC_OUTPLANE_COLOR, s=Config.SCATTER_SIZE,
                   alpha=Config.SCATTER_ALPHA, edgecolors='none', marker='D', label='MECC 面外')

    ax.set_xlabel('平均风速（m/s）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('加速度RMS（m/s²）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    if location_name:
        ax.set_title(f'涡激共振风速–振动RMS关系（{location_name}）[DL vs MECC]',
                     fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振平均风速–振动RMS散点图（DL vs MECC）")
    print("=" * 80)

    print("\n[步骤1] 加载 MECC 识别结果并构建风数据查找表...")
    mecc_result  = load_latest_result(Config.MECC_RESULT_GLOB)
    mecc_samples = get_viv_samples(mecc_result)
    print(f"  MECC VIV 样本：{len(mecc_samples)} 个")
    wind_lookup   = build_enriched_lookup(Config.ENRICHED_STATS_DIR)
    print(f"  风数据查找表：{len(wind_lookup)} 条记录")
    mecc_by_sensor = load_mecc_wind_by_sensor(mecc_samples, wind_lookup, Config.SENSOR_GROUPS)
    print(f"  MECC 匹配传感器：{list(mecc_by_sensor.keys())}")

    print("\n[步骤2] 按传感器生成图像并推送 WebUI...")
    PAGE = 'fig4_27 风速-RMS DL vs MECC'
    slot = 0
    stats_dir = Config.ENRICHED_STATS_DIR

    for location_name, filename in Config.SENSOR_GROUPS.items():
        json_file = stats_dir / filename
        print(f"\n  [加载 DL] {location_name}：{filename}")
        dl_data = load_wind_vib_data(json_file)
        n_dl = len(dl_data["wind_speeds"])
        if n_dl == 0:
            print("  跳过（DL 无有效样本）")
            continue
        print(f"  DL 有效样本：{n_dl}")

        mecc_data = mecc_by_sensor.get(location_name)
        if mecc_data:
            print(f"  MECC 匹配样本：{len(mecc_data['wind_speeds'])}")
        else:
            print(f"  MECC 无匹配样本（仅显示 DL）")

        fig = plot_wind_vs_vibration(dl_data, location_name=location_name, mecc_data=mecc_data)
        web_push(fig, page=PAGE, slot=slot, title=location_name,
                 page_cols=2 if slot == 0 else None)
        slot += 1
        print(f"  ✓ 图像推送完成（slot {slot - 1}）")

    print("\n" + "=" * 80)
    print(f"共推送 {slot} 张图表")
    print("=" * 80)


if __name__ == "__main__":
    main()
