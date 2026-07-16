import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import (
    filter_sensor_groups,
    get_enriched_class_dir,
)
from src.figure_paintings.figs_for_thesis.config import CN_FONT, FONT_SIZE, REC_FIG_SIZE, get_blue_color_map
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    NORMAL_VIB_CLASS_ID = 0
    FEATURE_BATCH_SIZE = 512

    FIG_SIZE = (REC_FIG_SIZE[0] * 1.35, REC_FIG_SIZE[1])
    LABEL_FONT_SIZE = FONT_SIZE
    TICK_FONT_SIZE = FONT_SIZE - 4
    LEGEND_FONT_SIZE = FONT_SIZE - 4

    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"

    _palette = get_blue_color_map(style="discrete", start_map_index=1, end_map_index=5).colors
    INPLANE_COLOR = _palette[2]
    OUTPLANE_COLOR = _palette[3]
    SHADE_ALPHA = 0.18
    LINE_WIDTH = 2.3
    MARKER_SIZE = 4

    WIND_BIN_WIDTH = 0.5
    MIN_BIN_SAMPLES = 80
    WIND_X_PERCENTILE = 99.5
    RMS_Y_PERCENTILE = 99.5
    AXIS_PAD = 1.05

    ENRICHED_STATS_DIR = get_enriched_class_dir(NORMAL_VIB_CLASS_ID)
    SENSOR_GROUPS = filter_sensor_groups(data_config.SENSOR_GROUPS_WIND)
    WEB_DASHBOARD_PORT = 15678
    SNAPSHOT_DIR = project_root / "results" / "chapter4_characteristics" / "figure_snapshots"
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_13_normal_vib_wind_rms.npz"


def _snapshot_config() -> dict:
    return {
        "figure": "fig4_13_normal_vib_wind_rms",
        "class_id": Config.NORMAL_VIB_CLASS_ID,
        "sensor_groups": Config.SENSOR_GROUPS,
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
        "data_source": data_config.DATA_SOURCE,
    }


def load_wind_rms_data(json_file: Path) -> dict:
    if not json_file.exists():
        raise FileNotFoundError(f"数据文件不存在：{json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    wind_speeds: list[float] = []
    rms_inplane: list[float] = []
    rms_outplane: list[float] = []

    for sample in data["samples"]:
        wind_list = sample.get("wind_stats") or []
        ts_in = sample.get("time_stats_inplane") or {}
        ts_out = sample.get("time_stats_outplane") or {}
        rms_in = ts_in.get("rms")
        rms_out = ts_out.get("rms")
        if not wind_list or rms_in is None or rms_out is None:
            continue

        rms_in = float(rms_in)
        rms_out = float(rms_out)
        speeds = [
            float(w["mean_wind_speed"])
            for w in wind_list
            if w.get("mean_wind_speed") is not None and np.isfinite(float(w["mean_wind_speed"]))
        ]
        if not speeds:
            continue
        if not np.isfinite(rms_in) or not np.isfinite(rms_out):
            continue

        wind_speeds.append(float(np.mean(speeds)))
        rms_inplane.append(rms_in)
        rms_outplane.append(rms_out)

    return {
        "wind_speeds": np.asarray(wind_speeds, dtype=np.float64),
        "rms_inplane": np.asarray(rms_inplane, dtype=np.float64),
        "rms_outplane": np.asarray(rms_outplane, dtype=np.float64),
    }


def load_all_locations() -> dict[str, dict]:
    ensure_enriched_for_figures(class_id=Config.NORMAL_VIB_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)

    by_location: dict[str, dict] = {}
    for location_name, filename in Config.SENSOR_GROUPS.items():
        json_file = Config.ENRICHED_STATS_DIR / filename
        print(f"  加载：{location_name} - {filename}")
        data = load_wind_rms_data(json_file)
        if len(data["wind_speeds"]) == 0:
            raise ValueError(f"{location_name} 无有效风速-RMS 样本：{json_file}")
        by_location[location_name] = data

    if not by_location:
        raise ValueError("无可绘制的随机振动风速-RMS 数据")
    return by_location


def load_snapshot(force_refresh: bool) -> dict[str, dict] | None:
    path = Config.SNAPSHOT_PATH
    if force_refresh or not path.exists():
        return None

    payload = np.load(path, allow_pickle=True)
    saved_cfg = json.loads(str(payload["config_json"]))
    if saved_cfg != _snapshot_config():
        print(f"  快照参数不匹配，将重新读取 enriched 数据：{path}")
        return None

    location_names = json.loads(str(payload["location_names_json"]))
    by_location: dict[str, dict] = {}
    for idx, location_name in enumerate(location_names):
        by_location[location_name] = {
            "wind_speeds": np.asarray(payload[f"wind_{idx}"], dtype=np.float64),
            "rms_inplane": np.asarray(payload[f"rms_inplane_{idx}"], dtype=np.float64),
            "rms_outplane": np.asarray(payload[f"rms_outplane_{idx}"], dtype=np.float64),
        }

    print(f"  读取结果快照：{path}")
    print(f"    created_at={payload['created_at']}  locations={len(by_location)}")
    return by_location


def save_snapshot(by_location: dict[str, dict]) -> None:
    path = Config.SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "created_at": np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "config_json": np.asarray(json.dumps(_snapshot_config(), ensure_ascii=False)),
        "location_names_json": np.asarray(json.dumps(list(by_location.keys()), ensure_ascii=False)),
    }
    for idx, data in enumerate(by_location.values()):
        arrays[f"wind_{idx}"] = np.asarray(data["wind_speeds"], dtype=np.float64)
        arrays[f"rms_inplane_{idx}"] = np.asarray(data["rms_inplane"], dtype=np.float64)
        arrays[f"rms_outplane_{idx}"] = np.asarray(data["rms_outplane"], dtype=np.float64)

    np.savez_compressed(path, **arrays)
    print(f"  写出结果快照：{path}")


def load_or_build_snapshot(force_refresh: bool) -> dict[str, dict]:
    cached = load_snapshot(force_refresh=force_refresh)
    if cached is not None:
        return cached

    print("  未命中快照，开始读取 enriched JSON ...")
    by_location = load_all_locations()
    save_snapshot(by_location)
    return by_location


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax) -> None:
    leg = ax.legend(loc="upper right", fontsize=Config.LEGEND_FONT_SIZE, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)


def _is_web_dashboard_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _shared_axis_limits(by_location: dict[str, dict]) -> tuple[float, float]:
    wind = np.concatenate([data["wind_speeds"] for data in by_location.values()])
    rms = np.concatenate(
        [
            np.concatenate([data["rms_inplane"], data["rms_outplane"]])
            for data in by_location.values()
        ]
    )
    wind = wind[np.isfinite(wind)]
    rms = rms[np.isfinite(rms)]
    if len(wind) == 0 or len(rms) == 0:
        raise ValueError("无有效有限值用于设置风速-RMS 坐标轴范围")
    x_max = max(float(np.percentile(wind, Config.WIND_X_PERCENTILE)) * Config.AXIS_PAD, 1e-6)
    y_max = max(float(np.percentile(rms, Config.RMS_Y_PERCENTILE)) * Config.AXIS_PAD, 1e-6)
    return x_max, y_max


def _bin_stats(wind: np.ndarray, rms: np.ndarray, bins: np.ndarray) -> dict:
    centers: list[float] = []
    q25: list[float] = []
    q50: list[float] = []
    q75: list[float] = []
    counts: list[int] = []

    bin_idx = np.digitize(wind, bins) - 1
    for i in range(len(bins) - 1):
        values = rms[bin_idx == i]
        values = values[np.isfinite(values)]
        if len(values) < Config.MIN_BIN_SAMPLES:
            continue
        centers.append(float((bins[i] + bins[i + 1]) / 2))
        q25.append(float(np.percentile(values, 25)))
        q50.append(float(np.percentile(values, 50)))
        q75.append(float(np.percentile(values, 75)))
        counts.append(int(len(values)))

    return {
        "centers": np.asarray(centers, dtype=np.float64),
        "q25": np.asarray(q25, dtype=np.float64),
        "q50": np.asarray(q50, dtype=np.float64),
        "q75": np.asarray(q75, dtype=np.float64),
        "counts": np.asarray(counts, dtype=np.int32),
    }


def _plot_side_band(ax, stats: dict, color: str, label: str) -> None:
    if len(stats["centers"]) == 0:
        return

    ax.fill_between(
        stats["centers"],
        stats["q25"],
        stats["q75"],
        color=color,
        alpha=Config.SHADE_ALPHA,
        linewidth=0,
    )
    ax.plot(
        stats["centers"],
        stats["q50"],
        color=color,
        linewidth=Config.LINE_WIDTH,
        marker="o",
        markersize=Config.MARKER_SIZE,
        label=label,
    )


def plot_wind_rms_trend(by_location: dict[str, dict]) -> plt.Figure:
    x_max, y_max = _shared_axis_limits(by_location)
    n_cols = len(by_location)
    fig, axes = plt.subplots(1, n_cols, figsize=Config.FIG_SIZE, sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes)
    bins = np.arange(0, x_max + Config.WIND_BIN_WIDTH, Config.WIND_BIN_WIDTH)

    for ax, (location_name, data) in zip(axes_arr, by_location.items()):
        wind = data["wind_speeds"]
        finite = np.isfinite(wind) & np.isfinite(data["rms_inplane"]) & np.isfinite(data["rms_outplane"])
        wind = wind[finite]
        rms_in = data["rms_inplane"][finite]
        rms_out = data["rms_outplane"][finite]

        stats_in = _bin_stats(wind, rms_in, bins)
        stats_out = _bin_stats(wind, rms_out, bins)
        _plot_side_band(ax, stats_in, Config.INPLANE_COLOR, "面内")
        _plot_side_band(ax, stats_out, Config.OUTPLANE_COLOR, "面外")
        ax.set_title(location_name, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE, pad=12)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
        _apply_grid(ax)

    axes_arr[0].set_ylabel(r"加速度 RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    for ax in axes_arr:
        ax.set_xlabel("平均风速（m/s）", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    _add_legend(axes_arr[-1])
    fig.tight_layout()
    return fig


def push_figure(fig: plt.Figure) -> None:
    if not _is_web_dashboard_available(Config.WEB_DASHBOARD_PORT):
        print(
            "  未检测到 VibDash 服务，跳过 WebUI 推送；"
            "如需预览请先运行：python -m src.visualize_tools.web_dashboard"
        )
        return

    page = "fig4_13 随机振动风速-RMS"
    web_push(fig, page=page, slot=0, title="随机振动风速-RMS关系", page_cols=1)
    print(f"[OK] 已推送到 WebUI：{page}")


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-13 随机振动风速-RMS 分箱趋势图（支持快照）")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略已有快照，强制从 enriched JSON 重建")
    args = parser.parse_args()

    print("=" * 80)
    print("图4-13 随机振动风速-RMS关系分箱趋势图")
    print("=" * 80)
    print(f"\n[步骤1] 加载随机振动风速-RMS 数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    print(f"  快照文件：{Config.SNAPSHOT_PATH}")
    if args.refresh_cache:
        print("  模式：--refresh-cache（强制刷新快照）")
    else:
        print("  模式：优先读快照，缺失才读取 enriched JSON")
    by_location = load_or_build_snapshot(force_refresh=args.refresh_cache)
    for location_name, data in by_location.items():
        print(
            f"  {location_name}: n={len(data['wind_speeds'])}, "
            f"wind={data['wind_speeds'].min():.2f}-{data['wind_speeds'].max():.2f} m/s, "
            f"RMS median 面内={float(np.median(data['rms_inplane'])):.4f}, "
            f"面外={float(np.median(data['rms_outplane'])):.4f}"
        )

    print("\n[步骤2] 绘制图像...")
    fig = plot_wind_rms_trend(by_location)
    push_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
