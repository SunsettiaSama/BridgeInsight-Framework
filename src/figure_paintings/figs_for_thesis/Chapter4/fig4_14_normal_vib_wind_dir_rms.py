"""图4-14：随机振动 风向–RMS–风速 极坐标散点图。

角向 = 风向，径向 = RMS，颜色 = 平均风速；面内 / 面外各一张。
合并全部拉索后随机抽取 int(1e5) 点写入快照；日常运行只读快照。
"""

from __future__ import annotations

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
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    get_red_color_map,
)
from src.visualize_tools.web_dashboard import push as web_push

FONT_SIZE = SQUARE_FONT_SIZE
_CMAP = get_red_color_map(style="gradient")


class Config:
    NORMAL_VIB_CLASS_ID = 0
    FEATURE_BATCH_SIZE = 512

    SAMPLE_N = int(1e5)
    RANDOM_SEED = 42

    AXIS_OF_BRIDGE = 10.6
    # 径向刻度轴方位（WSW）；标签在轴上方、数字在轴下方
    RLABEL_DEG = 245.0
    RLABEL_SIDE_OFFSET_DEG = 7.0
    FIG_W = SQUARE_FIG_SIZE[0]
    FIG_H = SQUARE_FIG_SIZE[1]
    GRID_ALPHA = 0.22
    CMAP = _CMAP

    SCATTER_SIZE = 6
    SCATTER_ALPHA = 0.28
    RMS_R_PERCENTILE = 99.5
    WIND_C_PERCENTILE = 99.5
    AXIS_PAD = 1.08

    ENRICHED_STATS_DIR = get_enriched_class_dir(NORMAL_VIB_CLASS_ID)
    SENSOR_GROUPS = filter_sensor_groups(data_config.SENSOR_GROUPS_WIND)
    WEB_DASHBOARD_PORT = 15678
    WEB_PAGE = "fig4_14 风向-RMS散点"
    SNAPSHOT_DIR = project_root / "results" / "chapter4_characteristics" / "figure_snapshots"
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_14_normal_vib_wind_dir_rms.npz"


def _snapshot_config() -> dict:
    return {
        "figure": "fig4_14_normal_vib_wind_dir_rms",
        "version": "polar_scatter_speed_color_1e5",
        "class_id": Config.NORMAL_VIB_CLASS_ID,
        "sensor_groups": Config.SENSOR_GROUPS,
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
        "data_source": data_config.DATA_SOURCE,
        "sample_n": Config.SAMPLE_N,
        "random_seed": Config.RANDOM_SEED,
    }


def _load_single(json_file: Path) -> dict:
    if not json_file.exists():
        raise FileNotFoundError(f"数据文件不存在：{json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dirs: list[float] = []
    speeds: list[float] = []
    r_in: list[float] = []
    r_out: list[float] = []

    for sample in raw["samples"]:
        wl = sample.get("wind_stats") or []
        ts_in = sample.get("time_stats_inplane") or {}
        ts_out = sample.get("time_stats_outplane") or {}

        ri = ts_in.get("rms")
        ro = ts_out.get("rms")
        d_list = [
            float(w["mean_wind_direction"])
            for w in wl
            if w.get("mean_wind_direction") is not None
            and np.isfinite(float(w["mean_wind_direction"]))
        ]
        s_list = [
            float(w["mean_wind_speed"])
            for w in wl
            if w.get("mean_wind_speed") is not None
            and np.isfinite(float(w["mean_wind_speed"]))
        ]
        if not d_list or not s_list or ri is None or ro is None:
            continue

        corrected = (360.0 - float(np.mean(d_list))) % 360.0
        speed = float(np.mean(s_list))
        ri_f = float(ri)
        ro_f = float(ro)
        if not (
            np.isfinite(corrected)
            and np.isfinite(speed)
            and np.isfinite(ri_f)
            and np.isfinite(ro_f)
        ):
            continue

        dirs.append(corrected)
        speeds.append(speed)
        r_in.append(ri_f)
        r_out.append(ro_f)

    return {
        "wind_dirs": np.asarray(dirs, dtype=np.float64),
        "wind_speeds": np.asarray(speeds, dtype=np.float64),
        "rms_in": np.asarray(r_in, dtype=np.float64),
        "rms_out": np.asarray(r_out, dtype=np.float64),
    }


def load_pooled() -> dict:
    ensure_enriched_for_figures(
        class_id=Config.NORMAL_VIB_CLASS_ID,
        batch_size=Config.FEATURE_BATCH_SIZE,
    )

    dir_parts: list[np.ndarray] = []
    spd_parts: list[np.ndarray] = []
    rms_in_parts: list[np.ndarray] = []
    rms_out_parts: list[np.ndarray] = []

    for location_name, filename in Config.SENSOR_GROUPS.items():
        json_file = Config.ENRICHED_STATS_DIR / filename
        print(f"  加载：{location_name} - {filename}")
        data = _load_single(json_file)
        n = len(data["wind_dirs"])
        if n == 0:
            print(f"  跳过（无有效样本）：{location_name}")
            continue
        print(f"    n={n}")
        dir_parts.append(data["wind_dirs"])
        spd_parts.append(data["wind_speeds"])
        rms_in_parts.append(data["rms_in"])
        rms_out_parts.append(data["rms_out"])

    if not dir_parts:
        raise ValueError("无可绘制的随机振动风向-风速-RMS 数据")

    return {
        "wind_dirs": np.concatenate(dir_parts),
        "wind_speeds": np.concatenate(spd_parts),
        "rms_in": np.concatenate(rms_in_parts),
        "rms_out": np.concatenate(rms_out_parts),
    }


def sample_points(data: dict, n: int, seed: int) -> dict:
    total = len(data["wind_dirs"])
    if total == 0:
        raise ValueError("有效样本数为 0，无法抽样")

    rng = np.random.default_rng(seed)
    if total <= n:
        print(f"  样本总量 {total} ≤ 目标 {n}，使用全部样本")
        idx = np.arange(total)
    else:
        print(f"  样本总量 {total}，随机抽取 {n} 点（seed={seed}）")
        idx = rng.choice(total, size=n, replace=False)

    return {key: values[idx] for key, values in data.items()}


def load_snapshot(force_refresh: bool) -> dict | None:
    path = Config.SNAPSHOT_PATH
    if force_refresh or not path.exists():
        return None

    payload = np.load(path, allow_pickle=True)
    saved_cfg = json.loads(str(payload["config_json"]))
    if saved_cfg != _snapshot_config():
        print(f"  快照参数不匹配，将重新读取 enriched 数据：{path}")
        return None

    required = ("wind_dirs", "wind_speeds", "rms_in", "rms_out")
    for key in required:
        if key not in payload:
            print(f"  快照缺少字段 {key}，将重新读取 enriched 数据：{path}")
            return None

    print(f"  读取结果快照：{path}")
    print(f"    created_at={payload['created_at']}  n={len(payload['wind_dirs'])}")
    return {key: np.asarray(payload[key], dtype=np.float64) for key in required}


def save_snapshot(data: dict) -> None:
    path = Config.SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config_json=np.asarray(json.dumps(_snapshot_config(), ensure_ascii=False)),
        wind_dirs=np.asarray(data["wind_dirs"], dtype=np.float64),
        wind_speeds=np.asarray(data["wind_speeds"], dtype=np.float64),
        rms_in=np.asarray(data["rms_in"], dtype=np.float64),
        rms_out=np.asarray(data["rms_out"], dtype=np.float64),
    )
    print(f"  写出结果快照：{path}")


def load_or_build_snapshot(force_refresh: bool) -> dict:
    cached = load_snapshot(force_refresh=force_refresh)
    if cached is not None:
        return cached

    print("  未命中快照，开始读取 enriched JSON 并抽样 ...")
    pooled = load_pooled()
    sampled = sample_points(pooled, n=Config.SAMPLE_N, seed=Config.RANDOM_SEED)
    save_snapshot(sampled)
    return sampled


def _axis_limits(rms: np.ndarray, wind: np.ndarray) -> tuple[float, float, float]:
    rms_fin = rms[np.isfinite(rms)]
    wind_fin = wind[np.isfinite(wind)]
    if len(rms_fin) == 0 or len(wind_fin) == 0:
        raise ValueError("无有效有限值用于设置坐标/颜色范围")
    r_max = max(float(np.percentile(rms_fin, Config.RMS_R_PERCENTILE)) * Config.AXIS_PAD, 1e-6)
    c_min = float(np.min(wind_fin))
    c_max = max(float(np.percentile(wind_fin, Config.WIND_C_PERCENTILE)), c_min + 1e-6)
    return r_max, c_min, c_max


def _radial_axis_label_style(theta_deg: float) -> tuple[float, bool]:
    """N=0、顺时针极坐标下，沿径向书写的 rotation，以及是否翻转（由外向内读）。"""
    screen = 90.0 - float(theta_deg)
    screen = (screen + 180.0) % 360.0 - 180.0
    if abs(screen) <= 90.0:
        return screen, False
    return screen + 180.0, True


def _annotate_radial_axis(ax, r_max: float, r_ticks: np.ndarray) -> None:
    """径向轴：上方写加速度 RMS，下方写刻度数字，文字均顺径向排布。"""
    theta0 = Config.RLABEL_DEG
    dtheta = Config.RLABEL_SIDE_OFFSET_DEG
    rot, flipped = _radial_axis_label_style(theta0)

    # 翻转时局部“上/下”与角向偏移对调，保证视觉上始终为上标签、下数字
    if flipped:
        theta_above = theta0 + dtheta
        theta_below = theta0 - dtheta
    else:
        theta_above = theta0 - dtheta
        theta_below = theta0 + dtheta

    ax.set_yticks(r_ticks)
    ax.set_yticklabels([])
    ax.set_rlabel_position(theta0)

    for r in r_ticks:
        ax.text(
            np.deg2rad(theta_below),
            r,
            f"{r:.3f}",
            rotation=rot,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontproperties=ENG_FONT,
            fontsize=FONT_SIZE - 6,
            clip_on=False,
            zorder=6,
        )

    ax.text(
        np.deg2rad(theta_above),
        r_max * 0.72,
        r"加速度 RMS ($m/s^{2}$)",
        rotation=rot,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 4,
        clip_on=False,
        zorder=6,
    )


def make_polar_scatter(
    wind_dirs: np.ndarray,
    rms: np.ndarray,
    wind_speeds: np.ndarray,
    title: str,
) -> plt.Figure:
    finite = np.isfinite(wind_dirs) & np.isfinite(rms) & np.isfinite(wind_speeds)
    wind_dirs = wind_dirs[finite]
    rms = rms[finite]
    wind_speeds = wind_speeds[finite]
    if len(wind_dirs) == 0:
        raise ValueError("无有效散点可绘制")

    r_max, c_min, c_max = _axis_limits(rms, wind_speeds)
    theta = np.deg2rad(wind_dirs)
    norm = plt.Normalize(vmin=c_min, vmax=c_max)

    fig = plt.figure(figsize=(Config.FIG_W, Config.FIG_H))
    # 左侧/下方留白，给径向轴末端标签腾位置
    ax = fig.add_axes([0.06, 0.08, 0.76, 0.80], projection="polar")
    cax = fig.add_axes([0.87, 0.12, 0.025, 0.62])

    sc = ax.scatter(
        theta,
        rms,
        c=wind_speeds,
        cmap=Config.CMAP,
        norm=norm,
        s=Config.SCATTER_SIZE,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(
        ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        fontproperties=ENG_FONT,
        fontsize=FONT_SIZE - 3,
    )
    ax.set_ylim(0, r_max)

    for ang in [Config.AXIS_OF_BRIDGE, Config.AXIS_OF_BRIDGE + 180]:
        ax.plot(
            [np.deg2rad(ang)] * 2,
            [0, r_max * 0.98],
            color="#CC3333",
            lw=1.2,
            ls="--",
            zorder=5,
        )
    ax.annotate(
        "桥轴线",
        xy=(np.deg2rad(Config.AXIS_OF_BRIDGE), r_max * 0.90),
        ha="center",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 5,
        color="#CC3333",
    )

    r_ticks = np.linspace(0, r_max, 5)[1:]
    ax.yaxis.grid(True, alpha=Config.GRID_ALPHA, ls="--")
    ax.xaxis.grid(True, alpha=Config.GRID_ALPHA, ls="--")
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE - 1, pad=16)
    _annotate_radial_axis(ax, r_max, r_ticks)

    cbar = fig.colorbar(sc, cax=cax, orientation="vertical")
    cbar.set_label("平均风速（m/s）", fontproperties=CN_FONT, fontsize=FONT_SIZE - 2)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 4)
    return fig


def _is_web_dashboard_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def push_figures(figures: list[tuple[plt.Figure, str]]) -> None:
    if not _is_web_dashboard_available(Config.WEB_DASHBOARD_PORT):
        print(
            "  未检测到 VibDash 服务，跳过 WebUI 推送；"
            "如需预览请先运行：python -m src.visualize_tools.web_dashboard"
        )
        return

    for slot, (fig, title) in enumerate(figures):
        web_push(
            fig,
            page=Config.WEB_PAGE,
            slot=slot,
            title=title,
            page_cols=2 if slot == 0 else None,
        )
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-14 随机振动风向-RMS-风速 极坐标散点图")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="忽略已有快照，强制从 enriched JSON 重建抽样快照",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("图4-14 随机振动风向–RMS–风速极坐标散点图（抽样 1e5，面内/面外）")
    print("=" * 80)
    print("\n[步骤1] 加载抽样快照（缺失时才读 enriched JSON）...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    print(f"  快照文件：{Config.SNAPSHOT_PATH}")
    if args.refresh_cache:
        print("  模式：--refresh-cache（强制刷新快照）")
    else:
        print("  模式：优先读快照")

    data = load_or_build_snapshot(force_refresh=args.refresh_cache)
    n = len(data["wind_dirs"])
    print(
        f"  绘制样本：n={n}, "
        f"dir={data['wind_dirs'].min():.1f}-{data['wind_dirs'].max():.1f}°, "
        f"wind={data['wind_speeds'].min():.2f}-{data['wind_speeds'].max():.2f} m/s, "
        f"RMS median 面内={float(np.median(data['rms_in'])):.4f}, "
        f"面外={float(np.median(data['rms_out'])):.4f}"
    )

    print("\n[步骤2] 绘制极坐标散点图（面内 / 面外）...")
    fig_in = make_polar_scatter(
        data["wind_dirs"],
        data["rms_in"],
        data["wind_speeds"],
        "随机振动  面内：风向–RMS（色=平均风速）",
    )
    fig_out = make_polar_scatter(
        data["wind_dirs"],
        data["rms_out"],
        data["wind_speeds"],
        "随机振动  面外：风向–RMS（色=平均风速）",
    )
    push_figures([
        (fig_in, "面内：风向–RMS"),
        (fig_out, "面外：风向–RMS"),
    ])
    plt.close(fig_in)
    plt.close(fig_out)
    print("=" * 80)


if __name__ == "__main__":
    main()
