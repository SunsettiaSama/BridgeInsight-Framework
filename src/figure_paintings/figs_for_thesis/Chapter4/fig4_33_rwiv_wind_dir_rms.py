"""图4-33：风雨振 风向–RMS–风速 极坐标散点图。

与图4-24 / 图4-14 散点图样式对齐：角向=风向，径向=RMS，颜色=平均风速。
样本池与 fig4_25 共用（合并副本或仅 DL）；风场按时间戳匹配，RMS 由识别窗计算。
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

from src.data_processer.io_unpacker import UNPACK
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.chapter4_characteristics.feature_analysis._wind import (
    build_wind_lookup,
    compute_wind_stats_for_sensor,
    get_wind_stats_for_sample,
    load_wind_metadata,
)
from src.chapter4_characteristics.settings import load_config
from src.figure_paintings.figs_for_thesis.Chapter4._rwiv_pipeline import (
    RWIV_SAMPLE_COPY_PATH,
    USE_MERGED_DATASET,
    add_dataset_switch_args,
    load_rwiv_samples_for_figures,
    resolve_use_merged,
)
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_25_rwiv_timeseries import (
    Config as SharedConfig,
)
from matplotlib.colors import LinearSegmentedColormap

from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    get_red_color_map,
)
from src.visualize_tools.web_dashboard import push as web_push

FONT_SIZE = SQUARE_FONT_SIZE
# 原红色谱浅端过淡：只取偏深的后半段拉高对比度
_RED_COLORS = list(get_red_color_map(style="discrete").colors)
_CMAP = LinearSegmentedColormap.from_list(
    "red_family_deep",
    _RED_COLORS[3:],  # '#EF8B67' → '#992224'
    N=256,
)


class Config:
    WINDOW_SIZE = SharedConfig.WINDOW_SIZE

    AXIS_OF_BRIDGE = 10.6
    RLABEL_DEG = 245.0
    RLABEL_SIDE_OFFSET_DEG = 7.0
    FIG_W = SQUARE_FIG_SIZE[0]
    FIG_H = SQUARE_FIG_SIZE[1]
    GRID_ALPHA = 0.22
    CMAP = _CMAP

    SCATTER_SIZE = 14
    SCATTER_ALPHA = 0.45
    RMS_R_PERCENTILE = 99.5
    WIND_C_PERCENTILE = 99.5
    AXIS_PAD = 1.08

    SNAPSHOT_DIR = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
    )
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_33_rwiv_wind_dir_rms.npz"
    WEB_PAGE = "fig4_33 风向-RMS散点"
    WEB_DASHBOARD_PORT = 15678


def _snapshot_config(use_merged: bool) -> dict:
    return {
        "figure": "fig4_33_rwiv_wind_dir_rms",
        "version": "polar_scatter_speed_color",
        "use_merged": bool(use_merged),
        "window_size": int(Config.WINDOW_SIZE),
        "dir_correction": "360_minus_mean",
        "sample_copy": str(RWIV_SAMPLE_COPY_PATH) if use_merged else "dl_only",
    }


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def _correct_direction(dirs: list[float]) -> float:
    return float((360.0 - float(np.mean(dirs))) % 360.0)


def _wind_for_timestamp(timestamp: list, wind_stats_by_ts: dict) -> tuple[float, float] | None:
    wl = get_wind_stats_for_sample(list(timestamp), wind_stats_by_ts)
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
    if not d_list or not s_list:
        return None
    return _correct_direction(d_list), float(np.mean(s_list))


def _compute_wind_stats_for_timestamps(timestamps: set[tuple[int, int, int]]) -> dict:
    cfg = load_config()
    wind_path = cfg.get("wind_metadata_path")
    if not wind_path:
        raise ValueError("配置中缺少 wind_metadata_path")
    meta = load_wind_metadata(str(wind_path))
    lookup = build_wind_lookup(meta)

    result: dict = {}
    n = len(timestamps)
    for i, ts in enumerate(sorted(timestamps), start=1):
        stats_list = []
        for rec in lookup.get(ts, []):
            stats = compute_wind_stats_for_sensor(rec)
            if stats is not None:
                stats_list.append(stats)
        result[ts] = stats_list
        if i % 5 == 0 or i == n:
            print(f"  风统计：{i}/{n} 时间戳")
    return result


def compute_scatter_arrays(samples: list[dict]) -> dict:
    need: set[tuple[int, int, int]] = set()
    for sample in samples:
        ts = sample.get("timestamp") or []
        if len(ts) >= 3:
            need.add((int(ts[0]), int(ts[1]), int(ts[2])))
    if not need:
        raise ValueError("风雨振样本缺少可用 timestamp，无法匹配风场")

    print(f"  唯一时间戳：{len(need)}，开始计算风统计量...")
    wind_by_ts = _compute_wind_stats_for_timestamps(need)

    unpacker = UNPACK(init_path=False)
    cache: dict[str, np.ndarray] = {}
    dirs: list[float] = []
    speeds: list[float] = []
    r_in: list[float] = []
    r_out: list[float] = []
    n_miss = 0
    n = len(samples)

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == n:
            print(f"  组装散点：{i + 1}/{n}")

        ts = sample.get("timestamp") or []
        if len(ts) < 3:
            n_miss += 1
            continue
        got = _wind_for_timestamp(ts, wind_by_ts)
        if got is None:
            n_miss += 1
            continue

        in_path = sample["inplane_file_path"]
        out_path = sample["outplane_file_path"]
        win = int(sample["window_idx"])
        if in_path not in cache:
            cache[in_path] = np.asarray(unpacker.VIC_DATA_Unpack(in_path), dtype=np.float64)
        if out_path not in cache:
            cache[out_path] = np.asarray(unpacker.VIC_DATA_Unpack(out_path), dtype=np.float64)

        start = win * Config.WINDOW_SIZE
        end = start + Config.WINDOW_SIZE
        raw_in = cache[in_path]
        raw_out = cache[out_path]
        if end > len(raw_in) or end > len(raw_out):
            raise ValueError(
                f"窗口越界：idx={sample.get('idx')} win={win} "
                f"len_in={len(raw_in)} len_out={len(raw_out)}"
            )

        dirs.append(got[0])
        speeds.append(got[1])
        r_in.append(_rms(raw_in[start:end]))
        r_out.append(_rms(raw_out[start:end]))

    if not dirs:
        raise ValueError("无有效风向-风速-RMS 配对")
    print(f"  有效配对：{len(dirs)}，缺失/无风：{n_miss}")
    return {
        "wind_dirs": np.asarray(dirs, dtype=np.float64),
        "wind_speeds": np.asarray(speeds, dtype=np.float64),
        "rms_in": np.asarray(r_in, dtype=np.float64),
        "rms_out": np.asarray(r_out, dtype=np.float64),
    }


def load_snapshot(use_merged: bool, force_refresh: bool) -> dict | None:
    path = Config.SNAPSHOT_PATH
    if force_refresh or not path.exists():
        return None

    payload = np.load(path, allow_pickle=True)
    saved_cfg = json.loads(str(payload["config_json"]))
    if saved_cfg != _snapshot_config(use_merged):
        print(f"  快照参数不匹配，将重新计算：{path}")
        return None

    required = ("wind_dirs", "wind_speeds", "rms_in", "rms_out")
    for key in required:
        if key not in payload:
            print(f"  快照缺少字段 {key}，将重新计算：{path}")
            return None

    print(f"  读取结果快照：{path}")
    print(f"    created_at={payload['created_at']}  n={len(payload['wind_dirs'])}")
    return {key: np.asarray(payload[key], dtype=np.float64) for key in required}


def save_snapshot(data: dict, use_merged: bool) -> None:
    path = Config.SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config_json=np.asarray(json.dumps(_snapshot_config(use_merged), ensure_ascii=False)),
        wind_dirs=np.asarray(data["wind_dirs"], dtype=np.float64),
        wind_speeds=np.asarray(data["wind_speeds"], dtype=np.float64),
        rms_in=np.asarray(data["rms_in"], dtype=np.float64),
        rms_out=np.asarray(data["rms_out"], dtype=np.float64),
    )
    print(f"  写出结果快照：{path}")


def load_or_compute(
    use_merged: bool,
    force_refresh: bool,
    refresh_sample_copy: bool,
) -> dict:
    cached = load_snapshot(use_merged=use_merged, force_refresh=force_refresh)
    if cached is not None:
        return cached

    print("  未命中快照，加载风雨振样本并组装风向-RMS-风速 ...")
    if use_merged:
        print(f"  副本路径：{RWIV_SAMPLE_COPY_PATH}")
    samples = load_rwiv_samples_for_figures(
        use_merged=use_merged,
        force_refresh=refresh_sample_copy,
    )
    print(f"  配对样本：{len(samples)}")
    data = compute_scatter_arrays(samples)
    save_snapshot(data, use_merged=use_merged)
    return data


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
    screen = 90.0 - float(theta_deg)
    screen = (screen + 180.0) % 360.0 - 180.0
    if abs(screen) <= 90.0:
        return screen, False
    return screen + 180.0, True


def _annotate_radial_axis(ax, r_max: float, r_ticks: np.ndarray) -> None:
    theta0 = Config.RLABEL_DEG
    dtheta = Config.RLABEL_SIDE_OFFSET_DEG
    rot, flipped = _radial_axis_label_style(theta0)

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
    parser = argparse.ArgumentParser(description="图4-33 风雨振风向-RMS-风速 极坐标散点图")
    add_dataset_switch_args(parser)
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="强制重算并覆盖快照",
    )
    args = parser.parse_args()
    use_merged = resolve_use_merged(args.use_merged)

    print("=" * 80)
    print("图4-33 风雨振风向–RMS–风速极坐标散点图")
    print(f"  默认开关 USE_MERGED_DATASET={USE_MERGED_DATASET}  → 本次 use_merged={use_merged}")
    print("=" * 80)

    print("\n[步骤1] 加载风向-RMS-风速 ...")
    data = load_or_compute(
        use_merged=use_merged,
        force_refresh=args.refresh_snapshot,
        refresh_sample_copy=args.refresh_sample_copy,
    )
    n = len(data["wind_dirs"])
    print(
        f"[OK] 绘制样本：n={n}, "
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
        "风雨振  面内：风向–RMS（色=平均风速）",
    )
    fig_out = make_polar_scatter(
        data["wind_dirs"],
        data["rms_out"],
        data["wind_speeds"],
        "风雨振  面外：风向–RMS（色=平均风速）",
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
