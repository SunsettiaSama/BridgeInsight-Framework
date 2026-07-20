from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    VIV_CLASS_ID = 1
    FEATURE_BATCH_SIZE = 512
    FREQ_THRESHOLD = 7.0

    FIG_SIZE = SQUARE_FIG_SIZE
    LABEL_FONT_SIZE = SQUARE_FONT_SIZE
    TICK_FONT_SIZE = SQUARE_FONT_SIZE - 4
    LEGEND_FONT_SIZE = SQUARE_FONT_SIZE - 4

    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    BAR_ALPHA = 0.78
    BAR_EDGE_COLOR = "white"
    BAR_EDGE_WIDTH = 0.4

    WIND_BIN_WIDTH = 0.5
    WIND_X_PERCENTILE = 99.5
    AXIS_PAD = 1.05

    ENRICHED_STATS_DIR = get_enriched_class_dir(VIV_CLASS_ID)
    SENSOR_GROUPS = filter_sensor_groups(data_config.SENSOR_GROUPS_WIND)
    WEB_DASHBOARD_PORT = 15678
    WEB_PAGE = "fig4_23 VIV风速分布"
    SNAPSHOT_DIR = project_root / "results" / "chapter4_characteristics" / "figure_snapshots"
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_23_viv_wind_speed_hist.npz"


def _snapshot_config() -> dict:
    return {
        "figure": "fig4_23_viv_wind_speed_hist",
        "version": "plane_split_low_high_freq",
        "class_id": Config.VIV_CLASS_ID,
        "freq_threshold": Config.FREQ_THRESHOLD,
        "sensor_groups": Config.SENSOR_GROUPS,
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
        "data_source": data_config.DATA_SOURCE,
        "wind_bin_width": Config.WIND_BIN_WIDTH,
    }


def _dominant_frequency(freqs, powers) -> float | None:
    if not freqs or not powers:
        return None
    dom_idx = int(np.argmax(powers))
    if dom_idx >= len(freqs):
        return None
    return float(freqs[dom_idx])


def _sample_mean_wind(sample: dict) -> float | None:
    wind_list = sample.get("wind_stats") or []
    if not wind_list:
        return None
    speeds = [
        float(w["mean_wind_speed"])
        for w in wind_list
        if w.get("mean_wind_speed") is not None and np.isfinite(float(w["mean_wind_speed"]))
    ]
    if not speeds:
        return None
    return float(np.mean(speeds))


def _append_plane_wind(
    buckets: dict[str, list[float]],
    key: str,
    wind: float,
    freqs,
    powers,
) -> None:
    dom = _dominant_frequency(freqs, powers)
    if dom is None:
        return
    if dom < Config.FREQ_THRESHOLD:
        buckets[f"low_{key}"].append(wind)
    else:
        buckets[f"high_{key}"].append(wind)


def load_wind_by_plane_and_band(json_file: Path) -> dict[str, np.ndarray]:
    if not json_file.exists():
        raise FileNotFoundError(f"数据文件不存在：{json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    buckets: dict[str, list[float]] = {
        "low_in": [],
        "low_out": [],
        "high_in": [],
        "high_out": [],
    }

    for sample in data["samples"]:
        wind = _sample_mean_wind(sample)
        if wind is None:
            continue
        psd_in = sample.get("psd_inplane") or {}
        psd_out = sample.get("psd_outplane") or {}
        _append_plane_wind(buckets, "in", wind, psd_in.get("frequencies"), psd_in.get("powers"))
        _append_plane_wind(buckets, "out", wind, psd_out.get("frequencies"), psd_out.get("powers"))

    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def load_pooled_data() -> dict[str, np.ndarray]:
    ensure_enriched_for_figures(class_id=Config.VIV_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)

    parts: dict[str, list[np.ndarray]] = {
        "low_in": [],
        "low_out": [],
        "high_in": [],
        "high_out": [],
    }

    for location_name, filename in Config.SENSOR_GROUPS.items():
        json_file = Config.ENRICHED_STATS_DIR / filename
        print(f"  加载：{location_name} - {filename}")
        data = load_wind_by_plane_and_band(json_file)
        n_total = sum(len(v) for v in data.values())
        if n_total == 0:
            print(f"  跳过（无有效样本）：{location_name}")
            continue
        print(
            f"    low 面内={len(data['low_in'])} 面外={len(data['low_out'])} | "
            f"high 面内={len(data['high_in'])} 面外={len(data['high_out'])}"
        )
        for key, arr in data.items():
            if len(arr) > 0:
                parts[key].append(arr)

    if not any(parts.values()):
        raise ValueError("无可绘制的涡激共振风速样本")

    return {
        key: np.concatenate(arrs) if arrs else np.empty(0, dtype=np.float64)
        for key, arrs in parts.items()
    }


def load_snapshot(force_refresh: bool) -> dict[str, np.ndarray] | None:
    path = Config.SNAPSHOT_PATH
    if force_refresh or not path.exists():
        return None

    payload = np.load(path, allow_pickle=True)
    saved_cfg = json.loads(str(payload["config_json"]))
    if saved_cfg != _snapshot_config():
        print(f"  快照参数不匹配，将重新读取 enriched 数据：{path}")
        return None

    data = {key: np.asarray(payload[key], dtype=np.float64) for key in ("low_in", "low_out", "high_in", "high_out")}
    print(f"  读取结果快照：{path}")
    print(f"    created_at={payload['created_at']}")
    return data


def save_snapshot(data: dict[str, np.ndarray]) -> None:
    path = Config.SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config_json=np.asarray(json.dumps(_snapshot_config(), ensure_ascii=False)),
        low_in=np.asarray(data["low_in"], dtype=np.float64),
        low_out=np.asarray(data["low_out"], dtype=np.float64),
        high_in=np.asarray(data["high_in"], dtype=np.float64),
        high_out=np.asarray(data["high_out"], dtype=np.float64),
    )
    print(f"  写出结果快照：{path}")


def load_or_build_snapshot(force_refresh: bool) -> dict[str, np.ndarray]:
    cached = load_snapshot(force_refresh=force_refresh)
    if cached is not None:
        return cached

    print("  未命中快照，开始读取 enriched JSON ...")
    data = load_pooled_data()
    save_snapshot(data)
    return data


def _is_web_dashboard_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _shared_bins(wind_in: np.ndarray, wind_out: np.ndarray) -> tuple[np.ndarray, float]:
    combined = np.concatenate([wind_in, wind_out])
    combined = combined[np.isfinite(combined)]
    if len(combined) == 0:
        raise ValueError("无有效有限风速值")
    x_max = max(
        float(np.percentile(combined, Config.WIND_X_PERCENTILE)) * Config.AXIS_PAD,
        Config.WIND_BIN_WIDTH,
    )
    bins = np.arange(0.0, x_max + Config.WIND_BIN_WIDTH, Config.WIND_BIN_WIDTH)
    return bins, x_max


def plot_band_histogram(
    wind_in: np.ndarray,
    wind_out: np.ndarray,
    title: str,
) -> plt.Figure:
    wind_in = wind_in[np.isfinite(wind_in)]
    wind_out = wind_out[np.isfinite(wind_out)]
    if len(wind_in) == 0 and len(wind_out) == 0:
        raise ValueError(f"{title} 无有效风速样本")

    bins, x_max = _shared_bins(wind_in, wind_out)
    centers = (bins[:-1] + bins[1:]) / 2.0
    width = Config.WIND_BIN_WIDTH * 0.42
    counts_in, _ = np.histogram(wind_in, bins=bins)
    counts_out, _ = np.histogram(wind_out, bins=bins)
    y_max = max(float(np.max(counts_in)), float(np.max(counts_out)), 1.0) * 1.08

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.bar(
        centers - width / 2,
        counts_in,
        width=width,
        color=Config.INPLANE_COLOR,
        alpha=Config.BAR_ALPHA,
        edgecolor=Config.BAR_EDGE_COLOR,
        linewidth=Config.BAR_EDGE_WIDTH,
        label=f"面内（n={len(wind_in):,}）",
        zorder=3,
    )
    ax.bar(
        centers + width / 2,
        counts_out,
        width=width,
        color=Config.OUTPLANE_COLOR,
        alpha=Config.BAR_ALPHA,
        edgecolor=Config.BAR_EDGE_COLOR,
        linewidth=Config.BAR_EDGE_WIDTH,
        label=f"面外（n={len(wind_out):,}）",
        zorder=3,
    )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE, pad=14)
    ax.set_xlabel("平均风速（m/s）", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel("VIV 样本数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{int(v):,}"))
    ax.grid(True, axis="y", color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)

    leg = ax.legend(fontsize=Config.LEGEND_FONT_SIZE, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)

    fig.tight_layout()
    return fig


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
    parser = argparse.ArgumentParser(description="图4-23 涡激共振风速分布（面内外×高低频，正方形）")
    parser.add_argument("--refresh-cache", action="store_true", help="忽略已有快照，强制从 enriched JSON 重建")
    args = parser.parse_args()

    thr = Config.FREQ_THRESHOLD
    title_low = f"低频主导（<{thr:g} Hz）VIV 平均风速分布"
    title_high = f"高频主导（≥{thr:g} Hz）VIV 平均风速分布"

    print("=" * 80)
    print(f"图4-23 涡激共振风速分布直方图（面内外；低频<{thr}Hz / 高频≥{thr}Hz）")
    print("=" * 80)
    print(f"\n[步骤1] 加载涡激共振风速样本...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    print(f"  快照文件：{Config.SNAPSHOT_PATH}")
    if args.refresh_cache:
        print("  模式：--refresh-cache（强制刷新快照）")
    else:
        print("  模式：优先读快照，缺失才读取 enriched JSON")

    data = load_or_build_snapshot(force_refresh=args.refresh_cache)
    print(
        f"  低频：面内={len(data['low_in']):,} 面外={len(data['low_out']):,} | "
        f"高频：面内={len(data['high_in']):,} 面外={len(data['high_out']):,}"
    )

    print("\n[步骤2] 绘制图像...")
    fig_low = plot_band_histogram(data["low_in"], data["low_out"], title=title_low)
    fig_high = plot_band_histogram(data["high_in"], data["high_out"], title=title_high)
    push_figures([
        (fig_low, title_low),
        (fig_high, title_high),
    ])
    plt.close(fig_low)
    plt.close(fig_high)


if __name__ == "__main__":
    main()
