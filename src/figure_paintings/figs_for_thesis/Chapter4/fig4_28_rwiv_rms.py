"""图4-28：风雨振 RMS 分布（直方图 + 面内–面外散点）。

样式对齐 fig4_10 / fig4_18。样本池与 fig4_25 共用（合并副本或仅 DL）。
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
from src.figure_paintings.figs_for_thesis.config import (
    ANNOTATION_COLOR,
    CN_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    WINDOW_SIZE = SharedConfig.WINDOW_SIZE

    N_BINS = 80
    RMS_X_PERCENTILE = 99.0
    # 与 fig4_18 一致：随机振动 RMS 直方图右边界（目视，m/s²）
    NORMAL_TAIL_RMS = 0.16
    SCATTER_AXIS_PERCENTILE = 99.5
    SCATTER_AXIS_PAD = 1.08
    SCATTER_MAX_POINTS = 120_000
    SCATTER_SEED = 42

    FIG_SIZE = SQUARE_FIG_SIZE
    LABEL_FONT_SIZE = SQUARE_FONT_SIZE
    TICK_FONT_SIZE = SQUARE_FONT_SIZE - 4
    LEGEND_FONT_SIZE = SQUARE_FONT_SIZE - 4

    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    SCATTER_COLOR = VIV_INPLANE_COLOR
    BAR_ALPHA = 0.72
    SCATTER_SIZE = 8
    SCATTER_ALPHA = 0.45

    TAIL_LINE_COLOR = ANNOTATION_COLOR
    TAIL_LINEWIDTH = 1.8
    TAIL_LINESTYLE = "-."

    SNAPSHOT_DIR = project_root / "results" / "chapter4_characteristics" / "figure_snapshots"
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_28_rwiv_rms.npz"
    WEB_PAGE = "fig4_28 风雨振 RMS"
    WEB_DASHBOARD_PORT = 15678


def _snapshot_config(use_merged: bool) -> dict:
    return {
        "figure": "fig4_28_rwiv_rms",
        "use_merged": bool(use_merged),
        "window_size": int(Config.WINDOW_SIZE),
        "sample_copy": str(RWIV_SAMPLE_COPY_PATH) if use_merged else "dl_only",
    }


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def compute_rms_from_samples(samples: list[dict]) -> dict:
    unpacker = UNPACK(init_path=False)
    cache: dict[str, np.ndarray] = {}
    inplane_rms: list[float] = []
    outplane_rms: list[float] = []

    n = len(samples)
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == n:
            print(f"  计算 RMS：{i + 1}/{n}")

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

        inplane_rms.append(_rms(raw_in[start:end]))
        outplane_rms.append(_rms(raw_out[start:end]))

    return {
        "inplane_rms": np.asarray(inplane_rms, dtype=np.float64),
        "outplane_rms": np.asarray(outplane_rms, dtype=np.float64),
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

    required = ("inplane_rms", "outplane_rms")
    for key in required:
        if key not in payload:
            print(f"  快照缺少字段 {key}，将重新计算：{path}")
            return None

    print(f"  读取结果快照：{path}")
    print(f"    created_at={payload['created_at']}  n={len(payload['inplane_rms'])}")
    return {key: np.asarray(payload[key], dtype=np.float64) for key in required}


def save_snapshot(data: dict, use_merged: bool) -> None:
    path = Config.SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config_json=np.asarray(json.dumps(_snapshot_config(use_merged), ensure_ascii=False)),
        inplane_rms=np.asarray(data["inplane_rms"], dtype=np.float64),
        outplane_rms=np.asarray(data["outplane_rms"], dtype=np.float64),
    )
    print(f"  写出结果快照：{path}")


def load_rms_data(use_merged: bool, force_refresh: bool, refresh_sample_copy: bool) -> dict:
    cached = load_snapshot(use_merged=use_merged, force_refresh=force_refresh)
    if cached is not None:
        return cached

    print("  未命中快照，加载风雨振样本并计算 RMS ...")
    if use_merged:
        print(f"  副本路径：{RWIV_SAMPLE_COPY_PATH}")
    samples = load_rwiv_samples_for_figures(
        use_merged=use_merged,
        force_refresh=refresh_sample_copy,
    )
    print(f"  配对样本：{len(samples)}")
    data = compute_rms_from_samples(samples)
    save_snapshot(data, use_merged=use_merged)
    return data


def _mark_normal_tail_hist(ax, x_tail: float) -> None:
    ax.axvline(
        x_tail,
        color=Config.TAIL_LINE_COLOR,
        linewidth=Config.TAIL_LINEWIDTH,
        linestyle=Config.TAIL_LINESTYLE,
        zorder=4,
        label="随机振动尾部",
    )


def _mark_normal_tail_scatter(ax, x_tail: float) -> None:
    ax.axvline(
        x_tail,
        color=Config.TAIL_LINE_COLOR,
        linewidth=Config.TAIL_LINEWIDTH,
        linestyle=Config.TAIL_LINESTYLE,
        zorder=4,
        label="随机振动尾部",
    )
    ax.axhline(
        x_tail,
        color=Config.TAIL_LINE_COLOR,
        linewidth=Config.TAIL_LINEWIDTH,
        linestyle=Config.TAIL_LINESTYLE,
        zorder=4,
    )


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax) -> None:
    leg = ax.legend(fontsize=Config.LEGEND_FONT_SIZE, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)


def _is_web_dashboard_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _calc_rms_histogram(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    in_rms = data["inplane_rms"]
    out_rms = data["outplane_rms"]
    combined = np.concatenate([in_rms, out_rms])
    x_max = float(np.percentile(combined, Config.RMS_X_PERCENTILE))
    x_max = max(x_max, 1e-6)

    bins = np.linspace(0, x_max, Config.N_BINS + 1)
    width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    counts_in, _ = np.histogram(in_rms[in_rms <= x_max], bins=bins)
    counts_out, _ = np.histogram(out_rms[out_rms <= x_max], bins=bins)
    return centers, counts_in, counts_out, width, x_max


def plot_rms_histogram(data: dict, normal_tail: float) -> plt.Figure:
    centers, counts_in, counts_out, width, x_max = _calc_rms_histogram(data)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    half = width * 0.46
    ax.bar(
        centers - half / 2,
        counts_in,
        width=half,
        color=Config.INPLANE_COLOR,
        alpha=Config.BAR_ALPHA,
        label="面内",
    )
    ax.bar(
        centers + half / 2,
        counts_out,
        width=half,
        color=Config.OUTPLANE_COLOR,
        alpha=Config.BAR_ALPHA,
        label="面外",
    )
    if 0 < normal_tail < x_max:
        _mark_normal_tail_hist(ax, normal_tail)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r"RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel("样本数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    _add_legend(ax)
    _apply_grid(ax)
    fig.tight_layout()
    return fig


def plot_rms_scatter(data: dict, normal_tail: float) -> plt.Figure:
    in_rms = data["inplane_rms"]
    out_rms = data["outplane_rms"]
    x_plot = out_rms
    y_plot = in_rms
    if len(x_plot) > Config.SCATTER_MAX_POINTS:
        rng = np.random.default_rng(Config.SCATTER_SEED)
        idx = rng.choice(len(x_plot), size=Config.SCATTER_MAX_POINTS, replace=False)
        x_plot = x_plot[idx]
        y_plot = y_plot[idx]

    combined_full = np.concatenate([out_rms, in_rms])
    xy_max = float(np.percentile(combined_full, Config.SCATTER_AXIS_PERCENTILE))
    xy_max = max(xy_max * Config.SCATTER_AXIS_PAD, 1e-6)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.scatter(
        x_plot,
        y_plot,
        s=Config.SCATTER_SIZE,
        color=Config.SCATTER_COLOR,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
    )
    if 0 < normal_tail < xy_max:
        _mark_normal_tail_scatter(ax, normal_tail)
        _add_legend(ax)
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)
    ax.set_xlabel(r"面外 RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel(r"面内 RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    _apply_grid(ax)
    fig.tight_layout()
    return fig


def push_figures(figures: list[tuple[plt.Figure, str]]) -> None:
    if not _is_web_dashboard_available(Config.WEB_DASHBOARD_PORT):
        print(
            "  未检测到 VibDash 服务，跳过 WebUI 推送；"
            "如需预览请先运行：python -m src.visualize_tools.web_dashboard"
        )
        return

    page = Config.WEB_PAGE
    for slot, (fig, title) in enumerate(figures):
        web_push(fig, page=page, slot=slot, title=title, page_cols=2 if slot == 0 else None)
    print(f"[OK] 已推送到 WebUI：{page}")


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-28 风雨振 RMS 分布")
    add_dataset_switch_args(parser)
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="强制重算 RMS 并覆盖快照",
    )
    args = parser.parse_args()
    use_merged = resolve_use_merged(args.use_merged)

    print("=" * 80)
    print("图4-28 风雨振 RMS 分布")
    print(f"  默认开关 USE_MERGED_DATASET={USE_MERGED_DATASET}  → 本次 use_merged={use_merged}")
    print("=" * 80)

    print("\n[步骤1] 加载风雨振 RMS ...")
    data = load_rms_data(
        use_merged=use_merged,
        force_refresh=args.refresh_snapshot,
        refresh_sample_copy=args.refresh_sample_copy,
    )
    print(f"[OK] 有效配对样本：{len(data['inplane_rms'])}")
    print(f"  面内 RMS median={float(np.median(data['inplane_rms'])):.4f}")
    print(f"  面外 RMS median={float(np.median(data['outplane_rms'])):.4f}")
    print(
        f"  面内 p99={float(np.percentile(data['inplane_rms'], 99)):.4f}  "
        f"面外 p99={float(np.percentile(data['outplane_rms'], 99)):.4f}"
    )

    normal_tail = Config.NORMAL_TAIL_RMS
    print(f"\n[步骤2] 随机振动尾部标记（与 fig4_18 一致）：{normal_tail:.2f} m/s^2")

    print("\n[步骤3] 绘制图像...")
    fig_hist = plot_rms_histogram(data, normal_tail)
    fig_scatter = plot_rms_scatter(data, normal_tail)
    print("[OK] 已生成 2 张独立图像")

    push_figures([
        (fig_hist, "风雨振 RMS 直方图"),
        (fig_scatter, "面内-面外 RMS 散点图"),
    ])
    plt.close(fig_hist)
    plt.close(fig_scatter)


if __name__ == "__main__":
    main()
