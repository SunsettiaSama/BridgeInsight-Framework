from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.config import (
    ANNOTATION_COLOR,
    CN_FONT,
    ENG_FONT,
    FONT_SIZE,
    LABEL_FONT_SIZE,
    REC_FIG_SIZE,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import DEFAULT_PORT as DASHBOARD_DEFAULT_PORT
from src.visualize_tools.web_dashboard import push as web_push

CACHE_PATH = PROJECT_ROOT / "results" / "full_vib_metadata" / "staycable_vib202409_index_cache.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figure_paintings" / "figs_for_thesis" / "Chapter3"
WEBUI_OUTPUT_DIR = PROJECT_ROOT / "results" / "webui" / "figs_for_thesis" / "Chapter3"
RMS_CACHE_PATH = OUTPUT_DIR / "fig3_x_inout_rms_scatter_training_cables_points.json"

REFERENCE_CABLES = (
    ("ST-VIC-C18-101-01", "ST-VIC-C18-101-02", "C18-101"),
    ("ST-VIC-C18-102-01", "ST-VIC-C18-102-02", "C18-102"),
    ("ST-VIC-C34-101-01", "ST-VIC-C34-101-02", "C34-101"),
    ("ST-VIC-C34-102-01", "ST-VIC-C34-102-02", "C34-102"),
)
TARGET_CABLES = (
    ("ST-VIC-C34-201-01", "ST-VIC-C34-201-02", "C34-201"),
    ("ST-VIC-C34-202-01", "ST-VIC-C34-202-02", "C34-202"),
    ("ST-VIC-C34-301-01", "ST-VIC-C34-301-02", "C34-301"),
    ("ST-VIC-C34-302-01", "ST-VIC-C34-302-02", "C34-302"),
)
ALL_CABLES = REFERENCE_CABLES + TARGET_CABLES

WINDOW_SIZE = 3000
FS = 50.0
SCATTER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("C18-101", "C18-102", "C34-101", "C34-102"),
    ("C34-201", "C34-202", "C34-301", "C34-302"),
)
FIGURE_STEMS = (
    "fig3_x_inout_rms_scatter_training_group1",
    "fig3_x_inout_rms_scatter_training_group2",
)
WEBUI_PAGE = "fig3_x 训练集面内外 RMS 散点"
LEGEND_FONT_SIZE = FONT_SIZE - 6
TICK_FONT_SIZE = FONT_SIZE - 6
MARKER_SIZE = 12
ALPHA = 0.34
THESIS_DPI = 300
THESIS_COLORS = list(get_full_color_map("discrete").colors)
THESIS_COLOR_BY_CABLE = {
    "C18-101": THESIS_COLORS[0],
    "C18-102": THESIS_COLORS[9],
    "C34-101": THESIS_COLORS[1],
    "C34-102": THESIS_COLORS[11],
    "C34-201": THESIS_COLORS[2],
    "C34-202": THESIS_COLORS[8],
    "C34-301": THESIS_COLORS[7],
    "C34-302": THESIS_COLORS[10],
}


@dataclass(frozen=True)
class CableStyle:
    cable_id: str
    color: str
    group: str
    legend_label: str
    marker_size: int
    alpha: float


CABLE_STYLES: dict[str, CableStyle] = {
    "C18-101": CableStyle(
        cable_id="C18-101",
        color=THESIS_COLOR_BY_CABLE["C18-101"],
        group="cable",
        legend_label="C18-101",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
    "C18-102": CableStyle(
        cable_id="C18-102",
        color=THESIS_COLOR_BY_CABLE["C18-102"],
        group="cable",
        legend_label="C18-102",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
    "C34-101": CableStyle(
        cable_id="C34-101",
        color=THESIS_COLOR_BY_CABLE["C34-101"],
        group="cable",
        legend_label="C34-101",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
    "C34-102": CableStyle(
        cable_id="C34-102",
        color=THESIS_COLOR_BY_CABLE["C34-102"],
        group="cable",
        legend_label="C34-102",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
    "C34-201": CableStyle(
        cable_id="C34-201",
        color=THESIS_COLOR_BY_CABLE["C34-201"],
        group="cable",
        legend_label="C34-201",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
    "C34-202": CableStyle(
        cable_id="C34-202",
        color=THESIS_COLOR_BY_CABLE["C34-202"],
        group="cable",
        legend_label="C34-202",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
    "C34-301": CableStyle(
        cable_id="C34-301",
        color=THESIS_COLOR_BY_CABLE["C34-301"],
        group="cable",
        legend_label="C34-301",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
    "C34-302": CableStyle(
        cable_id="C34-302",
        color=THESIS_COLOR_BY_CABLE["C34-302"],
        group="cable",
        legend_label="C34-302",
        marker_size=MARKER_SIZE,
        alpha=ALPHA,
    ),
}


def load_cache_samples(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        raise FileNotFoundError(f"索引缓存不存在：{cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{cache_path} 缺少 samples 列表")
    return samples


def filter_samples(samples: list[dict]) -> dict[str, list[tuple[str, str, int]]]:
    pair_lookup = {(in_id, out_id): label for in_id, out_id, label in ALL_CABLES}
    grouped: dict[str, list[tuple[str, str, int]]] = {label: [] for _, _, label in ALL_CABLES}
    for sample in samples:
        cable_pair = tuple(sample.get("cable_pair") or ())
        if len(cable_pair) != 2:
            continue
        label = pair_lookup.get((str(cable_pair[0]), str(cable_pair[1])))
        if label is None:
            continue
        in_fp = str(sample.get("inplane_file_path") or "")
        out_fp = str(sample.get("outplane_file_path") or "")
        if not in_fp or not out_fp:
            continue
        grouped[label].append((in_fp, out_fp, int(sample.get("window_idx", 0))))
    return grouped


def group_by_file_pair(rows: list[tuple[str, str, int]]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for in_fp, out_fp, window_idx in rows:
        grouped[(in_fp, out_fp)].append(int(window_idx))
    for key in grouped:
        grouped[key] = sorted(set(grouped[key]))
    return grouped


def window_rms(signal: np.ndarray) -> float:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def extract_window(raw: np.ndarray, window_idx: int, window_size: int) -> np.ndarray | None:
    start = int(window_idx) * int(window_size)
    end = start + int(window_size)
    if end > int(raw.size):
        return None
    return raw[start:end]


def process_file_pair(task: tuple[str, str, list[int], int]) -> list[tuple[float, float]]:
    in_fp, out_fp, window_indices, window_size = task
    unpacker = UNPACK(init_path=False)
    in_raw = np.asarray(unpacker.VIC_DATA_Unpack(in_fp), dtype=np.float64).reshape(-1)
    out_raw = np.asarray(unpacker.VIC_DATA_Unpack(out_fp), dtype=np.float64).reshape(-1)
    points: list[tuple[float, float]] = []
    for window_idx in window_indices:
        in_window = extract_window(in_raw, window_idx, window_size)
        out_window = extract_window(out_raw, window_idx, window_size)
        if in_window is None or out_window is None:
            continue
        in_rms = window_rms(in_window)
        out_rms = window_rms(out_window)
        if np.isfinite(in_rms) and np.isfinite(out_rms) and in_rms > 0 and out_rms > 0:
            points.append((in_rms, out_rms))
    return points


def compute_rms_points(
    grouped_samples: dict[str, list[tuple[str, str, int]]],
    window_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    tasks: list[tuple[str, str, list[int], int, str]] = []
    for label, rows in grouped_samples.items():
        for (in_fp, out_fp), window_indices in group_by_file_pair(rows).items():
            tasks.append((in_fp, out_fp, window_indices, int(window_size), label))

    per_cable: dict[str, list[tuple[float, float]]] = {label: [] for _, _, label in ALL_CABLES}
    if not tasks:
        raise ValueError("目标拉索在 202409 索引缓存中没有可用样本")

    worker_count = max(1, int(workers))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(process_file_pair, (in_fp, out_fp, window_indices, window_size)): label
            for in_fp, out_fp, window_indices, window_size, label in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            per_cable[label].extend(future.result())

    result: dict[str, np.ndarray] = {}
    for label in per_cable:
        arr = np.asarray(per_cable[label], dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"{label} 未计算出任何 RMS 点")
        result[label] = arr
    return result


def save_rms_cache(path: Path, points_by_cable: dict[str, np.ndarray], window_size: int) -> None:
    payload = {
        "window_size": int(window_size),
        "cables": {label: arr.tolist() for label, arr in points_by_cable.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_rms_cache(path: Path, window_size: int) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if int(payload.get("window_size", 0)) != int(window_size):
        return None
    cables = payload.get("cables")
    if not isinstance(cables, dict):
        return None
    required = {label for _, _, label in ALL_CABLES}
    if set(cables.keys()) != required:
        return None
    return {label: np.asarray(rows, dtype=np.float64) for label, rows in cables.items()}


def log_limits(points_by_cable: dict[str, np.ndarray]) -> tuple[float, float]:
    positive = []
    for arr in points_by_cable.values():
        mask = (arr[:, 0] > 0) & (arr[:, 1] > 0)
        positive.append(arr[mask])
    merged = np.concatenate(positive, axis=0)
    min_val = float(min(np.min(merged[:, 0]), np.min(merged[:, 1])))
    max_val = float(max(np.max(merged[:, 0]), np.max(merged[:, 1])))
    lo = max(min_val * 0.8, 1e-4)
    hi = max_val * 1.15
    return lo, hi


def setup_thesis_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.6, color=ANNOTATION_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE, direction="out", length=4, width=0.8)


def build_legend_handles(cable_labels: tuple[str, ...]) -> list[Line2D]:
    handles: list[Line2D] = []
    for label in cable_labels:
        style = CABLE_STYLES[label]
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=style.color,
                markeredgecolor=style.color,
                markeredgewidth=0.8,
                markersize=style.marker_size * 0.55,
                label=style.legend_label,
            )
        )
    return handles


def plot_scatter(
    points_by_cable: dict[str, np.ndarray],
    cable_labels: tuple[str, ...],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)
    subset = {label: points_by_cable[label] for label in cable_labels}
    lo, hi = log_limits(subset)

    for label in cable_labels:
        style = CABLE_STYLES[label]
        arr = points_by_cable[label]
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            s=style.marker_size,
            alpha=style.alpha,
            c=style.color,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("面内 RMS / m·s$^{-2}$", fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE, labelpad=8)
    ax.set_ylabel("面外 RMS / m·s$^{-2}$", fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE, labelpad=8)
    setup_thesis_axis(ax)

    legend = ax.legend(
        handles=build_legend_handles(cable_labels),
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=ANNOTATION_COLOR,
        fontsize=LEGEND_FONT_SIZE,
        prop=CN_FONT,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.6,
        borderpad=0.6,
        labelspacing=0.45,
    )
    legend.get_frame().set_linewidth(0.6)

    note = (
        f"2024年9月全量窗口；窗口长度 {WINDOW_SIZE / FS:.0f} s（{WINDOW_SIZE} 点 @ {FS:.0f} Hz）"
    )
    ax.text(
        0.99,
        0.02,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 10,
        color=ANNOTATION_COLOR,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)
    return fig


def build_group_summary(
    points_by_cable: dict[str, np.ndarray],
    cable_labels: tuple[str, ...],
    group_index: int,
    figure_stem: str,
) -> dict:
    summary = {
        "group_index": group_index,
        "figure_stem": figure_stem,
        "cables_in_group": list(cable_labels),
        "window_size": WINDOW_SIZE,
        "cache_path": str(CACHE_PATH),
        "rms_cache_path": str(RMS_CACHE_PATH),
        "cables": {},
    }
    for label in cable_labels:
        arr = points_by_cable[label]
        x = arr[:, 0]
        y = arr[:, 1]
        ratio = y / np.maximum(x, 1e-12)
        style = CABLE_STYLES[label]
        summary["cables"][label] = {
            "color": style.color,
            "legend_label": style.legend_label,
            "count": int(len(arr)),
            "in_rms_median": float(np.median(x)),
            "out_rms_median": float(np.median(y)),
            "ratio_median": float(np.median(ratio)),
            "in_rms_p95": float(np.percentile(x, 95)),
            "out_rms_p95": float(np.percentile(y, 95)),
        }
    return summary


def save_thesis_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=THESIS_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制论文版训练集拉索面内外 RMS 散点图（按训练集索分两组）"
    )
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--rms-cache", type=Path, default=RMS_CACHE_PATH)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--web-port", type=int, default=DASHBOARD_DEFAULT_PORT)
    args = parser.parse_args()

    points_by_cable = None
    if not args.recompute:
        points_by_cable = load_rms_cache(args.rms_cache, int(args.window_size))

    if points_by_cable is None:
        samples = load_cache_samples(args.cache_path)
        grouped = filter_samples(samples)
        counts = {label: len(rows) for label, rows in grouped.items()}
        print("window counts:", counts)
        points_by_cable = compute_rms_points(grouped, int(args.window_size), int(args.workers))
        save_rms_cache(args.rms_cache, points_by_cable, int(args.window_size))
        print(f"saved rms cache: {args.rms_cache}")

    WEBUI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    group_summaries: list[dict] = []

    for group_idx, (cable_labels, stem) in enumerate(
        zip(SCATTER_GROUPS, FIGURE_STEMS),
        start=1,
    ):
        figure = plot_scatter(points_by_cable, cable_labels)
        png_path, pdf_path = save_thesis_figure(figure, args.output_dir, stem)
        webui_png = WEBUI_OUTPUT_DIR / f"{stem}.png"
        figure.savefig(webui_png, dpi=THESIS_DPI, bbox_inches="tight", facecolor="white")

        summary = build_group_summary(points_by_cable, cable_labels, group_idx, stem)
        group_summaries.append(summary)
        summary_path = args.output_dir / f"{stem}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"group {group_idx}: {png_path}")
        print(f"group {group_idx}: {pdf_path}")
        print(f"group {group_idx}: {webui_png}")
        print(f"group {group_idx}: {summary_path}")

        if not args.no_web:
            web_push(
                figure,
                page=WEBUI_PAGE,
                slot=group_idx - 1,
                title=f"训练集面内外 RMS 散点（组{group_idx}）",
                port=int(args.web_port),
            )
        plt.close(figure)

    legacy_combined = [
        args.output_dir / "fig3_x_inout_rms_scatter_201_202_301_302.png",
        args.output_dir / "fig3_x_inout_rms_scatter_201_202_301_302.pdf",
        args.output_dir / "fig3_x_inout_rms_scatter_201_202_301_302_summary.json",
        WEBUI_OUTPUT_DIR / "fig3_x_inout_rms_scatter_201_202_301_302.png",
        args.output_dir / "fig3_x_inout_rms_scatter_group1.png",
        args.output_dir / "fig3_x_inout_rms_scatter_group1.pdf",
        args.output_dir / "fig3_x_inout_rms_scatter_group1_summary.json",
        args.output_dir / "fig3_x_inout_rms_scatter_group2.png",
        args.output_dir / "fig3_x_inout_rms_scatter_group2.pdf",
        args.output_dir / "fig3_x_inout_rms_scatter_group2_summary.json",
        WEBUI_OUTPUT_DIR / "fig3_x_inout_rms_scatter_group1.png",
        WEBUI_OUTPUT_DIR / "fig3_x_inout_rms_scatter_group2.png",
    ]
    for path in legacy_combined:
        if path.exists():
            path.unlink()
            print(f"removed legacy: {path}")


if __name__ == "__main__":
    main()
