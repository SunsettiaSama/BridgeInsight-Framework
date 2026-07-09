from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
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

RAW_RESULT_PATH = data_config.PROJECT_ROOT / data_config.CHAPTER4["predictions_enriched_raw"]
CACHE_PATH = (
    PROJECT_ROOT
    / "results"
    / "figure_paintings"
    / "figs_for_thesis"
    / "Chapter4"
    / "fig4_x_c34_201_202_rms_ratio_snapshot.json"
)
PAGE_NAME = "fig4_x C34-201-202 面内外RMS比值"
WINDOW_SIZE = 3000
FS = 50.0
TARGET_CABLES = {
    ("ST-VIC-C34-201-01", "ST-VIC-C34-201-02"): "C34-201",
    ("ST-VIC-C34-202-01", "ST-VIC-C34-202-02"): "C34-202",
}
CLASS_LABELS = {
    0: "随机振动",
    1: "涡激共振",
    2: "风雨振",
    3: "其他振动",
}
_palette = list(get_full_color_map("discrete").colors)
CLASS_COLORS = {
    0: _palette[3],
    1: _palette[0],
    2: _palette[9],
    3: _palette[7],
}
CABLE_COLORS = {
    "C34-201": _palette[2],
    "C34-202": _palette[8],
}


def _window_rms(signal: np.ndarray) -> float:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(np.square(arr))))


def _extract_window(raw: np.ndarray, window_idx: int, window_size: int) -> np.ndarray | None:
    start = int(window_idx) * int(window_size)
    end = start + int(window_size)
    if end > int(raw.size):
        return None
    return raw[start:end]


def _process_file_pair(task: tuple[str, str, list[dict[str, Any]], int, str]) -> list[dict[str, Any]]:
    in_fp, out_fp, rows, window_size, cable_id = task
    unpacker = UNPACK(init_path=False)
    in_raw = np.asarray(unpacker.VIC_DATA_Unpack(in_fp), dtype=np.float64).reshape(-1)
    out_raw = np.asarray(unpacker.VIC_DATA_Unpack(out_fp), dtype=np.float64).reshape(-1)

    records: list[dict[str, Any]] = []
    for row in rows:
        in_window = _extract_window(in_raw, int(row["window_idx"]), int(window_size))
        out_window = _extract_window(out_raw, int(row["window_idx"]), int(window_size))
        if in_window is None or out_window is None:
            continue
        rms_in = _window_rms(in_window)
        rms_out = _window_rms(out_window)
        if not (np.isfinite(rms_in) and np.isfinite(rms_out) and rms_in > 0 and rms_out > 0):
            continue
        ratio = rms_in / rms_out
        records.append(
            {
                "sample_idx": int(row["sample_idx"]),
                "cable_id": cable_id,
                "timestamp": row.get("timestamp", []),
                "window_idx": int(row["window_idx"]),
                "predicted_class": int(row["predicted_class"]),
                "inplane_sensor_id": row["inplane_sensor_id"],
                "outplane_sensor_id": row["outplane_sensor_id"],
                "rms_inplane": rms_in,
                "rms_outplane": rms_out,
                "rms_ratio_in_over_out": ratio,
                "log10_ratio": float(np.log10(ratio)),
            }
        )
    return records


def _task_key(task: tuple[str, str, list[dict[str, Any]], int, str]) -> str:
    in_fp, out_fp, _, _, cable_id = task
    return json.dumps([cable_id, in_fp, out_fp], ensure_ascii=False, separators=(",", ":"))


def _cache_aux_paths(cache_path: Path) -> tuple[Path, Path]:
    return (
        cache_path.with_suffix(".records.jsonl"),
        cache_path.with_suffix(".progress.json"),
    )


def _load_completed_tasks(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    with progress_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return set(str(x) for x in payload.get("completed_task_keys", []))


def _save_progress(progress_path: Path, completed: set[str], total_tasks: int) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = progress_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_tasks": int(total_tasks),
                "completed_tasks": int(len(completed)),
                "completed_task_keys": sorted(completed),
            },
            f,
            ensure_ascii=False,
        )
    tmp.replace(progress_path)


def _append_partial_records(partial_path: Path, task_key: str, rows: list[dict[str, Any]]) -> None:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"task_key": task_key, "record": row}, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def _load_partial_records(partial_path: Path) -> list[dict[str, Any]]:
    if not partial_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with partial_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line)["record"])
    return records


def _load_target_tasks(result_path: Path) -> list[tuple[str, str, list[dict[str, Any]], int, str]]:
    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    predictions = result.get("predictions", {})
    sample_metadata = result.get("sample_metadata", {})
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for idx_text, pred in predictions.items():
        meta = sample_metadata.get(str(idx_text))
        if meta is None:
            continue
        pair = (meta.get("inplane_sensor_id"), meta.get("outplane_sensor_id"))
        cable_id = TARGET_CABLES.get(pair)
        if cable_id is None:
            continue
        in_fp = str(meta.get("inplane_file_path") or "")
        out_fp = str(meta.get("outplane_file_path") or "")
        window_idx = meta.get("window_idx")
        if not in_fp or not out_fp or window_idx is None:
            continue
        grouped[(in_fp, out_fp, cable_id)].append(
            {
                "sample_idx": int(idx_text),
                "window_idx": int(window_idx),
                "timestamp": meta.get("timestamp", []),
                "predicted_class": int(pred),
                "inplane_sensor_id": pair[0],
                "outplane_sensor_id": pair[1],
            }
        )

    return [
        (in_fp, out_fp, sorted(rows, key=lambda r: int(r["window_idx"])), WINDOW_SIZE, cable_id)
        for (in_fp, out_fp, cable_id), rows in grouped.items()
    ]


def build_snapshot(
    result_path: Path,
    cache_path: Path,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    tasks = _load_target_tasks(result_path)
    if not tasks:
        raise ValueError(f"全量识别结果中没有 C34-201/202 样本：{result_path}")

    partial_path, progress_path = _cache_aux_paths(cache_path)
    completed = _load_completed_tasks(progress_path)
    pending = [task for task in tasks if _task_key(task) not in completed]
    print(
        f"  RMS 任务：total={len(tasks)} completed={len(completed)} pending={len(pending)} "
        f"workers={workers} batch_size={batch_size}",
        flush=True,
    )

    worker_count = max(1, int(workers))
    batch_count = max(1, int(batch_size))
    processed_now = 0
    for start in range(0, len(pending), batch_count):
        batch = pending[start:start + batch_count]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_process_file_pair, task): task for task in batch}
            for future in as_completed(futures):
                task = futures[future]
                key = _task_key(task)
                rows = future.result()
                _append_partial_records(partial_path, key, rows)
                completed.add(key)
                processed_now += 1
                total_done = len(completed)
                if processed_now % 20 == 0 or total_done == len(tasks):
                    _save_progress(progress_path, completed, len(tasks))
                    print(
                        f"  RMS 计算进度：{total_done}/{len(tasks)} file pairs, "
                        f"records+={sum(1 for _ in rows)}",
                        flush=True,
                    )
        _save_progress(progress_path, completed, len(tasks))

    records = _load_partial_records(partial_path)
    if not records:
        raise ValueError("C34-201/202 未计算出有效 RMS 记录")

    source_stat = result_path.stat()
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_result": str(result_path),
        "source_mtime": source_stat.st_mtime,
        "source_size": source_stat.st_size,
        "window_size": WINDOW_SIZE,
        "fs": FS,
        "target_cables": sorted(set(TARGET_CABLES.values())),
        "label_source": "augment_predictions_enriched_raw.predictions",
        "record_count": len(records),
        "records": sorted(records, key=lambda r: (r["cable_id"], r["sample_idx"])),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_snapshot(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        raise FileNotFoundError(f"RMS 快照不存在：{cache_path}")
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("label_source") != "augment_predictions_enriched_raw.predictions":
        raise ValueError("RMS 快照 label_source 不符合当前语义，请使用 --refresh-cache 重建")
    return payload


def _records_array(records: list[dict[str, Any]], cable_id: str) -> np.ndarray:
    return np.asarray(
        [
            [
                float(r["rms_inplane"]),
                float(r["rms_outplane"]),
                float(r["rms_ratio_in_over_out"]),
                int(r["predicted_class"]),
            ]
            for r in records
            if r["cable_id"] == cable_id
        ],
        dtype=np.float64,
    )


def _setup_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.6, color=ANNOTATION_COLOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 5)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=CLASS_COLORS[class_id],
            markeredgecolor="none",
            markersize=7,
            label=CLASS_LABELS[class_id],
        )
        for class_id in range(4)
    ]


def plot_snapshot(payload: dict[str, Any]) -> plt.Figure:
    records = payload["records"]
    fig, axes = plt.subplots(1, 2, figsize=REC_FIG_SIZE)

    # 左：面内 vs 面外 RMS，观察是否系统性偏离 1:1。
    ax = axes[0]
    all_positive = []
    for cable_id in ("C34-201", "C34-202"):
        arr = _records_array(records, cable_id)
        if arr.size == 0:
            continue
        all_positive.append(arr[:, :2])
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            s=12,
            alpha=0.38,
            c=[CLASS_COLORS[int(x)] for x in arr[:, 3]],
            edgecolors="none",
            rasterized=True,
            label=cable_id,
        )
    merged = np.concatenate(all_positive, axis=0)
    lo = max(float(np.min(merged)) * 0.75, 1e-5)
    hi = float(np.max(merged)) * 1.25
    ax.plot([lo, hi], [lo, hi], color=ANNOTATION_COLOR, linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("面内 RMS / m·s$^{-2}$", fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE - 2)
    ax.set_ylabel("面外 RMS / m·s$^{-2}$", fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE - 2)
    _setup_axis(ax)

    # 右：面内/面外 RMS 比值，按 cable 展示整体偏移。
    ax = axes[1]
    rng = np.random.default_rng(20240706)
    positions = {"C34-201": 0, "C34-202": 1}
    for cable_id in ("C34-201", "C34-202"):
        arr = _records_array(records, cable_id)
        if arr.size == 0:
            continue
        x = np.full(arr.shape[0], positions[cable_id], dtype=float)
        x += rng.normal(0.0, 0.045, size=arr.shape[0])
        ratio = arr[:, 2]
        ax.scatter(
            x,
            ratio,
            s=12,
            alpha=0.35,
            c=[CLASS_COLORS[int(v)] for v in arr[:, 3]],
            edgecolors="none",
            rasterized=True,
        )
        median = float(np.median(ratio))
        p10, p90 = np.percentile(ratio, [10, 90])
        ax.hlines(median, positions[cable_id] - 0.22, positions[cable_id] + 0.22, color=CABLE_COLORS[cable_id], linewidth=2.2)
        ax.vlines(positions[cable_id], p10, p90, color=CABLE_COLORS[cable_id], linewidth=3.0, alpha=0.65)
        ax.text(
            positions[cable_id],
            p90 * 1.08,
            f"median={median:.2f}\nn={len(ratio)}",
            ha="center",
            va="bottom",
            fontproperties=ENG_FONT,
            fontsize=FONT_SIZE - 8,
            color=ANNOTATION_COLOR,
        )

    ax.axhline(1.0, color=ANNOTATION_COLOR, linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["C34-201", "C34-202"], fontproperties=ENG_FONT, fontsize=FONT_SIZE - 4)
    ax.set_ylabel("面内 / 面外 RMS", fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE - 2)
    _setup_axis(ax)

    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        prop=CN_FONT,
        fontsize=FONT_SIZE - 6,
        bbox_to_anchor=(0.5, 1.01),
    )
    note = (
        "label 来自 augment 全量识别结果；"
        f"窗口 {WINDOW_SIZE / FS:.0f} s；快照样本 {payload['record_count']:,} 个"
    )
    fig.text(0.99, 0.01, note, ha="right", va="bottom", fontproperties=CN_FONT, fontsize=FONT_SIZE - 9, color=ANNOTATION_COLOR)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.88, wspace=0.25)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="探索 C34-201/202 的 2023 面内/面外 RMS 比值")
    parser.add_argument("--result", type=Path, default=RAW_RESULT_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--web-port", type=int, default=DASHBOARD_DEFAULT_PORT)
    args = parser.parse_args()

    if args.refresh_cache or not args.cache.exists():
        if args.refresh_cache:
            partial_path, progress_path = _cache_aux_paths(args.cache)
            for path in (args.cache, partial_path, progress_path):
                if path.exists():
                    path.unlink()
                    print(f"removed cache artifact: {path}")
        payload = build_snapshot(
            args.result,
            args.cache,
            int(args.workers),
            int(args.batch_size),
        )
        print(f"saved snapshot: {args.cache}")
    else:
        payload = load_snapshot(args.cache)
        print(f"loaded snapshot: {args.cache}")

    figure = plot_snapshot(payload)
    if not args.no_web:
        web_push(
            figure,
            page=PAGE_NAME,
            slot=0,
            title="C34-201/202 面内-面外 RMS 比值探索",
            port=int(args.web_port),
            page_cols=1,
        )
        print(f"pushed to WebUI: {PAGE_NAME}")
    plt.close(figure)


if __name__ == "__main__":
    main()
