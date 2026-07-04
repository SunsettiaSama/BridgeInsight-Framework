from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.config import CN_FONT, ENG_FONT
from src.visualize_tools.web_dashboard import push as web_push


ROUND_DIR = PROJECT_ROOT / "results" / "augment" / "rounds" / "round_06"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figure_paintings"
    / "figs_for_thesis"
    / "Chapter3"
    / "non_body"
)

LABEL_NAMES = ("Normal", "VIV", "RWIV", "Others")
LABEL_COLORS = ("#7895C1", "#8074C8", "#E3625D", "#992224")
TARGET_OUTPLANE_SENSORS = (
    "ST-VIC-C34-201-02",
    "ST-VIC-C34-202-02",
    "ST-VIC-C34-301-02",
    "ST-VIC-C34-302-02",
)
SENSOR_RE = re.compile(r"(ST-VIC-[A-Z0-9]+-[0-9]+-[0-9]+)")

WEBUI_PORT = 5678
WEBUI_PAGE = "fig2_x C34面外传感器统计与样本"
FS = 50.0
WINDOW_SIZE = 3000
TRIM_SECONDS = 20.0


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_rows(round_dir: Path) -> list[dict]:
    payload = load_json(round_dir / "merged_training.json")
    if isinstance(payload, list):
        return payload
    for key in ("records", "entries", "annotations"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError("merged_training.json 缺少 records/entries/annotations 列表")


def sensor_from_path(path: str | None) -> str:
    match = SENSOR_RE.search(str(path or ""))
    return match.group(1) if match else "unknown"


def row_label(row: dict) -> int:
    return int(row.get("annotation", row.get("class_id", row.get("prediction", 0))))


def normalized_counts(counter: Counter, total: int | None = None) -> np.ndarray:
    denom = int(total if total is not None else sum(counter.values()))
    if denom <= 0:
        return np.zeros(len(LABEL_NAMES), dtype=float)
    return np.array([counter.get(i, 0) / denom for i in range(len(LABEL_NAMES))], dtype=float)


def count_training_by_sensor(rows: list[dict]) -> tuple[dict[str, Counter], Counter]:
    by_sensor: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for row in rows:
        label = row_label(row)
        sensors = {
            sensor_from_path(row.get("file_path")),
            sensor_from_path(row.get("inplane_file_path")),
            sensor_from_path(row.get("outplane_file_path")),
        }
        sensors.discard("unknown")
        for sensor in sensors:
            by_sensor[sensor][label] += 1
            totals[sensor] += 1
    return by_sensor, totals


def count_inference_by_sensor(records: list[dict]) -> tuple[dict[str, Counter], Counter]:
    by_sensor: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for row in records:
        sensor = row.get("outplane_sensor_id") or sensor_from_path(row.get("outplane_file_path"))
        label = int(row.get("outplane_prediction", row.get("prediction", 0)))
        by_sensor[sensor][label] += 1
        totals[sensor] += 1
    return by_sensor, totals


def select_display_sensors(infer_totals: Counter, limit: int = 12) -> list[str]:
    selected = list(TARGET_OUTPLANE_SENSORS)
    for sensor, _ in infer_totals.most_common():
        if not sensor.endswith("-02"):
            continue
        if sensor not in selected:
            selected.append(sensor)
        if len(selected) >= limit:
            break
    return selected


def setup_axes(ax, title: str, ylabel: str = "Proportion") -> None:
    ax.set_title(title, fontproperties=ENG_FONT, fontsize=13, pad=10)
    ax.set_ylabel(ylabel, fontproperties=ENG_FONT)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    print(f"saved: {path}")
    return path


def plot_prediction_distribution(
    infer_by_sensor: dict[str, Counter],
    infer_totals: Counter,
    sensors: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(sensors))
    bottom = np.zeros(len(sensors), dtype=float)
    for label_idx, (label, color) in enumerate(zip(LABEL_NAMES, LABEL_COLORS)):
        values = np.array([
            normalized_counts(infer_by_sensor.get(sensor, Counter()), infer_totals.get(sensor, 0))[label_idx]
            for sensor in sensors
        ])
        ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.6)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.02)
    setup_axes(ax, "Round 6 Inference Label Distribution by Outplane Sensor")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    return fig


def plot_training_distribution(
    train_by_sensor: dict[str, Counter],
    train_totals: Counter,
    sensors: list[str],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(sensors))
    bottom = np.zeros(len(sensors), dtype=float)
    for label_idx, (label, color) in enumerate(zip(LABEL_NAMES, LABEL_COLORS)):
        values = np.array([
            normalized_counts(train_by_sensor.get(sensor, Counter()), train_totals.get(sensor, 0))[label_idx]
            for sensor in sensors
        ])
        ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.6)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.02)
    setup_axes(ax, "Round 6 Training Label Distribution on High-Others Sensors")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    return fig


def plot_train_vs_infer_others(
    train_by_sensor: dict[str, Counter],
    train_totals: Counter,
    infer_by_sensor: dict[str, Counter],
    infer_totals: Counter,
    sensors: list[str],
) -> plt.Figure:
    train_values = np.array([
        normalized_counts(train_by_sensor.get(sensor, Counter()), train_totals.get(sensor, 0))[3]
        for sensor in sensors
    ])
    infer_values = np.array([
        normalized_counts(infer_by_sensor.get(sensor, Counter()), infer_totals.get(sensor, 0))[3]
        for sensor in sensors
    ])
    x = np.arange(len(sensors))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, train_values, width=width, label="Training Others", color="#7895C1")
    ax.bar(x + width / 2, infer_values, width=width, label="Inference Others", color="#992224")
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    setup_axes(ax, "Others Ratio: Training Labels vs Round 6 Inference")
    ax.legend(frameon=False)
    for idx, value in enumerate(infer_values):
        ax.text(idx + width / 2, min(value + 0.025, 1.02), f"{value:.0%}", ha="center", va="bottom", fontsize=9)
    return fig


def plot_confidence_uncertainty(records: list[dict], sensors: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True)
    confidence_data = []
    uncertainty_data = []
    for sensor in sensors:
        subset = [row for row in records if outplane_sensor(row) == sensor]
        confidence_data.append([max(row.get("outplane_proba", row.get("proba", [0.0]))) for row in subset])
        uncertainty_data.append([float(row.get("outplane_uncertainty", row.get("uncertainty", 0.0))) for row in subset])

    axes[0].boxplot(confidence_data, showfliers=False, patch_artist=True)
    axes[1].boxplot(uncertainty_data, showfliers=False, patch_artist=True)
    for ax, title, ylabel in (
        (axes[0], "Outplane Max Probability", "Max probability"),
        (axes[1], "Outplane Uncertainty", "Uncertainty"),
    ):
        ax.set_xticks(np.arange(1, len(sensors) + 1))
        ax.set_xticklabels(sensors, rotation=35, ha="right", fontsize=8)
        setup_axes(ax, title, ylabel=ylabel)
    return fig


def outplane_sensor(row: dict) -> str:
    return row.get("outplane_sensor_id") or sensor_from_path(row.get("outplane_file_path"))


def select_sample_records(records: list[dict], sensors: list[str], samples_per_sensor: int) -> list[dict]:
    selected: list[dict] = []
    for sensor in sensors:
        candidates = [
            row
            for row in records
            if outplane_sensor(row) == sensor and int(row.get("outplane_prediction", row.get("prediction", 0))) == 3
        ]
        candidates.sort(
            key=lambda row: (
                float(row.get("outplane_proba", row.get("proba", [0, 0, 0, 0]))[3]),
                -float(row.get("outplane_uncertainty", row.get("uncertainty", 0.0))),
            ),
            reverse=True,
        )
        selected.extend(candidates[:samples_per_sensor])
    if not selected:
        raise ValueError("未筛选到目标传感器的 Others 推理样本")
    return selected


def load_window(unpacker: UNPACK, file_path: str, window_index: int) -> np.ndarray:
    raw = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=float)
    start = int(window_index) * WINDOW_SIZE
    end = start + WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window={window_index} len={len(raw)} file={file_path}")
    return raw[start:end]


def sample_title(row: dict) -> str:
    sensor = outplane_sensor(row)
    wi = int(row.get("window_index", 0))
    proba = row.get("outplane_proba", row.get("proba", [0.0, 0.0, 0.0, 0.0]))
    uncertainty = float(row.get("outplane_uncertainty", row.get("uncertainty", 0.0)))
    ts = row.get("timestamp", [])
    time_text = f"{int(ts[0]):02d}/{int(ts[1]):02d} {int(ts[2]):02d}:00" if len(ts) >= 3 else ""
    return f"{sensor} | win={wi} | P(Others)={float(proba[3]):.3f} | U={uncertainty:.3f} | {time_text}"


def plot_sample_triptych(row: dict, unpacker: UNPACK) -> plt.Figure:
    in_sig = load_window(unpacker, row["inplane_file_path"], int(row.get("window_index", 0)))
    out_sig = load_window(unpacker, row["outplane_file_path"], int(row.get("window_index", 0)))
    trim = min(int(TRIM_SECONDS * FS), len(in_sig), len(out_sig))
    time_axis = np.arange(trim) / FS
    in_plot = in_sig[:trim]
    out_plot = out_sig[:trim]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(time_axis, in_plot, color="#333333", linewidth=0.9)
    axes[0].set_title("Inplane Time Series", fontproperties=ENG_FONT, fontsize=12)
    axes[0].set_xlabel("Time (s)", fontproperties=ENG_FONT)
    axes[0].set_ylabel("Acceleration", fontproperties=ENG_FONT)

    axes[1].plot(time_axis, out_plot, color="#992224", linewidth=0.9)
    axes[1].set_title("Outplane Time Series", fontproperties=ENG_FONT, fontsize=12)
    axes[1].set_xlabel("Time (s)", fontproperties=ENG_FONT)

    axes[2].plot(in_plot, out_plot, color="#8074C8", linewidth=0.7, alpha=0.85)
    axes[2].scatter(in_plot[0], out_plot[0], s=28, color="#7895C1", label="start")
    axes[2].scatter(in_plot[-1], out_plot[-1], s=28, color="#E3625D", label="end")
    axes[2].set_title("Inplane-Outplane Trajectory", fontproperties=ENG_FONT, fontsize=12)
    axes[2].set_xlabel("Inplane acceleration", fontproperties=ENG_FONT)
    axes[2].set_ylabel("Outplane acceleration", fontproperties=ENG_FONT)
    axes[2].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(sample_title(row), fontproperties=ENG_FONT, fontsize=13, y=1.02)
    return fig


def plot_sample_overview(rows: list[dict], unpacker: UNPACK) -> plt.Figure:
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for row_idx, row in enumerate(rows):
        in_sig = load_window(unpacker, row["inplane_file_path"], int(row.get("window_index", 0)))
        out_sig = load_window(unpacker, row["outplane_file_path"], int(row.get("window_index", 0)))
        trim = min(int(TRIM_SECONDS * FS), len(in_sig), len(out_sig))
        time_axis = np.arange(trim) / FS
        in_plot = in_sig[:trim]
        out_plot = out_sig[:trim]
        axes[row_idx, 0].plot(time_axis, in_plot, color="#333333", linewidth=0.8)
        axes[row_idx, 1].plot(time_axis, out_plot, color="#992224", linewidth=0.8)
        axes[row_idx, 2].plot(in_plot, out_plot, color="#8074C8", linewidth=0.6)
        axes[row_idx, 0].set_ylabel(outplane_sensor(row), fontsize=9)
        axes[row_idx, 0].set_title("Inplane" if row_idx == 0 else "", fontproperties=ENG_FONT)
        axes[row_idx, 1].set_title("Outplane" if row_idx == 0 else "", fontproperties=ENG_FONT)
        axes[row_idx, 2].set_title("Trajectory" if row_idx == 0 else "", fontproperties=ENG_FONT)
        for ax in axes[row_idx]:
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.tick_params(labelsize=8)
    fig.suptitle("High-confidence Others Samples on C34 Outplane Sensors", fontproperties=ENG_FONT, fontsize=14)
    return fig


def push_and_save(figures: list[tuple[plt.Figure, str, str]], output_dir: Path) -> None:
    for fig, filename, _ in figures:
        save_figure(fig, output_dir, filename)

    for slot, (fig, _, title) in enumerate(figures):
        web_push(fig, page=WEBUI_PAGE, slot=slot, title=title, port=WEBUI_PORT, page_cols=2)
        plt.close(fig)


def print_summary(
    train_by_sensor: dict[str, Counter],
    train_totals: Counter,
    infer_by_sensor: dict[str, Counter],
    infer_totals: Counter,
    sensors: list[str],
) -> None:
    print("sensor,train_total,train_others_ratio,infer_total,infer_others_ratio")
    for sensor in sensors:
        train_ratio = normalized_counts(train_by_sensor.get(sensor, Counter()), train_totals.get(sensor, 0))[3]
        infer_ratio = normalized_counts(infer_by_sensor.get(sensor, Counter()), infer_totals.get(sensor, 0))[3]
        print(f"{sensor},{train_totals.get(sensor, 0)},{train_ratio:.4f},{infer_totals.get(sensor, 0)},{infer_ratio:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="C34 面外传感器 Others 偏置统计与样本图（非正文图）")
    parser.add_argument("--round-dir", type=Path, default=ROUND_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-sensors", type=int, default=12)
    parser.add_argument("--samples-per-sensor", type=int, default=1)
    args = parser.parse_args()

    train_rows = load_training_rows(args.round_dir)
    inference = load_json(args.round_dir / "inference.json")
    records = inference.get("records", [])
    if not records:
        raise ValueError("inference.json records 为空")

    train_by_sensor, train_totals = count_training_by_sensor(train_rows)
    infer_by_sensor, infer_totals = count_inference_by_sensor(records)
    sensors = select_display_sensors(infer_totals, limit=int(args.top_sensors))
    target_sensors = [sensor for sensor in TARGET_OUTPLANE_SENSORS if sensor in infer_totals]
    sample_rows = select_sample_records(records, target_sensors, int(args.samples_per_sensor))

    print_summary(train_by_sensor, train_totals, infer_by_sensor, infer_totals, sensors)
    unpacker = UNPACK(init_path=False)

    figures: list[tuple[plt.Figure, str, str]] = [
        (
            plot_prediction_distribution(infer_by_sensor, infer_totals, sensors),
            "fig2_x_outplane_sensor_inference_distribution.png",
            "面外传感器推理标签分布",
        ),
        (
            plot_training_distribution(train_by_sensor, train_totals, target_sensors),
            "fig2_x_outplane_sensor_training_distribution.png",
            "目标面外传感器训练标注分布",
        ),
        (
            plot_train_vs_infer_others(train_by_sensor, train_totals, infer_by_sensor, infer_totals, target_sensors),
            "fig2_x_outplane_sensor_others_ratio_compare.png",
            "训练/推理 Others 比例对比",
        ),
        (
            plot_confidence_uncertainty(records, target_sensors),
            "fig2_x_outplane_sensor_confidence_uncertainty.png",
            "目标面外传感器置信度与困惑度",
        ),
        (
            plot_sample_overview(sample_rows, unpacker),
            "fig2_x_outplane_sensor_sample_overview.png",
            "高置信 Others 抽样时程与轨迹总览",
        ),
    ]

    for index, row in enumerate(sample_rows, start=1):
        sensor = outplane_sensor(row)
        figures.append(
            (
                plot_sample_triptych(row, unpacker),
                f"fig2_x_outplane_sensor_sample_{index}_{sensor}.png",
                f"{sensor} 抽样面内/面外时程与轨迹",
            )
        )

    push_and_save(figures, args.output_dir)
    print(f"dashboard page: {WEBUI_PAGE}, port={WEBUI_PORT}")


if __name__ == "__main__":
    main()
