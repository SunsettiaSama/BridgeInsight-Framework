from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.config import ENG_FONT
from src.visualize_tools.web_dashboard import DEFAULT_PORT as DASHBOARD_DEFAULT_PORT
from src.visualize_tools.web_dashboard import push as web_push


ROUND_DIR = PROJECT_ROOT / "results" / "augment" / "rounds" / "round_09"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figure_paintings" / "figs_for_thesis" / "Chapter3"
TARGET_CABLES = ("ST-VIC-C34-201", "ST-VIC-C34-202", "ST-VIC-C34-301", "ST-VIC-C34-302")
SENSOR_RE = re.compile(r"(ST-VIC-[A-Z0-9]+-[0-9]+)-([0-9]+)")
WINDOW_SIZE = 3000
FS = 50.0
TRIM_SECONDS = 12.0
WEBUI_PAGE = "fig3_x 面内外传感器同源性随机抽样"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cable_key(path: str | None) -> str:
    match = SENSOR_RE.search(str(path or ""))
    return match.group(1) if match else ""


def sensor_suffix(path: str | None) -> str:
    match = SENSOR_RE.search(str(path or ""))
    return match.group(2) if match else "unknown"


def load_training_rows(round_dir: Path) -> list[dict]:
    payload = load_json(round_dir / "merged_training.json")
    if isinstance(payload, list):
        return payload
    for key in ("records", "entries", "annotations"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError("merged_training.json 缺少 records/entries/annotations 列表")


def paired_rows_by_cable(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {key: [] for key in TARGET_CABLES}
    for row in rows:
        in_path = row.get("inplane_file_path") or row.get("file_path")
        out_path = row.get("outplane_file_path")
        key = cable_key(in_path)
        if key not in grouped or cable_key(out_path) != key:
            continue
        grouped[key].append(row)
    return grouped


def select_random_rows(grouped: dict[str, list[dict]], samples_per_cable: int, seed: int) -> list[tuple[str, dict]]:
    rng = random.Random(int(seed))
    selected: list[tuple[str, dict]] = []
    for cable in TARGET_CABLES:
        rows = list(grouped.get(cable, []))
        if not rows:
            raise ValueError(f"没有找到配对样本：{cable}")
        rows.sort(key=lambda item: (str(item.get("inplane_file_path") or item.get("file_path")), int(item.get("window_index", 0))))
        same_file = [
            row
            for row in rows
            if str(row.get("inplane_file_path") or row.get("file_path")) == str(row.get("outplane_file_path"))
        ]
        canonical = [
            row
            for row in rows
            if sensor_suffix(row.get("inplane_file_path") or row.get("file_path")) == "01"
            and sensor_suffix(row.get("outplane_file_path")) == "02"
        ]
        chosen: list[dict] = []
        if same_file:
            chosen.extend(rng.sample(same_file, min(1, int(samples_per_cable), len(same_file))))
        remaining = int(samples_per_cable) - len(chosen)
        if remaining > 0 and canonical:
            chosen.extend(rng.sample(canonical, min(remaining, len(canonical))))
        remaining = int(samples_per_cable) - len(chosen)
        if remaining > 0:
            chosen.extend(rng.sample(rows, min(remaining, len(rows))))
        selected.extend((cable, row) for row in chosen)
    return selected


def load_window(unpacker: UNPACK, file_path: str, window_index: int, window_size: int) -> np.ndarray:
    raw = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float64).reshape(-1)
    start = int(window_index) * int(window_size)
    end = start + int(window_size)
    if end > int(raw.size):
        raise ValueError(f"窗口越界：window={window_index} len={raw.size} file={file_path}")
    return raw[start:end]


def standardize(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    std = float(np.std(arr))
    if std <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / std


def sample_metrics(in_sig: np.ndarray, out_sig: np.ndarray) -> dict[str, float | bool]:
    n = min(int(in_sig.size), int(out_sig.size))
    a = np.asarray(in_sig[:n], dtype=np.float64)
    b = np.asarray(out_sig[:n], dtype=np.float64)
    az = standardize(a)
    bz = standardize(b)
    corr = float(np.corrcoef(az, bz)[0, 1]) if n > 1 else float("nan")
    return {
        "corr": corr,
        "normalized_rmse": float(np.sqrt(np.mean((az - bz) ** 2))),
        "raw_max_abs_diff": float(np.max(np.abs(a - b))) if n else float("nan"),
        "raw_mean_abs_diff": float(np.mean(np.abs(a - b))) if n else float("nan"),
        "exact_equal": bool(np.array_equal(a, b)),
    }


def plot_samples(records: list[dict], metrics: list[dict], samples_per_cable: int, trim_seconds: float) -> plt.Figure:
    row_count = len(TARGET_CABLES)
    col_count = int(samples_per_cable)
    fig, axes = plt.subplots(row_count, col_count, figsize=(4.2 * col_count, 2.7 * row_count), sharex=True)
    if row_count == 1:
        axes = np.array([axes])
    if col_count == 1:
        axes = axes.reshape(row_count, 1)

    for index, item in enumerate(metrics):
        row_idx = TARGET_CABLES.index(str(item["cable"]))
        col_idx = int(item["sample_rank"])
        ax = axes[row_idx, col_idx]
        t = np.asarray(item["time_s"])
        ax.plot(t, item["in_z"], color="#333333", linewidth=0.8, label="Inplane")
        ax.plot(t, item["out_z"], color="#992224", linewidth=0.8, alpha=0.82, label="Outplane")
        ax.set_title(
            f"{item['pair_type']} | corr={item['corr']:.4f} rmse={item['normalized_rmse']:.3f}",
            fontproperties=ENG_FONT,
            fontsize=10,
        )
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(labelsize=8)
        if col_idx == 0:
            ax.set_ylabel(str(item["cable"]).replace("ST-VIC-C34-", "C34-"), fontproperties=ENG_FONT)
        if row_idx == row_count - 1:
            ax.set_xlabel("Time (s)", fontproperties=ENG_FONT)
        if row_idx == 0 and col_idx == 0:
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Random inplane/outplane waveform identity check ({trim_seconds:g}s windows)",
        fontproperties=ENG_FONT,
        fontsize=14,
        y=1.01,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="随机抽样检查 C34 201/202/301/302 面内外信号是否近似相同")
    parser.add_argument("--round-dir", type=Path, default=ROUND_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--samples-per-cable", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--trim-seconds", type=float, default=TRIM_SECONDS)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--web-port", type=int, default=DASHBOARD_DEFAULT_PORT)
    args = parser.parse_args()

    rows = load_training_rows(args.round_dir)
    grouped = paired_rows_by_cable(rows)
    selected = select_random_rows(grouped, args.samples_per_cable, args.seed)
    unpacker = UNPACK(init_path=False)

    plot_records: list[dict] = []
    metrics: list[dict] = []
    per_cable_rank = {key: 0 for key in TARGET_CABLES}
    trim_len = int(float(args.trim_seconds) * FS)
    for cable, row in selected:
        in_path = row.get("inplane_file_path") or row.get("file_path")
        out_path = row.get("outplane_file_path")
        window_index = int(row.get("window_index", 0))
        in_sig = load_window(unpacker, str(in_path), window_index, int(args.window_size))
        out_sig = load_window(unpacker, str(out_path), window_index, int(args.window_size))
        stats = sample_metrics(in_sig, out_sig)
        take = min(trim_len, int(in_sig.size), int(out_sig.size))
        rank = per_cable_rank[cable]
        per_cable_rank[cable] += 1
        item = {
            "cable": cable,
            "sample_rank": rank,
            "window_index": window_index,
            "inplane_file_path": str(in_path),
            "outplane_file_path": str(out_path),
            "pair_type": f"{sensor_suffix(str(in_path))}_{sensor_suffix(str(out_path))}",
            "same_file": bool(str(in_path) == str(out_path)),
            "time_s": (np.arange(take, dtype=float) / FS).tolist(),
            "in_z": standardize(in_sig[:take]).tolist(),
            "out_z": standardize(out_sig[:take]).tolist(),
            **stats,
        }
        metrics.append(item)
        plot_records.append(row)

    figure = plot_samples(plot_records, metrics, int(args.samples_per_cable), float(args.trim_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "fig3_x_inout_sensor_identity_random_samples.png"
    figure.tight_layout()
    figure.savefig(figure_path, dpi=220)

    metrics_path = args.output_dir / "fig3_x_inout_sensor_identity_random_samples_metrics.json"
    compact_metrics = [
        {key: value for key, value in item.items() if key not in {"time_s", "in_z", "out_z"}}
        for item in metrics
    ]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(compact_metrics, f, ensure_ascii=False, indent=2)

    print(f"saved: {figure_path}")
    print(f"saved: {metrics_path}")
    for cable in TARGET_CABLES:
        cable_metrics = [item for item in compact_metrics if item["cable"] == cable]
        mean_corr = float(np.mean([float(item["corr"]) for item in cable_metrics]))
        mean_rmse = float(np.mean([float(item["normalized_rmse"]) for item in cable_metrics]))
        exact_count = sum(1 for item in cable_metrics if bool(item["exact_equal"]))
        print(f"{cable}: mean_corr={mean_corr:.6f}, mean_norm_rmse={mean_rmse:.6f}, exact_equal={exact_count}/{len(cable_metrics)}")

    if not args.no_web:
        web_push(figure, page=WEBUI_PAGE, slot=0, title="C34 面内外信号随机抽样一致性检查", port=int(args.web_port))
    plt.close(figure)


if __name__ == "__main__":
    main()
