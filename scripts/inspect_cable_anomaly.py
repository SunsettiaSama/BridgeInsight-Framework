from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

ensure_paths()

TARGETS = ["201", "202", "301", "302"]
REF = ["101", "102"]
ALL = TARGETS + REF


def time_token(row: dict) -> str:
    raw = row.get("raw_time")
    if raw:
        return str(raw)
    return f"{row.get('hour', '')}{row.get('minute', '')}{row.get('second', '')}"


def main() -> None:
    meta = json.loads(
        (ROOT / "results/preprocessed_vibration_metadata/files_after_lackness_filter.json").read_text(
            encoding="utf-8"
        )
    )
    rows = meta if isinstance(meta, list) else meta["records"]

    print("=== NON-HOUR-ALIGNED FILES (minute!=00 or second!=00) ===")
    for cable in ALL:
        sensor_id = f"ST-VIC-C34-{cable}-01"
        sensor_rows = [row for row in rows if row.get("sensor_id") == sensor_id]
        non_hour = [
            row
            for row in sensor_rows
            if str(row.get("minute", "00")) != "00" or str(row.get("second", "00")) != "00"
        ]
        tokens = collections.Counter(time_token(row) for row in non_hour)
        ratio = 100.0 * len(non_hour) / max(len(sensor_rows), 1)
        print(
            f"C34-{cable}: total={len(sensor_rows)} non_hour={len(non_hour)} ({ratio:.1f}%) "
            f"top_tokens={tokens.most_common(8)}"
        )

    print("\n=== SHARED ANOMALOUS TIMESTAMPS ACROSS TARGET CABLES ===")
    token_sets = []
    for cable in TARGETS:
        sensor_id = f"ST-VIC-C34-{cable}-01"
        tokens = {time_token(row) for row in rows if row.get("sensor_id") == sensor_id}
        token_sets.append(tokens)
        ordered = sorted(tokens)
        print(
            f"C34-{cable}: unique_tokens={len(tokens)} "
            f"sample={ordered[:10]} ... {ordered[-5:]}"
        )
    common = set.intersection(*token_sets)
    print(f"common across 201/202/301/302 count={len(common)} sample={sorted(common)[:20]}")

    extractor = VICWindowExtractor(enable_denoise=False)
    print("\n=== DC OFFSET / RMS SURVEY (first 60 files per cable) ===")
    for cable in ALL:
        sensor_id = f"ST-VIC-C34-{cable}-01"
        sensor_rows = [row for row in rows if row.get("sensor_id") == sensor_id]
        means: list[float] = []
        stds: list[float] = []
        peaks: list[float] = []
        ok = 0
        bad = 0
        for row in sensor_rows[:60]:
            file_path = row["file_path"]
            data = extractor.load_file(file_path)
            signal = extractor.extract_window_from_data(
                data,
                0,
                3000,
                metadata={"window_index": 0},
                file_path=file_path,
            )
            if signal is None:
                bad += 1
                continue
            ok += 1
            arr = np.asarray(signal, dtype=np.float64)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            peaks.append(float(np.max(np.abs(arr))))
        print(
            f"C34-{cable}: ok={ok} bad={bad} mean_dc={statistics.mean(means):+.4f} "
            f"std_mean={statistics.mean(stds):.4f} peak_mean={statistics.mean(peaks):.4f} "
            f"peak_max={max(peaks):.4f}"
        )

    infer = json.loads(
        (ROOT / "results/augment/rounds/round_01/inference.json").read_text(encoding="utf-8")
    )
    print("\n=== SAMPLE INFERENCE FOR TARGET CABLES ===")
    for cable in TARGETS:
        subset = [
            row
            for row in infer["records"]
            if f"C34-{cable}-" in row.get("inplane_sensor_id", "")
        ][:3]
        for row in subset:
            print(
                f"C34-{cable}: {Path(row['inplane_file_path']).name} "
                f"pred={row['prediction']} in={row['inplane_prediction']} out={row['outplane_prediction']}"
            )


if __name__ == "__main__":
    main()
