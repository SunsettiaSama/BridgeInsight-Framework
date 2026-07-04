"""Audit in/out pairing logic for staycable index cache and metadata."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "results/preprocessed_vibration_metadata/files_after_lackness_filter.json"
CACHE_PATH = ROOT / "results/full_vib_metadata/staycable_vib202409_index_cache.json"

PAIRS = [
    ("ST-VIC-C34-101-01", "ST-VIC-C34-101-02", "C34-101"),
    ("ST-VIC-C34-102-01", "ST-VIC-C34-102-02", "C34-102"),
    ("ST-VIC-C34-201-01", "ST-VIC-C34-201-02", "C34-201"),
    ("ST-VIC-C34-202-01", "ST-VIC-C34-202-02", "C34-202"),
    ("ST-VIC-C34-301-01", "ST-VIC-C34-301-02", "C34-301"),
    ("ST-VIC-C34-302-01", "ST-VIC-C34-302-02", "C34-302"),
]
SENSOR_RE = re.compile(r"(ST-VIC-[A-Z0-9]+-[0-9]+)-([0-9]+)")


def ts_key(rec: dict) -> tuple[int, int, int]:
    return int(rec["month"]), int(rec["day"]), int(rec["hour"])


def raw_time(rec: dict) -> str:
    minute = rec.get("minute", "00")
    second = rec.get("second", "00")
    return str(rec.get("raw_time") or f"{rec.get('hour', ''):02}{minute:02}{second:02}")


def filename_time_token(path: str) -> str:
    return Path(path).stem.rsplit("_", 1)[-1]


def cable_base_from_path(path: str) -> str:
    name = Path(path).name
    return name.split("_", 1)[0].rsplit("-", 1)[0]


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    print("=== METADATA: hourly key collisions per sensor ===")
    for in_id, out_id, label in PAIRS:
        for sid in (in_id, out_id):
            rows = [r for r in meta if r.get("sensor_id") == sid]
            by_key: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
            for row in rows:
                by_key[ts_key(row)].append(row)
            multi = {k: v for k, v in by_key.items() if len(v) > 1}
            non_hour = sum(
                1
                for row in rows
                if str(row.get("minute", "00")) != "00" or str(row.get("second", "00")) != "00"
            )
            ch = sid.split("-")[-2] + "-" + sid.split("-")[-1]
            print(
                f"{label} {ch}: files={len(rows)} keys={len(by_key)} "
                f"multi_key_hours={len(multi)} non_hour_files={non_hour}"
            )
            if multi and label in {"C34-201", "C34-202", "C34-301", "C34-302"}:
                sample_key = next(iter(multi))
                sample_rows = multi[sample_key]
                print(f"  example collision key={sample_key}:")
                for row in sample_rows[:4]:
                    print(
                        f"    raw_time={raw_time(row)} file={Path(row['file_path']).name} "
                        f"len={row.get('actual_length')}"
                    )

    print("\n=== METADATA: in/out pairing simulation (dataset logic) ===")
    for in_id, out_id, label in PAIRS:
        in_lookup: dict[tuple[int, int, int], dict] = {}
        out_lookup: dict[tuple[int, int, int], dict] = {}
        for row in meta:
            if row.get("sensor_id") == in_id:
                in_lookup[ts_key(row)] = row
            if row.get("sensor_id") == out_id:
                out_lookup[ts_key(row)] = row
        common = sorted(set(in_lookup) & set(out_lookup))
        raw_mismatch = 0
        len_mismatch = 0
        for key in common:
            in_row = in_lookup[key]
            out_row = out_lookup[key]
            if raw_time(in_row) != raw_time(out_row):
                raw_mismatch += 1
            in_len = int(in_row.get("actual_length", 0) or 0)
            out_len = int(out_row.get("actual_length", 0) or 0)
            if in_len != out_len:
                len_mismatch += 1
        print(
            f"{label}: common_hours={len(common)} raw_time_mismatch={raw_mismatch} "
            f"length_mismatch={len_mismatch}"
        )

    print("\n=== INDEX CACHE: pairing integrity ===")
    for in_id, out_id, label in PAIRS:
        samples = [
            s
            for s in cache["samples"]
            if tuple(s.get("cable_pair") or ()) == (in_id, out_id)
        ]
        same_file = sum(
            1 for s in samples if s.get("inplane_file_path") == s.get("outplane_file_path")
        )
        bad_canonical = 0
        base_mismatch = 0
        token_mismatch = 0
        for sample in samples:
            in_fp = sample["inplane_file_path"]
            out_fp = sample["outplane_file_path"]
            in_match = SENSOR_RE.search(in_fp)
            out_match = SENSOR_RE.search(out_fp)
            if (
                not in_match
                or not out_match
                or in_match.group(2) != "01"
                or out_match.group(2) != "02"
                or in_match.group(1) != out_match.group(1)
            ):
                bad_canonical += 1
            if cable_base_from_path(in_fp) != cable_base_from_path(out_fp):
                base_mismatch += 1
            if filename_time_token(in_fp) != filename_time_token(out_fp):
                token_mismatch += 1
        print(
            f"{label}: samples={len(samples)} same_file={same_file} "
            f"bad_canonical={bad_canonical} cable_base_mismatch={base_mismatch} "
            f"filename_token_mismatch={token_mismatch}"
        )
        if token_mismatch and label in {"C34-201", "C34-202", "C34-301", "C34-302"}:
            shown = 0
            for sample in samples:
                in_fp = sample["inplane_file_path"]
                out_fp = sample["outplane_file_path"]
                if filename_time_token(in_fp) != filename_time_token(out_fp):
                    print(f"  token mismatch: {Path(in_fp).name} <-> {Path(out_fp).name}")
                    shown += 1
                    if shown >= 3:
                        break


if __name__ == "__main__":
    main()
