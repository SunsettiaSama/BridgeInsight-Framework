"""风雨振图共用样本池。

- use_merged=True：只读 augment 2024-09 训练+验证 RWIV，写成 chapter4 新副本后使用
- use_merged=False：仅用 chapter4 DL 识别结果中的 RWIV
不修改 results/augment 下任何原始文件。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import load_dl_result

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent

RWIV_CLASS_ID = 2
# 默认是否使用 2024-09 train+val 合并副本；各图可用 CLI 覆盖
USE_MERGED_DATASET = True

AUGMENT_MERGED_PATH = project_root / "results" / "augment" / "annotations" / "merged_for_training.json"
AUGMENT_SPLIT_PATH = project_root / "results" / "augment" / "split_indices.json"
RWIV_SAMPLE_COPY_PATH = (
    project_root
    / "results"
    / "chapter4_characteristics"
    / "figure_snapshots"
    / "rwiv_202409_train_val_pairs.json"
)


def _sibling_path(file_path: str) -> Path | None:
    path = Path(file_path)
    name = path.name
    if "-01_" in name:
        return path.with_name(name.replace("-01_", "-02_", 1))
    if "-02_" in name:
        return path.with_name(name.replace("-02_", "-01_", 1))
    return None


def _parse_timestamp(time_str: str, metadata: dict) -> list:
    month = metadata.get("month")
    day = metadata.get("day")
    hour = metadata.get("hour")
    if month is not None and day is not None and hour is not None:
        return [int(month), int(day), int(hour)]
    if isinstance(time_str, str) and "/" in time_str:
        date_part, _, clock_part = time_str.partition(" ")
        m_s, _, d_s = date_part.partition("/")
        h_s = clock_part.split(":")[0] if clock_part else "0"
        return [int(m_s), int(d_s), int(h_s)]
    return []


def _is_excluded_sensor(sensor_id: str) -> bool:
    return sensor_id in data_config.EXCLUDED_SENSOR_IDS


def _is_excluded_meta(meta: dict) -> bool:
    return (
        meta.get("inplane_sensor_id") in data_config.EXCLUDED_SENSOR_IDS
        or meta.get("outplane_sensor_id") in data_config.EXCLUDED_SENSOR_IDS
    )


def _load_train_val_key_sets() -> tuple[set, set]:
    if not AUGMENT_SPLIT_PATH.exists():
        raise FileNotFoundError(f"找不到 split：{AUGMENT_SPLIT_PATH}")
    with open(AUGMENT_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_keys = {annotation_key(k[0], k[1]) for k in split.get("train_keys", [])}
    val_keys = {annotation_key(k[0], k[1]) for k in split.get("val_keys", [])}
    return train_keys, val_keys


def _entry_to_pair(entry: dict) -> dict | None:
    file_path = entry.get("file_path")
    window_idx = entry.get("window_index")
    sensor_id = entry.get("sensor_id", "")
    if file_path is None or window_idx is None or not sensor_id:
        return None
    if _is_excluded_sensor(sensor_id):
        return None

    sibling = _sibling_path(file_path)
    if sibling is None or not sibling.exists():
        return None

    if sensor_id.endswith("-01"):
        in_id, out_id = sensor_id, sensor_id[:-3] + "-02"
        in_path, out_path = str(Path(file_path)), str(sibling)
    elif sensor_id.endswith("-02"):
        in_id, out_id = sensor_id[:-3] + "-01", sensor_id
        in_path, out_path = str(sibling), str(Path(file_path))
    else:
        return None

    if _is_excluded_sensor(in_id) or _is_excluded_sensor(out_id):
        return None

    meta = entry.get("metadata") or {}
    return {
        "idx": f"aug202409_{Path(in_path).stem}_{int(window_idx)}",
        "window_idx": int(window_idx),
        "inplane_sensor_id": in_id,
        "outplane_sensor_id": out_id,
        "inplane_file_path": in_path,
        "outplane_file_path": out_path,
        "timestamp": _parse_timestamp(entry.get("time", ""), meta),
        "source": "augment_202409",
        "split": entry.get("_split", "unknown"),
        "time_str": entry.get("time", ""),
    }


def load_dl_rwiv_samples() -> list[dict]:
    """仅从 chapter4 DL 识别结果筛选 RWIV（不使用 2024-09 副本）。"""
    result = load_dl_result()
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    samples = []
    for idx, pred_label in predictions.items():
        if pred_label != RWIV_CLASS_ID:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None or _is_excluded_meta(meta):
            continue
        inplane_path = meta.get("inplane_file_path")
        outplane_path = meta.get("outplane_file_path")
        if not inplane_path or not outplane_path:
            continue
        samples.append(
            {
                "idx": idx,
                "window_idx": meta["window_idx"],
                "inplane_sensor_id": meta.get("inplane_sensor_id", ""),
                "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
                "inplane_file_path": inplane_path,
                "outplane_file_path": outplane_path,
                "timestamp": meta.get("timestamp", []),
                "source": "dl_chapter4",
                "split": "dl",
                "time_str": "",
            }
        )
    print(f"  DL 识别 RWIV 配对样本：{len(samples)} 个")
    return samples


def build_rwiv_202409_copy(force_refresh: bool = False) -> list[dict]:
    """只读 augment，写出/读取 chapter4 下的 RWIV 样本副本。"""
    if RWIV_SAMPLE_COPY_PATH.exists() and not force_refresh:
        with open(RWIV_SAMPLE_COPY_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples = payload.get("samples", [])
        print(f"  读取 RWIV 副本：{RWIV_SAMPLE_COPY_PATH}  n={len(samples)}")
        return samples

    if not AUGMENT_MERGED_PATH.exists():
        raise FileNotFoundError(f"找不到 augment 标注：{AUGMENT_MERGED_PATH}")

    print(f"  只读 augment 标注：{AUGMENT_MERGED_PATH}")
    with open(AUGMENT_MERGED_PATH, "r", encoding="utf-8") as f:
        merged = json.load(f)

    train_keys, val_keys = _load_train_val_key_sets()
    print(f"  split：train={len(train_keys)}  val={len(val_keys)}")

    rwiv_entries: list[dict] = []
    n_train = 0
    n_val = 0
    for entry in merged:
        if int(entry.get("annotation", -1)) != RWIV_CLASS_ID:
            continue
        fp = entry.get("file_path")
        wi = entry.get("window_index", 0)
        if fp is None:
            continue
        key = annotation_key(fp, wi)
        if key in train_keys:
            item = dict(entry)
            item["_split"] = "train"
            rwiv_entries.append(item)
            n_train += 1
        elif key in val_keys:
            item = dict(entry)
            item["_split"] = "val"
            rwiv_entries.append(item)
            n_val += 1

    print(f"  2024-09 RWIV 单通道：train={n_train}  val={n_val}  合计={len(rwiv_entries)}")

    pairs: dict[tuple[str, str, int], dict] = {}
    for entry in rwiv_entries:
        pair = _entry_to_pair(entry)
        if pair is None:
            continue
        dedupe_key = (
            os.path.normcase(pair["inplane_file_path"]),
            os.path.normcase(pair["outplane_file_path"]),
            pair["window_idx"],
        )
        prev = pairs.get(dedupe_key)
        if prev is None:
            pairs[dedupe_key] = pair
            continue
        if prev.get("split") != pair.get("split"):
            prev["split"] = "train+val"

    samples = list(pairs.values())
    samples.sort(
        key=lambda s: (
            s.get("time_str", ""),
            s["inplane_sensor_id"],
            s["window_idx"],
        )
    )

    payload = {
        "version": "rwiv_202409_train_val_pairs_v1",
        "source_merged": str(AUGMENT_MERGED_PATH),
        "source_split": str(AUGMENT_SPLIT_PATH),
        "note": "只读自 augment，不修改原始标注；供 fig4_25/26/27 使用",
        "n_single_channel_train": n_train,
        "n_single_channel_val": n_val,
        "n_pairs": len(samples),
        "samples": samples,
    }
    RWIV_SAMPLE_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RWIV_SAMPLE_COPY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出 RWIV 副本：{RWIV_SAMPLE_COPY_PATH}  n_pairs={len(samples)}")
    return samples


def load_rwiv_samples_for_figures(
    use_merged: bool | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """按开关加载风雨振样本。

    use_merged=True  → 2024-09 train+val 副本
    use_merged=False → 仅 DL 识别结果
    """
    merged = USE_MERGED_DATASET if use_merged is None else bool(use_merged)
    if merged:
        print("  数据源开关：USE_MERGED_DATASET=True（2024-09 train+val 副本）")
        return build_rwiv_202409_copy(force_refresh=force_refresh)
    print("  数据源开关：USE_MERGED_DATASET=False（仅 DL 识别）")
    return load_dl_rwiv_samples()


def add_dataset_switch_args(parser) -> None:
    """为 argparse 增加 --use-merged / --no-merged / --refresh-sample-copy。"""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-merged",
        dest="use_merged",
        action="store_true",
        help="使用 2024-09 train+val 合并副本",
    )
    group.add_argument(
        "--no-merged",
        dest="use_merged",
        action="store_false",
        help="仅使用 DL 识别 RWIV，不使用 2024-09 副本",
    )
    parser.set_defaults(use_merged=None)
    parser.add_argument(
        "--refresh-sample-copy",
        action="store_true",
        help="重新从 augment 只读生成样本副本（不修改原始标注）",
    )


def resolve_use_merged(cli_value: bool | None) -> bool:
    return USE_MERGED_DATASET if cli_value is None else bool(cli_value)
