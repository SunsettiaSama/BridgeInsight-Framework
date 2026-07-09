from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from src.chapter3_identifier.augment.annotation.sample_key import (
    cable_base_from_sensor_id,
    pair_key_from_entry,
    pair_key_to_list,
    sensor_id_from_path,
)
from src.chapter3_identifier.augment.annotation.split import load_saved_split_key_sets
from src.chapter3_identifier.augment_eval_compare._bootstrap import ensure_paths
from src.data_processer.datasets.StayCable_Vib2023.StayCableVib2023Dataset import _SampleRecord

ensure_paths()

EvalSplit = Literal["val", "all_pairs"]


@dataclass(frozen=True)
class EvalPairEntry:
    pair_key: tuple
    inplane_file_path: str
    outplane_file_path: str
    window_index: int
    inplane_annotation: int
    outplane_annotation: int
    inplane_sensor_id: str
    outplane_sensor_id: str
    timestamp: tuple[int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_key": pair_key_to_list(self.pair_key),
            "inplane_file_path": self.inplane_file_path,
            "outplane_file_path": self.outplane_file_path,
            "window_index": self.window_index,
            "inplane_annotation": self.inplane_annotation,
            "outplane_annotation": self.outplane_annotation,
            "inplane_sensor_id": self.inplane_sensor_id,
            "outplane_sensor_id": self.outplane_sensor_id,
            "timestamp": list(self.timestamp),
        }


class EvalPairDataset:
    def __init__(self, samples: list[_SampleRecord], config: SimpleNamespace, entries: list[EvalPairEntry]):
        self._samples = samples
        self.config = config
        self.entries = entries


def _has_bidirectional_labels(entry: dict) -> bool:
    return entry.get("inplane_annotation") is not None and entry.get("outplane_annotation") is not None


def _normalize_entry(raw: dict) -> EvalPairEntry:
    if not _has_bidirectional_labels(raw):
        raise ValueError("entry 缺少双向标注")
    pair_key = pair_key_from_entry(raw)
    in_fp = str(raw.get("inplane_file_path") or raw.get("file_path"))
    out_fp = str(raw.get("outplane_file_path"))
    wi = int(raw.get("window_index", pair_key[5]))
    in_sensor = str(raw.get("inplane_sensor_id") or sensor_id_from_path(in_fp))
    out_sensor = str(raw.get("outplane_sensor_id") or sensor_id_from_path(out_fp))
    ts = raw.get("timestamp")
    if ts is None:
        timestamp = (int(pair_key[2]), int(pair_key[3]), int(pair_key[4]))
    else:
        timestamp = (int(ts[0]), int(ts[1]), int(ts[2]))
    return EvalPairEntry(
        pair_key=pair_key,
        inplane_file_path=in_fp,
        outplane_file_path=out_fp,
        window_index=wi,
        inplane_annotation=int(raw["inplane_annotation"]),
        outplane_annotation=int(raw["outplane_annotation"]),
        inplane_sensor_id=in_sensor,
        outplane_sensor_id=out_sensor,
        timestamp=timestamp,
    )


def load_pair_key_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"merged_training_pair_key.json 必须是列表：{path}")
    return payload


def filter_eval_entries(
    rows: list[dict],
    *,
    eval_split: EvalSplit,
    split_indices_path: str,
) -> list[EvalPairEntry]:
    labeled = [_normalize_entry(row) for row in rows if _has_bidirectional_labels(row)]
    if eval_split == "all_pairs":
        return labeled

    _, val_keys = load_saved_split_key_sets(split_indices_path)
    if not val_keys:
        raise ValueError(f"val_pair_keys 为空：{split_indices_path}")
    val_set = set(val_keys)
    return [entry for entry in labeled if entry.pair_key in val_set]


def _build_sample_record(entry: EvalPairEntry) -> _SampleRecord:
    month, day, hour = entry.timestamp
    in_meta = {
        "file_path": entry.inplane_file_path,
        "sensor_id": entry.inplane_sensor_id,
        "month": month,
        "day": day,
        "hour": hour,
    }
    out_meta = {
        "file_path": entry.outplane_file_path,
        "sensor_id": entry.outplane_sensor_id,
        "month": month,
        "day": day,
        "hour": hour,
    }
    in_base = cable_base_from_sensor_id(entry.inplane_sensor_id)
    out_base = cable_base_from_sensor_id(entry.outplane_sensor_id)
    cable_pair = (in_base, out_base)
    return _SampleRecord(
        inplane_meta=in_meta,
        outplane_meta=out_meta,
        wind_meta=None,
        window_idx=entry.window_index,
        cable_pair=cable_pair,
        timestamp_key=entry.timestamp,
        cable_pair_idx=0,
    )


def build_runtime_config(augment_cfg: dict) -> SimpleNamespace:
    return SimpleNamespace(
        window_size=int(augment_cfg.get("window_size", 3000)),
        enable_denoise=bool(augment_cfg.get("enable_denoise", False)),
        fs=float(augment_cfg.get("fs", 50.0)),
        nfft=int(augment_cfg.get("nfft", 2048)),
        freq_max_hz=float(augment_cfg.get("freq_max_hz", 25.0)),
    )


def build_eval_dataset(
    pair_key_path: Path,
    augment_cfg: dict,
    *,
    eval_split: EvalSplit = "val",
) -> EvalPairDataset:
    rows = load_pair_key_rows(pair_key_path)
    entries = filter_eval_entries(
        rows,
        eval_split=eval_split,
        split_indices_path=str(augment_cfg["split_indices_path"]),
    )
    if not entries:
        raise ValueError(f"评估集为空：split={eval_split}, path={pair_key_path}")
    samples = [_build_sample_record(entry) for entry in entries]
    config = build_runtime_config(augment_cfg)
    return EvalPairDataset(samples=samples, config=config, entries=entries)
