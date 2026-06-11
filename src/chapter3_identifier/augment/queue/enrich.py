from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key, build_gold_index
from src.chapter3_identifier.augment.annotation.store import load_cumulative_manual_edits
from src.chapter3_identifier.augment.figures.render.titles import format_sample_title
from src.chapter3_identifier.augment.labels import label_name


def build_manual_index(manual_entries: List[dict]) -> Dict[Tuple[str, int], dict]:
    index: Dict[Tuple[str, int], dict] = {}
    for entry in manual_entries:
        fp = entry.get("file_path")
        if fp is None:
            continue
        index[annotation_key(fp, entry.get("window_index", 0))] = entry
    return index


def resolve_card_kind(
    manual_ann: Optional[int],
    gold_label: Optional[int],
    prediction: int,
    is_gold: bool,
) -> tuple[str, str, bool]:
    if manual_ann is not None:
        return "manual", "已人工标注", False
    if is_gold and gold_label is not None:
        if prediction == gold_label:
            return "gold_agree", "金标准·模型一致", True
        return "gold_disagree", "金标准·模型分歧", False
    return "model", "模型预测", False


def enrich_record(
    record: dict,
    gold_index: dict,
    manual_index: dict,
    cfg: dict,
) -> dict:
    row = dict(record)
    in_fp = row.get("inplane_file_path", "")
    wi = int(row.get("window_index", 0))
    key = annotation_key(in_fp, wi) if in_fp else None

    gold_entry = gold_index.get(key) if key else None
    manual_entry = manual_index.get(key) if key else None
    manual_inplane_ann = (
        int(manual_entry.get("inplane_annotation", manual_entry.get("annotation", 0)))
        if manual_entry
        else None
    )
    manual_outplane_ann = (
        int(manual_entry.get("outplane_annotation", manual_entry.get("annotation", 0)))
        if manual_entry
        else None
    )
    gold_label = int(gold_entry["annotation"]) if gold_entry else None
    prediction = int(row.get("prediction", 0))
    inplane_prediction = int(row.get("inplane_prediction", prediction))
    outplane_prediction = int(row.get("outplane_prediction", prediction))

    if manual_entry is not None:
        suggested_inplane = manual_inplane_ann
        suggested_outplane = manual_outplane_ann
        suggested_source = "manual"
        label_origin = "manual"
    elif gold_entry is not None:
        suggested_inplane = gold_label
        suggested_outplane = gold_label
        suggested_source = "gold"
        label_origin = "gold"
    else:
        suggested_inplane = inplane_prediction
        suggested_outplane = outplane_prediction
        suggested_source = "prediction"
        label_origin = "model"

    row["is_gold"] = gold_entry is not None
    row["gold_label"] = gold_label
    row["gold_label_name"] = label_name(gold_label, cfg) if gold_label is not None else None
    row["manual_inplane_annotation"] = manual_inplane_ann
    row["manual_inplane_annotation_name"] = (
        label_name(manual_inplane_ann, cfg) if manual_inplane_ann is not None else None
    )
    row["manual_outplane_annotation"] = manual_outplane_ann
    row["manual_outplane_annotation_name"] = (
        label_name(manual_outplane_ann, cfg) if manual_outplane_ann is not None else None
    )
    row["manual_annotation"] = manual_inplane_ann
    row["manual_annotation_name"] = (
        label_name(manual_inplane_ann, cfg) if manual_inplane_ann is not None else None
    )
    row["suggested_inplane_label"] = suggested_inplane
    row["suggested_inplane_label_name"] = label_name(suggested_inplane, cfg)
    row["suggested_outplane_label"] = suggested_outplane
    row["suggested_outplane_label_name"] = label_name(suggested_outplane, cfg)
    row["suggested_label"] = suggested_inplane
    row["suggested_label_name"] = label_name(suggested_inplane, cfg)
    row["suggested_source"] = suggested_source
    row["label_origin"] = label_origin
    row["label_origin_name"] = {
        "gold": "金标准",
        "manual": "人工标注",
        "model": "模型预测",
    }.get(label_origin, label_origin)
    card_kind, card_kind_name, prediction_matches_gold = resolve_card_kind(
        manual_inplane_ann if manual_entry else None,
        gold_label,
        prediction,
        gold_entry is not None,
    )
    row["card_kind"] = card_kind
    row["card_kind_name"] = card_kind_name
    row["prediction_matches_gold"] = prediction_matches_gold
    row["prediction_name"] = label_name(prediction, cfg)
    row["inplane_time_label"] = format_sample_title(
        "面内",
        row.get("inplane_sensor_id", ""),
        row.get("inplane_file_path"),
        wi,
        row.get("timestamp"),
    )
    row["outplane_time_label"] = format_sample_title(
        "面外",
        row.get("outplane_sensor_id", ""),
        row.get("outplane_file_path"),
        wi,
        row.get("timestamp"),
    )
    return row


class AnnotationLookupCache:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._gold_index: Optional[dict] = None
        self._manual_by_round: dict[int, dict] = {}
        self._manual_epoch: dict[int, int] = {}
        self._manual_epoch_counter = 0

    def gold_index(self, gold_entries: List[dict]) -> dict:
        if self._gold_index is None:
            self._gold_index = build_gold_index(gold_entries)
        return self._gold_index

    def manual_index(self, round_idx: int) -> dict:
        if round_idx not in self._manual_by_round:
            entries = load_cumulative_manual_edits(self._cfg, round_idx)
            self._manual_by_round[round_idx] = build_manual_index(entries)
            self._manual_epoch[round_idx] = self._manual_epoch_counter
        return self._manual_by_round[round_idx]

    def manual_epoch(self, round_idx: int) -> int:
        return self._manual_epoch.get(round_idx, 0)

    def invalidate_manual(self) -> None:
        self._manual_by_round.clear()
        self._manual_epoch_counter += 1
        for round_idx in list(self._manual_epoch.keys()):
            self._manual_epoch[round_idx] = self._manual_epoch_counter
