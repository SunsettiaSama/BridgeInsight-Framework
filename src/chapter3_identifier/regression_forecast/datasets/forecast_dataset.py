from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from src.chapter3_identifier.regression_forecast.datasets.split import chronological_split_indices, leakage_report
from src.chapter3_identifier.regression_forecast.settings import (
    get_round_feature_cache_path,
    get_round_schema_path,
    get_round_split_path,
    load_config,
    read_json,
    read_jsonl,
    write_json,
)


def _clean(value: Any) -> float:
    f = float(value)
    if math.isfinite(f):
        return f
    return 0.0


class ForecastDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], schema: dict[str, Any], history_hours: int) -> None:
        self.records = records
        self.schema = schema
        self.history_hours = max(1, int(history_hours))
        self.feature_names = [str(x) for x in schema.get("input_feature_names", [])]
        self.class_names = [str(x) for x in schema.get("class_names", [])]
        self.horizons = [int(x) for x in schema.get("horizons_hours", [])]
        self.metric_names = [str(x) for x in schema.get("metric_names", [])]
        self._history_lookup = self._build_history_lookup()

    def _build_history_lookup(self) -> dict[int, list[dict[str, Any]]]:
        by_cable: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for idx, row in enumerate(self.records):
            by_cable[str(row.get("cable_key", ""))].append((idx, row))

        lookup: dict[int, list[dict[str, Any]]] = {}
        for _, rows in by_cable.items():
            ordered = sorted(rows, key=lambda item: int(item[1].get("hour_index", 0)))
            for pos, (global_idx, _) in enumerate(ordered):
                start = max(0, pos - self.history_hours + 1)
                hist = [r for _, r in ordered[start : pos + 1]]
                if len(hist) < self.history_hours:
                    hist = [hist[0]] * (self.history_hours - len(hist)) + hist
                lookup[global_idx] = hist
        return lookup

    def __len__(self) -> int:
        return len(self.records)

    def _history_tensor(self, idx: int) -> torch.Tensor:
        hist = self._history_lookup[idx]
        rows = []
        for row in hist:
            features = row.get("input_features", {})
            rows.append([_clean(features.get(name, 0.0)) for name in self.feature_names])
        return torch.tensor(rows, dtype=torch.float32)

    def _target_tensors(self, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_count = len(self.horizons)
        c_count = len(self.class_names)
        m_count = len(self.metric_names)
        event = torch.zeros((h_count, c_count), dtype=torch.float32)
        event_mask = torch.zeros((h_count,), dtype=torch.float32)
        metrics = torch.zeros((h_count, c_count, m_count), dtype=torch.float32)
        metric_mask = torch.zeros((h_count, c_count, m_count), dtype=torch.float32)
        targets = row.get("future_targets_by_horizon", {})
        for h_idx, horizon in enumerate(self.horizons):
            payload = targets.get(str(horizon), {})
            values = payload.get("event_target", [0.0] * c_count)
            event[h_idx] = torch.tensor([float(x) for x in values[:c_count]], dtype=torch.float32)
            event_mask[h_idx] = float(payload.get("mask", 0))
            by_class = payload.get("metrics_by_class", {})
            for c_idx, class_name in enumerate(self.class_names):
                class_metrics = by_class.get(class_name, {})
                for m_idx, metric_name in enumerate(self.metric_names):
                    value = class_metrics.get(metric_name)
                    if value is not None and math.isfinite(float(value)):
                        metrics[h_idx, c_idx, m_idx] = float(value)
                        metric_mask[h_idx, c_idx, m_idx] = 1.0
        return event, event_mask, metrics, metric_mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        event, event_mask, metrics, metric_mask = self._target_tensors(row)
        return {
            "x_hist": self._history_tensor(idx),
            "y_event": event,
            "event_mask": event_mask,
            "y_metric": metrics,
            "metric_mask": metric_mask,
            "sample_idx": int(row.get("sample_idx", idx)),
            "record_index": int(idx),
            "label_source": str(row.get("label_source", "model")),
            "meta": row,
        }


def load_forecast_dataset(round_idx: int = 1, config_path: str | None = None) -> tuple[ForecastDataset, dict[str, list[int]], dict[str, Any]]:
    cfg = load_config(config_path)
    cache_path = get_round_feature_cache_path(cfg, round_idx)
    if not cache_path.exists():
        from src.chapter3_identifier.regression_forecast.features.cache import build_feature_cache

        build_feature_cache(round_idx=round_idx, config_path=config_path)
    records = read_jsonl(cache_path)
    schema = read_json(get_round_schema_path(cfg, round_idx))
    dataset = ForecastDataset(records, schema, history_hours=int(cfg.get("history_hours", 6)))
    splits = chronological_split_indices(records, cfg)
    report = leakage_report(records, splits, cfg)
    split_payload = {"splits": splits, "leakage": report}
    write_json(get_round_split_path(cfg, round_idx), split_payload)
    if report["train_val_leak"] or report["val_test_leak"]:
        raise ValueError(f"时间切分存在泄漏：{report}")
    return dataset, splits, schema


def subset(dataset: ForecastDataset, indices: list[int]) -> Subset:
    return Subset(dataset, indices)

