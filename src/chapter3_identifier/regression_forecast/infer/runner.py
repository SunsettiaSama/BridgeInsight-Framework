from __future__ import annotations

from typing import Any

import torch

from src.chapter3_identifier.regression_forecast.features.target_schema import risk_class_from_event, risk_score_from_event
from src.chapter3_identifier.regression_forecast.train.trainer import make_loader


def _metrics_by_class(values: torch.Tensor, class_names: list[str], metric_names: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c_idx, class_name in enumerate(class_names):
        out[class_name] = {
            metric_name: float(values[c_idx, m_idx].detach().cpu())
            for m_idx, metric_name in enumerate(metric_names)
        }
    return out


def _true_metrics_by_class(values: torch.Tensor, mask: torch.Tensor, class_names: list[str], metric_names: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c_idx, class_name in enumerate(class_names):
        class_values: dict[str, float] = {}
        for m_idx, metric_name in enumerate(metric_names):
            if float(mask[c_idx, m_idx]) > 0.5:
                class_values[metric_name] = float(values[c_idx, m_idx].detach().cpu())
        if class_values:
            out[class_name] = class_values
    return out


class ForecastRunner:
    def __init__(self, model: torch.nn.Module, schema: dict[str, Any], device: torch.device) -> None:
        self.model = model.to(device)
        self.schema = schema
        self.device = device
        self.class_names = [str(x) for x in schema.get("class_names", [])]
        self.horizons = [int(x) for x in schema.get("horizons_hours", [])]
        self.metric_names = [str(x) for x in schema.get("metric_names", [])]

    def run(self, dataset, batch_size: int = 64) -> list[dict[str, Any]]:
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        records: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                output = self.model(batch["x_hist"].to(self.device))
                event_proba = torch.sigmoid(output["event_logits"]).detach().cpu()
                metric_pred = output["metric_pred"].detach().cpu()
                y_event = batch["y_event"].detach().cpu()
                event_mask = batch["event_mask"].detach().cpu()
                y_metric = batch["y_metric"].detach().cpu()
                metric_mask = batch["metric_mask"].detach().cpu()
                for b_idx, meta in enumerate(batch["meta"]):
                    horizons_payload = []
                    for h_idx, horizon in enumerate(self.horizons):
                        probs = [float(x) for x in event_proba[b_idx, h_idx]]
                        dominant_idx = max(range(len(probs)), key=lambda idx: probs[idx])
                        risk_class = risk_class_from_event(probs, self.class_names)
                        true_payload = None
                        if float(event_mask[b_idx, h_idx]) > 0.5:
                            true_payload = {
                                "event_target": {
                                    self.class_names[c_idx]: float(y_event[b_idx, h_idx, c_idx])
                                    for c_idx in range(len(self.class_names))
                                },
                                "metrics_by_class": _true_metrics_by_class(
                                    y_metric[b_idx, h_idx],
                                    metric_mask[b_idx, h_idx],
                                    self.class_names,
                                    self.metric_names,
                                ),
                            }
                        horizons_payload.append(
                            {
                                "horizon_hours": horizon,
                                "event_proba": {self.class_names[c_idx]: probs[c_idx] for c_idx in range(len(self.class_names))},
                                "dominant_class": self.class_names[dominant_idx],
                                "risk_class": risk_class,
                                "risk_score": risk_score_from_event(probs),
                                "metrics_by_class": _metrics_by_class(metric_pred[b_idx, h_idx], self.class_names, self.metric_names),
                                "y_true": true_payload,
                                "mask": int(float(event_mask[b_idx, h_idx]) > 0.5),
                                "label_source": str(meta.get("label_source", "model")),
                            }
                        )
                    records.append(
                        {
                            "sample_idx": int(meta.get("sample_idx", len(records))),
                            "sample_key": meta.get("sample_key"),
                            "cable_pair": meta.get("cable_pair", []),
                            "timestamp": meta.get("timestamp", {}),
                            "input_summary": {
                                "features": meta.get("input_features", {}),
                                "current_class": meta.get("current_class"),
                            },
                            "horizons": horizons_payload,
                        }
                    )
        return records

