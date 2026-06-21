from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_hist": torch.stack([item["x_hist"] for item in items]),
        "y_event": torch.stack([item["y_event"] for item in items]),
        "event_mask": torch.stack([item["event_mask"] for item in items]),
        "y_metric": torch.stack([item["y_metric"] for item in items]),
        "metric_mask": torch.stack([item["metric_mask"] for item in items]),
        "sample_idx": [int(item["sample_idx"]) for item in items],
        "record_index": [int(item["record_index"]) for item in items],
        "label_source": [str(item["label_source"]) for item in items],
        "meta": [item["meta"] for item in items],
    }


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), collate_fn=collate_batch)


class ForecastTrainer:
    def __init__(self, model: torch.nn.Module, cfg: dict[str, Any], device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        train_cfg = cfg.get("training", {})
        self.event_loss_weight = float(train_cfg.get("event_loss_weight", 1.0))
        self.metric_loss_weight = float(train_cfg.get("metric_loss_weight", 0.2))
        self.source_weights = {str(k): float(v) for k, v in train_cfg.get("label_source_weights", {}).items()}

    def _source_weight_tensor(self, sources: list[str], horizon_count: int, class_count: int) -> torch.Tensor:
        weights = [self.source_weights.get(source, 1.0) for source in sources]
        return torch.tensor(weights, dtype=torch.float32, device=self.device).view(-1, 1, 1).expand(-1, horizon_count, class_count)

    def _loss(self, batch: dict[str, Any], output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        y_event = batch["y_event"].to(self.device)
        event_mask = batch["event_mask"].to(self.device)
        y_metric = batch["y_metric"].to(self.device)
        metric_mask = batch["metric_mask"].to(self.device)
        event_logits = output["event_logits"]
        metric_pred = output["metric_pred"]
        source_weight = self._source_weight_tensor(
            batch["label_source"],
            event_logits.shape[1],
            event_logits.shape[2],
        )
        event_mask_full = event_mask.unsqueeze(-1).expand_as(event_logits)
        event_loss_raw = F.binary_cross_entropy_with_logits(event_logits, y_event, reduction="none")
        event_denom = torch.clamp((event_mask_full * source_weight).sum(), min=1.0)
        event_loss = (event_loss_raw * event_mask_full * source_weight).sum() / event_denom

        metric_loss_raw = F.mse_loss(metric_pred, y_metric, reduction="none")
        metric_denom = torch.clamp(metric_mask.sum(), min=1.0)
        metric_loss = (metric_loss_raw * metric_mask).sum() / metric_denom

        total = self.event_loss_weight * event_loss + self.metric_loss_weight * metric_loss
        return total, {"event_loss": float(event_loss.detach().cpu()), "metric_loss": float(metric_loss.detach().cpu())}

    def _metrics(self, batch: dict[str, Any], output: dict[str, torch.Tensor]) -> dict[str, float]:
        y_event = batch["y_event"].to(self.device)
        event_mask = batch["event_mask"].to(self.device).unsqueeze(-1).expand_as(y_event)
        event_pred = (torch.sigmoid(output["event_logits"]) >= 0.5).float()
        event_total = torch.clamp(event_mask.sum(), min=1.0)
        event_acc = ((event_pred == y_event).float() * event_mask).sum() / event_total

        y_metric = batch["y_metric"].to(self.device)
        metric_mask = batch["metric_mask"].to(self.device)
        abs_err = torch.abs(output["metric_pred"] - y_metric) * metric_mask
        sq_err = ((output["metric_pred"] - y_metric) ** 2) * metric_mask
        metric_total = torch.clamp(metric_mask.sum(), min=1.0)
        mae = abs_err.sum() / metric_total
        rmse = torch.sqrt(sq_err.sum() / metric_total)
        return {
            "event_accuracy": float(event_acc.detach().cpu()),
            "metric_mae": float(mae.detach().cpu()),
            "metric_rmse": float(rmse.detach().cpu()),
            "metric_mask_coverage": float((metric_mask.sum() / max(1, metric_mask.numel())).detach().cpu()),
        }

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            x_hist = batch["x_hist"].to(self.device)
            output = self.model(x_hist)
            loss, loss_parts = self._loss(batch, output)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metrics = self._metrics(batch, output)
            row = {"loss": float(loss.detach().cpu()), **loss_parts, **metrics}
            for key, value in row.items():
                if math.isfinite(float(value)):
                    totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1
        denom = max(1, n_batches)
        return {key: value / denom for key, value in totals.items()}

