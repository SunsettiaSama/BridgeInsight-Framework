from __future__ import annotations

import torch


class PersistenceBaseline:
    """Use the current event state and metrics as all future horizons."""

    def __init__(self, class_count: int, metric_count: int) -> None:
        self.class_count = int(class_count)
        self.metric_count = int(metric_count)

    def predict(self, x_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x_hist.shape[0]
        event = torch.zeros((batch, self.class_count), dtype=torch.float32, device=x_hist.device)
        metrics = torch.zeros((batch, self.class_count, self.metric_count), dtype=torch.float32, device=x_hist.device)
        return event, metrics

