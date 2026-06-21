from __future__ import annotations

import torch
from torch import nn


class MultiTaskForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        class_count: int,
        metric_count: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.horizon_count = int(horizon_count)
        self.class_count = int(class_count)
        self.metric_count = int(metric_count)
        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.event_head = nn.Linear(int(hidden_dim), self.horizon_count * self.class_count)
        self.metric_head = nn.Linear(int(hidden_dim), self.horizon_count * self.class_count * self.metric_count)

    def forward(self, x_hist: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.encoder(x_hist)
        encoded = hidden[-1]
        z = self.head(encoded)
        event_logits = self.event_head(z).view(-1, self.horizon_count, self.class_count)
        metric_pred = self.metric_head(z).view(-1, self.horizon_count, self.class_count, self.metric_count)
        return {"event_logits": event_logits, "metric_pred": metric_pred}

