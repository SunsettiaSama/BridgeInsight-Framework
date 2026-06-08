from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitFusion(nn.Module):
    def __init__(self, fusion_type: str = "weighted_sum", num_branches: int = 2):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "weighted_sum":
            self.weights = nn.Parameter(torch.zeros(num_branches))
        elif fusion_type == "concat_fc":
            raise ValueError("concat_fc 暂未实现")
        else:
            raise ValueError(f"未知 fusion_type={fusion_type}")

    def forward(self, *logits: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "weighted_sum":
            w = F.softmax(self.weights, dim=0)
            out = logits[0] * w[0]
            for i in range(1, len(logits)):
                out = out + logits[i] * w[i]
            return out
        raise ValueError(f"未知 fusion_type={self.fusion_type}")
