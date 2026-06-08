from __future__ import annotations

import torch
import torch.nn as nn

from src.chapter1_identifier.augment._bootstrap import ensure_paths
from src.chapter1_identifier.augment.settings import BranchConfig, DualStreamResCNNConfig
from src.chapter1_identifier.augment.models.fusion import LogitFusion
from src.training.deep_learning.models.res_cnn import ResCNN

ensure_paths()


class DualStreamResCNN(nn.Module):
    def __init__(self, config: DualStreamResCNNConfig):
        super().__init__()
        self.cfg = config
        self.time_branch = ResCNN(config.time_branch)
        self.spec_branch = ResCNN(config.spec_branch)
        self.fusion = LogitFusion(config.fusion_type, num_branches=2)

    def forward(self, time_x: torch.Tensor, spec_x: torch.Tensor) -> torch.Tensor:
        logits_time = self.time_branch(time_x)
        logits_spec = self.spec_branch(spec_x)
        return self.fusion(logits_time, logits_spec)

    def forward_branches(
        self, time_x: torch.Tensor, spec_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_time = self.time_branch(time_x)
        logits_spec = self.spec_branch(spec_x)
        logits_final = self.fusion(logits_time, logits_spec)
        return logits_time, logits_spec, logits_final

    @classmethod
    def build_default(cls, psd_bins: int, num_classes: int = 4) -> "DualStreamResCNN":
        cfg = DualStreamResCNNConfig.from_dict(
            {
                "num_classes": num_classes,
                "time_branch": {"input_size": 3000},
                "spec_branch": {"input_size": psd_bins},
            }
        )
        return cls(cfg)
