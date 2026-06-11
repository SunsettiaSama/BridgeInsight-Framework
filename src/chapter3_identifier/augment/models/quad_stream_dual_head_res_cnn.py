from __future__ import annotations

import torch
import torch.nn as nn

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.settings import BranchConfig
from src.training.deep_learning.models.res_cnn import ResCNN

ensure_paths()


class _ResCnnFeatureEncoder(nn.Module):
    def __init__(self, branch_cfg: BranchConfig):
        super().__init__()
        self.backbone = ResCNN(branch_cfg)
        linear_layers = [m for m in self.backbone.fc if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise ValueError("ResCNN fc 至少需要包含一个 Linear 层")
        final_linear = linear_layers[-1]
        self.feature_dim = int(final_linear.in_features)
        fc_layers = list(self.backbone.fc.children())
        self.feature_head = (
            nn.Sequential(*fc_layers[:-1]) if len(fc_layers) > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"输入必须是 3 维 (B, L, C) 或 (B, C, L)，当前：{x.shape}")
        if x.shape[1] != self.backbone.cfg.in_channels and x.shape[2] == self.backbone.cfg.in_channels:
            x = x.transpose(1, 2)
        x = self.backbone.stem(x)
        x = self.backbone.stages(x)
        x = self.backbone.gap(x).squeeze(-1)
        return self.feature_head(x)


class QuadStreamDualHeadResCNN(nn.Module):
    def __init__(
        self,
        time_branch_cfg: BranchConfig,
        spec_branch_cfg: BranchConfig,
        num_classes: int = 4,
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.fusion_dropout = float(fusion_dropout)

        self.in_time_encoder = _ResCnnFeatureEncoder(time_branch_cfg)
        self.in_spec_encoder = _ResCnnFeatureEncoder(spec_branch_cfg)
        self.out_time_encoder = _ResCnnFeatureEncoder(time_branch_cfg)
        self.out_spec_encoder = _ResCnnFeatureEncoder(spec_branch_cfg)

        concat_dim = (
            self.in_time_encoder.feature_dim
            + self.in_spec_encoder.feature_dim
            + self.out_time_encoder.feature_dim
            + self.out_spec_encoder.feature_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity(),
        )
        self.inplane_head = nn.Linear(self.fusion_hidden_dim, self.num_classes)
        self.outplane_head = nn.Linear(self.fusion_hidden_dim, self.num_classes)

    def forward(
        self,
        in_time_x: torch.Tensor,
        in_spec_x: torch.Tensor,
        out_time_x: torch.Tensor,
        out_spec_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        in_time_feat = self.in_time_encoder(in_time_x)
        in_spec_feat = self.in_spec_encoder(in_spec_x)
        out_time_feat = self.out_time_encoder(out_time_x)
        out_spec_feat = self.out_spec_encoder(out_spec_x)

        fused = self.fusion(
            torch.cat([in_time_feat, in_spec_feat, out_time_feat, out_spec_feat], dim=1)
        )
        return self.inplane_head(fused), self.outplane_head(fused)
