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
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.fusion_dropout = float(fusion_dropout)
        self.cross_attn_heads = int(max(1, cross_attn_heads))
        if self.fusion_hidden_dim % self.cross_attn_heads != 0:
            self.cross_attn_heads = 1

        self.in_time_encoder = _ResCnnFeatureEncoder(time_branch_cfg)
        self.in_spec_encoder = _ResCnnFeatureEncoder(spec_branch_cfg)
        self.out_time_encoder = _ResCnnFeatureEncoder(time_branch_cfg)
        self.out_spec_encoder = _ResCnnFeatureEncoder(spec_branch_cfg)

        in_concat_dim = self.in_time_encoder.feature_dim + self.in_spec_encoder.feature_dim
        out_concat_dim = self.out_time_encoder.feature_dim + self.out_spec_encoder.feature_dim
        self.inplane_fusion = nn.Sequential(
            nn.Linear(in_concat_dim, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity(),
        )
        self.outplane_fusion = nn.Sequential(
            nn.Linear(out_concat_dim, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_hidden_dim,
            num_heads=self.cross_attn_heads,
            dropout=self.fusion_dropout if self.fusion_dropout > 0 else 0.0,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.fusion_hidden_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity(),
            nn.Linear(self.fusion_hidden_dim * 2, self.fusion_hidden_dim),
        )
        self.cross_ffn_norm = nn.LayerNorm(self.fusion_hidden_dim)
        self.cross_out_dropout = (
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity()
        )

        self.inplane_head = nn.Linear(self.fusion_hidden_dim * 2, self.num_classes)
        self.outplane_head = nn.Linear(self.fusion_hidden_dim * 2, self.num_classes)

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

        # Stage-1: same-direction early fusion (inplane / outplane separately).
        in_local = self.inplane_fusion(torch.cat([in_time_feat, in_spec_feat], dim=1))
        out_local = self.outplane_fusion(torch.cat([out_time_feat, out_spec_feat], dim=1))

        # Stage-2: lightweight cross-attention over two direction tokens.
        tokens = torch.stack([in_local, out_local], dim=1)  # (B, 2, D)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.cross_attn_norm(tokens + self.cross_out_dropout(attn_out))
        ffn_out = self.cross_ffn(tokens)
        tokens = self.cross_ffn_norm(tokens + self.cross_out_dropout(ffn_out))

        in_cross = tokens[:, 0, :]
        out_cross = tokens[:, 1, :]
        in_feat = torch.cat([in_local, in_cross], dim=1)
        out_feat = torch.cat([out_local, out_cross], dim=1)
        return self.inplane_head(in_feat), self.outplane_head(out_feat)


class QuadStreamDualHeadContextResCNN(nn.Module):
    def __init__(
        self,
        time_branch_cfg: BranchConfig,
        spec_branch_cfg: BranchConfig,
        context_branch_cfg: BranchConfig,
        num_classes: int = 4,
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.1,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.fusion_dropout = float(fusion_dropout)
        self.cross_attn_heads = int(max(1, cross_attn_heads))
        if self.fusion_hidden_dim % self.cross_attn_heads != 0:
            self.cross_attn_heads = 1

        self.in_time_encoder = _ResCnnFeatureEncoder(time_branch_cfg)
        self.in_spec_encoder = _ResCnnFeatureEncoder(spec_branch_cfg)
        self.in_context_encoder = _ResCnnFeatureEncoder(context_branch_cfg)
        self.out_time_encoder = _ResCnnFeatureEncoder(time_branch_cfg)
        self.out_spec_encoder = _ResCnnFeatureEncoder(spec_branch_cfg)
        self.out_context_encoder = _ResCnnFeatureEncoder(context_branch_cfg)

        in_concat_dim = (
            self.in_time_encoder.feature_dim
            + self.in_spec_encoder.feature_dim
            + self.in_context_encoder.feature_dim
        )
        out_concat_dim = (
            self.out_time_encoder.feature_dim
            + self.out_spec_encoder.feature_dim
            + self.out_context_encoder.feature_dim
        )
        self.inplane_fusion = nn.Sequential(
            nn.Linear(in_concat_dim, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity(),
        )
        self.outplane_fusion = nn.Sequential(
            nn.Linear(out_concat_dim, self.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity(),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_hidden_dim,
            num_heads=self.cross_attn_heads,
            dropout=self.fusion_dropout if self.fusion_dropout > 0 else 0.0,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.fusion_hidden_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity(),
            nn.Linear(self.fusion_hidden_dim * 2, self.fusion_hidden_dim),
        )
        self.cross_ffn_norm = nn.LayerNorm(self.fusion_hidden_dim)
        self.cross_out_dropout = (
            nn.Dropout(p=self.fusion_dropout) if self.fusion_dropout > 0 else nn.Identity()
        )

        self.inplane_head = nn.Linear(self.fusion_hidden_dim * 2, self.num_classes)
        self.outplane_head = nn.Linear(self.fusion_hidden_dim * 2, self.num_classes)

    def forward(
        self,
        in_time_x: torch.Tensor,
        in_spec_x: torch.Tensor,
        in_context_x: torch.Tensor | None,
        out_time_x: torch.Tensor,
        out_spec_x: torch.Tensor,
        out_context_x: torch.Tensor | None,
        use_context: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        in_time_feat = self.in_time_encoder(in_time_x)
        in_spec_feat = self.in_spec_encoder(in_spec_x)
        out_time_feat = self.out_time_encoder(out_time_x)
        out_spec_feat = self.out_spec_encoder(out_spec_x)
        if use_context and in_context_x is not None and out_context_x is not None:
            in_context_feat = self.in_context_encoder(in_context_x)
            out_context_feat = self.out_context_encoder(out_context_x)
        else:
            in_context_feat = torch.zeros(
                in_time_feat.shape[0],
                self.in_context_encoder.feature_dim,
                device=in_time_feat.device,
                dtype=in_time_feat.dtype,
            )
            out_context_feat = torch.zeros(
                out_time_feat.shape[0],
                self.out_context_encoder.feature_dim,
                device=out_time_feat.device,
                dtype=out_time_feat.dtype,
            )

        in_local = self.inplane_fusion(torch.cat([in_time_feat, in_spec_feat, in_context_feat], dim=1))
        out_local = self.outplane_fusion(torch.cat([out_time_feat, out_spec_feat, out_context_feat], dim=1))

        tokens = torch.stack([in_local, out_local], dim=1)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.cross_attn_norm(tokens + self.cross_out_dropout(attn_out))
        ffn_out = self.cross_ffn(tokens)
        tokens = self.cross_ffn_norm(tokens + self.cross_out_dropout(ffn_out))

        in_cross = tokens[:, 0, :]
        out_cross = tokens[:, 1, :]
        in_feat = torch.cat([in_local, in_cross], dim=1)
        out_feat = torch.cat([out_local, out_cross], dim=1)
        return self.inplane_head(in_feat), self.outplane_head(out_feat)
