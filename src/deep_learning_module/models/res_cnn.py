import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.deep_learning_module.models.res_cnn import ResCNNConfig


class ResBlock1d(nn.Module):
    """
    1D 残差块：两层 Conv1d + BN + ReLU，跳跃连接自动匹配维度

    当 stride > 1 或 in_channels != out_channels 时，通过 1×1 卷积对齐跳跃分支。
    架构：
        input ─→ Conv1d → BN → ReLU → Conv1d → BN ─┐
              └─→ (1×1 Conv + BN，仅在需要时)      ─┘ → Add → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        needs_proj = (stride != 1) or (in_channels != out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if needs_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResCNN(nn.Module):
    """
    1D 残差卷积网络：专为时序振动信号分类任务设计

    架构：
        输入 (B, C, L)
        → 初始卷积层（宽卷积核，快速降采样）
        → N 个残差阶段（每阶段 num_blocks 个 ResBlock1d，首 block stride=2 降采样）
        → 全局平均池化 → 展平
        → FC 分类头

    输入格式：(B, in_channels, seq_len)
    输出格式：(B, num_classes)
    """

    def __init__(self, config: "ResCNNConfig"):
        super().__init__()
        self.cfg = config

        in_ch   = config.in_channels
        seq_len = config.input_size if isinstance(config.input_size, int) else config.input_size[0]

        if seq_len < 1:
            raise ValueError(f"input_size 必须 ≥ 1，当前：{seq_len}")
        if len(config.res_channels) == 0:
            raise ValueError("res_channels 不能为空，至少配置 1 个阶段")

        # ── 初始卷积（宽核，stride=2 快速降采样）──────────────────────────────
        init_ch = config.res_channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, init_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(init_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ── 残差阶段 ──────────────────────────────────────────────────────────
        stages: List[nn.Module] = []
        prev_ch = init_ch
        for stage_idx, out_ch in enumerate(config.res_channels):
            stride = 1 if stage_idx == 0 else 2
            blocks: List[nn.Module] = []
            for blk_idx in range(config.num_blocks):
                s = stride if blk_idx == 0 else 1
                blocks.append(ResBlock1d(prev_ch, out_ch, config.kernel_size, stride=s))
                prev_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        # ── 全局平均池化 + 分类头 ─────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=config.dropout_prob) if config.dropout_prob > 0 else nn.Identity()

        fc_in = prev_ch
        fc_layers: List[nn.Module] = []
        for hidden_dim in config.fc_hidden_dims:
            fc_layers.extend([
                nn.Linear(fc_in, hidden_dim),
                nn.ReLU(inplace=True),
                self.dropout,
            ])
            fc_in = hidden_dim
        fc_layers.append(nn.Linear(fc_in, config.num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (B, in_channels, seq_len)
        返回：
            logits: (B, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"输入必须是 3 维 (B, C, L)，当前：{x.shape}")
        # 数据集返回 (B, L, C)，自动转置为 Conv1d 期望的 (B, C, L)
        if x.shape[1] != self.cfg.in_channels and x.shape[2] == self.cfg.in_channels:
            x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ── 测试示例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from src.config.deep_learning_module.models.res_cnn import ResCNNConfig
    from src.deep_learning_module.models.res_cnn import ResCNN

    cfg = ResCNNConfig(
        in_channels=1,
        input_size=3000,
        res_channels=[64, 128, 256],
        num_blocks=2,
        kernel_size=3,
        fc_hidden_dims=[128],
        dropout_prob=0.5,
        num_classes=4,
    )
    model = ResCNN(cfg)

    x = torch.randn(8, 1, 3000)
    out = model(x)
    print(f"输入: {x.shape}  →  输出: {out.shape}  （预期: (8, 4)）")
    assert out.shape == (8, 4)

    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量：{total:,}")
    print("ResCNN 测试通过")
