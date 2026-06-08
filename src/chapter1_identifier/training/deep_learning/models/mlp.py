import torch
import torch.nn as nn
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.deep_learning_module.models.mlp import SimpleMLPConfig

# forecast 系任务：输出 (batch, predict_seq_len, output_feature_dim)
_FORECAST_TASKS = {"forecast", "seq2fixed"}
_SUPPORTED_TASKS = {"classification", "regression", "forecast", "seq2fixed"}


class MLP(nn.Module):
    """
    通用全连接网络（MLP），支持四类任务：

    task_type 说明：
      classification  → (batch, num_classes)                             [分类]
      regression      → (batch, regression_output_dim)                   [旧行为，向后兼容]
      forecast        → (batch, predict_seq_len, output_feature_dim)     [多步预测，推荐]
      seq2fixed       → 同 forecast，旧命名别名，向后兼容
    """

    def __init__(self, config: "SimpleMLPConfig"):
        super().__init__()
        self.cfg = config
        self._parse_config()
        self._validate_config()
        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_enable else nn.Identity()
        self.hidden_layers = self._build_hidden_layers()
        self._build_task_head()

    def _parse_config(self):
        self.input_shape  = self.cfg.input_shape
        self.hidden_dims  = self.cfg.hidden_dims
        self.activation   = self.cfg.get_activation_instance() or nn.ReLU()

        self.dropout_enable = self.cfg.dropout.enable
        self.dropout_prob   = self.cfg.dropout.prob

        self.task_type             = self.cfg.task_type
        self.num_classes           = self.cfg.num_classes
        self.regression_output_dim = self.cfg.regression_output_dim

        # forecast 专属
        self.predict_seq_len    = self.cfg.predict_seq_len
        # output_feature_dim 优先；未设置时退回 regression_output_dim（旧配置兼容）
        self.output_feature_dim = self.cfg.output_feature_dim or self.cfg.regression_output_dim

        self.input_flat_dim = self._calc_input_flat_dim()

    def _calc_input_flat_dim(self) -> int:
        flat = 1
        for d in self.input_shape:
            flat *= d
        return flat

    def _validate_config(self):
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims 不能为空，至少配置 1 个隐藏层维度")
        if self.input_flat_dim <= 0:
            raise ValueError(f"输入扁平化维度无效：{self.input_flat_dim}")
        if not (0 <= self.dropout_prob < 1):
            raise ValueError(f"dropout 概率必须在 [0,1) 区间，当前：{self.dropout_prob}")
        if self.task_type not in _SUPPORTED_TASKS:
            raise ValueError(f"task_type 仅支持 {sorted(_SUPPORTED_TASKS)}，当前：{self.task_type}")
        if self.task_type == "classification" and self.num_classes < 1:
            raise ValueError("分类任务 num_classes 必须≥1")
        if self.task_type == "regression" and self.regression_output_dim < 1:
            raise ValueError("regression 任务 regression_output_dim 必须≥1")
        if self.task_type in _FORECAST_TASKS:
            if self.predict_seq_len < 1:
                raise ValueError(f"{self.task_type} 任务 predict_seq_len 必须≥1")
            if self.output_feature_dim < 1:
                raise ValueError(f"{self.task_type} 任务 output_feature_dim 必须≥1")

    def _build_hidden_layers(self) -> nn.Sequential:
        layers = []
        prev = self.input_flat_dim
        for curr in self.hidden_dims:
            layers.extend([nn.Linear(prev, curr), self.activation, self.dropout])
            prev = curr
        return nn.Sequential(*layers)

    def _build_task_head(self):
        last = self.hidden_dims[-1]
        if self.task_type == "classification":
            self.task_head = nn.Linear(last, self.num_classes)
        elif self.task_type == "regression":
            self.task_head = nn.Linear(last, self.regression_output_dim)
        else:  # forecast / seq2fixed
            self.task_head = nn.Linear(last, self.predict_seq_len * self.output_feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        返回 shape：
          classification → (batch, num_classes)
          regression     → (batch, regression_output_dim)
          forecast 系    → (batch, predict_seq_len, output_feature_dim)
        """
        x = x.view(x.size(0), -1)
        x = self.hidden_layers(x)
        out = self.task_head(x)
        if self.task_type in _FORECAST_TASKS:
            out = out.reshape(-1, self.predict_seq_len, self.output_feature_dim)
        return out


# ── 向后兼容别名 ──────────────────────────────────────────────────────────────
SimpleMLP = MLP


# ── 测试示例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from src.training.deep_learning.models.mlp import MLP
    from src.config.deep_learning_module.models.mlp import SimpleMLPConfig, DropoutConfig

    ts_shape   = (300, 5)
    grid_shape = (1, 50, 60)

    # 测试1：分类
    print("=== 分类任务测试 ===")
    cfg = SimpleMLPConfig(
        input_shape=ts_shape, hidden_dims=[256, 128, 64],
        activation_type="ReLU", task_type="classification",
        num_classes=2, dropout=DropoutConfig(enable=True, prob=0.2)
    )
    out = MLP(cfg)(torch.randn(8, *ts_shape))
    print(f"输入 {(8,)+ts_shape}  →  输出 {tuple(out.shape)}  （预期 (8,2)）")
    assert out.shape == (8, 2)

    # 测试2：regression（旧行为，向后兼容）
    print("\n=== regression 任务测试（旧行为） ===")
    cfg = SimpleMLPConfig(
        input_shape=grid_shape, hidden_dims=[512, 256],
        activation_type="GELU", task_type="regression",
        regression_output_dim=1, dropout=DropoutConfig(enable=False)
    )
    out = MLP(cfg)(torch.randn(4, *grid_shape))
    print(f"输入 {(4,)+grid_shape}  →  输出 {tuple(out.shape)}  （预期 (4,1)）")
    assert out.shape == (4, 1)

    # 测试3：forecast（新任务，多步预测）
    print("\n=== forecast 任务测试（多步预测） ===")
    cfg = SimpleMLPConfig(
        input_shape=ts_shape, hidden_dims=[128, 64],
        activation_type="ReLU", task_type="forecast",
        output_feature_dim=5, predict_seq_len=4,
        dropout=DropoutConfig(enable=True, prob=0.1)
    )
    out = MLP(cfg)(torch.randn(8, *ts_shape))
    print(f"输入 {(8,)+ts_shape}  →  输出 {tuple(out.shape)}  （预期 (8,4,5)）")
    assert out.shape == (8, 4, 5)

    # 测试4：seq2fixed 旧命名兼容
    print("\n=== seq2fixed 旧命名兼容测试 ===")
    cfg = SimpleMLPConfig(
        input_shape=ts_shape, hidden_dims=[64],
        activation_type="Tanh", task_type="seq2fixed",
        output_feature_dim=3, predict_seq_len=2,
        dropout=DropoutConfig(enable=False)
    )
    out = MLP(cfg)(torch.randn(4, *ts_shape))
    print(f"输入 {(4,)+ts_shape}  →  输出 {tuple(out.shape)}  （预期 (4,2,3)）")
    assert out.shape == (4, 2, 3)

    print("\n所有 MLP 任务测试通过")

