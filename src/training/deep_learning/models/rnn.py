import warnings
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from src.config.deep_learning_module.models.rnn import RNNConfig


# forecast_tasks: 均表示"输入历史序列 → 预测未来固定长度序列"，语义等价
_FORECAST_TASKS = {"forecast", "regression", "seq2fixed"}


class RNN(nn.Module):
    """
    通用RNN网络：兼容三类时序任务（适配Pydantic配置类），与LSTM架构完全对齐

    支持的 task_type：
      classification  - 输入序列 → 单标签
      seq2seq         - 输入序列 → 等长输出序列
      forecast        - 输入历史序列 → 预测未来 predict_seq_len 步（推荐）
      regression      - 同 forecast，旧命名，向后兼容
      seq2fixed       - 同 forecast，旧命名，向后兼容

    核心差异：基于 nn.RNN 实现，仅维护隐藏状态（无细胞状态 c_n）
    """

    _SUPPORTED_TASKS = ["classification", "seq2seq", "forecast", "regression", "seq2fixed"]

    def __init__(self, config: RNNConfig):
        super().__init__()
        self.cfg = config
        self.task_type = self.cfg.task_type

        if self.task_type not in self._SUPPORTED_TASKS:
            raise ValueError(f"task_type 仅支持 {self._SUPPORTED_TASKS}，当前为 {self.task_type}")

        # ── 参数读取 ──────────────────────────────────────────────────────────
        self.input_size      = self.cfg.input_size
        # output_feature_dim 优先；若未设置则退回 num_output_dim（旧配置兼容）
        self.num_output_dim  = self.cfg.output_feature_dim or self.cfg.num_output_dim
        self.predict_seq_len = self.cfg.predict_seq_len
        self.hidden_size     = self.cfg.hidden_size
        self.num_layers      = self.cfg.num_layers
        self.bidirectional   = self.cfg.bidirectional
        self.dropout         = self.cfg.dropout
        self.batch_first     = self.cfg.batch_first
        self.seq_dropout     = self.cfg.seq_dropout

        # ── forecast 任务校验 ────────────────────────────────────────────────
        if self.task_type in _FORECAST_TASKS:
            if self.predict_seq_len < 1:
                raise ValueError(f"{self.task_type} 任务必须设置 predict_seq_len ≥ 1")
            if self.bidirectional:
                warnings.warn(
                    f"task_type='{self.task_type}' 时使用 bidirectional=True 会泄露未来信息，"
                    "请确认这是你期望的行为。预测任务通常应使用单向RNN。",
                    UserWarning,
                    stacklevel=2
                )

        # ── 网络层 ────────────────────────────────────────────────────────────
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=self.batch_first,
            nonlinearity=self.cfg.nonlinearity
        )

        self.rnn_output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        if self.task_type == "classification":
            self.output_fc    = nn.Linear(self.rnn_output_dim, self.num_output_dim)
            self.dropout_layer = nn.Dropout(self.cfg.classifier_dropout)

        elif self.task_type == "seq2seq":
            self.output_fc    = nn.Linear(self.rnn_output_dim, self.num_output_dim)
            self.dropout_layer = nn.Dropout(self.seq_dropout)

        else:  # forecast / regression / seq2fixed
            self.output_fc    = nn.Linear(self.rnn_output_dim, self.predict_seq_len * self.num_output_dim)
            self.dropout_layer = nn.Dropout(self.cfg.regression_dropout)

    # ── 辅助方法 ──────────────────────────────────────────────────────────────
    def _get_final_hidden(self, h_n: torch.Tensor) -> torch.Tensor:
        """
        提取最终有效隐藏状态
        h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        返回:      (batch_size, rnn_output_dim)
        """
        if self.bidirectional:
            return torch.cat([h_n[-2], h_n[-1]], dim=1)
        return h_n[-1]

    def _validate_input_dim(self, x: torch.Tensor) -> None:
        if self.batch_first:
            if x.ndim != 3 or x.shape[-1] != self.input_size:
                raise ValueError(
                    f"batch_first=True 时输入应为 (batch, seq_len, input_size)，"
                    f"当前 shape={x.shape}，input_size={self.input_size}"
                )
        else:
            if x.ndim != 3 or x.shape[2] != self.input_size:
                raise ValueError(
                    f"batch_first=False 时输入应为 (seq_len, batch, input_size)，"
                    f"当前 shape={x.shape}，input_size={self.input_size}"
                )

    # ── 前向传播 ──────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        参数：
            x:             输入张量
            hidden:        初始隐藏状态（可选）
            return_hidden: 为 True 时同时返回最终隐藏状态 h_n（默认 False）

        返回：
            return_hidden=False → out
            return_hidden=True  → (out, h_n)

        输出 shape：
            classification: (batch, num_output_dim)
            seq2seq:        (batch, seq_len, num_output_dim)
            forecast 系列:  (batch, predict_seq_len, num_output_dim)
        """
        self._validate_input_dim(x)

        rnn_out, h_n = self.rnn(x, hidden) if hidden is not None else self.rnn(x)

        if self.task_type == "classification":
            final_h = self._get_final_hidden(h_n)
            out = self.output_fc(self.dropout_layer(final_h))

        elif self.task_type == "seq2seq":
            out = self.output_fc(self.dropout_layer(rnn_out))

        else:  # forecast / regression / seq2fixed
            final_h = self._get_final_hidden(h_n)
            flat = self.output_fc(self.dropout_layer(final_h))
            out = flat.reshape(-1, self.predict_seq_len, self.num_output_dim)

        if return_hidden:
            return out, h_n
        return out


# ── 测试示例 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch
    from src.training.deep_learning.models.rnn import RNN
    from src.config.deep_learning_module.models.rnn import RNNConfig

    # 测试1：分类任务
    print("=== 分类任务测试 ===")
    cls_config = RNNConfig(
        input_size=16,
        num_output_dim=2,
        hidden_size=64,
        task_type="classification",
        nonlinearity="relu"
    )
    cls_model = RNN(cls_config)
    x = torch.randn(8, 100, 16)
    out = cls_model(x)
    print(f"输入: {x.shape}  →  输出: {out.shape}  （预期: (8, 2)）")
    assert out.shape == (8, 2)

    # 测试2：Seq2Seq任务，验证 return_hidden 接口
    print("\n=== Seq2Seq任务测试 ===")
    seq_config = RNNConfig(
        input_size=16,
        num_output_dim=16,
        hidden_size=64,
        bidirectional=True,
        task_type="seq2seq",
        nonlinearity="tanh"
    )
    seq_model = RNN(seq_config)
    x = torch.randn(8, 100, 16)
    out = seq_model(x)
    out_with_h, h_n = seq_model(x, return_hidden=True)
    print(f"输入: {x.shape}  →  输出: {out.shape}  （预期: (8, 100, 16)）")
    print(f"return_hidden=True 时 h_n: {h_n.shape}  （预期: (4, 8, 64)）")
    assert out.shape == (8, 100, 16)

    # 测试3：forecast 任务（推荐命名）
    print("\n=== Forecast任务测试（前100步预测后4步） ===")
    fc_config = RNNConfig(
        input_size=16,
        output_feature_dim=16,   # 使用新字段
        predict_seq_len=4,
        hidden_size=64,
        bidirectional=False,
        task_type="forecast"
    )
    fc_model = RNN(fc_config)
    x = torch.randn(8, 100, 16)
    out = fc_model(x)
    print(f"输入: {x.shape}  →  输出: {out.shape}  （预期: (8, 4, 16)）")
    assert out.shape == (8, 4, 16)

    # 测试4：regression / seq2fixed 旧命名兼容
    print("\n=== 旧命名兼容测试（regression / seq2fixed） ===")
    for alias in ["regression", "seq2fixed"]:
        cfg = RNNConfig(input_size=16, num_output_dim=16, predict_seq_len=4,
                        hidden_size=64, task_type=alias)
        m = RNN(cfg)
        out = m(torch.randn(8, 100, 16))
        assert out.shape == (8, 4, 16), f"{alias} 输出形状错误"
        print(f"  task_type='{alias}'  →  输出: {out.shape}  ✓")

    print("\n所有RNN任务测试通过")

