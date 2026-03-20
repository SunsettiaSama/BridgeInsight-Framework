import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
# 导入配置类（与LSTM配置类结构一致，可复用/微调）
from src.config.deep_learning_module.models.rnn import RNNConfig

class RNN(nn.Module):
    """
    通用RNN网络：兼容三类时序任务（适配Pydantic配置类），与LSTM架构完全对齐
    核心配置：
    - task_type: 任务类型，可选：
      ✅ classification（分类，默认）：输入序列→单标签
      ✅ seq2seq（序列到序列）：输入序列→等长输出序列
      ✅ regression(长序列→固定短序列）：输入长序列→输出指定长度的短序列（如前100步预测后4步）
    核心差异：基于nn.RNN实现，仅维护隐藏状态（无细胞状态c_n）
    """
    def __init__(self, config: RNNConfig):
        super().__init__()
        # 绑定配置实例
        self.cfg = config

        # ---------------------- 核心任务配置（从配置类读取+校验） ----------------------
        self.task_type = self.cfg.task_type
        supported_tasks = ["classification", "seq2seq", "regression"]
        if self.task_type not in supported_tasks:
            raise ValueError(f"task_type仅支持{supported_tasks}，当前为{self.task_type}")
        
        # ---------------------- 用户必填参数（配置类已提前校验） ----------------------
        self.input_size = self.cfg.input_size
        self.num_output_dim = self.cfg.num_output_dim  # 替换num_classes，语义更通用
        self.predict_seq_len = self.cfg.predict_seq_len  # regression必填
        
        # ---------------------- 中间层参数（从配置类读取） ----------------------
        self.hidden_size = self.cfg.hidden_size
        self.num_layers = self.cfg.num_layers
        self.bidirectional = self.cfg.bidirectional
        self.dropout = self.cfg.dropout
        self.batch_first = self.cfg.batch_first
        self.seq_dropout = self.cfg.seq_dropout
        
        # ---------------------- 网络层实现（适配RNN原生特性） ----------------------
        # 1. RNN核心层（双向/多层/ dropout兼容，无细胞状态）
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=self.batch_first,
            nonlinearity=self.cfg.nonlinearity  # RNN专属：激活函数（tanh/relu）
        )
        
        # 2. 输出维度适配（双向RNN需×2）
        self.rnn_output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # 3. 任务专属输出层（语义统一，替换num_classes为num_output_dim）
        if self.task_type == "classification":
            # 分类：隐藏层→类别数
            self.output_fc = nn.Linear(self.rnn_output_dim, self.num_output_dim)
            self.dropout_layer = nn.Dropout(self.cfg.classifier_dropout)
        elif self.task_type == "seq2seq":
            # Seq2Seq：每个时间步隐藏层→输出维度
            self.output_fc = nn.Linear(self.rnn_output_dim, self.num_output_dim)
            self.dropout_layer = nn.Dropout(self.seq_dropout)
        else:  # regression
            # 核心：用整个输入序列的最终隐藏状态→预测指定长度的序列
            self.output_fc = nn.Linear(self.rnn_output_dim, self.predict_seq_len * self.num_output_dim)
            self.dropout_layer = nn.Dropout(self.cfg.regression_dropout)

    def _get_final_hidden(self, h_n: torch.Tensor) -> torch.Tensor:
        """
        提取RNN最终有效隐藏状态（适配单向/双向、多层，无细胞状态）
        - 输入h_n: (num_layers * num_directions, batch_size, hidden_size)
        - 输出: (batch_size, rnn_output_dim)
        """
        num_directions = 2 if self.bidirectional else 1
        # 取最后一层的隐藏状态（双向则拼接前向+后向）
        if self.bidirectional:
            final_h = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        else:
            final_h = h_n[-1, :, :]
        return final_h

    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None  # RNN仅传入隐藏状态h_n，无c_n
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：按任务类型返回对应格式输出（适配RNN无细胞状态的特性）
        - classification: (batch_size, num_output_dim)
        - seq2seq: (batch_size, seq_len, num_output_dim) 或 (output, h_n)
        - regression: (batch_size, predict_seq_len, num_output_dim)
        """
        # 输入维度校验（兼容batch_first=True/False）
        self._validate_input_dim(x)
        
        # 1. RNN前向传播（支持传入初始隐藏状态，仅h_n）
        if hidden is not None:
            rnn_out, h_n = self.rnn(x, hidden)
        else:
            rnn_out, h_n = self.rnn(x)
        # rnn_out.shape: (batch_size, seq_len, rnn_output_dim) (batch_first=True)

        # 2. 按任务类型处理输出
        if self.task_type == "classification":
            # 分类：取最后一个时间步输出 或 最终隐藏状态（选隐藏状态更稳定）
            final_h = self._get_final_hidden(h_n)
            final_h = self.dropout_layer(final_h)
            out = self.output_fc(final_h)  # (batch_size, num_output_dim)
            return out

        elif self.task_type == "seq2seq":
            # Seq2Seq：全序列输出（每个时间步都映射）
            rnn_out = self.dropout_layer(rnn_out)
            out = self.output_fc(rnn_out)  # (batch_size, seq_len, num_output_dim)
            return out, h_n  # 仅返回h_n，无c_n

        else:  # regression
            # 核心逻辑：用整个输入序列的最终隐藏状态，一次性预测指定长度的序列
            final_h = self._get_final_hidden(h_n)
            final_h = self.dropout_layer(final_h)
            # 线性层映射：(batch_size, rnn_output_dim) → (batch_size, predict_seq_len * num_output_dim)
            out = self.output_fc(final_h)
            # 维度校验：避免reshape失败
            assert out.shape[-1] == self.predict_seq_len * self.num_output_dim, \
                f"输出层维度{out.shape[-1]}与predict_seq_len×num_output_dim={self.predict_seq_len * self.num_output_dim}不匹配"
            # Reshape到目标形状：(batch_size, predict_seq_len, num_output_dim)
            out = out.reshape(-1, self.predict_seq_len, self.num_output_dim)
            return out
    
    def _validate_input_dim(self, x: torch.Tensor) -> None:
        """校验输入维度是否符合batch_first配置，避免维度不匹配错误"""
        if self.batch_first:
            if x.ndim != 3 or x.shape[-1] != self.input_size:
                raise ValueError(
                    f"batch_first=True时，输入应为3维 (batch, seq_len, input_size)，当前shape={x.shape}，input_size={self.input_size}"
                )
        else:
            if x.ndim != 3 or x.shape[2] != self.input_size:
                raise ValueError(
                    f"batch_first=False时，输入应为3维 (seq_len, batch, input_size)，当前shape={x.shape}，input_size={self.input_size}"
                )




# ------------------------------ 测试示例（验证三类任务） ------------------------------
if __name__ == '__main__':

    import torch
    from src.deep_learning_module.models.rnn import RNN
    from src.config.deep_learning_module.models.rnn import RNNConfig

    # ==================== 测试1：分类任务 ====================
    print("=== 分类任务测试 ===")
    cls_config = RNNConfig(
        input_size=16,          # 输入特征维度
        num_output_dim=2,       # 二分类
        hidden_size=64,
        task_type="classification",
        nonlinearity="relu"
    )
    cls_model = RNN(cls_config)
    dummy_cls_input = torch.randn(8, 100, 16)  # (batch, seq_len=100, input_size)
    cls_output = cls_model(dummy_cls_input)
    print(f"分类输入形状: {dummy_cls_input.shape}")
    print(f"分类输出形状: {cls_output.shape} （预期：(8, 2)）\n")
    assert cls_output.shape == (8, 2), "分类任务输出形状错误"

    # ==================== 测试2：Seq2Seq任务（等长预测） ====================
    print("=== Seq2Seq任务测试（等长预测） ===")
    seq2seq_config = RNNConfig(
        input_size=16,
        num_output_dim=16,      # 每步输出16维特征
        hidden_size=64,
        bidirectional=True,
        task_type="seq2seq",
        nonlinearity="tanh"
    )
    seq2seq_model = RNN(seq2seq_config)
    dummy_seq_input = torch.randn(8, 100, 16)  # (batch, seq_len=100, input_size)
    seq_output, h_n = seq2seq_model(dummy_seq_input)
    print(f"Seq2Seq输入形状: {dummy_seq_input.shape}")
    print(f"Seq2Seq输出形状: {seq_output.shape} （预期：(8, 100, 16)）")
    print(f"隐藏状态形状: {h_n.shape} （预期：(4, 8, 64) → 2层×双向=4，batch=8，hidden_size=64）\n")
    assert seq_output.shape == (8, 100, 16), "Seq2Seq任务输出形状错误"

    # ==================== 测试3：Seq2Fixed任务（前100步预测后4步） ====================
    print("=== Seq2Fixed任务测试（前100步预测后4步） ===")
    seq2fixed_config = RNNConfig(
        input_size=16,
        num_output_dim=16,
        predict_seq_len=4,      # 预测后4步
        hidden_size=64,
        bidirectional=True,
        task_type="seq2fixed"
    )
    seq2fixed_model = RNN(seq2fixed_config)
    dummy_fixed_input = torch.randn(8, 100, 16)  # (batch, seq_len=100, input_size)
    fixed_output = seq2fixed_model(dummy_fixed_input)
    print(f"Seq2Fixed输入形状: {dummy_fixed_input.shape}")
    print(f"Seq2Fixed输出形状: {fixed_output.shape} （预期：(8, 4, 16)）")
    assert fixed_output.shape == (8, 4, 16), "Seq2Fixed任务输出形状错误"

    print("\n🎉 所有RNN任务测试通过！配置类与模型适配正常～")