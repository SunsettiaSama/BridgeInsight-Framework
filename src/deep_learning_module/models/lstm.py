import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
# 导入配置类（根据实际路径调整）
from src.config.deep_learning_module.models.lstm import LSTMConfig

class LSTM(nn.Module):
    """
    通用LSTM网络：兼容三类时序任务（适配Pydantic配置类）
    核心配置：
    - task_type: 任务类型，可选：
      ✅ classification（分类，默认）：输入序列→单标签
      ✅ seq2seq（序列到序列）：输入序列→等长输出序列
      ✅ regression（回归/长序列→短序列）：输入长序列→输出指定长度的短序列（如前100步预测后4步）
    """
    # 修改：参数从config_dict改为LSTMConfig实例
    def __init__(self, config: LSTMConfig):
        super().__init__()
        # 绑定配置实例（替换原config_dict）
        self.cfg = config

        # ---------------------- 核心任务配置（从配置类读取） ----------------------
        self.task_type = self.cfg.task_type
        supported_tasks = ["classification", "seq2seq", "regression"]
        if self.task_type not in supported_tasks:
            raise ValueError(f"task_type仅支持{supported_tasks}，当前为{self.task_type}")
        
        # ---------------------- 用户必填参数（已由配置类提前校验，无需重复判断None） ----------------------
        self.input_size = self.cfg.input_size
        self.num_classes = self.cfg.num_classes
        self.predict_seq_len = self.cfg.predict_seq_len  # seq2fixed必填（配置类已校验）
        
        # ---------------------- 中间层参数（从配置类读取，无需get方法） ----------------------
        self.hidden_size = self.cfg.hidden_size
        self.num_layers = self.cfg.num_layers
        self.bidirectional = self.cfg.bidirectional
        self.dropout = self.cfg.dropout
        self.batch_first = self.cfg.batch_first
        self.seq_dropout = self.cfg.seq_dropout
        
        # ---------------------- 网络层实现（逻辑不变，参数来源改为配置类） ----------------------
        # 1. LSTM核心层（双向/多层/ dropout兼容）
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=self.batch_first
        )
        
        # 2. 输出维度适配（双向LSTM需×2）
        self.lstm_output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # 3. 任务专属输出层（从配置类读取专属dropout）
        if self.task_type == "classification":
            # 分类：隐藏层→类别数
            self.output_fc = nn.Linear(self.lstm_output_dim, self.num_classes)
            self.dropout_layer = nn.Dropout(self.cfg.classifier_dropout)
        elif self.task_type == "seq2seq":
            # Seq2Seq：每个时间步隐藏层→输出维度
            self.output_fc = nn.Linear(self.lstm_output_dim, self.num_classes)
            self.dropout_layer = nn.Dropout(self.seq_dropout)
        else:  # regression（长序列→固定短序列）
            # 核心：用整个输入序列的最终隐藏状态→预测指定长度的序列
            self.output_fc = nn.Linear(self.lstm_output_dim, self.predict_seq_len * self.num_classes)
            self.dropout_layer = nn.Dropout(self.cfg.regression_dropout)

    def _get_final_hidden(self, h_n: torch.Tensor) -> torch.Tensor:
        """
        提取LSTM最终有效隐藏状态（适配单向/双向、多层）
        - 输入h_n: (num_layers * num_directions, batch_size, hidden_size)
        - 输出: (batch_size, lstm_output_dim)
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
        hidden: Optional[tuple] = None  # Seq2Seq解码阶段可传入隐藏状态
    ) -> torch.Tensor | tuple[torch.Tensor, tuple]:
        """
        前向传播：按任务类型返回对应格式输出
        - classification: (batch_size, num_classes)
        - seq2seq: (batch_size, seq_len, num_classes) 或 (output, (h_n, c_n))
        - regression: (batch_size, predict_seq_len, num_classes)
        """
        # 1. LSTM前向传播（支持传入初始隐藏状态）
        if hidden is not None:
            lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out.shape: (batch_size, seq_len, lstm_output_dim) (batch_first=True)

        # 2. 按任务类型处理输出
        if self.task_type == "classification":
            # 分类：取最后一个时间步输出 或 最终隐藏状态（二选一，这里选隐藏状态更稳定）
            final_h = self._get_final_hidden(h_n)
            final_h = self.dropout_layer(final_h)
            out = self.output_fc(final_h)  # (batch_size, num_classes)
            return out

        elif self.task_type == "seq2seq":
            # Seq2Seq：全序列输出（每个时间步都映射）
            lstm_out = self.dropout_layer(lstm_out)
            out = self.output_fc(lstm_out)  # (batch_size, seq_len, num_classes)
            return out, (h_n, c_n)

        else:  # regression（长序列→固定短序列）
            # 核心逻辑：用整个输入序列的最终隐藏状态，一次性预测指定长度的序列
            final_h = self._get_final_hidden(h_n)
            final_h = self.dropout_layer(final_h)
            # 线性层映射：(batch_size, lstm_output_dim) → (batch_size, predict_seq_len * num_classes)
            out = self.output_fc(final_h)
            # Reshape到目标形状：(batch_size, predict_seq_len, num_classes)
            out = out.reshape(-1, self.predict_seq_len, self.num_classes)
            return out
        




# ------------------------------ 测试示例（验证三类任务） ------------------------------
if __name__ == '__main__':

    import torch
    from src.deep_learning_module.models.lstm import LSTM
    from src.config.deep_learning_module.models.lstm import LSTMConfig

    # ==================== 测试1：分类任务 ====================
    print("=== 分类任务测试 ===")
    cls_config = LSTMConfig(
        input_size=16,          # 输入特征维度
        num_classes=2,          # 二分类
        hidden_size=64,
        task_type="classification"
    )
    cls_model = LSTM(cls_config)
    dummy_cls_input = torch.randn(8, 100, 16)  # (batch, seq_len=100, input_size)
    cls_output = cls_model(dummy_cls_input)
    print(f"分类输入形状: {dummy_cls_input.shape}")
    print(f"分类输出形状: {cls_output.shape} （预期：(8, 2)）\n")
    assert cls_output.shape == (8, 2), "分类任务输出形状错误"

    # ==================== 测试2：Seq2Seq任务（等长预测） ====================
    print("=== Seq2Seq任务测试（等长预测） ===")
    seq2seq_config = LSTMConfig(
        input_size=16,
        num_classes=16,         # 每步输出16维特征
        hidden_size=64,
        bidirectional=True,
        task_type="seq2seq"
    )
    seq2seq_model = LSTM(seq2seq_config)
    dummy_seq_input = torch.randn(8, 100, 16)  # (batch, seq_len=100, input_size)
    seq_output, (h_n, c_n) = seq2seq_model(dummy_seq_input)
    print(f"Seq2Seq输入形状: {dummy_seq_input.shape}")
    print(f"Seq2Seq输出形状: {seq_output.shape} （预期：(8, 100, 16)）\n")
    assert seq_output.shape == (8, 100, 16), "Seq2Seq任务输出形状错误"

    # ==================== 测试3：Seq2Fixed任务（前100步预测后4步） ====================
    print("=== Seq2Fixed任务测试（前100步预测后4步） ===")
    seq2fixed_config = LSTMConfig(
        input_size=16,
        num_classes=16,
        predict_seq_len=4,      # 预测后4步
        hidden_size=64,
        bidirectional=True,
        task_type="seq2fixed"
    )
    seq2fixed_model = LSTM(seq2fixed_config)
    dummy_fixed_input = torch.randn(8, 100, 16)  # (batch, seq_len=100, input_size)
    fixed_output = seq2fixed_model(dummy_fixed_input)
    print(f"Seq2Fixed输入形状: {dummy_fixed_input.shape}")
    print(f"Seq2Fixed输出形状: {fixed_output.shape} （预期：(8, 4, 16)）")
    assert fixed_output.shape == (8, 4, 16), "Seq2Fixed任务输出形状错误"

    print("\n🎉 所有LSTM任务测试通过！配置类与模型适配正常～")