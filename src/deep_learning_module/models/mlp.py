
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, TYPE_CHECKING

# 类型提示兼容
if TYPE_CHECKING:
    from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig


class MLP(nn.Module):
    """
    简洁版双任务全连接网络（MLP）
    核心特性：
    1. 任务可配置：通过task_type指定分类/回归模式，参考UNet架构；
    2. 轻量化设计：隐藏层结构简洁，无冗余计算，支持dropout正则化；
    3. 输入兼容：自动扁平化时序/网格输入，适配之前数据集的两种输出格式；
    4. 配置驱动：通过配置类管理参数，与UNet使用风格保持一致。
    """
    def __init__(self, config: "SimpleMLPConfig"):
        super(MLP, self).__init__()

        # 1. 核心配置绑定与解析
        self.cfg = config
        self._parse_config()

        # 2. 配置合法性校验
        self._validate_config()

        # 3. 通用层定义（dropout，默认恒等映射）
        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_enable else nn.Identity()

        # 4. 构建隐藏层（动态生成，简洁灵活）
        self.hidden_layers = self._build_hidden_layers()

        # 5. 构建任务适配头（分类/回归分支，参考SimpleCNN的_task_head设计）
        self._build_task_head()

    def _parse_config(self):
        """解析配置参数，与UNet的_parse_config逻辑对齐"""
        # 基础网络配置
        self.input_shape = self.cfg.input_shape  # 从数据集get_input_shape()获取
        self.hidden_dims = self.cfg.hidden_dims  # 隐藏层维度列表
        self.activation = self.cfg.get_activation_instance() or nn.ReLU()  # 激活函数

        # Dropout配置
        self.dropout_enable = self.cfg.dropout.enable
        self.dropout_prob = self.cfg.dropout.prob

        # 任务配置（核心：区分分类/回归）
        self.task_type = self.cfg.task_type  # "classification" / "regression"
        self.num_classes = self.cfg.num_classes  # 分类任务：类别数
        self.regression_output_dim = self.cfg.regression_output_dim  # 回归任务：输出维度

        # 计算输入扁平化维度
        self.input_flat_dim = self._calc_input_flat_dim()

    def _calc_input_flat_dim(self) -> int:
        """计算输入形状扁平化后的总特征数，兼容时序/网格输入"""
        flat_dim = 1
        for dim in self.input_shape:
            flat_dim *= dim
        return flat_dim

    def _validate_config(self):
        """配置合法性校验，参考UNet的_validate_config逻辑"""
        # 基础网络校验
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims不能为空，至少配置1个隐藏层维度")
        if self.input_flat_dim <= 0:
            raise ValueError(f"输入扁平化维度无效：{self.input_flat_dim}，请检查input_shape配置")
        if self.dropout_prob < 0 or self.dropout_prob >= 1:
            raise ValueError(f"dropout概率必须在[0,1)区间，当前：{self.dropout_prob}")

        # 任务类型校验
        if self.task_type not in ["classification", "regression"]:
            raise ValueError(f"task_type仅支持'classification'/'regression'，当前：{self.task_type}")

        # 分类任务专属校验
        if self.task_type == "classification" and self.num_classes < 1:
            raise ValueError("分类任务num_classes必须≥1")

        # 回归任务专属校验
        if self.task_type == "regression" and self.regression_output_dim < 1:
            raise ValueError("回归任务regression_output_dim必须≥1")

    def _build_hidden_layers(self) -> nn.Sequential:
        """构建隐藏层，动态生成，保持简洁轻量化"""
        layer_list = []
        prev_dim = self.input_flat_dim

        # 遍历隐藏层维度，构建全连接层+激活+dropout
        for curr_dim in self.hidden_dims:
            layer_list.extend([
                nn.Linear(prev_dim, curr_dim),
                self.activation,
                self.dropout
            ])
            prev_dim = curr_dim

        return nn.Sequential(*layer_list)

    def _build_task_head(self):
        """构建任务适配头，参考UNet的_build_task_head，区分分类/回归"""
        last_hidden_dim = self.hidden_dims[-1]

        if self.task_type == "classification":
            # 分类任务：全连接层映射到类别数（无激活，输出logits）
            self.task_head = nn.Linear(last_hidden_dim, self.num_classes)
        elif self.task_type == "regression":
            # 回归任务：全连接层映射到指定输出维度（无激活，直接输出连续值）
            self.task_head = nn.Linear(last_hidden_dim, self.regression_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，逻辑简洁清晰，参考UNet的forward结构
        Args:
            x: 输入张量，形状为(batch_size, *input_shape)（兼容时序/网格模式）
        Returns:
            output: 任务输出，分类为logits，回归为连续预测值
        """
        # 步骤1：扁平化输入（保留batch维度，适配任意输入形状）
        x = x.view(x.size(0), -1)  # 等价于x.flatten(1)

        # 步骤2：通过隐藏层
        x = self.hidden_layers(x)

        # 步骤3：通过任务头，输出最终结果
        output = self.task_head(x)

        return output



# ------------------------------ 测试示例（验证分类/回归双模式） ------------------------------
if __name__ == "__main__":
    import torch
    from src.deep_learning_module.models.mlp import MLP
    from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig

    # 1. 模拟数据集输入形状（两种模式）
    ts_input_shape = (300, 5)  # 时序模式：(seq_len, feat_dim)
    grid_input_shape = (1, 50, 60)  # 网格模式：(C, H, W)

    # 2. 测试1：分类任务（二分类，适配时序输入）
    print("===== 测试1：分类任务（时序输入） =====")
    cls_config = SimpleMLPConfig(
        input_shape=ts_input_shape,
        hidden_dims=[256, 128, 64],  # 顶层字段，直接传递隐藏层维度列表
        activation_type="ReLU",       # 顶层字段，字符串指定激活函数
        task_type="classification",
        num_classes=2,
        dropout=DropoutConfig(enable=True, prob=0.2)  # Dropout嵌套配置保持不变
    )
    cls_mlp = MLP(cls_config)
    cls_input = torch.randn(8, *ts_input_shape)  # batch_size=8
    cls_output = cls_mlp(cls_input)
    print(f"输入形状：{cls_input.shape}")
    print(f"输出形状：{cls_output.shape}（分类logits，形状为[batch_size, num_classes]）")
    assert cls_output.shape == (8, 2), f"分类任务输出形状错误，预期(8,2)，实际{cls_output.shape}"

    # 3. 测试2：回归任务（1维输出，适配网格输入）
    print("\n===== 测试2：回归任务（网格输入） =====")
    reg_config = SimpleMLPConfig(
        input_shape=grid_input_shape,
        hidden_dims=[512, 256],       # 顶层字段，隐藏层维度列表
        activation_type="GELU",        # 顶层字段，指定GELU激活函数
        task_type="regression",
        regression_output_dim=1,
        dropout=DropoutConfig(enable=False)  # 禁用Dropout
    )
    reg_mlp = MLP(reg_config)
    reg_input = torch.randn(4, *grid_input_shape)  # batch_size=4
    reg_output = reg_mlp(reg_input)
    print(f"输入形状：{reg_input.shape}")
    print(f"输出形状：{reg_output.shape}（回归预测值，形状为[batch_size, regression_output_dim]）")
    assert reg_output.shape == (4, 1), f"回归任务输出形状错误，预期(4,1)，实际{reg_output.shape}"

    # 4. 测试3：多分类任务（10分类，时序输入）
    print("\n===== 测试3：分类任务（多分类） =====")
    multi_cls_config = SimpleMLPConfig(
        input_shape=ts_input_shape,
        hidden_dims=[128, 64],         # 顶层字段，简洁隐藏层配置
        activation_type="LeakyReLU",   # 顶层字段，指定LeakyReLU激活函数
        task_type="classification",
        num_classes=10,
        dropout=DropoutConfig(enable=True, prob=0.1)  # 低丢弃概率的Dropout
    )
    multi_cls_mlp = MLP(multi_cls_config)
    multi_cls_input = torch.randn(2, *ts_input_shape)  # batch_size=2
    multi_cls_output = multi_cls_mlp(multi_cls_input)
    print(f"输入形状：{multi_cls_input.shape}")
    print(f"输出形状：{multi_cls_output.shape}（多分类logits，形状为[batch_size, 10]）")
    assert multi_cls_output.shape == (2, 10), f"多分类任务输出形状错误，预期(2,10)，实际{multi_cls_output.shape}"

    print("\n🎉 所有测试通过！MLP模型与Config类适配正常～")


# ------------------------------ 向后兼容别名 ------------------------------
# 保留SimpleMLP别名，用于向后兼容
SimpleMLP = MLP

