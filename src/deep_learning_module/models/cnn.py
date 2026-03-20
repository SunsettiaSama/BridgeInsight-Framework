import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.deep_learning_module.models.cnn import CNNConfig


class CNN(nn.Module):
    """
    通用CNN网络架构：支持多分类/回归任务，适配图像/时序信号输入
    
    核心特性：
    1. 双输入类型：图像（2D卷积）、时序信号（1D卷积）
    2. 双任务类型：多分类（任意类别数）、回归（单/多维度输出）
    3. 模块化设计：可配置卷积层数、通道数、池化方式、全连接层维度
    4. 工程化配置：输入尺寸校验、激活函数选择、Dropout正则化
    
    架构流程：
    - 图像输入：(B, C, H, W) → (Conv2d → BN2d → 激活 → Pool2d) × N → 展平 → FC → 输出
    - 时序输入：(B, C, L) → (Conv1d → BN1d → 激活 → Pool1d) × N → 展平 → FC → 输出
    - 多分类：输出层无激活（配合CrossEntropyLoss）
    - 回归：输出层无激活（配合MSELoss）
    """
    def __init__(self, config: "CNNConfig"):
        super(CNN, self).__init__()

        # 核心配置绑定
        self.cfg = config
        self._parse_config()

        # 配置合法性校验
        self._validate_config()

        # 通用层定义
        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_enable else nn.Identity()
        self.activation = self._get_activation_fn()

        # 动态选择卷积/池化/BN层类型（1D/2D）
        self.conv_layer_cls = self._get_conv_layer_cls()
        self.pool_layer_cls = self._get_pool_layer_cls()
        self.bn_layer_cls = self._get_bn_layer_cls()

        # 构建核心网络层
        self.conv_blocks = self._build_conv_blocks()
        self.fc_layers = self._build_fc_layers()

    def _parse_config(self):
        """解析Config配置，提取核心参数"""
        # 基础配置
        self.in_channels = self.cfg.in_channels
        self.input_type = self.cfg.input_type  # image/timeseries
        self.task_type = self.cfg.task_type    # classification/regression
        self.input_size = self.cfg.input_size  # 图像：(H,W)，时序：(seq_len,)
        self.dropout_enable = self.cfg.dropout.enable
        self.dropout_prob = self.cfg.dropout.prob
        self.activation_type = self.cfg.activation_type

        # 任务相关配置
        if self.task_type == "classification":
            self.num_classes = self.cfg.num_classes  # 多分类类别数
        elif self.task_type == "regression":
            self.num_outputs = self.cfg.num_outputs  # 回归输出维度（≥1）

        # 卷积层配置
        self.conv_channels = self.cfg.conv.conv_channels
        self.conv_kernel_size = self.cfg.conv.kernel_size
        self.conv_stride = self.cfg.conv.stride
        self.conv_padding = self.cfg.conv.padding
        
        # 池化层配置
        self.pool_type = self.cfg.pool.pool_type
        self.pool_kernel_size = self.cfg.pool.pool_kernel_size
        self.pool_stride = self.cfg.pool.pool_stride
        
        # 全连接层配置
        self.fc_hidden_dims = self.cfg.fc.fc_hidden_dims

    def _validate_config(self):
        """配置合法性校验"""
        # 基础校验
        if len(self.conv_channels) == 0:
            raise ValueError("conv.conv_channels 不能为空，至少配置1个通道数")
        if self.input_size is None:
            raise ValueError("input_size 必须指定：图像=(H,W)，时序=(seq_len,)")
        
        # 输入类型校验
        if self.input_type not in ["image", "timeseries"]:
            raise ValueError(f"input_type仅支持'image'/'timeseries'，当前：{self.input_type}")
        
        # 输入尺寸校验（匹配输入类型）
        if self.input_type == "image":
            if not isinstance(self.input_size, tuple) or len(self.input_size) != 2:
                raise ValueError(f"图像输入的input_size必须是2维tuple (H,W)，当前：{self.input_size}")
            h, w = self.input_size
            if h < 1 or w < 1:
                raise ValueError(f"图像input_size必须≥1，当前：{self.input_size}")
        elif self.input_type == "timeseries":
            # 时序输入size：支持int或1维tuple (seq_len,)
            if isinstance(self.input_size, int):
                self.input_size = (self.input_size,)
            elif not isinstance(self.input_size, tuple) or len(self.input_size) != 1:
                raise ValueError(f"时序输入的input_size必须是int或1维tuple (seq_len,)，当前：{self.input_size}")
            seq_len = self.input_size[0]
            if seq_len < 1:
                raise ValueError(f"时序seq_len必须≥1，当前：{seq_len}")
        
        # 任务类型校验
        if self.task_type not in ["classification", "regression"]:
            raise ValueError(f"task_type仅支持'classification'/'regression'，当前：{self.task_type}")
        
        # 任务参数校验
        if self.task_type == "classification":
            if getattr(self, "num_classes", 0) < 2:  # 多分类至少2类
                raise ValueError("多分类任务num_classes必须≥2")
        elif self.task_type == "regression":
            if getattr(self, "num_outputs", 0) < 1:  # 回归至少输出1个值
                raise ValueError("回归任务num_outputs必须≥1")
        
        # 激活函数/池化类型校验
        if self.activation_type not in ["relu", "leaky_relu"]:
            raise ValueError(f"activation_type仅支持'relu'/'leaky_relu'，当前：{self.activation_type}")
        if self.pool_type not in ["max", "avg"]:
            raise ValueError(f"pool_type仅支持'max'/'avg'，当前：{self.pool_type}")

    def _get_activation_fn(self) -> nn.Module:
        """获取激活函数"""
        if self.activation_type == "relu":
            return nn.ReLU(inplace=True)
        elif self.activation_type == "leaky_relu":
            return nn.LeakyReLU(inplace=True)

    def _get_conv_layer_cls(self) -> nn.Module:
        """根据输入类型选择卷积层（1D/2D）"""
        return nn.Conv1d if self.input_type == "timeseries" else nn.Conv2d

    def _get_pool_layer_cls(self) -> nn.Module:
        """根据输入类型选择池化层（1D/2D）"""
        if self.pool_type == "max":
            return nn.MaxPool1d if self.input_type == "timeseries" else nn.MaxPool2d
        elif self.pool_type == "avg":
            return nn.AvgPool1d if self.input_type == "timeseries" else nn.AvgPool2d

    def _get_bn_layer_cls(self) -> nn.Module:
        """根据输入类型选择BN层（1D/2D）"""
        return nn.BatchNorm1d if self.input_type == "timeseries" else nn.BatchNorm2d

    def _calc_fc_input_dim(self) -> int:
        """计算全连接层输入维度（适配1D/2D输入）"""
        prev_channels = self.in_channels
        
        if self.input_type == "image":
            # 图像输入：(H, W) → 计算卷积+池化后的(H', W')
            h, w = self.input_size
            for curr_channels in self.conv_channels:
                # 卷积层尺寸变化：H_out = (H_in + 2*padding - kernel_size) // stride + 1
                h = (h + 2 * self.conv_padding - self.conv_kernel_size) // self.conv_stride + 1
                w = (w + 2 * self.conv_padding - self.conv_kernel_size) // self.conv_stride + 1
                
                # 池化层尺寸变化：H_out = (H_in - kernel_size) // stride + 1
                h = (h - self.pool_kernel_size) // self.pool_stride + 1
                w = (w - self.pool_kernel_size) // self.pool_stride + 1
                
                prev_channels = curr_channels
            # 全连接输入维度 = 通道数 × H' × W'
            return prev_channels * h * w
        
        elif self.input_type == "timeseries":
            # 时序输入：(seq_len,) → 计算卷积+池化后的(seq_len')
            seq_len = self.input_size[0]
            for curr_channels in self.conv_channels:
                # 卷积层尺寸变化：L_out = (L_in + 2*padding - kernel_size) // stride + 1
                seq_len = (seq_len + 2 * self.conv_padding - self.conv_kernel_size) // self.conv_stride + 1
                
                # 池化层尺寸变化：L_out = (L_in - kernel_size) // stride + 1
                seq_len = (seq_len - self.pool_kernel_size) // self.pool_stride + 1
                
                prev_channels = curr_channels
            # 全连接输入维度 = 通道数 × seq_len'
            return prev_channels * seq_len

    def _build_conv_blocks(self) -> nn.ModuleList:
        """构建卷积块：Conv → BN → 激活 → 池化（适配1D/2D）"""
        conv_blocks = nn.ModuleList()
        prev_channels = self.in_channels
        pool_layer = self.pool_layer_cls(
            kernel_size=self.pool_kernel_size,
            stride=self.pool_stride,
            padding=0
        )

        for curr_channels in self.conv_channels:
            conv_block = nn.Sequential(
                self.conv_layer_cls(
                    in_channels=prev_channels,
                    out_channels=curr_channels,
                    kernel_size=self.conv_kernel_size,
                    stride=self.conv_stride,
                    padding=self.conv_padding,
                    bias=False  # BN层包含偏置，无需卷积层偏置
                ),
                self.bn_layer_cls(curr_channels),  # 批量归一化
                self.activation,
                pool_layer  # 池化层下采样
            )
            conv_blocks.append(conv_block)
            prev_channels = curr_channels

        return conv_blocks

    def _build_fc_layers(self) -> nn.Sequential:
        """构建全连接层（适配分类/回归任务）"""
        fc_layers = []
        fc_input_dim = self._calc_fc_input_dim()
        prev_dim = fc_input_dim

        # 构建全连接隐藏层
        for hidden_dim in self.fc_hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                self.dropout  # Dropout正则化
            ])
            prev_dim = hidden_dim

        # 构建任务输出层
        if self.task_type == "classification":
            # 多分类输出层：无激活（输出logits，交给CrossEntropyLoss）
            fc_layers.append(nn.Linear(prev_dim, self.num_classes))
        elif self.task_type == "regression":
            # 回归输出层：无激活（直接输出连续值，交给MSELoss）
            fc_layers.append(nn.Linear(prev_dim, self.num_outputs))

        return nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：适配图像/时序输入，分类/回归任务
        - 图像输入shape：(B, C, H, W)
        - 时序输入shape：(B, C, L)
        """
        # 1. 输入尺寸/维度校验
        if self.input_type == "image":
            if x.ndim != 4:
                raise ValueError(f"图像输入必须是4维 (B,C,H,W)，当前维度：{x.ndim}")
            if x.shape[2:] != self.input_size:
                raise ValueError(f"图像尺寸不匹配！配置={self.input_size}，实际={x.shape[2:]}")
        elif self.input_type == "timeseries":
            if x.ndim != 3:
                raise ValueError(f"时序输入必须是3维 (B,C,L)，当前维度：{x.ndim}")
            if x.shape[2] != self.input_size[0]:
                raise ValueError(f"时序长度不匹配！配置={self.input_size[0]}，实际={x.shape[2]}")

        # 2. 卷积块前向（特征提取）
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # 3. 展平（适配1D/2D特征）
        x = x.flatten(1)  # 保留batch维度，展平后续所有维度

        # 4. 全连接层前向（任务输出）
        x = self.fc_layers(x)

        return x