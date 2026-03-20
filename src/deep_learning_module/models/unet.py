import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from src.config.deep_learning_module.models.unet import UNetConfig


class UNet(nn.Module):
    """
    U-Net 工厂接口：根据输入配置自动选择 2D U-Net 或 1D U-Net
    
    核心设计：
    - 2D U-Net：用于图像数据，输入格式 (B, C, H, W)
    - 1D U-Net：用于时序数据，输入格式 (B, C, T) 或 (B, T, feat_dim)
    
    自动路由逻辑：
    - 如果 support_timeseries=True，使用 1D U-Net
    - 否则使用标准的 2D U-Net
    """
    def __new__(cls, config: "UNetConfig"):
        if getattr(config, "support_timeseries", False):
            return UNet1D(config)
        else:
            return UNet2D(config)


class UNet2D(nn.Module):
    """
    标准 2D U-Net 网络架构：对称的编码器-解码器结构，具备完整上下采样和跳跃连接
    
    适用于图像数据或 reshape 后的数据
    
    核心特性：
    1. 真正的U-Net架构：通过最大池化下采样，通过转置卷积上采样
    2. 完整的跳跃连接（Skip Connections）：编码器特征与解码器特征拼接融合
    3. 双卷积块（Double Convolution）：每个编码/解码步骤包含两个卷积层
    4. 可配置的网络深度和通道数：支持任意深度的U-Net
    5. 支持两种任务类型：
       - 分割任务：输出空间维度与输入相同
       - 多分类任务：通过全局平均池化和全连接层
    
    U-Net架构流程：
    - 编码阶段（收缩路径）：输入 → Conv2 → MaxPool → Conv2 → MaxPool → ...
    - 最深层：Conv2（最低分辨率）
    - 解码阶段（扩展路径）：UpConv → Concatenate(跳跃特征) → Conv2 → UpConv → ...
    - 最终输出：根据任务类型调整
    """
    def __init__(self, config: "UNetConfig"):
        super(UNet2D, self).__init__()

        self.cfg = config
        self._parse_config()
        self._validate_config()

        self.dropout = nn.Dropout2d(p=self.dropout_prob) if self.dropout_enable else nn.Identity()

        self.encoders = self._build_encoders()
        self.decoders = self._build_decoders()

        self._build_task_head()

    def _parse_config(self):
        """解析Config配置，提取核心参数"""
        self.in_channels = self.cfg.in_channels
        self.num_classes = self.cfg.num_classes
        self.feature_channels = self.cfg.encoder.feature_channels
        self.pool_size = self.cfg.encoder.pool_size if hasattr(self.cfg.encoder, 'pool_size') else 2
        self.dropout_enable = self.cfg.dropout.enable
        self.dropout_prob = self.cfg.dropout.prob
        
        self.task_type = getattr(self.cfg, "task_type", "segmentation")
        self.regression_output_dim = getattr(self.cfg, "regression_output_dim", 1)
        
        self.input_size = getattr(self.cfg, "input_size", None)
        self.output_size = getattr(self.cfg, "output_size", None)
        
        self.encoder_layers = len(self.feature_channels)

    def _validate_config(self):
        """配置合法性校验"""
        if len(self.feature_channels) == 0:
            raise ValueError("encoder.feature_channels 不能为空，至少配置1个通道数")
        if self.pool_size < 1:
            raise ValueError("encoder.pool_size 必须≥1（推荐2）")
        
        valid_tasks = ["segmentation", "classification", "regression"]
        if self.task_type not in valid_tasks:
            raise ValueError(f"task_type仅支持{valid_tasks}，当前：{self.task_type}")
        
        if self.task_type == "classification" and self.num_classes < 1:
            raise ValueError("分类任务num_classes必须≥1")

    def _build_encoders(self) -> nn.ModuleList:
        """构建 2D 编码器"""
        encoders = nn.ModuleList()
        prev_channels = self.in_channels

        for idx, curr_channels in enumerate(self.feature_channels):
            encoder_block = nn.Sequential(
                nn.Conv2d(prev_channels, curr_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(curr_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(curr_channels, curr_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(curr_channels),
                nn.ReLU(inplace=True),
                self.dropout
            )
            encoders.append(encoder_block)
            prev_channels = curr_channels

        return encoders

    def _build_decoders(self) -> nn.ModuleList:
        """构建 2D 解码器"""
        decoders = nn.ModuleList()
        feature_channels_rev = self.feature_channels[::-1]

        for idx in range(len(feature_channels_rev) - 1):
            current_channels = feature_channels_rev[idx]
            target_channels = feature_channels_rev[idx + 1]
            
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(current_channels, target_channels, kernel_size=2, stride=2, bias=False),
            )
            decoders.append(decoder_block)
            
            conv_block = nn.Sequential(
                nn.Conv2d(target_channels * 2, target_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(target_channels, target_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True),
                self.dropout
            )
            decoders.append(conv_block)

        return decoders

    def _build_task_head(self):
        """构建任务适配头：支持分割/分类/回归"""
        first_channel = self.feature_channels[0]
        
        if self.task_type == "segmentation":
            self.task_head = nn.Conv2d(first_channel, self.num_classes, kernel_size=1, stride=1, padding=0)
            
        elif self.task_type == "classification":
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.task_head = nn.Sequential(
                nn.Linear(first_channel, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_prob if self.dropout_enable else 0.0),
                nn.Linear(256, self.num_classes)
            )
            
        elif self.task_type == "regression":
            self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.task_head = nn.Sequential(
                nn.Linear(first_channel, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_prob if self.dropout_enable else 0.0),
                nn.Linear(128, self.regression_output_dim)
            )

    def _adapt_feature_size(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """自适应插值匹配尺寸"""
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if x.dim() != 4:
            raise ValueError(f"2D U-Net 输入维度应为 4（B, C, H, W），当前：{x.dim()}")
        
        batch_size = x.shape[0]
        input_H, input_W = x.shape[2], x.shape[3]
        
        encoder_feats = []
        x_down = x
        for idx, encoder in enumerate(self.encoders):
            x_down = encoder(x_down)
            encoder_feats.append(x_down)
            if idx < len(self.encoders) - 1:
                x_down = F.max_pool2d(x_down, kernel_size=self.pool_size, stride=self.pool_size)
        
        x_up = encoder_feats[-1]
        
        for idx in range(0, len(self.decoders), 2):
            up_conv = self.decoders[idx]
            conv_block = self.decoders[idx + 1]
            
            x_up = up_conv(x_up)
            
            skip_idx = len(encoder_feats) - 2 - (idx // 2)
            skip_feat = encoder_feats[skip_idx]
            
            x_up = self._adapt_feature_size(x_up, skip_feat.shape[2:])
            x_up = torch.cat([x_up, skip_feat], dim=1)
            x_up = conv_block(x_up)
        
        if self.task_type == "segmentation":
            x_out = self.task_head(x_up)
            x_out = self._adapt_feature_size(x_out, (input_H, input_W))
            
        elif self.task_type == "classification":
            x_pool = self.global_avg_pool(x_up)
            x_flat = x_pool.flatten(1)
            x_out = self.task_head(x_flat)
            
        elif self.task_type == "regression":
            x_pool = self.global_avg_pool(x_up)
            x_flat = x_pool.flatten(1)
            x_out = self.task_head(x_flat)
        
        return x_out


class UNet1D(nn.Module):
    """
    1D U-Net 网络架构：针对时序数据优化
    
    核心设计理念：
    - 完全使用 1D 卷积替代 2D 卷积
    - 保留 U-Net 核心架构：编码器 - 解码器 + 跳跃连接
    - 池化仅作用于时间维度（不会出现维度为 0 的问题）
    - 自然适配时序数据的 1D 结构
    
    输入格式：
    - (B, C, T)：(batch_size, feature_dim, time_steps)
    - 也支持 (B, T, C) 格式，会自动转置
    
    适用场景：
    - 时序分类
    - 时序分割
    - 时序回归
    """
    def __init__(self, config: "UNetConfig"):
        super(UNet1D, self).__init__()

        self.cfg = config
        self._parse_config()
        self._validate_config()

        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_enable else nn.Identity()

        self.encoders = self._build_encoders()
        self.decoders = self._build_decoders()

        self._build_task_head()

    def _parse_config(self):
        """解析配置"""
        self.in_channels = self.cfg.in_channels
        self.num_classes = self.cfg.num_classes
        self.feature_channels = self.cfg.encoder.feature_channels
        self.pool_size = self.cfg.encoder.pool_size if hasattr(self.cfg.encoder, 'pool_size') else 2
        self.dropout_enable = self.cfg.dropout.enable
        self.dropout_prob = self.cfg.dropout.prob
        
        self.task_type = getattr(self.cfg, "task_type", "segmentation")
        self.regression_output_dim = getattr(self.cfg, "regression_output_dim", 1)
        
        self.encoder_layers = len(self.feature_channels)

    def _validate_config(self):
        """配置合法性校验"""
        if len(self.feature_channels) == 0:
            raise ValueError("encoder.feature_channels 不能为空")
        if self.pool_size < 1:
            raise ValueError("encoder.pool_size 必须≥1")
        
        valid_tasks = ["segmentation", "classification", "regression"]
        if self.task_type not in valid_tasks:
            raise ValueError(f"task_type仅支持{valid_tasks}")

    def _build_encoders(self) -> nn.ModuleList:
        """构建 1D 编码器"""
        encoders = nn.ModuleList()
        prev_channels = self.in_channels

        for curr_channels in self.feature_channels:
            encoder_block = nn.Sequential(
                nn.Conv1d(prev_channels, curr_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(curr_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(curr_channels, curr_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(curr_channels),
                nn.ReLU(inplace=True),
                self.dropout
            )
            encoders.append(encoder_block)
            prev_channels = curr_channels

        return encoders

    def _build_decoders(self) -> nn.ModuleList:
        """构建 1D 解码器"""
        decoders = nn.ModuleList()
        feature_channels_rev = self.feature_channels[::-1]

        for idx in range(len(feature_channels_rev) - 1):
            current_channels = feature_channels_rev[idx]
            target_channels = feature_channels_rev[idx + 1]
            
            decoder_block = nn.Sequential(
                nn.ConvTranspose1d(current_channels, target_channels, kernel_size=2, stride=2, bias=False),
            )
            decoders.append(decoder_block)
            
            conv_block = nn.Sequential(
                nn.Conv1d(target_channels * 2, target_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(target_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(target_channels, target_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(target_channels),
                nn.ReLU(inplace=True),
                self.dropout
            )
            decoders.append(conv_block)

        return decoders

    def _build_task_head(self):
        """构建任务适配头"""
        first_channel = self.feature_channels[0]
        
        if self.task_type == "segmentation":
            self.task_head = nn.Conv1d(first_channel, self.num_classes, kernel_size=1, stride=1, padding=0)
            
        elif self.task_type == "classification":
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.task_head = nn.Sequential(
                nn.Linear(first_channel, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_prob if self.dropout_enable else 0.0),
                nn.Linear(256, self.num_classes)
            )
            
        elif self.task_type == "regression":
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.task_head = nn.Sequential(
                nn.Linear(first_channel, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_prob if self.dropout_enable else 0.0),
                nn.Linear(128, self.regression_output_dim)
            )

    def _adapt_length(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """自适应插值匹配时间长度"""
        if x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        return x

    def _process_timeseries_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理时序输入，确保格式为 (B, C, T)
        支持输入格式：
        - (B, C, T)：直接使用
        - (B, T, C)：自动转置为 (B, C, T)
        """
        if x.dim() == 3:
            if x.shape[2] == self.in_channels and x.shape[1] > self.in_channels:
                x = x.transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
               - 时序格式1：(B, C, T)
               - 时序格式2：(B, T, C)
        """
        if x.dim() == 3:
            x = self._process_timeseries_input(x)
        
        if x.dim() != 3:
            raise ValueError(f"1D U-Net 输入维度应为 3（B, C, T），当前：{x.dim()}")
        
        batch_size = x.shape[0]
        input_T = x.shape[2]
        
        encoder_feats = []
        x_down = x
        for idx, encoder in enumerate(self.encoders):
            x_down = encoder(x_down)
            encoder_feats.append(x_down)
            if idx < len(self.encoders) - 1:
                x_down = F.max_pool1d(x_down, kernel_size=self.pool_size, stride=self.pool_size)
        
        x_up = encoder_feats[-1]
        
        for idx in range(0, len(self.decoders), 2):
            up_conv = self.decoders[idx]
            conv_block = self.decoders[idx + 1]
            
            x_up = up_conv(x_up)
            
            skip_idx = len(encoder_feats) - 2 - (idx // 2)
            skip_feat = encoder_feats[skip_idx]
            
            x_up = self._adapt_length(x_up, skip_feat.shape[2])
            x_up = torch.cat([x_up, skip_feat], dim=1)
            x_up = conv_block(x_up)
        
        if self.task_type == "segmentation":
            x_out = self.task_head(x_up)
            x_out = self._adapt_length(x_out, input_T)
            
        elif self.task_type == "classification":
            x_pool = self.global_avg_pool(x_up)
            x_flat = x_pool.flatten(1)
            x_out = self.task_head(x_flat)
            
        elif self.task_type == "regression":
            x_pool = self.global_avg_pool(x_up)
            x_flat = x_pool.flatten(1)
            x_out = self.task_head(x_flat)
        
        return x_out

# 向后兼容别名
SimpleCNN = UNet
