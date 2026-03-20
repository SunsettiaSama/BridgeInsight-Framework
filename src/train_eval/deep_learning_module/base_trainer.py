from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
import os
import time
import json  # 新增：用于保存JSON信息
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score, top_k_accuracy_score,
    hamming_loss, jaccard_score
)

# 导入最新的训练器配置基类
from src.deep_learning_module.configs.base_config import BaseConfig
import torch.nn.functional as F  # 新增：用于FocalLoss计算
if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset

# --------------------------
# 自定义损失函数（如FocalLoss，PyTorch无内置）
# --------------------------
# 新增：自定义FocalLoss类（适配分类任务，支持多分类/二分类）
class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 多分类场景：inputs.shape [B, C, ...], targets.shape [B, ...]（类别索引）
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # [B, C, H*W]
            inputs = inputs.transpose(1, 2)  # [B, H*W, C]
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # [B*H*W, C]
            targets = targets.view(-1)  # [B*H*W]

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        # 计算置信度（对应目标类别的softmax概率）
        pt = torch.exp(-ce_loss)
        # Focal Loss核心公式：FL = -α*(1-pt)^γ * ce_loss（此处默认α=1，可按需扩展）
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 损失聚合
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss

class TrainStepState:
    """
    训练步状态记录类，统一管理训练单步中的所有关键参数与状态信息
    涵盖：基础定位、损失性能、优化器梯度、资源效率、特殊场景五大维度
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        # 新增：日志实例，优先使用传入的logger，否则使用默认根logger
        self.logger = logger or logging.getLogger(__name__)
        
        # ==================== 一、基础定位标识（必记） ====================
        self.global_step: int = 0          # 全局唯一步数（核心索引）
        self.epoch: int = 0                # 当前训练轮次
        self.batch_idx: int = 0            # 当前轮次内的批次索引
        self.is_resumed: bool = False      # 是否从断点续训

        # ==================== 二、损失与模型性能（核心监控） ====================
        self.batch_loss: float = 0.0       # 当前批次原始损失值
        self.normalized_loss: float = 0.0  # 梯度累积归一化后的损失值（实际用于反向传播的损失）
        self.running_loss: float = 0.0     # 累计平均损失（如最近N个批次的均值）
        self.running_loss_steps: int = 0   # 累计损失的步数（用于计算均值）
        self.batch_metrics: Dict[str, float] = {}  # 当前批次评估指标（acc/f1/mse等）
        self.loss_components: Dict[str, float] = {}  # 损失分解项（多任务/正则化损失等）

        # ==================== 三、优化器与梯度状态（调优依据） ====================
        self.learning_rates: Dict[str, float] = {}  # 学习率（支持分层学习率，key为参数组名称）
        self.gradient_norm: float = 0.0             # 梯度范数（全局梯度的L2范数）
        self.is_gradient_clipped: bool = False      # 当前步是否触发梯度裁剪
        self.accumulation_step: int = 1             # 梯度累积当前步数（如1/4表示第1步，共4步累积）
        self.total_accumulation_steps: int = 1      # 总梯度累积步数
        self.optimizer_momentum: Optional[float] = None  # 优化器动量值（SGD/Adam等）
        self.scaler_scale: Optional[float] = None   # 混合精度缩放因子（AMP场景）

        # ==================== 四、计算资源与效率（性能优化） ====================
        self.step_start_time: float = 0.0           # 当前步开始时间戳
        self.step_total_time: float = 0.0           # 当前步总耗时（秒）
        self.data_loading_time: float = 0.0         # 数据加载耗时（秒）
        self.forward_time: float = 0.0              # 模型前向传播耗时（秒）
        self.backward_time: float = 0.0             # 模型反向传播耗时（秒）
        self.optimizer_update_time: float = 0.0     # 优化器参数更新耗时（秒）
        self.gpu_memory_used: float = 0.0           # GPU显存占用（MB）
        self.gpu_utilization: float = 0.0           # GPU利用率（%）
        self.cpu_utilization: float = 0.0           # CPU利用率（%）

        # ==================== 五、特殊场景适配参数（按需记录） ====================
        self.weight_decay_applied: float = 0.0      # 实际生效的权重衰减值
        self.dropout_rate: Optional[float] = None   # 当前步dropout概率（动态调整场景）
        self.rank: int = 0                          # 分布式训练进程ID（单卡默认为0）
        self.sync_time: float = 0.0                 # 多卡梯度同步耗时（分布式场景，秒）
        self.batch_size_actual: int = 0             # 当前批次实际样本数（可能不足配置的batch_size）
        self.is_augmented: bool = False             # 当前批次是否启用数据增强

    def update(self, **kwargs) -> None:
        """
        灵活更新状态参数，支持关键字参数批量传入
        示例：state.update(global_step=100, batch_loss=0.5, learning_rates={"base": 1e-3})
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 替换print为logger.warning
                self.logger.warning(f"警告：TrainStepState中不存在属性 '{key}'，跳过该参数更新")

    def update_running_loss(self, loss_value: float, step_count: int = 1) -> None:
        """
        更新累计平均损失（滑动平均/累计均值）
        Args:
            loss_value: 待累积的损失值
            step_count: 本次累积的步数（默认1步）
        """
        self.running_loss = (self.running_loss * self.running_loss_steps + loss_value * step_count) / (self.running_loss_steps + step_count)
        self.running_loss_steps += step_count

    def start_step_timer(self) -> None:
        """记录当前训练步的开始时间（用于计算单步耗时）"""
        self.step_start_time = time.time()

    def end_step_timer(self) -> None:
        """计算当前训练步的总耗时（从start_step_timer调用开始）"""
        self.step_total_time = time.time() - self.step_start_time

    def reset_temp_params(self) -> None:
        """
        重置临时参数（每训练步结束后调用，保留持久化状态，清空临时状态）
        如：批次损失、单步耗时、梯度状态等临时参数
        """
        # 临时损失参数
        self.batch_loss = 0.0
        self.normalized_loss = 0.0
        self.data_loading_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optimizer_update_time = 0.0
        # 临时梯度/优化器参数
        self.gradient_norm = 0.0
        self.is_gradient_clipped = False
        self.accumulation_step = 1
        # 临时数据/增强参数
        self.batch_size_actual = 0
        self.is_augmented = False
        # 临时指标参数
        self.batch_metrics.clear()
        self.loss_components.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        导出当前状态为字典格式（便于保存日志、写入TensorBoard或断点）
        Returns:
            包含所有状态参数的字典
        """
        state_dict = {}
        for attr_name in dir(self):
            # 过滤私有属性和方法，仅保留实例变量（排除logger，避免JSON序列化问题）
            if not attr_name.startswith("_") and not callable(getattr(self, attr_name)) and attr_name != "logger":
                state_dict[attr_name] = getattr(self, attr_name)
        return state_dict

    def print_step_summary(self) -> None:
        """打印当前训练步的核心状态摘要（替换为logger.info输出）"""
        summary = f"""
        ==================== 训练步状态摘要 [Epoch: {self.epoch+1} | Batch: {self.batch_idx+1} | Global Step: {self.global_step}] ====================
        1. 性能指标：批次损失={self.batch_loss:.4f} | 累计平均损失={self.running_loss:.4f} | 批次指标={self.batch_metrics}
        2. 优化器状态：学习率={list(self.learning_rates.values())[0] if self.learning_rates else 0:.6f} | 梯度范数={self.gradient_norm:.4f} | 梯度裁剪={self.is_gradient_clipped}
        3. 训练配置：梯度累积={self.accumulation_step}/{self.total_accumulation_steps} | 续训={self.is_resumed} | 实际批次大小={self.batch_size_actual}
        4. 资源效率：单步耗时={self.step_total_time:.2f}s | GPU显存={self.gpu_memory_used:.0f}MB | GPU利用率={self.gpu_utilization:.1f}%
        """
        # 替换print为logger.info，保持原有格式
        self.logger.info(summary.strip())

    def reset_running_loss(self) -> None:
        """重置累计平均损失（如每轮结束后或手动触发滑动窗口重置）"""
        self.running_loss = 0.0
        self.running_loss_steps = 0


class BaseTrainer(ABC):
    """
    基于最新BaseConfig的训练器抽象基类，定义统一训练接口与通用功能
    核心特性：
    1.  接收并封装BaseConfig配置实例，替代原有全局配置，实现配置与训练器解耦；
    2.  兼容单卡/多卡分布式、混合精度、梯度累积、断点续训等通用场景；
    3.  封装通用功能（模型保存/加载、日志初始化、指标更新等），减少子类重复代码；
    4.  定义抽象方法，强制子类实现任务专属逻辑（数据加载、模型初始化、训练/验证步骤等）；
    5.  与配置类参数联动，自动适配训练配置，无需手动硬编码；
    6.  新增_get_loss和_get_optimizer方法，统一管理损失函数与优化器实例化；
    7.  集成TrainStepState管理训练步状态，支持模型结构+训练状态的JSON保存。
    """
    def __init__(self, trainer_config: BaseConfig):
        # 核心配置实例（最新BaseConfig）
        self.config: BaseConfig = trainer_config
        # 初始化基础属性
        self.epoch: int = 0
        self.global_step: int = 0
        self.best_metric: float = -float("inf") if "acc" in self.config.best_model_metric.lower() else float("inf")
        self.best_epoch: int = 0

        # 日志组件初始化（先初始化logger，再传给step_state）
        self.logger: logging.Logger = self._init_logger()

        # 新增：训练步状态实例（传入训练器的logger，实现日志统一输出）
        self.step_state: TrainStepState = TrainStepState(logger=self.logger)

        # 核心组件（子类通过_init_*方法初始化后赋值）
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[LRScheduler] = None
        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None

        # 损失函数（通过_get_loss方法获取）
        self.criterion: Optional[nn.Module] = None

        # 设备配置
        self.device: Union[torch.device, List[torch.device]] = self._init_device()

        # 混合精度训练组件
        self.scaler: Optional[GradScaler] = None
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()

        # 输出目录初始化（确保目录存在）
        self._init_output_dirs()

        # 断点续训（若配置指定）
        if self.config.resume_from_checkpoint is not None:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        # 初始化核心组件（子类实现具体逻辑）
        # 兼容延迟传入
        # self._init_dataloaders()
        # self._init_model()
        # 初始化优化器和调度器（子类可调用_get_optimizer获取优化器）
        # self._init_optimizer_scheduler()
        
        # 初始化损失函数（新增：统一通过基类方法获取）
        self.criterion = self._get_loss()
        self.logger.info(f"损失函数初始化完成：{self.config.loss_type if hasattr(self.config, 'loss_type') else '默认损失'}")

        # 打印配置信息
        self.logger.info("="*50)
        self.logger.info("训练器初始化完成，核心配置信息：")
        self.logger.info(f"训练轮数：{self.config.epochs} | 批次大小：{self.config.batch_size}")
        self.logger.info(f"初始学习率：{self.config.learning_rate} | 优化器：{self.config.optimizer}")
        self.logger.info(f"训练设备：{self.device} | 分布式训练：{self.config.use_distributed}")
        self.logger.info(f"混合精度：{self.config.use_mixed_precision}（精度类型：{self.config.mixed_precision_type}）")
        self.logger.info(f"输出目录：{self.config.output_dir} | 最优指标：{self.config.best_model_metric}")
        self.logger.info(f"损失函数：{self.criterion.__class__.__name__}")
        self.logger.info("="*50)


    # --------------------------
    # 新增：更新训练步状态（封装TrainStepState）
    # --------------------------
    def update_step_state(self, **kwargs) -> None:
        """
        简化子类对训练步状态的更新，直接传递关键字参数即可
        示例：self.update_step_state(global_step=100, batch_loss=0.5)
        """
        self.step_state.update(**kwargs)

    # --------------------------
    # 新增：获取模型结构与尺寸信息
    # --------------------------
    def _get_model_structure_info(self) -> Dict[str, Any]:
        """
        获取模型的层级结构、参数尺寸、参数总量等信息（用于JSON保存）
        Returns:
            包含模型详情的字典（可直接JSON序列化）
        """
        if self.model is None:
            self.logger.warning("模型尚未初始化，无法获取结构信息")
            return {"status": "error", "message": "模型未初始化"}
        
        # 处理多卡包装的模型（如DataParallel/DistributedDataParallel）
        model = self.model.module if hasattr(self.model, "module") else self.model
        structure_info = {
            "model_class": model.__class__.__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "layers": []
        }

        # 遍历模型各层，记录名称、类型、参数形状
        for layer_name, layer in model.named_children():
            layer_detail = {
                "layer_name": layer_name,
                "layer_type": layer.__class__.__name__,
                "parameters": {
                    param_name: list(param.shape)  # 转列表以支持JSON序列化
                    for param_name, param in layer.named_parameters()
                }
            }
            structure_info["layers"].append(layer_detail)
        
        return structure_info

    # --------------------------
    # 新增：通用损失函数获取方法
    # --------------------------
    def _get_loss(self) -> nn.Module:
        """
        从config中获取配置，统一实例化损失函数
        Returns:
            实例化后的损失函数对象
        """
        # 检查config是否包含loss_type属性（子类配置可能扩展）
        if not hasattr(self.config, "loss_type"):
            # 默认返回交叉熵损失（适配通用分类场景）
            self.logger.warning("配置中未指定loss_type，默认使用CrossEntropyLoss")
            return nn.CrossEntropyLoss()

        loss_type = self.config.loss_type
        try:
            # 标准损失函数实例化
            if loss_type == "CrossEntropyLoss":
                return nn.CrossEntropyLoss()
            elif loss_type == "MSELoss":
                return nn.MSELoss()
            elif loss_type == "L1Loss":
                return nn.L1Loss()
            elif loss_type == "BCELoss":
                return nn.BCELoss()
            elif loss_type == "BCEWithLogitsLoss":
                return nn.BCEWithLogitsLoss()
            # 自定义损失函数实例化
            elif loss_type == "FocalLoss":
                # 获取子类配置中的自定义参数（如SFTTrainerConfig的focal_gamma、num_classes）
                focal_gamma = getattr(self.config, "focal_gamma", 2.0)
                num_classes = getattr(self.config, "num_classes", 2)
                return FocalLoss(
                    gamma=focal_gamma,
                    num_classes=num_classes,
                    reduction="mean"
                )
            else:
                raise ValueError(f"不支持的损失函数类型：{loss_type}")
        except Exception as e:
            self.logger.error(f"损失函数实例化失败：{str(e)}")
            raise e

    # --------------------------
    # 新增：通用优化器获取方法
    # --------------------------
    def _get_optimizer(self, model_params) -> Optimizer:
        """
        从config中获取配置，统一实例化优化器
        Args:
            model_params: 模型可训练参数（如self.model.parameters()或分层参数组）
        Returns:
            实例化后的优化器对象
        """
        optimizer_type = self.config.optimizer
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        optimizer_extra_params = self.config.optimizer_params or {}

        try:
            # 实例化对应优化器
            if optimizer_type == "Adam":
                return Adam(
                    model_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    **optimizer_extra_params
                )
            elif optimizer_type == "AdamW":
                return AdamW(
                    model_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    **optimizer_extra_params
                )
            elif optimizer_type == "SGD":
                # 默认添加动量参数，可通过optimizer_extra_params覆盖
                default_params = {"momentum": 0.9, "nesterov": True}
                default_params.update(optimizer_extra_params)
                return SGD(
                    model_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    **default_params
                )
            elif optimizer_type == "RMSprop":
                return RMSprop(
                    model_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    **optimizer_extra_params
                )
            else:
                raise ValueError(f"不支持的优化器类型：{optimizer_type}")
        except Exception as e:
            self.logger.error(f"优化器实例化失败：{str(e)}")
            raise e

    # --------------------------
    # 通用初始化方法（基类实现，子类无需重写）
    # --------------------------
    def _init_device(self) -> Union[torch.device, List[torch.device]]:
        """初始化训练设备（兼容单卡/多卡/CPU）"""
        device_config = self.config.device
        if isinstance(device_config, str):
            if device_config.startswith("cuda") and not torch.cuda.is_available():
                self.logger.warning("CUDA不可用，自动切换为CPU设备")
                return torch.device("cpu")
            return torch.device(device_config)
        elif isinstance(device_config, list):
            devices = [torch.device(dev) for dev in device_config]
            for dev in devices:
                if dev.type == "cuda" and not torch.cuda.is_available():
                    self.logger.warning("部分CUDA设备不可用，自动切换为CPU设备")
                    return torch.device("cpu")
            return devices
        else:
            raise ValueError(f"无效的设备配置类型：{type(device_config)}")

    def _init_logger(self) -> logging.Logger:
        """初始化日志组件（兼容终端输出与文件保存）"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 终端处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 文件处理器（若配置开启）
        if self.config.save_log_file:
            log_file_path = Path(self.config.output_dir) / "train_log.txt"
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _init_output_dirs(self) -> None:
        """初始化输出目录（模型、日志、TensorBoard等）"""
        # 主输出目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard日志目录
        if self.config.use_tensorboard:
            tb_log_dir = Path(self.config.tensorboard_log_dir)
            tb_log_dir.mkdir(parents=True, exist_ok=True)

        # 检查点保存目录（可选，默认在主输出目录）
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # 抽象方法（子类必须实现）
    # --------------------------
    @abstractmethod
    def _init_dataloaders(self) -> None:
        """初始化训练/验证DataLoader，需赋值self.train_dataloader和self.val_dataloader"""
        pass

    @abstractmethod
    def _init_model(self) -> None:
        """初始化模型，需赋值self.model，并完成模型设备迁移"""
        pass

    @abstractmethod
    def _init_optimizer_scheduler(self) -> None:
        """初始化优化器和学习率调度器，需赋值self.optimizer和self.scheduler"""
        # 子类可通过以下方式获取优化器（示例）：
        # if self.model is not None:
        #     self.optimizer = self._get_optimizer(self.model.parameters())
        pass

    @abstractmethod
    def train_step(self, batch_data: Any) -> Dict[str, float]:
        """
        单步训练逻辑（单个batch）
        Args:
            batch_data: 从train_dataloader获取的批次数据
        Returns:
            训练指标字典（如{"train_loss": 0.5, "train_acc": 0.9}）
        注意：子类实现时需通过self.update_step_state更新训练步状态（如global_step、batch_loss等）
        """
        pass

    @abstractmethod
    def val_step(self, batch_data: Any) -> Dict[str, float]:
        """
        单步验证逻辑（单个batch）
        Args:
            batch_data: 从val_dataloader获取的批次数据
        Returns:
            验证指标字典（如{"val_loss": 0.4, "val_acc": 0.92}）
        """
        pass

    @abstractmethod
    def train(self,) -> None:
        """完整训练流程（包含epoch循环、训练/验证、指标更新、模型保存等）"""
        pass

    # --------------------------
    # 通用方法（基类实现，子类可按需重写）
    # --------------------------
    def save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None) -> None:
        """
        保存模型断点+模型结构+训练步状态（支持PTH+JSON双文件保存）
        Args:
            is_best: 是否为最优模型
            epoch: 当前训练轮数（默认使用self.epoch）
        """
        current_epoch = epoch or self.epoch
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"

        # 构建断点字典
        checkpoint = {
            "epoch": current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "criterion_state_dict": self.criterion.state_dict() if hasattr(self.criterion, "state_dict") else None,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "config": self.config.dict()
        }

        # 辅助函数：保存模型结构+训练步状态到JSON
        def save_info_json(pth_path: Path):
            model_info = self._get_model_structure_info()
            step_state_dict = self.step_state.to_dict()
            json_path = pth_path.with_suffix(".json")
            info_dict = {
                "metadata": {
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "checkpoint_path": str(pth_path)
                },
                "model_structure": model_info,
                "train_step_state": step_state_dict
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info_dict, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已保存断点信息至：{json_path}")

        # 保存最新断点+JSON
        latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_checkpoint_path)
        self.logger.info(f"已保存最新断点至：{latest_checkpoint_path}")
        save_info_json(latest_checkpoint_path)

        # 保存最优断点+JSON
        if is_best:
            best_checkpoint_path = checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"已保存最优断点至：{best_checkpoint_path}（最优指标：{self.best_metric:.4f}，第{current_epoch}轮）")
            save_info_json(best_checkpoint_path)

        # 保存中间断点+JSON
        if self.config.save_freq > 0 and current_epoch % self.config.save_freq == 0:
            intermediate_checkpoint_path = checkpoint_dir / f"epoch_{current_epoch}_checkpoint.pth"
            torch.save(checkpoint, intermediate_checkpoint_path)
            self.logger.info(f"已保存中间断点至：{intermediate_checkpoint_path}")
            save_info_json(intermediate_checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载模型断点+恢复训练步状态
        Args:
            checkpoint_path: 断点文件路径
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"断点文件不存在：{checkpoint_path}")

        # 加载断点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 恢复模型状态
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.logger.warning("模型尚未初始化，跳过模型状态恢复")

        # 恢复损失函数状态（若存在）
        if self.criterion is not None and "criterion_state_dict" in checkpoint and checkpoint["criterion_state_dict"] is not None:
            if hasattr(self.criterion, "load_state_dict"):
                self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
        else:
            self.logger.warning("损失函数尚未初始化或断点中无损失函数状态，跳过损失函数状态恢复")

        # 恢复优化器状态
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self.logger.warning("优化器尚未初始化或断点中无优化器状态，跳过优化器状态恢复")

        # 恢复调度器状态
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            self.logger.warning("调度器尚未初始化、断点中无调度器状态或未使用调度器，跳过调度器状态恢复")

        # 恢复混合精度缩放器状态
        if self.scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        elif self.config.use_mixed_precision:
            self.logger.warning("混合精度缩放器尚未初始化或断点中无缩放器状态，跳过缩放器状态恢复")

        # 恢复训练状态
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.best_epoch = checkpoint.get("best_epoch", 0)

        # 新增：恢复训练步状态
        self.step_state.update(
            epoch=self.epoch,
            global_step=self.global_step,
            is_resumed=True
        )

        self.logger.info(f"成功加载断点：{checkpoint_path}")
        self.logger.info(f"恢复训练状态：第{self.epoch}轮 | 全局步数{self.global_step} | 最优指标{self.best_metric:.4f}（第{self.best_epoch}轮）")

    def update_best_metric(self, current_metric: float) -> bool:
        """
        根据当前指标更新最优指标，返回是否为最优模型
        Args:
            current_metric: 当前验证指标值
        Returns:
            是否为最优模型（True/False）
        """
        # 判断指标优劣（分类任务指标越大越好，回归任务指标越小越好）
        is_better = False
        if "acc" in self.config.best_model_metric.lower() or "f1" in self.config.best_model_metric.lower() or "auc" in self.config.best_model_metric.lower():
            # 越大越好的指标
            if current_metric > self.best_metric:
                is_better = True
                self.best_metric = current_metric
                self.best_epoch = self.epoch
        else:
            # 越小越好的指标（loss/mse/mae等）
            if current_metric < self.best_metric:
                is_better = True
                self.best_metric = current_metric
                self.best_epoch = self.epoch

        return is_better

    def close(self) -> None:
        """训练结束后清理资源（子类可重写）"""
        self.logger.info("训练流程结束，开始清理资源...")
        # 关闭日志处理器
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        # 释放CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("资源清理完成")