from abc import ABC
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR, OneCycleLR
)
from torch.utils.data import DataLoader, Dataset  # 导入Dataset类
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score, top_k_accuracy_score,
    hamming_loss, jaccard_score
)
import torch.distributed as dist
from contextlib import nullcontext

# 导入基类和SFT配置类
from .base_trainer import BaseTrainer, FocalLoss
from ..configs.trainer.sft import SFTTrainerConfig

class SFTTrainer(BaseTrainer):
    """
    传统深度学习模型SFT（监督微调）专属训练器（适配CNN/LSTM/MLP等模型，非LLM）
    核心特性：
    1.  适配多任务类型：单标签分类/多标签分类/回归/时序分类/时序回归，自动匹配损失与指标；
    2.  支持分层训练：特征提取层与预测头不同学习率，支持排除偏置项权重衰减；
    3.  完善的模型冻结：支持整体冻结特征提取层或精细化冻结指定层前缀（兼容DDP模型）；
    4.  预训练权重加载：支持自定义预测头前缀过滤，自动忽略不匹配参数，兼容断点权重；
    5.  兼容混合精度/梯度累积/分布式训练，参数联动更严谨，容错性更强；
    6.  精细化指标监控：分类任务支持多标签适配，回归任务支持自定义评估指标；
    7.  支持多种传入模式：初始化/训练时传入Model/Dataset/DataLoader，自动转换Dataset为DataLoader；
    8.  增强调试模式：支持灵活控制训练/验证断点，TensorBoard记录更丰富。
    """
    def __init__(
        self,
        config: SFTTrainerConfig,
        model: Optional[nn.Module] = None,  # 改为可选参数
        train_dataloader: Optional[Union[DataLoader, Dataset]] = None,  # 支持Dataset类型
        val_dataloader: Optional[Union[DataLoader, Dataset]] = None,  # 支持Dataset类型
    ):
        """
        初始化SFT训练器（支持多种传入模式，兼容Dataset自动转换）
        Args:
            config: SFT专属训练配置
            model: 可选，外界传入的待训练模型（已定义好结构），可在train时补充传入
            train_dataloader: 可选，外界传入的训练数据加载器/数据集，可在train时补充传入
            val_dataloader: 可选，外界传入的验证数据加载器/数据集，可在train时补充传入
        """
        # 初始化基类（包含StepState、日志、设备等通用配置）
        super().__init__(config)

        # 类型断言，确保配置为SFT专属配置
        self.sft_config: SFTTrainerConfig = self.config  # type: ignore

        # 可选属性，支持延迟绑定（支持Dataset类型）
        self.model: Optional[nn.Module] = model
        self.train_dataloader: Optional[Union[DataLoader, Dataset]] = train_dataloader
        self.val_dataloader: Optional[Union[DataLoader, Dataset]] = val_dataloader

        # 初始化TensorBoard SummaryWriter（若开启，模型结构图延迟记录）
        self.writer: Optional[SummaryWriter] = None
        self._init_tensorboard(enable_graph=False)  # 暂时不记录模型结构图

        # 延迟初始化组件
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[object] = None
        self.scaler: Optional[GradScaler] = None
        self.criterion: Optional[nn.Module] = None  # 补充损失函数属性

        # 初始化混合精度scaler
        self._init_scaler()


    # --------------------------
    # 新增：Dataset转DataLoader辅助方法
    # --------------------------
    def _dataset_to_dataloader(
        self,
        dataset: Dataset,
        is_train: bool = True
    ) -> DataLoader:
        """
        将Dataset自动转换为DataLoader（依据配置获取参数）
        Args:
            dataset: 待转换的数据集实例
            is_train: 是否为训练集（训练集开启shuffle，验证集关闭）
        Returns:
            构建完成的DataLoader
        """
        # 从配置中获取DataLoader相关参数
        batch_size = self.sft_config.batch_size
        num_workers = self.sft_config.num_workers if hasattr(self.sft_config, "num_workers") else 0
        pin_memory = self.sft_config.pin_memory if hasattr(self.sft_config, "pin_memory") else True
        shuffle = self.sft_config.shuffle if hasattr(self.sft_config, "shuffle") else is_train
        drop_last = self.sft_config.drop_last if hasattr(self.sft_config, "drop_last") else is_train

        # 构建DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

        # 打印转换日志
        dataset_name = dataset.__class__.__name__
        self.logger.info(
            f"已自动将Dataset [{dataset_name}] 转换为DataLoader："
            f"batch_size={batch_size}, shuffle={shuffle}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}"
        )

        return dataloader

    # --------------------------
    # 新增：混合精度scaler初始化
    # --------------------------
    def _init_scaler(self) -> None:
        """初始化混合精度梯度缩放器"""
        if self.sft_config.use_mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("混合精度训练已启用，GradScaler初始化完成")
        else:
            self.scaler = None
    # --------------------------
    # 新增：损失函数初始化
    # --------------------------
    def _init_criterion(self, num_classes: int) -> None:
        """
        根据配置和模型类别数初始化损失函数（延迟初始化，依赖模型输出维度）
        Args:
            num_classes: 从模型提取的类别数（分类任务有效，回归任务可传1）
        """
        loss_type = self.sft_config.loss_type
        # 构建损失函数映射（新增FocalLoss，依赖num_classes）
        loss_map = {
            "CrossEntropyLoss": nn.CrossEntropyLoss(),
            "BCELoss": nn.BCELoss(),
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
            "MSELoss": nn.MSELoss(),
            "L1Loss": nn.L1Loss(),
            "FocalLoss": FocalLoss(num_classes=num_classes, gamma=self.sft_config.focal_gamma)
        }

        if loss_type not in loss_map:
            raise ValueError(f"不支持的损失函数类型：{loss_type}，支持类型：{list(loss_map.keys())}")
        
        self.criterion = loss_map[loss_type]
        self.logger.info(f"损失函数 {loss_type} 初始化完成（分类任务类别数：{num_classes}）")
    
    def _compute_metrics(
        self, 
        preds: np.ndarray, 
        targets: np.ndarray,
        num_classes: int,  # 新增：从外部传递模型提取的类别数
        is_multi_label: bool  # 新增：从外部传递任务是否为多标签
    ) -> Dict[str, float]:
        """
        根据任务类型计算评估指标（支持单/多标签分类、自定义top-k）
        Args:
            preds: 模型预测结果（分类：类别概率/索引；回归：数值）
            targets: 真实标签（分类：类别索引/多标签二进制矩阵；回归：真实数值）
            num_classes: 模型输出类别数（从模型动态提取，替代配置文件冗余配置）
            is_multi_label: 是否为多标签分类任务（从任务配置/模型判断，替代配置文件冗余配置）
        Returns:
            指标字典
        """
        metrics = {}
        # 仅保留必要配置的安全获取，移除num_classes和is_multi_label的配置依赖
        sft_task_type = getattr(self.sft_config, "sft_task_type", "classification")
        metric_names = getattr(self.sft_config, "get_train_evaluation_metrics", lambda: [])()
        top_k = getattr(self.sft_config, "top_k", 5)

        try:
            # 分类类任务（含时序分类，支持单/多标签）
            if sft_task_type in ["classification", "timeseries_classification"]:
                # 处理预测结果格式
                if preds.ndim == 2:
                    if is_multi_label:
                        # 多标签分类：概率转二进制（阈值0.5）
                        preds_binary = (preds >= 0.5).astype(int)
                        preds_argmax = preds_binary
                    else:
                        # 单标签分类：概率转类别索引
                        preds_argmax = np.argmax(preds, axis=1)
                else:
                    preds_argmax = preds.squeeze()
                    targets = targets.squeeze()

                # 计算分类指标（使用传入的num_classes和is_multi_label，不再依赖配置）
                if "accuracy" in metric_names and not is_multi_label:
                    metrics["accuracy"] = float(accuracy_score(targets, preds_argmax))
                if "f1" in metric_names:
                    average = "binary" if (num_classes == 2 and not is_multi_label) else "macro"
                    metrics["f1"] = float(f1_score(targets, preds_argmax, average=average, zero_division=0))
                if "precision" in metric_names:
                    average = "binary" if (num_classes == 2 and not is_multi_label) else "macro"
                    metrics["precision"] = float(precision_score(targets, preds_argmax, average=average, zero_division=0))
                if "recall" in metric_names:
                    average = "binary" if (num_classes == 2 and not is_multi_label) else "macro"
                    metrics["recall"] = float(recall_score(targets, preds_argmax, average=average, zero_division=0))
                if "top_k_accuracy" in metric_names and not is_multi_label and num_classes >= top_k:
                    metrics[f"top_{top_k}_accuracy"] = float(top_k_accuracy_score(targets, preds, k=top_k))
                # 多标签专属指标
                if is_multi_label:
                    if "hamming_loss" in metric_names:
                        metrics["hamming_loss"] = float(hamming_loss(targets, preds_argmax))
                    if "jaccard_score" in metric_names:
                        metrics["jaccard_score"] = float(jaccard_score(targets, preds_argmax, average="macro"))

            # 回归类任务（含时序回归）
            elif sft_task_type in ["regression", "timeseries_regression"]:
                preds_flat = preds.flatten()
                targets_flat = targets.flatten()

                # 过滤无效值（如NaN/Inf）
                valid_mask = np.isfinite(preds_flat) & np.isfinite(targets_flat)
                if not np.any(valid_mask):
                    self.logger.warning("预测结果或真实标签中无有效数值，回归指标计算失败")
                    metrics["loss"] = 0.0
                    return metrics
                preds_flat = preds_flat[valid_mask]
                targets_flat = targets_flat[valid_mask]

                # 计算回归指标
                if "mse" in metric_names:
                    metrics["mse"] = float(mean_squared_error(targets_flat, preds_flat))
                if "mae" in metric_names:
                    metrics["mae"] = float(mean_absolute_error(targets_flat, preds_flat))
                if "rmse" in metric_names:
                    metrics["rmse"] = float(np.sqrt(mean_squared_error(targets_flat, preds_flat)))
                if "r2_score" in metric_names:
                    metrics["r2_score"] = float(r2_score(targets_flat, preds_flat))

            # 损失指标后续填充
            metrics["loss"] = 0.0
            return metrics

        except Exception as e:
            self.logger.error(f"指标计算失败：{str(e)} | 预测形状：{preds.shape} | 标签形状：{targets.shape}")
            raise e
    
    # --------------------------
    # 优化：核心组件完整性校验（支持Dataset自动转换）
    # --------------------------
    def _validate_core_components(
        self,
        model: Optional[nn.Module] = None,
        train_dataloader: Optional[Union[DataLoader, Dataset]] = None,
        val_dataloader: Optional[Union[DataLoader, Dataset]] = None
    ) -> Tuple[nn.Module, DataLoader, DataLoader]:
        """
        校验模型、数据加载器等核心组件是否有效（支持Dataset自动转换为DataLoader）
        Args:
            model: 动态传入的模型（优先级高于实例属性）
            train_dataloader: 动态传入的训练数据加载器/数据集（优先级高于实例属性）
            val_dataloader: 动态传入的验证数据加载器/数据集（优先级高于实例属性）
        Returns:
            有效模型、训练数据加载器、验证数据加载器
        """
        # 优先级：动态传入 > 实例属性
        effective_model = model or self.model
        effective_train_data = train_dataloader or self.train_dataloader
        effective_val_data = val_dataloader or self.val_dataloader
        effective_optimizer = self.optimizer

        # 第一步：校验模型有效性
        if effective_model is None:
            raise RuntimeError("model 未正常提供，请在初始化或train调用时传入有效参数")
        if not isinstance(effective_model, nn.Module):
            raise TypeError(f"model必须是nn.Module实例，当前类型：{type(effective_model)}")

        # 第二步：处理训练数据（自动转换Dataset为DataLoader）
        effective_train_dl: Optional[DataLoader] = None
        if isinstance(effective_train_data, DataLoader):
            effective_train_dl = effective_train_data
        elif isinstance(effective_train_data, Dataset):
            # 转换Dataset为DataLoader（标记为训练集）
            effective_train_dl = self._dataset_to_dataloader(effective_train_data, is_train=True)
        elif effective_train_data is None:
            raise RuntimeError("train_dataloader/train_dataset 未正常提供，请在初始化或train调用时传入有效参数")
        else:
            raise TypeError(
                f"训练数据必须是DataLoader或Dataset实例，当前类型：{type(effective_train_data)}"
            )

        # 第三步：处理验证数据（自动转换Dataset为DataLoader）
        effective_val_dl: Optional[DataLoader] = None
        if isinstance(effective_val_data, DataLoader):
            effective_val_dl = effective_val_data
        elif isinstance(effective_val_data, Dataset):
            # 转换Dataset为DataLoader（标记为验证集）
            effective_val_dl = self._dataset_to_dataloader(effective_val_data, is_train=False)
        elif effective_val_data is None:
            raise RuntimeError("val_dataloader/val_dataset 未正常提供，请在初始化或train调用时传入有效参数")
        else:
            raise TypeError(
                f"验证数据必须是DataLoader或Dataset实例，当前类型：{type(effective_val_data)}"
            )

        # 第四步：校验优化器（未初始化则后续自动创建）
        if effective_optimizer is None:
            self.logger.warning("优化器尚未初始化，将在训练前自动创建")

        # 打印校验通过日志
        self.logger.info("核心组件完整性校验通过（含Dataset自动转换）")
        
        return effective_model, effective_train_dl, effective_val_dl

    # --------------------------
    # 优化：TensorBoard初始化（支持延迟记录模型结构图）
    # --------------------------
    def _init_tensorboard(self, enable_graph: bool = True) -> None:
        """
        初始化TensorBoard，支持延迟记录模型结构图
        Args:
            enable_graph: 是否立即记录模型结构图（若模型和数据未就绪，可后续调用时开启）
        """
        if not self.sft_config.use_tensorboard:
            return

        try:
            self.writer = SummaryWriter(log_dir=self.sft_config.tensorboard_log_dir)
            self.logger.info(f"TensorBoard日志已开启，保存目录：{self.sft_config.tensorboard_log_dir}")

            # 仅当启用且模型和数据就绪时，记录模型结构图
            if enable_graph and self.model is not None and self.train_dataloader is not None:
                self._record_model_graph()
        except Exception as e:
            self.logger.error(f"TensorBoard初始化失败：{str(e)}")
            self.writer = None

    def _record_model_graph(self, model: Optional[nn.Module] = None, dataloader: Optional[DataLoader] = None) -> None:
        """延迟记录模型结构图（支持动态传入模型和数据）"""
        if self.writer is None:
            return
        
        effective_model = model or self.model
        effective_dl = dataloader or self.train_dataloader
        if effective_model is None or effective_dl is None:
            self.logger.warning("模型或数据加载器未就绪，无法记录模型结构图")
            return

        try:
            sample_batch = next(iter(effective_dl))
            sample_inputs = sample_batch[0].to(self.device)
            self.writer.add_graph(effective_model, sample_inputs)
            self.logger.info("模型结构图已写入TensorBoard")
        except Exception as e:
            self.logger.warning(f"记录模型结构图失败：{str(e)}，不影响后续训练")

    # --------------------------
    # 优化：模型初始化（支持动态传入模型，兼容延迟初始化）
    # --------------------------
    def _init_model(self, model: nn.Module) -> nn.Module:
        """
        模型初始化：预训练权重加载 + 层冻结 + 设备迁移（支持动态传入模型）
        Args:
            model: 待初始化的模型（可以是新传入的模型）
        Returns:
            初始化完成的模型
        """
        if model is None:
            raise RuntimeError("模型未提供，无法进行初始化")

        # 1. 加载预训练权重
        self._load_pretrained_weights(model)

        # 2. 冻结模型层
        self._freeze_model_layers(model)

        # 3. 模型设备迁移（支持分布式/多卡/单卡，规范DDP初始化）
        target_model = model
        if isinstance(self.device, list) and len(self.device) > 1:
            if self.sft_config.use_distributed:
                # 规范DDP初始化（需先初始化进程组）
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend="nccl",
                        init_method=f"tcp://127.0.0.1:{self.sft_config.dist_port}",
                        rank=0,
                        world_size=len(self.device)
                    )
                target_model = nn.parallel.DistributedDataParallel(
                    target_model.to(self.device[0]),
                    device_ids=[self.device[0].index],
                    find_unused_parameters=self.sft_config.find_unused_parameters
                )
                self.logger.info(f"模型已启用DistributedDataParallel分布式训练（进程数：{len(self.device)}）")
            else:
                target_model = nn.DataParallel(target_model, device_ids=[d.index for d in self.device]).to(self.device[0])
                self.logger.info(f"模型已启用DataParallel多卡训练（显卡数量：{len(self.device)}）")
        else:
            target_device = self.device[0] if isinstance(self.device, list) else self.device
            target_model = target_model.to(target_device)
            self.logger.info(f"模型已迁移至设备：{target_device}")

        # 打印模型信息（优化参数量统计）
        raw_model = target_model.module if hasattr(target_model, "module") else target_model
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        self.logger.info(f"模型结构校验完成：")
        self.logger.info(f"  - 总参数量：{total_params:,}（{total_params/1e6:.2f}M）")
        self.logger.info(f"  - 可训练参数量：{trainable_params:,}（{trainable_params/1e6:.2f}M，占比：{trainable_params/total_params*100:.2f}%）")

        return target_model

    # --------------------------
    # 优化：模型层冻结（兼容动态传入模型）
    # --------------------------
    def _freeze_model_layers(self, model: nn.Module) -> None:
        """
        根据配置冻结模型层（特征提取层/指定前缀层，兼容DDP模型，支持动态传入模型）
        Args:
            model: 待冻结层的模型
        """
        if model is None:
            raise RuntimeError("模型尚未提供，无法进行层冻结操作")

        # 互斥校验：fix_feature_extractor与freeze_layer_prefixes不可同时使用
        if self.sft_config.fix_feature_extractor and len(self.sft_config.freeze_layer_prefixes) > 0:
            raise ValueError("fix_feature_extractor与freeze_layer_prefixes不可同时启用，需二选一")

        frozen_layers = []
        # 获取原始模型（兼容DDP/DataParallel包装）
        raw_model = model.module if hasattr(model, "module") else model

        # 方案1：整体冻结特征提取层（仅微调预测头）
        if self.sft_config.fix_feature_extractor:
            # 支持配置自定义特征提取层属性名
            feature_extractor_attrs = self.sft_config.feature_extractor_attrs or [
                "features", "backbone", "encoder", "lstm", "cnn", "mlp_feature"
            ]
            for attr in feature_extractor_attrs:
                if hasattr(raw_model, attr):
                    feature_extractor = getattr(raw_model, attr)
                    for param in feature_extractor.parameters():
                        param.requires_grad = False
                        frozen_layers.append(f"{attr}（特征提取层）")
            if not frozen_layers:
                self.logger.warning("未找到特征提取层（配置的feature_extractor_attrs中无匹配项），冻结操作无效")
        # 方案2：精细化冻结指定前缀的层
        elif len(self.sft_config.freeze_layer_prefixes) > 0:
            for name, param in model.named_parameters():
                # 去除DDP的module.前缀（若存在），避免前缀匹配失败
                clean_name = name.replace("module.", "")
                for prefix in self.sft_config.freeze_layer_prefixes:
                    if clean_name.startswith(prefix):
                        param.requires_grad = False
                        frozen_layers.append(name)
                        break

        # 打印冻结信息（优化日志展示）
        if frozen_layers:
            self.logger.info(f"成功冻结 {len(frozen_layers)} 个层/参数组：")
            # 去重并按前缀分组展示
            unique_frozen = list(set(frozen_layers))
            for idx, layer in enumerate(unique_frozen[:10]):  # 仅打印前10个，避免日志过长
                self.logger.info(f"  [{idx+1}] 冻结层：{layer}")
            if len(unique_frozen) > 10:
                self.logger.info(f"  ... 还有 {len(unique_frozen)-10} 个层已冻结（共{len(unique_frozen)}个唯一层）")
        else:
            self.logger.info("未冻结任何模型层，所有参数均参与训练")

    # --------------------------
    # 优化：预训练权重加载（支持动态传入模型）
    # --------------------------
    def _load_pretrained_weights(self, model: nn.Module) -> None:
        """
        加载预训练权重，支持自定义预测头前缀过滤，兼容各种权重格式（支持动态传入模型）
        Args:
            model: 待加载权重的模型
        """
        if model is None:
            raise RuntimeError("模型尚未提供，无法加载预训练权重")

        pretrained_path = self.sft_config.pretrained_weight_path
        if pretrained_path is None or not Path(pretrained_path).exists():
            self.logger.info("未指定有效预训练权重路径，模型将从头初始化")
            return

        try:
            # 加载预训练权重字典（先加载到CPU，避免设备不匹配）
            pretrained_state_dict = torch.load(pretrained_path, map_location="cpu")
            # 若断点包含model_state_dict，提取模型权重
            if "model_state_dict" in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict["model_state_dict"]
                self.logger.info("从断点文件中提取模型权重")
            self.logger.info(f"成功加载预训练权重文件：{pretrained_path}")

            # 选择性加载：过滤预测头权重（支持自定义预测头前缀）
            model_state_dict = model.state_dict()
            load_state_dict = {}
            # 优先使用配置的预测头前缀，无配置则使用默认值
            head_key_prefixes = self.sft_config.head_param_prefixes or [
                "head.", "classifier.", "regressor.", "fc_out."
            ]

            for k, v in pretrained_state_dict.items():
                # 情况1：加载预训练头，直接匹配
                if self.sft_config.load_pretrained_head:
                    if k in model_state_dict and v.shape == model_state_dict[k].shape:
                        load_state_dict[k] = v
                    else:
                        self.logger.debug(f"预训练权重键 {k} 不匹配模型结构，跳过加载")  # 改为debug日志，减少冗余输出
                # 情况2：不加载预训练头，过滤预测头权重
                else:
                    if any(k.startswith(prefix) for prefix in head_key_prefixes):
                        self.logger.debug(f"跳过预训练预测头权重：{k}")
                        continue
                    if k in model_state_dict and v.shape == model_state_dict[k].shape:
                        load_state_dict[k] = v
                    else:
                        self.logger.debug(f"预训练权重键 {k} 不匹配模型结构，跳过加载")

            # 加载过滤后的权重
            missing_keys, unexpected_keys = model.load_state_dict(load_state_dict, strict=False)
            self.logger.info(f"成功加载 {len(load_state_dict)} 个匹配的权重参数")

            # 打印加载详情（优化日志）
            if missing_keys:
                self.logger.warning(f"存在 {len(missing_keys)} 个缺失的权重键（多为预测头参数）：{missing_keys[:5]}..." if len(missing_keys)>5 else missing_keys)
            if unexpected_keys:
                self.logger.warning(f"存在 {len(unexpected_keys)} 个额外的权重键（预训练权重中多余参数）：{unexpected_keys[:5]}..." if len(unexpected_keys)>5 else unexpected_keys)

        except Exception as e:
            self.logger.error(f"预训练权重加载失败：{str(e)}")
            raise e

    # --------------------------
    # 优化：数据加载器校验（已整合到_validate_core_components，此处保留兼容）
    # --------------------------
    def _init_dataloaders(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader
    ) -> None:
        """
        验证数据加载器是否合法，打印数据集基本信息（支持动态传入）
        Args:
            train_dataloader: 训练数据加载器（可以是动态传入的）
            val_dataloader: 验证数据加载器（可以是动态传入的）
        """
        # 打印数据集信息
        train_dataset_size = len(train_dataloader.dataset)
        val_dataset_size = len(val_dataloader.dataset)
        train_batch_size = train_dataloader.batch_size
        val_batch_size = val_dataloader.batch_size

        self.logger.info(f"DataLoader校验完成：")
        self.logger.info(f"  - 训练集：样本数 {train_dataset_size:,} | 批次大小 {train_batch_size} | 批次数量 {len(train_dataloader)}")
        self.logger.info(f"  - 验证集：样本数 {val_dataset_size:,} | 批次大小 {val_batch_size} | 批次数量 {len(val_dataloader)}")

        # 校验批次数据格式
        try:
            sample_batch = next(iter(train_dataloader))
            if not isinstance(sample_batch, (tuple, list)) or len(sample_batch) < 2:
                raise ValueError("批次数据需为(inputs, targets)格式的元组/列表")
            self.logger.info(f"  - 批次数据格式校验通过：输入形状 {sample_batch[0].shape} | 标签形状 {sample_batch[1].shape}")
        except Exception as e:
            self.logger.error(f"批次数据格式校验失败：{str(e)}")
            raise e

    # --------------------------
    # 优化：优化器和调度器初始化（支持动态传入模型，延迟初始化）
    # --------------------------
    def _init_optimizer_scheduler(self, model: nn.Module) -> Tuple[torch.optim.Optimizer, Optional[object]]:
        """
        初始化优化器（分层学习率+排除偏置项权重衰减）和学习率调度器（新增OneCycleLR）
        Args:
            model: 待训练模型（用于获取可训练参数，支持动态传入）
        Returns:
            优化器实例、调度器实例
        """
        if model is None:
            raise RuntimeError("模型尚未提供，无法创建优化器")

        # 划分参数组（特征提取层+预测头，不同学习率，支持排除偏置项权重衰减）
        feature_params, head_params = [], []
        head_param_prefixes = self.sft_config.head_param_prefixes or [
            "head.", "classifier.", "regressor.", "fc_out."
        ]
        weight_decay = self.sft_config.weight_decay
        exclude_bias_from_weight_decay = self.sft_config.exclude_bias_from_weight_decay  # 新增配置

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 构建参数信息
            param_info = {"params": param}
            # 排除偏置项和批归一化参数的权重衰减
            if exclude_bias_from_weight_decay and (name.endswith(".bias") or "bn" in name.lower() or "batchnorm" in name.lower()):
                param_info["weight_decay"] = 0.0
            else:
                param_info["weight_decay"] = weight_decay

            # 区分特征提取层和预测头，设置不同学习率
            clean_name = name.replace("module.", "")
            if any(clean_name.startswith(prefix) for prefix in head_param_prefixes):
                # 预测头：高学习率（支持配置缩放因子）
                param_info["lr"] = self.sft_config.learning_rate * self.sft_config.head_lr_scale
                head_params.append(param_info)
            else:
                # 特征提取层：基础学习率
                param_info["lr"] = self.sft_config.learning_rate
                feature_params.append(param_info)

        # 合并参数组
        optimizer_params = feature_params + head_params
        if not optimizer_params:
            raise RuntimeError("无可用的可训练参数，无法创建优化器")

        # 调用基类通用方法创建优化器
        optimizer = self._get_optimizer(optimizer_params)
        self.logger.info(f"优化器 {self.sft_config.optimizer} 初始化完成：")
        self.logger.info(f"  - 分层参数组数量：特征提取层 {len(feature_params)} | 预测头 {len(head_params)}")
        self.logger.info(f"  - 特征提取层学习率：{self.sft_config.learning_rate} | 预测头学习率：{self.sft_config.learning_rate * self.sft_config.head_lr_scale}")
        self.logger.info(f"  - 权重衰减：{weight_decay}（{'排除偏置项和BN参数' if exclude_bias_from_weight_decay else '包含所有参数'}）")

        # 初始化学习率调度器（新增OneCycleLR支持）
        scheduler_type = self.sft_config.scheduler
        scheduler_params = self.sft_config.scheduler_params
        scheduler = None

        try:
            if scheduler_type is None:
                self.logger.info("未启用学习率调度器")
                return optimizer, scheduler

            scheduler_map = {
                "StepLR": StepLR,
                "CosineAnnealingLR": CosineAnnealingLR,
                "ReduceLROnPlateau": ReduceLROnPlateau,
                "ExponentialLR": ExponentialLR,
                "OneCycleLR": OneCycleLR  # 新增调度器
            }
            if scheduler_type not in scheduler_map:
                raise ValueError(f"不支持的调度器类型：{scheduler_type}，支持类型：{list(scheduler_map.keys())}")

            # OneCycleLR专属配置（需传入训练步数）
            if scheduler_type == "OneCycleLR" and "steps_per_epoch" not in scheduler_params:
                # 临时获取训练步数（若train_dataloader已就绪）
                steps_per_epoch = len(self.train_dataloader) if isinstance(self.train_dataloader, DataLoader) else 100
                scheduler_params["steps_per_epoch"] = steps_per_epoch
                scheduler_params["epochs"] = self.sft_config.epochs

            scheduler = scheduler_map[scheduler_type](optimizer, **scheduler_params)
            self.logger.info(f"学习率调度器 {scheduler_type} 初始化完成，参数：{scheduler_params}")
        except Exception as e:
            self.logger.error(f"调度器创建失败：{str(e)}")
            raise e

        return optimizer, scheduler

    # --------------------------
    # 优化：单步训练（传递num_classes和is_multi_label给_compute_metrics）
    # --------------------------
    def train_step(self, batch_data: Any, model: nn.Module) -> Dict[str, float]:
        """
        单步训练逻辑，同步更新StepState训练状态（支持动态传入模型）
        Args:
            batch_data: 格式为(inputs, targets)，从train_dataloader获取
            model: 当前训练的模型（可以是动态传入的）
        Returns:
            训练指标字典（含loss及对应任务指标）
        """
        if model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("模型/优化器/损失函数尚未初始化，无法进行训练步骤")

        # 1. 初始化步骤状态
        self.step_state.start_step_timer()
        model.train()

        # 批次数据合法性校验与拆分
        if not isinstance(batch_data, (tuple, list)) or len(batch_data) < 2:
            raise ValueError(f"批次数据格式无效，需为(inputs, targets)，当前格式：{type(batch_data)}")
        inputs, targets = batch_data[:2]
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # 2. 更新基础状态信息
        self.step_state.update(
            batch_size_actual=inputs.shape[0],
            is_augmented=getattr(batch_data, "is_augmented", False) if len(batch_data) > 2 else False
        )

        # 3. 前向传播（混合精度）
        forward_context = autocast(enabled=self.sft_config.use_mixed_precision)
        forward_start = time.time()
        with forward_context:
            outputs = model(inputs)
            # 优化损失函数适配
            loss_type = self.sft_config.loss_type
            task_type = self.sft_config.sft_task_type
            
            # 动态获取num_classes（已从模型提取，也可从outputs.shape获取）
            num_classes = outputs.shape[1] if len(outputs.shape) >= 2 and task_type in ["classification", "timeseries_classification"] else 1
            # 动态判断is_multi_label（优先从配置安全获取，无则从输出/任务类型判断）
            is_multi_label = getattr(self.sft_config, "is_multi_label", False)
            # 兜底判断：若配置未指定，多标签任务通常输出维度>1且标签为矩阵形式
            if not is_multi_label and task_type == "classification" and outputs.shape[1] > 1 and targets.ndim == 2:
                is_multi_label = True

            if task_type in ["classification", "timeseries_classification"]:
                # 二分类适配BCELoss/BCEWithLogitsLoss
                if num_classes == 1 and loss_type in ["BCELoss", "BCEWithLogitsLoss"]:
                    targets = targets.float().unsqueeze(1)
                # 多分类适配CrossEntropyLoss/FocalLoss
                elif num_classes > 1 and loss_type in ["CrossEntropyLoss", "FocalLoss"] and not is_multi_label:
                    if targets.ndim == outputs.ndim and targets.shape[1] == num_classes:
                        targets = torch.argmax(targets, dim=1)  # one-hot转类别索引
            # 回归任务适配
            else:
                if targets.ndim != outputs.ndim:
                    targets = targets.unsqueeze(-1)

            batch_loss = self.criterion(outputs, targets)
            # 梯度累积：损失归一化
            gradient_accum_steps = self.sft_config.gradient_accumulation_steps
            normalized_loss = batch_loss / gradient_accum_steps if gradient_accum_steps > 1 else batch_loss

        self.step_state.update(
            forward_time=time.time() - forward_start,
            batch_loss=batch_loss.item(),
            normalized_loss=normalized_loss.item()
        )

        # 4. 反向传播（混合精度）
        backward_start = time.time()
        if self.sft_config.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(normalized_loss).backward()
        else:
            normalized_loss.backward()
        self.step_state.update(backward_time=time.time() - backward_start)

        # 5. 梯度裁剪
        gradient_norm, is_clipped = 0.0, False
        clip_norm = self.sft_config.gradient_clip_norm
        if clip_norm and clip_norm > 0:
            if self.sft_config.use_mixed_precision and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_norm
            )
            is_clipped = gradient_norm >= clip_norm
        self.step_state.update(
            gradient_norm=gradient_norm.item(),
            is_gradient_clipped=is_clipped
        )

        # 6. 优化器更新
        update_start = time.time()
        global_step = self.step_state.global_step
        accumulation_step = (global_step % gradient_accum_steps) + 1
        self.step_state.update(accumulation_step=accumulation_step)

        if accumulation_step == gradient_accum_steps:
            if self.sft_config.use_mixed_precision and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step_state.update(scaler_scale=self.scaler.get_scale())
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        self.step_state.update(optimizer_update_time=time.time() - update_start)

        # 7. 指标计算与状态更新（传递num_classes和is_multi_label）
        preds = outputs.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        metrics = self._compute_metrics(preds, target_np, num_classes, is_multi_label)  # 新增参数传递
        metrics["loss"] = batch_loss.item()

        # 更新学习率、全局步数等核心状态
        learning_rates = {f"param_group_{i}": pg["lr"] for i, pg in enumerate(self.optimizer.param_groups)}
        self.step_state.update(
            global_step=global_step + 1,
            batch_metrics=metrics,
            learning_rates=learning_rates,
            optimizer_momentum=self.optimizer.param_groups[0].get("momentum", None)
        )

        # 8. 累积平均损失与计时结束
        self.step_state.update_running_loss(batch_loss.item())
        self.step_state.end_step_timer()

        return metrics

    # --------------------------
    # 优化：单步验证（传递num_classes和is_multi_label给_compute_metrics）
    # --------------------------
    def val_step(self, batch_data: Any, model: nn.Module) -> Dict[str, float]:
        """
        单步验证逻辑（增强容错，优化混合精度处理，支持动态传入模型）
        Args:
            batch_data: 格式为(inputs, targets)，从val_dataloader获取
            model: 当前验证的模型（可以是动态传入的）
        Returns:
            验证指标字典（含loss及对应任务指标）
        """
        if model is None or self.criterion is None:
            raise RuntimeError("模型/损失函数尚未初始化，无法进行验证步骤")

        model.eval()
        val_context = nullcontext() if not self.sft_config.use_mixed_precision else autocast(enabled=True)

        with torch.no_grad(), val_context:
            # 批次数据合法性校验
            if not isinstance(batch_data, (tuple, list)) or len(batch_data) < 2:
                raise ValueError(f"批次数据格式无效，需为(inputs, targets)，当前格式：{type(batch_data)}")
            inputs, targets = batch_data[:2]
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            # 动态获取num_classes和is_multi_label
            task_type = self.sft_config.sft_task_type
            num_classes = outputs.shape[1] if len(outputs.shape) >= 2 and task_type in ["classification", "timeseries_classification"] else 1
            is_multi_label = getattr(self.sft_config, "is_multi_label", False)
            if not is_multi_label and task_type == "classification" and outputs.shape[1] > 1 and targets.ndim == 2:
                is_multi_label = True

            # 损失函数适配
            if self.sft_config.loss_type == "BCELoss":
                outputs = torch.sigmoid(outputs)
                targets = targets.float()
                if targets.ndim != outputs.ndim:
                    targets = targets.unsqueeze(1)
            elif self.sft_config.loss_type == "BCEWithLogitsLoss":
                targets = targets.float()
                if targets.ndim != outputs.ndim:
                    targets = targets.unsqueeze(1)

            loss = self.criterion(outputs, targets)

            # 指标计算（传递num_classes和is_multi_label）
            preds = outputs.cpu().numpy()
            target_np = targets.cpu().numpy()
            metrics = self._compute_metrics(preds, target_np, num_classes, is_multi_label)  # 新增参数传递
            metrics["loss"] = loss.item()

        return metrics

    # --------------------------
    # 优化：完整训练流程（支持Dataset自动转换）
    # --------------------------
    def train(
        self,
        model: Optional[nn.Module] = None,
        train_dataloader: Optional[Union[DataLoader, Dataset]] = None,
        val_dataloader: Optional[Union[DataLoader, Dataset]] = None
    ) -> None:
        """
        完整SFT训练流程（支持动态传入model/dataset/dataloader，自动转换Dataset为DataLoader）
        Args:
            model: 可选，动态传入的待训练模型（优先级高于实例属性）
            train_dataloader: 可选，动态传入的训练数据加载器/数据集（优先级高于实例属性）
            val_dataloader: 可选，动态传入的验证数据加载器/数据集（优先级高于实例属性）
        """
        # 1. 校验并获取有效组件（自动转换Dataset为DataLoader）
        effective_model, effective_train_dl, effective_val_dl = self._validate_core_components(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

        # 2. 延迟初始化模型（若未初始化或传入新模型）
        if self.model != effective_model or not hasattr(effective_model, "_initialized"):
            effective_model = self._init_model(effective_model)
            # 标记模型已初始化，避免重复处理
            setattr(effective_model, "_initialized", True)
            # 更新实例属性（可选，保持一致性）
            self.model = effective_model

        # 【核心新增：提取模型输出类别数，初始化损失函数】
        # 2.1 构建dummy input，获取模型输出维度
        raw_model = effective_model.module if hasattr(effective_model, "module") else effective_model
        # 从训练数据加载器获取输入形状
        sample_batch = next(iter(effective_train_dl))
        sample_input_shape = sample_batch[0].shape[1:]  # [C, H, W] 或 [seq_len, feat_dim]
        dummy_input = torch.randn(1, *sample_input_shape).to(self.device)

        # 2.2 推理模型输出，提取num_classes
        with torch.no_grad():
            model_output = raw_model(dummy_input)
        if self.sft_config.sft_task_type in ["classification", "timeseries_classification"]:
            # 分类任务：输出维度[-1]或[1]为类别数
            num_classes = model_output.shape[1] if len(model_output.shape) >= 2 else 1
        else:
            # 回归任务：类别数传1，不影响损失函数
            num_classes = 1

        # 2.3 初始化损失函数（传入提取的num_classes）
        self._init_criterion(num_classes=num_classes)
        
        # 3. 延迟初始化优化器和调度器（若未初始化）
        if self.optimizer is None or self.scheduler is None:
            self.optimizer, self.scheduler = self._init_optimizer_scheduler(effective_model)

        # 4. 校验数据加载器并打印信息
        self._init_dataloaders(effective_train_dl, effective_val_dl)

        # 5. 延迟记录模型结构图（若未记录）
        if self.sft_config.use_tensorboard and self.writer is not None:
            try:
                self._record_model_graph(effective_model, effective_train_dl)
            except Exception as e:
                self.logger.warning(f"延迟记录模型结构图失败：{str(e)}")

        self.logger.info("="*60)
        self.logger.info("开始SFT模型训练流程")
        self.logger.info(f"训练轮数：{self.sft_config.epochs} | 全局初始步数：{self.step_state.global_step}")
        self.logger.info(f"调试模式：{self.sft_config.debug_mode}（最大步数：{self.sft_config.debug_max_steps} | 验证批次频率：{self.sft_config.debug_val_freq}）")
        self.logger.info(f"混合精度：{self.sft_config.use_mixed_precision} | 梯度累积步数：{self.sft_config.gradient_accumulation_steps}")
        self.logger.info("="*60)

        try:
            # 轮次循环
            for epoch in range(self.step_state.epoch, self.sft_config.epochs):
                self.step_state.update(epoch=epoch)
                self.logger.info(f"\n========== 第 {epoch+1}/{self.sft_config.epochs} 轮训练 ==========")

                # 训练阶段
                train_metrics_accum = {}
                train_batch_num = len(effective_train_dl)
                self.step_state.reset_running_loss()

                for batch_idx, batch_data in enumerate(effective_train_dl):
                    self.step_state.update(batch_idx=batch_idx)
                    # 单步训练（传入有效模型）
                    train_metrics = self.train_step(batch_data, effective_model)
                    # 动态累积指标（避免键不存在问题）
                    for k, v in train_metrics.items():
                        if k not in train_metrics_accum:
                            train_metrics_accum[k] = 0.0
                        train_metrics_accum[k] += v

                    # 打印批次日志（优化：添加进度百分比）
                    if (batch_idx + 1) % self.sft_config.log_freq == 0:
                        progress = (batch_idx + 1) / train_batch_num * 100
                        self.logger.info(f"[训练进度] {batch_idx+1}/{train_batch_num} ({progress:.1f}%)")
                        self.step_state.print_step_summary()

                    # 调试模式：提前终止
                    if self.sft_config.debug_mode and self.step_state.global_step >= self.sft_config.debug_max_steps:
                        self.logger.info(f"调试模式达到最大步数 {self.sft_config.debug_max_steps}，提前终止")
                        return

                    # 重置临时状态
                    self.step_state.reset_temp_params()

                # 训练平均指标（容错：避免除零错误）
                avg_train_metrics = {
                    k: v / max(train_batch_num, 1) for k, v in train_metrics_accum.items()
                }

                # 验证阶段
                val_metrics_accum = {}
                val_batch_num = len(effective_val_dl)

                for batch_idx, batch_data in enumerate(effective_val_dl):
                    # 单步验证（传入有效模型）
                    val_metrics = self.val_step(batch_data, effective_model)
                    # 动态累积指标
                    for k, v in val_metrics.items():
                        if k not in val_metrics_accum:
                            val_metrics_accum[k] = 0.0
                        val_metrics_accum[k] += v

                    # 调试模式验证日志（优化：打印完整临时指标）
                    if self.sft_config.debug_mode and (batch_idx + 1) % self.sft_config.debug_val_freq == 0:
                        temp_avg_metrics = {
                            k: v / (batch_idx + 1) for k, v in val_metrics_accum.items()
                        }
                        self.logger.info(f"[验证进度] Batch: {batch_idx+1}/{val_batch_num}")
                        for k, v in temp_avg_metrics.items():
                            self.logger.info(f"  临时平均{k}：{v:.4f}")

                # 验证平均指标（容错：避免除零错误）
                avg_val_metrics = {
                    k: v / max(val_batch_num, 1) for k, v in val_metrics_accum.items()
                }

                # 轮次汇总日志（优化：格式化输出）
                self.logger.info("="*50)
                self.logger.info(f"[Epoch {epoch+1} 汇总]")
                self.logger.info(f"训练平均损失：{avg_train_metrics.get('loss', 0.0):.4f} | 验证平均损失：{avg_val_metrics.get('loss', 0.0):.4f}")
                for metric in self.sft_config.get_train_evaluation_metrics():
                    if metric == "loss":
                        continue
                    train_val = avg_train_metrics.get(metric, 0.0)
                    val_val = avg_val_metrics.get(metric, 0.0)
                    self.logger.info(f"训练{metric}：{train_val:.4f} | 验证{metric}：{val_val:.4f}")
                self.logger.info("="*50)

                # TensorBoard记录（优化：添加学习率曲线）
                if self.writer is not None:
                    for k, v in avg_train_metrics.items():
                        self.writer.add_scalar(f"Train/{k}", v, epoch)
                    for k, v in avg_val_metrics.items():
                        self.writer.add_scalar(f"Val/{k}", v, epoch)
                    # 记录所有参数组的学习率
                    for i, pg in enumerate(self.optimizer.param_groups):
                        self.writer.add_scalar(f"LearningRate/param_group_{i}", pg["lr"], epoch)

                # 模型保存（优化：添加保存条件判断）
                current_metric = avg_val_metrics.get(self.sft_config.best_model_metric, 0.0)
                is_best = self.update_best_metric(current_metric)
                save_conditions = [
                    (self.sft_config.save_best_model and is_best, "最优模型"),
                    (self.sft_config.save_freq > 0 and (epoch+1) % self.sft_config.save_freq == 0, "定期保存")
                ]
                for condition, desc in save_conditions:
                    if condition:
                        self.save_checkpoint(is_best=is_best, epoch=epoch+1)
                        self.logger.info(f"[{desc}] 已保存模型断点（Epoch {epoch+1}）")

                # 学习率调度（优化：支持按步数调度）
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(current_metric)
                    elif not isinstance(self.scheduler, OneCycleLR):  # OneCycleLR按步数更新
                        self.scheduler.step()

                # 调试模式：验证后终止
                if self.sft_config.debug_mode:
                    self.logger.info("调试模式验证完成，提前终止训练")
                    return

            # 训练结束保存最终模型
            self.save_checkpoint(is_best=False, epoch=self.sft_config.epochs)
            self.logger.info(f"训练流程正常结束，共完成 {self.sft_config.epochs} 轮训练")

        except Exception as e:
            self.logger.error(f"训练流程异常终止：{str(e)}", exc_info=True)  # 打印异常堆栈
            self.save_checkpoint(is_best=False, epoch=self.step_state.epoch)
            raise e
        finally:
            # 确保资源清理（无论正常结束还是异常终止）
            self._after_train()

    # --------------------------
    # 新增：独立评估方法（支持Dataset自动转换）
    # --------------------------
    def eval(
        self,
        model: Optional[nn.Module] = None,
        val_dataloader: Optional[Union[DataLoader, Dataset]] = None
    ) -> Dict[str, float]:
        """
        独立评估流程（支持动态传入model/dataset/dataloader，自动转换Dataset为DataLoader）
        Args:
            model: 可选，动态传入的待评估模型（优先级高于实例属性）
            val_dataloader: 可选，动态传入的验证数据加载器/数据集（优先级高于实例属性）
        Returns:
            评估指标字典（含平均损失和各类任务指标）
        """
        # 1. 先通过核心组件校验（自动转换Dataset为DataLoader）
        effective_model, _, effective_val_dl = self._validate_core_components(
            model=model,
            train_dataloader=self.train_dataloader,  # 仅为校验通过，实际评估不使用
            val_dataloader=val_dataloader
        )

        # 2. 模型设备迁移（若未迁移）
        raw_model = effective_model.module if hasattr(effective_model, "module") else effective_model
        if next(raw_model.parameters()).device != self.device:
            target_device = self.device[0] if isinstance(self.device, list) else self.device
            effective_model = effective_model.to(target_device)
            self.logger.info(f"评估模型已迁移至设备：{target_device}")

        self.logger.info("="*60)
        self.logger.info("开始独立评估流程")
        self.logger.info(f"验证集：样本数 {len(effective_val_dl.dataset):,} | 批次数量 {len(effective_val_dl)}")
        self.logger.info("="*60)

        # 【新增：评估时若损失函数未初始化，自动初始化】
        if self.criterion is None:
            # 提取num_classes
            sample_batch = next(iter(effective_val_dl))
            sample_input_shape = sample_batch[0].shape[1:]
            dummy_input = torch.randn(1, *sample_input_shape).to(self.device)
            with torch.no_grad():
                model_output = raw_model(dummy_input)
            if self.sft_config.sft_task_type in ["classification", "timeseries_classification"]:
                num_classes = model_output.shape[1] if len(model_output.shape) >= 2 else 1
            else:
                num_classes = 1
            # 初始化损失函数
            self._init_criterion(num_classes=num_classes)

            # 3. 评估阶段
            val_metrics_accum = {}
            val_batch_num = len(effective_val_dl)

        for batch_idx, batch_data in enumerate(effective_val_dl):
            val_metrics = self.val_step(batch_data, effective_model)
            # 动态累积指标
            for k, v in val_metrics.items():
                if k not in val_metrics_accum:
                    val_metrics_accum[k] = 0.0
                val_metrics_accum[k] += v

            # 打印评估进度
            if (batch_idx + 1) % self.sft_config.log_freq == 0:
                progress = (batch_idx + 1) / val_batch_num * 100
                self.logger.info(f"[评估进度] {batch_idx+1}/{val_batch_num} ({progress:.1f}%)")

        # 4. 计算平均指标
        avg_val_metrics = {
            k: v / max(val_batch_num, 1) for k, v in val_metrics_accum.items()
        }

        # 5. 打印评估汇总
        self.logger.info("="*50)
        self.logger.info("[独立评估汇总]")
        for k, v in avg_val_metrics.items():
            self.logger.info(f"平均{k}：{v:.4f}")
        self.logger.info("="*50)

        return avg_val_metrics

    # --------------------------
    # 优化：训练后资源清理（增强鲁棒性）
    # --------------------------
    def _after_train(self) -> None:
        """训练结束后清理资源（增强鲁棒性，确保所有资源释放）"""
        # 关闭TensorBoard
        if self.writer is not None:
            try:
                self.writer.close()
                self.logger.info("TensorBoard SummaryWriter已关闭")
            except Exception as e:
                self.logger.warning(f"关闭TensorBoard失败：{str(e)}")

        # 关闭分布式进程组
        if self.sft_config.use_distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
                self.logger.info("分布式进程组已销毁")
            except Exception as e:
                self.logger.warning(f"销毁分布式进程组失败：{str(e)}")

        # 调用基类清理方法
        self.close()

        # 打印最优模型信息（优化：格式化输出）
        self.logger.info("="*60)
        self.logger.info("最优模型信息汇总")
        self.logger.info(f"  - 最优轮次：{self.best_epoch+1} 轮")
        self.logger.info(f"  - 最优{self.sft_config.best_model_metric}：{self.best_metric:.4f}")
        self.logger.info(f"  - 模型输出目录：{self.sft_config.output_dir}")
        if self.model is not None:
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            self.logger.info(f"  - 可训练参数量：{trainable_params:,}")
        self.logger.info("="*60)