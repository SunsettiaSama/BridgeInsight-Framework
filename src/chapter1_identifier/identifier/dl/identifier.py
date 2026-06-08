import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

# 模型类型 → 输入形状变换方式
_CNN_TYPES  = {"cnn", "res_cnn"}
_SEQ_TYPES  = {"rnn", "lstm"}
_FLAT_TYPES = {"mlp"}


class DLVibrationIdentifier:
    """
    深度学习振动分类识别器。

    与数据集完全解耦：仅接收信号张量，返回类别预测。
    支持 MLP / CNN / RNN / LSTM 四种模型架构。

    输入约定
    --------
    predict_batch 接收形状 (B, window_size, 1) 的 float32 张量，
    内部按 model_type 自动变换为各模型期望的输入格式。
    """

    NUM_CLASSES_DEFAULT = 4
    LABEL_NAMES = {0: "Normal", 1: "VIV", 2: "RWIV", 3: "Transition"}

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        num_classes: int = NUM_CLASSES_DEFAULT,
        device: Optional[str] = None,
    ):
        self.model_type  = model_type.lower()
        self.num_classes = num_classes
        self.device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model       = model.to(self.device)
        self.model.eval()
        logger.info(
            f"DLVibrationIdentifier 就绪：model_type={self.model_type}, "
            f"device={self.device}, num_classes={self.num_classes}"
        )

    # ------------------------------------------------------------------ #
    # 工厂方法                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str,
        model_config_path: str,
        num_classes: int = NUM_CLASSES_DEFAULT,
        device: Optional[str] = None,
    ) -> "DLVibrationIdentifier":
        """
        从训练好的 checkpoint 构建识别器。

        Parameters
        ----------
        checkpoint_path   : .pth 文件路径（SFTTrainer 保存格式）
        model_type        : "mlp" / "cnn" / "rnn" / "lstm"
        model_config_path : 对应模型的 YAML 配置文件路径
        num_classes       : 分类类别数
        device            : "cuda" / "cpu" / None（自动选择）
        """
        from src.training.deep_learning.model_factory import get_model
        from src.training.deep_learning.model_registry import MODEL_CONFIG_REGISTRY

        key = model_type.lower()
        if key not in MODEL_CONFIG_REGISTRY:
            raise ValueError(
                f"未知 model_type='{model_type}'，"
                f"可用值：{list(MODEL_CONFIG_REGISTRY.keys())}"
            )
        config_cls = MODEL_CONFIG_REGISTRY[key]

        with open(model_config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # YAML 结构可能为嵌套格式 {model_type: {fields...}}，提取内层字典
        if key in config_dict and isinstance(config_dict[key], dict):
            config_dict = config_dict[key]

        model_config = config_cls(**config_dict)
        model        = get_model(model_config)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"checkpoint 已加载：{checkpoint_path}")

        return cls(model=model, model_type=model_type, num_classes=num_classes, device=device)

    # ------------------------------------------------------------------ #
    # 推理接口                                                              #
    # ------------------------------------------------------------------ #

    def predict_batch(self, x: torch.Tensor) -> np.ndarray:
        """
        批量预测。

        Parameters
        ----------
        x : Tensor，形状 (B, window_size, 1)

        Returns
        -------
        np.ndarray，形状 (B,)，dtype int32，每个元素为预测类别 ID
        """
        x = self._prepare_input(x).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        return logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

    def predict_batch_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        批量预测各类别 softmax 概率。

        Parameters
        ----------
        x : Tensor，形状 (B, window_size, 1)

        Returns
        -------
        np.ndarray，形状 (B, num_classes)，dtype float32
        """
        x = self._prepare_input(x).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        return torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    # 内部工具                                                              #
    # ------------------------------------------------------------------ #

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """将 (B, window_size, 1) 变换为各模型期望的输入形状。"""
        if x.ndim == 3 and x.shape[-1] == 1:
            if self.model_type in _CNN_TYPES:
                return x.permute(0, 2, 1)        # → (B, 1, window_size)
            if self.model_type in _SEQ_TYPES:
                return x                          # → (B, window_size, 1) 已符合
            # MLP 及其他：展平
            return x.squeeze(-1)                  # → (B, window_size)
        return x

