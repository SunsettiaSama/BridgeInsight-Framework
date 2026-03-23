from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging

from .sft import SFTTrainer
from src.config.trainer.base_config import BaseConfig


def create_trainer(config: BaseConfig) -> SFTTrainer:
    """
    根据配置创建SFT训练器实例
    Args:
        config: 训练配置对象
    Returns:
        初始化完成的SFTTrainer实例
    """
    trainer = SFTTrainer(config=config)
    return trainer


def train_model(
    trainer: SFTTrainer,
    model: nn.Module,
    train_dataloader: Union[DataLoader, Dataset],
    val_dataloader: Optional[Union[DataLoader, Dataset]] = None,
    epochs: Optional[int] = None
) -> None:
    """
    执行完整的模型训练流程
    Args:
        trainer: SFTTrainer实例
        model: 待训练的模型
        train_dataloader: 训练数据加载器或数据集
        val_dataloader: 验证数据加载器或数据集（可选）
        epochs: 可选，覆盖配置中的轮数
    """
    if epochs is not None:
        trainer.sft_config.epochs = epochs
    
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )


def main(
    config: BaseConfig,
    model: nn.Module,
    train_dataloader: Union[DataLoader, Dataset],
    val_dataloader: Optional[Union[DataLoader, Dataset]] = None,
) -> None:
    """
    主训练入口函数
    Args:
        config: 训练配置
        model: 待训练的模型
        train_dataloader: 训练数据加载器或数据集
        val_dataloader: 验证数据加载器或数据集（可选）
    """
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("初始化训练流程")
    logger.info("="*60)
    
    trainer = create_trainer(config=config)
    
    logger.info("开始训练")
    train_model(
        trainer=trainer,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    logger.info("="*60)
    logger.info("训练流程完成")
    logger.info("="*60)
