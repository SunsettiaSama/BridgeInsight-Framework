from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import yaml

from .sft import SFTTrainer
from src.config.train_eval.deep_learning_module.sft import SFTTrainerConfig
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.training.deep_learning.model_factory import get_model
from src.data_processer.datasets.data_factory import get_dataset

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> dict:
    """
    从YAML文件加载配置
    Args:
        config_path: YAML配置文件路径
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"YAML配置文件为空：{config_path}")
    
    return config


def create_trainer(config: SFTTrainerConfig) -> SFTTrainer:
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


def train_from_configs(
    dataset_config_path: str,
    model_config_path: str,
    trainer_config_path: str
) -> None:
    """
    主训练入口函数 - 从三个独立的YAML配置文件读取并执行训练
    
    Args:
        dataset_config_path: 数据集配置YAML文件路径（默认：config/train/datasets/xxx.yaml）
        model_config_path: 模型配置YAML文件路径（默认：config/train/xxx.yaml）
        trainer_config_path: 训练器配置YAML文件路径（默认：config/train/xxx.yaml）
    
    示例用法：
        train_from_configs(
            dataset_config_path="config/train/datasets/annotation_dataset.yaml",
            model_config_path="config/train/models/simple_mlp.yaml",
            trainer_config_path="config/train/trainer_sft.yaml"
        )
    """
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("开始统一训练流程（从YAML配置加载）")
    logger.info("="*70)
    
    try:
        # 1. 加载数据集配置
        logger.info("\n[1/4] 加载数据集配置...")
        dataset_config_dict = load_yaml_config(dataset_config_path)
        dataset_config = AnnotationDatasetConfig(**dataset_config_dict)
        logger.info(f"✓ 数据集配置加载成功")
        logger.info(f"  - 标注文件：{dataset_config.annotation_file}")
        logger.info(f"  - 任务类型：{dataset_config.task_type}")
        logger.info(f"  - 类别数：{dataset_config.num_classes}")
        
        # 2. 加载模型配置
        logger.info("\n[2/4] 加载模型配置...")
        model_config_dict = load_yaml_config(model_config_path)
        # 这里假设模型工厂能够识别配置类型，您可能需要根据实际情况调整
        logger.info(f"✓ 模型配置加载成功")
        logger.info(f"  - 配置内容：{model_config_dict}")
        
        # 3. 加载训练器配置
        logger.info("\n[3/4] 加载训练器配置...")
        trainer_config_dict = load_yaml_config(trainer_config_path)
        trainer_config = SFTTrainerConfig(**trainer_config_dict)
        logger.info(f"✓ 训练器配置加载成功")
        logger.info(f"  - 轮数：{trainer_config.epochs}")
        logger.info(f"  - 批次大小：{trainer_config.batch_size}")
        logger.info(f"  - 学习率：{trainer_config.learning_rate}")
        logger.info(f"  - 优化器：{trainer_config.optimizer}")
        logger.info(f"  - 设备：{trainer_config.device}")
        
        # 4. 创建数据集和模型
        logger.info("\n[4/4] 初始化数据集、模型和训练器...")
        
        # 创建数据集
        dataset = get_dataset(dataset_config)
        train_dataset = dataset.get_train_dataset()
        val_dataset = dataset.get_val_dataset()
        
        # 从数据集获取模型所需的参数
        sample_data, sample_label = train_dataset[0]
        input_shape = sample_data.shape
        num_classes = dataset.get_num_classes()
        
        logger.info(f"✓ 数据集初始化成功")
        logger.info(f"  - 训练集大小：{len(train_dataset)}")
        logger.info(f"  - 验证集大小：{len(val_dataset)}")
        logger.info(f"  - 输入形状：{input_shape}")
        logger.info(f"  - 类别数：{num_classes}")
        
        # 创建模型
        model_config_dict['input_shape'] = input_shape
        model_config_dict['num_classes'] = num_classes
        model = get_model(model_config_dict)
        
        logger.info(f"✓ 模型创建成功")
        logger.info(f"  - 模型类型：{model.__class__.__name__}")
        
        # 创建训练器并开始训练
        trainer = create_trainer(trainer_config)
        
        logger.info(f"✓ 训练器初始化成功")
        logger.info("\n" + "="*70)
        logger.info("开始模型训练...")
        logger.info("="*70 + "\n")
        
        train_model(
            trainer=trainer,
            model=model,
            train_dataloader=train_dataset,
            val_dataloader=val_dataset
        )
        
        logger.info("\n" + "="*70)
        logger.info("训练流程完成")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"训练流程异常终止：{str(e)}", exc_info=True)
        raise


def main(
    config: SFTTrainerConfig,
    model: nn.Module,
    train_dataloader: Union[DataLoader, Dataset],
    val_dataloader: Optional[Union[DataLoader, Dataset]] = None,
) -> None:
    """
    主训练入口函数（兼容旧接口）
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
