import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import logging
import json
import os
import torch
from torch.utils.data import DataLoader
import yaml

from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.config.deep_learning_module.models.mlp import SimpleMLPConfig, DropoutConfig
from src.config.train_eval.deep_learning_module.sft import SFTTrainerConfig
from src.deep_learning_module.model_factory import get_model
from src.train_eval.deep_learning_module.trainer.sft import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_CONFIG_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\datasets\annotation_dataset.yaml"
MODEL_CONFIG_PATH   = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\models\mlp.yaml"
TRAINER_CONFIG_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\trainer\sft.yaml"
MODEL_SAVE_DIR      = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\mlp"
SEARCH_RESULT_PATH  = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\search_best_hyperparams\mlp_search_result.json"


def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    """从YAML配置文件加载数据集配置"""
    logger.info(f"加载数据集配置：{config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    config_dict['auto_split'] = True

    config = AnnotationDatasetConfig(**config_dict)
    logger.info(f"数据集配置加载完成（auto_split=True）")
    
    return config


def load_model_config(config_path: str) -> dict:
    logger.info(f"加载模型配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    logger.info("模型配置加载完成")
    return config_dict


def load_trainer_config(config_path: str) -> dict:
    logger.info(f"加载 Trainer 配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Trainer 配置加载完成")
    return cfg


def load_best_params(search_result_path: str) -> dict:
    """从超参数搜索结果中加载最佳参数"""
    logger.info(f"加载最佳参数：{search_result_path}")
    
    with open(search_result_path, 'r', encoding='utf-8') as f:
        search_data = json.load(f)
    
    best_params = search_data.get('best_params', {})
    logger.info(f"最佳参数加载完成")
    logger.info(f"  batch_size: {best_params.get('batch_size')}")
    logger.info(f"  learning_rate: {best_params.get('learning_rate')}")
    logger.info(f"  weight_decay: {best_params.get('weight_decay')}")
    logger.info(f"  gradient_clip_norm: {best_params.get('gradient_clip_norm')}")
    
    return best_params


def create_dataloaders(
    dataset_config: AnnotationDatasetConfig,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True
):
    """创建训练/验证数据加载器（8:2比例划分）"""
    logger.info("创建数据集...")
    
    dataset = AnnotationDataset(dataset_config)
    
    logger.info(f"总样本数：{len(dataset)}")
    logger.info(f"类别数：{dataset.get_num_classes()}")
    
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    
    logger.info(f"训练集大小：{len(train_dataset)}")
    logger.info(f"验证集大小：{len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader, dataset.get_num_classes()


def train_mlp(
    dataset_config_path: str,
    model_config_path: str,
    trainer_config_path: str = None,
    best_params: dict = None,
    epochs: int = None,
    output_dir: str = None
):
    """
    MLP模型训练主流程（使用搜索得到的最佳超参数）

    Trainer 通用配置（优化器/调度器/损失函数等）读取自 sft.yaml，
    best_params 中的搜索参数（batch_size/learning_rate/weight_decay/
    gradient_clip_norm）覆盖 YAML 对应字段；
    epochs 与 output_dir 若不传则沿用 YAML 中的值。
    """
    if trainer_config_path is None:
        trainer_config_path = TRAINER_CONFIG_PATH
    if output_dir is None:
        output_dir = MODEL_SAVE_DIR
    if best_params is None:
        raise ValueError("best_params 不能为空，请提供搜索得到的最佳参数")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始MLP模型训练（使用搜索最佳参数）")
    logger.info("=" * 60)

    # 1. 加载配置和数据集
    logger.info("\n阶段1：配置加载与数据集准备")
    logger.info("-" * 60)

    dataset_config = load_dataset_config(dataset_config_path)
    model_config   = load_model_config(model_config_path)
    trainer_cfg    = load_trainer_config(trainer_config_path)

    # 用 best_params 覆盖 YAML 中的搜索参数
    trainer_cfg.update({
        'batch_size':         best_params['batch_size'],
        'learning_rate':      best_params['learning_rate'],
        'weight_decay':       best_params['weight_decay'],
        'gradient_clip_norm': best_params['gradient_clip_norm'],
        'output_dir':         output_dir,
    })
    if epochs is not None:
        trainer_cfg['epochs'] = epochs

    batch_size    = trainer_cfg['batch_size']
    learning_rate = trainer_cfg['learning_rate']
    weight_decay  = trainer_cfg['weight_decay']
    gradient_clip_norm = trainer_cfg['gradient_clip_norm']

    train_dataloader, val_dataloader, num_classes = create_dataloaders(
        dataset_config, batch_size=batch_size
    )

    # 2. 创建模型
    logger.info("\n阶段2：模型创建")
    logger.info("-" * 60)

    sample_data, _ = train_dataloader.dataset[0]
    input_shape = sample_data.shape

    logger.info(f"输入形状：{input_shape}  类别数：{num_classes}")

    hidden_dims  = model_config.get('hidden_dims', [256, 128, 64])
    dropout_prob = model_config.get('dropout', {}).get('prob', 0.5)

    model_config_obj = SimpleMLPConfig(
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        task_type="classification",
        activation_type="ReLU",
        dropout=DropoutConfig(enable=True, prob=dropout_prob)
    )
    model = get_model(model_config_obj)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型：{model.__class__.__name__}  总参数：{total_params:,}  可训练：{trainable_params:,}")

    # 3. 创建训练配置（以 YAML 为基础，已叠加 best_params）
    logger.info("\n阶段3：训练配置")
    logger.info("-" * 60)

    train_config = SFTTrainerConfig(**trainer_cfg)
    
    logger.info(f"设备：{train_config.device}  优化器：{train_config.optimizer}  调度器：{train_config.scheduler}")
    logger.info(f"学习率：{train_config.learning_rate}  权重衰减：{train_config.weight_decay}  梯度裁剪：{train_config.gradient_clip_norm}")
    logger.info(f"损失函数：{train_config.loss_type}  最优指标：{train_config.best_model_metric}  训练轮数：{train_config.epochs}")
    
    # 4. 创建训练器并执行训练
    logger.info("\n阶段4：模型训练")
    logger.info("-" * 60)
    
    trainer = SFTTrainer(
        config=train_config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    logger.info("开始训练...")
    trainer.train()
    
    # 5. 收集训练元数据
    logger.info("\n阶段5：结果整理")
    logger.info("-" * 60)
    
    training_metadata = trainer.get_training_metadata()
    
    # 提取最优指标对应的epoch数据
    best_epoch_data = None
    if training_metadata.get('epoch_states'):
        epoch_states = training_metadata['epoch_states']
        best_epoch_idx = training_metadata.get('best_epoch', 1) - 1
        if 0 <= best_epoch_idx < len(epoch_states):
            best_epoch_data = epoch_states[best_epoch_idx]
    
    final_result = {
        'model_config': {
            'model_type': 'SimpleMLP',
            'input_shape': list(input_shape),
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
            'activation_type': 'ReLU',
            'dropout_enable': True,
            'dropout_prob': dropout_prob,
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'train_config': {
            'epochs': train_config.epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer': train_config.optimizer,
            'scheduler': train_config.scheduler,
            'scheduler_params': train_config.scheduler_params,
            'loss_type': train_config.loss_type,
            'gradient_clip_norm': gradient_clip_norm,
            'device': train_config.device,
            'use_mixed_precision': train_config.use_mixed_precision
        },
        'dataset_info': {
            'train_samples': len(train_dataloader.dataset),
            'val_samples': len(val_dataloader.dataset),
            'train_batches': len(train_dataloader),
            'val_batches': len(val_dataloader)
        },
        'training_metadata': training_metadata
    }
    
    # 6. 保存结果
    result_save_path = os.path.join(output_dir, "mlp_train_result.json")
    
    os.makedirs(os.path.dirname(result_save_path) or '.', exist_ok=True)
    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4, default=str)
    
    logger.info(f"训练结果已保存至：{result_save_path}")
    
    # 7. 打印训练总结
    logger.info("\n" + "=" * 80)
    logger.info("【MLP训练总结】")
    logger.info("=" * 80)
    logger.info(f"模型类型：{final_result['model_config']['model_type']}")
    logger.info(f"总参数数：{final_result['model_config']['total_params']:,}")
    logger.info(f"可训练参数：{final_result['model_config']['trainable_params']:,}")
    
    logger.info(f"\n【最优模型指标】")
    logger.info(f"  - 最优指标类型：{training_metadata.get('best_metric_name', 'unknown')}")
    logger.info(f"  - 最优指标值：{training_metadata.get('best_metric', 0):.6f}")
    logger.info(f"  - 最优epoch：{training_metadata.get('best_epoch', 0)}")
    
    if best_epoch_data:
        val_metrics = best_epoch_data.get('val_metrics', {})
        train_metrics = best_epoch_data.get('train_metrics', {})
        logger.info(f"\n【最优epoch的详细指标】")
        logger.info(f"  验证集指标：")
        logger.info(f"    - Accuracy：{val_metrics.get('accuracy', 0):.6f}")
        logger.info(f"    - F1：{val_metrics.get('f1', 0):.6f}")
        logger.info(f"    - Precision：{val_metrics.get('precision', 0):.6f}")
        logger.info(f"    - Recall：{val_metrics.get('recall', 0):.6f}")
        logger.info(f"    - Loss：{val_metrics.get('loss', 0):.6f}")
        logger.info(f"  训练集指标：")
        logger.info(f"    - Accuracy：{train_metrics.get('accuracy', 0):.6f}")
        logger.info(f"    - F1：{train_metrics.get('f1', 0):.6f}")
        logger.info(f"    - Precision：{train_metrics.get('precision', 0):.6f}")
        logger.info(f"    - Recall：{train_metrics.get('recall', 0):.6f}")
        logger.info(f"    - Loss：{train_metrics.get('loss', 0):.6f}")
    
    logger.info(f"\n【训练配置】")
    logger.info(f"  - 总轮数：{final_result['train_config']['epochs']}")
    logger.info(f"  - 批次大小：{final_result['train_config']['batch_size']}")
    logger.info(f"  - 学习率：{final_result['train_config']['learning_rate']}")
    logger.info(f"  - 权重衰减：{final_result['train_config']['weight_decay']}")
    logger.info(f"  - 梯度裁剪范数：{final_result['train_config']['gradient_clip_norm']}")
    logger.info(f"  - 优化器：{final_result['train_config']['optimizer']}")
    logger.info(f"  - 调度器：{final_result['train_config']['scheduler']}")
    
    logger.info(f"\n【数据集信息】")
    logger.info(f"  - 训练集大小：{final_result['dataset_info']['train_samples']}")
    logger.info(f"  - 验证集大小：{final_result['dataset_info']['val_samples']}")
    
    logger.info("=" * 80)
    
    return final_result


def main():
    best_params = load_best_params(SEARCH_RESULT_PATH)

    result = train_mlp(
        dataset_config_path=DATASET_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
        trainer_config_path=TRAINER_CONFIG_PATH,
        best_params=best_params,
        output_dir=MODEL_SAVE_DIR
    )
    
    logger.info("\n最终结果已保存，训练完成！")


if __name__ == "__main__":
    main()
