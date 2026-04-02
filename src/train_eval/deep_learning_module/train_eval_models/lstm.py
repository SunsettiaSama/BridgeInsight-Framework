import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import logging
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.config.deep_learning_module.models.lstm import LSTMConfig
from src.config.train_eval.deep_learning_module.sft import SFTTrainerConfig
from src.deep_learning_module.model_factory import get_model
from src.train_eval.deep_learning_module.trainer.sft import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_CONFIG_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\datasets\annotation_dataset.yaml"
MODEL_CONFIG_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\models\lstm.yaml"
MODEL_SAVE_DIR = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\lstm"
SEARCH_RESULT_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\search_best_hyperparams\mlp_search_result.json"


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
    """从YAML配置文件加载模型网络结构配置"""
    logger.info(f"加载模型配置：{config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    logger.info(f"模型配置加载完成")
    return config_dict


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


def train_lstm(
    dataset_config_path: str,
    model_config_path: str,
    best_params: dict = None,
    epochs: int = 100,
    output_dir: str = None
):
    """
    LSTM模型训练主流程（使用搜索得到的最佳参数）
    
    参数：
        dataset_config_path: 数据集配置文件路径
        model_config_path: 模型配置文件路径
        best_params: 最佳超参数字典，包含batch_size、learning_rate、weight_decay、gradient_clip_norm
        epochs: 训练轮数
        output_dir: 输出目录
    
    返回：
        训练结果字典，包含所有关键参数和指标
    """
    if output_dir is None:
        output_dir = MODEL_SAVE_DIR
    
    if best_params is None:
        raise ValueError("best_params 不能为空，请提供搜索得到的最佳参数")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("开始LSTM模型训练（使用搜索最佳参数）")
    logger.info("=" * 60)
    
    # 1. 加载配置和数据集
    logger.info("\n阶段1：配置加载与数据集准备")
    logger.info("-" * 60)
    
    dataset_config = load_dataset_config(dataset_config_path)
    model_config = load_model_config(model_config_path)
    
    batch_size = best_params.get('batch_size', 16)
    learning_rate = best_params.get('learning_rate', 1e-4)
    weight_decay = best_params.get('weight_decay', 1e-5)
    gradient_clip_norm = best_params.get('gradient_clip_norm', 0.5)
    
    train_dataloader, val_dataloader, num_classes = create_dataloaders(
        dataset_config,
        batch_size=batch_size
    )
    
    # 2. 创建模型
    logger.info("\n阶段2：模型创建")
    logger.info("-" * 60)
    
    sample_data, _ = train_dataloader.dataset[0]
    input_size = sample_data.shape[-1] if sample_data.ndim > 1 else 1
    
    logger.info(f"输入大小：{input_size}")
    logger.info(f"类别数：{num_classes}")
    
    hidden_size = model_config.get('hidden_size', 128)
    num_layers = model_config.get('num_layers', 1)
    bidirectional = model_config.get('bidirectional', False)
    dropout = model_config.get('dropout', 0.5)
    batch_first = model_config.get('batch_first', True)
    seq_dropout = model_config.get('seq_dropout', 0.5)
    classifier_dropout = model_config.get('classifier_dropout', 0.5)
    
    model_config_obj = LSTMConfig(
        input_size=input_size,
        num_classes=num_classes,
        predict_seq_len=None,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        batch_first=batch_first,
        task_type="classification",
        seq_dropout=seq_dropout,
        classifier_dropout=classifier_dropout
    )
    
    model = get_model(model_config_obj)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型创建成功：{model.__class__.__name__}")
    logger.info(f"总参数数：{total_params:,}")
    logger.info(f"可训练参数：{trainable_params:,}")
    
    # 3. 创建训练配置
    logger.info("\n阶段3：训练配置")
    logger.info("-" * 60)
    
    train_config = SFTTrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer="AdamW",
        scheduler="CosineAnnealingLR",
        scheduler_params={"T_max": epochs, "eta_min": 1e-6},
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=output_dir,
        loss_type="CrossEntropyLoss",
        sft_task_type="classification",
        best_model_metric="accuracy",
        save_best_model=True,
        save_freq=10,
        use_tensorboard=False,
        save_log_file=False,
        log_freq=10,
        use_mixed_precision=False
    )
    
    logger.info(f"设备：{train_config.device}")
    logger.info(f"优化器：{train_config.optimizer}")
    logger.info(f"学习率：{train_config.learning_rate}")
    logger.info(f"权重衰减：{train_config.weight_decay}")
    logger.info(f"梯度裁剪范数：{train_config.gradient_clip_norm}")
    logger.info(f"损失函数：{train_config.loss_type}")
    logger.info(f"最优模型评估指标：{train_config.best_model_metric}")
    
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
    
    final_result = {
        'model_config': {
            'model_type': 'LSTM',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'num_classes': num_classes,
            'dropout': dropout,
            'batch_first': batch_first,
            'seq_dropout': seq_dropout,
            'classifier_dropout': classifier_dropout,
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
    result_save_path = os.path.join(output_dir, "lstm_train_result.json")
    
    os.makedirs(os.path.dirname(result_save_path) or '.', exist_ok=True)
    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4, default=str)
    
    logger.info(f"训练结果已保存至：{result_save_path}")
    
    # 7. 打印训练总结
    logger.info("\n" + "=" * 60)
    logger.info("【LSTM训练总结】")
    logger.info("=" * 60)
    logger.info(f"模型类型：{final_result['model_config']['model_type']}")
    logger.info(f"总参数数：{final_result['model_config']['total_params']:,}")
    logger.info(f"可训练参数：{final_result['model_config']['trainable_params']:,}")
    logger.info(f"\n模型配置：")
    logger.info(f"  - 隐藏层大小：{final_result['model_config']['hidden_size']}")
    logger.info(f"  - 隐藏层数：{final_result['model_config']['num_layers']}")
    logger.info(f"  - 双向：{final_result['model_config']['bidirectional']}")
    logger.info(f"\n训练配置：")
    logger.info(f"  - 总轮数：{final_result['train_config']['epochs']}")
    logger.info(f"  - 批次大小：{final_result['train_config']['batch_size']}")
    logger.info(f"  - 学习率：{final_result['train_config']['learning_rate']}")
    logger.info(f"  - 权重衰减：{final_result['train_config']['weight_decay']}")
    logger.info(f"  - 梯度裁剪范数：{final_result['train_config']['gradient_clip_norm']}")
    logger.info(f"  - 优化器：{final_result['train_config']['optimizer']}")
    logger.info(f"  - 调度器：{final_result['train_config']['scheduler']}")
    logger.info(f"\n数据集信息：")
    logger.info(f"  - 训练集大小：{final_result['dataset_info']['train_samples']}")
    logger.info(f"  - 验证集大小：{final_result['dataset_info']['val_samples']}")
    logger.info("=" * 60)
    
    return final_result


def main():
    """主函数"""
    best_params = load_best_params(SEARCH_RESULT_PATH)
    
    result = train_lstm(
        dataset_config_path=DATASET_CONFIG_PATH,
        model_config_path=MODEL_CONFIG_PATH,
        best_params=best_params,
        epochs=100,
        output_dir=MODEL_SAVE_DIR
    )
    
    logger.info("\n最终结果已保存，训练完成！")


if __name__ == "__main__":
    main()
