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
from src.config.deep_learning_module.models.res_cnn import ResCNNConfig
from src.config.train_eval.deep_learning_module.sft import SFTTrainerConfig
from src.deep_learning_module.model_factory import get_model
from src.train_eval.deep_learning_module.trainer.sft import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_CONFIG_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\datasets\annotation_dataset.yaml"
MODEL_CONFIG_PATH   = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\models\res_cnn.yaml"
MODEL_SAVE_DIR      = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\res_cnn"
SEARCH_RESULT_PATH  = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\search_best_hyperparams\mlp_search_result.json"


def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    logger.info(f"加载数据集配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config_dict['auto_split'] = True
    config = AnnotationDatasetConfig(**config_dict)
    logger.info("数据集配置加载完成（auto_split=True）")
    return config


def load_model_config(config_path: str) -> dict:
    logger.info(f"加载模型配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    config_dict = raw.get('res_cnn', raw)
    logger.info("模型配置加载完成")
    return config_dict


def load_best_params(search_result_path: str) -> dict:
    logger.info(f"加载最佳参数：{search_result_path}")
    with open(search_result_path, 'r', encoding='utf-8') as f:
        search_data = json.load(f)
    best_params = search_data.get('best_params', {})
    logger.info("最佳参数加载完成")
    logger.info(f"  batch_size: {best_params.get('batch_size')}")
    logger.info(f"  learning_rate: {best_params.get('learning_rate')}")
    logger.info(f"  weight_decay: {best_params.get('weight_decay')}")
    logger.info(f"  gradient_clip_norm: {best_params.get('gradient_clip_norm')}")
    return best_params


def create_dataloaders(
    dataset_config: AnnotationDatasetConfig,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    logger.info("创建数据集...")
    dataset = AnnotationDataset(dataset_config)
    logger.info(f"总样本数：{len(dataset)}")
    logger.info(f"类别数：{dataset.get_num_classes()}")

    train_dataset = dataset.get_train_dataset()
    val_dataset   = dataset.get_val_dataset()
    logger.info(f"训练集大小：{len(train_dataset)}")
    logger.info(f"验证集大小：{len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=shuffle_train, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    return train_dataloader, val_dataloader, dataset.get_num_classes()


def train_res_cnn(
    dataset_config_path: str,
    model_config_path: str,
    best_params: dict = None,
    epochs: int = 100,
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = MODEL_SAVE_DIR
    if best_params is None:
        raise ValueError("best_params 不能为空，请提供搜索得到的最佳参数")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始 ResCNN 模型训练（使用搜索最佳参数）")
    logger.info("=" * 60)

    # 1. 加载配置和数据集
    logger.info("\n阶段1：配置加载与数据集准备")
    logger.info("-" * 60)

    dataset_config = load_dataset_config(dataset_config_path)
    model_config   = load_model_config(model_config_path)

    batch_size         = best_params.get('batch_size', 16)
    learning_rate      = best_params.get('learning_rate', 1e-4)
    weight_decay       = best_params.get('weight_decay', 1e-5)
    gradient_clip_norm = best_params.get('gradient_clip_norm', 0.5)

    train_dataloader, val_dataloader, num_classes = create_dataloaders(
        dataset_config, batch_size=batch_size
    )

    # 2. 创建模型
    logger.info("\n阶段2：模型创建")
    logger.info("-" * 60)

    # CNN 的 input_size = seq_len（shape[0]）；in_channels = feature_dim（shape[-1]）
    sample_data, _ = train_dataloader.dataset[0]
    input_size  = sample_data.shape[0]                                  # seq_len
    in_channels = sample_data.shape[-1] if sample_data.ndim > 1 else 1  # feat_dim

    logger.info(f"序列长度：{input_size}")
    logger.info(f"输入通道数：{in_channels}")
    logger.info(f"类别数：{num_classes}")

    model_config_obj = ResCNNConfig(
        in_channels  = in_channels,
        input_size   = input_size,
        res_channels = model_config.get('res_channels', [64, 128, 256]),
        num_blocks   = model_config.get('num_blocks', 2),
        kernel_size  = model_config.get('kernel_size', 3),
        fc_hidden_dims = model_config.get('fc_hidden_dims', [128]),
        dropout_prob   = model_config.get('dropout_prob', 0.5),
        num_classes    = num_classes,
    )

    model = get_model(model_config_obj)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"模型创建成功：{model.__class__.__name__}")
    logger.info(f"总参数数：{total_params:,}")
    logger.info(f"可训练参数：{trainable_params:,}")

    # 3. 训练配置
    logger.info("\n阶段3：训练配置")
    logger.info("-" * 60)

    train_config = SFTTrainerConfig(
        epochs             = epochs,
        batch_size         = batch_size,
        learning_rate      = learning_rate,
        optimizer          = "AdamW",
        scheduler          = "CosineAnnealingLR",
        scheduler_params   = {"T_max": epochs, "eta_min": 1e-6},
        weight_decay       = weight_decay,
        gradient_clip_norm = gradient_clip_norm,
        device             = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir         = output_dir,
        loss_type          = "CrossEntropyLoss",
        sft_task_type      = "classification",
        best_model_metric  = "accuracy",
        save_best_model    = True,
        save_freq          = 10,
        use_tensorboard    = False,
        save_log_file      = False,
        log_freq           = 10,
        use_mixed_precision = False,
    )

    logger.info(f"设备：{train_config.device}")
    logger.info(f"优化器：{train_config.optimizer}")
    logger.info(f"学习率：{train_config.learning_rate}")
    logger.info(f"权重衰减：{train_config.weight_decay}")
    logger.info(f"梯度裁剪范数：{train_config.gradient_clip_norm}")
    logger.info(f"损失函数：{train_config.loss_type}")
    logger.info(f"最优模型评估指标：{train_config.best_model_metric}")

    # 4. 训练
    logger.info("\n阶段4：模型训练")
    logger.info("-" * 60)

    trainer = SFTTrainer(
        config           = train_config,
        model            = model,
        train_dataloader = train_dataloader,
        val_dataloader   = val_dataloader,
    )

    logger.info("开始训练...")
    trainer.train()

    # 5. 收集元数据
    logger.info("\n阶段5：结果整理")
    logger.info("-" * 60)

    training_metadata = trainer.get_training_metadata()

    final_result = {
        'model_config': {
            'model_type':      'ResCNN',
            'in_channels':     in_channels,
            'input_size':      input_size,
            'res_channels':    model_config_obj.res_channels,
            'num_blocks':      model_config_obj.num_blocks,
            'kernel_size':     model_config_obj.kernel_size,
            'fc_hidden_dims':  model_config_obj.fc_hidden_dims,
            'dropout_prob':    model_config_obj.dropout_prob,
            'num_classes':     num_classes,
            'total_params':    total_params,
            'trainable_params': trainable_params,
        },
        'train_config': {
            'epochs':            train_config.epochs,
            'batch_size':        batch_size,
            'learning_rate':     learning_rate,
            'weight_decay':      weight_decay,
            'optimizer':         train_config.optimizer,
            'scheduler':         train_config.scheduler,
            'scheduler_params':  train_config.scheduler_params,
            'loss_type':         train_config.loss_type,
            'gradient_clip_norm': gradient_clip_norm,
            'device':            train_config.device,
            'use_mixed_precision': train_config.use_mixed_precision,
        },
        'dataset_info': {
            'train_samples': len(train_dataloader.dataset),
            'val_samples':   len(val_dataloader.dataset),
            'train_batches': len(train_dataloader),
            'val_batches':   len(val_dataloader),
        },
        'training_metadata': training_metadata,
    }

    # 6. 保存结果
    result_save_path = os.path.join(output_dir, "res_cnn_train_result.json")
    os.makedirs(os.path.dirname(result_save_path) or '.', exist_ok=True)
    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4, default=str)
    logger.info(f"训练结果已保存至：{result_save_path}")

    # 7. 训练总结
    logger.info("\n" + "=" * 60)
    logger.info("【ResCNN 训练总结】")
    logger.info("=" * 60)
    logger.info(f"模型类型：{final_result['model_config']['model_type']}")
    logger.info(f"总参数数：{final_result['model_config']['total_params']:,}")
    logger.info(f"可训练参数：{final_result['model_config']['trainable_params']:,}")
    logger.info(f"\n模型配置：")
    logger.info(f"  - 输入通道数：{final_result['model_config']['in_channels']}")
    logger.info(f"  - 序列长度：{final_result['model_config']['input_size']}")
    logger.info(f"  - 残差阶段通道：{final_result['model_config']['res_channels']}")
    logger.info(f"  - 每阶段块数：{final_result['model_config']['num_blocks']}")
    logger.info(f"  - 卷积核大小：{final_result['model_config']['kernel_size']}")
    logger.info(f"  - FC 隐藏层：{final_result['model_config']['fc_hidden_dims']}")
    logger.info(f"  - Dropout：{final_result['model_config']['dropout_prob']}")
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
    best_params = load_best_params(SEARCH_RESULT_PATH)
    train_res_cnn(
        dataset_config_path = DATASET_CONFIG_PATH,
        model_config_path   = MODEL_CONFIG_PATH,
        best_params         = best_params,
        epochs              = 100,
        output_dir          = MODEL_SAVE_DIR,
    )
    logger.info("\n最终结果已保存，训练完成！")


if __name__ == "__main__":
    main()
