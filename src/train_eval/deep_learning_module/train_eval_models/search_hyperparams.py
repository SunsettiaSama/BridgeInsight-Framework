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
from itertools import product
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
MODEL_SAVE_DIR = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\mlp"

# ==================== 超参数网格 ====================
# 网络结构由YAML配置固定，Dropout概率默认0.5
PARAM_GRID = {
    'batch_size': [8, 16, 32],
    'learning_rate': [1e-4, 1e-3, 5e-3],
    'weight_decay': [1e-5, 1e-4],
    'gradient_clip_norm': [0.5, 1.0]
}


def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    """从YAML配置文件加载数据集配置"""
    logger.info(f"加载数据集配置：{config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config_dict['auto_split'] = True
    
    config = AnnotationDatasetConfig(**config_dict)
    logger.info(f"数据集配置加载完成（auto_split=True）")
    
    return config


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


def train_single_mlp(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    input_shape: tuple,
    num_classes: int,
    params: dict,
    output_dir: str,
    combo_idx: int,
    total_combos: int,
    epochs: int = 30
):
    """训练单个MLP模型配置（网络结构和Dropout由YAML配置）"""
    logger.info(f"\n[{combo_idx + 1}/{total_combos}] 测试参数组合")
    
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    gradient_clip_norm = params['gradient_clip_norm']
    
    # 固定的网络配置参数
    hidden_dims = [256, 128, 64]
    dropout_prob = 0.5
    
    params_display = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'gradient_clip_norm': gradient_clip_norm
    }
    logger.info(f"  {params_display}")
    
    model_config = SimpleMLPConfig(
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        task_type="classification",
        activation_type="ReLU",
        dropout=DropoutConfig(enable=True, prob=dropout_prob)
    )
    
    model = get_model(model_config)
    
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
        save_best_model=False,
        save_freq=10,
        use_tensorboard=False,
        save_log_file=False,
        log_freq=10,
        use_mixed_precision=False
    )
    
    trainer = SFTTrainer(
        config=train_config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    
    train_results = trainer.train()
    
    if isinstance(train_results, dict) and 'val_metrics' in train_results:
        val_metrics = train_results.get('val_metrics', {})
        accuracy = val_metrics.get('accuracy', 0.0)
        precision = val_metrics.get('precision', 0.0)
        recall = val_metrics.get('recall', 0.0)
        f1 = val_metrics.get('f1', 0.0)
    else:
        accuracy = precision = recall = f1 = 0.0
    
    result = {
        'params': params,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'epochs': epochs
    }
    
    logger.info(f"  准确率: {accuracy:.4f} | F1: {f1:.4f} | 精确率: {precision:.4f} | 召回率: {recall:.4f}")
    
    return result


def hyperparameter_search_mlp(
    dataset_config_path: str,
    param_grid: dict = None,
    output_dir: str = None,
    epochs_search: int = 30
):
    """MLP网络超参数搜索"""
    if param_grid is None:
        param_grid = PARAM_GRID
    
    if output_dir is None:
        output_dir = MODEL_SAVE_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("开始MLP网络超参数搜索")
    logger.info("=" * 80)
    
    logger.info("\n阶段1：数据集准备")
    logger.info("-" * 80)
    
    dataset_config = load_dataset_config(dataset_config_path)
    
    train_dataloader, val_dataloader, num_classes = create_dataloaders(
        dataset_config,
        batch_size=16
    )
    
    sample_data, _ = train_dataloader.dataset[0]
    input_shape = sample_data.shape
    
    logger.info(f"输入形状：{input_shape}")
    logger.info(f"类别数：{num_classes}")
    
    logger.info("\n阶段2：超参数搜索")
    logger.info("-" * 80)
    logger.info("网络结构配置（来自YAML）：hidden_dims=[256, 128, 64], dropout=0.5")
    
    logger.info(f"超参数网格配置：")
    for key, values in param_grid.items():
        logger.info(f"  {key}: {values}")
    
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(product(*param_values))
    
    total_combinations = len(param_combinations)
    logger.info(f"超参数组合总数：{total_combinations}")
    
    search_results = []
    best_accuracy = 0
    best_f1 = 0
    best_params_acc = None
    best_params_f1 = None
    
    for combo_idx, param_values_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, param_values_tuple))
        
        result = train_single_mlp(
            train_dataloader,
            val_dataloader,
            input_shape,
            num_classes,
            params,
            output_dir,
            combo_idx,
            total_combinations,
            epochs=epochs_search
        )
        
        search_results.append(result)
        
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_params_acc = params.copy()
        
        if result['f1'] > best_f1:
            best_f1 = result['f1']
            best_params_f1 = params.copy()
    
    logger.info("\n" + "=" * 80)
    logger.info("超参数搜索完成！")
    logger.info("=" * 80)
    logger.info(f"基于准确率的最优参数（Acc={best_accuracy:.4f}）:")
    logger.info(f"  {best_params_acc}")
    logger.info(f"\n基于F1分数的最优参数（F1={best_f1:.4f}）:")
    logger.info(f"  {best_params_f1}")
    logger.info("=" * 80)
    
    return best_params_acc, search_results


def analyze_search_results(search_results: list):
    """分析超参数搜索结果，输出排名"""
    if not search_results:
        logger.warning("搜索结果为空")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("搜索结果分析")
    logger.info("=" * 80)
    
    sorted_by_acc = sorted(search_results, key=lambda x: x['accuracy'], reverse=True)
    sorted_by_f1 = sorted(search_results, key=lambda x: x['f1'], reverse=True)
    
    logger.info("\n【TOP 5 最高准确率的参数组合】")
    for rank, result in enumerate(sorted_by_acc[:5], 1):
        logger.info(f"{rank}. Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}")
        logger.info(f"   {result['params']}")
    
    logger.info("\n【TOP 5 最高F1分数的参数组合】")
    for rank, result in enumerate(sorted_by_f1[:5], 1):
        logger.info(f"{rank}. F1={result['f1']:.4f}, Acc={result['accuracy']:.4f}")
        logger.info(f"   {result['params']}")
    
    logger.info("=" * 80)


def save_search_results(search_results: list, best_params: dict, output_dir: str):
    """保存超参数搜索结果"""
    result_save_path = os.path.join(output_dir, "mlp_search_result.json")
    
    os.makedirs(os.path.dirname(result_save_path) or '.', exist_ok=True)
    
    serializable_results = []
    for result in search_results:
        params = result['params'].copy()
        
        serializable_results.append({
            'params': params,
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'epochs': result['epochs']
        })
    
    best_params_serializable = best_params.copy()
    
    final_result = {
        'best_params': best_params_serializable,
        'search_results': serializable_results,
        'total_combinations': len(search_results),
        'best_accuracy': max(r['accuracy'] for r in search_results) if search_results else 0.0,
        'best_f1': max(r['f1'] for r in search_results) if search_results else 0.0
    }
    
    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4, default=str)
    
    logger.info(f"搜索结果已保存至：{result_save_path}")
    
    return result_save_path


def main_search():
    """执行超参数搜索的主函数"""
    logger.info("=" * 80)
    logger.info("MLP网络超参数搜索流程")
    logger.info("=" * 80)
    
    best_params, search_results = hyperparameter_search_mlp(
        dataset_config_path=DATASET_CONFIG_PATH,
        param_grid=PARAM_GRID,
        output_dir=MODEL_SAVE_DIR,
        epochs_search=30
    )
    
    analyze_search_results(search_results)
    
    save_search_results(search_results, best_params, MODEL_SAVE_DIR)
    
    logger.info("\n" + "=" * 80)
    logger.info("超参数搜索完成！")
    logger.info("=" * 80)
    
    return best_params, search_results


if __name__ == "__main__":
    main_search()
