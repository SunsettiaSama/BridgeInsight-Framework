import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import logging
import json
import os
import torch
from torch.utils.data import DataLoader
import yaml
from itertools import product

from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset, log_split_distribution
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.config.deep_learning_module.models.mlp import SimpleMLPConfig, DropoutConfig
from src.config.train_eval.deep_learning_module.sft import SFTTrainerConfig
from src.training.deep_learning.model_factory import get_model
from src.training.deep_learning.trainer.sft import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_CONFIG_PATH  = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\datasets\annotation_dataset.yaml"
MODEL_CONFIG_PATH    = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\models\mlp.yaml"
TRAINER_CONFIG_PATH  = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\trainer\sft.yaml"
MODEL_SAVE_DIR       = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\search_best_hyperparams"

# ==================== 超参数网格 ====================
# 网络结构由YAML配置固定，Dropout概率默认0.5
# PARAM_GRID = {
#     'batch_size': [8],
#     'learning_rate': [1e-4],
#     'weight_decay': [1e-5],
#     'gradient_clip_norm': [0.5],
#     'label_smoothing': [0.1],
# }

PARAM_GRID = {
    'batch_size': [8, 16, 32],
    'learning_rate': [1e-4, 1e-3, 5e-3],
    'weight_decay': [1e-5, 1e-4],
    'gradient_clip_norm': [0.5, 1.0],
    'label_smoothing': [0.0, 0.05, 0.1, 0.15],
}
SEARCH_EPOCH = 100


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


def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    """从YAML配置文件加载数据集配置"""
    logger.info(f"加载数据集配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = AnnotationDatasetConfig(**config_dict)
    logger.info(f"数据集配置加载完成（split_ratio={config.split_ratio}, auto_split={config.auto_split}）")
    return config


def create_dataloaders(
    dataset_config: AnnotationDatasetConfig,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True,
    dataset: AnnotationDataset = None
):
    """创建训练/验证数据加载器（8:2比例划分）
    
    参数:
        dataset_config: 数据集配置
        batch_size: batch大小
        num_workers: 加载数据的进程数
        shuffle_train: 是否打乱训练集
        dataset: 可选，已经初始化的数据集实例。如果为None则创建新的数据集
    
    返回:
        (train_dataloader, val_dataloader, num_classes)
    """
    if dataset is None:
        logger.info("创建数据集...")
        dataset = AnnotationDataset(dataset_config)
        logger.info(f"总样本数：{len(dataset)}")
        logger.info(f"类别数：{dataset.get_num_classes()}")
        log_split_distribution(dataset)
    else:
        logger.debug(f"复用已有数据集（batch_size={batch_size}）")

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
    model_config: dict,
    trainer_base_config: dict,
    output_dir: str,
    combo_idx: int,
    total_combos: int,
    epochs: int = 30
):
    logger.info(f"\n[{combo_idx + 1}/{total_combos}] 测试参数组合")
    logger.info(f"  {params}")

    hidden_dims = model_config.get('hidden_dims', [256, 128, 64])
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

    # 以 YAML 为基础，仅覆盖网格搜索参数及搜索专用设置
    trainer_cfg = trainer_base_config.copy()
    trainer_cfg.update({
        'epochs':             epochs,
        'batch_size':         params['batch_size'],
        'learning_rate':      params['learning_rate'],
        'weight_decay':       params['weight_decay'],
        'gradient_clip_norm': params['gradient_clip_norm'],
        'label_smoothing':    params['label_smoothing'],
        'output_dir':         output_dir,
        'save_best_model':    False,
        'save_freq':          0,
    })
    train_config = SFTTrainerConfig(**trainer_cfg)

    trainer = SFTTrainer(
        config=train_config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )
    trainer.train()

    training_metadata = trainer.get_training_metadata()
    best_metric_value = float(trainer.best_metric)
    best_epoch = int(trainer.best_epoch + 1) if hasattr(trainer, 'best_epoch') else 0

    logger.info(f"  最佳 {train_config.best_model_metric}={best_metric_value:.4f}  (Epoch {best_epoch})")

    return {
        'params': params,
        'best_metrics': {
            'metric_name':  train_config.best_model_metric,
            'metric_value': best_metric_value,
            'epoch':        best_epoch,
        },
        'best_metric_value': best_metric_value,
        'epochs':            epochs,
        'training_metadata': training_metadata,
    }


def hyperparameter_search_mlp(
    dataset_config_path: str,
    model_config_path: str,
    trainer_config_path: str = None,
    param_grid: dict = None,
    output_dir: str = None,
    epochs_search: int = 30
):
    if param_grid is None:
        param_grid = PARAM_GRID
    if output_dir is None:
        output_dir = MODEL_SAVE_DIR
    if trainer_config_path is None:
        trainer_config_path = TRAINER_CONFIG_PATH

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("开始MLP网络超参数搜索")
    logger.info("=" * 80)

    logger.info("\n阶段1：配置与数据集准备（仅初始化一次）")
    logger.info("-" * 80)

    model_config = load_model_config(model_config_path)
    trainer_base_config = load_trainer_config(trainer_config_path)
    dataset_config = load_dataset_config(dataset_config_path)

    train_dataloader, val_dataloader, num_classes = create_dataloaders(
        dataset_config, batch_size=16, num_workers=0
    )

    dataset = train_dataloader.dataset
    sample_data, _ = dataset[0]
    input_shape = sample_data.shape

    logger.info(f"输入形状：{input_shape}  类别数：{num_classes}")
    logger.info(f"网络结构（YAML）：hidden_dims={model_config.get('hidden_dims')}  dropout={model_config.get('dropout')}")
    logger.info(f"Trainer 基础配置（YAML）：optimizer={trainer_base_config.get('optimizer')}  "
                f"scheduler={trainer_base_config.get('scheduler')}  "
                f"loss={trainer_base_config.get('loss_type')}  "
                f"best_metric={trainer_base_config.get('best_model_metric')}")

    logger.info("\n阶段2：超参数搜索")
    logger.info("-" * 80)
    for key, values in param_grid.items():
        logger.info(f"  {key}: {values}")

    param_names = list(param_grid.keys())
    param_combinations = list(product(*[param_grid[n] for n in param_names]))
    total_combinations = len(param_combinations)
    logger.info(f"超参数组合总数：{total_combinations}")

    search_results = []
    best_overall_params = None

    # 根据指标类型选择初始哨兵值和比较方向，与 update_best_metric 保持一致
    _metric_name = trainer_base_config.get('best_model_metric', '').lower()
    _higher_is_better = any(k in _metric_name for k in ('acc', 'f1', 'auc', 'precision', 'recall'))
    best_overall_metric = -float('inf') if _higher_is_better else float('inf')

    original_dataset = train_dataloader.dataset
    if hasattr(original_dataset, 'dataset'):
        original_dataset = original_dataset.dataset

    for combo_idx, param_values_tuple in enumerate(param_combinations):
        params = dict(zip(param_names, param_values_tuple))

        train_dataloader_combo, val_dataloader_combo, _ = create_dataloaders(
            dataset_config,
            batch_size=params['batch_size'],
            num_workers=0,
            dataset=original_dataset
        )

        result = train_single_mlp(
            train_dataloader_combo,
            val_dataloader_combo,
            input_shape,
            num_classes,
            params,
            model_config,
            trainer_base_config,
            output_dir,
            combo_idx,
            total_combinations,
            epochs=epochs_search
        )

        search_results.append(result)

        val = result['best_metric_value']
        is_better = (val > best_overall_metric) if _higher_is_better else (val < best_overall_metric)
        if is_better:
            best_overall_metric = val
            best_overall_params = params.copy()

    logger.info("\n" + "=" * 80)
    logger.info("超参数搜索完成！")
    logger.info(f"全局最优参数（{trainer_base_config.get('best_model_metric')}={best_overall_metric:.4f}）：")
    logger.info(f"  {best_overall_params}")
    logger.info("=" * 80)

    return best_overall_params, search_results


def analyze_search_results(search_results: list):
    """分析超参数搜索结果，输出排名"""
    if not search_results:
        logger.warning("搜索结果为空")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("搜索结果分析")
    logger.info("=" * 80)
    
    sorted_by_metric = sorted(search_results, key=lambda x: x['best_metric_value'], reverse=True)
    metric_name = sorted_by_metric[0].get('best_metrics', {}).get('metric_name', 'metric')

    logger.info(f"\n【TOP 5 最优参数组合（按 {metric_name} 排序）】")
    for rank, result in enumerate(sorted_by_metric[:5], 1):
        logger.info(f"{rank}. {metric_name}={result['best_metric_value']:.4f}")
        logger.info(f"   {result['params']}")
    
    logger.info("=" * 80)


def save_search_results(search_results: list, best_params: dict, output_dir: str):
    """保存超参数搜索结果（包含训练元数据）"""
    result_save_path = os.path.join(output_dir, "mlp_search_result.json")
    
    os.makedirs(os.path.dirname(result_save_path) or '.', exist_ok=True)
    
    serializable_results = []
    for result in search_results:
        params = result['params'].copy()
        
        serializable_result = {
            'params': params,
            'best_metric_value': result['best_metric_value'],
            'epochs': result['epochs'],
            'best_metrics': result.get('best_metrics', {}),
        }
        
        if 'training_metadata' in result:
            serializable_result['training_metadata'] = result['training_metadata']
        
        serializable_results.append(serializable_result)
    
    best_params_serializable = best_params.copy() if best_params is not None else {}
    
    final_result = {
        'best_params': best_params_serializable,
        'search_results': serializable_results,
        'total_combinations': len(search_results),
        'best_metric_value': max(r['best_metric_value'] for r in search_results) if search_results else 0.0
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
        model_config_path=MODEL_CONFIG_PATH,
        trainer_config_path=TRAINER_CONFIG_PATH,
        param_grid=PARAM_GRID,
        output_dir=MODEL_SAVE_DIR,
        epochs_search=SEARCH_EPOCH
    )
    
    analyze_search_results(search_results)
    
    save_search_results(search_results, best_params, MODEL_SAVE_DIR)
    
    logger.info("\n" + "=" * 80)
    logger.info("超参数搜索完成！")
    logger.info("=" * 80)
    
    return best_params, search_results


if __name__ == "__main__":
    main_search()

