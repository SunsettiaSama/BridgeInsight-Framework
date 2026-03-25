import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
import pickle
import os
import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from config.machine_learning_module.svm.train_config import SVMTrainConfig
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 从配置中加载参数
CONFIG = SVMTrainConfig()
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.USE_GPU) else "cpu")

_SENTINEL = object()

# -------------------------- 核心函数 --------------------------
def _process_batch(batch_data, batch_label):
    """处理单个批次数据的辅助函数"""
    batch_data = batch_data.to(DEVICE).cpu().numpy()
    batch_label = batch_label.to(DEVICE).cpu().numpy()
    batch_data = batch_data.reshape(batch_data.shape[0], -1)
    return batch_data, batch_label


def extract_data_from_dataloader(dataloader, num_workers: int = 4, use_progress_bar: bool = True):
    """
    从PyTorch DataLoader中提取特征和标签，转换为numpy数组（适配SVM）
    
    :param dataloader: PyTorch DataLoader，返回格式为 (data, label)
    :param num_workers: 多线程工作进程数（默认4）
    :param use_progress_bar: 是否显示进度条（默认True）
    :return: features (np.array), labels (np.array)
    """
    features = []
    labels = []
    
    total_batches = len(dataloader)
    
    if use_progress_bar:
        pbar = tqdm(dataloader, total=total_batches, desc="提取数据", unit="batch")
    else:
        pbar = dataloader
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for batch_data, batch_label in pbar:
            future = executor.submit(_process_batch, batch_data, batch_label)
            futures.append(future)
        
        for future in tqdm(futures, desc="处理结果", unit="batch", disable=not use_progress_bar):
            batch_data, batch_label = future.result()
            features.append(batch_data)
            labels.append(batch_label)
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    logger.info(f"数据提取完成: {len(features)} 个样本，特征维度 {features.shape[1]}")
    
    return features, labels

def train_svm(train_dataloader, val_dataloader=None, model_save_path=_SENTINEL, result_save_path=_SENTINEL):
    """
    训练SVM模型，并评估性能
    :param train_dataloader: 训练集DataLoader
    :param val_dataloader: 验证集DataLoader（可选）
    :param model_save_path: 模型保存路径，None 则不保存，不传则用配置默认
    :param result_save_path: 训练结果保存路径，None 则不保存，不传则用配置默认
    :return: 训练结果字典
    """
    model_path = CONFIG.MODEL_SAVE_PATH if model_save_path is _SENTINEL else model_save_path
    result_path = CONFIG.RESULT_SAVE_PATH if result_save_path is _SENTINEL else result_save_path
    # 1. 提取训练数据
    logger.info("开始提取训练数据...")
    train_features, train_labels = extract_data_from_dataloader(train_dataloader)
    
    # 2. 特征标准化（SVM关键步骤，不做会严重影响性能）
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    
    # 3. 初始化SVM模型（可根据需求调整参数）
    svm_model = SVC(
        kernel=CONFIG.KERNEL,
        C=CONFIG.C,
        gamma=CONFIG.GAMMA,
        probability=CONFIG.PROBABILITY,
        random_state=CONFIG.RANDOM_STATE
    )
    
    # 4. 训练模型
    logger.info("开始训练SVM模型...")
    svm_model.fit(train_features, train_labels)
    
    # 5. 评估训练集性能
    logger.info("开始评估训练集性能...")
    train_pred = svm_model.predict(train_features)
    train_metrics = {
        "accuracy": float(accuracy_score(train_labels, train_pred)),
        "precision": float(precision_score(train_labels, train_pred, average='weighted')),
        "recall": float(recall_score(train_labels, train_pred, average='weighted')),
        "f1": float(f1_score(train_labels, train_pred, average='weighted'))
    }
    
    # 6. 评估验证集（如果有）
    val_metrics = {}
    if val_dataloader is not None:
        logger.info("开始评估验证集性能...")
        val_features, val_labels = extract_data_from_dataloader(val_dataloader)
        val_features = scaler.transform(val_features)  # 用训练集的scaler标准化
        val_pred = svm_model.predict(val_features)
        val_metrics = {
            "accuracy": float(accuracy_score(val_labels, val_pred)),
            "precision": float(precision_score(val_labels, val_pred, average='weighted')),
            "recall": float(recall_score(val_labels, val_pred, average='weighted')),
            "f1": float(f1_score(val_labels, val_pred, average='weighted'))
        }
    
    # 7. 整理训练结果
    train_result = {
        "model_params": {
            "kernel": svm_model.kernel,
            "C": float(svm_model.C),
            "gamma": svm_model.gamma
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_samples": len(train_features),
        "val_samples": len(val_features) if val_dataloader is not None else 0,
        "feature_dim": train_features.shape[1],
        "class_num": len(np.unique(train_labels))
    }
    
    # 8. 保存模型和结果（路径为 None 则不保存）
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({"model": svm_model, "scaler": scaler}, f)
        logger.info(f"模型已保存至：{model_path}")
    if result_path is not None:
        os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(train_result, f, ensure_ascii=False, indent=4)
        logger.info(f"训练结果已保存至：{result_path}")
    
    return train_result

# -------------------------- 测试用例（示例） --------------------------
if __name__ == "__main__":
    # 这里替换为你自己的DataLoader（示例用随机数据模拟）
    # 1. 模拟PyTorch Dataset和DataLoader
    from torch.utils.data import Dataset, DataLoader
    
    class MockDataset(Dataset):
        def __len__(self):
            return 1000
        
        def __getitem__(self, idx):
            # 模拟特征：10维向量，标签：0-9的整数
            data = torch.randn(10)  # 特征维度10
            label = torch.randint(0, 10, (1,)).item()
            return data, label
    
    # 构建训练/验证DataLoader
    train_dataset = MockDataset()
    val_dataset = MockDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 启动训练
    result = train_svm(train_dataloader, val_dataloader)
    print("训练完成，核心结果：")
    print(f"训练集准确率：{result['train_metrics']['accuracy']:.4f}")
    print(f"验证集准确率：{result['val_metrics']['accuracy']:.4f}")