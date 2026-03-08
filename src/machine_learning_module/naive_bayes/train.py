import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
import pickle
import os
import logging
import time
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from config.machine_learning_module.naive_bayes.train_config import NaiveBayesTrainConfig

logger = logging.getLogger(__name__)

CONFIG = NaiveBayesTrainConfig()
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.USE_GPU) else "cpu")

_SENTINEL = object()

# -------------------------- 核心函数 --------------------------
def extract_data_from_dataloader(dataloader):
    """
    从PyTorch DataLoader提取特征和标签，转换为numpy数组
    :param dataloader: PyTorch DataLoader，返回 (data, label)
    :return: features (np.array), labels (np.array)
    :raise: 数据提取失败时抛出异常并记录日志
    """
    try:
        logger.info("开始提取DataLoader中的数据")
        features, labels = [], []
        for batch_data, batch_label in dataloader:
            # 转移到CPU并转换为numpy
            batch_data = batch_data.to(DEVICE).cpu().numpy()
            batch_label = batch_label.to(DEVICE).cpu().numpy()
            
            # 展平特征（适配朴素贝叶斯一维输入）
            batch_data = batch_data.reshape(batch_data.shape[0], -1)
            
            features.append(batch_data)
            labels.append(batch_label)
        
        # 合并批次
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        logger.info(f"数据提取完成：特征维度{features.shape}，样本数{len(labels)}")
        return features, labels
    except Exception as e:
        logger.error(f"数据提取失败：{str(e)}", exc_info=True)
        raise

def train_naive_bayes(train_dataloader, val_dataloader=None, model_save_path=_SENTINEL, result_save_path=_SENTINEL):
    """
    训练朴素贝叶斯模型，评估性能并保存结果
    :param train_dataloader: 训练集DataLoader
    :param val_dataloader: 验证集DataLoader（可选）
    :param model_save_path: 模型保存路径，None 则不保存，不传则用配置默认
    :param result_save_path: 训练结果保存路径，None 则不保存，不传则用配置默认
    :return: 训练结果字典
    """
    model_path = CONFIG.MODEL_SAVE_PATH if model_save_path is _SENTINEL else model_save_path
    result_path = CONFIG.RESULT_SAVE_PATH if result_save_path is _SENTINEL else result_save_path
    try:
        # 1. 提取训练数据
        train_features, train_labels = extract_data_from_dataloader(train_dataloader)
        
        # 2. 特征标准化（高斯朴素贝叶斯对尺度敏感，提升性能）
        logger.info("开始特征标准化处理")
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        
        # 3. 初始化高斯朴素贝叶斯模型
        logger.info("初始化高斯朴素贝叶斯模型")
        nb_model = GaussianNB()
        
        # 4. 训练模型
        logger.info("开始训练朴素贝叶斯模型")
        nb_model.fit(train_features_scaled, train_labels)
        
        # 5. 评估训练集性能
        logger.info("评估训练集性能")
        train_pred = nb_model.predict(train_features_scaled)
        train_metrics = {
            "accuracy": float(accuracy_score(train_labels, train_pred)),
            "precision": float(precision_score(train_labels, train_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(train_labels, train_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(train_labels, train_pred, average='weighted', zero_division=0))
        }
        logger.info(f"训练集指标：{train_metrics}")
        
        # 6. 评估验证集（如有）
        val_metrics = {}
        if val_dataloader is not None:
            logger.info("评估验证集性能")
            val_features, val_labels = extract_data_from_dataloader(val_dataloader)
            val_features_scaled = scaler.transform(val_features)
            val_pred = nb_model.predict(val_features_scaled)
            val_metrics = {
                "accuracy": float(accuracy_score(val_labels, val_pred)),
                "precision": float(precision_score(val_labels, val_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(val_labels, val_pred, average='weighted', zero_division=0)),
                "f1": float(f1_score(val_labels, val_pred, average='weighted', zero_division=0))
            }
            logger.info(f"验证集指标：{val_metrics}")
        
        # 7. 整理训练结果
        train_result = {
            "model_type": "GaussianNB",
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_samples": len(train_features),
            "val_samples": len(val_features) if val_dataloader is not None else 0,
            "feature_dim": train_features.shape[1],
            "class_num": len(np.unique(train_labels)),
            "train_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # 8. 保存模型和结果（路径为 None 则不保存）
        if model_path is not None:
            os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump({"model": nb_model, "scaler": scaler}, f)
            logger.info(f"模型已保存至：{model_path}")
        if result_path is not None:
            os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(train_result, f, ensure_ascii=False, indent=4)
            logger.info(f"训练结果已保存至：{result_path}")
        
        return train_result
    except Exception as e:
        logger.error(f"训练过程失败：{str(e)}", exc_info=True)
        raise

# -------------------------- 测试用例 --------------------------
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    
    class MockDataset(Dataset):
        def __len__(self):
            return 1000
        
        def __getitem__(self, idx):
            data = torch.randn(10)
            label = torch.randint(0, 10, (1,)).item()
            return data, label
    
    train_dataset = MockDataset()
    val_dataset = MockDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    logger.info("启动朴素贝叶斯训练流程")
    result = train_naive_bayes(train_dataloader, val_dataloader)
    logger.info(f"训练流程完成，核心结果：训练集准确率{result['train_metrics']['accuracy']:.4f}，验证集准确率{result['val_metrics']['accuracy']:.4f}")