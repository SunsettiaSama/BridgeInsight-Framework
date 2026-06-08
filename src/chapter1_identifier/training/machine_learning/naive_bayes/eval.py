import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
import pickle
import os
import logging
from sklearn.metrics import accuracy_score
from config.machine_learning_module.naive_bayes.eval_config import NaiveBayesEvalConfig

logger = logging.getLogger(__name__)

CONFIG = NaiveBayesEvalConfig()
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.USE_GPU) else "cpu")

_SENTINEL = object()

# -------------------------- 核心函数 --------------------------
def load_nb_model(model_path):
    """
    加载训练好的朴素贝叶斯模型和标准化器
    :param model_path: 模型保存路径
    :return: nb_model, scaler
    :raise: 模型加载失败时抛出异常
    """
    try:
        logger.info(f"尝试加载模型：{model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        
        with open(model_path, 'rb') as f:
            saved_dict = pickle.load(f)
        
        nb_model = saved_dict["model"]
        scaler = saved_dict["scaler"]
        logger.info("模型加载成功")
        return nb_model, scaler
    except Exception as e:
        logger.error(f"模型加载失败：{str(e)}", exc_info=True)
        raise

def infer_naive_bayes(infer_dataloader, model_path, has_label=True, infer_result_path=_SENTINEL):
    """
    朴素贝叶斯推理流程
    :param infer_dataloader: 推理集DataLoader（返回 (data, label) 或仅data）
    :param model_path: 模型路径
    :param has_label: 推理数据是否含标签（有则计算准确率）
    :param infer_result_path: 推理结果保存路径，None 则不保存，不传则用配置默认
    :return: 推理结果字典
    """
    result_path = CONFIG.INFER_RESULT_PATH if infer_result_path is _SENTINEL else infer_result_path
    try:
        # 1. 加载模型和标准化器
        nb_model, scaler = load_nb_model(model_path)
        
        # 2. 提取推理数据
        logger.info("开始提取推理数据")
        infer_features = []
        infer_labels = []
        sample_ids = []  # 样本ID，用于溯源
        
        for idx, batch in enumerate(infer_dataloader):
            if has_label:
                batch_data, batch_label = batch
                batch_label = batch_label.to(DEVICE).cpu().numpy()
                infer_labels.extend(batch_label.tolist())
            else:
                batch_data = batch  # 无标签时仅返回data
            
            # 处理特征
            batch_data = batch_data.to(DEVICE).cpu().numpy()
            batch_data = batch_data.reshape(batch_data.shape[0], -1)  # 展平特征
            infer_features.append(batch_data)
            
            # 生成样本ID
            batch_size = batch_data.shape[0]
            sample_ids.extend([f"sample_{idx}_{i}" for i in range(batch_size)])
        
        # 合并特征并标准化
        infer_features = np.concatenate(infer_features, axis=0)
        infer_features_scaled = scaler.transform(infer_features)
        logger.info(f"推理数据提取完成：样本数{len(sample_ids)}，特征维度{infer_features.shape}")
        
        # 3. 推理预测
        logger.info("开始执行推理预测")
        infer_preds = nb_model.predict(infer_features_scaled)
        # 预测概率（高斯朴素贝叶斯支持predict_proba）
        infer_probs = nb_model.predict_proba(infer_features_scaled)
        max_probs = np.max(infer_probs, axis=1)  # 最大置信度
        
        # 4. 整理推理结果
        infer_result = {
            "sample_num": len(sample_ids),
            "class_num": infer_probs.shape[1],
            "predictions": [],
            "metrics": {}
        }
        
        # 逐个样本记录结果
        for idx, sample_id in enumerate(sample_ids):
            sample_result = {
                "sample_id": sample_id,
                "pred_label": int(infer_preds[idx]),
                "max_confidence": float(max_probs[idx]),
                "all_confidences": infer_probs[idx].tolist()
            }
            if has_label:
                sample_result["true_label"] = int(infer_labels[idx])
            infer_result["predictions"].append(sample_result)
        
        # 5. 计算指标（如有标签）
        if has_label and len(infer_labels) > 0:
            accuracy = float(accuracy_score(infer_labels, infer_preds))
            infer_result["metrics"]["accuracy"] = accuracy
            logger.info(f"推理准确率：{accuracy:.4f}")
        
        # 6. 保存推理结果（路径为 None 则不保存）
        if result_path is not None:
            os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(infer_result, f, ensure_ascii=False, indent=4)
            logger.info(f"推理结果已保存至：{result_path}")
        
        return infer_result
    except Exception as e:
        logger.error(f"推理过程失败：{str(e)}", exc_info=True)
        raise

# -------------------------- 测试用例 --------------------------
if __name__ == "__main__":
    # 模拟推理DataLoader（替换为你的真实数据）
    from torch.utils.data import Dataset, DataLoader
    
    class MockInferDataset(Dataset):
        def __len__(self):
            return 200
        
        def __getitem__(self, idx):
            data = torch.randn(10)  # 与训练集特征维度一致
            label = torch.randint(0, 10, (1,)).item()
            return data, label
    
    # 构建推理DataLoader
    infer_dataset = MockInferDataset()
    infer_dataloader = DataLoader(infer_dataset, batch_size=32, shuffle=False)
    
    # 启动推理
    logger.info("启动朴素贝叶斯推理流程")
    result = infer_naive_bayes(infer_dataloader, CONFIG.MODEL_LOAD_PATH, has_label=True)
    logger.info(f"推理流程完成：共处理{result['sample_num']}个样本，准确率{result['metrics']['accuracy']:.4f}")