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
from config.machine_learning_module.svm.eval_config import SVMEvalConfig

logger = logging.getLogger(__name__)

# 从配置中加载参数
CONFIG = SVMEvalConfig()
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.USE_GPU) else "cpu")

_SENTINEL = object()

# -------------------------- 核心函数 --------------------------
def load_svm_model(model_path):
    """
    加载训练好的SVM模型和标准化器
    :param model_path: 模型保存路径
    :return: svm_model, scaler
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    
    with open(model_path, 'rb') as f:
        saved_dict = pickle.load(f)
    
    return saved_dict["model"], saved_dict["scaler"]

def infer_svm(infer_dataloader, model_path, has_label=True, infer_result_path=_SENTINEL):
    """
    SVM推理过程
    :param infer_dataloader: 推理集DataLoader（返回 (data, label) 或仅 data）
    :param model_path: 训练好的模型路径
    :param has_label: 推理数据是否有标签（有则计算准确率）
    :param infer_result_path: 推理结果保存路径，None 则不保存，不传则用配置默认
    :return: 推理结果字典
    """
    result_path = CONFIG.INFER_RESULT_PATH if infer_result_path is _SENTINEL else infer_result_path
    # 1. 加载模型和标准化器
    logger.info("加载SVM模型...")
    svm_model, scaler = load_svm_model(model_path)
    
    # 2. 提取推理数据
    logger.info("提取推理数据...")
    infer_features = []
    infer_labels = []
    sample_ids = []
    
    for idx, batch in enumerate(infer_dataloader):
        if has_label:
            batch_data, batch_label = batch
            batch_label = batch_label.to(DEVICE).cpu().numpy()
            infer_labels.extend(batch_label.tolist())
        else:
            batch_data = batch
        
        batch_data = batch_data.to(DEVICE).cpu().numpy()
        batch_data = batch_data.reshape(batch_data.shape[0], -1)
        infer_features.append(batch_data)
        
        batch_size = batch_data.shape[0]
        sample_ids.extend([f"{idx}_{i}" for i in range(batch_size)])
    
    # 合并特征
    infer_features = np.concatenate(infer_features, axis=0)
    # 标准化特征（用训练集的scaler）
    infer_features = scaler.transform(infer_features)
    
    # 3. 推理预测
    logger.info("开始推理...")
    infer_preds = svm_model.predict(infer_features)
    # 预测概率（可选，返回每个类别的置信度）
    infer_probs = svm_model.predict_proba(infer_features)
    # 最大置信度
    max_probs = np.max(infer_probs, axis=1)
    
    # 4. 整理结果
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
    
    # 5. 计算推理指标（如果有标签）
    if has_label and len(infer_labels) > 0:
        infer_result["metrics"]["accuracy"] = float(accuracy_score(infer_labels, infer_preds))
    
    # 6. 保存推理结果（路径为 None 则不保存）
    if result_path is not None:
        os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(infer_result, f, ensure_ascii=False, indent=4)
        logger.info(f"推理结果已保存至：{result_path}")
    return infer_result

# -------------------------- 测试用例（示例） --------------------------
if __name__ == "__main__":
    # 替换为你自己的推理DataLoader
    from torch.utils.data import Dataset, DataLoader
    
    class MockInferDataset(Dataset):
        def __len__(self):
            return 200
        
        def __getitem__(self, idx):
            data = torch.randn(10)  # 同训练集的特征维度
            label = torch.randint(0, 10, (1,)).item()
            return data, label
    
    # 构建推理DataLoader
    infer_dataset = MockInferDataset()
    infer_dataloader = DataLoader(infer_dataset, batch_size=32, shuffle=False)
    
    # 启动推理
    result = infer_svm(infer_dataloader, CONFIG.MODEL_LOAD_PATH, has_label=True)
    print("推理完成，核心结果：")
    print(f"推理样本数：{result['sample_num']}")
    print(f"推理准确率：{result['metrics']['accuracy']:.4f}")
    print(f"第一个样本预测标签：{result['predictions'][0]['pred_label']}，真实标签：{result['predictions'][0]['true_label']}")