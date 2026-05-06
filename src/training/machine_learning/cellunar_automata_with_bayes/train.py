import torch
import numpy as np
import json
import pickle
import os
import logging
from scipy.stats import entropy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# -------------------------- 日志配置 --------------------------
def setup_logger():
    """配置训练日志（控制台+文件）"""
    logger = logging.getLogger("ca_nb_train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    
    # 控制台+文件处理器
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("ca_nb_train.log", encoding="utf-8")
    
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# -------------------------- 配置参数 --------------------------
# CA参数（可根据数据调整）
CA_GRID_SIZE = 128  # 元胞网格长度（一维）
CA_RULE = 30        # CA演化规则（经典Rule 30，可换110/184等）
CA_EVOLVE_STEPS = 10  # 演化步数
# 保存路径
_RESULTS_BASE = "results/classification_results/machine_learning/ca_bayes"
CA_PARAMS_PATH = f"{_RESULTS_BASE}/ca_params.pkl"
NB_MODEL_PATH = f"{_RESULTS_BASE}/ca_nb_model.pkl"
TRAIN_RESULT_PATH = f"{_RESULTS_BASE}/ca_nb_train_result.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SENTINEL = object()

# -------------------------- 元胞自动机核心实现 --------------------------
def rule30(neighborhood):
    """Rule 30 演化规则（一维CA核心）：3位邻居→新状态"""
    # neighborhood: [左, 中, 右] → 转十进制 → 查Rule30映射
    rule_map = {
        0b111: 0, 0b110: 0, 0b101: 0, 0b100: 1,
        0b011: 1, 0b010: 1, 0b001: 1, 0b000: 0
    }
    val = int(''.join(map(str, neighborhood)), 2)
    return rule_map[val]

def init_ca_grid(feature_vector, grid_size=CA_GRID_SIZE):
    """
    将特征向量映射为CA初始网格（归一化→二值化）
    :param feature_vector: 一维特征向量 (n_features,)
    :param grid_size: CA网格长度
    :return: 一维CA初始网格 (grid_size,)
    """
    # 归一化到[0,1]
    feat_norm = (feature_vector - feature_vector.min()) / (feature_vector.max() - feature_vector.min() + 1e-8)
    # 插值到网格大小
    grid = np.interp(np.linspace(0, len(feat_norm)-1, grid_size), 
                     np.arange(len(feat_norm)), feat_norm)
    # 二值化（0/1）
    grid = (grid > 0.5).astype(int)
    return grid

def ca_evolve(grid, steps=CA_EVOLVE_STEPS, rule=rule30):
    """
    一维CA演化
    :param grid: 初始网格 (grid_size,)
    :param steps: 演化步数
    :param rule: 演化规则函数
    :return: 演化后的所有状态 (steps+1, grid_size)
    """
    grid_size = len(grid)
    history = [grid.copy()]
    current = grid.copy()
    
    for _ in range(steps):
        next_grid = np.zeros(grid_size, dtype=int)
        # 遍历每个元胞（边界用0填充）
        for i in range(grid_size):
            left = current[i-1] if i > 0 else 0
            mid = current[i]
            right = current[i+1] if i < grid_size-1 else 0
            next_grid[i] = rule([left, mid, right])
        current = next_grid
        history.append(current.copy())
    
    return np.array(history)

def extract_ca_features(ca_history):
    """
    从CA演化历史提取统计特征（核心：把CA模式转成可分类的数值特征）
    :param ca_history: CA演化历史 (steps+1, grid_size)
    :return: 提取的特征向量 (n_features,)
    """
    # 1. 稳态密度（最后一步1的占比）
    final_grid = ca_history[-1]
    density = np.mean(final_grid)
    
    # 2. 熵（稳态分布的不确定性）
    counts = np.bincount(final_grid, minlength=2)
    entropy_val = entropy(counts / counts.sum())
    
    # 3. 最大连通域长度（稳态中连续1的最长长度）
    consecutive_1s = []
    current_len = 0
    for val in final_grid:
        if val == 1:
            current_len += 1
        else:
            consecutive_1s.append(current_len)
            current_len = 0
    consecutive_1s.append(current_len)
    max_conn = max(consecutive_1s) if consecutive_1s else 0
    
    # 4. 演化活跃度（不同步数间的差异总和）
    activity = np.sum(np.abs(np.diff(ca_history, axis=0))) / (ca_history.shape[0] * ca_history.shape[1])
    
    # 5. 全局均值（所有演化步骤的均值）
    global_mean = np.mean(ca_history)
    
    # 拼接特征
    features = np.array([density, entropy_val, max_conn, activity, global_mean])
    return features

# -------------------------- 数据处理 & 训练核心函数 --------------------------
def process_dataloader_to_ca_features(dataloader):
    """
    从DataLoader提取数据 → 转CA → 提取CA特征
    :param dataloader: PyTorch DataLoader (data, label)
    :return: ca_features (n_samples, n_ca_features), labels (n_samples,)
    """
    try:
        logger.info("开始从DataLoader提取数据并转换为CA特征")
        ca_features_list = []
        labels_list = []
        
        for batch_data, batch_label in dataloader:
            # 转numpy + 展平特征
            batch_data = batch_data.to(DEVICE).cpu().numpy().reshape(batch_data.shape[0], -1)
            batch_label = batch_label.to(DEVICE).cpu().numpy()
            
            # 逐个样本处理
            for feat, label in zip(batch_data, batch_label):
                # 1. 初始化CA网格
                ca_grid = init_ca_grid(feat)
                # 2. CA演化
                ca_hist = ca_evolve(ca_grid)
                # 3. 提取CA特征
                ca_feat = extract_ca_features(ca_hist)
                ca_features_list.append(ca_feat)
                labels_list.append(label)
        
        ca_features = np.array(ca_features_list)
        labels = np.array(labels_list)
        logger.info(f"CA特征提取完成：样本数{len(labels)}，特征维度{ca_features.shape}")
        return ca_features, labels
    except Exception as e:
        logger.error(f"CA特征提取失败：{str(e)}", exc_info=True)
        raise

def train_ca_nb(train_dataloader, val_dataloader=None, ca_params_path=_SENTINEL, nb_model_path=_SENTINEL, result_save_path=_SENTINEL):
    """
    :param ca_params_path: CA参数保存路径，None 则不保存，不传则用默认
    :param nb_model_path: NB模型保存路径，None 则不保存，不传则用默认
    :param result_save_path: 训练结果保存路径，None 则不保存，不传则用默认
    """
    ca_path = CA_PARAMS_PATH if ca_params_path is _SENTINEL else ca_params_path
    nb_path = NB_MODEL_PATH if nb_model_path is _SENTINEL else nb_model_path
    res_path = TRAIN_RESULT_PATH if result_save_path is _SENTINEL else result_save_path
    """
    训练CA+朴素贝叶斯模型
    :param train_dataloader: 训练集DataLoader
    :param val_dataloader: 验证集DataLoader（可选）
    :return: 训练结果字典
    """
    try:
        # 1. 提取CA特征
        train_ca_feat, train_labels = process_dataloader_to_ca_features(train_dataloader)
        
        # 2. 特征标准化（提升朴素贝叶斯性能）
        logger.info("开始CA特征标准化")
        scaler = StandardScaler()
        train_ca_feat_scaled = scaler.fit_transform(train_ca_feat)
        
        # 3. 初始化朴素贝叶斯模型
        logger.info("初始化高斯朴素贝叶斯模型")
        nb_model = GaussianNB()
        
        # 4. 训练模型
        logger.info("开始训练朴素贝叶斯模型")
        nb_model.fit(train_ca_feat_scaled, train_labels)
        
        # 5. 评估训练集性能
        logger.info("评估训练集性能")
        train_pred = nb_model.predict(train_ca_feat_scaled)
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
            val_ca_feat, val_labels = process_dataloader_to_ca_features(val_dataloader)
            val_ca_feat_scaled = scaler.transform(val_ca_feat)
            val_pred = nb_model.predict(val_ca_feat_scaled)
            val_metrics = {
                "accuracy": float(accuracy_score(val_labels, val_pred)),
                "precision": float(precision_score(val_labels, val_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(val_labels, val_pred, average='weighted', zero_division=0)),
                "f1": float(f1_score(val_labels, val_pred, average='weighted', zero_division=0))
            }
            logger.info(f"验证集指标：{val_metrics}")
        
        # 7. 保存模型和参数（路径为 None 则不保存）
        ca_params = {
            "grid_size": CA_GRID_SIZE,
            "rule": CA_RULE,
            "evolve_steps": CA_EVOLVE_STEPS
        }
        if nb_path is not None:
            os.makedirs(os.path.dirname(nb_path) or '.', exist_ok=True)
            with open(nb_path, 'wb') as f:
                pickle.dump({"nb_model": nb_model, "scaler": scaler}, f)
            logger.info(f"NB模型已保存至：{nb_path}")
        if ca_path is not None:
            os.makedirs(os.path.dirname(ca_path) or '.', exist_ok=True)
            with open(ca_path, 'wb') as f:
                pickle.dump(ca_params, f)
            logger.info(f"CA参数已保存至：{ca_path}")
        
        # 8. 整理训练结果
        train_result = {
            "ca_params": ca_params,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_samples": len(train_labels),
            "val_samples": len(val_labels) if val_dataloader is not None else 0,
            "ca_feature_dim": train_ca_feat.shape[1],
            "class_num": len(np.unique(train_labels))
        }
        if res_path is not None:
            os.makedirs(os.path.dirname(res_path) or '.', exist_ok=True)
            with open(res_path, 'w', encoding='utf-8') as f:
                json.dump(train_result, f, ensure_ascii=False, indent=4)
            logger.info(f"训练结果已保存至：{res_path}")
        
        return train_result
    except Exception as e:
        logger.error(f"训练过程失败：{str(e)}", exc_info=True)
        raise

# -------------------------- 测试用例 --------------------------
if __name__ == "__main__":
    # 模拟PyTorch DataLoader（替换为你的真实数据）
    from torch.utils.data import Dataset, DataLoader
    
    class MockDataset(Dataset):
        def __len__(self):
            return 1000
        
        def __getitem__(self, idx):
            # 模拟10维特征 + 0-9分类标签
            data = torch.randn(10)
            label = torch.randint(0, 10, (1,)).item()
            return data, label
    
    # 构建DataLoader
    train_dataset = MockDataset()
    val_dataset = MockDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 启动训练
    logger.info("启动CA+朴素贝叶斯训练流程")
    result = train_ca_nb(train_dataloader, val_dataloader)
    logger.info(f"训练完成：训练集准确率{result['train_metrics']['accuracy']:.4f}，验证集准确率{result['val_metrics']['accuracy']:.4f}")