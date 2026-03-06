# ====================== 1. 导入核心库 ======================
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader  # 对齐PyTorch数据加载框架
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple, Dict  # 类型注解提升可读性

# ====================== 2. 泛化配置（修改为高斯分布测试） ======================
# 数据集配置（新增gaussian类型，无需外部文件）
DATA_CONFIG = {
    "dataset_type": "gaussian",         # 改为高斯分布生成
    "csv_path": "your_dataset.csv",     # 失效（仅gaussian生效）
    "feature_cols": None,               # 失效
    "label_col": None,                  # 失效
    "test_size": 0.3,
    "random_state": 42,
    "batch_size": 32,                   # DataLoader批次大小
    # 高斯分布参数（新增）
    "gaussian_params": {
        "n_samples_per_class": 500,     # 每类样本数
        "class0_mean": [0, 0],          # 类别0均值（2维特征）
        "class0_cov": [[1, 0.5], [0.5, 1]],  # 类别0协方差
        "class1_mean": [3, 3],          # 类别1均值
        "class1_cov": [[1, 0.5], [0.5, 1]]   # 类别1协方差
    }
}

# SVM模型配置（保持默认）
SVM_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "random_state": 42
}

# ====================== 3. 自定义Dataset（对齐PyTorch风格） ======================
class CustomClassificationDataset(Dataset):
    """
    泛化分类数据集类，继承PyTorch Dataset，适配任意结构化数据
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
        """
        self.features = features.astype(np.float32)  # 统一数据类型
        self.labels = labels.astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        核心接口：返回单样本（对齐PyTorch __getitem__）
        Returns:
            dict: {"features": 单样本特征, "labels": 单样本标签}
        """
        return {
            "features": self.features[idx],
            "labels": self.labels[idx]
        }

    def __len__(self) -> int:
        """核心接口：返回数据集总样本数"""
        return len(self.features)

# ====================== 4. 数据预处理类（新增高斯分布生成逻辑） ======================
class DataPreprocessor:
    """泛化数据预处理类，封装拆分、标准化逻辑"""
    def __init__(self, test_size: float, random_state: int):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()  # 标准化器（SVM核心）
        # 固定随机种子，保证生成数据可复现
        np.random.seed(self.random_state)

    def _generate_gaussian_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成双高斯分布的二类分类数据"""
        params = DATA_CONFIG["gaussian_params"]
        # 生成类别0数据
        X0 = np.random.multivariate_normal(
            mean=params["class0_mean"],
            cov=params["class0_cov"],
            size=params["n_samples_per_class"]
        )
        y0 = np.zeros(params["n_samples_per_class"], dtype=int)
        
        # 生成类别1数据
        X1 = np.random.multivariate_normal(
            mean=params["class1_mean"],
            cov=params["class1_cov"],
            size=params["n_samples_per_class"]
        )
        y1 = np.ones(params["n_samples_per_class"], dtype=int)
        
        # 合并数据并打乱
        X = np.vstack((X0, X1))
        y = np.hstack((y0, y1))
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        return X, y

    def load_and_split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载/生成并拆分数据集（新增gaussian分支）"""
        # 生成双高斯分布数据
        if DATA_CONFIG["dataset_type"] == "gaussian":
            X, y = self._generate_gaussian_data()
        # 加载sklearn内置数据集（保留）
        elif DATA_CONFIG["dataset_type"] == "sklearn_builtin":
            from sklearn import datasets
            iris = datasets.load_iris()
            X, y = iris.data, iris.target
        # 加载自定义CSV数据集（保留）
        elif DATA_CONFIG["dataset_type"] == "csv":
            df = pd.read_csv(DATA_CONFIG["csv_path"])
            X = df[DATA_CONFIG["feature_cols"]].values
            y = df[DATA_CONFIG["label_col"]].values
        else:
            raise ValueError(f"不支持的数据集类型：{DATA_CONFIG['dataset_type']}")
        
        # 拆分训练/测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def process(self) -> Tuple[CustomClassificationDataset, CustomClassificationDataset]:
        """完整预处理流程：加载→拆分→标准化→封装为Dataset"""
        X_train, X_test, y_train, y_test = self.load_and_split_data()
        
        # 标准化（训练集拟合+转换，测试集仅转换）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 封装为PyTorch风格的Dataset
        train_dataset = CustomClassificationDataset(X_train_scaled, y_train)
        test_dataset = CustomClassificationDataset(X_test_scaled, y_test)
        
        return train_dataset, test_dataset

# ====================== 5. 工具函数（从DataLoader提取完整数据） ======================
def extract_data_from_dataloader(dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    从PyTorch DataLoader中提取完整的特征和标签（适配SVM批量训练特性）
    Args:
        dataloader: 数据加载器
    Returns:
        X: 完整特征矩阵, y: 完整标签数组
    """
    all_features = []
    all_labels = []
    for batch in dataloader:
        all_features.append(batch["features"].numpy())  # 从tensor转numpy（适配sklearn）
        all_labels.append(batch["labels"].numpy())
    # 拼接所有batch
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

# ====================== 6. 核心训练/评估流程 ======================
def main():
    # 1. 初始化预处理类并生成Dataset
    preprocessor = DataPreprocessor(
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"]
    )
    train_dataset, test_dataset = preprocessor.process()
    
    # 打印数据基本信息（验证生成）
    print(f"训练集样本数：{len(train_dataset)} | 测试集样本数：{len(test_dataset)}")
    print(f"特征维度：{train_dataset[0]['features'].shape[0]} | 类别数：{len(np.unique([sample['labels'] for sample in train_dataset]))}")
    
    # 2. 构建PyTorch DataLoader（对齐框架核心）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=True,  # 训练集打乱
        num_workers=0  # 新手建议设0，避免多进程问题
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=False,  # 测试集不打乱
        num_workers=0
    )
    
    # 3. 从DataLoader提取完整数据（SVM需全量训练）
    X_train, y_train = extract_data_from_dataloader(train_dataloader)
    X_test, y_test = extract_data_from_dataloader(test_dataloader)
    
    # 4. 初始化并训练SVM模型
    svm_model = SVC(**SVM_CONFIG)
    svm_model.fit(X_train, y_train)
    
    # 5. 模型评估
    y_pred = svm_model.predict(X_test)
    print("\n" + "="*50 + " 模型评估结果 " + "="*50)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print("\n分类报告:\n", classification_report(y_test, y_pred))
    
    # 6. 预测新样本（高斯分布示例）
    print("\n" + "="*50 + " 新样本预测 " + "="*50)
    # 生成2个新样本（分别贴近类别0和类别1）
    new_samples = np.array([[0.1, 0.2], [2.9, 3.1]])
    new_samples_scaled = preprocessor.scaler.transform(new_samples)  # 标准化
    pred_labels = svm_model.predict(new_samples_scaled)
    for i, (sample, label) in enumerate(zip(new_samples, pred_labels)):
        print(f"新样本{i+1}特征: {sample} | 预测标签: {label} (类别{label})")

if __name__ == "__main__":
    main()