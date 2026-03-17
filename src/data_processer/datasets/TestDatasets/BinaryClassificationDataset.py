import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Union
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processer.datasets.BaseDataset import BaseDataset
from src.config.data_processer.datasets.TestDatasets.BinaryClassificationDatasetConfig import BinaryClassificationDatasetConfig


logger = logging.getLogger(__name__)


class BinaryClassificationDataset(BaseDataset):
    """
    二分类测试数据集（sin函数 vs arctan函数）
    
    特性：
    - 在线生成数据（无需预先存储）
    - sin函数映射为类型0
    - arctan函数映射为类型1
    - 返回格式：(input_tensor, label_tensor)
    """
    
    def __init__(self, config: BinaryClassificationDatasetConfig):
        self.config = config
        self.seq_length = config.seq_length
        self.amplitude = config.amplitude
        self.frequency = config.frequency
        self.noise_std = config.noise_std
        self.num_samples_per_class = config.num_samples_per_class
        self.num_classes = config.num_classes
        
        # 设置数据集模式
        self._dataset_mode = "full"
        self.full_file_paths = list(range(self.num_samples_per_class * self.num_classes))
        
        # 缓存配置
        self.cache_in_memory = config.use_cache
        self._cache = {} if self.cache_in_memory else None
        
        logger.info(f"二分类数据集初始化完成：每类{self.num_samples_per_class}个样本，序列长度{self.seq_length}")
    
    def _generate_sin_data(self, idx: int) -> Tuple[np.ndarray, int]:
        """生成sin函数数据（类型0）"""
        np.random.seed(idx)
        x = np.linspace(0, 4 * np.pi, self.seq_length)
        y = self.amplitude * np.sin(self.frequency * x)
        y = y + np.random.normal(0, self.noise_std, self.seq_length)
        return y.astype(np.float32), 0
    
    def _generate_arctan_data(self, idx: int) -> Tuple[np.ndarray, int]:
        """生成arctan函数数据（类型1）"""
        np.random.seed(idx)
        x = np.linspace(-5, 5, self.seq_length)
        y = self.amplitude * np.arctan(x)
        y = y + np.random.normal(0, self.noise_std, self.seq_length)
        return y.astype(np.float32), 1
    
    def _parse_sample(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        解析样本（实现基类抽象方法）
        由于数据在线生成，此方法不实际使用
        """
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取样本
        Args:
            idx: 样本索引
        Returns:
            (input_tensor, label_tensor)
        """
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]
        
        if idx < self.num_samples_per_class:
            data, label = self._generate_sin_data(idx)
        else:
            data, label = self._generate_arctan_data(idx - self.num_samples_per_class)
        
        input_tensor = torch.from_numpy(data).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        result = (input_tensor, label_tensor)
        
        if self.cache_in_memory:
            self._cache[idx] = result
        
        return result
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples_per_class * self.num_classes
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self._cache:
            self._cache.clear()
            logger.info("二分类数据集缓存已清空")
