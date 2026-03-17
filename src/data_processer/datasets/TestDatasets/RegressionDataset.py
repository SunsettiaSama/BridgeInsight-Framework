import torch
import numpy as np
from pathlib import Path
from typing import Tuple
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processer.datasets.BaseDataset import BaseDataset
from src.config.data_processer.datasets.TestDatasets.RegressionDatasetConfig import RegressionDatasetConfig


logger = logging.getLogger(__name__)


class RegressionDataset(BaseDataset):
    """
    回归测试数据集（sin -> arctan映射）
    
    特性：
    - 在线生成数据（无需预先存储）
    - 输入：sin函数的时序数据
    - 输出：对应sin函数值映射到arctan函数的目标值
    - 返回格式：(input_tensor, target_tensor)
    """
    
    def __init__(self, config: RegressionDatasetConfig):
        self.config = config
        self.seq_length = config.seq_length
        self.amplitude = config.amplitude
        self.frequency = config.frequency
        self.noise_std = config.noise_std
        self.num_samples = config.num_samples
        
        # 设置数据集模式
        self._dataset_mode = "full"
        self.full_file_paths = list(range(self.num_samples))
        
        # 缓存配置
        self.cache_in_memory = config.use_cache
        self._cache = {} if self.cache_in_memory else None
        
        logger.info(f"回归数据集初始化完成：样本数{self.num_samples}，序列长度{self.seq_length}")
    
    def _parse_sample(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
            (input_tensor, target_tensor)
            input_tensor: sin函数数据 (1, seq_length)
            target_tensor: arctan映射目标 (1, seq_length)
        """
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]
        
        np.random.seed(idx)
        
        x = np.linspace(0, 4 * np.pi, self.seq_length)
        
        sin_data = self.amplitude * np.sin(self.frequency * x)
        sin_data = sin_data + np.random.normal(0, self.noise_std, self.seq_length)
        
        x_scaled = (x - x.mean()) / (x.std() + 1e-8)
        arctan_target = self.amplitude * np.arctan(x_scaled)
        
        input_tensor = torch.from_numpy(sin_data.astype(np.float32)).unsqueeze(0)
        target_tensor = torch.from_numpy(arctan_target.astype(np.float32)).unsqueeze(0)
        
        result = (input_tensor, target_tensor)
        
        if self.cache_in_memory:
            self._cache[idx] = result
        
        return result
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self._cache:
            self._cache.clear()
            logger.info("回归数据集缓存已清空")
