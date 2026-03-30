import os
from typing import List, Optional, Tuple, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Subset
import copy
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.data_processer.datasets.data_factory import BaseDatasetConfig

# 类型变量，用于约束数据集实例类型
DatasetType = TypeVar("DatasetType", bound="BaseDataset")
# 配置日志（替代print，更规范可配置）
logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC, Generic[DatasetType]):
    """
    通用数据集基类（适配新Config体系，支持自动划分控制+独立实例获取）
    核心职责：
    1.  从 BaseDatasetConfig 实例读取通用参数（路径/缓存/采样/划分开关）；
    2.  封装「文件夹遍历+文件路径筛选+样本限制」核心逻辑；
    3.  支持可配置的自动划分（开关控制），兼容官方划分/自定义划分；
    4.  提供独立训练/验证/测试集实例获取能力，底层数据共享+差异化配置；
    5.  封装缓存管理通用能力，支持内存缓存独立维护；
    6.  定义抽象方法，强制子类实现数据解析/预处理逻辑；
    子类仅需实现：_parse_sample（解析单样本）、__getitem__（适配任务格式）
    """
    def __init__(self, config: BaseDatasetConfig):
        # 1. 从Config实例解析通用参数（强类型+已校验，无需二次验证）
        self.config = config
        self.data_dir = Path(config.data_dir)  # 统一转为Path对象（现代路径处理）
        self.max_samples = config.max_samples
        self.cache_in_memory = config.cache_in_memory
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        
        # 新增：自动划分控制参数（核心开关）
        self.auto_split = config.auto_split  # 布尔值，默认True
        self._dataset_mode: str = "full"  # 标识当前实例模式：full/train/val/test

        # 2. 核心逻辑：加载并筛选文件路径（完整数据集路径）
        self.full_file_paths = self._get_file_list()
        # 初始化划分后路径（默认与完整路径一致，关闭划分时使用）
        self.train_paths: List[Path] = self.full_file_paths.copy()
        self.val_paths: List[Path] = []
        self.test_paths: List[Path] = []

        # 3. 数据划分（通用能力：训练/验证/测试集拆分，受auto_split控制）
        if self.auto_split:
            self.train_paths, self.val_paths, self.test_paths = self._split_dataset()

        # 4. 内存缓存初始化（按需开启，每个实例独立维护缓存）
        self._cache = {} if self.cache_in_memory else None

        # 日志提示初始化状态
        self._log_init_info()

    def _get_file_list(self) -> List[Path]:
        """
        优化版文件遍历：支持后缀筛选、样本数限制，基于Pathlib实现
        Returns:
            筛选后的文件路径列表（Path对象）
        Raises:
            FileNotFoundError: 数据目录不存在/无有效文件时触发
        """
        # 基础校验：目录存在性
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据集根目录不存在：{self.data_dir.absolute()}")
        
        # 遍历目录：仅保留文件（排除文件夹）
        file_paths = [p for p in self.data_dir.iterdir() if p.is_file()]
        
        # 校验：是否有有效文件
        if not file_paths:
            raise FileNotFoundError(f"数据集目录「{self.data_dir}」下无任何文件")
        
        # 样本数限制（调试用）
        if self.max_samples and len(file_paths) > self.max_samples:
            file_paths = file_paths[:self.max_samples]
            logger.info(f"调试模式：限制样本数为 {self.max_samples}")
        
        return file_paths

    def _split_dataset(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        通用数据划分逻辑（对齐Config的划分配置，仅当auto_split=True时生效）
        Returns:
            train_paths: 训练集文件路径
            val_paths: 验证集文件路径
            test_paths: 测试集文件路径
        """
        # 情况1：使用官方划分（需目录包含train/val/test子文件夹）
        if self.config.use_official_split:
            train_dir = self.data_dir / "train"
            val_dir = self.data_dir / "val"
            test_dir = self.data_dir / "test"
            
            train_paths = list(train_dir.glob("*")) if train_dir.exists() else []
            val_paths = list(val_dir.glob("*")) if val_dir.exists() else []
            test_paths = list(test_dir.glob("*")) if test_dir.exists() else []
            
            if not train_paths:
                raise ValueError("启用官方划分，但未找到train子目录或子目录下无文件")
            logger.info(f"使用官方目录划分：train/{len(train_paths)} | val/{len(val_paths)} | test/{len(test_paths)}")
            return train_paths, val_paths, test_paths
        
        # 情况2：自定义划分（基于split_ratio/test_ratio）
        total = len(self.full_file_paths)
        if self.config.split_ratio <= 0 or self.config.split_ratio >= 1:
            raise ValueError(f"split_ratio必须在(0,1)范围内，当前值：{self.config.split_ratio}")
        
        np.random.seed(self.config.split_seed)  # 固定种子保证可复现
        shuffled_paths = np.random.permutation(self.full_file_paths).tolist()
        
        # 计算划分索引
        train_size = int(total * self.config.split_ratio)
        train_paths = shuffled_paths[:train_size]
        
        remaining = shuffled_paths[train_size:]
        test_paths = []
        val_paths = []
        
        if self.config.test_ratio:
            if self.config.test_ratio < 0 or self.config.test_ratio >= (1 - self.config.split_ratio):
                raise ValueError(f"test_ratio需在[0, {1-self.config.split_ratio})范围内，当前值：{self.config.test_ratio}")
            test_size = int(total * self.config.test_ratio)
            test_paths = remaining[:test_size]
            val_paths = remaining[test_size:]
        else:
            val_paths = remaining
        
        logger.info(f"自定义数据划分完成：训练集{len(train_paths)} | 验证集{len(val_paths)} | 测试集{len(test_paths)}")
        return train_paths, val_paths, test_paths

    def _get_cached_sample(self, idx: int) -> Optional[Union[np.ndarray, Tuple]]:
        """通用缓存读取逻辑（内存缓存，当前实例独立）"""
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]
        return None

    def _cache_sample(self, idx: int, sample: Union[np.ndarray, Tuple]) -> None:
        """通用缓存写入逻辑（内存缓存，当前实例独立）"""
        if self.cache_in_memory:
            self._cache[idx] = sample

    def _log_init_info(self) -> None:
        """日志输出初始化信息，提升可追溯性"""
        logger.info(f"数据集初始化完成 | 数据目录：{self.data_dir.absolute()}")
        logger.info(f"自动划分开启：{self.auto_split} | 当前实例模式：{self._dataset_mode}")
        logger.info(f"完整样本数：{len(self.full_file_paths)} | 训练集：{len(self.train_paths)} | 验证集：{len(self.val_paths)} | 测试集：{len(self.test_paths)}")
        logger.info(f"内存缓存开启：{self.cache_in_memory} | 最大样本限制：{self.max_samples if self.max_samples else '无'}")

    def _create_dataset_instance(self, mode: str) -> DatasetType:
        """
        ⚠️ 已废弃：使用 get_train_dataset()/get_val_dataset()/get_test_dataset() 代替
        这些新方法使用 Subset 复用原实例，避免重复初始化
        """
        raise NotImplementedError("请使用 get_train_dataset()/get_val_dataset()/get_test_dataset() 代替")

    def _setup_subset_instance(self, instance, mode: str):
        """⚠️ 已废弃：不再需要此方法"""
        pass

    def get_train_dataset(self) -> Union[Subset, "BaseDataset"]:
        """✅ 修复：基于索引划分，而非文件匹配"""
        total = len(self.full_file_paths)
        train_size = len(self.train_paths)
        train_indices = list(range(train_size))  # 直接用前N个索引（和划分顺序一致）
        return Subset(self, train_indices)

    def get_val_dataset(self) -> Union[Subset, "BaseDataset"]:
        """✅ 修复：基于索引划分"""
        total = len(self.full_file_paths)
        train_size = len(self.train_paths)
        val_size = len(self.val_paths)
        val_indices = list(range(train_size, train_size + val_size))
        return Subset(self, val_indices)

    def get_test_dataset(self) -> Union[Subset, "BaseDataset"]:
        """✅ 修复：基于索引划分"""
        total = len(self.full_file_paths)
        train_size = len(self.train_paths)
        val_size = len(self.val_paths)
        test_indices = list(range(train_size + val_size, total))
        return Subset(self, test_indices)

    def get_full_dataset(self) -> "BaseDataset":
        """获取完整数据集实例"""
        return self

    # --------------------------
    # 抽象方法：强制子类实现（核心个性化逻辑）
    # --------------------------
    @abstractmethod
    def _parse_sample(self, file_path: Path) -> Union[np.ndarray, Tuple]:
        """
        解析单个样本文件（子类必须实现）
        Args:
            file_path: 样本文件路径（Path对象）
        Returns:
            解析后的原始数据（如时序数组、图像数组+标签）
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple:
        """
        生成模型输入格式（子类必须实现）
        Args:
            index: 样本索引
        Returns:
            (输入张量, 标签) 或 仅输入张量（无标注场景）
        """
        pass

    # --------------------------
    # 通用接口：保持兼容性+扩展便捷性
    # --------------------------
    def __len__(self) -> int:
        """
        数据集长度：根据当前实例模式返回对应长度
        - full模式：返回完整数据集长度
        - train模式：返回训练集长度
        - val模式：返回验证集长度
        - test模式：返回测试集长度
        """
        mode_len_map = {
            "full": len(self.full_file_paths),
            "train": len(self.train_paths),
            "val": len(self.val_paths),
            "test": len(self.test_paths)
        }
        return mode_len_map.get(self._dataset_mode, len(self.full_file_paths))

    def get_val_len(self) -> int:
        """验证集长度（兼容原有接口）"""
        return len(self.val_paths)

    def get_test_len(self) -> int:
        """测试集长度（兼容原有接口）"""
        return len(self.test_paths)

    def get_full_len(self) -> int:
        """完整数据集长度（新增便捷接口）"""
        return len(self.full_file_paths)

    def clear_cache(self) -> None:
        """清空当前实例的内存缓存（避免内存泄漏）"""
        if self._cache and len(self._cache) > 0:
            self._cache.clear()
            logger.info(f"{self._dataset_mode}模式数据集内存缓存已清空")

    def update_config(self, **kwargs) -> None:
        """
        便捷更新配置参数（支持子类差异化配置，如数据增强开关）
        Args:
            kwargs: 要更新的配置键值对，如transform=train_transform
        """
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
                logger.info(f"更新配置参数：{k} = {v}")
            else:
                logger.warning(f"配置中无参数{k}，跳过更新")