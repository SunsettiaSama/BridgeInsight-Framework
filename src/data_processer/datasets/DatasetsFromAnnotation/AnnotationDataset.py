import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_processer.datasets.DatasetsFromAnnotation.BaseDataset import BaseDataset
from config.data_processer.datasets.DatasetsFromAnnotation.AnnotationDatasetConfig import (
    AnnotationDatasetConfig
)

logger = logging.getLogger(__name__)


class AnnotationDataset(BaseDataset):
    """
    基于标注文件的时序分类数据集
    
    核心功能：
    1. 加载标注JSON文件
    2. 匹配数据文件和标注信息
    3. 支持标签过滤和转换
    4. 自动处理时序数据的加载和预处理
    """
    
    def __init__(self, config: AnnotationDatasetConfig):
        """
        初始化标注数据集
        
        参数:
            config: AnnotationDatasetConfig配置实例
        """
        # 1. 加载并解析标注文件
        self.annotation_data = self._load_annotations(config.annotation_file)
        
        # 2. 过滤和转换标注
        self.sample_annotations = self._process_annotations(config)
        
        # 3. 构建文件路径列表
        self.file_paths = self._build_file_paths(config)
        
        # 4. 存储配置特有参数（在调用基类之前）
        self.annotation_config = config
        self.label_mapping = config.label_to_class if config.enable_label_mapping else None
        
        # 5. 调用基类初始化（处理路径、划分、缓存等）
        super().__init__(config)
        
        # 6. 计算全局归一化统计
        self.global_norm_stats = self._calc_global_normalize_stats()
        
        logger.info(f"标注数据集加载完成：{len(self.file_paths)} 个样本")
    
    def _get_file_list(self) -> List[Path]:
        """
        重写基类方法：从标注文件中获取文件列表
        
        返回:
            文件路径列表
        """
        # 标注数据集的文件列表已在 _build_file_paths 中构建
        return self.file_paths if hasattr(self, 'file_paths') else []
    
    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """
        加载标注JSON文件
        
        参数:
            annotation_file: 标注JSON文件路径
        
        返回:
            标注数据列表
        """
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            if not isinstance(annotations, list):
                raise ValueError("标注文件必须是列表格式")
            
            logger.info(f"加载标注文件：{annotation_file}，共 {len(annotations)} 条记录")
            return annotations
        
        except Exception as e:
            logger.error(f"加载标注文件失败：{e}")
            raise
    
    def _process_annotations(self, config: AnnotationDatasetConfig) -> Dict[str, Dict]:
        """
        处理标注数据：过滤、验证、转换
        
        参数:
            config: 配置实例
        
        返回:
            处理后的标注字典 {sample_id: {标注信息}}
        """
        sample_annotations = {}
        filtered_count = 0
        
        for item in self.annotation_data:
            # 1. 提取关键信息
            sample_id = item.get(config.sample_id_field)
            annotation = item.get(config.annotation_field)
            data_path = item.get(config.data_path_field)
            
            if not sample_id or not data_path:
                logger.warning(f"跳过不完整的标注记录：{item}")
                continue
            
            # 2. 检查是否为空标注
            if config.only_annotated and not annotation:
                filtered_count += 1
                continue
            
            # 3. 检查标签过滤（include/exclude）
            if config.include_labels and annotation not in config.include_labels:
                filtered_count += 1
                continue
            
            if config.exclude_labels and annotation in config.exclude_labels:
                filtered_count += 1
                continue
            
            # 4. 标签转换
            if config.enable_label_mapping and config.label_to_class:
                if annotation in config.label_to_class:
                    class_id = config.label_to_class[annotation]
                else:
                    if config.unknown_label_class == -1:
                        logger.warning(f"未知标签：{annotation}，跳过该样本")
                        filtered_count += 1
                        continue
                    class_id = config.unknown_label_class
            else:
                # 如果不启用映射，保持原始标注（可能是字符串或数值）
                class_id = annotation
            
            # 5. 存储处理后的标注
            sample_annotations[sample_id] = {
                "annotation": annotation,
                "class_id": class_id,
                "file_path": data_path,
                "metadata": item  # 保存原始元数据
            }
        
        logger.info(f"标注数据处理完成：{len(sample_annotations)} 个有效样本，过滤 {filtered_count} 个")
        return sample_annotations
    
    def _build_file_paths(self, config: AnnotationDatasetConfig) -> List[Path]:
        """
        构建有效的文件路径列表（仅包含已标注的文件）
        
        参数:
            config: 配置实例
        
        返回:
            文件路径列表
        """
        file_paths = []
        missing_files = 0
        
        for sample_id, anno_info in self.sample_annotations.items():
            file_path = anno_info["file_path"]
            
            # 尝试绝对和相对路径
            if not Path(file_path).exists():
                # 尝试相对于data_dir的路径
                alt_path = Path(config.data_dir) / file_path
                if alt_path.exists():
                    file_path = str(alt_path)
                else:
                    logger.warning(f"样本 {sample_id} 对应的数据文件不存在：{file_path}")
                    missing_files += 1
                    continue
            
            file_paths.append(Path(file_path))
        
        if missing_files > 0:
            logger.warning(f"有 {missing_files} 个样本的数据文件不存在")
        
        return file_paths
    
    def _parse_sample(self, file_path: Path) -> dict:
        """
        解析单个样本文件
        
        参数:
            file_path: 样本文件路径
        
        返回:
            {data, label, sample_id}
        """
        config = self.annotation_config
        
        try:
            if config.data_format == "mat":
                loaded = sio.loadmat(str(file_path))
                data = loaded.get(config.mat_data_key)
                
                if data is None:
                    raise ValueError(f"未找到数据键 '{config.mat_data_key}' 在文件 {file_path}")
                
                # 展平到2D (seq_len, feat_dim)
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
                
            elif config.data_format == "vic":
                # .VIC格式需要特殊处理（依赖io_unpacker）
                from src.data_processer.io_unpacker import DataManager
                manager = DataManager()
                data = manager.load_vic_file(str(file_path))
            
            elif config.data_format == "npy":
                data = np.load(str(file_path))
            
            else:
                raise ValueError(f"不支持的数据格式：{config.data_format}")
            
            # 匹配样本到标注
            sample_id = file_path.stem  # 使用文件名作为sample_id
            
            # 查找对应的标注
            anno_info = None
            for sid, info in self.sample_annotations.items():
                if file_path.samefile(Path(info["file_path"])):
                    anno_info = info
                    break
            
            if anno_info is None:
                raise ValueError(f"未找到样本 {sample_id} 的标注信息")
            
            label = anno_info["class_id"]
            
            return {
                "data": data.astype(np.float32),
                "label": label,
                "sample_id": sample_id,
                "metadata": anno_info
            }
        
        except Exception as e:
            logger.error(f"解析样本文件失败 {file_path}：{e}")
            raise
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            (data_tensor, label_tensor)
        """
        file_path = self.file_paths[idx]
        sample = self._parse_sample(file_path)
        
        data = sample["data"]  # shape: (seq_len, feat_dim)
        label = sample["label"]
        
        # 处理序列长度
        if self.annotation_config.fix_seq_len is not None:
            data = self._process_sequence_length(data)
        
        # 归一化
        if self.annotation_config.normalize and self.global_norm_stats:
            data = self._normalize_sequence(data)
        
        # 转为Tensor
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return data_tensor, label_tensor
    
    def _process_sequence_length(self, data: np.ndarray) -> np.ndarray:
        """
        处理序列长度（补全/截断）
        
        参数:
            data: 原始数据
        
        返回:
            处理后的数据
        """
        config = self.annotation_config
        target_len = config.fix_seq_len
        current_len = data.shape[0]
        
        if current_len == target_len:
            return data
        
        elif current_len < target_len:
            # 补全
            pad_len = target_len - current_len
            if config.pad_mode == "zero":
                pad = np.zeros((pad_len, data.shape[1]), dtype=data.dtype)
            elif config.pad_mode == "repeat":
                pad = np.tile(data[-1:], (pad_len, 1))
            elif config.pad_mode == "mean":
                pad = np.tile(data.mean(axis=0), (pad_len, 1))
            else:
                raise ValueError(f"未知的补全模式：{config.pad_mode}")
            
            return np.vstack([data, pad])
        
        else:
            # 截断
            if config.trunc_mode == "head":
                return data[:target_len]
            elif config.trunc_mode == "tail":
                return data[-target_len:]
            else:
                raise ValueError(f"未知的截断模式：{config.trunc_mode}")
    
    def _normalize_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        使用全局统计量归一化序列
        
        参数:
            data: 原始数据
        
        返回:
            归一化后的数据
        """
        config = self.annotation_config
        stats = self.global_norm_stats
        
        if config.normalize_type == "z-score":
            mean = stats.get("mean", 0)
            std = stats.get("std", 1)
            return (data - mean) / (std + 1e-8)
        
        elif config.normalize_type == "min-max":
            vmin = stats.get("min", data.min())
            vmax = stats.get("max", data.max())
            return (data - vmin) / (vmax - vmin + 1e-8)
        
        else:
            return data
    
    def _calc_global_normalize_stats(self) -> dict:
        """
        计算全局归一化统计量
        
        返回:
            统计量字典 {mean, std, min, max}
        """
        config = self.annotation_config
        
        if not config.normalize or len(self.file_paths) == 0:
            return {}
        
        logger.info("计算全局归一化统计量...")
        
        all_data = []
        for file_path in self.file_paths[:min(100, len(self.file_paths))]:  # 采样前100个文件
            try:
                sample = self._parse_sample(file_path)
                all_data.append(sample["data"].flatten())
            except:
                continue
        
        if not all_data:
            logger.warning("无法计算全局统计量，使用默认值")
            return {"mean": 0, "std": 1, "min": 0, "max": 1}
        
        all_data = np.concatenate(all_data)
        
        stats = {
            "mean": float(all_data.mean()),
            "std": float(all_data.std()),
            "min": float(all_data.min()),
            "max": float(all_data.max())
        }
        
        logger.info(f"全局统计：mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        return stats
    
    def get_train_dataset(self) -> "AnnotationDataset":
        """获取训练集实例（继承自BaseDataset）"""
        return self._create_dataset_instance("train")
    
    def get_val_dataset(self) -> "AnnotationDataset":
        """获取验证集实例（继承自BaseDataset）"""
        return self._create_dataset_instance("val")
    
    def __len__(self) -> int:
        """获取数据集大小"""
        return len(self.file_paths)
