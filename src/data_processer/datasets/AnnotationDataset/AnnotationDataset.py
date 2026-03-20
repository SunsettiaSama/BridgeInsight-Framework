import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data_processer.datasets.BaseDataset import BaseDataset
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import (
    AnnotationDatasetConfig
)
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor, LRUVICCache

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
        
        # 5. 初始化VIC窗口提取器（始终初始化，因为会在运行时自动检测格式）
        enable_denoise = getattr(config, 'enable_denoise', False)
        enable_extreme_window = getattr(config, 'enable_extreme_window', False)
        self.vic_extractor = VICWindowExtractor(
            enable_denoise=enable_denoise,
            enable_extreme_window=enable_extreme_window
        )
        
        # 6. 初始化LRU缓存（始终初始化，因为会在运行时自动检测格式）
        cache_max_items = getattr(config, 'cache_max_items', 1000)
        self.vic_cache = LRUVICCache(max_items=cache_max_items)
        logger.info(f"初始化缓存: max_items={cache_max_items}")
        
        # 7. 调用基类初始化（处理路径、划分、缓存等）
        super().__init__(config)
        
        # 8. 计算全局归一化统计
        self.global_norm_stats = self._calc_global_normalize_stats()
        
        # 9. 回归任务：构建滑窗索引映射
        if config.task_type == "regression":
            self.window_indices = self._build_regression_window_indices()
            logger.info(f"回归任务：生成 {len(self.window_indices)} 个窗口样本")
        else:
            self.window_indices = None
        
        # 10. 初始化可视化器（延迟初始化，在第一次调用show()时创建）
        self._visualizer = None
        
        logger.info(f"标注数据集加载完成：{len(self.file_paths)} 个样本，任务类型={config.task_type}")
    
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
        sample_counter = 0
        
        for item in self.annotation_data:
            # 1. 提取关键信息
            sample_id = item.get(config.sample_id_field)
            annotation = item.get(config.annotation_field)
            data_path = item.get(config.data_path_field)
            
            # 检查必需字段
            if not data_path:
                logger.warning(f"跳过：缺少 data_path_field('{config.data_path_field}') 字段")
                filtered_count += 1
                continue
            
            # 如果没有 sample_id，自动生成一个唯一ID
            if not sample_id:
                # 使用文件路径和window_index生成唯一ID
                window_index = item.get("window_index", 0)
                file_name = Path(data_path).stem
                sample_id = f"{file_name}_w{window_index}_{sample_counter}"
                sample_counter += 1
            
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
        解析单个样本文件（基于标注中的window_index提取VIC数据窗口）
        
        使用缓存机制避免频繁IO读取：
        1. 查询LRU缓存
        2. 缓存未命中时通过VICWindowExtractor读取
        3. 缓存结果用于后续访问
        
        参数:
            file_path: 样本文件路径
        
        返回:
            {data, label, sample_id, metadata}
            其中 data shape: (window_size, 1) 表示单个窗口的振动序列
        """
        config = self.annotation_config
        
        try:
            # 1. 查找对应的标注信息（包含window_index）
            anno_info = None
            for sid, info in self.sample_annotations.items():
                if file_path.samefile(Path(info["file_path"])):
                    anno_info = info
                    break
            
            if anno_info is None:
                raise ValueError(f"未找到样本 {file_path} 的标注信息")
            
            # 2. 自动检测数据格式（优先使用元数据，其次使用文件扩展名，最后使用配置）
            data_format = config.data_format
            
            # 尝试从元数据中获取 data_type
            metadata_data_type = anno_info["metadata"].get("data_type", "").lower()
            if metadata_data_type:
                if metadata_data_type == "vic":
                    data_format = "vic"
                elif metadata_data_type in ["npy", "numpy"]:
                    data_format = "npy"
            
            # 如果没有获取到，尝试从文件扩展名判断
            if data_format == config.data_format:
                file_ext = Path(file_path).suffix.lower()
                if file_ext == ".vic":
                    data_format = "vic"
                elif file_ext in [".npy", ".npz"]:
                    data_format = "npy"
                elif file_ext == ".csv":
                    data_format = "csv"
            
            # 3. 根据检测到的数据格式加载数据
            if data_format == "vic":
                # 使用VICWindowExtractor提取窗口
                window_index = anno_info["metadata"].get("window_index", 0)
                window_size = config.window_size
                
                # 构建缓存键
                cache_key = (str(file_path), window_index, window_size)
                
                # 先检查缓存
                if self.vic_cache is not None:
                    cached_data = self.vic_cache.get(cache_key)
                    if cached_data is not None:
                        logger.debug(f"缓存命中: {Path(file_path).name} @ window {window_index}")
                        data = cached_data
                    else:
                        # 缓存未命中，通过提取器读取（传递完整metadata）
                        data = self.vic_extractor.extract_window(
                            str(file_path),
                            window_index,
                            window_size,
                            metadata=anno_info["metadata"]
                        )
                        # 将结果存入缓存
                        self.vic_cache.put(cache_key, data)
                else:
                    # 没有缓存，直接使用提取器（传递完整metadata）
                    data = self.vic_extractor.extract_window(
                        str(file_path),
                        window_index,
                        window_size,
                        metadata=anno_info["metadata"]
                    )
            
            elif data_format == "npy":
                # NPY格式也支持窗口提取
                data = np.load(str(file_path))
                if hasattr(config, 'window_size'):
                    window_index = anno_info["metadata"].get("window_index", 0)
                    window_size = config.window_size
                    start_idx = window_index * window_size
                    end_idx = start_idx + window_size
                    data = data[start_idx:end_idx]
                    
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
            
            else:
                raise ValueError(f"不支持的数据格式：{data_format}。仅支持 'vic' 和 'npy'")
            
            # 4. 获取标签和样本ID
            sample_id = file_path.stem
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
            分类任务：(data_tensor, label_tensor)
            回归任务：(input_tensor, output_tensor)
        """
        config = self.annotation_config
        
        if config.task_type == "classification":
            return self._getitem_classification(idx)
        elif config.task_type == "regression":
            return self._getitem_regression(idx)
        else:
            raise ValueError(f"未知的任务类型：{config.task_type}")
    
    def _getitem_classification(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分类任务：获取单个样本及其标签
        
        返回：(data_tensor, label_tensor)
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
    
    def _getitem_regression(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        回归任务：获取输入和预测窗口
        
        返回：(input_tensor, output_tensor)
        其中：
        - input_tensor: shape (look_back, feat_dim)
        - output_tensor: shape (forecast_steps, feat_dim)
        """
        config = self.annotation_config
        file_idx, start_idx, end_idx = self.window_indices[idx]
        
        file_path = self.file_paths[file_idx]
        sample = self._parse_sample(file_path)
        data = sample["data"]  # shape: (seq_len, feat_dim)
        
        # 提取窗口
        window_data = data[start_idx:end_idx]  # shape: (look_back + forecast_steps, feat_dim)
        
        # 分割为输入和输出
        input_data = window_data[:config.look_back]    # shape: (look_back, feat_dim)
        output_data = window_data[config.look_back:]   # shape: (forecast_steps, feat_dim)
        
        # 归一化
        if config.normalize and self.global_norm_stats:
            input_data = self._normalize_sequence(input_data)
            output_data = self._normalize_sequence(output_data)
        
        # 转为Tensor
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()
        
        return input_tensor, output_tensor
    
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
    
    def _build_regression_window_indices(self) -> List[Tuple[int, int, int]]:
        """
        为回归任务构建滑窗索引映射
        
        返回:
            窗口索引列表 [(file_idx, start_idx, end_idx), ...]
            其中：
            - file_idx: 文件索引
            - start_idx: 窗口起始点在该文件中的时间步索引
            - end_idx: 窗口终点（包含forecast_steps）
        """
        config = self.annotation_config
        look_back = config.look_back
        forecast_steps = config.forecast_steps
        stride = config.regression_stride
        
        window_indices = []
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                # 加载数据获取长度
                sample = self._parse_sample(file_path)
                data = sample["data"]
                seq_len = data.shape[0]
                
                # 生成该文件中的所有窗口
                # 窗口需要：前look_back步 + 预测forecast_steps步
                max_start = seq_len - look_back - forecast_steps + 1
                
                if max_start > 0:
                    for start_idx in range(0, max_start, stride):
                        end_idx = start_idx + look_back + forecast_steps
                        window_indices.append((file_idx, start_idx, end_idx))
            
            except Exception as e:
                logger.warning(f"构建文件 {file_path} 的回归窗口时出错：{e}")
                continue
        
        if len(window_indices) == 0:
            logger.warning("未能生成任何回归窗口！检查look_back和forecast_steps配置")
        
        return window_indices
    
    def get_train_dataset(self) -> "AnnotationDataset":
        """获取训练集实例（继承自BaseDataset）"""
        return self._create_dataset_instance("train")
    
    def get_val_dataset(self) -> "AnnotationDataset":
        """获取验证集实例（继承自BaseDataset）"""
        return self._create_dataset_instance("val")
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        返回:
            缓存统计字典（仅当使用VIC格式时有效）
        """
        if self.vic_cache is not None:
            return self.vic_cache.get_stats()
        else:
            return {"status": "cache disabled"}
    
    def log_cache_stats(self):
        """打印缓存统计信息"""
        if self.vic_cache is not None:
            self.vic_cache.log_stats()
        else:
            logger.info("缓存未启用（非VIC格式或不支持缓存）")
    
    def clear_cache(self):
        """清空所有缓存"""
        if self.vic_cache is not None:
            self.vic_cache.clear()
            logger.info("VIC窗口缓存已清空")
    
    def __len__(self) -> int:
        """获取数据集大小"""
        if self.annotation_config.task_type == "regression":
            return len(self.window_indices) if self.window_indices else 0
        else:
            return len(self.file_paths)
    
    # --------------------------
    # 可视化便捷方法
    # --------------------------
    def _get_visualizer(self) -> "AnnotationDatasetVisualizer":
        """
        获取或初始化可视化器（延迟初始化）
        
        返回：
            AnnotationDatasetVisualizer 实例
        """
        if self._visualizer is None:
            logger.info("初始化数据集可视化器...")
            self._visualizer = AnnotationDatasetVisualizer(self)
        return self._visualizer
    
    def visualize_sample(self, idx: int) -> plt.Figure:
        """
        可视化单个样本
        
        参数：
            idx: 样本索引
        
        返回：
            matplotlib Figure 对象
        """
        visualizer = self._get_visualizer()
        return visualizer.visualize_sample(idx)
    
    def visualize_batch(self, batch_indices: List[int], figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        批量可视化样本
        
        参数：
            batch_indices: 要可视化的样本索引列表
            figsize: 图像大小
        
        返回：
            matplotlib Figure 对象
        """
        visualizer = self._get_visualizer()
        return visualizer.visualize_batch(batch_indices, figsize)
    
    def show(self):
        """
        显示数据集的所有可视化图表
        
        在Tkinter窗口中显示所有已生成的图表，支持前后翻页和保存。
        如果还未生成任何图表，则显示警告信息。
        """
        visualizer = self._get_visualizer()
        
        # 检查是否有图表
        if not visualizer.figs:
            logger.warning("没有生成任何图表。")
            logger.info("提示：请先调用 visualize_sample() 或 visualize_batch() 生成图表")
            logger.info("示例：")
            logger.info("  dataset.visualize_sample(0)      # 可视化单个样本")
            logger.info("  dataset.visualize_batch([0,1,2]) # 批量可视化样本")
            logger.info("  dataset.show()                   # 显示所有图表")
            return
        
        logger.info(f"显示数据集的{len(visualizer.figs)}个图表...")
        visualizer.show()


# --------------------------
# 可视化模块：基于PlotLib的AnnotationDataset可视化器
# --------------------------
from src.visualize_tools.utils import PlotLib


class AnnotationDatasetVisualizer(PlotLib):
    """
    标注数据集可视化器（继承自PlotLib）
    - 支持分类和回归任务
    - 显示输入、输出和标签信息
    - 集成时域/频域分析
    - 内嵌show()方法用于可视化显示
    """
    
    def __init__(self, dataset: AnnotationDataset):
        """
        初始化可视化器
        
        参数：
            dataset: AnnotationDataset实例
        """
        super().__init__()  # 初始化PlotLib
        
        self.dataset = dataset
        self.config = dataset.annotation_config
        
        # 绘图配置
        self.fs = 50.0  # 采样频率
        self.nfft = 2048
        self.font_size = 11
        self.label_font_size = 12
        
        # 中文字体配置
        self.cn_font = plt.matplotlib.font_manager.FontProperties(
            family='SimHei', size=self.label_font_size
        )
        self.eng_font = plt.matplotlib.font_manager.FontProperties(
            family='DejaVu Sans', size=self.label_font_size
        )
        
        logger.info(f"初始化可视化器：任务类型={self.config.task_type}")
    
    def visualize_sample(self, idx: int) -> plt.Figure:
        """
        可视化单个样本并添加到图表列表
        
        参数：
            idx: 样本索引
        
        返回：
            matplotlib Figure 对象
        """
        if self.config.task_type == "classification":
            fig = self._visualize_classification_sample(idx)
        elif self.config.task_type == "regression":
            fig = self._visualize_regression_sample(idx)
        else:
            raise ValueError(f"未知的任务类型：{self.config.task_type}")
        
        # 自动添加到图表列表（PlotLib管理）
        self.figs.append(fig)
        return fig
    
    def _visualize_classification_sample(self, idx: int) -> plt.Figure:
        """
        可视化分类任务样本
        显示：原始数据 | 时域 | 频域 + 标签信息
        """
        data_tensor, label_tensor = self.dataset[idx]
        data = data_tensor.numpy()  # shape: (seq_len, feat_dim)
        label = label_tensor.item()
        
        # 如果多维，取第一个特征
        if data.ndim > 1:
            data_to_plot = data[:, 0]
        else:
            data_to_plot = data
        
        # 创建图像
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1：时域波形
        ax1 = fig.add_subplot(131)
        time_axis = np.arange(len(data_to_plot)) / self.fs
        ax1.plot(time_axis, data_to_plot, color='#333333', linewidth=1.0)
        ax1.set_title('时域波形', fontproperties=self.cn_font, fontsize=self.label_font_size)
        ax1.set_xlabel('时间 (s)', fontproperties=self.cn_font)
        ax1.set_ylabel('幅度', fontproperties=self.cn_font)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：频域谱
        ax2 = fig.add_subplot(132)
        f, psd = signal.welch(data_to_plot, fs=self.fs, nperseg=int(self.nfft/2), 
                              noverlap=int(self.nfft/4), nfft=self.nfft)
        freq_limit = 25
        mask = f <= freq_limit
        ax2.plot(f[mask], psd[mask], color='#333333', linewidth=1.0)
        ax2.set_title('频域谱', fontproperties=self.cn_font, fontsize=self.label_font_size)
        ax2.set_xlabel('频率 (Hz)', fontproperties=self.cn_font)
        ax2.set_ylabel('PSD', fontproperties=self.cn_font)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, freq_limit)
        
        # 子图3：标签信息
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        # 构建标签文本
        info_text = f"""
【分类任务信息】

样本索引: {idx}
序列长度: {len(data_to_plot)}
特征维度: {data.shape[1] if data.ndim > 1 else 1}

【标签映射】
原始标注: {self._get_original_label(idx)}
    ↓
类别ID: {label}

【数据统计】
最小值: {data_to_plot.min():.4f}
最大值: {data_to_plot.max():.4f}
平均值: {data_to_plot.mean():.4f}
标准差: {data_to_plot.std():.4f}
        """
        
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes,
                fontproperties=self.cn_font, fontsize=self.font_size,
                verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        return fig
    
    def _visualize_regression_sample(self, idx: int) -> plt.Figure:
        """
        可视化回归任务样本
        显示：输入窗口 | 输出窗口 | 映射关系
        """
        input_tensor, output_tensor = self.dataset[idx]
        input_data = input_tensor.numpy()    # shape: (look_back, feat_dim)
        output_data = output_tensor.numpy()  # shape: (forecast_steps, feat_dim)
        
        # 如果多维，取第一个特征
        if input_data.ndim > 1:
            input_to_plot = input_data[:, 0]
        else:
            input_to_plot = input_data
            
        if output_data.ndim > 1:
            output_to_plot = output_data[:, 0]
        else:
            output_to_plot = output_data
        
        # 创建图像
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1：输入窗口（历史数据）
        ax1 = fig.add_subplot(131)
        time_axis_in = np.arange(len(input_to_plot)) / self.fs
        ax1.plot(time_axis_in, input_to_plot, color='#2E86AB', linewidth=1.5, label='输入窗口')
        ax1.set_title(f'输入: 前{self.config.look_back}步', fontproperties=self.cn_font, 
                     fontsize=self.label_font_size)
        ax1.set_xlabel('时间 (s)', fontproperties=self.cn_font)
        ax1.set_ylabel('幅度', fontproperties=self.cn_font)
        ax1.grid(True, alpha=0.3)
        ax1.legend(prop=self.cn_font)
        
        # 子图2：输出窗口（预测数据）
        ax2 = fig.add_subplot(132)
        time_axis_out = np.arange(len(output_to_plot)) / self.fs
        ax2.plot(time_axis_out, output_to_plot, color='#A23B72', linewidth=1.5, label='输出窗口')
        ax2.set_title(f'输出: 预测{self.config.forecast_steps}步', fontproperties=self.cn_font, 
                     fontsize=self.label_font_size)
        ax2.set_xlabel('时间 (s)', fontproperties=self.cn_font)
        ax2.set_ylabel('幅度', fontproperties=self.cn_font)
        ax2.grid(True, alpha=0.3)
        ax2.legend(prop=self.cn_font)
        
        # 子图3：输入→输出映射关系
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        # 构建映射信息
        info_text = f"""
【回归任务信息】

样本索引: {idx}

【窗口配置】
输入步数 (look_back): {self.config.look_back}
输出步数 (forecast_steps): {self.config.forecast_steps}
滑窗步长 (stride): {self.config.regression_stride}

【映射关系】
历史 {self.config.look_back} 步数据
    ↓↓↓ (神经网络)
预测 {self.config.forecast_steps} 步数据

【数据统计】
输入:
  最小值: {input_to_plot.min():.4f}
  最大值: {input_to_plot.max():.4f}
  平均值: {input_to_plot.mean():.4f}

输出:
  最小值: {output_to_plot.min():.4f}
  最大值: {output_to_plot.max():.4f}
  平均值: {output_to_plot.mean():.4f}
        """
        
        ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes,
                fontproperties=self.cn_font, fontsize=self.font_size,
                verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        return fig
    
    def _get_original_label(self, idx: int) -> str:
        """
        获取原始标注（分类任务）
        """
        try:
            file_path = self.dataset.file_paths[idx]
            for sample_id, anno_info in self.dataset.sample_annotations.items():
                if file_path.samefile(Path(anno_info["file_path"])):
                    return anno_info.get("annotation", "未知")
        except:
            pass
        return "未知"
    
    def visualize_batch(self, batch_indices: List[int], figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        批量可视化多个样本并添加到图表列表
        
        参数：
            batch_indices: 要可视化的样本索引列表
            figsize: 图像大小
        
        返回：
            matplotlib Figure 对象
        """
        n_samples = len(batch_indices)
        n_cols = min(3, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=figsize)
        
        for i, idx in enumerate(batch_indices):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            if self.config.task_type == "classification":
                data_tensor, label_tensor = self.dataset[idx]
                data = data_tensor.numpy()
                if data.ndim > 1:
                    data_to_plot = data[:, 0]
                else:
                    data_to_plot = data
                
                label = label_tensor.item()
                ax.plot(data_to_plot, linewidth=1.0)
                ax.set_title(f'样本{idx} - 类别{label}', fontproperties=self.cn_font)
                
            elif self.config.task_type == "regression":
                input_tensor, output_tensor = self.dataset[idx]
                input_data = input_tensor.numpy()
                if input_data.ndim > 1:
                    input_to_plot = input_data[:, 0]
                else:
                    input_to_plot = input_data
                
                ax.plot(input_to_plot, color='#2E86AB', linewidth=1.0, label='输入')
                ax.set_title(f'样本{idx} - 窗口{idx}', fontproperties=self.cn_font)
                ax.legend(prop=self.cn_font)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('时间步', fontproperties=self.cn_font)
            ax.set_ylabel('幅度', fontproperties=self.cn_font)
        
        plt.tight_layout()
        
        # 自动添加到图表列表（PlotLib管理）
        self.figs.append(fig)
        return fig
    
    def show(self):
        """
        显示所有可视化的图表（使用PlotLib的show方法）
        
        在Tkinter窗口中显示所有生成的图表，支持前后翻页和保存
        """
        if not self.figs:
            logger.warning("没有图表可显示，请先调用visualize_sample()或visualize_batch()")
            return
        
        logger.info(f"开始显示{len(self.figs)}个图表")
        super().show()  # 调用PlotLib的show()方法
