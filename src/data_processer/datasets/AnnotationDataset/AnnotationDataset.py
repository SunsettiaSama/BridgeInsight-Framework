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
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data_processer.datasets.BaseDataset import BaseDataset
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import (
    AnnotationDatasetConfig
)
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor, LRUVICCache

logger = logging.getLogger(__name__)


def _parse_sample_standalone(file_path: Path, config_dict: dict, idx: int, anno_info: Dict) -> Tuple[int, Optional[dict]]:
    """
    独立函数：用于多进程预加载数据
    不需要实例化 AnnotationDataset，直接执行解析逻辑
    
    参数:
        file_path: 文件路径
        config_dict: 配置字典（序列化友好）
        idx: 样本索引
        anno_info: 单条标注信息（预查询，避免跨进程拷贝整个字典）
    
    返回:
        (样本索引, 解析结果字典或None)
    """
    try:
        if anno_info is None:
            raise ValueError(f"未找到样本 {file_path} 的标注信息")
        
        vic_extractor = VICWindowExtractor(
            enable_denoise=config_dict.get('enable_denoise', False),
            enable_extreme_window=config_dict.get('enable_extreme_window', False)
        )
        
        data_format = config_dict.get('data_format', 'vic')
        metadata_data_type = anno_info["metadata"].get("data_type", "").lower()
        if metadata_data_type:
            if metadata_data_type == "vic":
                data_format = "vic"
            elif metadata_data_type in ["npy", "numpy"]:
                data_format = "npy"
        
        if data_format == config_dict.get('data_format', 'vic'):
            file_ext = Path(file_path).suffix.lower()
            if file_ext == ".vic":
                data_format = "vic"
            elif file_ext in [".npy", ".npz"]:
                data_format = "npy"
            elif file_ext == ".csv":
                data_format = "csv"
        
        if data_format == "vic":
            window_index = anno_info["metadata"].get("window_index", 0)
            window_size = config_dict.get('window_size', 3000)
            data = vic_extractor.extract_window(
                str(file_path),
                window_index,
                window_size,
                metadata=anno_info["metadata"]
            )
        elif data_format == "npy":
            data = np.load(str(file_path))
            if 'window_size' in config_dict:
                window_index = anno_info["metadata"].get("window_index", 0)
                window_size = config_dict['window_size']
                start_idx = window_index * window_size
                end_idx = start_idx + window_size
                data = data[start_idx:end_idx]
                
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
        else:
            raise ValueError(f"不支持的数据格式：{data_format}。仅支持 'vic' 和 'npy'")
        
        sample_id = file_path.stem
        label = anno_info["class_id"]
        
        return (idx, {
            "data": data.astype(np.float32),
            "label": label,
            "sample_id": sample_id,
            "metadata": anno_info
        })
    
    except Exception as e:
        logger.warning(f"预加载样本 {idx} 失败：{e}")
        return (idx, None)


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
        ✅ 修复：先调用基类 → 再构建子类数据 → 强制覆盖基类文件列表
        """
        # 0. ✅ 第一步：必须先调用基类初始化（修复顺序核心）
        super().__init__(config)
        
        # 1. 存储配置
        self.annotation_config = config
        self.label_mapping = config.label_to_class if config.enable_label_mapping else None
        
        # 2. 加载并解析标注文件
        self.annotation_data = self._load_annotations(config.annotation_file)
        
        # 3. 过滤和转换标注
        self.sample_annotations = self._process_annotations(config)
        
        # 4. 构建标注过滤后的有效文件列表
        self.file_paths = self._build_file_paths(config)
        
        # 5. ✅ 强制覆盖基类的文件列表（让基类用子类的有效样本划分）
        self.full_file_paths = self.file_paths
        # 清空基类错误划分的结果，重新划分
        self.train_paths = []
        self.val_paths = []
        self.test_paths = []
        
        # 6. ✅ 手动触发正确的数据集划分（用子类有效样本）
        if self.auto_split:
            self.train_paths, self.val_paths, self.test_paths = self._split_dataset()
        
        # 7. 后续原有代码不变
        enable_denoise = getattr(config, 'enable_denoise', False)
        enable_extreme_window = getattr(config, 'enable_extreme_window', False)
        self.vic_extractor = VICWindowExtractor(
            enable_denoise=enable_denoise,
            enable_extreme_window=enable_extreme_window
        )
        
        cache_max_items = getattr(config, 'cache_max_items', 1000)
        self.vic_cache = LRUVICCache(max_items=cache_max_items)
        logger.info(f"初始化缓存: max_items={cache_max_items}")
        
        self.preload_cache = {}
        self.global_norm_stats = self._calc_global_normalize_stats()
        
        if config.task_type == "regression":
            self.window_indices = self._build_regression_window_indices()
            logger.info(f"回归任务：生成 {len(self.window_indices)} 个窗口样本")
        else:
            self.window_indices = None
        
        enable_preload = getattr(config, 'enable_preload_cache', False)
        if enable_preload:
            num_workers = getattr(config, 'preload_num_workers', 4)
            # 📌 支持进度条控制参数（可由调用方通过 config 指定）
            show_progress = getattr(config, 'show_preload_progress', False)
            # 📌 验证config的字典化是否可序列化（预检查）
            self._verify_config_serializability()
            self._preload_all_data(num_workers, show_progress=show_progress)
        
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
    
    def _verify_config_serializability(self):
        """
        ✅ 验证配置字典是否可序列化（用于 ProcessPoolExecutor）
        
        在多进程预加载前检查，确保 config_dict 能被正确 pickle，
        避免隐形的多进程崩溃问题
        """
        import pickle
        
        config = self.annotation_config
        config_dict = {
            'enable_denoise': getattr(config, 'enable_denoise', False),
            'enable_extreme_window': getattr(config, 'enable_extreme_window', False),
            'data_format': config.data_format,
            'window_size': config.window_size,
        }
        
        try:
            pickle.dumps(config_dict)
            logger.info("✅ 配置字典可序列化，多进程预加载正常")
        except Exception as e:
            logger.error(f"❌ 配置字典不可序列化: {e}")
            logger.warning("⚠️ 多进程预加载可能失败，将自动回退到单进程模式")
    
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
    
    def _convert_to_relative_path(self, file_path: str) -> str:
        """
        将绝对路径转换为相对路径（从项目根目录开始）
        
        通过字符串匹配找到项目根目录标识符，然后提取相对路径
        支持不同驱动器号的情况（Windows）
        
        参数:
            file_path: 原始路径（绝对或相对）
        
        返回:
            转换后的相对路径
        """
        if not file_path:
            return file_path
        
        file_path = str(file_path)
        
        try:
            project_marker = self.annotation_config.project_root_marker
            
            if project_marker in file_path:
                idx = file_path.find(project_marker)
                relative_part = file_path[idx + len(project_marker):].lstrip(os.sep).lstrip('/')
                logger.debug(f"路径转换成功：{file_path} -> {relative_part}")
                return relative_part
            
            logger.debug(f"未找到项目标识符 '{project_marker}' 在路径中，保持原路径：{file_path}")
            return file_path
        
        except Exception as e:
            logger.warning(f"路径转换失败 ({file_path})：{e}，保持原路径")
            return file_path
    
    def _resolve_file_path(self, file_path: str, config: AnnotationDatasetConfig) -> str:
        """
        解析文件路径，支持相对路径转换
        
        1. 如果启用路径转换，先将绝对路径转换为相对路径
        2. 如果是相对路径，基于data_dir进行解析
        3. 返回最终的完整文件路径
        
        参数:
            file_path: 原始文件路径
            config: 配置实例
        
        返回:
            解析后的完整文件路径
        """
        if not file_path:
            return file_path
        
        file_path = str(file_path).strip()
        
        if config.enable_path_conversion:
            file_path = self._convert_to_relative_path(file_path)
        
        path_obj = Path(file_path)
        
        if path_obj.is_absolute() and path_obj.exists():
            return str(path_obj)
        
        if not path_obj.is_absolute():
            alt_path = Path(config.data_dir) / file_path
            if alt_path.exists():
                return str(alt_path.resolve())
        
        return file_path
    
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
            
            # 解析路径（支持相对路径转换）
            resolved_path = self._resolve_file_path(data_path, config)
            
            # 如果没有 sample_id，自动生成一个唯一ID
            if not sample_id:
                window_index = item.get("window_index", 0)
                file_name = Path(resolved_path).stem
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
                class_id = annotation
            
            # 5. 存储处理后的标注
            sample_annotations[sample_id] = {
                "annotation": annotation,
                "class_id": class_id,
                "file_path": resolved_path,
                "metadata": item
            }
        
        logger.info(f"标注数据处理完成：{len(sample_annotations)} 个有效样本，过滤 {filtered_count} 个")
        if config.enable_path_conversion:
            logger.info(f"路径转换已启用（项目标识符: {config.project_root_marker}）")
        return sample_annotations
    
    def _find_file_with_search_strategy(self, rel_file_path: str, config: AnnotationDatasetConfig) -> Optional[str]:
        """
        使用多策略查找文件（处理跨驱动器迁移的情况）
        
        搜索策略：
        1. 尝试原始路径（可能是绝对路径）
        2. 尝试相对于 data_dir 的路径
        3. 尝试从项目根目录相对查找
        4. 尝试从项目根目录的 data 文件夹查找
        
        参数:
            rel_file_path: 相对路径或文件名
            config: 配置实例
        
        返回:
            找到的完整文件路径，或 None
        """
        file_name = Path(rel_file_path).name
        
        search_paths = []
        
        search_paths.append(Path(rel_file_path))
        
        search_paths.append(Path(config.data_dir) / rel_file_path)
        
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            search_paths.append(project_root / rel_file_path)
            search_paths.append(project_root / "data" / rel_file_path)
        except:
            pass
        
        for search_path in search_paths:
            if search_path.exists():
                logger.debug(f"文件找到：{search_path}")
                return str(search_path.resolve())
        
        return None
    
    def _build_file_paths(self, config: AnnotationDatasetConfig) -> List[Path]:
        """
        构建有效的文件路径列表（仅包含已标注的文件）
        
        支持跨驱动器迁移：当启用路径转换时，会使用多策略查找文件
        
        参数:
            config: 配置实例
        
        返回:
            文件路径列表
        """
        file_paths = []
        missing_files = 0
        
        for sample_id, anno_info in self.sample_annotations.items():
            file_path = anno_info["file_path"]
            
            found_path = None
            
            if Path(file_path).exists():
                found_path = file_path
            elif config.enable_path_conversion:
                rel_file_path = self._convert_to_relative_path(file_path)
                if rel_file_path != file_path:
                    found_path = self._find_file_with_search_strategy(rel_file_path, config)
                    if found_path:
                        logger.info(f"跨驱动器查找成功：原路径 {file_path[:60]}... -> 找到 {Path(found_path).name}")
            
            if not found_path:
                alt_path = Path(config.data_dir) / Path(file_path).name
                if alt_path.exists():
                    found_path = str(alt_path)
                else:
                    logger.warning(f"样本 {sample_id} 对应的数据文件不存在：{file_path}")
                    missing_files += 1
                    continue
            
            file_paths.append(Path(found_path))
        
        if missing_files > 0:
            logger.warning(f"有 {missing_files} 个样本的数据文件不存在")
        
        return file_paths
    
    @staticmethod
    def _parse_sample_worker(args: Tuple[Path, AnnotationDatasetConfig, int, Dict[str, Dict]]) -> Tuple[int, Optional[dict]]:
        """⚠️ 已废弃：使用模块级函数 _parse_sample_standalone 代替"""
        raise NotImplementedError("请使用模块级函数 _parse_sample_standalone 代替")
    
    def _preload_all_data(self, num_workers: int = 4, show_progress: bool = False):
        """
        预加载所有数据到内存
        
        在初始化时将所有样本数据加载到内存中，使用多进程并行加载以加快速度
        
        📌 【设计原则】进度条显示由调用方控制，库函数提供钩子但不主动决策
        
        参数:
            num_workers: 并行加载的进程数
            show_progress: 是否显示进度条（由调用方/配置决定）
                - False (默认): 完全静默，无任何副作用
                - True: 显示 tqdm 进度条（仅在终端环境中有效）
        """
        logger.info(f"开始预加载数据集（{len(self.file_paths)} 个样本），使用 {num_workers} 个进程...")
        start_time = time.time()
        
        config = self.annotation_config
        
        # 📌 修复1：将 config 转换为字典（可序列化，解决 pickle 问题）
        config_dict = {
            'enable_denoise': getattr(config, 'enable_denoise', False),
            'enable_extreme_window': getattr(config, 'enable_extreme_window', False),
            'data_format': config.data_format,
            'window_size': config.window_size,
        }
        
        # 📌 修复2+3：预先建立路径映射，在主进程中完成路径查询
        # 避免跨进程拷贝整个 sample_annotations 字典
        file_to_anno = {}
        for idx, file_path in enumerate(self.file_paths):
            file_path_resolved = str(Path(file_path).resolve())
            anno_info = None
            
            # 使用唯一键（已resolve的路径字符串）查找标注
            for sid, info in self.sample_annotations.items():
                anno_resolved = str(Path(info["file_path"]).resolve())
                if file_path_resolved == anno_resolved:
                    anno_info = info
                    break
            
            if anno_info is not None:
                file_to_anno[file_path_resolved] = anno_info
        
        # 📌 修复2+3：构建 task_args，每条只传递单个 anno_info，不传整个字典
        task_args = [
            (file_path, config_dict, idx, file_to_anno.get(str(Path(file_path).resolve())))
            for idx, file_path in enumerate(self.file_paths)
            if str(Path(file_path).resolve()) in file_to_anno
        ]
        
        total_tasks = len(task_args)
        loaded_count = 0
        failed_count = 0
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(_parse_sample_standalone, *args): args[2]
                    for args in task_args
                }
                
                # 📌 【关键修复】根据 show_progress 参数决定是否显示进度条
                if show_progress:
                    # 仅在需要时导入 tqdm，避免 import 时触发终端检测
                    iterator = tqdm(
                        as_completed(future_to_idx),
                        total=total_tasks,
                        desc="📦 预加载数据集",
                        unit="样本",
                        ncols=100,
                        mininterval=0.5,
                        leave=True
                    )
                else:
                    # 静默模式：不显示进度条，无任何副作用
                    iterator = as_completed(future_to_idx)
                
                for future in iterator:
                    idx = future_to_idx[future]
                    try:
                        result_idx, result = future.result(timeout=60)
                        if result is not None:
                            cache_key = f"preload_{result_idx}"
                            self.preload_cache[cache_key] = result
                            loaded_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"样本 {idx} 预加载失败: {e}")
        
        except Exception as e:
            logger.error(f"多进程预加载失败，回退到单进程加载：{e}")
            # 回退到单进程
            if show_progress:
                file_paths_iter = tqdm(self.file_paths, desc="⚠️ 单进程回退加载")
            else:
                file_paths_iter = self.file_paths
                
            for file_idx, file_path in enumerate(file_paths_iter):
                try:
                    result = self._parse_sample(file_path)
                    cache_key = f"preload_{file_idx}"
                    self.preload_cache[cache_key] = result
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"预加载样本 {file_idx} 失败：{e}")
                    failed_count += 1
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n✅ 预加载完成！")
        logger.info(f"   成功: {loaded_count}/{total_tasks} 样本")
        logger.info(f"   失败: {failed_count} 样本")
        logger.info(f"   耗时: {elapsed_time:.2f}s，平均 {elapsed_time/max(1, total_tasks):.3f}s/样本")
        logger.info(f"   缓存占用: 约 {len(self.preload_cache) * 300 / 1024:.2f} MB")
    
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
        # 检查预加载缓存
        cache_key = f"preload_{idx}"
        if cache_key in self.preload_cache:
            sample = self.preload_cache[cache_key]
            logger.debug(f"从预加载缓存读取样本 {idx}")
        else:
            file_path = self.full_file_paths[idx]
            sample = self._parse_sample(file_path)
        
        data = sample["data"]
        label = int(sample["label"])
        
        if self.annotation_config.fix_seq_len is not None:
            data = self._process_sequence_length(data)
        
        if self.annotation_config.normalize and self.global_norm_stats:
            data = self._normalize_sequence(data)
        
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
        
        # 检查预加载缓存
        cache_key = f"preload_{file_idx}"
        if cache_key in self.preload_cache:
            sample = self.preload_cache[cache_key]
            logger.debug(f"从预加载缓存读取回归样本 {file_idx}")
        else:
            file_path = self.full_file_paths[file_idx]
            sample = self._parse_sample(file_path)
        
        data = sample["data"]
        
        window_data = data[start_idx:end_idx]
        
        input_data = window_data[:config.look_back]
        output_data = window_data[config.look_back:]
        
        if config.normalize and self.global_norm_stats:
            input_data = self._normalize_sequence(input_data)
            output_data = self._normalize_sequence(output_data)
        
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
    
    def get_preload_cache_stats(self) -> dict:
        """
        获取预加载缓存统计信息
        
        返回:
            包含缓存大小、占用内存等信息的字典
        """
        cache_size = len(self.preload_cache)
        memory_mb = cache_size * 300 / 1024
        
        stats = {
            "enabled": len(self.preload_cache) > 0,
            "loaded_samples": cache_size,
            "total_samples": len(self.file_paths),
            "memory_usage_mb": memory_mb,
            "cache_hit_rate": f"{100 * cache_size / len(self.file_paths):.1f}%" if len(self.file_paths) > 0 else "0%"
        }
        
        return stats
    
    def clear_preload_cache(self):
        """清空预加载缓存"""
        cache_size = len(self.preload_cache)
        self.preload_cache.clear()
        logger.info(f"预加载缓存已清空（释放 {cache_size} 个样本，约 {cache_size * 300 / 1024:.2f} MB）")
    
    def get_train_dataset(self):
        """获取训练集（使用基类的Subset实现，无需重新初始化）"""
        return super().get_train_dataset()
    
    def get_val_dataset(self):
        """获取验证集（使用基类的Subset实现，无需重新初始化）"""
        return super().get_val_dataset()
    
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
            return len(self.full_file_paths)
    
    def get_num_classes(self) -> int:
        """
        获取分类任务的类别总数
        
        返回：
            num_classes: 类别总数
        """
        return self.annotation_config.num_classes
    
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
