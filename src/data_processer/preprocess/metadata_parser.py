"""
元数据解析处理器 - Metadata Parser

统一管理振动和风数据元数据，支持队列式批量解析
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple, Literal, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.preprocess.vib_metadata2data import (
    parse_single_metadata_to_vibration_data,
)
from src.data_processer.preprocess.wind_metadata2data import (
    parse_single_metadata_to_wind_data,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 超参数配置 ====================
class ParseConfig:
    """数据解析超参数配置"""
    
    # 批处理配置
    DEFAULT_BATCH_SIZE = 32
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 1024
    
    # 进程配置
    DEFAULT_NUM_WORKERS = None  # None=单进程（推荐PyTorch）
    
    # 数据窗口配置
    DEFAULT_WINDOW_INDEX = 0  # 默认窗口索引
    
    # 日志配置
    ENABLE_LOGGING_DEFAULT = True
    LOG_TO_CONSOLE_DEFAULT = False
    
    @classmethod
    def validate_batch_size(cls, batch_size: int) -> int:
        """验证并限制batch_size"""
        return max(cls.MIN_BATCH_SIZE, min(batch_size, cls.MAX_BATCH_SIZE))


class LogConfig:
    """日志配置管理类"""
    
    _enable_logging = True
    _console_handler = None
    
    @classmethod
    def enable_logging(cls, enable: bool = True, to_console: bool = False) -> None:
        """
        启用或禁用日志
        
        参数:
            enable: 是否启用日志
            to_console: 是否输出到控制台
        """
        cls._enable_logging = enable
        
        if enable and to_console:
            # 避免重复添加处理器
            if cls._console_handler is None:
                cls._console_handler = logging.StreamHandler()
                cls._console_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                # 移除旧的处理器（如果存在）
                logger.handlers.clear()
                logger.addHandler(cls._console_handler)
        elif not to_console and cls._console_handler is not None:
            # 移除控制台处理器
            logger.removeHandler(cls._console_handler)
            cls._console_handler = None
    
    @classmethod
    def is_enabled(cls) -> bool:
        """检查日志是否启用"""
        return cls._enable_logging
    
    @classmethod
    def disable_logging(cls) -> None:
        """禁用日志"""
        cls._enable_logging = False
    
    @classmethod
    def get_logger(cls, name: str):
        """获取日志对象"""
        return logging.getLogger(name)


def _log_info(msg: str) -> None:
    """条件日志记录（信息级别）"""
    if LogConfig.is_enabled():
        logger.info(msg)


def _log_warning(msg: str) -> None:
    """条件日志记录（警告级别）"""
    if LogConfig.is_enabled():
        logger.warning(msg)


def _log_debug(msg: str) -> None:
    """条件日志记录（调试级别）"""
    if LogConfig.is_enabled():
        logger.debug(msg)


def _log_error(msg: str) -> None:
    """条件日志记录（错误级别）"""
    if LogConfig.is_enabled():
        logger.error(msg)


class VibrationMetadata:
    """振动数据元数据容器"""
    
    def __init__(self, metadata: Dict):
        """
        初始化振动元数据
        
        参数:
            metadata: 元数据字典，通常包含：
                - file_path: 数据文件路径
                - sensor_id: 传感器ID
                - month, day, hour: 时间信息
                - actual_length: 实际数据长度
                - missing_rate: 缺失率
                - extreme_rms_indices: 极端RMS窗口索引
        """
        self.metadata = metadata
        self.file_path = metadata.get('file_path')
        self.sensor_id = metadata.get('sensor_id')
    
    def is_valid(self) -> bool:
        """检查元数据是否有效"""
        return self.file_path is not None and os.path.exists(self.file_path)


class WindMetadata:
    """风数据元数据容器"""
    
    def __init__(self, metadata: Dict):
        """
        初始化风元数据
        
        参数:
            metadata: 元数据字典，通常包含：
                - file_path: 数据文件路径
                - sensor_id: 传感器ID
                - month, day, hour: 时间信息
        """
        self.metadata = metadata
        self.file_path = metadata.get('file_path')
        self.sensor_id = metadata.get('sensor_id')
    
    def is_valid(self) -> bool:
        """检查元数据是否有效"""
        return self.file_path is not None and os.path.exists(self.file_path)


def _parse_vibration_worker(args: Tuple) -> Dict:
    """
    振动数据解析工作函数（用于多进程）
    
    参数:
        args: (metadata, enable_extreme_window, enable_denoise, index)
    
    返回:
        解析结果字典，包含索引以确保顺序对应
    """
    metadata, enable_extreme_window, enable_denoise, index = args
    
    try:
        result = parse_single_metadata_to_vibration_data(
            metadata,
            enable_extreme_window=enable_extreme_window,
            enable_denoise=enable_denoise,
        )
        return {
            'status': 'success',
            'data': result,
            'error': None,
            'metadata_id': metadata.get('file_path', 'unknown'),
            'index': index,
        }
    except Exception as e:
        return {
            'status': 'failed',
            'data': None,
            'error': str(e),
            'metadata_id': metadata.get('file_path', 'unknown'),
            'index': index,
        }


def _parse_wind_worker(args: Tuple) -> Dict:
    """
    风数据解析工作函数（用于多进程）
    
    参数:
        args: (metadata, enable_denoise, index)
    
    返回:
        解析结果字典，包含索引以确保顺序对应
    """
    metadata, enable_denoise, index = args
    
    try:
        result = parse_single_metadata_to_wind_data(
            metadata,
            enable_denoise=enable_denoise,
        )
        return {
            'status': 'success',
            'data': result,
            'error': None,
            'metadata_id': metadata.get('file_path', 'unknown'),
            'index': index,
        }
    except Exception as e:
        return {
            'status': 'failed',
            'data': None,
            'error': str(e),
            'metadata_id': metadata.get('file_path', 'unknown'),
            'index': index,
        }


class MetadataParser:
    """
    元数据解析处理器
    
    功能：
    - 存储振动和风数据元数据
    - 批量解析元数据为原始数据
    - 支持多进程并行处理
    - 队列式批处理
    """
    
    def __init__(
        self,
        vibration_metadata: Optional[List[Dict]] = None,
        wind_metadata: Optional[List[Dict]] = None,
        num_workers: Optional[int] = None,
        validate_metadata: bool = False,
    ):
        """
        初始化元数据解析器
        
        参数:
            vibration_metadata: 振动元数据列表
            wind_metadata: 风元数据列表
            num_workers: 处理进程数
                - None: 单进程模式（用于 PyTorch DataLoader） ⭐ 推荐
                - 0 或其他值: 同None，归于单进程模式（与参数0等同）
                  注: 当前不支持自动CPU核心数检测，建议显式指定进程数或使用None
            validate_metadata: 是否验证元数据有效性（检查文件是否存在）
                - False: 不验证（默认，用于已预处理的数据）
                - True: 验证并过滤无效元数据
        """
        vib_metadata_objs = [
            VibrationMetadata(m) for m in (vibration_metadata or [])
        ]
        wind_metadata_objs = [
            WindMetadata(m) for m in (wind_metadata or [])
        ]
        
        self.validate_metadata = validate_metadata
        self.vibration_metadata_list = self._filter_valid_metadata(vib_metadata_objs)
        self.wind_metadata_list = self._filter_valid_metadata(wind_metadata_objs)
        
        # num_workers处理: 0或其他值都归于单进程模式(None)
        self.num_workers = num_workers if num_workers is not None and num_workers > 0 else None
        
        vib_filtered_out = len(vib_metadata_objs) - len(self.vibration_metadata_list)
        wind_filtered_out = len(wind_metadata_objs) - len(self.wind_metadata_list)
        
        _log_info(
            f"初始化 MetadataParser: "
            f"振动元数据 {len(self.vibration_metadata_list)} 条"
            f"{'(过滤掉 ' + str(vib_filtered_out) + ' 条无效)' if vib_filtered_out > 0 else ''}, "
            f"风元数据 {len(self.wind_metadata_list)} 条"
            f"{'(过滤掉 ' + str(wind_filtered_out) + ' 条无效)' if wind_filtered_out > 0 else ''}"
        )
        _log_info(
            f"多进程模式: {'单进程' if self.num_workers is None else '多进程'} "
            f"(num_workers={self.num_workers})"
        )
        _log_info(
            f"元数据验证: {'启用' if validate_metadata else '禁用'}"
        )
    
    def _filter_valid_metadata(
        self,
        metadata_list: List[Union['VibrationMetadata', 'WindMetadata']]
    ) -> List[Union['VibrationMetadata', 'WindMetadata']]:
        """
        过滤无效元数据（文件不存在）
        
        只有当 validate_metadata=True 时才会执行过滤
        """
        if not self.validate_metadata:
            return metadata_list
        
        valid_list = []
        
        for meta in metadata_list:
            if meta.is_valid():
                valid_list.append(meta)
            else:
                _log_debug(
                    f"过滤掉无效元数据: {meta.file_path} (文件不存在或路径无效)"
                )
        
        return valid_list
    
    def add_vibration_metadata(self, metadata_list: List[Dict]):
        """添加振动元数据"""
        self.vibration_metadata_list.extend([
            VibrationMetadata(m) for m in metadata_list
        ])
    
    def add_wind_metadata(self, metadata_list: List[Dict]):
        """添加风元数据"""
        self.wind_metadata_list.extend([
            WindMetadata(m) for m in metadata_list
        ])
    
    def parse_data(
        self,
        mode: Literal['vibration', 'wind', 'both'] = 'both',
        batch_size: Optional[int] = None,
        enable_extreme_window: bool = False,
        enable_denoise: bool = False,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        统一数据解析接口
        
        参数:
            mode: 返回模式
                - 'vibration': 仅返回振动数据
                - 'wind': 仅返回风数据
                - 'both': 返回同时对齐的振动和风数据
            batch_size: 批处理大小（若为None，使用ParseConfig.DEFAULT_BATCH_SIZE）
                - 多进程模式下：每批提交给进程池的任务数
                - 单进程模式下：每批处理的任务数（用于进度展示）
            enable_extreme_window: 是否提取极端窗口（仅对振动数据有效）
            enable_denoise: 去噪开关
            show_progress: 是否显示进度条
        
        返回:
            根据 mode 返回不同格式：
            
            - mode='vibration': 
              List[Dict] 每个元素为: {
                'vibration': {单个振动数据结果},
                'statistics': {
                    'index': 在原始列表中的索引,
                    'metadata_id': 元数据标识,
                    'status': 'success' 或 'failed',
                    'error': 错误信息（失败时）
                }
              }
            
            - mode='wind': 
              List[Dict] 同上结构
            
            - mode='both': 
              List[Dict] 每个元素为: {
                'vibration': {单个振动数据结果或None},
                'wind': {单个风数据结果或None},
                'statistics': {
                    'vib_status': 振动数据状态,
                    'wind_status': 风数据状态,
                    'vib_metadata_id': 振动元数据标识,
                    'wind_metadata_id': 风元数据标识,
                }
              }
              
              **注意**: 在 'both' 模式下，通过配对索引确保数据对齐。
                     如果数据量不等，短的列表用 None 补齐。
        """
        # 使用默认配置
        if batch_size is None:
            batch_size = ParseConfig.DEFAULT_BATCH_SIZE
        else:
            batch_size = ParseConfig.validate_batch_size(batch_size)
        
        _log_info(f"开始解析数据 (mode={mode}, batch_size={batch_size})")
        
        if mode == 'vibration':
            return self._parse_vibration_data(
                batch_size=batch_size,
                enable_extreme_window=enable_extreme_window,
                enable_denoise=enable_denoise,
                show_progress=show_progress,
            )
        elif mode == 'wind':
            return self._parse_wind_data(
                batch_size=batch_size,
                enable_denoise=enable_denoise,
                show_progress=show_progress,
            )
        elif mode == 'both':
            return self._parse_both_data(
                batch_size=batch_size,
                enable_extreme_window=enable_extreme_window,
                enable_denoise=enable_denoise,
                show_progress=show_progress,
            )
        raise ValueError(f"不支持的 mode: {mode}，应为 'vibration', 'wind' 或 'both'")
    
    def _process_batch_multiprocess(
        self,
        worker_args_list: List[Tuple],
        worker_func,
        batch_size: int,
        desc: str,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        多进程批处理
        
        参数:
            worker_args_list: 工作参数列表
            worker_func: 工作函数
            batch_size: 每批大小
            desc: 进度条描述
            show_progress: 是否显示进度条
        
        返回:
            结果列表（保持原始顺序）
        """
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for batch_start in range(0, len(worker_args_list), batch_size):
                batch_end = min(batch_start + batch_size, len(worker_args_list))
                batch_args = worker_args_list[batch_start:batch_end]
                
                futures = {
                    executor.submit(worker_func, args): idx
                    for idx, args in enumerate(batch_args, start=batch_start)
                }
                
                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(futures), desc=f"{desc} (batch {batch_start//batch_size + 1})")
                
                for future in iterator:
                    result = future.result()
                    results[result['index']] = result
        
        sorted_results = [results[i] for i in range(len(results))]
        return sorted_results
    
    def _process_batch_singleprocess(
        self,
        worker_args_list: List[Tuple],
        worker_func,
        batch_size: int,
        desc: str,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        单进程批处理（PyTorch DataLoader 兼容模式）
        
        参数:
            worker_args_list: 工作参数列表
            worker_func: 工作函数
            batch_size: 每批大小（用于进度展示）
            desc: 进度条描述
            show_progress: 是否显示进度条
        
        返回:
            结果列表（保持原始顺序）
        """
        results = {}
        iterator = enumerate(worker_args_list)
        
        if show_progress:
            iterator = tqdm(iterator, total=len(worker_args_list), desc=desc)
        
        for idx, args in iterator:
            result = worker_func(args)
            results[result['index']] = result
        
        sorted_results = [results[i] for i in range(len(results))]
        return sorted_results
    
    def _parse_vibration_data(
        self,
        batch_size: int = 32,
        enable_extreme_window: bool = False,
        enable_denoise: bool = False,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        内部方法：解析仅振动数据
        
        返回:
            List[Dict] 格式同 parse_data 中 mode='vibration' 的说明
        """
        _log_info(f"开始解析振动数据: 共 {len(self.vibration_metadata_list)} 条元数据")
        
        if len(self.vibration_metadata_list) == 0:
            _log_warning("无振动元数据可解析")
            return []
        
        worker_args = [
            (m.metadata, enable_extreme_window, enable_denoise, idx)
            for idx, m in enumerate(self.vibration_metadata_list)
        ]
        
        if self.num_workers is None:
            _log_info("使用单进程模式处理数据（PyTorch DataLoader 兼容）")
            raw_results = self._process_batch_singleprocess(
                worker_args,
                _parse_vibration_worker,
                batch_size,
                "解析振动数据",
                show_progress,
            )
        else:
            _log_info(f"使用多进程模式处理数据（num_workers={self.num_workers}）")
            raw_results = self._process_batch_multiprocess(
                worker_args,
                _parse_vibration_worker,
                batch_size,
                "解析振动数据",
                show_progress,
            )
        
        # 转换为标准返回格式
        results = []
        for raw_result in raw_results:
            formatted_result = {
                'vibration': raw_result.get('data'),
                'statistics': {
                    'index': raw_result.get('index'),
                    'metadata_id': raw_result.get('metadata_id'),
                    'status': raw_result.get('status'),
                    'error': raw_result.get('error')
                }
            }
            results.append(formatted_result)
        
        _log_info(f"振动数据解析完成: 共 {len(results)} 条")
        return results
    
    def _parse_wind_data(
        self,
        batch_size: int = 32,
        enable_denoise: bool = False,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        内部方法：解析仅风数据
        
        返回:
            List[Dict] 格式同 parse_data 中 mode='wind' 的说明
        """
        _log_info(f"开始解析风数据: 共 {len(self.wind_metadata_list)} 条元数据")
        
        if len(self.wind_metadata_list) == 0:
            _log_warning("无风元数据可解析")
            return []
        
        worker_args = [
            (m.metadata, enable_denoise, idx)
            for idx, m in enumerate(self.wind_metadata_list)
        ]
        
        if self.num_workers is None:
            _log_info("使用单进程模式处理数据（PyTorch DataLoader 兼容）")
            raw_results = self._process_batch_singleprocess(
                worker_args,
                _parse_wind_worker,
                batch_size,
                "解析风数据",
                show_progress,
            )
        else:
            _log_info(f"使用多进程模式处理数据（num_workers={self.num_workers}）")
            raw_results = self._process_batch_multiprocess(
                worker_args,
                _parse_wind_worker,
                batch_size,
                "解析风数据",
                show_progress,
            )
        
        # 转换为标准返回格式
        results = []
        for raw_result in raw_results:
            formatted_result = {
                'wind': raw_result.get('data'),
                'statistics': {
                    'index': raw_result.get('index'),
                    'metadata_id': raw_result.get('metadata_id'),
                    'status': raw_result.get('status'),
                    'error': raw_result.get('error')
                }
            }
            results.append(formatted_result)
        
        _log_info(f"风数据解析完成: 共 {len(results)} 条")
        return results
    
    def _parse_both_data(
        self,
        batch_size: int = 32,
        enable_extreme_window: bool = False,
        enable_denoise: bool = False,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        内部方法：同时解析并对齐振动和风数据
        
        关键要点：
        - 通过配对索引确保两个数据窗口对齐
        - 如果数据量不相等，较短的列表用 None 补齐
        - 返回的列表长度为 max(vib数, wind数)
        
        返回:
            List[Dict] 格式同 parse_data 中 mode='both' 的说明
        """
        _log_info(
            f"开始同时解析振动和风数据 "
            f"(vib={len(self.vibration_metadata_list)}, "
            f"wind={len(self.wind_metadata_list)})"
        )
        
        vib_results = self._parse_vibration_data(
            batch_size=batch_size,
            enable_extreme_window=enable_extreme_window,
            enable_denoise=enable_denoise,
            show_progress=show_progress,
        )
        
        wind_results = self._parse_wind_data(
            batch_size=batch_size,
            enable_denoise=enable_denoise,
            show_progress=show_progress,
        )
        
        max_len = max(len(vib_results), len(wind_results))
        
        paired_results = []
        for i in range(max_len):
            vib_result = vib_results[i] if i < len(vib_results) else None
            wind_result = wind_results[i] if i < len(wind_results) else None
            
            paired_item = {
                'vibration': vib_result['data'] if vib_result and vib_result.get('status') == 'success' else None,
                'wind': wind_result['data'] if wind_result and wind_result.get('status') == 'success' else None,
                'statistics': {
                    'vib_index': i if vib_result else None,
                    'wind_index': i if wind_result else None,
                    'vib_status': vib_result.get('status', 'missing') if vib_result else 'missing',
                    'wind_status': wind_result.get('status', 'missing') if wind_result else 'missing',
                    'vib_metadata_id': vib_result.get('metadata_id') if vib_result else None,
                    'wind_metadata_id': wind_result.get('metadata_id') if wind_result else None,
                    'vib_error': vib_result.get('error') if vib_result else None,
                    'wind_error': wind_result.get('error') if wind_result else None,
                }
            }
            paired_results.append(paired_item)
        
        _log_info(f"数据对齐完成: 共 {len(paired_results)} 对")
        return paired_results
    
    def get_vibration_metadata_count(self) -> int:
        """获取振动元数据数量"""
        return len(self.vibration_metadata_list)
    
    def get_wind_metadata_count(self) -> int:
        """获取风元数据数量"""
        return len(self.wind_metadata_list)
    
    def get_vibration_metadata(self) -> List[Dict]:
        """获取所有振动元数据"""
        return [m.metadata for m in self.vibration_metadata_list]
    
    def get_wind_metadata(self) -> List[Dict]:
        """获取所有风元数据"""
        return [m.metadata for m in self.wind_metadata_list]


class AnnotatedDatasetParser(MetadataParser):
    """
    标注数据集解析器 - 继承自 MetadataParser
    
    功能:
    - 基于标注结果和原始数据构建训练数据集
    - 支持分类任务（返回标签0、1、2、3）
    - 支持预测任务（指定预测步长，返回输入和标签）
    - 生成与 PyTorch Dataset 兼容的格式
    
    工作流程:
    1. 加载标注结果 JSON 文件
    2. 与解析的数据进行对齐
    3. 支持两种数据基准：
       - 'annotation': 以标注结果为基准，仅包含已标注的样本
       - 'data': 以原始数据为基准，未标注的样本标记为 -1 或 'unknown'
    4. 生成数据集
    """
    
    def __init__(
        self,
        vibration_metadata: Optional[List[Dict]] = None,
        wind_metadata: Optional[List[Dict]] = None,
        annotation_result_path: Optional[str] = None,
        num_workers: Optional[int] = None,
        validate_metadata: bool = False,
    ):
        """
        初始化标注数据集解析器
        
        参数:
            vibration_metadata: 振动元数据列表
            wind_metadata: 风元数据列表
            annotation_result_path: 标注结果 JSON 文件路径
            num_workers: 处理进程数
            validate_metadata: 是否验证元数据有效性
        """
        super().__init__(
            vibration_metadata=vibration_metadata,
            wind_metadata=wind_metadata,
            num_workers=num_workers,
            validate_metadata=validate_metadata,
        )
        
        self.annotation_result_path = annotation_result_path
        self.annotation_data = []
        
        if annotation_result_path and os.path.exists(annotation_result_path):
            self._load_annotations()
    
    def _load_annotations(self) -> None:
        """从 JSON 文件加载标注结果"""
        _log_info(f"加载标注结果: {self.annotation_result_path}")
        
        with open(self.annotation_result_path, 'r', encoding='utf-8') as f:
            self.annotation_data = json.load(f)
        
        _log_info(f"加载完成: 共 {len(self.annotation_data)} 条标注")
    
    def set_annotation_result_path(self, path: str) -> None:
        """设置标注结果路径并加载"""
        self.annotation_result_path = path
        if os.path.exists(path):
            self._load_annotations()
        else:
            logger.warning(f"标注文件不存在: {path}")
            self.annotation_data = []
    
    def build_classification_dataset(
        self,
        data_base: Literal['annotation', 'data'] = 'annotation',
        mode: Literal['vibration', 'wind', 'both'] = 'vibration',
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        构建分类任务数据集
        
        参数:
            data_base: 数据基准
                - 'annotation': 以标注结果为基准，仅包含已标注的样本
                - 'data': 以原始数据为基准，未标注的标记为 -1
            mode: 数据模式（'vibration', 'wind', 'both'）
            show_progress: 是否显示进度条
        
        返回:
            List[Dict] 每个元素为:
            {
                'data': {
                    'vibration': 一维或二维数组 (或None),
                    'wind': 一维或二维数组 (或None),
                },
                'label': 0/1/2/3 (或 -1 表示未标注),
                'metadata': {
                    'sensor_id': str,
                    'file_path': str,
                    'window_index': int,
                    'annotation_raw': 原始标注字符串,
                },
                'statistics': {
                    'vib_status': 'success'/'failed'/'missing',
                    'wind_status': 'success'/'failed'/'missing',
                }
            }
        """
        _log_info(f"构建分类数据集 (base={data_base}, mode={mode})")
        
        parsed_data = self.parse_data(mode=mode, show_progress=show_progress)
        
        dataset = []
        
        if data_base == 'annotation':
            dataset = self._build_classification_from_annotation(
                parsed_data=parsed_data,
                mode=mode,
            )
        elif data_base == 'data':
            dataset = self._build_classification_from_data(
                parsed_data=parsed_data,
                mode=mode,
            )
        else:
            raise ValueError(f"不支持的 data_base: {data_base}")
        
        _log_info(f"分类数据集构建完成: 共 {len(dataset)} 个样本")
        return dataset
    
    def _build_classification_from_annotation(
        self,
        parsed_data: List[Dict],
        mode: str,
    ) -> List[Dict]:
        """从标注结果构建分类数据集"""
        dataset = []
        
        for ann in tqdm(self.annotation_data, desc="对齐标注数据"):
            file_path = ann.get('file_path')
            window_index = ann.get('window_index')
            annotation_raw = ann.get('annotation', '')
            label = self._convert_annotation_to_label(annotation_raw)
            
            matching_item = self._find_matching_parsed_data(
                parsed_data,
                file_path,
                window_index,
            )
            
            if matching_item is None:
                logger.warning(
                    f"未找到匹配的数据: {file_path} @ window {window_index}"
                )
                continue
            
            dataset_item = {
                'data': self._extract_data_from_parsed(matching_item, mode),
                'label': label,
                'metadata': {
                    'sensor_id': ann.get('sensor_id'),
                    'file_path': file_path,
                    'window_index': window_index,
                    'annotation_raw': annotation_raw,
                    'time': ann.get('time'),
                },
                'statistics': {
                    'vib_status': matching_item.get('statistics', {}).get(
                        'vib_status', 'unknown'
                    ),
                    'wind_status': matching_item.get('statistics', {}).get(
                        'wind_status', 'unknown'
                    ),
                }
            }
            dataset.append(dataset_item)
        
        return dataset
    
    def _build_classification_from_data(
        self,
        parsed_data: List[Dict],
        mode: str,
    ) -> List[Dict]:
        """从原始数据构建分类数据集，未标注的标记为 -1"""
        dataset = []
        annotation_map = self._build_annotation_map()
        
        for item in tqdm(parsed_data, desc="对齐数据集"):
            file_path = item.get('statistics', {}).get('vib_metadata_id')
            window_index = ParseConfig.DEFAULT_WINDOW_INDEX
            
            annotation_info = annotation_map.get((file_path, window_index))
            
            if annotation_info:
                label = self._convert_annotation_to_label(
                    annotation_info.get('annotation', '')
                )
                annotation_raw = annotation_info.get('annotation', '')
            else:
                label = -1
                annotation_raw = 'unknown'
            
            dataset_item = {
                'data': self._extract_data_from_parsed(item, mode),
                'label': label,
                'metadata': {
                    'sensor_id': annotation_info.get('sensor_id') if annotation_info else 'unknown',
                    'file_path': file_path,
                    'window_index': window_index,
                    'annotation_raw': annotation_raw,
                    'time': annotation_info.get('time') if annotation_info else 'unknown',
                },
                'statistics': {
                    'vib_status': item.get('statistics', {}).get(
                        'vib_status', 'unknown'
                    ),
                    'wind_status': item.get('statistics', {}).get(
                        'wind_status', 'unknown'
                    ),
                }
            }
            dataset.append(dataset_item)
        
        return dataset
    
    def build_prediction_dataset(
        self,
        prediction_step: int,
        data_base: Literal['annotation', 'data'] = 'annotation',
        mode: Literal['vibration', 'wind', 'both'] = 'vibration',
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        构建预测任务数据集
        
        参数:
            prediction_step: 预测步长（如100表示用100个样本预测接下来的4步）
            data_base: 数据基准 ('annotation' 或 'data')
            mode: 数据模式
            show_progress: 是否显示进度条
        
        返回:
            List[Dict] 每个元素为:
            {
                'input': {
                    'vibration': 形状为 (prediction_step,) 的数组,
                    'wind': 形状为 (prediction_step,) 的数组,
                },
                'label': {
                    'vibration': 形状为 (4,) 的数组,
                    'wind': 形状为 (4,) 的数组,
                },
                'metadata': {
                    'sensor_id': str,
                    'file_path': str,
                    'annotation_raw': 标注字符串,
                },
                'statistics': {...}
            }
        
        注意:
            - label 总是返回 4 步的预测
            - 如果数据不足，会用 NaN 或 0 补齐
        """
        _log_info(
            f"构建预测数据集 "
            f"(step={prediction_step}, base={data_base}, mode={mode})"
        )
        
        parsed_data = self.parse_data(mode=mode, show_progress=show_progress)
        dataset = []
        
        for item in tqdm(parsed_data, desc="构建预测样本"):
            vib_data = item.get('vibration')
            wind_data = item.get('wind')
            
            file_path = item.get('statistics', {}).get('vib_metadata_id')
            annotation_info = self._get_annotation_by_file(file_path)
            
            input_seq = {
                'vibration': self._prepare_input_sequence(vib_data, prediction_step),
                'wind': self._prepare_input_sequence(wind_data, prediction_step),
            }
            
            label_seq = {
                'vibration': self._prepare_label_sequence(vib_data, prediction_step),
                'wind': self._prepare_label_sequence(wind_data, prediction_step),
            }
            
            dataset_item = {
                'input': input_seq,
                'label': label_seq,
                'metadata': {
                    'sensor_id': annotation_info.get('sensor_id') if annotation_info else 'unknown',
                    'file_path': file_path,
                    'annotation_raw': annotation_info.get('annotation', '') if annotation_info else 'unknown',
                },
                'statistics': {
                    'vib_status': item.get('statistics', {}).get('vib_status', 'unknown'),
                    'wind_status': item.get('statistics', {}).get('wind_status', 'unknown'),
                }
            }
            dataset.append(dataset_item)
        
        _log_info(f"预测数据集构建完成: 共 {len(dataset)} 个样本")
        return dataset
    
    def save_dataset_as_torch(
        self,
        dataset: List[Dict],
        output_path: str,
        format: Literal['npz', 'pt'] = 'pt',
    ) -> None:
        """
        将数据集保存为 PyTorch 或 NumPy 格式
        
        参数:
            dataset: 数据集列表
            output_path: 输出路径
            format: 保存格式 ('npz' 或 'pt')
        """
        _log_info(f"保存数据集为 {format} 格式: {output_path}")
        
        import numpy as np
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if format == 'npz':
            self._save_as_npz(dataset, output_path)
        elif format == 'pt':
            self._save_as_torch(dataset, output_path)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        _log_info(f"保存完成: {output_path}")
    
    def _save_as_npz(self, dataset: List[Dict], output_path: str) -> None:
        """
        保存为 NPZ 格式
        
        NPZ 文件结构:
        - 'labels': 标签数组
        - 'metadata.json': 元数据和数据引用（JSON 格式）
        - 'data_*.npy': 各样本数据文件
        """
        import numpy as np
        
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        labels = np.array([item.get('label') for item in dataset])
        
        metadata_list = []
        data_files = {}
        
        for idx, item in enumerate(dataset):
            data = item.get('data')
            meta = item.get('metadata', {})
            
            if data is not None:
                data_vib = data.get('vibration')
                data_wind = data.get('wind')
                
                if data_vib is not None:
                    vib_file = f"data_{idx}_vibration.npy"
                    np.save(os.path.join(output_dir, vib_file), np.asarray(data_vib))
                    data_files[f'{idx}_vibration'] = vib_file
                
                if data_wind is not None:
                    wind_file = f"data_{idx}_wind.npy"
                    np.save(os.path.join(output_dir, wind_file), np.asarray(data_wind))
                    data_files[f'{idx}_wind'] = wind_file
            
            metadata_list.append({
                'index': idx,
                'label': int(item.get('label', -1)),
                'metadata': meta,
                'statistics': item.get('statistics', {}),
                'data_files': data_files.get(f'{idx}_vibration') or data_files.get(f'{idx}_wind') or None,
            })
        
        metadata_json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        np.savez_compressed(
            output_path,
            labels=labels,
            metadata_json=metadata_json_path,
        )
        
        _log_info(f"NPZ 数据集已保存: {output_path}")
        _log_info(f"元数据已保存: {metadata_json_path}")
        _log_info(f"数据文件数: {len(data_files)}")
    
    def _save_as_torch(self, dataset: List[Dict], output_path: str) -> None:
        """保存为 PyTorch 格式"""
        import torch
        
        data_list = [item.get('data') for item in dataset]
        labels = torch.tensor(
            [item.get('label') for item in dataset],
            dtype=torch.long
        )
        
        torch.save({
            'data': data_list,
            'labels': labels,
            'metadata': [item.get('metadata') for item in dataset],
        }, output_path)
    
    # ============== 辅助方法 ==============
    
    def _convert_annotation_to_label(self, annotation: str) -> int:
        """将标注字符串转换为标签数字 (0/1/2/3)"""
        annotation_str = str(annotation).strip()
        
        try:
            label = int(annotation_str)
            if label in [0, 1, 2, 3]:
                return label
        except ValueError:
            pass
        
        mapping = {
            'normal': 0,
            '正常': 0,
            'abnormal': 1,
            '异常': 1,
            'severe': 2,
            '严重': 2,
            'unknown': 3,
            '未知': 3,
        }
        
        return mapping.get(annotation_str.lower(), 3)
    
    def _find_matching_parsed_data(
        self,
        parsed_data: List[Dict],
        file_path: str,
        window_index: int,
    ) -> Optional[Dict]:
        """在解析数据中找到匹配的项"""
        for item in parsed_data:
            stats = item.get('statistics', {})
            item_file_path = stats.get('vib_metadata_id') or stats.get('wind_metadata_id')
            
            if item_file_path == file_path:
                return item
        
        return None
    
    def _extract_data_from_parsed(
        self,
        parsed_item: Dict,
        mode: str,
    ) -> Dict:
        """从解析结果中提取数据"""
        if mode == 'vibration':
            return {
                'vibration': parsed_item.get('vibration'),
                'wind': None,
            }
        elif mode == 'wind':
            return {
                'vibration': None,
                'wind': parsed_item.get('wind'),
            }
        elif mode == 'both':
            return {
                'vibration': parsed_item.get('vibration'),
                'wind': parsed_item.get('wind'),
            }
        
        return {}
    
    def _build_annotation_map(self) -> Dict[Tuple, Dict]:
        """构建标注映射字典，key为 (file_path, window_index)"""
        ann_map = {}
        
        for ann in self.annotation_data:
            key = (ann.get('file_path'), ann.get('window_index'))
            ann_map[key] = ann
        
        return ann_map
    
    def _get_annotation_by_file(self, file_path: str) -> Optional[Dict]:
        """按文件路径获取标注信息（返回第一个匹配）"""
        for ann in self.annotation_data:
            if ann.get('file_path') == file_path:
                return ann
        
        return None
    
    def _prepare_input_sequence(self, data, prediction_step: int):
        """准备输入序列"""
        import numpy as np
        
        if data is None:
            return np.full(prediction_step, np.nan)
        
        data_array = np.asarray(data).flatten()
        
        if len(data_array) >= prediction_step:
            return data_array[:prediction_step]
        else:
            padded = np.full(prediction_step, np.nan)
            padded[:len(data_array)] = data_array
            return padded
    
    def _prepare_label_sequence(self, data, prediction_step: int, num_steps: int = 4):
        """准备标签序列（预测接下来的 num_steps 步）"""
        import numpy as np
        
        if data is None:
            return np.full(num_steps, np.nan)
        
        data_array = np.asarray(data).flatten()
        
        if len(data_array) > prediction_step:
            label_data = data_array[prediction_step:prediction_step + num_steps]
            
            if len(label_data) < num_steps:
                padded = np.full(num_steps, np.nan)
                padded[:len(label_data)] = label_data
                return padded
            
            return label_data
        else:
            return np.full(num_steps, np.nan)
