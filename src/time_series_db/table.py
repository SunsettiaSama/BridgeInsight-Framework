



# ########################
# 需要把对外的接口都统一一下,统一成df的输出即可
# ########################




"""

Index           timestamp  hour  day  month                                  ST-UAN-G02-002-01
0        0 2025-01-01 00:00:00     0    1      1                                               None
1        1 2025-09-01 00:00:00     0    1      9  {'file_path': 'F:/Research/Vibration Character...
2        2 2025-09-01 01:00:00     1    1      9  {'file_path': 'F:/Research/Vibration Character...
3        3 2025-09-01 02:00:00     2    1      9  {'file_path': 'F:/Research/Vibration Character...
4        4 2025-09-01 03:00:00     3    1      9  {'file_path': 'F:/Research/Vibration Character...
..     ...                 ...   ...  ...    ...                                                ...
716    716 2025-09-30 19:00:00    19   30      9  {'file_path': 'F:/Research/Vibration Character...
717    717 2025-09-30 20:00:00    20   30      9  {'file_path': 'F:/Research/Vibration Character...
718    718 2025-09-30 21:00:00    21   30      9  {'file_path': 'F:/Research/Vibration Character...
719    719 2025-09-30 22:00:00    22   30      9  {'file_path': 'F:/Research/Vibration Character...
720    720 2025-09-30 23:00:00    23   30      9  {'file_path': 'F:/Research/Vibration Character...

[721 rows x 6 columns]


"""









import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import json
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import concurrent.futures
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class FileTypeMappingConfig:
    """
    文件类型映射配置类，负责管理所有映射规则
    
    支持:
    1. 从YAML/JSON文件加载配置
    2. 默认映射规则
    3. 动态更新映射规则
    4. 多种映射策略（扩展名、传感器ID、文件名模式等）
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """初始化映射配置，可选从文件加载"""
        # 初始化默认配置
        self.config = self._get_default_config()
        
        # 从文件加载配置（如果提供）
        if config_file:
            self.load_from_file(config_file)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认映射配置"""
        return {
            'extension_mapping': {
                '.vic': 'acceleration_of_cable',
                '.uan': 'wind_speed',
                '.vib': 'vibration_of_tower',
                '.dis': 'displacement_of_girder',
                '.rhs': 'humidity_and_temperature',
                '.dsp': 'displacement',
                '.txt': 'text_data',
                '.csv': 'csv_data'
            },
            'sensor_id_mapping': {
                'ST-UAN': 'vibration',
                'ST-ACC': 'acceleration',
                'ST-WND': 'wind_speed',
                'ST-TMP': 'temperature',
                'ST-HMD': 'humidity',
                'ST-PRS': 'pressure',
                'ST-DSP': 'displacement'
            },
            'sensor_type_mapping': {
                'UAN': 'vibration',
                'ACC': 'acceleration',
                'WND': 'wind_speed',
                'TMP': 'temperature',
                'HMD': 'humidity',
                'PRS': 'pressure',
                'DSP': 'displacement'
            },
            'filename_pattern_mapping': [
                {'pattern': r'vibration', 'data_type': 'vibration'},
                {'pattern': r'acceleration|accel', 'data_type': 'acceleration'},
                {'pattern': r'wind|wnd', 'data_type': 'wind_speed'},
                {'pattern': r'temperature|temp|tmp', 'data_type': 'temperature'},
                {'pattern': r'humidity|humid|hmd', 'data_type': 'humidity'},
                {'pattern': r'pressure|press|prs', 'data_type': 'pressure'},
                {'pattern': r'displacement|displ|dsp', 'data_type': 'displacement'}
            ],
            'default_data_type': 'unknown',
            'mapping_priority': [
                'extension',
                'sensor_id',
                'sensor_type',
                'filename_pattern',
                'default'
            ]
        }
    
    def load_from_file(self, file_path: str) -> None:
        """
        从YAML或JSON文件加载映射配置
        
        Args:
            file_path: 配置文件路径
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式或无效的配置
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"映射配置文件不存在: {file_path}")
        
        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                raise ValueError("仅支持YAML和JSON格式的映射配置文件")
            
            # 验证并应用配置
            self._validate_and_apply_config(config_data)
            print(f"成功加载映射配置: {file_path}")
        
        except Exception as e:
            raise ValueError(f"加载映射配置失败: {str(e)}")
    
    def _validate_and_apply_config(self, config_data: Dict[str, Any]) -> None:
        """
        验证并应用配置数据
        
        Args:
            config_data: 配置字典
        """
        valid_keys = [
            'extension_mapping', 'sensor_id_mapping', 'sensor_type_mapping',
            'filename_pattern_mapping', 'default_data_type', 'mapping_priority'
        ]
        
        # 验证配置结构
        for key in config_data.keys():
            if key not in valid_keys:
                raise ValueError(f"无效的映射配置键: {key}. 有效键为: {valid_keys}")
        
        # 应用配置
        for key, value in config_data.items():
            if key in ['extension_mapping', 'sensor_id_mapping', 'sensor_type_mapping']:
                # 验证映射字典
                if not isinstance(value, dict):
                    raise ValueError(f"配置项 {key} 必须是字典类型")
                self.config[key] = value
            elif key == 'filename_pattern_mapping':
                # 验证模式映射列表
                if not isinstance(value, list):
                    raise ValueError(f"配置项 {key} 必须是列表类型")
                for item in value:
                    if not isinstance(item, dict) or 'pattern' not in item or 'data_type' not in item:
                        raise ValueError(f"无效的文件名模式映射项: {item}")
                self.config[key] = value
            elif key == 'mapping_priority':
                # 验证优先级列表
                valid_priorities = ['extension', 'sensor_id', 'sensor_type', 'filename_pattern', 'default']
                if not isinstance(value, list):
                    raise ValueError(f"配置项 {key} 必须是列表类型")
                for item in value:
                    if item not in valid_priorities:
                        raise ValueError(f"无效的映射优先级: {item}. 有效值为: {valid_priorities}")
                self.config[key] = value
            else:
                # 其他配置项
                self.config[key] = value
    
    def add_extension_mapping(self, extension: str, data_type: str) -> None:
        """
        添加文件扩展名映射
        
        Args:
            extension: 文件扩展名（如'.uan'）
            data_type: 对应的数据类型
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        self.config['extension_mapping'][extension.lower()] = data_type
    
    def add_sensor_id_mapping(self, sensor_id_prefix: str, data_type: str) -> None:
        """
        添加传感器ID前缀映射
        
        Args:
            sensor_id_prefix: 传感器ID前缀（如'ST-WND'）
            data_type: 对应的数据类型
        """
        self.config['sensor_id_mapping'][sensor_id_prefix] = data_type
    
    def add_sensor_type_mapping(self, sensor_type_code: str, data_type: str) -> None:
        """
        添加传感器类型代码映射
        
        Args:
            sensor_type_code: 传感器类型代码（如'WND'）
            data_type: 对应的数据类型
        """
        self.config['sensor_type_mapping'][sensor_type_code.upper()] = data_type
    
    def add_filename_pattern_mapping(self, pattern: str, data_type: str) -> None:
        """
        添加文件名模式映射
        
        Args:
            pattern: 正则表达式模式
            data_type: 对应的数据类型
        """
        self.config['filename_pattern_mapping'].append({
            'pattern': pattern,
            'data_type': data_type
        })
    
    def set_default_data_type(self, data_type: str) -> None:
        """设置默认数据类型"""
        self.config['default_data_type'] = data_type
    
    def set_mapping_priority(self, priority_list: List[str]) -> None:
        """
        设置映射策略优先级
        
        Args:
            priority_list: 优先级列表，例如['extension', 'sensor_id', 'filename_pattern']
        """
        valid_priorities = ['extension', 'sensor_id', 'sensor_type', 'filename_pattern', 'default']
        for item in priority_list:
            if item not in valid_priorities:
                raise ValueError(f"无效的映射优先级: {item}. 有效值为: {valid_priorities}")
        self.config['mapping_priority'] = priority_list
    
    def get_data_type_by_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        根据元数据确定数据类型
        
        Args:
            metadata: 文件元数据字典，应包含:
                - file_extension: 文件扩展名
                - sensor_id: 传感器ID
                - sensor_type: 传感器类型
                - filename: 文件名（不含扩展名）
        
        Returns:
            确定的数据类型
        """
        # 按优先级应用映射策略
        for strategy in self.config['mapping_priority']:
            if strategy == 'extension':
                result = self._map_by_extension(metadata)
                if result != 'unknown':
                    return result
            elif strategy == 'sensor_id':
                result = self._map_by_sensor_id(metadata)
                if result != 'unknown':
                    return result
            elif strategy == 'sensor_type':
                result = self._map_by_sensor_type(metadata)
                if result != 'unknown':
                    return result
            elif strategy == 'filename_pattern':
                result = self._map_by_filename_pattern(metadata)
                if result != 'unknown':
                    return result
            elif strategy == 'default':
                return self.config['default_data_type']
        
        # 如果没有匹配的策略，返回默认类型
        return self.config['default_data_type']
    
    def _map_by_extension(self, metadata: Dict[str, Any]) -> str:
        """通过文件扩展名映射"""
        ext = metadata.get('file_extension', '').lower()
        if ext in self.config.get('extension_mapping', {}):
            return self.config['extension_mapping'][ext]
        return 'unknown'
    
    def _map_by_sensor_id(self, metadata: Dict[str, Any]) -> str:
        """通过传感器ID前缀映射"""
        sensor_id = metadata.get('sensor_id', '')
        if not sensor_id:
            return 'unknown'
        
        # 检查映射配置中的传感器ID前缀
        for prefix, data_type in self.config.get('sensor_id_mapping', {}).items():
            if sensor_id.startswith(prefix):
                return data_type
        
        return 'unknown'
    
    def _map_by_sensor_type(self, metadata: Dict[str, Any]) -> str:
        """通过传感器类型代码映射"""
        sensor_type = metadata.get('sensor_type', '')
        if not sensor_type:
            return 'unknown'
        
        sensor_type_upper = sensor_type.upper()
        if sensor_type_upper in self.config.get('sensor_type_mapping', {}):
            return self.config['sensor_type_mapping'][sensor_type_upper]
        
        return 'unknown'
    
    def _map_by_filename_pattern(self, metadata: Dict[str, Any]) -> str:
        """通过文件名模式映射"""
        filename = metadata.get('filename', '').lower()
        
        for rule in self.config.get('filename_pattern_mapping', []):
            pattern = rule.get('pattern', '')
            data_type = rule.get('data_type', 'unknown')
            
            if re.search(pattern, filename, re.IGNORECASE):
                return data_type
        
        return 'unknown'
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置的副本"""
        return self.config.copy()
    
    def save_to_file(self, file_path: str) -> None:
        """
        将当前配置保存到YAML文件
        
        Args:
            file_path: 保存的文件路径
        """
        file_path = Path(file_path)
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, sort_keys=False, allow_unicode=True)
            elif file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError("仅支持保存为YAML或JSON格式")
            
            print(f"成功保存映射配置到: {file_path}")
        except Exception as e:
            raise ValueError(f"保存映射配置失败: {str(e)}")
    
    def __repr__(self):
        """返回对象的字符串表示"""
        return (f"FileTypeMappingConfig("
                f"strategies={self.config['mapping_priority']}, "
                f"default_type={self.config['default_data_type']})")

class TimeSeriesFileIndex:
    """
    时序文件索引表
    
    特性:
    1. 时间戳作为唯一标识符（精确到小时）
    2. 支持按月、日、时进行高效检索
    3. 按数据类型分类存储文件列表
    4. 使用独立配置类管理文件类型映射
    """
    
    def __init__(self, mapping_config: Optional[Union[str, FileTypeMappingConfig]] = None):
        """
        初始化时序文件索引表
        
        Args:
            mapping_config: 可以是FileTypeMappingConfig实例，也可以是配置文件路径
        """
        # 初始化DataFrame
        self.df = pd.DataFrame(columns=[
            'month', 'day', 'hour', 'data_types', 'file_groups'
        ])
        self.df.index.name = 'timestamp'
        self.df = self.df.astype({
            'month': 'int8',
            'day': 'int8', 
            'hour': 'int8',
            'data_types': 'object',
            'file_groups': 'object'
        })
        
        # 初始化映射配置
        if isinstance(mapping_config, FileTypeMappingConfig):
            self.mapping_config = mapping_config
        elif isinstance(mapping_config, str):
            self.mapping_config = FileTypeMappingConfig(mapping_config)
        else:
            self.mapping_config = FileTypeMappingConfig()
    
    def add_file(self, file_path: str, timestamp: datetime = None, 
                data_type: str = None, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加单个文件到索引表
        
        Args:
            file_path: 文件的完整路径
            timestamp: 文件对应的时间戳，如果为None则从路径解析
            data_type: 数据类型，如果为None则通过映射规则推断
            extra_metadata: 额外的元数据，可用于映射决策
        """
        # 解析文件路径
        path = Path(file_path)
        
        # 如果没有提供时间戳，尝试从路径解析
        if timestamp is None:
            metadata = self._extract_metadata_from_path(file_path)
        else:
            # 保留现有逻辑，但需要构建基本metadata
            metadata = {
                'timestamp': timestamp,
                'file_extension': path.suffix.lower(),
                'filename': path.stem,
                'full_path': str(path),
                'parent_dirs': list(path.parent.parts)
            }
        
        # 如果没有提供数据类型，通过映射规则推断
        if data_type is None:
            # 合并元数据
            all_metadata = metadata.copy()
            if extra_metadata:
                all_metadata.update(extra_metadata)
            data_type = self.mapping_config.get_data_type_by_metadata(all_metadata)
        
        # 确保timestamp是datetime对象
        timestamp = metadata['timestamp']
        if not isinstance(timestamp, datetime):
            raise ValueError("timestamp必须是datetime对象")
        
        # 提取时间组件
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        
        # 创建时间戳索引（精确到小时）
        hourly_timestamp = datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        
        # 检查是否已存在该小时的记录
        if hourly_timestamp in self.df.index:
            # 更新现有记录
            idx = hourly_timestamp
            
            # 获取现有分组
            file_groups = self.df.at[idx, 'file_groups']
            data_types = self.df.at[idx, 'data_types']
            
            # 初始化该数据类型的文件列表
            if data_type not in file_groups:
                file_groups[data_type] = []
                if data_type not in data_types:
                    data_types.append(data_type)
            
            # 添加文件路径（去重）
            if file_path not in file_groups[data_type]:
                file_groups[data_type].append(file_path)
            
            # 更新DataFrame
            self.df.at[idx, 'file_groups'] = file_groups
            self.df.at[idx, 'data_types'] = data_types
        else:
            # 创建新记录
            file_groups = {data_type: [file_path]}
            data_types = [data_type]
            
            new_row = pd.DataFrame({
                'month': [month],
                'day': [day],
                'hour': [hour],
                'data_types': [data_types],
                'file_groups': [file_groups]
            }, index=[hourly_timestamp])
            new_row.index.name = 'timestamp'
            
            self.df = pd.concat([self.df, new_row])
    
    def add_files(self, file_paths: List[str], 
                metadata_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
                max_workers: Optional[int] = None,
                progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, List[str]]:
        """
        批量添加文件到索引表（多线程版本）
        
        Args:
            file_paths: 文件路径列表
            metadata_provider: 可选的元数据提供函数，接收文件路径，返回元数据字典
            max_workers: 最大工作线程数，None表示使用CPU核心数*2
            progress_callback: 进度回调函数，接收(已处理数, 总数)参数
        
        Returns:
            包含处理结果的字典，包括成功和失败的文件列表
        """
        # 确保线程安全的锁
        index_lock = threading.Lock()
        error_lock = threading.Lock()
        
        # 存储结果
        results = {
            'success': [],
            'failed': [],
            'errors': {}
        }
        
        # 设置默认线程数
        if max_workers is None:
            # I/O密集型任务，可以使用更多线程
            max_workers = min(32, (os.cpu_count() or 1) * 4)
        
        # 初始化进度
        total_files = len(file_paths)
        processed = 0
        
        def process_single_file(file_path):
            """处理单个文件的内部函数"""
            nonlocal processed
            try:
                extra_metadata = None
                if metadata_provider:
                    extra_metadata = metadata_provider(file_path)
                
                # 使用锁确保线程安全地添加文件
                with index_lock:
                    self.add_file(file_path, extra_metadata=extra_metadata)
                
                # 更新进度
                with error_lock:
                    processed += 1
                    results['success'].append(file_path)
                    
                    # 调用进度回调
                    if progress_callback and processed % max(1, total_files // 100) == 0:
                        progress_callback(processed, total_files)
                
                return file_path, True, None
            except Exception as e:
                error_msg = str(e)
                with error_lock:
                    processed += 1
                    results['failed'].append(file_path)
                    results['errors'][file_path] = error_msg
                    
                    # 调用进度回调
                    if progress_callback and processed % max(1, total_files // 100) == 0:
                        progress_callback(processed, total_files)
                
                return file_path, False, error_msg
        
        # 显示开始信息
        print(f"开始处理 {total_files} 个文件，使用 {max_workers} 个工作线程...")
        start_time = time.time()
        
        # 使用线程池处理文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(process_single_file, fp): fp for fp in file_paths}
            
            # 处理完成的任务（可选，用于实时显示）
            for future in as_completed(future_to_file):
                file_path, success, error = future.result()
                if not success:
                    # 可以选择在这里打印错误，或者等到最后汇总
                    pass

        # 按时间戳索引排序，确保时间序列顺序正确
        self.df = self.df.sort_index()

        # 显示处理总结
        elapsed_time = time.time() - start_time
        success_count = len(results['success'])
        failed_count = len(results['failed'])
        
        print(f"\n处理完成! 总耗时: {elapsed_time:.2f} 秒")
        print(f"成功: {success_count} 个文件 ({success_count/total_files*100:.1f}%)")
        print(f"失败: {failed_count} 个文件 ({failed_count/total_files*100:.1f}%)")
        
        if failed_count > 0:
            print("\n错误摘要:")
            for i, (file_path, error) in enumerate(results['errors'].items()):
                if i >= 5:  # 只显示前5个错误
                    print(f"  ... 及其他 {failed_count-5} 个错误")
                    break
                print(f"  {file_path}: {error}")
        
        return results
        
    def get_files_by_time_and_type(self, timestamp: datetime, data_type: str = None) -> List[str]:
        """
        根据时间戳和数据类型获取文件列表
        
        Args:
            timestamp: 查询的时间戳
            data_type: 数据类型，如果为None则返回所有类型
        
        Returns:
            匹配的文件路径列表
        """
        hourly_timestamp = datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        
        if hourly_timestamp not in self.df.index:
            return []
        
        file_groups = self.df.at[hourly_timestamp, 'file_groups']
        
        if data_type is None:
            # 返回所有类型的文件
            all_files = []
            for files in file_groups.values():
                all_files.extend(files)
            return all_files
        else:
            # 返回指定类型的文件
            return file_groups.get(data_type, [])
    
    def get_files_by_hour_and_type(self, month: int, day: int, hour: int, 
                                 data_type: str = None, year: Optional[int] = None) -> List[str]:
        """
        根据月、日、时和数据类型获取文件列表
        
        Args:
            month: 月份 (1-12)
            day: 日期 (1-31)
            hour: 小时 (0-23)
            data_type: 数据类型，如果为None则返回所有类型
            year: 年份，如果为None则使用当前年份
        
        Returns:
            匹配的文件路径列表
        """
        if year is None:
            year = datetime.now().year
        
        try:
            hourly_timestamp = datetime(year, month, day, hour)
            return self.get_files_by_time_and_type(hourly_timestamp, data_type)
        except ValueError:
            return []
    
    def get_available_data_types(self, timestamp: datetime = None) -> List[str]:
        """
        获取可用的数据类型列表
        
        Args:
            timestamp: 可选的时间戳，如果提供则只返回该时间点可用的数据类型
        
        Returns:
            数据类型列表
        """
        if timestamp is None:
            # 获取所有时间点的数据类型
            all_types = set()
            for types in self.df['data_types']:
                all_types.update(types)
            return sorted(list(all_types))
        else:
            # 获取特定时间点的数据类型
            hourly_timestamp = datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
            if hourly_timestamp in self.df.index:
                return self.df.at[hourly_timestamp, 'data_types']
            return []
    
    def get_file_groups_by_hour(self, month: int, day: int, hour: int, year: Optional[int] = None) -> Dict[str, List[str]]:
        """
        获取指定小时的所有数据类型分组
        
        Args:
            month: 月份
            day: 日期
            hour: 小时
            year: 年份，如果为None则使用当前年份
        
        Returns:
            {data_type: [file_paths]} 字典
        """
        if year is None:
            year = datetime.now().year
        
        try:
            hourly_timestamp = datetime(year, month, day, hour)
            if hourly_timestamp in self.df.index:
                return self.df.at[hourly_timestamp, 'file_groups']
            return {}
        except ValueError:
            return {}
    
    def get_data_coverage(self) -> Dict[str, Any]:
        """
        获取数据覆盖情况统计
        
        Returns:
            包含数据类型覆盖、时间覆盖等信息的字典
        """
        # 统计每种数据类型的文件数
        type_counts = {}
        for groups in self.df['file_groups']:
            for data_type, files in groups.items():
                if data_type not in type_counts:
                    type_counts[data_type] = 0
                type_counts[data_type] += len(files)
        
        # 计算总文件数
        total_files = sum(len(files) for groups in self.df['file_groups'] for files in groups.values())
        
        return {
            'data_types': sorted(list(type_counts.keys())),
            'type_counts': type_counts,
            'total_files': total_files,
            'total_hours': len(self.df),
            'time_coverage': {
                'start': self.df.index.min().isoformat() if not self.df.empty else None,
                'end': self.df.index.max().isoformat() if not self.df.empty else None,
            },
            'hours_per_type': self._calculate_hours_per_type()
        }

    def get_files_by_sensor_id(self, sensor_id: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: Optional[int] = None,
                            max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        根据sensor_id直接查找相关文件，使用多线程加速，不依赖预配置映射
        
        Args:
            sensor_id: 传感器ID
            start_time: 开始时间（包含），None表示不限制
            end_time: 结束时间（包含），None表示不限制
            limit: 限制返回结果数量，None表示不限制
            max_workers: 最大工作线程数，None表示使用默认值（CPU核心数*2）
        
        Returns:
            包含匹配文件信息的字典列表，每个字典包含:
            - file_path: 文件路径
            - timestamp: 文件所属的时间戳
            - data_type: 数据类型
            - sensor_id: 传感器ID
            - metadata: 其他元数据
        """
        start_time_total = time.time()
        
        # 确定要搜索的时间范围
        search_df = self.df.copy()
        
        # 应用时间过滤
        if start_time is not None:
            # 处理时区
            if start_time.tzinfo is None and search_df.index.tz is not None:
                start_time = start_time.replace(tzinfo=search_df.index.tz)
            elif start_time.tzinfo is not None and search_df.index.tz is None:
                start_time = start_time.replace(tzinfo=None)
            search_df = search_df[search_df.index >= start_time]
        
        if end_time is not None:
            # 处理时区
            if end_time.tzinfo is None and search_df.index.tz is not None:
                end_time = end_time.replace(tzinfo=search_df.index.tz)
            elif end_time.tzinfo is not None and search_df.index.tz is None:
                end_time = end_time.replace(tzinfo=None)
            search_df = search_df[search_df.index <= end_time]
        
        if search_df.empty:
            sys.stdout.write("\r警告: 按时间范围筛选后无数据\n")
            sys.stdout.flush()
            return []
        
        # 设置默认的最大工作线程数
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 2)  # 合理的上限
        
        total_points = len(search_df)
        sys.stdout.write(f"\r开始多线程搜索: sensor_id='{sensor_id}', 数据点数量: {total_points}, "
                        f"工作线程数: {max_workers}")
        sys.stdout.flush()
        
        # 将DataFrame分割成块，每个线程处理一个块
        df_blocks = np.array_split(search_df, max_workers)
        df_blocks = [block for block in df_blocks if not block.empty]  # 移除空块
        
        # 共享结果列表和锁
        results = []
        results_lock = threading.Lock()
        stop_event = threading.Event()  # 用于提前终止
        completed_blocks = 0
        block_lock = threading.Lock()
        block_start_times = [time.time()] * len(df_blocks)  # 记录每个块的开始时间
        block_match_counts = [0] * len(df_blocks)  # 记录每个块找到的匹配数
        
        # 创建处理单个块的函数
        def process_block(block_df, block_id):
            """处理DataFrame的一个块"""
            block_results = []
            block_start_time = time.time()
            block_start_times[block_id] = block_start_time
            
            try:
                for timestamp, row in block_df.iterrows():
                    # 检查是否需要提前终止
                    if stop_event.is_set() or (limit is not None and len(results) >= limit):
                        break
                        
                    # 检查每个数据类型的文件
                    for data_type, files in row['file_groups'].items():
                        # 再次检查终止条件
                        if stop_event.is_set() or (limit is not None and len(results) >= limit):
                            break
                            
                        for file_info in files:
                            # 提取文件路径和元数据
                            if isinstance(file_info, str):
                                file_path = file_info
                                metadata = {}
                            else:
                                file_path = file_info.get('path', file_info) if isinstance(file_info, dict) else file_info
                                metadata = file_info.get('metadata', {}) if isinstance(file_info, dict) else {}
                            
                            # 检查sensor_id是否匹配
                            file_sensor_id = None
                            if isinstance(metadata, dict):
                                # 尝试常见字段名
                                for key in ['sensor_id', 'sensorId', 'sensor', 'device_id', 'deviceId']:
                                    if key in metadata:
                                        file_sensor_id = str(metadata[key])
                                        break
                            
                            # 如果元数据中没有，尝试从文件路径中提取
                            if file_sensor_id is None:
                                file_sensor_id = self._extract_sensor_id_from_path(file_path)
                            
                            # 检查是否匹配
                            if file_sensor_id and file_sensor_id == str(sensor_id):
                                block_results.append({
                                    'file_path': file_path,
                                    'timestamp': timestamp,
                                    'data_type': data_type,
                                    'sensor_id': file_sensor_id,
                                    'hour': row['hour'],
                                    'day': row['day'],
                                    'month': row['month'],
                                    'metadata': metadata
                                })
                
                # 将块结果添加到全局结果
                with results_lock:
                    # 记录这个块找到的匹配数
                    block_match_counts[block_id] = len(block_results)
                    
                    # 再次检查limit，避免超过限制
                    if limit is not None and len(results) + len(block_results) > limit:
                        remaining = limit - len(results)
                        if remaining > 0:
                            results.extend(block_results[:remaining])
                    else:
                        results.extend(block_results)
                
                return len(block_results)
                
            except Exception as e:
                print(f"\n处理块 {block_id} 时出错: {str(e)}")
                return 0
            finally:
                # 更新完成块计数
                with block_lock:
                    nonlocal completed_blocks
                    completed_blocks += 1
        
        # 进度显示线程
        def update_progress():
            """定期更新进度显示"""
            last_update = 0
            start_time_progress = time.time()
            
            while not stop_event.is_set():
                current_time = time.time()
                if current_time - last_update < 1.0:  # 每秒更新一次
                    time.sleep(0.1)
                    continue
                    
                last_update = current_time
                
                # 计算进度
                progress = (completed_blocks / len(df_blocks)) * 100 if df_blocks else 0
                total_matches = len(results)
                
                # 计算速度
                elapsed = current_time - start_time_progress
                speed = total_points / elapsed if elapsed > 0 else 0
                
                # 计算剩余时间
                if completed_blocks > 0:
                    avg_time_per_block = elapsed / completed_blocks
                    remaining_blocks = len(df_blocks) - completed_blocks
                    est_remaining = avg_time_per_block * remaining_blocks
                else:
                    est_remaining = 0
                
                # 构建进度字符串（限制在一行内）
                progress_str = (
                    f"\r进度: {progress:5.1f}% ({completed_blocks}/{len(df_blocks)} 块) | "
                    f"匹配: {total_matches} | "
                    f"速度: {speed:5.1f} 点/秒 | "
                    f"剩余: {est_remaining:5.1f}秒"
                )
                
                # 确保不超过终端宽度
                terminal_width = 80  # 默认宽度，可以根据需要调整
                try:
                    terminal_width = os.get_terminal_size().columns
                except:
                    pass
                    
                if len(progress_str) > terminal_width:
                    progress_str = progress_str[:terminal_width-4] + "..."
                
                sys.stdout.write(progress_str)
                sys.stdout.flush()
                
                # 检查是否需要停止
                if completed_blocks >= len(df_blocks) or (limit is not None and total_matches >= limit):
                    break
                    
                time.sleep(0.1)
            
            # 最后确保换行
            sys.stdout.write("\n")
            sys.stdout.flush()
        
        # 使用线程池执行
        progress_thread = None
        try:
            # 启动进度线程
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_block = {
                    executor.submit(process_block, block_df, i): i 
                    for i, block_df in enumerate(df_blocks)
                }
                
                # 等待所有任务完成
                for future in concurrent.futures.as_completed(future_to_block):
                    block_id = future_to_block[future]
                    try:
                        future.result()
                        
                        # 检查是否达到limit
                        if limit is not None and len(results) >= limit:
                            stop_event.set()
                            # 取消未开始的任务
                            for f in future_to_block:
                                if not f.done():
                                    f.cancel()
                            break
                            
                    except Exception as e:
                        print(f"\n块 {block_id} 处理失败: {str(e)}")
        
        except KeyboardInterrupt:
            print("\n用户中断操作，正在清理...")
            stop_event.set()
            # 等待进度线程结束
            if progress_thread and progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
            raise
        finally:
            # 确保进度线程结束
            stop_event.set()
            if progress_thread and progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
        
        # 按时间戳排序结果
        results.sort(key=lambda x: x['timestamp'])
        
        # 应用limit限制（再次确保）
        if limit is not None and len(results) > limit:
            results = results[:limit]
        
        elapsed_total = time.time() - start_time_total
        total_matches = len(results)
        
        # 清晰的最终报告
        sys.stdout.write(f"\r搜索完成: 找到 {total_matches} 个与 sensor_id '{sensor_id}' 匹配的文件")
        sys.stdout.flush()
        
        # 详细统计（只在有结果时显示）
        if total_matches > 0:
            # 计算时间范围
            first_time = min(r['timestamp'] for r in results)
            last_time = max(r['timestamp'] for r in results)
            time_span = last_time - first_time if len(results) > 1 else "Unkown"
            
            sys.stdout.write(
                f"\n时间范围: {first_time} 至 {last_time} (跨度: {time_span})"
                f"\n总耗时: {elapsed_total:.2f} 秒, 平均处理速度: {total_points/elapsed_total:.1f} 时间点/秒"
            )
            sys.stdout.flush()
        
        if limit is not None and total_matches == limit:
            sys.stdout.write(f"\n注意: 结果数量已达到限制 {limit}")
            sys.stdout.flush()
        
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        return results

    def _extract_sensor_id_from_path(self, file_path: str) -> Optional[str]:
        """
        从文件路径中提取sensor_id，特别针对形如 'ST-UAN-T01-003-01_000000.UAN' 的文件名格式
        
        Args:
            file_path: 文件路径
        
        Returns:
            提取的sensor_id或None
        """
        import os
        
        # 获取文件名（不含扩展名）
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # 主要策略：处理 'ST-UAN-T01-003-01_000000' 格式的文件名
        # 以第一个下划线为分界，前半部分作为sensor_id
        if '_' in name_without_ext:
            parts = name_without_ext.split('_', 1)  # 只分割一次
            sensor_id_candidate = parts[0]
            
            # 验证是否符合传感器ID的格式特征
            # 通常包含字母、数字和连字符，且长度合理
            if self._is_valid_sensor_id_format(sensor_id_candidate):
                return sensor_id_candidate
        
        # 备选策略1：如果没有下划线，但文件名包含连字符且看起来像传感器ID
        if '-' in name_without_ext and self._is_valid_sensor_id_format(name_without_ext):
            return name_without_ext
        
        # 备选策略2：尝试查找符合 'XX-XXX-XX-XXX-XX' 模式的部分
        import re
        pattern = r'([A-Z]{2}-[A-Z0-9]{3}-[A-Z0-9]{2}-\d{3}-\d{2})'
        match = re.search(pattern, name_without_ext)
        if match:
            return match.group(1)
        
        # 备选策略3：查找开头部分包含字母和连字符的模式
        prefix_match = re.match(r'^([A-Z0-9\-]+?)(?:[_\s\d]|$)', name_without_ext)
        if prefix_match:
            candidate = prefix_match.group(1)
            # 去除末尾可能的连字符
            candidate = candidate.rstrip('-')
            if len(candidate) >= 5 and self._is_valid_sensor_id_format(candidate):
                return candidate
        
        # 无法识别，返回None
        return None

    def _is_valid_sensor_id_format(self, candidate: str) -> bool:
        """
        验证字符串是否符合传感器ID的格式特征
        
        Args:
            candidate: 待验证的字符串
        
        Returns:
            是否符合传感器ID格式
        """
        if not candidate or len(candidate) < 5 or len(candidate) > 30:
            return False
        
        # 检查是否包含有效的字符（字母、数字、连字符）
        if not all(c.isalnum() or c == '-' for c in candidate):
            return False
        
        # 检查连字符使用是否合理（不在开头/结尾，不连续）
        if candidate.startswith('-') or candidate.endswith('-') or '--' in candidate:
            return False
        
        # 检查是否包含至少一个字母（避免纯数字ID，除非是特殊情况）
        if not any(c.isalpha() for c in candidate):
            # 允许特定格式的纯数字，如包含连字符分隔
            if not re.match(r'^\d{2,3}-\d{3}-\d{2}$', candidate):
                return False
        
        return True
    
    def _search_sensor_ids_by_df_block(self, df_block, target_sensor_ids, include_metadata=True):
        """
        处理DataFrame块，搜索匹配的sensor_ids路径
        
        Args:
            df_block: 要处理的DataFrame块，包含时间序列数据
            target_sensor_ids: 要匹配的sensor_id列表
            include_metadata: 是否包含元数据信息
        
        Returns:
            pd.DataFrame: 包含处理结果的DataFrame，列包括:
                - timestamp: 时间戳索引
                - hour/day/month: 时间组件
                - 每个target_sensor_ids对应的列，值为文件路径信息或None
        """
        # 创建结果字典
        results = {
            'timestamp': [],
            'month': [], 
            'day': [],
            'hour': [],
        }
        
        # 初始化每个sensor_id的结果列
        for sensor_id in target_sensor_ids:
            results[sensor_id] = []
        
        # 遍历DataFrame块的每一行
        for timestamp, row in df_block.iterrows():
            # 添加时间信息
            results['timestamp'].append(timestamp)
            results['hour'].append(row.get('hour') if not pd.isna(row.get('hour', np.nan)) else None)
            results['day'].append(row.get('day') if not pd.isna(row.get('day', np.nan)) else None)
            results['month'].append(row.get('month') if not pd.isna(row.get('month', np.nan)) else None)
            
            # 初始化当前行的sensor数据
            row_sensor_data = {sensor_id: None for sensor_id in target_sensor_ids}
            
            # 检查是否有文件数据
            if 'file_groups' in row and isinstance(row['file_groups'], dict):
                # 创建sensor_id集合以提高查找效率
                target_sensor_set = set(target_sensor_ids)
                
                # 遍历所有数据类型和文件
                for data_type, files in row['file_groups'].items():
                    if not isinstance(files, list):
                        continue
                        
                    for file_info in files:
                        # 处理文件信息格式
                        if isinstance(file_info, str):
                            file_path = file_info
                            metadata = {}
                        elif isinstance(file_info, dict):
                            file_path = file_info.get('path', file_info.get('file_path', str(file_info)))
                            metadata = file_info.get('metadata', {})
                        else:
                            continue
                        
                        # 从元数据中提取sensor_id
                        sensor_id = None
                        if isinstance(metadata, dict):
                            for key in ['sensor_id', 'sensorId', 'sensor', 'device_id', 'deviceId']:
                                if key in metadata:
                                    sensor_id = str(metadata[key])
                                    break
                        
                        # 如果元数据中没有，尝试从路径中提取
                        if sensor_id is None:
                            sensor_id = self._extract_sensor_id_from_path(file_path)
                        
                        # 检查sensor_id是否在目标列表中
                        if sensor_id and sensor_id in target_sensor_set:
                            # 构建文件信息
                            file_data = {
                                'file_path': file_path,
                                'data_type': data_type # 存储了文件类型
                            }
                            
                            if include_metadata:
                                file_data['metadata'] = metadata
                            
                            # 处理同一sensor_id的多个文件
                            existing = row_sensor_data[sensor_id]
                            if existing is None:
                                row_sensor_data[sensor_id] = file_data
                            elif isinstance(existing, list):
                                existing.append(file_data)
                            else:
                                row_sensor_data[sensor_id] = [existing, file_data]
            
            # 添加当前行的sensor数据到结果
            for sensor_id in target_sensor_ids:
                results[sensor_id].append(row_sensor_data[sensor_id])
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results)
        
        # 设置timestamp为索引
        result_df.set_index('timestamp', inplace=True)
        
        return result_df

    def search_sensor_ids(self, 
                        target_sensor_ids: List[str],
                        start_time: Optional[Union[datetime, str, tuple, pd.Timestamp]] = None,
                        end_time: Optional[Union[datetime, str, pd.Timestamp]] = None,
                        include_metadata: bool = True,
                        max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        多线程搜索指定sensor_ids的文件路径
        
        Args:
            target_sensor_ids: 要搜索的sensor_id列表
            start_time: 开始时间，可以是:
                - datetime 对象
                - ISO 格式的字符串 (如 "2023-01-01 08:00:00")
                - (start_time, end_time) 元组
                - None 表示从最早时间开始
            end_time: 结束时间，格式同 start_time，None 表示到最晚时间结束
            include_metadata: 是否包含元数据
            max_workers: 最大工作线程数，None表示自动确定
        
        Returns:
            pd.DataFrame: 合并后的结果DataFrame，包含所有时间点和sensor_id的映射
        """
        # 处理元组输入
        if isinstance(start_time, tuple) and end_time is None:
            start_time, end_time = start_time
        
        # 转换时间为 pandas Timestamp
        if start_time is not None:
            if isinstance(start_time, str):
                start_time = pd.Timestamp(start_time)
            elif isinstance(start_time, datetime):
                start_time = pd.Timestamp(start_time)
        
        if end_time is not None:
            if isinstance(end_time, str):
                end_time = pd.Timestamp(end_time)
            elif isinstance(end_time, datetime):
                end_time = pd.Timestamp(end_time)
        
        # 筛选时间范围的DataFrame
        if start_time is None and end_time is None:
            # 使用全部数据
            time_range_df = self.df.copy()
        else:
            # 创建时间掩码
            mask = pd.Series(True, index=self.df.index)
            
            if start_time is not None:
                mask = mask & (self.df.index >= start_time)
            
            if end_time is not None:
                mask = mask & (self.df.index <= end_time)
            
            time_range_df = self.df[mask].copy()
        
        # 检查是否为空
        if time_range_df.empty:
            logging.info(f"在指定时间范围内没有找到数据: {start_time} 到 {end_time}")
            return pd.DataFrame(columns=['Index', 'timestamp', 'hour', 'day', 'month'] + target_sensor_ids)
        
        # 设置默认线程数
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        
        # 按时间顺序排序确保结果有序
        time_range_df = time_range_df.sort_index()
        
        # 将DataFrame按时间索引分割成块
        n_splits = min(max_workers, len(time_range_df))
        if n_splits == 0:
            return pd.DataFrame(columns=['Index', 'timestamp', 'hour', 'day', 'month'] + target_sensor_ids)
        
        df_blocks = np.array_split(time_range_df, n_splits)
        
        # 存储结果的列表
        result_blocks = [None] * n_splits
        block_lock = threading.Lock()
        
        # 处理单个块的包装函数
        def process_block(block, block_idx):
            try:
                block_result = self._search_sensor_ids_by_df_block(
                    df_block=block,
                    target_sensor_ids=target_sensor_ids,
                    include_metadata=include_metadata
                )
                with block_lock:
                    result_blocks[block_idx] = block_result
                return block_idx, True
            except Exception as e:
                logging.error(f"处理块 {block_idx} 时出错: {str(e)}")
                return block_idx, False
        
        # 创建进度跟踪
        total_blocks = len(df_blocks)
        completed_blocks = 0
        progress_lock = threading.Lock()
        
        # 进度显示函数
        def show_progress():
            nonlocal completed_blocks
            start_time = time.time()
            last_update = 0
            
            while completed_blocks < total_blocks:
                current_time = time.time()
                if current_time - last_update < 0.5:  # 每0.5秒更新一次
                    time.sleep(0.1)
                    continue
                    
                last_update = current_time
                progress = (completed_blocks / total_blocks) * 100
                elapsed = current_time - start_time
                estimated_total = elapsed / (completed_blocks / total_blocks) if completed_blocks > 0 else 0
                remaining = estimated_total - elapsed
                
                sys.stdout.write(
                    f"\r多线程处理进度: {progress:5.1f}% ({completed_blocks}/{total_blocks} 块) | "
                    f"预计剩余时间: {remaining:.1f}秒"
                )
                sys.stdout.flush()
                time.sleep(0.1)
        
        # 启动进度线程
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        # 使用线程池处理所有块
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_block, block, idx) 
                for idx, block in enumerate(df_blocks) if not block.empty
            ]
            
            for future in concurrent.futures.as_completed(futures):
                block_idx, success = future.result()
                with progress_lock:
                    completed_blocks += 1
        
        # 等待进度线程结束
        progress_thread.join(timeout=1.0)
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        # 合并所有结果块
        valid_blocks = [block for block in result_blocks if block is not None and not block.empty]
        
        if not valid_blocks:
            logging.warning("没有成功处理任何数据块")
            return pd.DataFrame(columns=['Index', 'timestamp', 'hour', 'day', 'month'] + target_sensor_ids)
        
        # 按时间索引排序合并
        final_result = pd.concat(valid_blocks, axis=0)
        final_result = final_result.sort_index()
        
        # 确保所有sensor_id列都存在
        for sensor_id in target_sensor_ids:
            if sensor_id not in final_result.columns:
                final_result[sensor_id] = None
        
        # 重置索引并添加行号
        final_result = final_result.reset_index()
        final_result.insert(0, 'Index', range(len(final_result)))
        
        return final_result

    def save_to_parquet(self, file_dir: str) -> None:
        """
        保存索引表到指定目录，生成index.parquet和config.yaml文件
        
        Args:
            file_dir: 保存的目录路径
        """
        # 确保目录存在
        Path(file_dir).mkdir(parents=True, exist_ok=True)
        
        # 定义文件路径
        index_path = Path(file_dir) / "index.parquet"
        config_path = Path(file_dir) / "config.yaml"
        
        # 将复杂对象转换为JSON字符串以便保存
        df_copy = self.df.copy()
        df_copy['data_types'] = df_copy['data_types'].apply(json.dumps)
        df_copy['file_groups'] = df_copy['file_groups'].apply(json.dumps)
        
        # 保存DataFrame
        df_copy.to_parquet(index_path)
        print(f"索引表已保存至: {index_path}")
        
        # 保存映射配置
        self.mapping_config.save_to_file(config_path)
        print(f"配置文件已保存至: {config_path}")
    
    @classmethod
    def load_from_parquet(cls, index_path: str, config_path: Optional[str] = None, 
                        load_mapping_config: bool = True) -> 'TimeSeriesFileIndex':
        """
        从Parquet文件加载索引表，可选加载映射配置
        
        Args:
            index_path: 索引Parquet文件的路径
            config_path: 映射配置文件的路径（可选）。如果为None且load_mapping_config为True，
                        将尝试从索引文件所在目录加载config.yaml
            load_mapping_config: 是否加载映射配置文件
        
        Returns:
            加载完成的TimeSeriesFileIndex实例
        """
        index_path = Path(index_path)
        
        # 检查索引文件是否存在
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        # 加载DataFrame
        df_loaded = pd.read_parquet(index_path)
        
        # 将JSON字符串转换回Python对象
        df_loaded['data_types'] = df_loaded['data_types'].apply(json.loads)
        df_loaded['file_groups'] = df_loaded['file_groups'].apply(json.loads)
        
        # 创建新实例
        instance = cls()
        
        # 设置DataFrame
        instance.df = df_loaded
        
        # 处理配置文件加载
        if load_mapping_config:
            # 确定配置文件路径
            if config_path is None:
                # 尝试使用默认配置文件名（与索引文件同目录）
                default_config_path = index_path.parent / "config.yaml"
                config_to_load = str(default_config_path)
            else:
                config_to_load = config_path
            
            config_path_obj = Path(config_to_load)
            
            if config_path_obj.exists():
                try:
                    instance.mapping_config = FileTypeMappingConfig(str(config_path_obj))
                    print(f"成功加载映射配置: {config_path_obj}")
                except Exception as e:
                    print(f"加载映射配置失败，使用默认配置: {str(e)}")
            else:
                # 尝试旧格式的配置文件（向后兼容）
                legacy_config_path = str(index_path).replace('.parquet', '_mapping.yaml')
                if os.path.exists(legacy_config_path):
                    try:
                        instance.mapping_config = FileTypeMappingConfig(legacy_config_path)
                        print(f"成功加载旧格式映射配置: {legacy_config_path}")
                    except Exception as e:
                        print(f"加载旧格式映射配置失败，使用默认配置: {str(e)}")
                else:
                    print(f"未找到映射配置文件 ({config_to_load})，使用默认配置")
        
        print(f"成功加载索引表，包含 {len(df_loaded)} 条记录")
        return instance
    
    def _extract_metadata_from_path(self, file_path: str) -> Dict[str, Any]:
        """
        从文件路径提取元数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            包含元数据的字典
        """
        path = Path(file_path)
        
        # 默认值
        current_year = datetime.now().year
        month = 1
        day = 1
        hour = 0
        minute = 0
        second = 0
        sensor_id = ""
        
        # 获取路径的所有部分
        parts = list(path.parent.parts)
        
        # 从文件夹结构提取月份和日期（倒数第二个文件夹是月，倒数第一个是日）
        if len(parts) >= 2:
            try:
                # 倒数第一个文件夹是日
                day_str = parts[-1]
                # 倒数第二个文件夹是月
                month_str = parts[-2]
                
                # 处理可能的前导零
                month = int(month_str.lstrip('0') or '1')
                day = int(day_str.lstrip('0') or '1')
                
                # 验证月份和日期范围
                if not 1 <= month <= 12:
                    month = 1
                if not 1 <= day <= 31:  # 简单验证，更精确的验证在datetime构造时处理
                    day = 1
            except (ValueError, TypeError, IndexError):
                # 如果解析失败，使用默认值
                month = 1
                day = 1
        
        # 从文件名提取时间信息
        filename = path.stem
        
        # 尝试匹配文件名中的时间模式：_后面跟6位数字（HHMMSS）
        time_match = re.search(r'_(\d{6})$', filename)
        if time_match:
            time_str = time_match.group(1)
            try:
                hour = int(time_str[0:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                
                # 传感器ID是文件名去掉时间部分（包括前面的下划线）
                sensor_id = filename[:time_match.start()]
            except (ValueError, IndexError):
                sensor_id = filename
        else:
            # 如果没有匹配到时间格式，尝试其他常见模式
            alt_time_match = re.search(r'(\d{2})[hH]', filename)  # 尝试匹配如"05h"格式
            if alt_time_match:
                try:
                    hour = int(alt_time_match.group(1))
                    sensor_id = re.sub(r'(\d{2})[hH]', '', filename)
                except ValueError:
                    sensor_id = filename
            else:
                sensor_id = filename
        
        # 从传感器ID中提取传感器类型（如果需要）
        sensor_type = ""
        sensor_match = re.search(r'([A-Z]{2}-([A-Z0-9]+)-[A-Z0-9]+-[0-9]+-[0-9]+)', sensor_id)
        if sensor_match:
            full_id = sensor_match.group(1)
            type_code = sensor_match.group(2)
            if type_code:
                sensor_type = type_code
            # 确保sensor_id是完整的ID
            sensor_id = full_id
        
        # 构建时间戳，处理无效日期
        try:
            timestamp = datetime(current_year, month, day, hour, minute, second)
        except ValueError as e:
            # 记录错误但继续处理
            print(f"警告: 无效日期 {current_year}-{month}-{day} {hour}:{minute}:{second} - {e}")
            # 尝试使用最接近的有效日期
            try:
                # 尝试使用当月的第一天
                timestamp = datetime(current_year, month, 1, hour, minute, second)
            except ValueError:
                # 最后手段：使用1月1日
                timestamp = datetime(current_year, 1, 1, hour, minute, second)
        
        return {
            'timestamp': timestamp,
            'file_extension': path.suffix.lower(),
            'filename': filename,
            'sensor_id': sensor_id,
            'sensor_type': sensor_type,
            'full_path': str(path),
            'parent_dirs': list(path.parent.parts),
            'month': month,
            'day': day,
            'hour': hour
        }
    
    def _calculate_hours_per_type(self) -> Dict[str, int]:
        """计算每种数据类型覆盖的小时数"""
        hours_per_type = {}
        for _, row in self.df.iterrows():
            for data_type in row['data_types']:
                if data_type not in hours_per_type:
                    hours_per_type[data_type] = 0
                hours_per_type[data_type] += 1
        return hours_per_type
    
    def get_mapping_config(self) -> FileTypeMappingConfig:
        """获取当前使用的映射配置"""
        return self.mapping_config
    
    def set_mapping_config(self, config: FileTypeMappingConfig) -> None:
        """设置新的映射配置"""
        if not isinstance(config, FileTypeMappingConfig):
            raise ValueError("config必须是FileTypeMappingConfig实例")
        self.mapping_config = config

    def __len__(self):
        """返回索引中的小时数"""
        return len(self.df)
    
    def __repr__(self):
        """返回对象的字符串表示"""
        data_coverage = self.get_data_coverage()
        return (f"TimeSeriesFileIndex(hours={len(self.df)}, "
                f"files={data_coverage['total_files']}, "
                f"data_types={len(data_coverage['data_types'])})")

    def __str__(self, max_rows: int = 10) -> str:
        """
        返回索引表的摘要信息字符串，便于通过print()快速了解数据情况
        
        Args:
            max_rows: 最大显示行数
        
        Returns:
            格式化的字符串表示
        """
        if self.df.empty:
            return "索引表为空"
        
        # 获取数据覆盖情况
        coverage = self.get_data_coverage()
        
        # 构建输出字符串
        output_lines = []
        output_lines.append("\n" + "="*60)
        output_lines.append(f"时序文件索引表摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        output_lines.append("="*60)
        output_lines.append(f"总小时数: {len(self.df):,}")
        output_lines.append(f"总文件数: {coverage['total_files']:,}")
        output_lines.append(f"数据类型数: {len(coverage['data_types'])}")
        
        time_coverage = coverage['time_coverage']
        if time_coverage['start'] and time_coverage['end']:
            output_lines.append(f"时间范围: {time_coverage['start']} 至 {time_coverage['end']}")
        
        # 数据类型分布
        output_lines.append("\n" + "-"*60)
        output_lines.append("数据类型分布:")
        output_lines.append("-"*60)
        
        # 按文件数量排序
        sorted_types = sorted(coverage['type_counts'].items(), key=lambda x: x[1], reverse=True)
        for data_type, count in sorted_types:
            hours = coverage['hours_per_type'].get(data_type, 0)
            percentage = (hours / len(self.df)) * 100 if len(self.df) > 0 else 0
            output_lines.append(f"{data_type:15} | 文件数: {count:5,} | 覆盖小时: {hours:4} ({percentage:5.1f}%)")
        
        # 显示数据样本
        output_lines.append("\n" + "-"*60)
        output_lines.append(f"数据摘要 (最多显示 {max_rows} 行):")
        output_lines.append("-"*60)
        output_lines.append(f"{'时间戳':<20} | {'月':<2} | {'日':<2} | {'时':<2} | {'数据类型数':<6} | {'文件总数'}")
        output_lines.append("-"*60)
        
        # 确定要显示的行
        show_ellipsis = False
        if len(self.df) <= max_rows:
            display_df = self.df
        else:
            # 取前半部分和后半部分，确保省略号有位置
            half_rows = max(1, (max_rows - 1) // 2)  # 至少显示1行，为省略号留出空间
            first_part = self.df.head(half_rows)
            last_part = self.df.tail(half_rows)
            display_df = pd.concat([first_part, last_part])
            show_ellipsis = True
        
        # 添加数据行
        row_count = 0
        total_display_rows = len(display_df)
        for timestamp, row in display_df.iterrows():
            total_files = sum(len(files) for files in row['file_groups'].values())
            output_lines.append(
                f"{timestamp.strftime('%Y-%m-%d %H:%M'):<20} | "
                f"{row['month']:<2} | {row['day']:<2} | {row['hour']:<2} | "
                f"{len(row['data_types']):<6} | {total_files}"
            )
            row_count += 1
            
            # 在前半部分结束后插入省略号（如果有）
            if show_ellipsis and row_count == half_rows and row_count < total_display_rows:
                output_lines.append(" " * 25 + "...")
        
        # 映射配置摘要
        output_lines.append("\n" + "-"*60)
        output_lines.append("映射配置:")
        output_lines.append("-"*60)
        output_lines.append(f"映射策略优先级: {self.mapping_config.config['mapping_priority']}")
        output_lines.append(f"默认数据类型: {self.mapping_config.config['default_data_type']}")
        output_lines.append(f"扩展名映射数: {len(self.mapping_config.config['extension_mapping'])}")
        output_lines.append(f"传感器ID映射数: {len(self.mapping_config.config['sensor_id_mapping'])}")
        
        output_lines.append("="*60)
        
        return "\n".join(output_lines)

# 示例YAML配置文件内容示例 (mapping_config.yaml)
"""
# 文件类型映射配置示例

extension_mapping:
  .uan: wind_speed
  .tmp: temperature
  .wnd: wind_speed
  .acc: acceleration
  .hmd: humidity

sensor_id_mapping:
  ST-WND: wind_speed
  ST-TMP: temperature
  ST-HMD: humidity
  ST-ACC: acceleration

sensor_type_mapping:
  WND: wind_speed
  TMP: temperature
  HMD: humidity
  ACC: acceleration

filename_pattern_mapping:
  - pattern: 'wind.*sensor'
    data_type: wind_speed
  - pattern: 'temp.*sensor'
    data_type: temperature
  - pattern: 'humidity.*sensor'
    data_type: humidity
  - pattern: 'accel.*sensor'
    data_type: acceleration

default_data_type: unknown

mapping_priority:
  - extension
  - sensor_id
  - filename_pattern
  - default
"""
