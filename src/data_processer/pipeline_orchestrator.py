import numpy as np
import os
import struct
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Union, Tuple, Optional, Any, Callable
from collections import defaultdict
import concurrent.futures
import threading
import math
import yaml
import json
import gc


from .persistence_utils import save_dataframe_to_parquet


class TimeSeriesDataSegmenter:
    """
    时序数据切分处理器，基于DataFrame接口，所有内部方法设为私有
    
    特性:
    1. 通过DataFrame接口接收数据，完全解耦
    2. process_dataframe作为唯一公开处理接口
    3. 支持自定义统计量计算函数
    4. 提供对齐时间表生成功能
    5. 所有内部处理方法以"_"开头，保持封装性
    """
    
    def __init__(self):
        """
        初始化数据切分处理器
        """
        # 注册默认文件解析函数
        self._parsers = {
            'VIC': self._parse_vic_file,
            'UAN': self._parse_wind_file,
            'wind_speed': self._parse_wind_file,
            'acceleration': self._parse_vic_file,
            'vibration': self._parse_vic_file
        }

        # 添加默认统计函数
        self._statistical_functions = {
            'mean': np.mean,
            'std': np.std,
            'max': np.max,
            'min': np.min,
            'median': np.median,
            'rms': lambda x: np.sqrt(np.mean(np.square(x))),
            'peak_to_peak': lambda x: np.max(x) - np.min(x)
        }

    def register_parser(self, key: str, parser_func: Callable[[str], Any]) -> None:
        """
        注册自定义文件解析函数
        
        Args:
            key: 解析器标识符（文件扩展名或数据类型）
            parser_func: 解析函数，接收文件路径，返回解析后的数据
        """
        self._parsers[key] = parser_func

    def register_statistical_function(self, name: str, func: Callable) -> None:
        """
        注册自定义统计函数
        
        Args:
            name: 统计函数标识名
            func: 统计函数，接收一维数组，返回统计值
        """
        self._statistical_functions[name] = func
        
    def _get_file_parser(self, file_path: str, data_type: Optional[str] = None) -> Callable:
        """
        获取适当的文件解析函数
        
        Args:
            file_path: 文件路径
            data_type: 可选的数据类型
            
        Returns:
            适当的解析函数
        """
        # 优先使用数据类型匹配
        if data_type and data_type in self._parsers:
            return self._parsers[data_type]
        
        # 其次使用文件扩展名匹配
        ext = os.path.splitext(file_path)[1][1:].upper()
        if ext in self._parsers:
            return self._parsers[ext]
        
        # 默认使用VIC解析器
        return self._parsers['VIC']
    
    def _parse_vic_file(self, file_path: str) -> np.ndarray:
        """
        解析VIC振动数据文件
        
        Args:
            file_path: VIC文件路径
            
        Returns:
            解析后的浮点数数组
        """
        split_str = '_'
        floats = np.array([])
        
        try:
            with open(file_path, "rb") as f:
                data = f.read()
                filename = os.path.basename(file_path)
                if split_str:
                    string = filename.split(split_str)[0].encode('utf-8')
                else:
                    string = filename.encode('utf-8')
                
                try:
                    idx_in_data = data.index(string) + len(string) + 1
                    num_floats = (len(data) - idx_in_data) // 4
                    if num_floats > 0:
                        floats = np.array(struct.unpack("f" * num_floats, data[idx_in_data:idx_in_data + num_floats * 4]))
                except ValueError:
                    # 从文件末尾读取1小时的数据作为后备方案
                    sample_rate = 50  # Hz
                    hour_samples = sample_rate * 3600
                    bytes_needed = hour_samples * 4
                    
                    if len(data) >= bytes_needed:
                        data_section = data[-bytes_needed:]
                        floats = np.array(struct.unpack("f" * hour_samples, data_section))
                    else:
                        num_floats = len(data) // 4
                        floats = np.array(struct.unpack("f" * num_floats, data[:num_floats * 4]))
        except Exception as e:
            print(f"解析VIC文件时出错 {file_path}: {str(e)}")
        
        return floats
    
    def _parse_wind_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        解析风数据文件
        
        Args:
            file_path: 风数据文件路径
            
        Returns:
            包含风速、风向和风攻角的字典
        """
        speed, direction, angle = [], [], []
        
        try:
            with open(file_path, "rb") as f:
                f_content = f.read()
                try:
                    content = f_content.decode("GB18030")
                except:
                    content = str(f_content)
                
                matches = re.findall(r"\d+\.\d*\,\d+\.\d*\,\d+\.\d*", content)
                
                for match in matches:
                    parts = match.split(",")
                    if len(parts) >= 3:
                        try:
                            speed.append(float(parts[0]))
                            direction.append(float(parts[1]))
                            angle.append(round(float(parts[2]) - 60.0, 1))
                        except ValueError:
                            continue
            
            return {
                'speed': np.array(speed),
                'direction': np.array(direction),
                'angle': np.array(angle)
            }
        except Exception as e:
            print(f"解析风数据文件时出错 {file_path}: {str(e)}")
            return {
                'speed': np.array([]),
                'direction': np.array([]),
                'angle': np.array([])
            }
    
    def _process_file(self, file_path: str, data_type: Optional[str] = None) -> Any:
        """
        统一文件处理接口
        
        根据文件路径和/或指定的数据类型自动选择适当的解析器，
        并返回解析后的数据。
        
        Args:
            file_path: 要处理的文件路径
            data_type: 可选的数据类型标识符，用于覆盖自动检测
            
        Returns:
            解析后的数据，类型取决于具体的解析器:
            - VIC文件: numpy数组，包含振动数据
            - 风速文件: 字典，包含'speed', 'direction', 'angle'三个键
            
        Raises:
            FileNotFoundError: 如果指定的文件不存在
            ValueError: 如果无法确定合适的解析器
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        parser_func = self._get_file_parser(file_path, data_type)
        return parser_func(file_path)
    
    def _segment_by_time_window(self, 
                                data: Union[List, np.ndarray], 
                                window_size_sec: float, 
                                step_size_sec: float, 
                                sampling_rate_hz: float) -> List[List]:
        """
        按时间窗口进行滑动切分的私有接口
        
        将时序数据按照指定的时间窗口大小和步长进行切分，返回切分后的数据段列表。
        
        Args:
            data: 输入的时序数据序列，支持列表或numpy数组
            window_size_sec: 窗口大小（秒）
            step_size_sec: 滑动步长（秒）
            sampling_rate_hz: 数据采样率（Hz，每秒样本数）
            
        Returns:
            List[List]: 切分后的数据，每个子列表代表一个时间窗口内的数据
            
        Raises:
            ValueError: 参数无效时抛出
            TypeError: 数据类型不支持时抛出
        """
        # 验证输入数据类型
        if not isinstance(data, (list, np.ndarray)):
            try:
                data = list(data)
            except Exception as e:
                raise TypeError(f"不支持的数据类型，无法转换为列表: {str(e)}")
        
        # 验证参数
        if sampling_rate_hz <= 0:
            raise ValueError("采样率必须大于0")
        if window_size_sec <= 0:
            raise ValueError("窗口大小必须大于0")
        if step_size_sec <= 0:
            raise ValueError("步长必须大于0")
        
        # 将时间参数转换为样本数
        window_samples = int(round(window_size_sec * sampling_rate_hz))
        step_samples = int(round(step_size_sec * sampling_rate_hz))
        
        # 验证转换后的样本数
        if window_samples <= 0:
            raise ValueError("窗口大小转换为样本数后必须大于0")
        if step_samples <= 0:
            raise ValueError("步长转换为样本数后必须大于0")
        
        # 验证数据长度是否足够
        data_length = len(data)
        if data_length < window_samples:
            raise ValueError(f"数据长度({data_length})小于窗口大小({window_samples}个样本)")
        
        # 确保数据是列表格式
        if isinstance(data, np.ndarray):
            data_list = data.tolist()
        else:
            data_list = data
        
        # 执行滑动窗口切分
        segments = []
        start_idx = 0
        
        while start_idx + window_samples <= data_length:
            end_idx = start_idx + window_samples
            segment = data_list[start_idx:end_idx]
            segments.append(segment)
            start_idx += step_samples
        
        # 处理边界情况：如果最后剩余的数据足够接近窗口大小，也作为一个段
        remaining_samples = data_length - start_idx
        if remaining_samples > 0 and remaining_samples >= window_samples * 0.5:  # 至少有窗口一半的大小
            segment = data_list[start_idx:]
            segments.append(segment)
        
        return segments

    def _select_best_file_from_list(self, file_list: List[Dict], target_time: datetime) -> Optional[Dict]:
        """
        从文件列表中选择最佳文件，优先选择分秒为00:00的文件
        
        Args:
            file_list: 包含文件信息的字典列表，每个字典应有'file_path'键
            target_time: 目标时间戳，用于匹配最接近的文件
        
        Returns:
            Optional[Dict]: 最佳文件信息字典，如果找不到则返回None
        """
        # 优先收集整点文件（分秒为00:00）
        zero_minute_files = []
        # 其次收集其他有效文件
        valid_files = []
        
        for file_info in file_list:
            file_path = file_info['file_path']
            filename = os.path.basename(file_path)
            
            # 专门处理示例中的文件名格式（如ST-UAN-G02-002-01_090000.UAN）
            # 1. 尝试匹配文件名末尾的时间戳（在最后一个下划线和扩展名之间）
            time_match = re.search(r'_(\d{6})\.[^.]+$', filename)
            
            # 2. 如果没有匹配到，尝试匹配更通用的格式
            if not time_match:
                time_match = re.search(r'(\d{6})\.[^.]+$', filename)
            
            # 3. 如果仍未匹配，尝试匹配完整时间戳（14位）
            if not time_match:
                time_match = re.search(r'(\d{14})', filename)
            
            # 4. 最后尝试匹配8位日期+6位时间的组合
            if not time_match:
                time_match = re.search(r'(\d{8}[-_]?\d{6})', filename)
            
            if time_match:
                time_str = time_match.group(1)
                # 处理不同格式的时间字符串
                if len(time_str) == 6:  # 仅时间部分 (HHMMSS)
                    # 使用目标时间的日期部分
                    date_part = target_time.strftime('%Y%m%d')
                    full_time_str = date_part + time_str
                    time_format = '%Y%m%d%H%M%S'
                elif len(time_str) == 8:  # 仅日期部分 (YYYYMMDD)
                    # 使用目标时间的时分秒部分
                    time_part = target_time.strftime('%H%M%S')
                    full_time_str = time_str + time_part
                    time_format = '%Y%m%d%H%M%S'
                elif len(time_str) == 14:  # 完整时间戳 (YYYYMMDDHHMMSS)
                    full_time_str = time_str
                    time_format = '%Y%m%d%H%M%S'
                else:  # 处理带分隔符的格式 (如20240909_090000)
                    # 移除分隔符
                    clean_str = re.sub(r'[-_]', '', time_str)
                    if len(clean_str) == 14:
                        full_time_str = clean_str
                        time_format = '%Y%m%d%H%M%S'
                    else:
                        # 无法处理的格式，跳过
                        continue
                
                try:
                    file_time = datetime.strptime(full_time_str, time_format)
                    # 检查是否为整点文件（分钟和秒为0）
                    if file_time.minute == 0 and file_time.second == 0:
                        zero_minute_files.append((file_info, file_time))
                    else:
                        valid_files.append((file_info, file_time))
                except (ValueError, TypeError):
                    # 无法解析时间，跳过此文件
                    continue
            else:
                # 无法匹配时间格式，跳过
                continue
        
        # 优先选择整点文件
        if zero_minute_files:
            # 选择最接近目标时间的整点文件
            best_match = min(zero_minute_files, key=lambda x: abs((x[1] - target_time).total_seconds()))
            return best_match[0]
        
        # 如果没有整点文件，选择最接近目标时间的有效文件
        if valid_files:
            best_match = min(valid_files, key=lambda x: abs((x[1] - target_time).total_seconds()))
            return best_match[0]
        
        # 最终后备：返回列表中的第一个文件（即使无法解析时间）
        if file_list:
            return file_list[0]
        
        return None


    def _process_sensor_column(self, df: pd.DataFrame, sensor_col: str, data_type: Optional[str]) -> List[Dict]:
        """
        处理单个传感器列的所有行（线程安全）
        
        Args:
            df: 输入DataFrame
            sensor_col: 传感器列名
            data_type: 传感器数据类型
        
        Returns:
            List[Dict]: 该列所有行的处理结果列表
        """
        results = []
        for idx, row in df.iterrows():
            try:
                processed_result = self._process_single_row_for_sensor(
                    row=row,
                    sensor_col=sensor_col,
                    data_type=data_type,
                    row_idx=idx
                )
                results.append(processed_result)
            except Exception as e:
                error_info = {
                    'error': f"行处理失败: {str(e)}",
                    'timestamp': row['timestamp'],
                    'sensor_id': sensor_col,
                    'row_idx': idx,
                    'exception_type': type(e).__name__
                }
                results.append(error_info)
                print(f"线程 {threading.current_thread().name} 处理行 {idx} 传感器 {sensor_col} 时出错: {str(e)}")
        return results

    def _process_single_row_for_sensor(self, row: pd.Series, sensor_col: str, 
                                    data_type: Optional[str], row_idx: int) -> Dict:
        """
        处理单行数据中的单个传感器列
        
        Args:
            row: DataFrame的当前行
            sensor_col: 要处理的传感器列名
            data_type: 传感器数据类型（如'UAN'或'VIC'）
            row_idx: 当前行索引（用于错误报告）
        
        Returns:
            Dict: 处理结果字典（包含数据或错误信息）
        """
        sensor_value = row[sensor_col]
        
        # 处理空值情况
        if sensor_value is None:
            return {
                'error': '传感器值为空',
                'timestamp': row['timestamp'],
                'sensor_id': sensor_col,
                'row_idx': row_idx,
                'raw_value': None
            }
        
        # 处理列表类型的传感器数据（包含多个文件路径）
        if isinstance(sensor_value, list) and len(sensor_value) > 0:
            # 筛选最佳文件：优先选择分秒为00:00的文件
            best_file_info = self._select_best_file_from_list(sensor_value, row['timestamp'])
            if best_file_info is None:
                return {
                    'error': '无法从文件列表中选择有效文件',
                    'timestamp': row['timestamp'],
                    'sensor_id': sensor_col,
                    'row_idx': row_idx,
                    'raw_value': str(sensor_value)[:100] + '...' if len(str(sensor_value)) > 100 else str(sensor_value)
                }
            file_path = best_file_info['file_path']
            original_files = sensor_value
        # 处理字典类型的传感器数据
        elif isinstance(sensor_value, dict) and 'file_path' in sensor_value:
            file_path = sensor_value['file_path']
            original_files = [sensor_value]
        # 处理字符串类型的文件路径
        elif isinstance(sensor_value, str):
            file_path = sensor_value
            original_files = [{'file_path': sensor_value}]
        else:
            # 无法识别的数据类型
            return {
                'error': f'不支持的传感器数据类型: {type(sensor_value).__name__}',
                'timestamp': row['timestamp'],
                'sensor_id': sensor_col,
                'row_idx': row_idx,
                'raw_value': str(sensor_value)[:100] + '...' if len(str(sensor_value)) > 100 else str(sensor_value)
            }
        
        try:
            # 处理文件
            processed_data = self._process_file(file_path, data_type)
            
            # 为结果添加时间戳和元数据
            timestamp = row['timestamp']
            if isinstance(processed_data, dict):
                # 风速等数据类型，添加时间戳到字典
                processed_data_with_time = processed_data.copy()
                processed_data_with_time.update({
                    'timestamp': timestamp,
                    'file_path': file_path,
                    'original_files': original_files,
                    'sensor_id': sensor_col,
                    'row_idx': row_idx,
                    'data_type': data_type
                })
            else:
                # 振动等数据类型，创建包含时间戳的结构
                processed_data_with_time = {
                    'data': processed_data,
                    'timestamp': timestamp,
                    'file_path': file_path,
                    'original_files': original_files,
                    'sensor_id': sensor_col,
                    'row_idx': row_idx,
                    'data_type': data_type
                }
            
            return processed_data_with_time
        
        except Exception as e:
            # 捕获文件处理异常
            return {
                'error': str(e),
                'timestamp': row['timestamp'],
                'sensor_id': sensor_col,
                'row_idx': row_idx,
                'file_path': file_path,
                'data_type': data_type,
                'raw_value': str(sensor_value)[:100] + '...' if len(str(sensor_value)) > 100 else str(sensor_value)
            }

    def _get_sensor_data_type(self, sensor_col: str, data_type_map: Optional[Dict[str, str]]) -> Optional[str]:
        """
        获取传感器的数据类型
        
        Args:
            sensor_col: 传感器列名
            data_type_map: 传感器类型映射字典
        
        Returns:
            str: 传感器数据类型，如'UAN'或'VIC'
        """
        # 从映射字典中获取
        if data_type_map and sensor_col in data_type_map:
            return data_type_map[sensor_col]
        
        # 从列名推断
        col_lower = sensor_col.lower()
        if 'uan' in col_lower or 'wind' in col_lower:
            return 'UAN'
        elif 'vic' in col_lower or 'vibration' in col_lower or 'acceleration' in col_lower:
            return 'VIC'
        
        return None


    def _process_file_dataframe(self, df: pd.DataFrame, sensor_columns: Optional[List[str]] = None, 
                        data_type_map: Optional[Dict[str, str]] = None, 
                        max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        多线程处理DataFrame中的时序数据，直接替换原始传感器列，并按时间戳排序
        
        Args:
            df: 输入的DataFrame，必须包含timestamp列和传感器数据列
            sensor_columns: 要处理的传感器列名列表，如果为None则自动检测所有非时间相关列
            data_type_map: 传感器类型映射字典，键为传感器列名，值为数据类型(如'VIC','UAN'等)
            max_workers: 最大线程数，None表示使用CPU核心数*2
        
        Returns:
            pd.DataFrame: 处理后的DataFrame，原始传感器列被替换为处理结果，按时间戳排序
        """
        # 创建结果DataFrame的深拷贝，避免修改原始数据
        result_df = df.copy(deep=True)
        
        # 自动检测传感器列（排除时间相关列）
        if sensor_columns is None:
            time_columns = ['timestamp', 'hour', 'day', 'month', 'Index']
            sensor_columns = [col for col in df.columns if col not in time_columns and not col.endswith('_processed')]
        
        # 确保timestamp列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        
        # 设置默认线程数
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 2)  # 限制最大32线程
        
        # 准备传感器列处理任务
        sensor_tasks = []
        for sensor_col in sensor_columns:
            data_type = self._get_sensor_data_type(sensor_col, data_type_map)
            sensor_tasks.append((sensor_col, data_type))
        
        # 使用线程池并行处理传感器列
        processed_columns = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有传感器列处理任务
            future_to_sensor = {
                executor.submit(
                    self._process_sensor_column, 
                    result_df, 
                    sensor_col, 
                    data_type
                ): sensor_col 
                for sensor_col, data_type in sensor_tasks
            }
            
            # 收集处理结果
            for future in concurrent.futures.as_completed(future_to_sensor):
                sensor_col = future_to_sensor[future]
                try:
                    processed_column = future.result()
                    processed_columns[sensor_col] = processed_column
                except Exception as e:
                    print(f"传感器列 {sensor_col} 处理失败: {str(e)}")
                    # 创建错误列
                    error_col = [{
                        'error': f"列处理失败: {str(e)}",
                        'sensor_id': sensor_col,
                        'row_idx': idx
                    } for idx in range(len(result_df))]
                    processed_columns[sensor_col] = error_col
        
        # 将处理结果应用到DataFrame
        for sensor_col, processed_column in processed_columns.items():
            result_df[sensor_col] = processed_column
        
        # 按时间戳排序
        result_df = result_df.sort_values(by='timestamp').reset_index(drop=True)
        
        return result_df
    
    def _split_sensor_columns(self, df, components_to_extract=None, exclude_keys=None, extract_errors=False):
        """
        高性能传感器字典列拆分方法
        允许用户指定要提取的组件类型和排除的键
        
        参数:
        df : pandas DataFrame
            包含传感器字典列的DataFrame
        components_to_extract : list or dict, optional
            要提取的组件名称:
            - None: 自动提取所有非排除组件
            - list: 为所有传感器提取指定组件 (e.g., ['speed', 'direction'])
            - dict: 按传感器指定组件 (e.g., {'ST-UAN-G02-002-01': ['speed'], 'ST-VIC-C08-G06-01': ['data']})
        exclude_keys : list or dict, optional
            要排除的键:
            - None: 只使用默认排除列表
            - list: 全局排除列表，对所有传感器生效 (e.g., ['timestamp', 'unit'])
            - dict: 按传感器指定排除键 (e.g., {'ST-UAN-G02-002-01': ['metadata'], 'ST-VIC-C08-G06-01': ['unit']})
        extract_errors : bool, default=False
            是否提取error信息到单独列
            
        返回:
        pandas DataFrame: 拆分后的新DataFrame
        """
        import pandas as pd
        import numpy as np
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 定义基础列（非传感器列）
        base_columns = ['Index', 'timestamp', 'month', 'day', 'hour']
        base_columns = [col for col in base_columns if col in df.columns]
        sensor_columns = [col for col in df.columns if col not in base_columns]
        
        if not sensor_columns:
            return df.copy()
        
        result_df = df[base_columns].copy()
        new_columns = {}
        
        # 处理排除键参数
        default_exclude = ['error', 'timestamp', 'metadata', 'unit', 'name', 'id', 'sensor_id']
        
        # 构建完整的排除键映射
        exclude_map = {}
        for sensor_col in sensor_columns:
            if exclude_keys is None:
                exclude_map[sensor_col] = set(default_exclude)
            elif isinstance(exclude_keys, list):
                # 全局排除列表
                exclude_map[sensor_col] = set(default_exclude + exclude_keys)
            elif isinstance(exclude_keys, dict):
                # 传感器特定排除
                sensor_exclude = exclude_keys.get(sensor_col, [])
                exclude_map[sensor_col] = set(default_exclude + sensor_exclude)
            else:
                raise ValueError("exclude_keys must be None, list, or dict")
        
        # 标准化components_to_extract参数
        if components_to_extract is None:
            # 自动模式：收集所有非排除组件
            auto_components = {}
            for sensor_col in sensor_columns:
                auto_components[sensor_col] = set()
                current_exclude = exclude_map[sensor_col]
                
                # 逐行检查传感器数据
                for idx, cell_value in enumerate(df[sensor_col]):
                    # 检查是否为有效字典
                    if isinstance(cell_value, dict):
                        # 跳过包含error的字典（除非用户明确要求提取error）
                        if 'error' in cell_value and not extract_errors:
                            continue
                        
                        # 收集所有有效组件键
                        for key, value in cell_value.items():
                            # 跳过排除的键
                            if key in current_exclude:
                                continue
                            
                            # 确保值是有效数据（列表或基础类型）
                            valid_types = (list, np.ndarray, int, float, str, bool)
                            if isinstance(value, valid_types):
                                auto_components[sensor_col].add(key)
                            # 处理嵌套字典（展平为 sensor_key 形式）
                            elif isinstance(value, dict):
                                for nested_key in value.keys():
                                    # 跳过嵌套的排除键
                                    if nested_key in current_exclude:
                                        continue
                                    auto_components[sensor_col].add(f"{key}_{nested_key}")
                
                # 如果没有找到任何组件，使用默认组件
                if not auto_components[sensor_col]:
                    auto_components[sensor_col] = {'data'}  # 默认组件名
                
            components_map = auto_components
        elif isinstance(components_to_extract, list):
            # 统一模式：所有传感器使用相同组件
            components_map = {}
            for sensor_col in sensor_columns:
                current_exclude = exclude_map[sensor_col]
                # 过滤掉被排除的组件
                filtered_components = [comp for comp in components_to_extract if comp not in current_exclude]
                
                # 验证组件是否存在
                valid_components = []
                for component in filtered_components:
                    has_component = False
                    for cell_value in df[sensor_col]:
                        if isinstance(cell_value, dict) and component in cell_value:
                            if 'error' not in cell_value or extract_errors:
                                has_component = True
                                break
                    if has_component:
                        valid_components.append(component)
                
                components_map[sensor_col] = valid_components if valid_components else ['data']
        elif isinstance(components_to_extract, dict):
            # 自定义模式：每个传感器指定组件
            components_map = {}
            for sensor_col in sensor_columns:
                current_exclude = exclude_map[sensor_col]
                if sensor_col in components_to_extract:
                    # 过滤掉被排除的组件
                    requested_components = [comp for comp in components_to_extract[sensor_col] 
                                        if comp not in current_exclude]
                    
                    # 验证组件是否存在
                    valid_components = []
                    for component in requested_components:
                        has_component = False
                        for cell_value in df[sensor_col]:
                            if isinstance(cell_value, dict) and component in cell_value:
                                if 'error' not in cell_value or extract_errors:
                                    has_component = True
                                    break
                        if has_component:
                            valid_components.append(component)
                    
                    components_map[sensor_col] = valid_components if valid_components else ['data']
                else:
                    components_map[sensor_col] = []  # 该传感器不提取任何组件
        else:
            raise ValueError("components_to_extract must be None, list, or dict")
        
        # 并行处理函数
        def process_sensor_column(sensor_col, components):
            """处理单个传感器列，可并行执行"""
            sensor_data = df[sensor_col]
            comp_data = {}
            current_exclude = exclude_map[sensor_col]
            
            # 提取指定组件
            for component in components:
                # 跳过被排除的组件
                if component in current_exclude:
                    continue
                    
                new_col_name = f"{sensor_col}-{component}"
                # 向量化提取，只处理有效数据
                comp_data[new_col_name] = sensor_data.apply(
                    lambda x: x.get(component) 
                    if isinstance(x, dict) and 
                    (component in x) and 
                    ('error' not in x or extract_errors)
                    else None
                )
            
            # 按需提取error信息（但受排除规则限制）
            if extract_errors and 'error' not in current_exclude:
                error_col_name = f"{sensor_col}-error"
                comp_data[error_col_name] = sensor_data.apply(
                    lambda x: x.get('error') if isinstance(x, dict) and 'error' in x else None
                )
            
            return comp_data
        
        # 并行处理所有传感器列
        max_workers = min(len(sensor_columns), max(1, os.cpu_count() or 2))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sensor = {
                executor.submit(process_sensor_column, sensor_col, components): sensor_col
                for sensor_col, components in components_map.items()
            }
            
            for future in as_completed(future_to_sensor):
                sensor_col = future_to_sensor[future]
                try:
                    comp_results = future.result()
                    for col_name, data in comp_results.items():
                        new_columns[col_name] = data
                except Exception as e:
                    print(f"Error processing sensor {sensor_col}: {str(e)}")
                    # 出错时创建空列
                    for component in components_map.get(sensor_col, ['data']):
                        if component not in exclude_map[sensor_col]:
                            col_name = f"{sensor_col}-{component}"
                            new_columns[col_name] = pd.Series([None] * len(df))
        
        # 批量添加所有新列
        for col_name, data in new_columns.items():
            result_df[col_name] = data
        
        return result_df
    
    def _segment_sensor_data_by_columns(self, 
                                        df: pd.DataFrame, interval_nums: int, 
                                        time_col: str = 'timestamp', 
                                        exclude_cols: Optional[List[str]] = None,
                                        n_threads: int = 4) -> pd.DataFrame:
        """
        将每小时传感器数据按指定粒度切分（多线程版本）
        
        参数:
        df -- 输入的DataFrame，包含每小时数据，传感器数据以列表形式存储
        interval_nums -- 要切分的间隔数量（例如60表示每小时切分为60个1分钟间隔，
                        20表示每小时切分为20个3分钟间隔）
        time_col -- 时间列的名称，默认为'timestamp'
        exclude_cols -- 用户指定的非传感器列列表，这些列将被排除在传感器数据处理之外
        n_threads -- 用于并行处理的线程数，默认为4
        
        返回:
        新的DataFrame，其中每行代表一个时间间隔的数据，保持原始列顺序
        """
        # 1. 保存原始列顺序
        original_columns = list(df.columns)
        
        # 2. 确定新列'minute'的插入位置（在timestamp之后）
        minute_col_pos = 1  # 默认在第2位（索引1）
        if time_col in original_columns:
            time_col_idx = original_columns.index(time_col)
            minute_col_pos = time_col_idx + 1
        
        # 3. 定义默认的非传感器关键词
        default_non_sensor_keywords = [
            'time', 'date', 'index', 'id', 'serial', 'num', 'count', 
            'label', 'type', 'name', 'status', 'flag', 'code', 'desc'
        ]
        
        # 4. 收集所有需要排除的列
        cols_to_exclude = set()
        
        # 添加用户指定的排除列
        if exclude_cols:
            cols_to_exclude.update(exclude_cols)
        
        # 添加默认排除的时间组件列
        time_components = ['month', 'day', 'hour', 'year', 'week', 'weekday', 'quarter', 'minute', 'second']
        cols_to_exclude.update(time_components)
        
        # 添加包含非传感器关键词的列
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in default_non_sensor_keywords):
                cols_to_exclude.add(col)
            if col_lower in time_components:
                cols_to_exclude.add(col)
        
        # 确保时间列被排除
        cols_to_exclude.add(time_col)
        
        # 5. 确定传感器数据列：所有不在排除列表中的列
        sensor_cols = [col for col in df.columns if col not in cols_to_exclude]
        
        if not sensor_cols:
            raise ValueError(
                f"未检测到传感器数据列。所有列都被视为非传感器列。"
                f"\n排除的列: {sorted(cols_to_exclude)}"
                f"\n所有列: {sorted(df.columns)}"
            )
        
        # 6. 准备多线程处理
        # 将DataFrame分成多个chunk
        n_rows = len(df)
        chunk_size = max(1, math.ceil(n_rows / (n_threads * 2)))  # 每个线程处理多个小chunk
        chunks = [(i, df.iloc[i:i+chunk_size]) for i in range(0, n_rows, chunk_size)]
        
        # 7. 定义处理单个chunk的函数
        def process_chunk(chunk_idx: int, chunk_df: pd.DataFrame) -> List[Dict[str, Any]]:
            """处理单个数据块，返回新行列表"""
            chunk_rows = []
            
            for _, row in chunk_df.iterrows():
                hour_time = row[time_col]
                
                for interval_idx in range(interval_nums):
                    new_row = {}
                    
                    # 创建新的时间戳
                    minutes_offset = (60 / interval_nums) * interval_idx
                    interval_start_time = hour_time + timedelta(minutes=minutes_offset)
                    
                    # 设置所有原始列
                    for col in original_columns:
                        if col == time_col:
                            new_row[col] = interval_start_time
                        elif col in sensor_cols:
                            sensor_data = row[col]
                            
                            # 处理非列表数据
                            if not isinstance(sensor_data, (list, np.ndarray, pd.Series)):
                                new_row[col] = sensor_data
                                continue
                            
                            # 转换为numpy数组以便分割
                            try:
                                data_array = np.array(sensor_data)
                                total_points = len(data_array)
                                
                                if total_points == 0:
                                    new_row[col] = []
                                    continue
                                
                                # 使用array_split进行均匀分割
                                split_arrays = np.array_split(data_array, interval_nums)
                                
                                if interval_idx < len(split_arrays):
                                    interval_data = split_arrays[interval_idx].tolist()
                                else:
                                    interval_data = []
                                
                                new_row[col] = interval_data
                            except Exception as e:
                                # 出错时保留原始数据
                                new_row[col] = sensor_data
                        else:
                            # 非传感器列和特殊列，直接复制
                            new_row[col] = row[col]
                    
                    # 添加minute列
                    new_row['minute'] = interval_start_time.minute
                    
                    chunk_rows.append(new_row)
            
            return chunk_rows
        
        # 8. 使用多线程处理所有chunk
        all_new_rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            # 提交所有任务
            future_to_chunk = {
                executor.submit(process_chunk, chunk_idx, chunk_df): chunk_idx
                for chunk_idx, chunk_df in chunks
            }
            
            # 按原始顺序收集结果
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_rows = future.result()
                    all_new_rows.extend(chunk_rows)
                except Exception as e:
                    print(f"处理chunk {chunk_idx}时出错: {str(e)}")
        
        # 9. 按原始时间顺序排序（因为多线程可能打乱顺序）
        all_new_rows.sort(key=lambda x: (x[time_col], x['minute']))
        
        # 10. 创建新的DataFrame
        new_df = pd.DataFrame(all_new_rows)
        
        # 11. 确保列顺序与原始DataFrame一致，并插入minute列
        new_columns = original_columns.copy()
        if 'minute' not in new_columns:
            new_columns.insert(new_columns.index("hour") + 1, 'minute')
        
        # 确保所有需要的列都存在
        final_columns = []
        for col in new_columns:
            if col in new_df.columns:
                final_columns.append(col)
        
        # 添加可能遗漏的列
        for col in new_df.columns:
            if col not in final_columns:
                final_columns.append(col)
        
        new_df = new_df[final_columns]
        
        # 12. 重置索引
        new_df = new_df.reset_index(drop=True)
        
        return new_df


    def process_df(self, df: pd.DataFrame, interval_nums = 60):
        """
        传入df表，获取细粒度的原始数据：
            文件解析        -> 原始传感器数据表
            列数据拆分      -> 获取展平后的数据表
            按照时间戳拆分  -> 细粒度的df原始数据表
        rparams: 
            df: pd.DataFrame，其内容对齐TimeSeriesFileIndex的搜索结果，包含索引、时间戳、时间、传感器文件及元数据等
                Index           timestamp  month  day  hour                                  ST-UAN-G02-002-01                                  ST-VIC-C08-G06-01
                0        0 2025-01-01 00:00:00      1    1     0                                               None                                               None
                1        1 2025-09-01 00:00:00      9    1     0  {'file_path': 'F:/Research/Vibration Character...  {'file_path': 'F:/Research/Vibration Character...
                ..     ...                 ...    ...  ...   ...                                                ...                                                ...
                719    719 2025-09-30 22:00:00      9   30    22  {'file_path': 'F:/Research/Vibration Character...  {'file_path': 'F:/Research/Vibration Character...
                720    720 2025-09-30 23:00:00      9   30    23  {'file_path': 'F:/Research/Vibration Character...  {'file_path': 'F:/Research/Vibration Character...
                [721 rows x 7 columns]

            interval_nums: int，切分粒度，1h文件数据会被切分为interval_nums份
            
        return:
            df: 切分好的原始数据表
                Index           timestamp  month  ...                        ST-UAN-G02-002-01-direction                             ST-VIC-C08-G06-01-data           minute
                0          0 2025-01-01 00:00:00      1  ...                                               None                                               None      0
                1          0 2025-01-01 00:01:00      1  ...                                               None                                               None      1
                2          0 2025-01-01 00:02:00      1  ...                                               None                                               None      2
                3          0 2025-01-01 00:03:00      1  ...                                               None                                               None      3
                4          0 2025-01-01 00:04:00      1  ...                                               None                                               None      4
                ...      ...                 ...    ...  ...                                                ...                                                ...    ...
                43255    720 2025-09-30 23:55:00      9  ...  [146.0, 145.6, 145.1, 145.3, 145.9, 146.3, 146...  [-0.06532228738069534, -0.2127375304698944, -0...     55
                43256    720 2025-09-30 23:56:00      9  ...  [145.4, 145.2, 145.3, 145.3, 145.6, 145.9, 146...  [0.004542321898043156, -0.07423175871372223, 0...     56
                43257    720 2025-09-30 23:57:00      9  ...  [146.3, 146.3, 146.3, 146.4, 146.4, 145.8, 145...  [-0.2574847936630249, -0.2277340292930603, 0.2...     57
                43258    720 2025-09-30 23:58:00      9  ...  [145.0, 144.3, 143.9, 143.4, 142.9, 142.7, 142...  [-0.5360044240951538, -0.17414997518062592, 0....     58
                43259    720 2025-09-30 23:59:00      9  ...  [144.1, 146.1, 147.0, 146.3, 145.8, 145.2, 144...  [0.12180100381374359, -0.06454096734523773, -0...     59
                [43260 rows x 10 columns]

        """
        result = self._process_file_dataframe(df)
        result = self._split_sensor_columns(result, exclude_keys = ['file_path', 'data_type', 'row_idx', 'original_files'])
        result = self._segment_sensor_data_by_columns(result, interval_nums = interval_nums)
        return result
    
def build_new_raw_data_mapping(sensor_ids: List, interval_nums: int = 60):
    """
    构建新的分段数据映射表，并且返回
    params: 
        sensor_ids: List，传感器列表，形如 ["ST-UAN-G02-002-01", "ST-VIC-C08-G06-01"]
        interval_nums: int，区间切分数量
    return:
        df: pd.DataFrame，其中每一个传感器的数据都被切分并且存储为新的列
    """
    # 下述文件中包含了所有的路径结果
    from ..time_series_db.workflow import get_index_table
    tb = get_index_table()


    # 这一部分就是文件处理了
    segmentor = TimeSeriesDataSegmenter()
    df = tb.search_sensor_ids(sensor_ids)
    result = segmentor.process_df(df, interval_nums)

    return result

def build_database_chunks(config_path: str, interval_nums: int = 60):
    """
    分块保存传感器数据到本地，每组传感器保存为独立文件，并原地更新配置文件
    
    params: 
        config_path: str，传感器配置文件的路径（YAML格式）
        interval_nums: int，区间切分数量
    
    return:
        saved_files: Dict，保存的文件路径字典，键为传感器组名，值为文件路径
        updated_config_path: str，更新后的配置文件路径
    """
    # 确保配置路径为绝对路径
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    
    # 模块1: 加载和验证配置（移除核心传感器相关验证）
    config_data = _load_and_validate_config(config_path)
    
    # 模块2: 准备存储环境
    base_save_dir = _prepare_storage_environment(config_data, config_dir)
    
    # 模块3: 处理和保存传感器组数据（直接处理所有组，无核心数据单独处理）
    saved_files = {}
    processed_groups = _process_sensor_groups(
        config_data, base_save_dir, interval_nums, saved_files
    )
    
    # 模块4: 更新配置文件（移除核心列相关配置）
    updated_config_path = _update_config_files(
        config_data, saved_files, base_save_dir, 
        config_path, interval_nums, processed_groups
    )
    
    print(f"\n数据库构建完成! 共处理 {processed_groups} 个传感器组")
    return saved_files, updated_config_path

def _load_and_validate_config(config_path: str):
    """模块1: 加载和验证配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        print(f"成功加载配置文件: {config_path}")
        
        # 仅验证必需配置项：sensor_groups（移除核心传感器相关所有逻辑）
        if 'sensor_groups' not in config_data:
            raise ValueError("配置文件中缺少必需的 'sensor_groups' 配置")
        
        # 移除核心传感器ID的获取与赋值逻辑
        return config_data
        
    except Exception as e:
        print(f"加载配置文件时出错: {str(e)}")
        raise

def _prepare_storage_environment(config_data, config_dir):
    """模块2: 准备存储环境"""
    # 获取或创建基础保存目录（使用绝对路径）
    base_save_dir = config_data.get('base_save_dir', os.path.join(config_dir, 'data', 'sensor_id_db'))
    base_save_dir = os.path.abspath(base_save_dir)
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"数据将保存至目录: {base_save_dir}")
    
    return base_save_dir

def _process_sensor_groups(config_data, base_save_dir, interval_nums, saved_files):
    """模块4: 处理和保存传感器组数据"""
    
    sensor_groups = config_data['sensor_groups']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_groups = 0
    
    # 定义通用基础必备列（时间相关，所有组均需保留，替代原核心列中的时间类列）
    base_required_columns = ['timestamp', 'month', 'day', 'hour', 'minute']
    
    for group_name, sensor_ids in sensor_groups.items():
        print(f"\n处理传感器组: {group_name}, 包含传感器: {sensor_ids}")
        
        try:
            # 构建该组的数据
            group_df = build_new_raw_data_mapping(sensor_ids, interval_nums)
            
            if group_df.empty:
                print(f"  警告: 传感器组 '{group_name}' 数据为空，跳过处理")
                continue
            
            # 筛选当前组实际存在的基础必备列（避免列不存在报错）
            available_base_columns = [col for col in base_required_columns if col in group_df.columns]
            if not available_base_columns:
                print(f"  警告: 未找到基础时间列，跳过该组（需包含至少一个时间相关列）")
                continue
            
            # 获取当前组的传感器列（无需排除核心列，直接匹配传感器ID）
            sensor_columns = []
            for sensor_id in sensor_ids:
                matching_cols = [col for col in group_df.columns if str(sensor_id) in str(col)]
                if not matching_cols:
                    print(f"    警告: 未找到传感器 {sensor_id} 的对应列")
                sensor_columns.extend(matching_cols)
            
            sensor_columns = list(set(sensor_columns))  # 去重
            if not sensor_columns:
                print(f"  警告: 未找到传感器组 '{group_name}' 的有效列")
                continue
            
            # 只保留基础必备列和当前组的传感器列（替代原"核心列+传感器列"）
            columns_to_keep = available_base_columns + sensor_columns
            group_df = group_df[columns_to_keep].copy()
            
            # 生成保存文件名
            filename = f"{group_name}_data_{timestamp}.parquet"
            save_path = os.path.join(base_save_dir, filename)
            
            # 保存到本地
            save_dataframe_to_parquet(group_df, save_path)
            saved_files[group_name] = save_path
            
            print(f"传感器组 '{group_name}' 已保存至: {save_path}, 数据形状: {group_df.shape}")
            print(f"  保留列: {available_base_columns} (基础必备列) + {len(sensor_columns)} 个传感器列")
            
            # 释放内存
            del group_df
            gc.collect()
            processed_groups += 1
            
        except Exception as e:
            print(f"处理传感器组 '{group_name}' 时出错: {str(e)}")
            continue
    
    return processed_groups

def _update_config_files(config_data, saved_files, base_save_dir, 
                         config_path, interval_nums, processed_groups):
    """模块5: 更新配置文件"""
    try:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        current_time = datetime.now()
        
        # 确保配置中有file_mappings字段
        if 'file_mappings' not in config_data:
            config_data['file_mappings'] = {}
        
        # 确保配置中有metadata字段
        if 'metadata' not in config_data:
            config_data['metadata'] = {}
        
        # 更新文件映射 - 使用绝对路径（无需区分核心组）
        print("\n更新文件映射 (使用绝对路径):")
        
        for group_name, file_path in saved_files.items():
            if file_path and os.path.exists(file_path):
                abs_file_path = os.path.abspath(file_path)
                config_data['file_mappings'][group_name] = abs_file_path
                print(f"  - {group_name}: {abs_file_path}")
        
        # 更新元数据（移除核心列/核心传感器相关字段）
        config_data['metadata'].update({
            'last_updated': current_time.strftime("%Y-%m-%d %H:%M:%S"),
            'interval_nums': interval_nums,
            'total_groups': processed_groups,
            'base_save_dir': base_save_dir,
            'build_status': 'success'
        })
        
        # 原地更新配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, sort_keys=False, allow_unicode=True)
        
        print(f"\n配置文件已成功原地更新: {config_path}")
        print(f"文件映射已更新，共 {len(config_data['file_mappings'])} 个传感器组")
        
        # 保存HuggingFace风格配置（移除核心相关字段，保留通用信息）
        hf_style_config = {
            "base_dir": base_save_dir,
            "sensor_groups": config_data['sensor_groups'],  # 保留所有组，无需过滤core
            "file_mappings": {k: os.path.abspath(v) for k, v in config_data['file_mappings'].items() 
                            if not any([isinstance(k, dict), isinstance(v, dict)])},
            "interval_nums": interval_nums,
            "metadata": config_data['metadata'],
            "build_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_version": "1.0"
        }
        
        # 保存为JSON配置文件
        hf_config_path = os.path.join(base_save_dir, "config.json")
        with open(hf_config_path, 'w', encoding='utf-8') as f:
            json.dump(hf_style_config, f, indent=2, ensure_ascii=False)
        
        print(f"HuggingFace风格配置已保存至: {hf_config_path}")
        
        return config_path
        
    except Exception as e:
        print(f"更新配置文件时出错: {str(e)}")
        # 出错时创建备份配置（备份逻辑无需核心相关字段）
        _create_backup_config(config_data, saved_files, base_save_dir, e)
        raise

def _create_backup_config(config_data, saved_files, base_save_dir, error):
    """创建备份配置文件"""
    try:
        backup_config_path = os.path.join(base_save_dir, f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        with open(backup_config_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'original_config': config_data,
                'file_mappings': saved_files,
                'error': str(error),
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, sort_keys=False, allow_unicode=True)
        print(f"已创建备份配置文件: {backup_config_path}")
    except Exception as backup_error:
        print(f"创建备份配置文件失败: {str(backup_error)}")


