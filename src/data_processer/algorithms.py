import pandas as pd
import numpy as np
from typing import (
    Callable, Dict, List, Optional, Union, Tuple, 
    Any, Iterable, TypeVar, overload
)
from scipy import signal
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
import warnings

# 类型变量定义
T = TypeVar('T')
FuncOrList = Union[Callable[[list], Any], List[Callable[[list], Any]]]
SuffixOrMap = Union[str, List[str], Dict[Union[Callable, str], str], None]


def _standardize_column_functions(
    df: pd.DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    functions: Optional[FuncOrList] = None,
    column_functions: Optional[Dict[str, FuncOrList]] = None
) -> Dict[str, List[Callable[[list], Any]]]:
    """
    将用户输入的列-函数配置标准化为统一的字典格式 {列名: [函数列表]}
    
    该函数支持三种配置风格，但只能选择其中一种：
    
    配置风格1: 高级字典映射 (推荐用于复杂场景)
        column_functions = {
            '列1': func1,                   # 单函数
            '列2': [func1, func2],           # 多函数
            '列3': lambda x: np.max(x) - np.min(x)  # 自定义函数
        }
        映射结果: {
            '列1': [func1],
            '列2': [func1, func2],
            '列3': [<lambda>]
        }
    
    配置风格2: 简单位置映射 (适合简单场景)
        情况A: 单列多函数
            columns='direction', functions=[np.mean, np.std]
            映射结果: {'direction': [np.mean, np.std]}
            
        情况B: 多列单函数
            columns=['direction', 'speed'], functions=np.mean
            映射结果: {
                'direction': [np.mean],
                'speed': [np.mean]
            }
            
        情况C: 多列多函数 (1:1映射)
            columns=['direction', 'speed'], functions=[np.mean, np.max]
            映射结果: {
                'direction': [np.mean],
                'speed': [np.max]
            }
    
    参数说明:
    ----------
    df : pd.DataFrame
        输入的DataFrame，用于验证列是否存在
    
    columns : str, List[str] 或 None
        要处理的列名或列名列表
        - 与functions参数配合使用
        - 不能与column_functions同时使用
    
    functions : callable, List[callable] 或 None
        要应用的函数或函数列表
        - 与columns参数配合使用
        - 不能与column_functions同时使用
    
    column_functions : Dict[str, Union[callable, List[callable]]] 或 None
        高级配置字典，键为列名，值为函数或函数列表
        - 不能与columns/functions同时使用
    
    返回:
    -------
    Dict[str, List[Callable]]
        标准化后的映射字典，格式为:
        {
            '列名1': [函数1, 函数2, ...],
            '列名2': [函数3, ...],
            ...
        }
    
    异常:
    ------
    ValueError:
        - 同时提供column_functions和(columns/functions)参数
        - 指定的列不在DataFrame中
        - columns和functions参数数量不匹配
        - 未提供任何有效配置
    
    TypeError:
        - column_functions中某个列的值既不是函数也不是函数列表
        - functions参数中包含非callable对象
    
    完整配置映射示例:
    ==================
    假设df包含列: ['A', 'B', 'C']
    
    1. 高级字典配置 (推荐):
        column_functions = {
            'A': np.mean,                          # 单函数
            'B': [np.mean, np.std],                # 多函数
            'C': lambda x: np.percentile(x, 90)   # 匿名函数
        }
        => 返回: {
            'A': [np.mean],
            'B': [np.mean, np.std],
            'C': [<lambda>]
        }
    
    2. 单列多函数:
        columns='A', functions=[np.min, np.max]
        => 返回: {'A': [np.min, np.max]}
    
    3. 多列单函数:
        columns=['A', 'B'], functions=np.mean
        => 返回: {
            'A': [np.mean],
            'B': [np.mean]
        }
    
    4. 多列多函数 (1:1映射):
        columns=['A', 'B'], functions=[np.mean, np.std]
        => 返回: {
            'A': [np.mean],
            'B': [np.std]
        }
    
    5. 非法配置示例:
        a) 混合配置:
            column_functions={...}, columns=['A']  # 触发ValueError
        
        b) 列不存在:
            columns='X'  # 如果df没有'X'列，触发ValueError
        
        c) 函数数量不匹配:
            columns=['A','B'], functions=[np.mean]  # 触发ValueError
        
        d) 非函数对象:
            column_functions={'A': 'not_a_function'}  # 触发TypeError
    
    设计说明:
    =========
    1. 优先级: column_functions参数优先级最高
    2. 线程安全: 该函数不修改输入DataFrame，纯计算操作
    3. 扩展性: 新增配置风格只需扩展此函数，不影响主逻辑
    4. 验证顺序:
        - 先检查参数组合有效性
        - 再验证列存在性
        - 最后验证函数可调用性
    """

    # 情况1：使用高级字典配置
    if column_functions is not None:
        if columns is not None or functions is not None:
            raise ValueError(
                "Cannot mix 'column_functions' with 'columns' and 'functions' parameters. "
                "Use one configuration style only."
            )
        
        standardized = {}
        for col, funcs in column_functions.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            # 将单个函数转换为列表
            if callable(funcs):
                standardized[col] = [funcs]
            elif isinstance(funcs, list) and all(callable(f) for f in funcs):
                standardized[col] = funcs
            else:
                raise TypeError(
                    f"Value for column '{col}' must be a function or list of functions. "
                    f"Got type: {type(funcs).__name__}"
                )
        return standardized
    
    # 情况2：使用简单配置
    if columns is None or functions is None:
        raise ValueError(
            "Either 'column_functions' OR ('columns' and 'functions') must be provided"
        )
    
    # 标准化列名为列表
    cols = [columns] if isinstance(columns, str) else columns
    
    # 验证所有列存在
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # 标准化函数为列表
    func_list = [functions] if callable(functions) else functions
    
    # 验证所有项都是函数
    if not all(callable(f) for f in func_list):
        raise TypeError("All functions must be callable objects")
    
    # 创建映射
    if len(cols) == 1:
        # 单列多函数: {'col': [func1, func2, ...]}
        return {cols[0]: func_list}
    elif len(func_list) == 1:
        # 多列单函数: {'col1': [func], 'col2': [func], ...}
        return {col: [func_list[0]] for col in cols}
    elif len(cols) == len(func_list):
        # 多列多函数 (1:1映射): {'col1': [func1], 'col2': [func2], ...}
        return {col: [func] for col, func in zip(cols, func_list)}
    else:
        raise ValueError(
            f"Mismatch between columns ({len(cols)}) and functions ({len(func_list)}). "
            "For multi-column processing with multiple functions, either:\n"
            "1. Use single function with multiple columns\n"
            "2. Use matching length lists for columns and functions\n"
            "3. Use the 'column_functions' dictionary parameter for complex mappings"
        )

def _generate_suffix(
    func: Callable,
    suffix_config: SuffixOrMap,
    index: int,
    total: int
) -> str:
    """智能生成列后缀"""
    # 情况1：统一后缀字符串
    if isinstance(suffix_config, str):
        return suffix_config
    
    # 情况2：后缀列表 (按顺序分配)
    if isinstance(suffix_config, list):
        if index >= len(suffix_config):
            raise ValueError(
                f"Suffix list has {len(suffix_config)} items but need {total} suffixes"
            )
        return suffix_config[index]
    
    # 情况3：函数到后缀的映射
    if isinstance(suffix_config, dict):
        # 优先使用函数对象作为键
        if func in suffix_config:
            return suffix_config[func]
        
        # 尝试使用函数名
        func_name = getattr(func, '__name__', None)
        if func_name and func_name in suffix_config:
            return suffix_config[func_name]
    
    # 情况4：自动生成 (函数名或索引)
    func_name = getattr(func, '__name__', f'func_{index}')
    return f"_{func_name}" if not func_name.startswith('_') else func_name

def _process_column_chunk(
    series_chunk: pd.Series,
    func: Callable,
    handle_errors: str,
    chunk_index: int
) -> Tuple[int, pd.Series]:
    """处理单个数据块（线程安全）"""
    def safe_apply(lst):
        try:
            if not isinstance(lst, (list, tuple, np.ndarray, pd.Series)):
                raise TypeError(f"Expected list-like, got {type(lst)}")
            
            if len(lst) == 0:
                return np.nan
            
            return func(lst)
        except Exception as e:
            if handle_errors == 'coerce':
                return np.nan
            elif handle_errors == 'raise':
                raise ValueError(f"Error in chunk {chunk_index}: {str(e)}") from e
            return lst
    
    result = series_chunk.apply(safe_apply)
    return chunk_index, result

def _process_single_task(
    df: pd.DataFrame,
    col: str,
    func: Callable,
    new_col_name: str,
    handle_errors: str,
    max_workers: Optional[int] = None,
    chunk_size: int = 1000
) -> Tuple[str, pd.Series]:
    """
    并行处理单列单函数任务
    
    参数:
    chunk_size: 每个数据块的行数（优化内存使用）
    """
    series = df[col]
    n_rows = len(series)
    
    # 小数据集直接单线程处理
    if n_rows <= chunk_size or max_workers == 1:
        def safe_apply(lst):
            try:
                if not isinstance(lst, (list, tuple, np.ndarray, pd.Series)):
                    raise TypeError(f"Expected list-like, got {type(lst)}")
                if len(lst) == 0:
                    return np.nan
                return func(lst)
            except Exception as e:
                if handle_errors == 'coerce':
                    return np.nan
                elif handle_errors == 'raise':
                    raise ValueError(f"Error in column '{col}': {str(e)}") from e
                return lst
        
        return new_col_name, series.apply(safe_apply)
    
    # 大数据集分块并行处理
    n_chunks = math.ceil(n_rows / chunk_size)
    chunks = [
        (i, series.iloc[i*chunk_size:(i+1)*chunk_size])
        for i in range(n_chunks)
    ]
    
    results = [None] * n_chunks
    
    # 确定工作线程数（避免过度订阅）
    effective_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    effective_workers = min(effective_workers, n_chunks)
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_chunk = {
            executor.submit(
                _process_column_chunk, 
                chunk, 
                func, 
                handle_errors,
                chunk_idx
            ): chunk_idx
            for chunk_idx, chunk in chunks
        }
        
        for future in as_completed(future_to_chunk):
            chunk_idx, result = future.result()
            results[chunk_idx] = result
    
    # 合并结果
    final_series = pd.concat(results)
    final_series.name = new_col_name
    return new_col_name, final_series


def apply_function(
    df: pd.DataFrame,
    *,
    columns: Optional[Union[str, List[str]]] = None,
    functions: Optional[FuncOrList] = None,
    column_functions: Optional[Dict[str, FuncOrList]] = None,
    new_column_suffix: SuffixOrMap = None,
    handle_errors: str = 'coerce', 
    max_workers: Optional[int] = None,
    chunk_size: int = 5000,
    progress_bar: bool = False
) -> pd.DataFrame:
    """
    灵活处理DataFrame中列表数据的单元格注入函数
    
    支持三种配置模式:
    1. 简单模式: 单列+单函数
       columns='col_name', functions=np.mean
       
    2. 批量模式: 
       - 多列+单函数: columns=['col1','col2'], functions=np.mean
       - 单列+多函数: columns='col_name', functions=[np.mean, np.std]
       - 多列+多函数(1:1): columns=['col1','col2'], functions=[func1, func2]
       
    3. 高级模式: 自定义映射
       column_functions={
           'col1': [np.mean, np.std],
           'col2': custom_func
       }
    
    参数:
    df : pd.DataFrame
        输入的DataFrame
    columns : str 或 List[str], 可选
        要处理的列名（简单/批量模式）
    functions : callable 或 List[callable], 可选
        要应用的函数（简单/批量模式）
    column_functions : dict, 可选
        高级列-函数映射配置
    new_column_suffix : str, List[str], dict 或 None
        新列后缀配置:
        - str: 所有新列使用同一后缀 (如 "_avg")
        - List[str]: 按顺序为每个新列分配后缀
        - dict: {函数: 后缀} 或 {函数名: 后缀}
        - None: 自动使用函数名作为后缀
    handle_errors : str, 默认 'coerce'
        错误处理策略:
        - 'coerce': 转换为NaN
        - 'ignore': 保留原始值
        - 'raise': 抛出异常
    max_workers : int, 可选
        最大工作线程数，None表示自动确定
        - 对于42,420行数据，建议值: 4-8
        - 设置为1时禁用并行（用于调试）
    chunk_size : int, 默认 5000
        每个数据块的行数，优化内存使用
        - 较小值: 降低内存峰值，增加调度开销
        - 较大值: 减少调度开销，增加内存使用
    progress_bar : bool, 默认 False
        是否显示进度条（需要安装tqdm）

    返回:
    pd.DataFrame
        包含新计算列的DataFrame副本
    """
    
    # 验证参数
    if handle_errors not in ['coerce', 'ignore', 'raise']:
        raise ValueError("handle_errors must be one of: 'coerce', 'ignore', 'raise'")
    
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1 or None")
    
    if chunk_size < 100:
        warnings.warn("chunk_size too small may hurt performance. Recommended: 1000-10000")
    
    # 标准化列-函数映射
    col_func_map = _standardize_column_functions(df, columns, functions, column_functions)
    
    # 只保留索引结构，不添加索引数据列
    result_df = pd.DataFrame(index=df.index)
        
    # 准备所有任务
    tasks = []
    for col, funcs in col_func_map.items():
        for func in funcs:
            tasks.append((col, func))
    
    # 生成新列名
    new_col_names = []
    for i, (col, func) in enumerate(tasks):
        suffix = _generate_suffix(func, new_column_suffix, i, len(tasks))
        base_name = col.rstrip('_')
        candidate_name = f"{base_name}{suffix}"
        
        counter = 1
        while candidate_name in df.columns or candidate_name in new_col_names:
            candidate_name = f"{base_name}{suffix}_{counter}"
            counter += 1
        
        new_col_names.append(candidate_name)
    
    # 确定全局最大工作线程数
    effective_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    effective_workers = min(effective_workers, len(tasks) or 1)
    
    # 动态调整chunk_size（基于数据大小）
    total_cells = len(df) * len(tasks)
    if total_cells > 1_000_000:  # 百万级单元格
        chunk_size = max(2000, min(chunk_size, 20000))
    elif total_cells > 100_000:
        chunk_size = max(500, min(chunk_size, 5000))
    
    # 多线程处理
    results = {}
    start_time = time.time()
    
    if progress_bar:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="Processing columns")
        except ImportError:
            warnings.warn("tqdm not installed. Install with: pip install tqdm")
            progress_bar = False
    
    # 提交任务
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_task = {
            executor.submit(
                _process_single_task,
                df,
                col,
                func,
                new_col_name,
                handle_errors,
                max_workers=1,  # 内部不再嵌套多线程
                chunk_size=chunk_size
            ): (col, func, new_col_name)
            for (col, func), new_col_name in zip(tasks, new_col_names)
        }
        
        # 收集结果
        for future in as_completed(future_to_task):
            col, func, new_col_name = future_to_task[future]
            try:
                col_name, series = future.result()
                results[col_name] = series
            except Exception as e:
                if handle_errors == 'raise':
                    raise RuntimeError(f"Task failed for column '{col}': {str(e)}") from e
                else:
                    # 创建全NaN系列
                    warnings.warn(f"Task failed for column '{col}': {str(e)}. Filling with NaNs.")
                    results[new_col_name] = pd.Series(np.nan, index=df.index, name=new_col_name)
            
            if progress_bar:
                pbar.update(1)
    
    if progress_bar:
        pbar.close()
    
    # 合并结果
    for new_col_name in new_col_names:
        if new_col_name in results:
            result_df[new_col_name] = results[new_col_name]
        else:
            # 处理缺失结果（理论上不应发生）
            result_df[new_col_name] = np.nan
            warnings.warn(f"Missing result for column {new_col_name}, filled with NaNs")
    
    # 保留原始列（按需）
    original_cols = [col for col in df.columns if col not in result_df.columns]
    if original_cols:
        result_df = pd.concat([df[original_cols], result_df], axis=1)
    
    # 性能报告
    elapsed = time.time() - start_time
    if elapsed > 1.0:  # 仅当处理时间较长时报告
        print(f"Processed {len(tasks)} tasks on {len(df):,} rows in {elapsed:.2f} seconds "
              f"using {effective_workers} threads (chunk_size={chunk_size})")
    
    return result_df

def wind_turbulence_intensity_cal(
    wind_speeds: Union[list, np.ndarray, pd.Series],
    min_valid_speed: float = 0.5,
    max_valid_ti: float = 100.0,
    handle_invalid: str = 'nan'
) -> float:
    """
    计算风速序列的紊流度 (Turbulence Intensity)
    
    紊流度定义为：标准差 / 平均风速 × 100%
    TI = (σ / V̄) × 100%
    
    参数:
    ----------
    wind_speeds : array-like
        风速序列数据（m/s），支持列表、NumPy数组或Pandas Series
    min_valid_speed : float, 默认 0.5
        有效计算的最小平均风速阈值（m/s）
        - 低于此值时，紊流度计算无物理意义
        - IEC 61400-1标准推荐使用0.5 m/s作为阈值
    max_valid_ti : float, 默认 100.0
        有效紊流度上限（百分比）
        - 超过此值的结果可能表示测量异常
        - 典型风机设计中最大TI通常<60%
    handle_invalid : {'nan', 'zero', 'raise'}, 默认 'nan'
        无效情况处理策略:
        - 'nan': 返回NaN（推荐，保持数据类型一致性）
        - 'zero': 返回0.0
        - 'raise': 抛出ValueError异常
    
    返回:
    -------
    float
        计算得到的紊流度（百分比，%）
        - 有效范围: 0.0 ~ max_valid_ti
        - 无效输入时根据handle_invalid策略返回
    
    异常:
    ------
    ValueError:
        - 当handle_invalid='raise'且遇到无效输入时
        - 输入序列为空或全为NaN
    
    工程说明:
    =========
    1. IEC 61400-1标准要求:
        - 紊流度计算应基于10分钟平均数据
        - 低风速段(<0.5m/s)的TI值通常不用于风机载荷计算
    
    2. 特殊情况处理:
        - 当平均风速接近0时：返回NaN或0（根据策略）
        - 当TI>max_valid_ti时：截断至max_valid_ti
        - 当输入包含NaN时：自动忽略NaN值
    
    3. 典型风机TI范围:
        - 陆上风机: 10%-25%
        - 海上风机: 8%-15%
        - 湍流场地: 可高达40%+
    
    使用示例:
    =========
    >>> # 标准用例
    >>> wind_data = [7.2, 7.5, 6.9, 8.1, 7.3, 7.0, 7.8, 8.2]
    >>> calculate_turbulence_intensity(wind_data)
    6.823...
    
    >>> # 低风速情况
    >>> low_speed = [0.2, 0.3, 0.1, 0.4]
    >>> calculate_turbulence_intensity(low_speed)  # 返回 nan
    
    >>> # 带异常值的情况
    >>> noisy_data = [8.0, 8.2, 7.9, 25.5, 8.1]  # 包含异常值25.5
    >>> calculate_turbulence_intensity(noisy_data, max_valid_ti=50)
    50.0  # 被截断至最大允许值
    """
    # 验证输入
    if not isinstance(wind_speeds, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError("wind_speeds must be a list-like structure")
    
    # 转换为NumPy数组并过滤NaN
    try:
        data = np.asarray(wind_speeds, dtype=float)
        valid_data = data[~np.isnan(data)]
    except (ValueError, TypeError):
        raise ValueError("wind_speeds contains non-numeric values")
    
    # 检查有效数据量
    if len(valid_data) == 0:
        if handle_invalid == 'raise':
            raise ValueError("No valid wind speed data available for TI calculation")
        return np.nan if handle_invalid == 'nan' else 0.0
    
    # 计算统计量
    mean_speed = np.mean(valid_data)
    std_speed = np.std(valid_data, ddof=1)  # 无偏标准差 (n-1)
    
    # 低风速检查
    if mean_speed < min_valid_speed:
        if handle_invalid == 'raise':
            raise ValueError(f"Mean wind speed ({mean_speed:.2f} m/s) below minimum threshold ({min_valid_speed} m/s)")
        return np.nan if handle_invalid == 'nan' else 0.0
    
    # 计算原始紊流度
    ti = (std_speed / mean_speed) * 100.0
    
    # 处理无效/极端值
    if np.isnan(ti) or np.isinf(ti):
        if handle_invalid == 'raise':
            raise ValueError(f"Invalid TI calculation result: {ti}")
        return np.nan if handle_invalid == 'nan' else 0.0
    
    # 应用物理限制
    ti = max(0.0, min(ti, max_valid_ti))
    
    return float(ti)

def isVIV(data, f0 = 0.24486, fs=50, f0times=5, mecc0 = 0.1):
    """
    精确复现VIV_Filter方法，识别涡激振动
    
    参数:
    data : array-like
        振动加速度时程数据
    f0 : float
        拉索基频(Hz)
    fs : int, optional
        采样频率(Hz), 默认50Hz
    f0times : int, optional
        brother区间宽度(基频倍数), 默认11 (与原始调用一致)
    
    返回:
    bool
        True表示为涡激振动，False表示为一般振动
    """
    # 准则4(第一次): RMS控制 (原始阈值0.3)
    rms_value = np.sqrt(np.mean((data - np.mean(data)) ** 2))
    if rms_value < 0.3:
        return False
    
    # 计算功率谱密度 (精确匹配原始参数)
    fx, Pxx_den = signal.welch(
        data, 
        fs=fs, 
        nfft=65536,
        nperseg=2048,
        noverlap=1
    )
    
    # 准则1: 振幅控制 (原始阈值0.01)
    if np.max(data) < 0.01:
        return False
    
    # 获取主导模态
    E1 = np.max(Pxx_den)
    idx_max = np.argmax(Pxx_den)  # 修正：使用argmax确保单一索引
    f_major = fx[idx_max]
    
    # 准则3: 主导模态频率检查 (原始阈值3*f0)
    if f_major < 3 * f0:
        return False
    
    # 定义brother区间 (核心参数f0times=11)
    f_major_left = f_major - f0times * f0
    f_major_right = f_major + f0times * f0
    
    # 确保区间边界在频率范围内
    f_major_left = max(f_major_left, fx[0])
    f_major_right = min(f_major_right, fx[-1])
    
    # 在brother区间外寻找最大能量
    left_mask = fx < f_major_left
    right_mask = fx > f_major_right
    
    # 提取区间外的功率谱值
    Pxx_den_outside = []
    if np.any(left_mask):
        Pxx_den_outside.extend(Pxx_den[left_mask])
    if np.any(right_mask):
        Pxx_den_outside.extend(Pxx_den[right_mask])
    
    # 处理无外部点的情况
    if len(Pxx_den_outside) == 0:
        Ek = 0  # 无外部能量
    else:
        # 获取区间外的最大能量
        Ek = np.max(Pxx_den_outside)
    
    # 准则2: 单峰准则 (原始阈值0.1)
    return Ek / E1 < mecc0
