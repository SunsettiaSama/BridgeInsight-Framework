

# 这个文件的工具专门用于做数据处理

import pandas as pd
import os
import numpy as np
from typing import (
    Callable, Dict, List, Optional, Union, Tuple, 
    Any, Iterable, TypeVar, overload
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
import warnings

# 类型变量定义
T = TypeVar('T')
FuncOrList = Union[Callable[[list], Any], List[Callable[[list], Any]]]
SuffixOrMap = Union[str, List[str], Dict[Union[Callable, str], str], None]

def save_dataframe_to_parquet(df: pd.DataFrame, save_path: str) -> None:
    """
    将DataFrame保存为Parquet文件（原生支持列表类型）
    :param df: 需要保存的DataFrame
    :param file_path: 保存路径（例如 'data/timeseries.parquet'）
    """
    # 自动创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 直接保存（Parquet原生支持列表类型，无需转换）
    df.to_parquet(save_path, index=False)
    print(f"✅ 数据已保存到: {save_path} (共 {len(df)} 条记录)")

def load_dataframe_from_parquet(file_path: str) -> pd.DataFrame:
    """
    从Parquet文件加载DataFrame
    :param file_path: Parquet文件路径
    :return: 加载的DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 文件不存在: {file_path}")
    
    # 直接加载（列表类型自动恢复为Python列表）
    df = pd.read_parquet(file_path)
    print(f"✅ 数据已加载: {file_path} (共 {len(df)} 条记录)")
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗时间序列数据：
    1. 删除 Index 列（因为有错位问题）
    2. 删除包含 None 占位符的整行
    :param df: 原始DataFrame
    :return: 清洗后的DataFrame
    """
    # 1. 删除 Index 列（如果存在）
    if 'Index' in df.columns:
        df = df.drop(columns=['Index'])
    
    # 2. 删除包含 None 的整行（注意：这里检查的是 Python 的 None，不是字符串 'None'）
    # 使用 isin([None]) 会失效，因为 None 在 pandas 中会被视为 NaN
    # 正确做法：使用 isna() 检查缺失值（包括 None）
    cleaned_df = df.dropna(how='any')
    
    print(f"🧹 数据清洗完成！原数据 {len(df)} 行 → 清洗后 {len(cleaned_df)} 行")
    print(f"✅ 已删除 {len(df) - len(cleaned_df)} 行（包含 None 占位符）")
    return cleaned_df

def clean_by_length(df, config=None, min_samples=10, lower_percentile=0.05, upper_percentile=0.95):
    """
    根据列表列的长度清洗DataFrame
    
    参数:
    df : pandas.DataFrame
        输入数据表
    config : dict, optional
        列长度配置字典，格式:
        {
            '列名1': (min_length, max_length),  # 允许的长度区间
            '列名2': fixed_length,              # 固定长度
            ...
        }
        若为None则使用统计方法自动确定
    min_samples : int, default=10
        自动模式下计算统计量所需的最小有效样本数
    lower_percentile : float, default=0.05
        自动模式下计算下界分位数(5%)
    upper_percentile : float, default=0.95
        自动模式下计算上界分位数(95%)
    
    返回:
    pandas.DataFrame
        清洗后的DataFrame
    """
    # 创建副本避免修改原始数据
    df_clean = df.copy()
    
    # 1. 确定需要处理的列
    if config is not None:
        # 用户指定模式：仅处理配置中指定的列
        target_cols = list(config.keys())
        # 验证列是否存在
        missing_cols = [col for col in target_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"配置中指定的列不存在: {missing_cols}")
    else:
        # 自动模式：识别包含列表的列
        target_cols = []
        for col in df.columns:
            # 检查前100个非空值是否为列表
            sample = df[col].dropna().head(100)
            if not sample.empty and (all(isinstance(x, list) for x in sample) or all(isinstance(x, np.ndarray) for x in sample)):
                target_cols.append(col)
    
    if not target_cols:
        print("未找到需要处理的列表列，返回原始DataFrame")
        return df_clean
    
    print(f"检测到需处理的列: {target_cols}")
    
    # 2. 为每列生成有效长度掩码
    valid_mask = pd.Series(True, index=df.index)
    
    for col in target_cols:
        print(f"\n处理列: '{col}'")
        
        # 2.1 获取列配置
        if config is not None:
            # 用户配置模式
            setting = config[col]
            if isinstance(setting, (int, float)):
                # 固定长度
                min_len = max_len = int(setting)
                print(f"  使用固定长度: {min_len}")
            elif isinstance(setting, tuple) and len(setting) == 2:
                # 长度区间
                min_len, max_len = map(int, setting)
                print(f"  使用长度区间: [{min_len}, {max_len}]")
            else:
                raise ValueError(f"列 '{col}' 的配置无效，应为整数或二元元组")
        else:
            # 自动统计模式
            # 提取有效长度（仅列表类型）
            lengths = df[col].apply(
                lambda x: len(x) if (isinstance(x, list) or isinstance(x, np.ndarray)) and x is not None else np.nan
            )
            
            # 过滤无效值
            valid_lengths = lengths.dropna()
            n_valid = len(valid_lengths)
            
            if n_valid < min_samples:
                print(f"  警告: 有效样本不足({n_valid}<{min_samples})，跳过该列")
                continue
            
            # 计算统计边界 (5%/95%分位数)
            min_len = np.percentile(valid_lengths, lower_percentile * 100)
            max_len = np.percentile(valid_lengths, upper_percentile * 100)
            
            # 确保边界为整数且合理
            min_len = max(0, int(np.floor(min_len)))
            max_len = int(np.ceil(max_len))
            
            print(f"  自动确定长度区间: [{min_len}, {max_len}] "
                  f"(基于{n_valid}个有效样本，{lower_percentile*100}%-{upper_percentile*100}%分位数)")
        
        # 2.2 生成当前列的有效掩码
        col_mask = df[col].apply(
            lambda x: ((isinstance(x, list) or isinstance(x, np.ndarray)) and 
                      min_len <= len(x) <= max_len) if any(pd.notnull(x)) else False
        )
        
        # 2.3 更新全局掩码
        before_count = valid_mask.sum()
        valid_mask &= col_mask
        removed = before_count - valid_mask.sum()
        print(f"  该列过滤掉 {removed} 行异常数据")
    
    # 3. 应用最终掩码
    original_count = len(df_clean)
    df_clean = df_clean[valid_mask]
    remaining_count = len(df_clean)
    
    print(f"\n清洗完成: 原始 {original_count} 行 -> 剩余 {remaining_count} 行 "
          f"(移除 {original_count - remaining_count} 行, {100*(1-remaining_count/original_count):.2f}%)")
    
    return df_clean







