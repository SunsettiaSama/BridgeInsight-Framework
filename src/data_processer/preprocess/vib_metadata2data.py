import os
import sys
import json
import numpy as np
from typing import Dict, Optional, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.config.data_processer.preprocess.vibration_io_process.config import (
    FS,
    TIME_WINDOW,
)
from src.config.data_processer.preprocess.vib_metadata2data_config import (
    STRATIFIED_DENOISE_ENABLED,
    DOMINANT_FREQ_PERCENTILE,
    DOMINANT_FREQ_RESULT_PATH,
    WAVELET_NAME,
    WAVELET_DECOMPOSITION_LEVEL,
    THRESHOLD_TYPE,
    THRESHOLD_METHOD,
    LAYER_WISE_THRESHOLD,
)
from src.data_processer.signals.wavelets.denoise import denoise


def parse_single_metadata_to_vibration_data(
    metadata: Dict,
    enable_extreme_window: bool = False,
    enable_denoise: bool = False,
) -> Dict[str, any]:
    """
    从单个元数据对象解析获取原始振动数据。
    
    该函数负责：
    1. 从元数据中提取文件路径和相关信息
    2. 加载振动数据文件
    3. 可选地进行极端窗口处理
    4. 可选地进行去噪处理
    
    参数:
        metadata: 单个元数据字典，包含以下字段：
            - file_path: 振动数据文件路径（必需）
            - sensor_id: 传感器ID（可选）
            - month: 月份（可选）
            - day: 日期（可选）
            - hour: 小时（可选）
            - actual_length: 实际数据长度（可选）
            - missing_rate: 缺失率（可选）
            - extreme_rms_indices: 极端RMS窗口索引列表（可选）
            
        enable_extreme_window: 是否提取极端窗口数据
            - False (默认): 返回完整的原始数据
            - True: 返回仅包含极端窗口的数据
            
        enable_denoise: 是否进行去噪处理
            - False (默认): 不进行去噪
            - True: 应用分层去噪策略
            
            分层去噪策略说明：
            - 低频窗口 (主频 <= 95%分位数): 应用小波去噪
            - 高频/极端窗口 (主频 > 95%分位数): 保留原始特征，不去噪
    
    返回:
        包含解析结果的字典，结构如下：
        {
            "metadata": {
                "sensor_id": str,
                "month": str,
                "day": str,
                "hour": str,
                "file_path": str,
                "actual_length": int,
                "missing_rate": float,
                "extreme_rms_indices": list,
            },
            "data": np.ndarray,  # 处理后的振动加速度数据
            "data_length": int,  # 数据点数
            "processing_info": {
                "enable_extreme_window": bool,
                "enable_denoise": bool,
                "denoise_info": {  # 去噪处理详情
                    "stratified_denoise_enabled": bool,
                    "freq_threshold": float,  # 95%分位数阈值 (Hz)
                    "denoise_applied": bool,
                    "num_windows_denoised": int,  # 应用去噪的窗口数
                    "num_windows_preserved": int,  # 保留的极端窗口数
                    "percentile": int,  # 使用的分位数（如95）
                },
                "fs": int,  # 采样频率 (Hz)
                "time_window": float,  # 时间窗口长度 (秒)
                "window_size": int,  # 窗口采样点数
                "num_windows": int,  # 总窗口数
            }
        }
    
    异常:
        FileNotFoundError: 当文件不存在时
        ValueError: 当metadata缺少必需字段或数据格式不正确时
        Exception: 数据解析失败时
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata 必须是字典类型")
    
    if "file_path" not in metadata:
        raise ValueError("metadata 必须包含 'file_path' 字段")
    
    file_path = metadata["file_path"]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"振动数据文件不存在: {file_path}")
    
    unpacker = UNPACK(init_path=False)
    
    raw_data = unpacker.VIC_DATA_Unpack(file_path)
    
    if len(raw_data) == 0:
        raise ValueError(f"无法从文件中解析出有效数据: {file_path}")
    
    window_size = int(TIME_WINDOW * FS)
    num_windows = len(raw_data) // window_size
    
    processed_data = raw_data
    
    if enable_extreme_window and "extreme_rms_indices" in metadata:
        extreme_indices = metadata.get("extreme_rms_indices", [])
        if isinstance(extreme_indices, list) and len(extreme_indices) > 0:
            extreme_windows = []
            for idx in extreme_indices:
                start = int(idx * window_size)
                end = int((idx + 1) * window_size)
                if end <= len(raw_data):
                    extreme_windows.append(raw_data[start:end])
            
            if extreme_windows:
                processed_data = np.concatenate(extreme_windows)
    
    if enable_denoise:
        processed_data, denoise_info = _apply_denoise_placeholder(
            processed_data,
            metadata=metadata,
            enable_denoise=enable_denoise,
        )
    else:
        denoise_info = {
            "stratified_denoise_enabled": False,
            "freq_threshold": None,
            "denoise_applied": False,
            "num_windows_denoised": 0,
            "num_windows_preserved": 0,
        }
    
    result = {
        "metadata": {
            "sensor_id": metadata.get("sensor_id", None),
            "month": metadata.get("month", None),
            "day": metadata.get("day", None),
            "hour": metadata.get("hour", None),
            "file_path": file_path,
            "actual_length": metadata.get("actual_length", len(raw_data)),
            "missing_rate": metadata.get("missing_rate", 0.0),
            "extreme_rms_indices": metadata.get("extreme_rms_indices", []),
        },
        "data": processed_data,
        "data_length": len(processed_data),
        "processing_info": {
            "enable_extreme_window": enable_extreme_window,
            "enable_denoise": enable_denoise,
            "denoise_info": denoise_info,
            "fs": FS,
            "time_window": TIME_WINDOW,
            "window_size": window_size,
            "num_windows": num_windows,
        }
    }
    
    return result


def _load_dominant_freq_statistics() -> Dict:
    """
    加载主频统计结果，获取95%分位数阈值。
    
    返回:
        包含主频统计信息的字典：
        {
            "freq_p95": float,  # 95%分位数频率值
            "freq_stats": {
                "mean": float,
                "median": float,
                "std": float,
                "min": float,
                "max": float,
            },
            "all_dominant_frequencies": list,
        }
    
    异常:
        FileNotFoundError: 当统计结果文件不存在时
        json.JSONDecodeError: 当JSON格式错误时
    """
    if not os.path.exists(DOMINANT_FREQ_RESULT_PATH):
        raise FileNotFoundError(
            f"主频统计结果文件不存在: {DOMINANT_FREQ_RESULT_PATH}\n"
            f"请先运行 vibration_io_process 工作流来生成统计结果。"
        )
    
    with open(DOMINANT_FREQ_RESULT_PATH, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    return stats


def _calculate_freq_threshold(freq_statistics: Dict) -> float:
    """
    根据主频统计结果计算频率阈值。
    
    当前使用95%分位数作为阈值，但可以通过配置灵活调整。
    
    参数:
        freq_statistics: 主频统计信息字典
    
    返回:
        频率阈值（Hz）
    """
    percentile = DOMINANT_FREQ_PERCENTILE
    
    if 'freq_p95' in freq_statistics:
        threshold = freq_statistics['freq_p95']
    elif 'freq_stats' in freq_statistics and 'p95' in freq_statistics['freq_stats']:
        threshold = freq_statistics['freq_stats']['p95']
    else:
        raise ValueError(
            f"无法从统计结果中找到 {int(percentile * 100)}% 分位数。"
            f"可用字段: {list(freq_statistics.keys())}"
        )
    
    return float(threshold)


def _apply_denoise_placeholder(
    data: np.ndarray,
    metadata: Optional[Dict] = None,
    enable_denoise: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    分层去噪处理函数。
    
    实现分层去噪策略：
    - 低频信号 (频率 <= 95%分位数): 应用小波去噪
    - 高频/极端信号 (频率 > 95%分位数): 保留原始特征，不去噪
    
    参数:
        data: 输入的振动加速度数据
        metadata: 元数据字典，包含极端RMS窗口索引等信息
        enable_denoise: 是否启用去噪
    
    返回:
        (去噪后的数据, 去噪处理信息)
        处理信息包含：
        {
            "stratified_denoise_enabled": bool,
            "freq_threshold": float,
            "denoise_applied": bool,
            "num_windows_denoised": int,
            "num_windows_preserved": int,
            "freq_threshold_unit": "Hz",
        }
    
    异常:
        FileNotFoundError: 当主频统计文件不存在时
        ValueError: 当数据或配置错误时
    """
    denoise_info = {
        "stratified_denoise_enabled": False,
        "freq_threshold": None,
        "denoise_applied": False,
        "num_windows_denoised": 0,
        "num_windows_preserved": 0,
        "freq_threshold_unit": "Hz",
    }
    
    if not enable_denoise or not STRATIFIED_DENOISE_ENABLED:
        return data, denoise_info
    
    try:
        freq_stats = _load_dominant_freq_statistics()
        freq_threshold = _calculate_freq_threshold(freq_stats)
    except Exception as e:
        raise ValueError(
            f"分层去噪初始化失败: {str(e)}\n"
            f"请确保已生成主频统计结果。"
        )
    
    window_size = int(TIME_WINDOW * FS)
    num_windows = len(data) // window_size
    
    extreme_indices = metadata.get("extreme_rms_indices", []) if metadata else []
    
    denoised_data_list = []
    num_windows_denoised = 0
    num_windows_preserved = 0
    
    for window_idx in range(num_windows):
        start = int(window_idx * window_size)
        end = int((window_idx + 1) * window_size)
        window_data = data[start:end]
        
        is_extreme_window = window_idx in extreme_indices
        
        if is_extreme_window:
            denoised_data_list.append(window_data)
            num_windows_preserved += 1
        else:
            try:
                denoised_window, _ = denoise(
                    window_data,
                    wavelet=WAVELET_NAME,
                    level=WAVELET_DECOMPOSITION_LEVEL,
                    threshold_type=THRESHOLD_TYPE,
                    threshold_method=THRESHOLD_METHOD,
                    layer_wise_threshold=LAYER_WISE_THRESHOLD,
                )
                denoised_data_list.append(denoised_window)
                num_windows_denoised += 1
            except Exception:
                denoised_data_list.append(window_data)
                num_windows_preserved += 1
    
    remaining = len(data) % window_size
    if remaining > 0:
        denoised_data_list.append(data[-remaining:])
    
    processed_data = np.concatenate(denoised_data_list)
    
    denoise_info = {
        "stratified_denoise_enabled": True,
        "freq_threshold": freq_threshold,
        "denoise_applied": True,
        "num_windows_denoised": num_windows_denoised,
        "num_windows_preserved": num_windows_preserved,
        "freq_threshold_unit": "Hz",
        "percentile": int(DOMINANT_FREQ_PERCENTILE * 100),
    }
    
    return processed_data, denoise_info
