import os
import sys
import numpy as np
from typing import Dict, Optional, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.config.data_processer.preprocess.wind_metadata2data_config import (
    WIND_SAMPLING_FREQUENCY,
    WIND_TIME_WINDOW,
    DENOISE_ENABLED,
    DENOISE_METHOD,
    SMOOTH_WINDOW_SIZE,
    WIND_SPEED_MIN,
    WIND_SPEED_MAX,
    WIND_DIRECTION_MIN,
    WIND_DIRECTION_MAX,
    WIND_ATTACK_ANGLE_MIN,
    WIND_ATTACK_ANGLE_MAX,
)


def parse_single_metadata_to_wind_data(
    metadata: Dict,
    enable_denoise: bool = False,
) -> Dict[str, any]:
    """
    从单个元数据对象解析获取原始风数据。
    
    该函数负责：
    1. 从元数据中提取文件路径和相关信息
    2. 加载风数据文件（包含风速、风向、风攻角）
    3. 可选地进行去噪处理
    
    参数:
        metadata: 单个元数据字典，包含以下字段：
            - file_path: 风数据文件路径（必需）
            - sensor_id: 传感器ID（可选）
            - month: 月份（可选）
            - day: 日期（可选）
            - hour: 小时（可选）
            
        enable_denoise: 是否进行去噪处理
            - False (默认): 不进行去噪
            - True: 应用去噪策略（占位符，待后续实现）
    
    返回:
        包含解析结果的字典，结构如下：
        {
            "metadata": {
                "sensor_id": str,
                "month": str,
                "day": str,
                "hour": str,
                "file_path": str,
            },
            "data": {
                "wind_speed": np.ndarray,  # 风速数据 (m/s)
                "wind_direction": np.ndarray,  # 风向数据 (度)
                "wind_attack_angle": np.ndarray,  # 风攻角数据 (度)
            },
            "data_length": int,  # 数据点数
            "processing_info": {
                "enable_denoise": bool,
                "denoise_info": dict,  # 去噪处理详情
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
        raise FileNotFoundError(f"风数据文件不存在: {file_path}")
    
    unpacker = UNPACK(init_path=False)
    
    wind_data = unpacker.Wind_Data_Unpack(file_path)
    
    if not wind_data or len(wind_data) != 3:
        raise ValueError(f"无法从文件中解析出有效风数据: {file_path}")
    
    wind_speed = np.array(wind_data[0], dtype=np.float32)
    wind_direction = np.array(wind_data[1], dtype=np.float32)
    wind_attack_angle = np.array(wind_data[2], dtype=np.float32)
    
    data_length = len(wind_speed)
    
    if len(wind_direction) != data_length or len(wind_attack_angle) != data_length:
        raise ValueError(
            f"风数据三个分量长度不一致: "
            f"wind_speed={len(wind_speed)}, "
            f"wind_direction={len(wind_direction)}, "
            f"wind_attack_angle={len(wind_attack_angle)}"
        )
    
    window_size = int(WIND_TIME_WINDOW * WIND_SAMPLING_FREQUENCY)
    num_windows = data_length // window_size
    
    processed_data = {
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "wind_attack_angle": wind_attack_angle,
    }
    
    if enable_denoise:
        processed_data, denoise_info = _apply_wind_denoise_placeholder(
            processed_data,
            metadata=metadata,
            enable_denoise=enable_denoise,
        )
    else:
        denoise_info = {
            "denoise_enabled": False,
            "denoise_applied": False,
            "denoise_method": "none",
        }
    
    result = {
        "metadata": {
            "sensor_id": metadata.get("sensor_id", None),
            "month": metadata.get("month", None),
            "day": metadata.get("day", None),
            "hour": metadata.get("hour", None),
            "file_path": file_path,
        },
        "data": processed_data,
        "data_length": data_length,
        "processing_info": {
            "enable_denoise": enable_denoise,
            "denoise_info": denoise_info,
            "fs": WIND_SAMPLING_FREQUENCY,
            "time_window": WIND_TIME_WINDOW,
            "window_size": window_size,
            "num_windows": num_windows,
        }
    }
    
    return result


def _smooth_with_sliding_window(
    data: np.ndarray,
    window_size: int,
    lower_bound: float,
    upper_bound: float,
) -> Tuple[np.ndarray, int]:
    """
    对数据使用滑动窗口平滑去噪。
    
    对于越界点（outliers），向前k个点和向后k个点收集非离群点，
    取平均值替换离群点。不包含离群点本身和临近离群点。
    
    参数:
        data: 输入的一维数据数组
        window_size: 向前向后各收集的非离群点个数k
        lower_bound: 数据下界
        upper_bound: 数据上界
    
    返回:
        (平滑后的数据, 修复的离群点个数)
    """
    smoothed_data = data.copy()
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    
    if len(outlier_indices) == 0:
        return smoothed_data, 0
    
    num_fixed = 0
    
    for idx in outlier_indices:
        neighbors = []
        
        left_count = 0
        left_idx = idx - 1
        while left_count < window_size and left_idx >= 0:
            if not outlier_mask[left_idx]:
                neighbors.append(data[left_idx])
                left_count += 1
            left_idx -= 1
        
        right_count = 0
        right_idx = idx + 1
        while right_count < window_size and right_idx < len(data):
            if not outlier_mask[right_idx]:
                neighbors.append(data[right_idx])
                right_count += 1
            right_idx += 1
        
        if len(neighbors) > 0:
            smoothed_data[idx] = np.mean(neighbors)
            num_fixed += 1
    
    return smoothed_data, num_fixed


def _apply_wind_denoise_placeholder(
    data: Dict,
    metadata: Optional[Dict] = None,
    enable_denoise: bool = False,
) -> Tuple[Dict, Dict]:
    """
    风数据去噪处理函数 - 滑动窗口平滑。
    
    实现分维度的滑动窗口平滑去噪：
    - 风速: 使用 [WIND_SPEED_MIN, WIND_SPEED_MAX] 范围识别离群点
    - 风向: 使用 [WIND_DIRECTION_MIN, WIND_DIRECTION_MAX] 范围识别离群点
    - 风攻角: 使用 [WIND_ATTACK_ANGLE_MIN, WIND_ATTACK_ANGLE_MAX] 范围识别离群点
    
    对于每个离群点，向前k个和向后k个非离群点取平均值替换。
    
    参数:
        data: 风数据字典，包含wind_speed, wind_direction, wind_attack_angle
        metadata: 元数据字典
        enable_denoise: 是否启用去噪
    
    返回:
        (处理后的风数据, 去噪处理信息)
    """
    denoise_info = {
        "denoise_enabled": False,
        "denoise_applied": False,
        "denoise_method": "none",
        "num_fixed_speed": 0,
        "num_fixed_direction": 0,
        "num_fixed_angle": 0,
    }
    
    if not enable_denoise or not DENOISE_ENABLED:
        return data, denoise_info
    
    smoothed_data = {
        "wind_speed": data["wind_speed"].copy(),
        "wind_direction": data["wind_direction"].copy(),
        "wind_attack_angle": data["wind_attack_angle"].copy(),
    }
    
    num_fixed_speed = 0
    num_fixed_direction = 0
    num_fixed_angle = 0
    
    try:
        smoothed_speed, num_fixed_speed = _smooth_with_sliding_window(
            data["wind_speed"],
            SMOOTH_WINDOW_SIZE,
            WIND_SPEED_MIN,
            WIND_SPEED_MAX,
        )
        smoothed_data["wind_speed"] = smoothed_speed
    except Exception:
        pass
    
    try:
        smoothed_direction, num_fixed_direction = _smooth_with_sliding_window(
            data["wind_direction"],
            SMOOTH_WINDOW_SIZE,
            WIND_DIRECTION_MIN,
            WIND_DIRECTION_MAX,
        )
        smoothed_data["wind_direction"] = smoothed_direction
    except Exception:
        pass
    
    try:
        smoothed_angle, num_fixed_angle = _smooth_with_sliding_window(
            data["wind_attack_angle"],
            SMOOTH_WINDOW_SIZE,
            WIND_ATTACK_ANGLE_MIN,
            WIND_ATTACK_ANGLE_MAX,
        )
        smoothed_data["wind_attack_angle"] = smoothed_angle
    except Exception:
        pass
    
    total_fixed = num_fixed_speed + num_fixed_direction + num_fixed_angle
    
    denoise_info = {
        "denoise_enabled": True,
        "denoise_applied": total_fixed > 0,
        "denoise_method": DENOISE_METHOD,
        "smooth_window_size": SMOOTH_WINDOW_SIZE,
        "num_fixed_speed": num_fixed_speed,
        "num_fixed_direction": num_fixed_direction,
        "num_fixed_angle": num_fixed_angle,
        "total_fixed": total_fixed,
    }
    
    return smoothed_data, denoise_info
