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



"""
VIC数据窗口提取器和缓存管理模块

职责：
1. 从VIC文件加载完整数据
2. 根据window_index和window_size提取指定窗口
3. 通过LRU缓存避免频繁IO
4. 支持可配置的缓存大小限制
"""

import logging
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)


class VICWindowExtractor:
    """
    VIC文件窗口提取器
    
    职责：
    - 从VIC文件加载数据并提取指定窗口
    - 支持去噪和极端窗口处理
    - 实现元数据到数据的对齐逻辑
    - 提供缓存接口支持外部缓存机制
    """
    
    WINDOW_SIZE = 3000  # 默认窗口大小：60秒 @ 50Hz
    FS = 50.0           # 采样频率
    
    def __init__(
        self,
        enable_denoise: bool = False,
        enable_extreme_window: bool = False,
        freq_threshold: Optional[float] = None,
    ):
        """
        初始化VIC窗口提取器
        
        参数:
            enable_denoise: 是否启用去噪处理。False 时跳过所有去噪。
            enable_extreme_window: 是否只提取极端窗口
            freq_threshold: 分层去噪频率阈值（Hz）。
                仅在 enable_denoise=True 且 STRATIFIED_DENOISE_ENABLED=True 时生效。
                None 时由 _get_freq_threshold() 从统计文件读取 95% 分位数；
                仍不可用时降级为全窗口去噪。
        """
        from src.data_processer.io_unpacker import UNPACK
        self.unpacker = UNPACK(init_path=False)
        self.enable_denoise = enable_denoise
        self.enable_extreme_window = enable_extreme_window
        self.freq_threshold = freq_threshold
    
    def extract_window(
        self,
        file_path: str,
        window_index: int,
        window_size: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        从VIC文件中提取单个窗口
        
        📌 【关键约束】每个样本 = 单个独立窗口，长度固定为 window_size，NO拼接
        
        参数:
            file_path: VIC文件路径
            window_index: 窗口号（从0开始）
            window_size: 窗口大小（采样点数），默认3000
            metadata: 元数据字典，包含extreme_freq_indices和其他信息
        
        返回:
            窗口数据，shape: (window_size, 1) 的2D数组
        
        异常:
            ValueError: 窗口超出范围或参数错误
        """
        if window_size is None:
            window_size = self.WINDOW_SIZE
        
        # 1. 加载完整VIC数据
        vic_data = self.unpacker.VIC_DATA_Unpack(str(file_path))
        
        # 2. 计算窗口索引范围
        start_idx = window_index * window_size
        end_idx = start_idx + window_size
        
        # 3. 边界检查
        if end_idx > len(vic_data):
            raise ValueError(
                f"窗口超出数据范围: window_index={window_index}, "
                f"需要索引范围[{start_idx}, {end_idx}], "
                f"但数据长度={len(vic_data)}"
            )
        
        if start_idx < 0:
            raise ValueError(f"窗口起始索引不能为负: {start_idx}")
        
        # 4. 提取窗口（显式拷贝，随即释放整文件数组）
        window_data = np.array(vic_data[start_idx:end_idx])
        del vic_data
        
        # 5. 应用去噪（可选）
        # metadata 为 None 时 _apply_denoise 内部自动走实时 FFT fallback
        if self.enable_denoise:
            window_data = self._apply_denoise(window_data, window_index, metadata)
        
        # 6. 转为2D格式 (seq_len, 1)
        if window_data.ndim == 1:
            window_data = window_data.reshape(-1, 1)
        
        return window_data.astype(np.float32)
    
    def _get_freq_threshold(self) -> Optional[float]:
        """
        获取分层去噪频率阈值（Hz）。

        优先级：
        1. self.freq_threshold 硬编码值（来自 YAML denoise_freq_threshold）
        2. 从主频统计文件读取全量数据 95% 分位数
        3. 两者均不可用 → 返回 None
        """
        if self.freq_threshold is not None:
            return self.freq_threshold
        try:
            freq_stats = _load_dominant_freq_statistics()
            return _calculate_freq_threshold(freq_stats)
        except Exception as e:
            logger.warning(
                f"无法从统计文件获取主频阈值，分层去噪将退化为全窗口去噪: {e}"
            )
            return None

    def _compute_dominant_freq(self, window_data: np.ndarray) -> float:
        """
        实时计算窗口信号的主频（Hz）。

        使用 FFT 幅值谱最大值（跳过 DC 分量）。
        仅在 dominant_freq_per_window 缺失时作为 fallback 调用。
        """
        data_1d = window_data.ravel().astype(np.float64)
        n = len(data_1d)
        fft_amp = np.abs(np.fft.rfft(data_1d))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.FS)
        dominant_idx = int(np.argmax(fft_amp[1:])) + 1
        return float(freqs[dominant_idx])

    def _apply_denoise(self, window_data: np.ndarray, window_index: int, metadata: Dict) -> np.ndarray:
        """
        对窗口应用分层去噪策略。

        分层判断依据：窗口主频 vs 全量数据主频的 95% 分位数阈值。
        主频 > 阈值 → 跳过去噪（保留原始特征）；否则 → 应用小波去噪。

        注意：extreme_rms_indices / extreme_freq_indices 仅供标注使用，
              不参与去噪决策。

        判断优先级（由高到低）：

        ① dominant_freq_per_window 存在（预处理写入）+ 阈值可获取
            → 直接使用预处理主频值与阈值比对（与全量统计完全一致）

        ② dominant_freq_per_window 缺失 + 阈值可获取
            → 实时 FFT 计算窗口主频与阈值比对（fallback）

        ③ 阈值不可获取
            → 对所有窗口应用去噪（兜底）
        """
        if not STRATIFIED_DENOISE_ENABLED:
            pass  # 全局分层开关关闭 → 无差别去噪，直接落入末尾去噪逻辑
        else:
            threshold = self._get_freq_threshold()

            if threshold is not None:
                per_window_freqs = metadata.get("dominant_freq_per_window") if metadata else None

                if per_window_freqs is not None and window_index < len(per_window_freqs):
                    # ① 预处理主频值（与全量 95% 分位数统计口径一致）
                    dominant_freq = float(per_window_freqs[window_index])
                else:
                    # ② 实时 FFT fallback（仅在无预处理主频时使用）
                    dominant_freq = self._compute_dominant_freq(window_data)
                    logger.debug(
                        f"dominant_freq_per_window 缺失，使用实时 FFT: "
                        f"window_index={window_index}, freq={dominant_freq:.3f} Hz"
                    )

                if dominant_freq > threshold:
                    logger.debug(
                        f"跳过去噪（主频 {dominant_freq:.3f} Hz > 阈值 {threshold:.3f} Hz）: "
                        f"window_index={window_index}"
                    )
                    return window_data
            # ③ 阈值不可获取 → 兜底，继续执行去噪

        try:
            denoised_data, _ = denoise(
                window_data,
                wavelet=WAVELET_NAME,
                level=WAVELET_DECOMPOSITION_LEVEL,
                threshold_type=THRESHOLD_TYPE,
                threshold_method=THRESHOLD_METHOD,
                layer_wise_threshold=LAYER_WISE_THRESHOLD,
            )
            logger.debug(f"应用去噪: window_index={window_index}")
            return denoised_data
        except Exception as e:
            logger.warning(f"去噪处理失败: {e}，使用原始数据")
            return window_data


class LRUVICCache:
    """
    LRU缓存管理器，用于缓存VIC窗口数据
    
    特性：
    - LRU淘汰策略：当达到容量时，自动删除最少使用的条目
    - 缓存命中统计
    - 可配置的最大条目数
    - 📌 【关键修复】缓存键包含处理标志，避免缓存污染
    
    缓存键格式: (file_path, window_index, window_size, enable_denoise, is_extreme_window)
    """
    
    def __init__(self, max_items: int = 1000):
        """
        初始化LRU缓存
        
        参数:
            max_items: 最多缓存的条目数（默认1000）
        """
        self.max_items = max_items
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"初始化LRU缓存: max_items={max_items}")
    
    @staticmethod
    def _build_cache_key(
        file_path: str,
        window_index: int,
        window_size: int,
        enable_denoise: bool = False,
        is_extreme_window: bool = False
    ) -> Tuple:
        """
        📌 构建包含处理标志的缓存键
        
        参数:
            file_path: 文件路径
            window_index: 窗口索引
            window_size: 窗口大小
            enable_denoise: 是否启用去噪
            is_extreme_window: 是否是极端窗口
        
        返回:
            缓存键元组
        """
        return (
            str(file_path),
            int(window_index),
            int(window_size),
            bool(enable_denoise),
            bool(is_extreme_window)
        )
    
    def get(self, key: Tuple) -> Optional[np.ndarray]:
        """
        从缓存中获取数据
        
        参数:
            key: 缓存键 (file_path, window_index, window_size, enable_denoise, is_extreme_window)
        
        返回:
            缓存的数据，或None（缓存未命中）
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        # 移到末尾（标记为最近使用）
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]
    
    def put(self, key: Tuple, data: np.ndarray):
        """
        将数据存入缓存
        
        参数:
            key: 缓存键 (file_path, window_index, window_size, enable_denoise, is_extreme_window)
            data: 要缓存的数据
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = data
            return
        
        # 如果缓存满了，删除最旧的条目
        if len(self.cache) >= self.max_items:
            oldest_key, _ = self.cache.popitem(last=False)
            logger.debug(f"缓存已满，删除最旧条目: {oldest_key}")
        
        self.cache[key] = data
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("LRU缓存已清空")
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        返回:
            统计字典
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_items
        }
    
    def log_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        logger.info(
            f"缓存统计 - 命中: {stats['hits']}, "
            f"未命中: {stats['misses']}, "
            f"命中率: {stats['hit_rate']:.1f}%, "
            f"当前大小: {stats['size']}/{stats['max_size']}"
        )




def load_full_or_extreme_segments_from_file(
    metadata: Dict,
    enable_extreme_window: bool = False,
    enable_denoise: bool = False,
) -> Dict[str, any]:
    """
    📌 【重命名】从单个元数据对象加载完整或极端分段数据
    
    ⚠️ 【关键约束】此函数用于全文件级加载，用于异常检测等任务
    如果每个样本需要为单个固定长度窗口（用于分类），请使用 VICWindowExtractor
    
    该函数负责：
    1. 从元数据中提取文件路径和相关信息
    2. 加载振动数据文件
    3. 可选地提取极端主频窗口（仍为多段拼接）
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
            - extreme_freq_indices: 极端主频窗口索引列表（可选）
            
        enable_extreme_window: 是否提取极端主频窗口数据
            - False (默认): 返回完整的原始数据
            - True: 返回仅包含极端主频窗口的数据（多段拼接）
            
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
                "extreme_freq_indices": list,  # 📌 修复：使用 extreme_freq_indices
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
    
    注意:
        如果需要单样本 = 单窗口的场景，请改用 VICWindowExtractor.extract_window()
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
    
    # 📌 【修复1】使用 extreme_freq_indices 而非 extreme_rms_indices
    if enable_extreme_window and "extreme_freq_indices" in metadata:
        extreme_indices = metadata.get("extreme_freq_indices", [])
        if isinstance(extreme_indices, list) and len(extreme_indices) > 0:
            extreme_windows = []
            for idx in extreme_indices:
                start = int(idx * window_size)
                end = int((idx + 1) * window_size)
                if end <= len(raw_data):
                    extreme_windows.append(raw_data[start:end])
            
            if extreme_windows:
                # 📌 【关键说明】此处确实会拼接多个窗口
                # ⚠️ 这是设计选择：此函数用于全文件加载，不用于单样本DataLoader
                processed_data = np.concatenate(extreme_windows)
    
    if enable_denoise:
        processed_data, denoise_info = _apply_denoise_placeholder(
            processed_data,
            metadata=metadata,
            enable_denoise=enable_denoise,
            freq_threshold_override=metadata.get("denoise_freq_threshold"),
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
            "extreme_freq_indices": metadata.get("extreme_freq_indices", []),  # 📌 修复
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


# 保留向后兼容的别名（已废弃）
def parse_single_metadata_to_vibration_data(
    metadata: Dict,
    enable_extreme_window: bool = False,
    enable_denoise: bool = False,
) -> Dict[str, any]:
    """
    ⚠️ 【已废弃】使用 load_full_or_extreme_segments_from_file 代替
    
    此函数仅为向后兼容保留
    """
    logger.warning(
        "❌ parse_single_metadata_to_vibration_data 已废弃，"
        "请改用 load_full_or_extreme_segments_from_file"
    )
    return load_full_or_extreme_segments_from_file(
        metadata,
        enable_extreme_window=enable_extreme_window,
        enable_denoise=enable_denoise
    )


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
    freq_threshold_override: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    分层去噪处理函数。
    
    实现分层去噪策略：
    - 低频信号 (频率 <= 阈值): 应用小波去噪
    - 高频/极端信号 (频率 > 阈值): 保留原始特征，不去噪

    阈值来源优先级：
    1. freq_threshold_override 不为 None → 直接使用，跳过统计文件
    2. 否则 → 从 dominant_freq_statistics_result.json 读取 95% 分位数
    
    参数:
        data: 输入的振动加速度数据
        metadata: 元数据字典，包含极端RMS窗口索引等信息
        enable_denoise: 是否启用去噪
        freq_threshold_override: 硬编码频率阈值（Hz）；不为 None 时跳过统计文件读取
    
    返回:
        (去噪后的数据, 去噪处理信息)
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

    if freq_threshold_override is not None:
        freq_threshold = float(freq_threshold_override)
        logger.debug(f"使用硬编码频率阈值: {freq_threshold} Hz")
    else:
        try:
            freq_stats = _load_dominant_freq_statistics()
            freq_threshold = _calculate_freq_threshold(freq_stats)
        except Exception as e:
            raise ValueError(
                f"分层去噪初始化失败: {str(e)}\n"
                f"请确保已生成主频统计结果，或在配置中设置 denoise_freq_threshold。"
            )
    
    window_size = int(TIME_WINDOW * FS)
    num_windows = len(data) // window_size

    # 预处理写入的逐窗口主频列表（与全量统计口径一致）
    per_window_freqs = metadata.get("dominant_freq_per_window") if metadata else None

    denoised_data_list = []
    num_windows_denoised = 0
    num_windows_preserved = 0

    for window_idx in range(num_windows):
        start = int(window_idx * window_size)
        end = int((window_idx + 1) * window_size)
        window_data = data[start:end]

        # 获取当前窗口主频：优先使用预处理值，否则实时 FFT
        if per_window_freqs is not None and window_idx < len(per_window_freqs):
            dominant_freq = float(per_window_freqs[window_idx])
        else:
            fft_amp = np.abs(np.fft.rfft(window_data.ravel().astype(np.float64)))
            freqs = np.fft.rfftfreq(len(window_data), d=1.0 / FS)
            dominant_freq = float(freqs[int(np.argmax(fft_amp[1:])) + 1])

        # 主频超过阈值 → 跳过去噪，保留原始特征
        if dominant_freq > freq_threshold:
            denoised_data_list.append(window_data)
            num_windows_preserved += 1
            continue

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
