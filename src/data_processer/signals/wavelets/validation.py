import numpy as np
import pywt
import warnings
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def validate_input_signal(signal: Union[np.ndarray, list]) -> np.ndarray:
    """
    验证和转换输入信号为一维numpy数组
    """
    try:
        signal = np.asarray(signal, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"输入信号转换失败！必须是数值类型（list/numpy数组），错误：{str(e)}")
    
    if len(signal.shape) != 1:
        raise ValueError(f"输入信号必须是一维！当前维度：{len(signal.shape)}")
    
    signal_len = len(signal)
    if signal_len < 8:
        raise ValueError(f"信号长度过短（<8），无法进行小波分解！当前长度：{signal_len}")
    elif signal_len < 32:
        warnings.warn(f"信号长度较短（{signal_len}），小波分解效果可能不佳，建议长度≥64")
    
    return signal


def validate_wavelet(wavelet: str) -> pywt.Wavelet:
    """
    验证小波基是否支持
    """
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
        return wavelet_obj
    except ValueError:
        supported_wavelets = pywt.wavelist()
        common_wavelets = ['db4', 'sym8', 'haar', 'coif5', 'bior2.2']
        raise ValueError(
            f"小波基{wavelet}不支持！\n"
            f"常用小波基：{common_wavelets}\n"
            f"全部支持的小波基共{len(supported_wavelets)}种，前10种：{supported_wavelets[:10]}"
        )


def validate_and_set_decomposition_level(
    signal_len: int,
    level: Optional[int],
    wavelet_obj: pywt.Wavelet
) -> int:
    """
    验证和设置分解层数
    """
    max_possible_level = pywt.dwt_max_level(signal_len, wavelet_obj.dec_len)
    
    if level is None:
        level = min(2 if signal_len < 128 else 4, max_possible_level)
    else:
        level = int(level)
        if level < 1:
            raise ValueError(f"分解层数必须≥1！当前值：{level}")
        if level > max_possible_level:
            warnings.warn(f"分解层数{level}超过最大支持层数{max_possible_level}，自动降级为{max_possible_level}")
            level = max_possible_level
    
    logger.info(f"小波分解层数：{level}（最大支持：{max_possible_level}）")
    return level


def validate_threshold_type(threshold_type: str) -> str:
    """
    验证阈值类型
    """
    threshold_type = threshold_type.lower()
    if threshold_type not in ['soft', 'hard']:
        raise ValueError(f"阈值类型仅支持'soft'/'hard'！当前值：{threshold_type}")
    return threshold_type


def validate_threshold_method(threshold_method: str, custom_threshold: Optional[float]) -> str:
    """
    验证阈值计算方法
    """
    threshold_method = threshold_method.lower()
    if custom_threshold is None and threshold_method not in ['sqtwolog', 'rigrsure', 'heursure', 'minimaxi']:
        raise ValueError(f"阈值方法仅支持sqtwolog/rigrsure/heursure/minimaxi！当前值：{threshold_method}")
    return threshold_method


def validate_custom_threshold(custom_threshold: Optional[float]) -> Optional[float]:
    """
    验证自定义阈值
    """
    if custom_threshold is not None:
        custom_threshold = float(custom_threshold)
        if custom_threshold < 0:
            raise ValueError("自定义阈值必须≥0！")
    return custom_threshold
