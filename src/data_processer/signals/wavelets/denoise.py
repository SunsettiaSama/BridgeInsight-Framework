import numpy as np
import pywt
import logging
from typing import Union, Tuple, Optional

from .validation import (
    validate_input_signal,
    validate_wavelet,
    validate_and_set_decomposition_level,
    validate_threshold_type,
    validate_threshold_method,
    validate_custom_threshold
)
from .utils import calculate_sigma
from .core import (
    perform_wavelet_decomposition,
    calculate_layer_wise_thresholds,
    calculate_global_threshold,
    apply_threshold_to_coefficients,
    reconstruct_signal
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def denoise(
    signal: Union[np.ndarray, list],
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_type: str = 'soft',
    threshold_method: str = 'sqtwolog',
    custom_threshold: Optional[float] = None,
    layer_wise_threshold: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    健壮的小波去噪接口（主要入口）
    
    参数:
        signal: 一维时序信号（list/numpy数组）
        wavelet: 小波基名称（如'db4'/'sym8'/'haar'）
        level: 分解层数（None则自动计算最大合理层数）
        threshold_type: 阈值类型，仅支持'soft'/'hard'
        threshold_method: 阈值计算方法（sqtwolog/rigrsure/heursure/minimaxi）
        custom_threshold: 自定义阈值（优先级高于自动计算）
        layer_wise_threshold: 是否分层计算阈值（推荐True，适配不同层噪声特性）
    
    返回:
        denoised_signal: 去噪后的信号
        info: 包含关键信息的字典（便于调试/分析）
    
    异常:
        所有参数错误都会抛出清晰的ValueError，避免模糊的AttributeError/TypeError
    """
    # ==================== 输入参数全校验 ====================
    signal = validate_input_signal(signal)
    signal_len = len(signal)
    
    wavelet_obj = validate_wavelet(wavelet)
    
    level = validate_and_set_decomposition_level(signal_len, level, wavelet_obj)
    
    threshold_type = validate_threshold_type(threshold_type)
    
    threshold_method = validate_threshold_method(threshold_method, custom_threshold)
    
    custom_threshold = validate_custom_threshold(custom_threshold)
    
    # ==================== 小波分解 ====================
    coeffs, original_len = perform_wavelet_decomposition(signal, wavelet, level)
    cA = coeffs[0]
    cD_list = coeffs[1:]
    
    # ==================== 阈值计算 ====================
    thresholds = []
    threshold = None
    
    if custom_threshold is not None:
        threshold = float(custom_threshold)
        thresholds = [threshold] * len(cD_list)
        logger.info(f"使用自定义阈值：{threshold}（所有细节层）")
    else:
        if layer_wise_threshold:
            logger.info("分层计算阈值（适配不同层噪声特性）")
            thresholds = calculate_layer_wise_thresholds(cD_list, method=threshold_method)
            threshold = thresholds[0] if thresholds else None
        else:
            threshold, thresholds = calculate_global_threshold(cD_list, method=threshold_method)
    
    # ==================== 阈值处理 ====================
    cD_processed = apply_threshold_to_coefficients(cD_list, thresholds, threshold_type)
    
    # ==================== 小波重构 ====================
    denoised_signal = reconstruct_signal(cA, cD_processed, wavelet, original_len)
    
    # ==================== 返回信息 ====================
    info = {
        'original_signal_shape': signal.shape,
        'original_signal_stats': {
            'mean': round(np.mean(signal), 4),
            'std': round(np.std(signal), 4),
            'min': round(np.min(signal), 4),
            'max': round(np.max(signal), 4)
        },
        'wavelet': wavelet,
        'level': level,
        'max_possible_level': pywt.dwt_max_level(signal_len, wavelet_obj.dec_len),
        'threshold': threshold,
        'thresholds': thresholds,
        'threshold_type': threshold_type,
        'threshold_method': threshold_method if custom_threshold is None else 'custom',
        'layer_wise_threshold': layer_wise_threshold,
        'coeffs_original': coeffs,
        'coeffs_processed': [cA] + cD_processed,
        'pywavelets_version': pywt.__version__,
        'denoised_signal_stats': {
            'mean': round(np.mean(denoised_signal), 4),
            'std': round(np.std(denoised_signal), 4),
            'min': round(np.min(denoised_signal), 4),
            'max': round(np.max(denoised_signal), 4)
        }
    }
    
    return denoised_signal, info
