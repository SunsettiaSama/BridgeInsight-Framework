import numpy as np
import pywt
import warnings
import logging
from typing import Tuple, Optional

from .utils import calculate_sigma
from .threshold import manual_estimate_threshold

logger = logging.getLogger(__name__)


def perform_wavelet_decomposition(signal: np.ndarray, wavelet: str, level: int) -> Tuple[list, int]:
    """
    执行小波分解，返回分解系数和原始信号长度
    """
    try:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        return coeffs, len(signal)
    except Exception as e:
        raise RuntimeError(f"小波分解失败！错误：{str(e)}")


def calculate_layer_wise_thresholds(
    cD_list: list,
    method: str = 'sqtwolog'
) -> list:
    """
    分层计算阈值：每一层细节系数单独计算
    """
    thresholds = []
    for idx, cD in enumerate(cD_list):
        sigma = calculate_sigma(cD)
        thresh = manual_estimate_threshold(cD, method=method, sigma=sigma)
        thresholds.append(thresh)
        logger.info(f"细节层{idx+1} - 阈值：{thresh:.4f}（sigma：{sigma:.4f}）")
    return thresholds


def calculate_global_threshold(
    cD_list: list,
    method: str = 'sqtwolog'
) -> Tuple[float, list]:
    """
    全局阈值：所有细节系数拼接计算
    """
    all_cD = np.concatenate(cD_list)
    sigma = calculate_sigma(all_cD)
    threshold = manual_estimate_threshold(all_cD, method=method, sigma=sigma)
    thresholds = [threshold] * len(cD_list)
    logger.info(f"全局阈值：{threshold:.4f}（sigma：{sigma:.4f}）")
    return threshold, thresholds


def apply_threshold_to_coefficients(
    cD_list: list,
    thresholds: list,
    threshold_type: str = 'soft'
) -> list:
    """
    将阈值应用到细节系数
    """
    cD_processed = []
    for idx, (cD, thresh) in enumerate(zip(cD_list, thresholds)):
        try:
            cD_filtered = pywt.threshold(
                data=cD,
                value=thresh,
                mode=threshold_type,
                substitute=0.0
            )
            cD_processed.append(cD_filtered)
            logger.debug(f"细节层{idx+1} - 原始系数范围：[{np.min(cD):.4f}, {np.max(cD):.4f}]，处理后范围：[{np.min(cD_filtered):.4f}, {np.max(cD_filtered):.4f}]")
        except Exception as e:
            raise RuntimeError(f"细节层{idx+1}阈值处理失败！错误：{str(e)}")
    return cD_processed


def reconstruct_signal(
    cA: np.ndarray,
    cD_processed: list,
    wavelet: str,
    original_signal_len: int
) -> np.ndarray:
    """
    小波重构信号
    """
    try:
        coeffs_processed = [cA] + cD_processed
        denoised_signal = pywt.waverec(coeffs_processed, wavelet=wavelet)
    except Exception as e:
        raise RuntimeError(f"小波重构失败！错误：{str(e)}")
    
    denoised_signal = denoised_signal[:original_signal_len]
    if len(denoised_signal) != original_signal_len:
        warnings.warn(f"重构信号长度({len(denoised_signal)})与原信号({original_signal_len})不一致，已裁剪")
    
    return denoised_signal
