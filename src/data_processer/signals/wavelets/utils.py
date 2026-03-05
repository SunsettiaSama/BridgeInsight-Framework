import numpy as np
import pywt
import logging

logger = logging.getLogger(__name__)


def calculate_sigma(data: np.ndarray) -> float:
    """
    鲁棒计算噪声标准差（中位数法），处理极端场景（全零/常量信号）
    """
    data = np.asarray(data)
    abs_data = np.abs(data)
    median = np.median(abs_data)
    
    if median == 0:
        return np.max([np.std(data), 1e-6])
    return median / 0.6745


def calculate_sure_risk(data: np.ndarray, threshold: float, sigma: float) -> float:
    """
    计算Stein无偏风险（SURE），标准化输入避免尺度问题
    """
    data = np.asarray(data) / sigma
    n = len(data)
    if n == 0:
        return np.inf
    
    abs_data = np.abs(data)
    k = np.sum(abs_data > threshold)
    risk = (n - 2 * k + np.sum(np.minimum(abs_data**2, threshold**2)) + k * threshold**2) / n
    return risk
