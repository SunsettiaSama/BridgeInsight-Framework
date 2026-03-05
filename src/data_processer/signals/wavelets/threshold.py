import numpy as np
import logging
from typing import Optional
from .utils import calculate_sigma, calculate_sure_risk

logger = logging.getLogger(__name__)


def manual_estimate_threshold(
    data: np.ndarray,
    method: str = 'sqtwolog',
    sigma: Optional[float] = None
) -> float:
    """
    手动实现阈值计算（兼容低版本PyWavelets），修正heursure为标准实现
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return 0.0
    
    sigma = sigma or calculate_sigma(data)
    if sigma <= 0:
        sigma = 1e-6
    
    normalized_data = data / sigma

    if method == 'sqtwolog':
        threshold = np.sqrt(2 * np.log(n))
    
    elif method == 'rigrsure':
        sorted_abs_sq = np.sort(np.abs(normalized_data))**2
        n_coeffs = len(sorted_abs_sq)
        risks = (n_coeffs - 2 * np.arange(1, n_coeffs+1) + 
                 np.cumsum(sorted_abs_sq) + 
                 (n_coeffs - np.arange(1, n_coeffs+1)) * sorted_abs_sq) / n_coeffs
        min_risk_idx = np.argmin(risks)
        threshold = np.sqrt(sorted_abs_sq[min_risk_idx])
    
    elif method == 'heursure':
        t_sqt = np.sqrt(2 * np.log(n))
        risk_sqt = calculate_sure_risk(normalized_data, t_sqt, sigma=1.0)
        
        t_rigr = manual_estimate_threshold(data, method='rigrsure', sigma=sigma) / sigma
        risk_rigr = calculate_sure_risk(normalized_data, t_rigr, sigma=1.0)
        
        if risk_sqt < risk_rigr:
            threshold = t_sqt
        else:
            threshold = t_rigr
    
    elif method == 'minimaxi':
        threshold = 0.3936 + 0.1829 * np.log(n) / np.log(2)
    
    else:
        raise ValueError(f"不支持的阈值方法：{method}，可选：sqtwolog/rigrsure/heursure/minimaxi")
    
    threshold = threshold * sigma
    return max(threshold, 1e-6)
