"""
Backward Compatibility Layer for Wavelet Module

This module provides backward compatibility by delegating to the new wavelets submodule.
The main functionality has been reorganized into the wavelets package for better maintainability.

旧版兼容层 - 将所有调用代理到新的wavelets子模块中
"""

import warnings
from typing import Union, Tuple, Optional
import numpy as np

from .wavelets import denoise

warnings.warn(
    "导入 'src.data_processer.signals.wavelet' 已弃用。"
    "请改用 'from src.data_processer.signals.wavelets import denoise'",
    DeprecationWarning,
    stacklevel=2
)


def wavelet_denoise(
    signal: Union[np.ndarray, list],
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_type: str = 'soft',
    threshold_method: str = 'sqtwolog',
    custom_threshold: Optional[float] = None,
    layer_wise_threshold: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Deprecated: Use denoise() from wavelets module instead.
    
    This is a backward compatibility wrapper that delegates to the new denoise function.
    """
    return denoise(
        signal=signal,
        wavelet=wavelet,
        level=level,
        threshold_type=threshold_type,
        threshold_method=threshold_method,
        custom_threshold=custom_threshold,
        layer_wise_threshold=layer_wise_threshold
    )