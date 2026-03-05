"""
小波去噪模块 - Wavelet Denoising Module

这个模块提供了健壮的小波去噪功能，支持多种小波基、阈值方法和处理选项。

主要接口:
    denoise: 主要的去噪接口，参数稳定，不会因后续更新而改变

使用示例:
    from src.data_processer.signals.wavelets import denoise
    
    denoised_signal, info = denoise(
        signal=your_signal,
        wavelet='db4',
        level=4,
        threshold_type='soft',
        threshold_method='sqtwolog'
    )
"""

from .denoise import denoise

__all__ = ['denoise']
