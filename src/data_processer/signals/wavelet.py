import wave
import numpy as np
import pywt
import warnings
from typing import Union, Tuple, Optional

# 检查PyWavelets版本，兼容低版本
PYWAVELETS_VERSION = tuple(map(int, pywt.__version__.split('.')[:2]))
SUPPORT_ESTIMATE_THRESHOLD = PYWAVELETS_VERSION >= (1, 3)

def _manual_estimate_threshold(
    data: np.ndarray,
    method: str = 'sqtwolog',
    sigma: Optional[float] = None
) -> float:
    """
    手动实现阈值计算（兼容低版本PyWavelets），避免依赖内置estimate_threshold
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return 0.0
    
    # 计算噪声标准差（默认用中位数法）
    if sigma is None:
        sigma = np.median(np.abs(data)) / 0.6745
    
    # 不同阈值方法的实现
    if method == 'sqtwolog':
        threshold = sigma * np.sqrt(2 * np.log(n))
    elif method == 'rigrsure':
        # 无偏风险估计
        sorted_data = np.sort(np.abs(data))**2
        risks = (n - 2 * np.arange(1, n+1) + np.cumsum(sorted_data) + (n - np.arange(1, n+1)) * sorted_data) / n
        min_risk_idx = np.argmin(risks)
        threshold = np.sqrt(sorted_data[min_risk_idx])
    elif method == 'heursure':
        # 混合策略
        sqtwolog_thresh = sigma * np.sqrt(2 * np.log(n))
        r = sigma**2 * np.log(n) / np.mean(data**2)
        if r >= 1:
            threshold = sqtwolog_thresh
        else:
            rigrsure_thresh = _manual_estimate_threshold(data, method='rigrsure', sigma=sigma)
            threshold = min(sqtwolog_thresh, rigrsure_thresh)
    elif method == 'minimaxi':
        # 极小极大阈值
        threshold = sigma * (0.3936 + 0.1829 * np.log(n) / np.log(2))
    else:
        raise ValueError(f"不支持的阈值方法：{method}，可选：sqtwolog/rigrsure/heursure/minimaxi")
    
    return threshold

def wavelet_denoise(
    signal: Union[np.ndarray, list],
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_type: str = 'soft',
    threshold_method: str = 'sqtwolog',
    custom_threshold: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    健壮的小波去噪接口
    
    参数:
        signal: 一维时序信号（list/numpy数组）
        wavelet: 小波基名称（如'db4'/'sym8'/'haar'）
        level: 分解层数（None则自动计算最大合理层数）
        threshold_type: 阈值类型，仅支持'soft'/'hard'
        threshold_method: 阈值计算方法（sqtwolog/rigrsure/heursure/minimaxi）
        custom_threshold: 自定义阈值（优先级高于自动计算）
    
    返回:
        denoised_signal: 去噪后的信号
        info: 包含关键信息的字典（便于调试/分析）
    
    异常:
        所有参数错误都会抛出清晰的ValueError，避免模糊的AttributeError/TypeError
    """
    # -------------------------- 1. 输入参数全校验（避免低级错误） --------------------------
    # 信号校验：转为一维numpy数组
    try:
        signal = np.asarray(signal, dtype=np.float64)
    except:
        raise ValueError("输入信号必须是可转为一维数组的数值类型（list/numpy数组）")
    
    if len(signal.shape) != 1:
        raise ValueError(f"输入信号必须是一维！当前维度：{len(signal.shape)}")
    
    if len(signal) < 16:
        warnings.warn("信号长度过短（<16），小波分解效果可能不佳")
    
    # 小波基校验：检查是否支持
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except ValueError:
        supported_wavelets = pywt.wavelist()
        raise ValueError(f"小波基{wavelet}不支持！支持的小波基列表：{supported_wavelets[:10]}...（共{len(supported_wavelets)}种）")
    
    # 分解层数校验：自动计算最大合理层数
    max_possible_level = pywt.dwt_max_level(len(signal), wavelet_obj.dec_len)
    if level is None:
        level = min(3, max_possible_level)  # 默认3层（兼顾效果和速度）
    else:
        level = int(level)
        if level < 1:
            raise ValueError(f"分解层数必须≥1！当前值：{level}")
        if level > max_possible_level:
            warnings.warn(f"分解层数{level}超过最大支持层数{max_possible_level}，自动降级为{max_possible_level}")
            level = max_possible_level
    
    # 阈值类型校验
    threshold_type = threshold_type.lower()
    if threshold_type not in ['soft', 'hard']:
        raise ValueError(f"阈值类型仅支持'soft'/'hard'！当前值：{threshold_type}")
    
    # 阈值方法校验（仅当自定义阈值为None时生效）
    threshold_method = threshold_method.lower()
    if custom_threshold is None and threshold_method not in ['sqtwolog', 'rigrsure', 'heursure', 'minimaxi']:
        raise ValueError(f"阈值方法仅支持sqtwolog/rigrsure/heursure/minimaxi！当前值：{threshold_method}")
    
    # -------------------------- 2. 小波分解 --------------------------
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    cA = coeffs[0]  # 近似系数（低频，信号主体）
    cD_list = coeffs[1:]  # 细节系数（高频，噪声为主）
    
    # -------------------------- 3. 阈值计算（避免AttributeError） --------------------------
    if custom_threshold is not None:
        threshold = float(custom_threshold)
        if threshold < 0:
            raise ValueError("自定义阈值必须≥0！")
    else:
        # 提取所有细节系数（平坦化）
        all_cD = np.concatenate(cD_list)
        # 兼容不同PyWavelets版本的阈值计算
        if SUPPORT_ESTIMATE_THRESHOLD:
            try:
                threshold = pywt.thresholding.estimate_threshold(all_cD, threshold_method)
            except:
                # 兜底：用手动实现
                threshold = _manual_estimate_threshold(all_cD, threshold_method)
        else:
            threshold = _manual_estimate_threshold(all_cD, threshold_method)
    
    # -------------------------- 4. 阈值处理（仅处理细节系数） --------------------------
    cD_processed = []
    for cD in cD_list:
        # 阈值处理（严格校验mode参数，避免低级错误）
        cD_filtered = pywt.threshold(
            data=cD,
            value=threshold,
            mode=threshold_type,
            substitute=0.0  # 小于阈值的系数置0（固定值，避免错误）
        )
        cD_processed.append(cD_filtered)
    
    # -------------------------- 5. 小波重构（保证长度一致） --------------------------
    coeffs_processed = [cA] + cD_processed
    denoised_signal = pywt.waverec(coeffs_processed, wavelet=wavelet)
    # 裁剪到原信号长度（重构可能有±1的长度偏差）
    denoised_signal = denoised_signal[:len(signal)]
    
    # -------------------------- 6. 整理返回信息 --------------------------
    info = {
        'original_signal_shape': signal.shape,
        'wavelet': wavelet,
        'level': level,
        'max_possible_level': max_possible_level,
        'threshold': threshold,
        'threshold_type': threshold_type,
        'threshold_method': threshold_method if custom_threshold is None else 'custom',
        'coeffs_original': coeffs,
        'coeffs_processed': coeffs_processed,
        'pywavelets_version': pywt.__version__
    }
    
    return denoised_signal, info




