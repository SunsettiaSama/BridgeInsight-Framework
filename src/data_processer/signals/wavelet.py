import wave
import numpy as np
import pywt
import warnings
import logging
from typing import Union, Tuple, Optional

# 配置日志（便于调试）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查PyWavelets版本，兼容低版本
PYWAVELETS_VERSION = tuple(map(int, pywt.__version__.split('.')[:2]))
SUPPORT_ESTIMATE_THRESHOLD = PYWAVELETS_VERSION >= (1, 3)

def _calculate_sigma(data: np.ndarray) -> float:
    """
    鲁棒计算噪声标准差（中位数法），处理极端场景（全零/常量信号）
    """
    data = np.asarray(data)
    abs_data = np.abs(data)
    median = np.median(abs_data)
    
    # 处理常量/全零信号（避免sigma为0）
    if median == 0:
        return np.max([np.std(data), 1e-6])  # 兜底最小值
    return median / 0.6745

def _calculate_sure_risk(data: np.ndarray, threshold: float, sigma: float) -> float:
    """
    计算Stein无偏风险（SURE），标准化输入避免尺度问题
    """
    data = np.asarray(data) / sigma  # 标准化
    n = len(data)
    if n == 0:
        return np.inf
    
    abs_data = np.abs(data)
    k = np.sum(abs_data > threshold)
    # 标准SURE风险公式
    risk = (n - 2 * k + np.sum(np.minimum(abs_data**2, threshold**2)) + k * threshold**2) / n
    return risk

def _manual_estimate_threshold(
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
    
    # 鲁棒计算sigma
    sigma = sigma or _calculate_sigma(data)
    if sigma <= 0:
        sigma = 1e-6
    
    # 标准化系数（避免幅值尺度影响阈值）
    normalized_data = data / sigma

    # 不同阈值方法的实现
    if method == 'sqtwolog':
        threshold = np.sqrt(2 * np.log(n))
    
    elif method == 'rigrsure':
        # 无偏风险估计（Rigrsure）
        sorted_abs_sq = np.sort(np.abs(normalized_data))**2
        n_coeffs = len(sorted_abs_sq)
        # 计算所有候选阈值的风险
        risks = (n_coeffs - 2 * np.arange(1, n_coeffs+1) + 
                 np.cumsum(sorted_abs_sq) + 
                 (n_coeffs - np.arange(1, n_coeffs+1)) * sorted_abs_sq) / n_coeffs
        min_risk_idx = np.argmin(risks)
        threshold = np.sqrt(sorted_abs_sq[min_risk_idx])
    
    elif method == 'heursure':
        # 标准Heursure实现：对比sqtwolog和rigrsure的SURE风险，选更优者
        # 步骤1：计算sqtwolog阈值及对应风险
        t_sqt = np.sqrt(2 * np.log(n))
        risk_sqt = _calculate_sure_risk(normalized_data, t_sqt, sigma=1.0)  # 已标准化，sigma=1
        
        # 步骤2：计算rigrsure阈值及对应风险
        t_rigr = _manual_estimate_threshold(data, method='rigrsure', sigma=sigma) / sigma  # 标准化阈值
        risk_rigr = _calculate_sure_risk(normalized_data, t_rigr, sigma=1.0)
        
        # 步骤3：选择风险更小的阈值
        if risk_sqt < risk_rigr:
            threshold = t_sqt
        else:
            threshold = t_rigr
    
    elif method == 'minimaxi':
        # 极小极大阈值（稳健阈值，适合低信噪比）
        threshold = 0.3936 + 0.1829 * np.log(n) / np.log(2)
    
    else:
        raise ValueError(f"不支持的阈值方法：{method}，可选：sqtwolog/rigrsure/heursure/minimaxi")
    
    # 反标准化阈值（还原到原始尺度）
    threshold = threshold * sigma
    return max(threshold, 1e-6)  # 避免阈值为0

def wavelet_denoise(
    signal: Union[np.ndarray, list],
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_type: str = 'soft',
    threshold_method: str = 'sqtwolog',
    custom_threshold: Optional[float] = None,
    layer_wise_threshold: bool = True  # 新增：是否分层计算阈值
) -> Tuple[np.ndarray, dict]:
    """
    健壮的小波去噪接口（完善版）
    
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
    # -------------------------- 1. 输入参数全校验（增强版） --------------------------
    # 信号校验：转为一维numpy数组，处理极端长度
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
    
    # 小波基校验：更友好的提示
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except ValueError:
        supported_wavelets = pywt.wavelist()
        common_wavelets = ['db4', 'sym8', 'haar', 'coif5', 'bior2.2']
        raise ValueError(
            f"小波基{wavelet}不支持！\n"
            f"常用小波基：{common_wavelets}\n"
            f"全部支持的小波基共{len(supported_wavelets)}种，前10种：{supported_wavelets[:10]}"
        )
    
    # 分解层数校验：更合理的默认值
    max_possible_level = pywt.dwt_max_level(signal_len, wavelet_obj.dec_len)
    if level is None:
        # 自适应默认层数：短信号少分层，长信号多分层
        level = min(2 if signal_len < 128 else 4, max_possible_level)
    else:
        level = int(level)
        if level < 1:
            raise ValueError(f"分解层数必须≥1！当前值：{level}")
        if level > max_possible_level:
            warnings.warn(f"分解层数{level}超过最大支持层数{max_possible_level}，自动降级为{max_possible_level}")
            level = max_possible_level
    logger.info(f"小波分解层数：{level}（最大支持：{max_possible_level}）")
    
    # 阈值类型校验
    threshold_type = threshold_type.lower()
    if threshold_type not in ['soft', 'hard']:
        raise ValueError(f"阈值类型仅支持'soft'/'hard'！当前值：{threshold_type}")
    
    # 阈值方法校验
    threshold_method = threshold_method.lower()
    if custom_threshold is None and threshold_method not in ['sqtwolog', 'rigrsure', 'heursure', 'minimaxi']:
        raise ValueError(f"阈值方法仅支持sqtwolog/rigrsure/heursure/minimaxi！当前值：{threshold_method}")
    
    # -------------------------- 2. 小波分解 --------------------------
    try:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    except Exception as e:
        raise RuntimeError(f"小波分解失败！错误：{str(e)}")
    cA = coeffs[0]  # 近似系数（低频，信号主体）
    cD_list = coeffs[1:]  # 细节系数（高频，噪声为主）
    
    # -------------------------- 3. 阈值计算（分层/全局可选） --------------------------
    thresholds = []  # 存储各层阈值（分层模式）
    threshold = None  # 主阈值（用于返回信息）
    if custom_threshold is not None:
        # 自定义阈值：所有层使用同一阈值
        threshold = float(custom_threshold)
        if threshold < 0:
            raise ValueError("自定义阈值必须≥0！")
        thresholds = [threshold] * len(cD_list)
        logger.info(f"使用自定义阈值：{threshold}（所有细节层）")
    else:
        if layer_wise_threshold:
            # 分层计算阈值（推荐）：每一层细节系数单独计算
            logger.info("分层计算阈值（适配不同层噪声特性）")
            for idx, cD in enumerate(cD_list):
                sigma = _calculate_sigma(cD)
                thresh = _manual_estimate_threshold(cD, method=threshold_method, sigma=sigma)
                thresholds.append(thresh)
                logger.info(f"细节层{idx+1} - 阈值：{thresh:.4f}（sigma：{sigma:.4f}）")
            # 分层模式下，使用第一层的阈值作为代表值
            threshold = thresholds[0] if thresholds else None
        else:
            # 全局阈值（兼容原逻辑）：所有细节系数拼接计算
            all_cD = np.concatenate(cD_list)
            sigma = _calculate_sigma(all_cD)
            threshold = _manual_estimate_threshold(all_cD, method=threshold_method, sigma=sigma)
            thresholds = [threshold] * len(cD_list)
            logger.info(f"全局阈值：{threshold:.4f}（sigma：{sigma:.4f}）")
    
    # -------------------------- 4. 阈值处理（仅处理细节系数） --------------------------
    cD_processed = []
    for idx, (cD, thresh) in enumerate(zip(cD_list, thresholds)):
        try:
            # 阈值处理（严格校验参数）
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
    
    # -------------------------- 5. 小波重构（保证长度一致） --------------------------
    try:
        coeffs_processed = [cA] + cD_processed
        denoised_signal = pywt.waverec(coeffs_processed, wavelet=wavelet)
    except Exception as e:
        raise RuntimeError(f"小波重构失败！错误：{str(e)}")
    
    # 精确裁剪到原信号长度（避免重构长度偏差）
    denoised_signal = denoised_signal[:signal_len]
    if len(denoised_signal) != signal_len:
        warnings.warn(f"重构信号长度({len(denoised_signal)})与原信号({signal_len})不一致，已裁剪")
    
    # -------------------------- 6. 整理返回信息（增强版） --------------------------
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
        'max_possible_level': max_possible_level,
        'threshold': threshold,  # 主阈值（代表值）
        'thresholds': thresholds,  # 各层阈值列表
        'threshold_type': threshold_type,
        'threshold_method': threshold_method if custom_threshold is None else 'custom',
        'layer_wise_threshold': layer_wise_threshold,
        'coeffs_original': coeffs,
        'coeffs_processed': coeffs_processed,
        'pywavelets_version': pywt.__version__,
        'denoised_signal_stats': {
            'mean': round(np.mean(denoised_signal), 4),
            'std': round(np.std(denoised_signal), 4),
            'min': round(np.min(denoised_signal), 4),
            'max': round(np.max(denoised_signal), 4)
        }
    }
    
    return denoised_signal, info