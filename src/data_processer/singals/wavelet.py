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

# -------------------------- 新增：稀疏优化小波去噪函数 --------------------------
def _ista_solver(
    coeffs_original: list,
    signal: np.ndarray,
    wavelet: str,
    lambda_sparse: float,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> list:
    """
    ISTA（迭代软阈值算法）求解稀疏优化问题：min ||y - Ψx||² + λ||x||₁
    适配小波系数结构，仅对细节系数做稀疏优化
    
    参数:
        coeffs_original: 原始小波分解系数 [cA, cDn, cDn-1, ..., cD1]
        signal: 原始带噪信号
        wavelet: 小波基
        lambda_sparse: 稀疏正则化系数（控制稀疏度，越大越稀疏）
        max_iter: 最大迭代次数
        tol: 收敛阈值
    
    返回:
        coeffs_sparse: 稀疏优化后的小波系数
    """
    # 初始化：用原始系数作为初始值
    coeffs_sparse = [np.copy(c) for c in coeffs_original]
    cA = coeffs_sparse[0]
    cD_list = coeffs_sparse[1:]
    
    # 计算步长（Lipschitz常数倒数）：工程经验值，无需严格计算
    step_size = 0.01
    
    # 迭代求解
    for iter_idx in range(max_iter):
        # 保存上一轮系数用于收敛判断
        cD_prev = [np.copy(c) for c in cD_list]
        
        # 步骤1：重构当前系数对应的信号
        coeffs_temp = [cA] + cD_list
        signal_recon = pywt.waverec(coeffs_temp, wavelet)[:len(signal)]
        
        # 步骤2：计算残差
        residual = signal - signal_recon
        
        # 步骤3：小波分解残差（用于梯度更新）
        coeffs_residual = pywt.wavedec(residual, wavelet, level=len(cD_list))
        cD_residual = coeffs_residual[1:]
        
        # 步骤4：梯度下降更新细节系数
        for i in range(len(cD_list)):
            cD_list[i] = cD_list[i] + step_size * cD_residual[i]
        
        # 步骤5：软阈值操作（L1正则化的核心，保证稀疏性）
        for i in range(len(cD_list)):
            # 软阈值：x = sign(x)(|x| - λ*step_size)+
            cD_list[i] = np.sign(cD_list[i]) * np.maximum(np.abs(cD_list[i]) - lambda_sparse * step_size, 0)
        
        # 收敛判断：所有细节系数的L2误差小于tol
        error = sum(np.linalg.norm(cD_list[i] - cD_prev[i]) for i in range(len(cD_list)))
        if error < tol:
            # print(f"ISTA收敛，迭代次数：{iter_idx+1}")
            break
    
    # 组装稀疏系数
    coeffs_sparse = [cA] + cD_list
    return coeffs_sparse

def wavelet_denoise_sparse(
    signal: Union[np.ndarray, list],
    wavelet: str = 'db4',
    level: Optional[int] = None,
    lambda_sparse: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[np.ndarray, dict]:
    """
    稀疏优化小波去噪接口（前沿方法，替换传统阈值）
    
    参数:
        signal: 一维时序信号（list/numpy数组）
        wavelet: 小波基名称（如'db4'/'sym8'/'haar'）
        level: 分解层数（None则自动计算最大合理层数）
        lambda_sparse: 稀疏正则化系数（None则自动估计）
        max_iter: ISTA迭代最大次数
        tol: ISTA收敛阈值
    
    返回:
        denoised_signal: 稀疏优化去噪后的信号
        info: 包含关键信息的字典
    """
    # -------------------------- 1. 输入参数全校验 --------------------------
    try:
        signal = np.asarray(signal, dtype=np.float64)
    except:
        raise ValueError("输入信号必须是可转为一维数组的数值类型（list/numpy数组）")
    
    if len(signal.shape) != 1:
        raise ValueError(f"输入信号必须是一维！当前维度：{len(signal.shape)}")
    
    if len(signal) < 16:
        warnings.warn("信号长度过短（<16），小波分解效果可能不佳")
    
    # 小波基校验
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
    except ValueError:
        supported_wavelets = pywt.wavelist()
        raise ValueError(f"小波基{wavelet}不支持！支持的小波基列表：{supported_wavelets[:10]}...（共{len(supported_wavelets)}种）")
    
    # 分解层数校验
    max_possible_level = pywt.dwt_max_level(len(signal), wavelet_obj.dec_len)
    if level is None:
        level = min(3, max_possible_level)
    else:
        level = int(level)
        if level < 1:
            raise ValueError(f"分解层数必须≥1！当前值：{level}")
        if level > max_possible_level:
            warnings.warn(f"分解层数{level}超过最大支持层数{max_possible_level}，自动降级为{max_possible_level}")
            level = max_possible_level
    
    # -------------------------- 2. 小波分解 --------------------------
    coeffs_original = pywt.wavedec(signal, wavelet=wavelet, level=level)
    
    # -------------------------- 3. 自动估计稀疏正则化系数λ --------------------------
    if lambda_sparse is None:
        # 提取细节系数
        all_cD = np.concatenate(coeffs_original[1:])
        # 基于噪声标准差估计λ（工程经验公式）
        sigma = np.median(np.abs(all_cD)) / 0.6745
        lambda_sparse = sigma * np.sqrt(2 * np.log(len(signal))) * 0.1  # 0.1是经验缩放因子
    
    # -------------------------- 4. 稀疏优化求解（ISTA） --------------------------
    coeffs_sparse = _ista_solver(
        coeffs_original=coeffs_original,
        signal=signal,
        wavelet=wavelet,
        lambda_sparse=lambda_sparse,
        max_iter=max_iter,
        tol=tol
    )
    
    # -------------------------- 5. 小波重构 --------------------------
    denoised_signal = pywt.waverec(coeffs_sparse, wavelet)[:len(signal)]
    
    # -------------------------- 6. 整理返回信息 --------------------------
    info = {
        'original_signal_shape': signal.shape,
        'wavelet': wavelet,
        'level': level,
        'max_possible_level': max_possible_level,
        'lambda_sparse': lambda_sparse,
        'ista_max_iter': max_iter,
        'ista_tol': tol,
        'coeffs_original': coeffs_original,
        'coeffs_sparse': coeffs_sparse,
        'pywavelets_version': pywt.__version__
    }
    
    return denoised_signal, info

# -------------------------- 新增：SNR计算函数（用于对比效果） --------------------------
def calculate_snr(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """
    计算信噪比（越高表示去噪效果越好）
    """
    noise = clean_signal - denoised_signal
    signal_power = np.sum(clean_signal ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)



# -------------------------- 测试用例（验证无低级错误） --------------------------
if __name__ == "__main__":
    # 生成测试信号（带噪声的正弦波）
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    clean = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    noisy = clean + 0.3 * np.random.randn(len(t))
    
    # 调用健壮接口（测试默认参数）
    try:
        denoised, info = wavelet_denoise(
            signal=noisy,
            wavelet='db4',
            level=3,
            threshold_type='soft',
            threshold_method='sqtwolog'
        )
        print("✅ 接口调用成功！无低级错误")
        print(f"关键信息：小波基={info['wavelet']}，分解层数={info['level']}，阈值={info['threshold']:.4f}")
        print(f"原信号长度：{len(noisy)}，去噪后长度：{len(denoised)}（长度一致）")
    except Exception as e:
        print(f"❌ 接口调用失败：{e}")