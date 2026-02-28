import numpy as np
import pywt
import matplotlib.pyplot as plt
from typing import Tuple

def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3,
    threshold_type: str = 'soft',
    threshold_method: str = 'sqtwolog'
) -> Tuple[np.ndarray, dict]:
    if len(signal.shape) != 1:
        raise ValueError("输入信号必须是一维数组！")
    
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    if level < 1 or level > max_level:
        raise ValueError(f"分解层数 level={level} 无效，对于当前信号长度和基函数，最大可用层数为 {max_level}")
    
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cA = coeffs[0]
    cD_list = coeffs[1:]
    
    # 估计噪声强度 sigma，通常使用最细尺度（第一层）的细节系数计算 MAD
    # coeffs 的顺序是 [cA_n, cD_n, cD_{n-1}, ..., cD_1]，所以 cD_1 是最后一个元素
    cD1 = coeffs[-1]
    sigma = np.median(np.abs(cD1)) / 0.6745
    
    n = len(signal)
    if threshold_method == 'sqtwolog':
        threshold = sigma * np.sqrt(2 * np.log(n))
    else:
        # 默认使用通用阈值 (VisuShrink)
        threshold = sigma * np.sqrt(2 * np.log(n))
    
    cD_processed = []
    for cD in cD_list:
        cD_filtered = pywt.threshold(
            cD, 
            value=threshold, 
            mode=threshold_type,
            substitute=0
        )
        cD_processed.append(cD_filtered)
    
    coeffs_processed = [cA] + cD_processed
    denoised_signal = pywt.waverec(coeffs_processed, wavelet)
    
    denoised_signal = denoised_signal[:len(signal)]
    
    info = {
        'original_coeffs': coeffs,
        'processed_coeffs': coeffs_processed,
        'threshold': threshold,
        'sigma': sigma,
        'wavelet': wavelet,
        'level': level
    }
    
    return denoised_signal, info

# ---------------------- 测试用例 ----------------------
if __name__ == "__main__":
    # 1. 生成带噪声的时序信号（正弦信号 + 高斯噪声）
    np.random.seed(42)  # 固定随机种子，结果可复现
    t = np.linspace(0, 10, 1000)  # 时间轴
    clean_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    noisy_signal = clean_signal + 0.3 * np.random.randn(len(t))  # 加噪声
    
    # 2. 调用小波去噪函数
    denoised_signal, info = wavelet_denoise(
        signal=noisy_signal,
        wavelet='db4',
        level=3,
        threshold_type='soft',
        threshold_method='sqtwolog'
    )
    
    # 3. 可视化结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, clean_signal, label='原始无噪信号', color='blue')
    plt.title('原始无噪信号')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal, label='带噪声信号', color='orange')
    plt.title('带噪声的时序信号')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, denoised_signal, label='小波过滤后信号', color='green')
    plt.title(f'小波过滤后信号（阈值={info["threshold"]:.4f}）')
    plt.xlabel('时间')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 输出关键信息
    print(f"使用的小波基：{info['wavelet']}")
    print(f"分解层数：{info['level']}")
    print(f"计算的阈值：{info['threshold']:.4f}")