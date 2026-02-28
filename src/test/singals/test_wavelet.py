"""
小波变换去噪功能测试
测试wavelet_denoise函数的功能和性能
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 获取项目根目录
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
src_dir = os.path.join(current_dir, "..", "..")
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_processer.singals.wavelet import wavelet_denoise
from figs.figs_for_thesis.config import CN_FONT, ENG_FONT, FONT_SIZE


def generate_test_signal(n_samples=1000, freq1=1.0, freq2=5.0, noise_level=0.3, seed=42):
    """
    生成测试信号：多频率正弦波 + 高斯噪声
    
    参数:
        n_samples: 采样点数
        freq1: 第一个频率成分 (Hz)
        freq2: 第二个频率成分 (Hz)
        noise_level: 噪声标准差
        seed: 随机种子
    
    返回:
        clean_signal: 无噪信号
        noisy_signal: 带噪信号
        time_axis: 时间轴
    """
    np.random.seed(seed)
    time_axis = np.linspace(0, 10, n_samples)
    
    # 生成干净信号
    clean_signal = (
        np.sin(2 * np.pi * freq1 * time_axis) + 
        0.5 * np.sin(2 * np.pi * freq2 * time_axis)
    )
    
    # 添加高斯噪声
    noise = noise_level * np.random.randn(len(time_axis))
    noisy_signal = clean_signal + noise
    
    return clean_signal, noisy_signal, time_axis


def calculate_snr(signal, reference):
    """计算信噪比 (dB)"""
    signal_power = np.mean(signal ** 2)
    error_power = np.mean((signal - reference) ** 2)
    snr = 10 * np.log10(signal_power / error_power) if error_power > 0 else float('inf')
    return snr


def test_wavelet_denoise():
    """
    测试小波去噪函数
    """
    print("=" * 80)
    print("小波变换去噪功能测试")
    print("=" * 80)
    
    # 生成测试信号
    print("\n[步骤1] 生成测试信号...")
    clean_signal, noisy_signal, time_axis = generate_test_signal(
        n_samples=1000,
        freq1=1.0,
        freq2=5.0,
        noise_level=0.3
    )
    print(f"✓ 已生成 {len(noisy_signal)} 个采样点的测试信号")
    
    # 计算原始噪声信号的SNR
    original_snr = calculate_snr(noisy_signal, clean_signal)
    print(f"✓ 原始信号 SNR: {original_snr:.2f} dB")
    
    # 测试不同参数组合
    print("\n[步骤2] 测试不同参数组合...")
    test_configs = [
        {'wavelet': 'db4', 'level': 3, 'threshold_type': 'soft', 'threshold_method': 'sqtwolog'},
        {'wavelet': 'db4', 'level': 4, 'threshold_type': 'soft', 'threshold_method': 'sqtwolog'},
        {'wavelet': 'sym8', 'level': 3, 'threshold_type': 'soft', 'threshold_method': 'sqtwolog'},
        {'wavelet': 'db4', 'level': 3, 'threshold_type': 'hard', 'threshold_method': 'sqtwolog'},
    ]
    
    results = []
    for i, config in enumerate(test_configs, 1):
        print(f"\n  配置 {i}: {config}")
        denoised_signal, info = wavelet_denoise(noisy_signal, **config)
        
        # 计算去噪效果
        denoised_snr = calculate_snr(denoised_signal, clean_signal)
        mse = np.mean((denoised_signal - clean_signal) ** 2)
        snr_improvement = denoised_snr - original_snr
        
        results.append({
            'config': config,
            'denoised_signal': denoised_signal,
            'info': info,
            'snr': denoised_snr,
            'mse': mse,
            'snr_improvement': snr_improvement
        })
        
        print(f"    ✓ SNR: {denoised_snr:.2f} dB (改善 {snr_improvement:.2f} dB)")
        print(f"    ✓ MSE: {mse:.6f}")
        print(f"    ✓ 阈值: {info['threshold']:.4f}")
    
    # 选择最好的结果
    print("\n[步骤3] 对比分析...")
    best_result = max(results, key=lambda x: x['snr_improvement'])
    print(f"\n✓ 最佳配置: {best_result['config']}")
    print(f"  - SNR改善: {best_result['snr_improvement']:.2f} dB")
    print(f"  - 最终SNR: {best_result['snr']:.2f} dB")
    print(f"  - MSE: {best_result['mse']:.6f}")
    
    # 输出测试总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"\n原始信号 SNR: {original_snr:.2f} dB")
    print(f"最佳去噪配置:")
    for key, value in best_result['config'].items():
        print(f"  {key}: {value}")
    print(f"\n最终 SNR: {best_result['snr']:.2f} dB")
    print(f"SNR 改善: {best_result['snr_improvement']:.2f} dB")
    print(f"MSE: {best_result['mse']:.6f}")
    
    print("\n所有配置的对比结果:")
    print("-" * 80)
    print(f"{'配置':<50} {'SNR改善(dB)':<15} {'最终SNR(dB)':<15}")
    print("-" * 80)
    for result in results:
        config_str = f"{result['config']['wavelet']}, level={result['config']['level']}, {result['config']['threshold_type']}"
        print(f"{config_str:<50} {result['snr_improvement']:>14.2f} {result['snr']:>14.2f}")
    
    print("=" * 80)


def visualize_denoising_effect():
    """
    可视化展示去噪前后的效果
    """
    print("\n" + "=" * 80)
    print("去噪效果可视化展示")
    print("=" * 80)
    
    # 生成测试信号
    print("\n[生成信号] 创建测试信号...")
    clean_signal, noisy_signal, time_axis = generate_test_signal(
        n_samples=1000,
        freq1=1.0,
        freq2=5.0,
        noise_level=0.5
    )
    
    # 进行小波去噪
    print("[去噪处理] 使用小波变换去噪...")
    denoised_signal, info = wavelet_denoise(
        noisy_signal,
        wavelet='db4',
        level=3,
        threshold_type='soft',
        threshold_method='sqtwolog'
    )
    
    # 计算性能指标
    original_snr = calculate_snr(noisy_signal, clean_signal)
    denoised_snr = calculate_snr(denoised_signal, clean_signal)
    snr_improvement = denoised_snr - original_snr
    mse_before = np.mean((noisy_signal - clean_signal) ** 2)
    mse_after = np.mean((denoised_signal - clean_signal) ** 2)
    
    print(f"✓ 原始信号 SNR: {original_snr:.2f} dB, MSE: {mse_before:.6f}")
    print(f"✓ 去噪后 SNR: {denoised_snr:.2f} dB, MSE: {mse_after:.6f}")
    print(f"✓ SNR改善: {snr_improvement:.2f} dB")
    
    # 绘制对比图
    print("[绘图] 生成可视化图表...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('小波去噪效果展示 (Wavelet Denoising)', fontproperties=CN_FONT, fontsize=18, fontweight='bold')
    
    # 第一个图：原始信号 vs 带噪信号
    axes[0].plot(time_axis, clean_signal, 'g-', linewidth=2, label='清洁信号 (Clean Signal)', alpha=0.8)
    axes[0].plot(time_axis, noisy_signal, 'r.', markersize=2, alpha=0.5, label='带噪信号 (Noisy Signal)')
    axes[0].set_ylabel('幅度 (Amplitude)', fontproperties=CN_FONT, fontsize=12, fontweight='bold')
    axes[0].set_title(f'原始信号对比 (Original) - SNR: {original_snr:.2f} dB', fontproperties=CN_FONT, fontsize=13)
    axes[0].legend(loc='upper right', fontsize=10, prop=CN_FONT)
    axes[0].grid(True, alpha=0.3)
    
    # 第二个图：去噪后的信号 vs 清洁信号
    axes[1].plot(time_axis, clean_signal, 'g-', linewidth=2, label='清洁信号 (Clean Signal)', alpha=0.8)
    axes[1].plot(time_axis, denoised_signal, 'b-', linewidth=1.5, label='去噪信号 (Denoised Signal)', alpha=0.8)
    axes[1].set_ylabel('幅度 (Amplitude)', fontproperties=CN_FONT, fontsize=12, fontweight='bold')
    axes[1].set_title(f'去噪后信号对比 (After Denoising) - SNR: {denoised_snr:.2f} dB', fontproperties=CN_FONT, fontsize=13)
    axes[1].legend(loc='upper right', fontsize=10, prop=CN_FONT)
    axes[1].grid(True, alpha=0.3)
    
    # 第三个图：三信号对比（部分区间放大）
    zoom_start, zoom_end = 100, 300
    axes[2].plot(time_axis[zoom_start:zoom_end], clean_signal[zoom_start:zoom_end], 
                 'g-', linewidth=2.5, label='清洁信号 (Clean)', marker='o', markersize=3, alpha=0.8)
    axes[2].plot(time_axis[zoom_start:zoom_end], noisy_signal[zoom_start:zoom_end], 
                 'r.', markersize=4, alpha=0.6, label='带噪信号 (Noisy)')
    axes[2].plot(time_axis[zoom_start:zoom_end], denoised_signal[zoom_start:zoom_end], 
                 'b--', linewidth=2, label='去噪信号 (Denoised)', alpha=0.8)
    axes[2].set_xlabel('时间 (Time)', fontproperties=CN_FONT, fontsize=12, fontweight='bold')
    axes[2].set_ylabel('幅度 (Amplitude)', fontproperties=CN_FONT, fontsize=12, fontweight='bold')
    axes[2].set_title(f'局部放大对比 (Zoomed Detail: t={time_axis[zoom_start]:.1f}~{time_axis[zoom_end]:.1f}s)', fontproperties=CN_FONT, fontsize=13)
    axes[2].legend(loc='upper right', fontsize=10, prop=CN_FONT)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印对比摘要
    print("\n[性能摘要]")
    print("-" * 80)
    print(f"{'指标':<30} {'去噪前':<20} {'去噪后':<20} {'改善':<10}")
    print("-" * 80)
    print(f"{'SNR (dB)':<30} {original_snr:>19.2f} {denoised_snr:>19.2f} {snr_improvement:>9.2f}")
    print(f"{'MSE':<30} {mse_before:>19.6f} {mse_after:>19.6f} {mse_before-mse_after:>9.6f}")
    print("-" * 80)
    print(f"\n去噪参数：")
    print(f"  小波基: {info['wavelet']}")
    print(f"  分解层数: {info['level']}")
    print(f"  阈值类型: {info['threshold_type']}")
    print(f"  阈值方法: {info['threshold_method']}")
    print(f"  计算阈值: {info['threshold']:.6f}")


def test_edge_cases():
    """
    测试边界情况和异常处理
    """
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)
    
    # 测试 1: 输入二维数组
    print("\n[测试1] 输入二维数组应该抛出异常...")
    try:
        signal_2d = np.random.randn(100, 2)
        wavelet_denoise(signal_2d)
        print("❌ 未能捕获异常")
    except ValueError as e:
        print(f"✓ 正确抛出异常: {e}")
    
    # 测试 2: 分解层数过高
    print("\n[测试2] 分解层数过高应该抛出异常...")
    try:
        signal = np.random.randn(100)
        wavelet_denoise(signal, level=15)
        print("❌ 未能捕获异常")
    except ValueError as e:
        print(f"✓ 正确抛出异常: {e}")
    
    # 测试 3: 分解层数为0
    print("\n[测试3] 分解层数为0应该抛出异常...")
    try:
        signal = np.random.randn(100)
        wavelet_denoise(signal, level=0)
        print("❌ 未能捕获异常")
    except ValueError as e:
        print(f"✓ 正确抛出异常: {e}")
    
    # 测试 4: 长信号处理
    print("\n[测试4] 长信号处理...")
    try:
        signal = np.random.randn(100000)
        denoised, info = wavelet_denoise(signal)
        assert len(denoised) == len(signal), "输出长度与输入不匹配"
        print(f"✓ 成功处理 {len(signal)} 点的长信号")
        print(f"  输出长度一致性: {len(denoised) == len(signal)}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
    
    # 测试 5: 短信号处理
    print("\n[测试5] 短信号处理...")
    try:
        signal = np.random.randn(32)
        denoised, info = wavelet_denoise(signal, level=2)
        assert len(denoised) == len(signal), "输出长度与输入不匹配"
        print(f"✓ 成功处理 {len(signal)} 点的短信号")
        print(f"  输出长度一致性: {len(denoised) == len(signal)}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    visualize_denoising_effect()
    test_wavelet_denoise()
    test_edge_cases()
