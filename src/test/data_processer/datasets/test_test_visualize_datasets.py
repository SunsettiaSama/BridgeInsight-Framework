import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
os.chdir(str(project_root))

from src.data_processer.datasets.TestDatasets.BinaryClassificationDataset import BinaryClassificationDataset
from src.data_processer.datasets.TestDatasets.RegressionDataset import RegressionDataset
from src.config.data_processer.datasets.Test.BinaryClassificationDatasetConfig import BinaryClassificationDatasetConfig
from src.config.data_processer.datasets.Test.RegressionDatasetConfig import RegressionDatasetConfig

from src.figure_paintings.figs_for_thesis.config import CN_FONT, ENG_FONT

from src.visualize_tools.utils import PlotLib



def visualize_binary_classification():
    """可视化二分类数据集"""
    print("正在加载二分类数据集...")

    config = BinaryClassificationDatasetConfig(data_dir="./data/binary_classification")
    dataset = BinaryClassificationDataset(config)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('二分类数据集可视化 (Sin vs Arctan)', fontsize=16, fontweight='bold', fontproperties=CN_FONT)
    
    sin_samples = []
    arctan_samples = []
    
    for i in range(5):
        data, label = dataset[i]
        sin_samples.append(data.numpy().squeeze())
    
    for i in range(5):
        data, label = dataset[config.num_samples_per_class + i]
        arctan_samples.append(data.numpy().squeeze())
    
    sin_avg = np.mean(sin_samples, axis=0)
    arctan_avg = np.mean(arctan_samples, axis=0)
    
    axes[0, 0].plot(sin_samples[0], label='样本1', alpha=0.7)
    axes[0, 0].plot(sin_samples[1], label='样本2', alpha=0.7)
    axes[0, 0].plot(sin_samples[2], label='样本3', alpha=0.7)
    axes[0, 0].set_title('Sin函数样本 (类型0) - 前3个样本', fontweight='bold', fontproperties=CN_FONT)
    axes[0, 0].set_xlabel('时间步', fontproperties=CN_FONT)
    axes[0, 0].set_ylabel('幅度', fontproperties=CN_FONT)
    for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    axes[0, 0].legend(prop=CN_FONT)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(arctan_samples[0], label='样本1', alpha=0.7)
    axes[0, 1].plot(arctan_samples[1], label='样本2', alpha=0.7)
    axes[0, 1].plot(arctan_samples[2], label='样本3', alpha=0.7)
    axes[0, 1].set_title('Arctan函数样本 (类型1) - 前3个样本', fontweight='bold', fontproperties=CN_FONT)
    axes[0, 1].set_xlabel('时间步', fontproperties=CN_FONT)
    axes[0, 1].set_ylabel('幅度', fontproperties=CN_FONT)
    for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    axes[0, 1].legend(prop=CN_FONT)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(sin_avg, label='Sin平均', linewidth=2, color='blue')
    axes[1, 0].plot(arctan_avg, label='Arctan平均', linewidth=2, color='orange')
    axes[1, 0].set_title('类型对比 - 平均信号', fontweight='bold', fontproperties=CN_FONT)
    axes[1, 0].set_xlabel('时间步', fontproperties=CN_FONT)
    axes[1, 0].set_ylabel('幅度', fontproperties=CN_FONT)
    for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    axes[1, 0].legend(prop=CN_FONT)
    axes[1, 0].grid(True, alpha=0.3)
    
    sin_std = [np.std(s) for s in sin_samples]
    arctan_std = [np.std(s) for s in arctan_samples]
    
    axes[1, 1].boxplot([sin_std, arctan_std], labels=['Sin (类型0)', 'Arctan (类型1)'])
    axes[1, 1].set_title('标准差分布对比', fontweight='bold', fontproperties=CN_FONT)
    axes[1, 1].set_ylabel('标准差', fontproperties=CN_FONT)
    for label in axes[1, 1].get_xticklabels() + axes[1, 1].get_yticklabels():
        label.set_fontproperties(CN_FONT)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    return fig


def visualize_regression():
    """可视化回归数据集"""
    print("正在加载回归数据集...")
    
    config = RegressionDatasetConfig(data_dir="./data/regression")
    dataset = RegressionDataset(config)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('回归数据集可视化 (Sin -> Arctan映射)', fontsize=16, fontweight='bold', fontproperties=CN_FONT)
    
    sin_samples = []
    arctan_targets = []
    
    for i in range(5):
        sin_input, arctan_target = dataset[i]
        sin_samples.append(sin_input.numpy().squeeze())
        arctan_targets.append(arctan_target.numpy().squeeze())
    
    sin_avg = np.mean(sin_samples, axis=0)
    arctan_avg = np.mean(arctan_targets, axis=0)
    
    axes[0, 0].plot(sin_samples[0], label='样本1', alpha=0.7)
    axes[0, 0].plot(sin_samples[1], label='样本2', alpha=0.7)
    axes[0, 0].plot(sin_samples[2], label='样本3', alpha=0.7)
    axes[0, 0].set_title('输入: Sin函数 - 前3个样本', fontweight='bold', fontproperties=CN_FONT)
    axes[0, 0].set_xlabel('时间步', fontproperties=CN_FONT)
    axes[0, 0].set_ylabel('幅度', fontproperties=CN_FONT)
    for label in axes[0, 0].get_xticklabels() + axes[0, 0].get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    axes[0, 0].legend(prop=CN_FONT)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(arctan_targets[0], label='样本1', alpha=0.7)
    axes[0, 1].plot(arctan_targets[1], label='样本2', alpha=0.7)
    axes[0, 1].plot(arctan_targets[2], label='样本3', alpha=0.7)
    axes[0, 1].set_title('输出: Arctan映射 - 前3个样本', fontweight='bold', fontproperties=CN_FONT)
    axes[0, 1].set_xlabel('时间步', fontproperties=CN_FONT)
    axes[0, 1].set_ylabel('幅度', fontproperties=CN_FONT)
    for label in axes[0, 1].get_xticklabels() + axes[0, 1].get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    axes[0, 1].legend(prop=CN_FONT)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(sin_avg, label='Sin输入平均', linewidth=2, color='blue', marker='o', markersize=3, markevery=50)
    axes[1, 0].plot(arctan_avg, label='Arctan输出平均', linewidth=2, color='orange', marker='s', markersize=3, markevery=50)
    axes[1, 0].set_title('输入-输出对比 - 平均信号', fontweight='bold', fontproperties=CN_FONT)
    axes[1, 0].set_xlabel('时间步', fontproperties=CN_FONT)
    axes[1, 0].set_ylabel('幅度', fontproperties=CN_FONT)
    for label in axes[1, 0].get_xticklabels() + axes[1, 0].get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    axes[1, 0].legend(prop=CN_FONT)
    axes[1, 0].grid(True, alpha=0.3)
    
    sample_idx = 2
    axes[1, 1].scatter(sin_samples[sample_idx], arctan_targets[sample_idx], alpha=0.5, s=20)
    axes[1, 1].set_title(f'输入-输出关系散点图 (样本{sample_idx})', fontweight='bold', fontproperties=CN_FONT)
    axes[1, 1].set_xlabel('Sin输入值', fontproperties=CN_FONT)
    axes[1, 1].set_ylabel('Arctan输出值', fontproperties=CN_FONT)
    for label in axes[1, 1].get_xticklabels() + axes[1, 1].get_yticklabels():
        label.set_fontproperties(ENG_FONT)
    axes[1, 1].grid(True, alpha=0.3)
    
    return fig



def generate_summary():
    """生成数据集摘要"""
    print("\n" + "="*60)
    print("数据集摘要")
    print("="*60)
    
    binary_config = BinaryClassificationDatasetConfig(data_dir="./data/binary_classification")
    regression_config = RegressionDatasetConfig(data_dir="./data/regression")
    
    print("\n【二分类数据集】")
    print(f"  • 数据集类型: {binary_config.dataset_type}")
    print(f"  • 分类类别数: {binary_config.num_classes}")
    print(f"  • 每类样本数: {binary_config.num_samples_per_class}")
    print(f"  • 总样本数: {binary_config.num_samples_per_class * binary_config.num_classes}")
    print(f"  • 序列长度: {binary_config.seq_length}")
    print(f"  • 幅度: {binary_config.amplitude}")
    print(f"  • 频率: {binary_config.frequency}")
    print(f"  • 噪声标准差: {binary_config.noise_std}")
    print(f"  • 映射关系:")
    print(f"      - 类型0: Sin函数")
    print(f"      - 类型1: Arctan函数")
    
    print("\n【回归数据集】")
    print(f"  • 数据集类型: {regression_config.dataset_type}")
    print(f"  • 总样本数: {regression_config.num_samples}")
    print(f"  • 序列长度: {regression_config.seq_length}")
    print(f"  • 幅度: {regression_config.amplitude}")
    print(f"  • 频率: {regression_config.frequency}")
    print(f"  • 噪声标准差: {regression_config.noise_std}")
    print(f"  • 映射关系:")
    print(f"      - 输入: Sin函数数据")
    print(f"      - 输出: Arctan函数映射")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    generate_summary()
    
    print("开始可视化数据集...")
    print("-" * 60)
    
    fig1 = visualize_binary_classification()
    fig2 = visualize_regression()
    
    plot_lib = PlotLib()
    plot_lib.figs.append(fig1)
    plot_lib.figs.append(fig2)
    plot_lib.show()
    
    print("-" * 60)
    print("✓ 所有可视化已完成！")
    
