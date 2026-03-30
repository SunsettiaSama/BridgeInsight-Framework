import yaml
import sys
import os
from pathlib import Path
from collections import Counter
import random
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset
from src.visualize_tools.utils import PlotLib


def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    """从YAML配置文件加载数据集配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    config_dict['auto_split'] = True
    config = AnnotationDatasetConfig(**config_dict)
    
    return config


def get_label_distribution(subset):
    """获取数据集中的类别分布"""
    labels = []
    for i in range(len(subset)):
        _, label = subset[i]
        labels.append(label.item())
    return Counter(labels)


def plot_sample_signal(data, label, dataset_type, sample_idx, ploter):
    """绘制单个样本的时域信号"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if isinstance(data, np.ndarray):
        signal_data = data.squeeze()
    else:
        signal_data = data.cpu().numpy().squeeze()
    
    time_axis = np.arange(len(signal_data))
    
    ax.plot(time_axis, signal_data, linewidth=1.5, color='steelblue', alpha=0.8)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title(f'{dataset_type} Dataset - Sample #{sample_idx} - Label: {int(label)}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    ploter.figs.append(fig)
    
    print(f"  ✓ 绘制{dataset_type}集样本 #{sample_idx}，标签: {int(label)}")


def visualize_random_samples(train_dataset, val_dataset, num_samples=5):
    """随机抽样并绘制数据集样本"""
    print("\n" + "=" * 80)
    print("=== 数据对齐检查：随机抽样并绘制信号 ===")
    print("=" * 80)
    
    ploter = PlotLib()
    
    print(f"\n正在从训练集中随机抽样 {num_samples} 个样本...")
    train_indices = random.sample(range(len(train_dataset)), min(num_samples, len(train_dataset)))
    for sample_idx in train_indices:
        data, label = train_dataset[sample_idx]
        plot_sample_signal(data, label, "Train", sample_idx, ploter)
    
    print(f"\n正在从验证集中随机抽样 {num_samples} 个样本...")
    val_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    for sample_idx in val_indices:
        data, label = val_dataset[sample_idx]
        plot_sample_signal(data, label, "Val", sample_idx, ploter)
    
    print(f"\n✓ 生成了 {len(ploter.figs)} 张样本图表")
    
    return ploter


def main():
    config_path = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\datasets\annotation_dataset.yaml"
    
    print("=" * 80)
    print("正在加载数据集...")
    print("=" * 80)
    
    dataset_config = load_dataset_config(config_path)
    dataset = AnnotationDataset(dataset_config)
    
    print("\n✓ 数据集加载完成")
    
    print("\n获取训练集和验证集...")
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    
    print("✓ 数据集获取完成")
    
    train_dist = get_label_distribution(train_dataset)
    val_dist = get_label_distribution(val_dataset)
    
    print("\n" + "=" * 80)
    print("=== 训练集类别分布 ===")
    print("=" * 80)
    total_train = sum(train_dist.values())
    for cls in sorted(train_dist.keys()):
        cnt = train_dist[cls]
        percentage = 100 * cnt / total_train
        print(f"  类别 {cls}: {cnt:4d} 样本 ({percentage:5.1f}%)")
    print(f"  总计: {total_train} 样本")
    
    print("\n" + "=" * 80)
    print("=== 验证集类别分布 ===")
    print("=" * 80)
    total_val = sum(val_dist.values())
    for cls in sorted(val_dist.keys()):
        cnt = val_dist[cls]
        percentage = 100 * cnt / total_val
        print(f"  类别 {cls}: {cnt:4d} 样本 ({percentage:5.1f}%)")
    print(f"  总计: {total_val} 样本")
    
    print("\n" + "=" * 80)
    print("=== 数据集总体统计 ===")
    print("=" * 80)
    print(f"训练集样本数: {total_train}")
    print(f"验证集样本数: {total_val}")
    print(f"总样本数: {total_train + total_val}")
    print(f"训练/验证比例: {100 * total_train / (total_train + total_val):.1f}% / {100 * total_val / (total_train + total_val):.1f}%")
    print("=" * 80 + "\n")
    
    ploter = visualize_random_samples(train_dataset, val_dataset, num_samples=5)
    
    print("\n" + "=" * 80)
    print("正在显示样本可视化界面...")
    print("=" * 80 + "\n")
    
    ploter.show()


if __name__ == "__main__":
    main()
