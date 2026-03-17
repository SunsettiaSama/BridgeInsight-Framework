"""
数据集展示图像绘制脚本
绘制两个主要图像:
1. 数据集样本展示 (20-50个子图的网格展示)
2. 标签分布饼状图
"""

import os
import sys
import json
import random
from collections import Counter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


FS = 50.0
NFFT = 2048
FREQ_LIMIT = 25
WINDOW_SIZE = 3000


def _get_config():
    """延迟导入配置以避免numpy兼容性问题"""
    from src.figure_paintings.figs_for_thesis.config import (
        SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE,
        CN_FONT, ENG_FONT, get_viridis_color_map
    )
    return SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, get_viridis_color_map


def load_annotation_results(annotation_path):
    """加载标注结果JSON文件"""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_sample_data(file_path, window_index, window_size=WINDOW_SIZE):
    """加载单个样本数据"""
    from src.data_processer.io_unpacker import UNPACK
    import numpy as np
    
    unpacker = UNPACK(init_path=False)
    vibration_data = np.array(unpacker.VIC_DATA_Unpack(file_path))
    
    start_sample = window_index * window_size
    end_sample = (window_index + 1) * window_size
    
    return vibration_data[start_sample:end_sample]


def plot_time_domain(ax, data, fs=FS):
    """绘制时域波形"""
    import numpy as np
    
    _, _, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, _ = _get_config()
    
    time_axis = np.arange(len(data)) / fs
    ax.plot(time_axis, data, color='#333333', linewidth=0.8)
    ax.set_ylabel(r'$a$ $(m/s^2)$', fontproperties=ENG_FONT, fontsize=8)
    ax.set_xlabel('Time (s)', fontproperties=ENG_FONT, fontsize=8)
    ax.grid(True, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=7)


def plot_frequency_domain(ax, data, fs=FS):
    """绘制频域谱"""
    import numpy as np
    from scipy import signal
    
    _, _, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, _ = _get_config()
    
    f, psd = signal.welch(data, fs=fs, nperseg=int(NFFT/2), 
                          noverlap=int(NFFT/4), nfft=NFFT)
    
    mask = f <= FREQ_LIMIT
    f_limited = f[mask]
    psd_limited = psd[mask]
    
    ax.plot(f_limited, psd_limited, color='#333333', linewidth=0.8)
    ax.set_ylabel('PSD', fontproperties=ENG_FONT, fontsize=8)
    ax.set_xlabel('Freq (Hz)', fontproperties=ENG_FONT, fontsize=8)
    ax.grid(True, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim(0, FREQ_LIMIT)


def plot_frequency_evolution(ax, data, fs=FS):
    """绘制频域演变(时频域)"""
    import numpy as np
    from scipy import signal
    
    SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, get_viridis_color_map = _get_config()
    
    fs_int = int(fs)
    window_total = int(len(data) // fs_int)
    
    psd_list = []
    
    for i in range(window_total):
        start_idx = i * fs_int
        end_idx = (i + 1) * fs_int
        
        if end_idx <= len(data):
            segment = data[start_idx:end_idx]
            f, psd = signal.welch(segment, fs=fs, nperseg=int(fs * 0.8),
                                 noverlap=int(fs * 0.4), nfft=NFFT)
            
            mask = f <= FREQ_LIMIT
            psd_limited = psd[mask]
            psd_list.append(psd_limited)
    
    if not psd_list:
        return
    
    spec_array = np.array(psd_list)
    f_limited = f[f <= FREQ_LIMIT]
    
    cmap_gray = get_viridis_color_map(start_gray=0.2)
    
    im = ax.imshow(spec_array, aspect='auto', origin='lower', cmap=cmap_gray,
                   extent=[0, FREQ_LIMIT, 0, window_total],
                   interpolation='bilinear')
    
    ax.set_ylabel('Time (s)', fontproperties=ENG_FONT, fontsize=8)
    ax.set_xlabel('Freq (Hz)', fontproperties=ENG_FONT, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)


def create_dataset_display_figure(annotation_data, num_samples=30):
    """
    创建数据集展示图(20-50个子图)
    
    Args:
        annotation_data: 标注数据列表
        num_samples: 要显示的样本数 (20-50)
    
    Returns:
        fig: matplotlib图句柄
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, get_viridis_color_map = _get_config()
    
    num_samples = min(max(num_samples, 20), 50)
    
    # 随机选择样本
    if len(annotation_data) > num_samples:
        selected_indices = random.sample(range(len(annotation_data)), num_samples)
        selected_data = [annotation_data[i] for i in selected_indices]
    else:
        selected_data = annotation_data[:num_samples]
    
    # 计算网格布局 (尽量接近正方形)
    num_cols = int(np.ceil(np.sqrt(num_samples)))
    num_rows = int(np.ceil(num_samples / num_cols))
    
    # 创建图像，子图高度为行数*2，宽度为列数*2
    fig, axes = plt.subplots(num_rows, num_cols * 3, figsize=SQUARE_FIG_SIZE)
    
    # 展平axes以便遍历
    if num_rows == 1 and num_cols * 3 == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols * 3 == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(num_rows, num_cols * 3)
    
    # 绘制每个样本
    for idx, sample_data in enumerate(selected_data):
        row = idx // num_cols
        col_base = (idx % num_cols) * 3
        
        ax_time = axes[row, col_base] if num_rows > 1 else axes[col_base]
        ax_freq = axes[row, col_base + 1] if num_rows > 1 else axes[col_base + 1]
        ax_evo = axes[row, col_base + 2] if num_rows > 1 else axes[col_base + 2]
        
        file_path = sample_data['file_path']
        window_index = sample_data['window_index']
        
        data = load_sample_data(file_path, window_index)
        
        plot_time_domain(ax_time, data)
        plot_frequency_domain(ax_freq, data)
        plot_frequency_evolution(ax_evo, data)
    
    # 隐藏未使用的子图
    total_subplots = num_rows * num_cols * 3
    used_subplots = len(selected_data) * 3
    
    for idx in range(used_subplots, total_subplots):
        row = idx // (num_cols * 3)
        col = idx % (num_cols * 3)
        if num_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def create_label_distribution_pie_chart(annotation_data):
    """
    创建标签分布饼状图
    
    Args:
        annotation_data: 标注数据列表
    
    Returns:
        fig: matplotlib图句柄
    """
    import matplotlib.pyplot as plt
    
    SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, get_viridis_color_map = _get_config()
    
    # 统计标签分布
    labels = [item['annotation'] for item in annotation_data]
    label_counts = Counter(labels)
    
    # 创建饼状图
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    
    # 获取标签和计数
    label_names = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # 定义颜色
    colors = ['#8074C8', '#7895C1', '#A8CBDF', '#D6EFF4', '#F2FAFC',
              '#F7FBC9', '#F5EBAE', '#F0C284', '#EF8B67', '#E3625D']
    colors = colors[:len(label_names)]
    
    # 绘制饼状图
    wedges, texts, autotexts = ax.pie(counts, labels=label_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    
    # 设置文字属性
    for text in texts:
        text.set_fontsize(FONT_SIZE)
        text.set_fontproperties(CN_FONT)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(FONT_SIZE)
        autotext.set_fontweight('bold')
    
    ax.set_title('标签分布', fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
    
    return fig


def main():
    """主函数"""
    from src.visualize_tools.utils import PlotLib
    
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    # 标注结果路径
    annotation_path = os.path.join(project_root, "results", "dataset_annotation", "annotation_results.json")
    
    # 加载标注数据
    print("加载标注数据...")
    annotation_data = load_annotation_results(annotation_path)
    print(f"✓ 加载了 {len(annotation_data)} 条标注记录")
    
    # 创建图像1: 数据集展示
    print("\n生成图像1: 数据集样本展示...")
    fig1 = create_dataset_display_figure(annotation_data, num_samples=30)
    print("✓ 数据集样本展示图生成完成")
    
    # 创建图像2: 标签分布饼状图
    print("\n生成图像2: 标签分布饼状图...")
    fig2 = create_label_distribution_pie_chart(annotation_data)
    print("✓ 标签分布饼状图生成完成")
    
    # 使用PlotLib进行显示和保存
    print("\n显示图像...")
    ploter = PlotLib()
    ploter.figs.append(fig1)
    ploter.figs.append(fig2)
    ploter.show()


if __name__ == "__main__":
    main()
