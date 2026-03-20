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


def plot_time_domain(ax, data, fs=FS):
    """绘制时域波形"""
    import numpy as np
    
    time_axis = np.arange(len(data)) / fs
    ax.plot(time_axis, data, color='#333333', linewidth=0.8)
    
    # 隐藏标注文字，但保留坐标轴和边框
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', alpha=0.2, linewidth=0.3)


def plot_frequency_domain(ax, data, fs=FS):
    """绘制频域谱"""
    import numpy as np
    from scipy import signal
    
    f, psd = signal.welch(data, fs=fs, nperseg=int(NFFT/2), 
                          noverlap=int(NFFT/4), nfft=NFFT)
    
    mask = f <= FREQ_LIMIT
    f_limited = f[mask]
    psd_limited = psd[mask]
    
    ax.plot(f_limited, psd_limited, color='#333333', linewidth=0.8)
    ax.set_xlim(0, FREQ_LIMIT)
    
    # 隐藏标注文字，但保留坐标轴和边框
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', alpha=0.2, linewidth=0.3)


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
    
    # 隐藏标注文字，但保留坐标轴和边框
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def create_dataset_display_figure(annotation_data, num_samples=30):
    """
    创建数据集展示图(20-50个子图)
    对齐annotation.py的逻辑：先读文件 → 提取窗口 → 绘图
    
    布局: 每个样本占用2行3列，左侧占2/3，右侧占1/3
    - [0,0]: 时域波形 (左上，占2/3宽度)
    - [1,0]: 频域谱   (左下，占2/3宽度)  
    - [0,1:3], [1,1:3]: 时频域演化 (右侧占1/3宽度)
    
    Args:
        annotation_data: 标注数据列表
        num_samples: 要显示的样本数 (20-50)
    
    Returns:
        fig: matplotlib图句柄
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from collections import defaultdict
    
    SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, get_viridis_color_map = _get_config()
    
    num_samples = min(max(num_samples, 20), 50)
    
    # 随机选择样本
    if len(annotation_data) > num_samples:
        selected_indices = random.sample(range(len(annotation_data)), num_samples)
        selected_data = [annotation_data[i] for i in selected_indices]
    else:
        selected_data = annotation_data[:num_samples]
    
    # 计算网格布局 - 每个样本占2行3列（时域1列占2/3，演化2列占1/3）
    num_cols = int(np.ceil(np.sqrt(num_samples)))
    num_rows = int(np.ceil(num_samples / num_cols))
    
    # 总的子图网格：2*num_rows 行，3*num_cols 列
    total_rows = 2 * num_rows
    total_cols = 3 * num_cols
    
    # 设置列宽比例：左侧占2/3，右侧占1/3
    # 每个样本的比例是 [2, 1, 1]
    width_ratios = [2, 1, 1] * num_cols
    
    fig = plt.figure(figsize=SQUARE_FIG_SIZE)
    
    # 添加总标题（对齐annotation.py风格）
    total_labels = len(set(item['annotation'] for item in selected_data))
    fig_title = f"Dataset Display - {num_samples} Samples ({total_labels} Labels)"
    fig.suptitle(fig_title, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, 
                fontweight='bold', y=0.98)
    
    gs = GridSpec(total_rows, total_cols, figure=fig, hspace=0.3, wspace=0.3, width_ratios=width_ratios)
    
    # 缓存已加载的数据文件，避免重复加载
    file_cache = {}
    
    # 绘制每个样本
    for idx, sample_data in enumerate(selected_data):
        sample_row = idx // num_cols  # 样本在网格中的行
        sample_col = idx % num_cols   # 样本在网格中的列
        
        # 计算在大网格中的位置
        ax_row_start = sample_row * 2
        ax_col_start = sample_col * 3
        
        # 获取子图
        ax_time = fig.add_subplot(gs[ax_row_start, ax_col_start])           # 时域 (上左)
        ax_freq = fig.add_subplot(gs[ax_row_start + 1, ax_col_start])       # 频域 (下左)
        ax_evo = fig.add_subplot(gs[ax_row_start:ax_row_start+2, ax_col_start+1:ax_col_start+3])  # 演化 (右侧)
        
        file_path = sample_data['file_path']
        window_index = sample_data['window_index']
        
        # 对齐annotation.py的逻辑：使用缓存避免重复加载同一文件
        if file_path not in file_cache:
            try:
                from src.data_processer.io_unpacker import UNPACK
                unpacker = UNPACK(init_path=False)
                vibration_data = np.array(unpacker.VIC_DATA_Unpack(file_path))
                file_cache[file_path] = vibration_data
            except Exception as e:
                print(f"⚠ 加载文件失败 {file_path}: {e}")
                continue
        
        vibration_data = file_cache[file_path]
        
        # 提取窗口数据
        start_sample = window_index * WINDOW_SIZE
        end_sample = (window_index + 1) * WINDOW_SIZE
        
        if end_sample > len(vibration_data):
            print(f"⚠ 窗口范围超出数据长度: [{start_sample}, {end_sample}] 超过 {len(vibration_data)}")
            continue
        
        window_data = vibration_data[start_sample:end_sample]
        
        # 绘图
        plot_time_domain(ax_time, window_data)
        plot_frequency_domain(ax_freq, window_data)
        plot_frequency_evolution(ax_evo, window_data)
    
    return fig


def create_label_distribution_pie_chart(annotation_data):
    """
    创建标签分布饼状图 (3D效果、倾斜、使用legend、带有元数据标题)
    - 饼图内显示百分比
    - Legend显示标签名和计数
    - 字体统一使用ENG_FONT
    
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
    
    # 获取标题信息：总样本数和标签类别数
    total_samples = len(annotation_data)
    total_labels = len(label_counts)
    fig_title = f"Label Distribution - {total_samples} Samples, {total_labels} Categories"
    fig.suptitle(fig_title, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, 
                fontweight='bold', y=0.98)
    
    # 获取标签和计数
    label_names = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # 定义颜色
    colors = ['#8074C8', '#7895C1', '#A8CBDF', '#D6EFF4', '#F2FAFC',
              '#F7FBC9', '#F5EBAE', '#F0C284', '#EF8B67', '#E3625D']
    colors = colors[:len(label_names)]
    
    # 绘制饼状图，添加3D效果和倾斜
    # 通过explode参数创建分离效果（类似3D）
    explode = [0.05] * len(label_names)
    
    # 绘制饼状图（不提供labels，避免直接标注）
    wedges, autotexts = ax.pie(
        counts, 
        colors=colors,
        explode=explode,
        startangle=45,  # 倾斜角度
        shadow=True,    # 添加阴影创建3D效果
        textprops={'fontproperties': ENG_FONT, 'fontsize': FONT_SIZE - 2, 'fontweight': 'bold'}
    )
    
    # 设置autotexts（百分比）属性 - 统一字体风格
    total_count = sum(counts)
    for i, autotext in enumerate(autotexts):
        autotext.set_color('white')
        autotext.set_fontsize(FONT_SIZE)
        autotext.set_fontweight('bold')
        autotext.set_fontproperties(ENG_FONT)
        # 更新文本为百分比格式
        percentage = (counts[i] / total_count) * 100
        autotext.set_text(f'{percentage:.1f}%')
    
    # 使用legend展示标签名称和计数信息
    legend_labels = [f'{name}: {count} ({count/sum(counts)*100:.1f}%)' 
                     for name, count in zip(label_names, counts)]
    legend = ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=FONT_SIZE - 2, frameon=True, fancybox=True, shadow=True)
    
    # 统一legend的字体
    for text in legend.get_texts():
        text.set_fontproperties(ENG_FONT)
        text.set_fontsize(FONT_SIZE - 2)
    
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
