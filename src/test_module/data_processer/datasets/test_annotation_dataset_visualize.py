import sys
import os
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

current_dir  = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig

CONFIG_PATH = "config/train/datasets/annotation_dataset.yaml"

SAMPLES_PER_CLASS = 1

def _get_config():
    """延迟导入配置以避免numpy兼容性问题"""
    from src.figure_paintings.figs_for_thesis.config import (
        SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE,
        CN_FONT, ENG_FONT, get_viridis_color_map
    )
    return SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, get_viridis_color_map


def load_config(config_path: str) -> AnnotationDatasetConfig:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['enable_preload_cache'] = False
    return AnnotationDatasetConfig(**cfg)


def _build_path_to_class(dataset: AnnotationDataset) -> dict:
    """构建 resolved_path_str → class_id 的反向映射

    直接复用 dataset._idx_to_annotation（与 full_file_paths 顺序严格对齐），
    避免在测试侧重复实现路径匹配逻辑。
    """
    mapping: dict = {}
    unmatched = 0
    for fp, anno in zip(dataset.full_file_paths, dataset._idx_to_annotation):
        if anno is not None:
            mapping[str(Path(fp).resolve())] = int(anno["class_id"])
        else:
            unmatched += 1

    if unmatched:
        print(f"[警告] _build_path_to_class: {unmatched} 个路径无法匹配到任何标注")

    return mapping


def collect_indices_by_label(dataset: AnnotationDataset, n_per_class: int = SAMPLES_PER_CLASS) -> dict:
    """
    按 class_id 收集前 n_per_class 个样本的全局索引
    返回 {class_id: [idx, ...]}
    """
    path_to_class = _build_path_to_class(dataset)
    label_to_indices: dict = {}

    for idx, file_path in enumerate(dataset.full_file_paths):
        resolved = str(Path(file_path).resolve())
        cid = path_to_class.get(resolved)
        if cid is None:
            continue
        bucket = label_to_indices.setdefault(cid, [])
        if len(bucket) < n_per_class:
            bucket.append(idx)

    return label_to_indices


def test_visualize_single_samples(dataset: AnnotationDataset):
    """对每个类别分别可视化 SAMPLES_PER_CLASS 个样本"""
    print("\n" + "=" * 60)
    print("单样本可视化（每类各取若干样本）")
    print("=" * 60)

    label_to_indices = collect_indices_by_label(dataset, n_per_class=SAMPLES_PER_CLASS)
    print(f"找到标签：{sorted(label_to_indices.keys())}")

    for class_id in sorted(label_to_indices.keys()):
        indices = label_to_indices[class_id]
        print(f"\n  类别 {class_id}：展示 {len(indices)} 个样本（索引 {indices}）")
        for idx in indices:
            dataset.visualize_sample(idx)


def test_visualize_batch(dataset: AnnotationDataset):
    """跨类别批量可视化"""
    print("\n" + "=" * 60)
    print("批量可视化（每类取1个样本，拼一张图）")
    print("=" * 60)

    label_to_indices = collect_indices_by_label(dataset, n_per_class=1)
    batch_indices = [v[0] for v in label_to_indices.values() if v]
    print(f"批次索引：{batch_indices}")

    dataset.visualize_batch(batch_indices)


def print_dataset_summary(dataset: AnnotationDataset):
    print("\n" + "=" * 60)
    print("数据集摘要")
    print("=" * 60)
    print(f"  总样本数   : {len(dataset.full_file_paths)}")
    print(f"  类别数     : {dataset.get_num_classes()}")
    print(f"  任务类型   : {dataset.annotation_config.task_type}")
    print(f"  窗口大小   : {dataset.annotation_config.window_size}")
    print(f"  归一化     : {dataset.annotation_config.normalize}")

    label_to_indices = collect_indices_by_label(dataset, n_per_class=9999)
    print("\n  各类别样本数：")
    for cid in sorted(label_to_indices.keys()):
        print(f"    类别 {cid} : {len(label_to_indices[cid])} 个")
    print("=" * 60)

def create_split_ratio_pie_chart(dataset: AnnotationDataset):
    """
    创建数据集分割比例的饼状图
    分别展示训练集和验证集的四类样本占比
    
    参考 fig2_27_dataset_display.py 的实现逻辑
    """
    import numpy as np
    
    SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, FONT_SIZE, LABEL_FONT_SIZE, CN_FONT, ENG_FONT, get_viridis_color_map = _get_config()
    
    path_to_class = _build_path_to_class(dataset)

    train_labels = []
    train_miss = 0
    for fp in dataset.train_paths:
        resolved = str(Path(fp).resolve())
        cid = path_to_class.get(resolved)
        if cid is not None:
            train_labels.append(cid)
        else:
            train_miss += 1

    val_labels = []
    val_miss = 0
    for fp in dataset.val_paths:
        resolved = str(Path(fp).resolve())
        cid = path_to_class.get(resolved)
        if cid is not None:
            val_labels.append(cid)
        else:
            val_miss += 1

    print(f"  [诊断] train_paths={len(dataset.train_paths)}, 命中={len(train_labels)}, 未命中={train_miss}")
    print(f"  [诊断] val_paths  ={len(dataset.val_paths)},  命中={len(val_labels)},  未命中={val_miss}")

    if not train_labels:
        raise RuntimeError("训练集标签全部未命中，请检查路径匹配逻辑")
    if not val_labels:
        raise RuntimeError("验证集标签全部未命中，请检查 auto_split 配置或路径匹配逻辑")
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    
    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    fig_title = "Dataset Split Distribution"
    fig.suptitle(fig_title, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, 
                fontweight='bold', y=0.98)
    
    colors = ['#8074C8', '#7895C1', '#A8CBDF', '#D6EFF4']
    
    train_keys = sorted(train_counts.keys())
    train_values = [train_counts[cid] for cid in train_keys]
    train_labels_display = [f"Class {cid}" for cid in train_keys]
    
    val_keys = sorted(val_counts.keys())
    val_values = [val_counts[cid] for cid in val_keys]
    val_labels_display = [f"Class {cid}" for cid in val_keys]
    
    explode = [0.05] * len(train_labels_display)
    
    wedges_train, _, autotexts_train = axes[0].pie(
        train_values,
        colors=colors[:len(train_labels_display)],
        explode=explode,
        startangle=45,
        shadow=True,
        autopct='%1.1f%%',
        textprops={'fontproperties': ENG_FONT, 'fontsize': FONT_SIZE - 2, 'fontweight': 'bold'}
    )
    
    axes[0].set_title(f"Train Set ({train_total} samples)", fontproperties=ENG_FONT, 
                     fontsize=FONT_SIZE, fontweight='bold', pad=20)
    
    for autotext in autotexts_train:
        autotext.set_color('white')
        autotext.set_fontsize(FONT_SIZE)
        autotext.set_fontweight('bold')
        autotext.set_fontproperties(ENG_FONT)
    
    legend_labels_train = [f'{name}: {count} ({count/train_total*100:.1f}%)' 
                          for name, count in zip(train_labels_display, train_values)]
    legend_train = axes[0].legend(legend_labels_train, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                                 fontsize=FONT_SIZE - 2, frameon=True, fancybox=True, shadow=True)
    for text in legend_train.get_texts():
        text.set_fontproperties(ENG_FONT)
        text.set_fontsize(FONT_SIZE - 2)
    
    explode_val = [0.05] * len(val_labels_display)
    
    wedges_val, _, autotexts_val = axes[1].pie(
        val_values,
        colors=colors[:len(val_labels_display)],
        explode=explode_val,
        startangle=45,
        shadow=True,
        autopct='%1.1f%%',
        textprops={'fontproperties': ENG_FONT, 'fontsize': FONT_SIZE - 2, 'fontweight': 'bold'}
    )
    
    axes[1].set_title(f"Validation Set ({val_total} samples)", fontproperties=ENG_FONT, 
                     fontsize=FONT_SIZE, fontweight='bold', pad=20)
    
    for autotext in autotexts_val:
        autotext.set_color('white')
        autotext.set_fontsize(FONT_SIZE)
        autotext.set_fontweight('bold')
        autotext.set_fontproperties(ENG_FONT)
    
    legend_labels_val = [f'{name}: {count} ({count/val_total*100:.1f}%)' 
                        for name, count in zip(val_labels_display, val_values)]
    legend_val = axes[1].legend(legend_labels_val, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                               fontsize=FONT_SIZE - 2, frameon=True, fancybox=True, shadow=True)
    for text in legend_val.get_texts():
        text.set_fontproperties(ENG_FONT)
        text.set_fontsize(FONT_SIZE - 2)
    
    return fig


if __name__ == "__main__":
    print("加载 AnnotationDataset 配置...")
    config  = load_config(CONFIG_PATH)
    dataset = AnnotationDataset(config)

    print_dataset_summary(dataset)

    test_visualize_single_samples(dataset)
    test_visualize_batch(dataset)

    print("\n生成数据集分割比例饼状图...")
    fig_split = create_split_ratio_pie_chart(dataset)
    
    print("\n整合所有图表...")
    visualizer = dataset._get_visualizer()
    visualizer.figs.insert(0, fig_split)
    
    print(f"✓ 总共生成了 {len(visualizer.figs)} 个图表")
    visualizer.show()


