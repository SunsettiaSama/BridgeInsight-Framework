import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from matplotlib.font_manager import FontProperties
from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, ENG_FONT, CN_FONT, get_blue_color_map

CMAP = get_blue_color_map(style='gradient')
COLORS = CMAP(np.linspace(0, 1, 256))
TRAIN_CURVE_COLOR = COLORS[255]
VAL_CURVE_COLOR = COLORS[128]

MLP_RESULT_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module\mlp\mlp_train_result.json"


def load_mlp_results(result_path):
    """加载MLP训练结果"""
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_metrics_by_epoch(training_metadata):
    """从训练元数据中提取各个指标按epoch的变化"""
    epochs = []
    train_loss = []
    train_precision = []
    train_recall = []
    train_f1 = []
    val_loss = []
    val_precision = []
    val_recall = []
    val_f1 = []
    
    for epoch_state in training_metadata['epoch_states']:
        epoch = epoch_state['epoch']
        epochs.append(epoch)
        
        train_metrics = epoch_state['train_metrics']
        train_loss.append(train_metrics['loss'])
        train_precision.append(train_metrics['precision'])
        train_recall.append(train_metrics['recall'])
        train_f1.append(train_metrics['f1'])
        
        val_metrics = epoch_state['val_metrics']
        val_loss.append(val_metrics['loss'])
        val_precision.append(val_metrics['precision'])
        val_recall.append(val_metrics['recall'])
        val_f1.append(val_metrics['f1'])
    
    return {
        'epochs': np.array(epochs),
        'train_loss': np.array(train_loss),
        'train_precision': np.array(train_precision),
        'train_recall': np.array(train_recall),
        'train_f1': np.array(train_f1),
        'val_loss': np.array(val_loss),
        'val_precision': np.array(val_precision),
        'val_recall': np.array(val_recall),
        'val_f1': np.array(val_f1),
    }


def plot_loss_curve(metrics, ploter):
    """绘制损失函数曲线"""
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    
    epochs = metrics['epochs']
    ax.plot(epochs, metrics['train_loss'], color=TRAIN_CURVE_COLOR, linewidth=2.5, marker='o', 
            markersize=4, label='Train Loss', alpha=0.8)
    ax.plot(epochs, metrics['val_loss'], color=VAL_CURVE_COLOR, linewidth=2.5, marker='s', 
            markersize=4, label='Val Loss', alpha=0.8)
    
    ax.set_xlabel('Epoch', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel('Loss', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_title('Training Loss vs Epoch', fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, pad=15)
    ax.legend(loc='best', fontsize=SQUARE_FONT_SIZE - 2, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)
    
    plt.tight_layout()
    ploter.figs.append(fig)


def plot_precision_curve(metrics, ploter):
    """绘制精确度曲线"""
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    
    epochs = metrics['epochs']
    ax.plot(epochs, metrics['train_precision'], color=TRAIN_CURVE_COLOR, linewidth=2.5, marker='o', 
            markersize=4, label='Train Precision', alpha=0.8)
    ax.plot(epochs, metrics['val_precision'], color=VAL_CURVE_COLOR, linewidth=2.5, marker='s', 
            markersize=4, label='Val Precision', alpha=0.8)
    
    ax.set_xlabel('Epoch', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel('Precision', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_title('Precision vs Epoch', fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, pad=15)
    ax.set_ylim((0, 1))
    ax.legend(loc='best', fontsize=SQUARE_FONT_SIZE - 2, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)
    
    plt.tight_layout()
    ploter.figs.append(fig)


def plot_recall_curve(metrics, ploter):
    """绘制召回率曲线"""
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    
    epochs = metrics['epochs']
    ax.plot(epochs, metrics['train_recall'], color=TRAIN_CURVE_COLOR, linewidth=2.5, marker='o', 
            markersize=4, label='Train Recall', alpha=0.8)
    ax.plot(epochs, metrics['val_recall'], color=VAL_CURVE_COLOR, linewidth=2.5, marker='s', 
            markersize=4, label='Val Recall', alpha=0.8)
    
    ax.set_xlabel('Epoch', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel('Recall', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_title('Recall vs Epoch', fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, pad=15)
    ax.set_ylim((0, 1))
    ax.legend(loc='best', fontsize=SQUARE_FONT_SIZE - 2, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)
    
    plt.tight_layout()
    ploter.figs.append(fig)


def plot_f1_curve(metrics, ploter):
    """绘制F1分数曲线"""
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    
    epochs = metrics['epochs']
    ax.plot(epochs, metrics['train_f1'], color=TRAIN_CURVE_COLOR, linewidth=2.5, marker='o', 
            markersize=4, label='Train F1', alpha=0.8)
    ax.plot(epochs, metrics['val_f1'], color=VAL_CURVE_COLOR, linewidth=2.5, marker='s', 
            markersize=4, label='Val F1', alpha=0.8)
    
    ax.set_xlabel('Epoch', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel('F1 Score', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_title('F1 Score vs Epoch', fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, pad=15)
    ax.set_ylim((0, 1))
    ax.legend(loc='best', fontsize=SQUARE_FONT_SIZE - 2, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)
    
    plt.tight_layout()
    ploter.figs.append(fig)


def main():
    """主函数"""
    print("=" * 80)
    print("开始绘制深度学习模型训练结果曲线图")
    print("=" * 80)
    
    if not os.path.exists(MLP_RESULT_PATH):
        print(f"❌ 结果文件不存在：{MLP_RESULT_PATH}")
        raise FileNotFoundError(f"Cannot find result file: {MLP_RESULT_PATH}")
    
    data = load_mlp_results(MLP_RESULT_PATH)
    training_metadata = data.get('training_metadata', {})
    
    if not training_metadata.get('epoch_states'):
        print("❌ 训练数据为空")
        raise ValueError("No training data found in epoch_states")
    
    metrics = extract_metrics_by_epoch(training_metadata)
    
    print(f"\n找到 {len(metrics['epochs'])} 个epoch的训练数据")
    
    ploter = PlotLib()
    
    print("\n正在绘制Loss曲线...")
    plot_loss_curve(metrics, ploter)
    
    print("正在绘制Precision曲线...")
    plot_precision_curve(metrics, ploter)
    
    print("正在绘制Recall曲线...")
    plot_recall_curve(metrics, ploter)
    
    print("正在绘制F1分数曲线...")
    plot_f1_curve(metrics, ploter)
    
    print(f"\n生成了 {len(ploter.figs)} 张曲线图")
    
    ploter.show()
    
    print("\n" + "=" * 80)
    print("所有曲线图绘制完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
