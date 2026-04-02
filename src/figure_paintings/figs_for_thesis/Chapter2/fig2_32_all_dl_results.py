import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.config import SQUARE_FIG_SIZE, SQUARE_FONT_SIZE, ENG_FONT

RESULT_BASE = r"F:\Research\Vibration Characteristics In Cable Vibration\results\training_result\deep_learning_module"

MODEL_CONFIGS = [
    {"name": "MLP",    "key": "mlp",     "path": os.path.join(RESULT_BASE, "mlp",     "mlp_train_result.json")},
    {"name": "RNN",    "key": "rnn",     "path": os.path.join(RESULT_BASE, "rnn",     "rnn_train_result.json")},
    {"name": "LSTM",   "key": "lstm",    "path": os.path.join(RESULT_BASE, "lstm",    "lstm_train_result.json")},
    {"name": "CNN",    "key": "cnn",     "path": os.path.join(RESULT_BASE, "cnn",     "cnn_train_result.json")},
    {"name": "ResCNN", "key": "res_cnn", "path": os.path.join(RESULT_BASE, "res_cnn", "res_cnn_train_result.json")},
]

# 从 config.py 色盘中选取对比度明显的 5 个颜色（跳过 #D6EFF4/#F2FAFC 等极浅色）
MODEL_COLORS = ['#8074C8', '#7895C1', '#F0C284', '#E3625D', '#992224']

MARKERS = ['o', 's', '^', 'D', 'v']



def load_result(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_val_metrics(training_metadata):
    epochs, val_loss, val_precision, val_recall, val_f1 = [], [], [], [], []
    for epoch_state in training_metadata['epoch_states']:
        epochs.append(epoch_state['epoch'])
        vm = epoch_state['val_metrics']
        val_loss.append(vm['loss'])
        val_precision.append(vm['precision'])
        val_recall.append(vm['recall'])
        val_f1.append(vm['f1'])
    return {
        'epochs':        np.array(epochs),
        'val_loss':      np.array(val_loss),
        'val_precision': np.array(val_precision),
        'val_recall':    np.array(val_recall),
        'val_f1':        np.array(val_f1),
    }


def plot_comparison_curve(all_metrics, metric_key, ylabel, title, ploter, ylim=None):
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    for i, (cfg, metrics) in enumerate(zip(MODEL_CONFIGS, all_metrics)):
        ax.plot(
            metrics['epochs'], metrics[metric_key],
            color=MODEL_COLORS[i], linewidth=2.5,
            marker=MARKERS[i], markersize=4,
            label=cfg['name'], alpha=0.85
        )
    ax.set_xlabel('Epoch', labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_ylabel(ylabel, labelpad=10, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE)
    ax.set_title(title, fontproperties=ENG_FONT, fontsize=SQUARE_FONT_SIZE, pad=15)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc='best', fontsize=SQUARE_FONT_SIZE - 4, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)
    plt.tight_layout()
    ploter.figs.append(fig)


def print_best_results(all_metrics):
    print("\n" + "=" * 80)
    print("各网络最优验证集结果（按 Val F1 排序）")
    print("=" * 80)
    print(f"{'Model':<10} {'Best Epoch':>12} {'Val F1':>10} {'Val Precision':>15} {'Val Recall':>12} {'Val Loss':>12}")
    print("-" * 80)
    summary = []
    for cfg, metrics in zip(MODEL_CONFIGS, all_metrics):
        best_idx = int(np.argmax(metrics['val_f1']))
        summary.append({
            'name':      cfg['name'],
            'epoch':     int(metrics['epochs'][best_idx]),
            'val_f1':    metrics['val_f1'][best_idx],
            'val_p':     metrics['val_precision'][best_idx],
            'val_r':     metrics['val_recall'][best_idx],
            'val_loss':  metrics['val_loss'][best_idx],
        })
    summary.sort(key=lambda x: x['val_f1'], reverse=True)
    for r in summary:
        print(f"{r['name']:<10} {r['epoch']:>12} {r['val_f1']:>10.4f} {r['val_p']:>15.4f} {r['val_r']:>12.4f} {r['val_loss']:>12.6f}")
    print("=" * 80)


def main():
    print("=" * 80)
    print("开始绘制所有深度学习模型验证集指标对比曲线图")
    print("=" * 80)

    all_metrics = []
    for cfg in MODEL_CONFIGS:
        if not os.path.exists(cfg['path']):
            raise FileNotFoundError(f"结果文件不存在: {cfg['path']}")
        data = load_result(cfg['path'])
        training_metadata = data.get('training_metadata', {})
        if not training_metadata.get('epoch_states'):
            raise ValueError(f"{cfg['name']} 训练数据为空")
        metrics = extract_val_metrics(training_metadata)
        all_metrics.append(metrics)
        print(f"  {cfg['name']}: 共 {len(metrics['epochs'])} 个 epoch")

    print_best_results(all_metrics)

    ploter = PlotLib()

    print("\n正在绘制 Val F1 对比曲线...")
    plot_comparison_curve(all_metrics, 'val_f1', 'F1 Score', 'Validation F1 Score vs Epoch', ploter, ylim=(0, 1))

    print("正在绘制 Val Precision 对比曲线...")
    plot_comparison_curve(all_metrics, 'val_precision', 'Precision', 'Validation Precision vs Epoch', ploter, ylim=(0, 1))

    print("正在绘制 Val Recall 对比曲线...")
    plot_comparison_curve(all_metrics, 'val_recall', 'Recall', 'Validation Recall vs Epoch', ploter, ylim=(0, 1))

    print("正在绘制 Val Loss 对比曲线...")
    plot_comparison_curve(all_metrics, 'val_loss', 'Loss', 'Validation Loss vs Epoch', ploter)

    print(f"\n共生成 {len(ploter.figs)} 张对比曲线图")
    ploter.show()

    print("\n" + "=" * 80)
    print("所有曲线图绘制完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
