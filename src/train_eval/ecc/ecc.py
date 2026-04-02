import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader
import yaml
import json
import os
import logging
from datetime import datetime

from src.visualize_tools.utils import PlotLib
from src.identifier.cable_analysis_methods.base_mode_calculator import Cal_Mount
from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig
from src.figure_paintings.figs_for_thesis.config import (
    ENG_FONT, CN_FONT, SQUARE_FIG_SIZE, REC_FIG_SIZE, SQUARE_FONT_SIZE,
    get_viridis_color_map, get_blue_color_map
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== 全局配置与常量定义 =====================
CONFIG_PATH = "config/train/datasets/annotation_dataset.yaml"
SAVE_DIR = r"F:\Research\Vibration Characteristics In Cable Vibration\results\ecc_results"
RESULT_JSON_PATH = os.path.join(SAVE_DIR, 'ecc_search_results.json')
RESULT_TXT_PATH = os.path.join(SAVE_DIR, 'result.txt')

# 二分类标签（label=1/3为过渡态，排除；仅保留0/1）
LABEL_IDS = [0, 1]
LABELS = ['Normal Vibration', 'VIV']

# 最优参数（参数搜索后手动更新）
BEST_THRESHOLD = 0.1   # ECC阈值：次大峰/最大峰 < threshold → VIV
BEST_SIGMA = 0.1       # 起振阈值：RMS < sigma → 直接判定为Normal

# 参数搜索网格（一维，仅搜索 threshold）
ECC_PARAM_GRID = {
    'threshold': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# 全量数据拿来进行验证
TRAIN_RATIO = 0.99

# 强制重新搜索（True：忽略缓存；False：有缓存则直接绘图）
FORCE_RECOMPUTE = False

CMAP = get_blue_color_map()
CMAP_CURVE = get_blue_color_map(style='gradient')

# ===================== 数据集加载函数 =====================
def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    logger.info(f"加载数据集配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config_dict['auto_split'] = True
    config_dict['split_ratio'] = TRAIN_RATIO
    config = AnnotationDatasetConfig(**config_dict)
    logger.info("数据集配置加载完成（auto_split=True）")
    return config


def create_dataloaders(
    dataset_config: AnnotationDatasetConfig,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True
):
    logger.info("创建数据集...")
    dataset = AnnotationDataset(dataset_config)
    logger.info(f"总样本数：{len(dataset)}")
    logger.info(f"类别数：{dataset.get_num_classes()}")

    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    logger.info(f"训练集大小：{len(train_dataset)}")
    logger.info(f"验证集大小：{len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, dataset.get_num_classes()


def extract_data_from_dataloader(dataloader):
    features_list = []
    labels_list = []
    with tqdm(total=len(dataloader), desc="提取数据") as pbar:
        for batch_data, batch_labels in dataloader:
            features_list.append(batch_data.numpy())
            labels_list.append(batch_labels.numpy())
            pbar.update(1)
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def filter_binary_samples(features: np.ndarray, labels: np.ndarray):
    """过滤仅保留label=0和label=2的样本（剔除过渡态1/3）"""
    mask = np.isin(labels, LABEL_IDS)
    kept = int(mask.sum())
    total = len(labels)
    logger.info(f"二分类过滤：{total} → {kept} 个样本（保留label 0/2）")
    return features[mask], labels[mask]


# ===================== ECC核心分类函数 =====================
def ecc_classify(sig: np.ndarray, mount: Cal_Mount, threshold: float, sigma: float) -> int:
    """
    ECC分类：先判断RMS是否超过起振阈值sigma，再用ECC阈值判断VIV
    - RMS < sigma → Normal(0)
    - 次大峰/最大峰 < threshold → VIV(2)
    - 否则 → Normal(0)
    """
    sig = np.array(sig, dtype=np.float64)
    rms = np.sqrt(np.mean(sig ** 2))
    if rms < sigma:
        return 0

    fx, pxxden = signal.welch(sig, fs=50, nfft=65536, nperseg=2048, noverlap=1)
    fxtopk, pxxtopk, _, _ = mount.peaks(fx, pxxden, return_intervals=True)
    pxxtopk = np.array(pxxtopk, dtype=np.float64)

    if len(pxxtopk) < 2:
        return 0

    pxxtopk_sorted = sorted(pxxtopk)
    return 2 if (pxxtopk_sorted[-2] / pxxtopk_sorted[-1]) < threshold else 0


# ===================== 指标计算 =====================
def calculate_metrics(y_true, y_pred):
    precision_per_class = precision_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    recall_per_class    = recall_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    f1_per_class        = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)

    return {
        'per_class': {
            LABELS[i]: {
                'Precision': float(precision_per_class[i]),
                'Recall':    float(recall_per_class[i]),
                'F1':        float(f1_per_class[i])
            }
            for i in range(len(LABELS))
        },
        'weighted': {
            'Precision': float(precision_score(y_true, y_pred, labels=LABEL_IDS, average='weighted', zero_division=0)),
            'Recall':    float(recall_score(y_true, y_pred, labels=LABEL_IDS, average='weighted', zero_division=0)),
            'F1':        float(f1_score(y_true, y_pred, labels=LABEL_IDS, average='weighted', zero_division=0))
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    }


def calculate_metrics_for_params(
    threshold: float,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    mount: Cal_Mount
):
    true_labels = []
    pred_labels = []

    for data, label in zip(val_features, val_labels):
        sig = data[:, 0] if len(data.shape) > 1 else data
        true_labels.append(int(label))
        pred_labels.append(ecc_classify(sig, mount, threshold, BEST_SIGMA))

    metrics = calculate_metrics(true_labels, pred_labels)
    return metrics, metrics['weighted']['F1']


# ===================== 参数搜索 =====================
def ecc_hyperparameter_search(
    val_features: np.ndarray,
    val_labels: np.ndarray,
    mount: Cal_Mount,
    param_grid: dict = None
):
    """一维参数搜索：threshold"""
    if param_grid is None:
        param_grid = ECC_PARAM_GRID

    threshold_values = param_grid['threshold']
    total = len(threshold_values)

    logger.info("=" * 60)
    logger.info(f"开始ECC一维参数搜索，组合数：{total}")
    logger.info(f"threshold: {threshold_values}")
    logger.info(f"固定 sigma={BEST_SIGMA}")
    logger.info("=" * 60)

    search_results = []
    param_metrics = {}  # key: threshold
    best_f1 = 0.0
    best_params = None

    with tqdm(total=total, desc="ECC参数搜索") as pbar:
        for threshold in threshold_values:
            metrics, f1 = calculate_metrics_for_params(
                threshold, val_features, val_labels, mount
            )
            param_metrics[threshold] = metrics

            result = {
                'params': {'threshold': threshold},
                'precision': metrics['weighted']['Precision'],
                'recall':    metrics['weighted']['Recall'],
                'f1':        f1
            }
            search_results.append(result)

            if f1 > best_f1:
                best_f1 = f1
                best_params = {'threshold': threshold}
            pbar.update(1)

    logger.info(f"搜索完成！最优参数: {best_params}，最优F1: {best_f1:.4f}")
    return best_params, search_results, param_metrics


# ===================== 绘图函数 =====================
def plot_ecc_f1_curve(search_results: list, param_grid: dict, ploter: PlotLib):
    """绘制 threshold vs F1 的一维曲线图"""
    colors = CMAP_CURVE(np.linspace(0, 1, 256))
    curve_color = colors[255]

    threshold_values = param_grid['threshold']
    f1_scores = np.array([r['f1'] for r in search_results])

    x_positions = np.arange(len(threshold_values))
    param_str_values = [str(t) for t in threshold_values]

    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = threshold_values[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\n{'=' * 80}")
    print(f"【ECC - 最优参数】")
    print(f"{'=' * 80}")
    print(f"  threshold = {best_threshold}")
    print(f"  sigma     = {BEST_SIGMA}  (固定)")
    print(f"  Best Weighted F1 = {best_f1:.4f}")
    print(f"{'=' * 80}\n")

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)

    ax.plot(x_positions, f1_scores, color=curve_color, linewidth=3, marker='o', markersize=8)
    ax.fill_between(x_positions, f1_scores, alpha=0.3, color=curve_color)

    ax.set_xlabel(r"$C'_{ECC}$ (threshold)", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('F1 Score', labelpad=10, fontproperties=ENG_FONT)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(param_str_values, rotation=45, ha='right', fontproperties=ENG_FONT)
    ax.set_ylim((0, 1))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelsize=SQUARE_FONT_SIZE - 4)

    plt.tight_layout()
    ploter.figs.append(fig)

    save_path = os.path.join(SAVE_DIR, 'ecc_f1_curve.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"F1曲线图已保存：{save_path}")
    return fig


def plot_ecc_confusion_matrix(best_params: dict, f1_value: float, cm: np.ndarray, ploter: PlotLib):
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=CMAP,
        xticklabels=LABELS, yticklabels=LABELS,
        ax=ax, cbar=False, annot_kws={'size': 20}
    )
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('True Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_title(
        f"threshold={best_params['threshold']}\nWeighted F1: {f1_value:.3f}",
        pad=20, fontproperties=ENG_FONT
    )
    plt.tight_layout()
    ploter.figs.append(fig)

    save_path = os.path.join(SAVE_DIR, f"confusion_matrix_th{best_params['threshold']}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"混淆矩阵图已保存：{save_path}")


# ===================== 缓存读取 =====================
def load_ecc_cache():
    if not os.path.exists(RESULT_JSON_PATH):
        return None, None, None
    logger.info(f"发现缓存文件，直接加载：{RESULT_JSON_PATH}")
    with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best_params    = data['best_params']
    search_results = data['search_results']
    param_metrics  = {
        float(k): {
            'weighted':         v['weighted'],
            'per_class':        v['per_class'],
            'confusion_matrix': np.array(v['confusion_matrix'])
        }
        for k, v in data['param_metrics'].items()
    }
    logger.info(f"缓存加载完成，最优参数：{best_params}")
    return best_params, search_results, param_metrics


# ===================== 结果存储 =====================
def save_ecc_search_results(search_results: list, best_params: dict, param_metrics: dict):
    os.makedirs(SAVE_DIR, exist_ok=True)

    result_data = {
        'best_params': best_params,
        'search_results': [
            {
                'params': r['params'],
                'precision': r['precision'],
                'recall':    r['recall'],
                'f1':        r['f1']
            }
            for r in search_results
        ],
        'param_metrics': {
            str(k): {
                'weighted':        v['weighted'],
                'per_class':       v['per_class'],
                'confusion_matrix': v['confusion_matrix'].tolist()
            }
            for k, v in param_metrics.items()
        }
    }

    with open(RESULT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    logger.info(f"ECC搜索结果已保存：{RESULT_JSON_PATH}")


def write_result_file(best_params: dict, best_metrics: dict, search_results: list, dataset_size: int):
    result_path = os.path.join(SAVE_DIR, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ECC参数搜索结果报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集配置: {CONFIG_PATH}\n")
        f.write(f"有效样本数量（label 0/2）: {dataset_size}\n")
        f.write(f"搜索网格: threshold={ECC_PARAM_GRID['threshold']}\n")
        f.write(f"固定参数: sigma={BEST_SIGMA}\n")
        f.write(f"最优参数: threshold={best_params['threshold']}\n")
        f.write("=" * 80 + "\n\n")

        f.write("【最优参数详细指标】\n")
        f.write(f"加权Precision: {best_metrics['weighted']['Precision']:.4f}\n")
        f.write(f"加权Recall:    {best_metrics['weighted']['Recall']:.4f}\n")
        f.write(f"加权F1 Score:  {best_metrics['weighted']['F1']:.4f}\n")
        f.write("\n类别级指标:\n")
        for cls_name, cls_m in best_metrics['per_class'].items():
            f.write(f"  {cls_name}: P={cls_m['Precision']:.4f}  R={cls_m['Recall']:.4f}  F1={cls_m['F1']:.4f}\n")
        cm = best_metrics['confusion_matrix']
        f.write("\n混淆矩阵:\n")
        f.write(f"          预测：{LABELS[0]}  {LABELS[1]}\n")
        f.write(f"真实：{LABELS[0]}  {cm[0,0]}      {cm[0,1]}\n")
        f.write(f"真实：{LABELS[1]}  {cm[1,0]}      {cm[1,1]}\n")
        f.write("=" * 80 + "\n\n")

        f.write("【所有参数组合搜索结果】\n")
        f.write(f"{'threshold':<12} {'加权P':<12} {'加权R':<12} {'加权F1':<12}\n")
        f.write("-" * 50 + "\n")
        for r in search_results:
            p = r['params']
            f.write(f"{p['threshold']:<12.2f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}\n")

    logger.info(f"结果文件已保存：{result_path}")


# ===================== 最优参数评估 =====================
def evaluate_best_params(config_path: str = CONFIG_PATH):
    """基于BEST_THRESHOLD和BEST_SIGMA评估最优参数"""
    logger.info("=" * 60)
    logger.info(f"评估最优参数：threshold={BEST_THRESHOLD}, sigma={BEST_SIGMA}")
    logger.info("=" * 60)

    dataset_config = load_dataset_config(config_path)
    _, val_dataloader, num_classes = create_dataloaders(dataset_config, batch_size=16)
    logger.info(f"数据集加载完成，类别数：{num_classes}")

    val_features, val_labels = extract_data_from_dataloader(val_dataloader)
    val_features, val_labels = filter_binary_samples(val_features, val_labels)

    mount = Cal_Mount()
    metrics, f1 = calculate_metrics_for_params(
        BEST_THRESHOLD, val_features, val_labels, mount
    )

    logger.info(f"Weighted F1={metrics['weighted']['F1']:.4f}  P={metrics['weighted']['Precision']:.4f}  R={metrics['weighted']['Recall']:.4f}")
    for cls_name, cls_m in metrics['per_class'].items():
        logger.info(f"  [{cls_name}] P={cls_m['Precision']:.3f}  R={cls_m['Recall']:.3f}  F1={cls_m['F1']:.3f}")

    cm = metrics['confusion_matrix']
    ploter = PlotLib()
    best_params = {'threshold': BEST_THRESHOLD}
    plot_ecc_confusion_matrix(best_params, f1, cm, ploter)
    ploter.show()

    return metrics


# ===================== 主流程 =====================
def run_ecc_param_search(config_path: str = CONFIG_PATH, param_grid: dict = None):
    """主流程：优先从缓存加载 → 否则搜索 → 绘图"""
    if param_grid is None:
        param_grid = ECC_PARAM_GRID

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_params, search_results, param_metrics = (None, None, None) if FORCE_RECOMPUTE else load_ecc_cache()

    if best_params is None:
        logger.info("=" * 60)
        logger.info("开始ECC参数搜索")
        logger.info("=" * 60)

        dataset_config = load_dataset_config(config_path)
        _, val_dataloader, num_classes = create_dataloaders(dataset_config, batch_size=16)
        logger.info(f"数据集加载完成，类别数：{num_classes}")

        val_features, val_labels = extract_data_from_dataloader(val_dataloader)
        val_features, val_labels = filter_binary_samples(val_features, val_labels)
        dataset_size = len(val_labels)

        mount = Cal_Mount()

        best_params, search_results, param_metrics = ecc_hyperparameter_search(
            val_features, val_labels, mount, param_grid
        )

        save_ecc_search_results(search_results, best_params, param_metrics)

        best_key     = best_params['threshold']
        best_metrics = param_metrics[best_key]
        write_result_file(best_params, best_metrics, search_results, dataset_size)
    else:
        logger.info("使用缓存结果，跳过参数搜索直接绘图")

    best_key     = best_params['threshold']
    best_metrics = param_metrics[best_key]
    best_f1      = best_metrics['weighted']['F1']

    ploter = PlotLib()
    plot_ecc_f1_curve(search_results, param_grid, ploter)
    plot_ecc_confusion_matrix(best_params, best_f1, best_metrics['confusion_matrix'], ploter)
    ploter.show()

    logger.info("=" * 60)
    logger.info(f"ECC参数搜索完成  最优参数: {best_params}  F1: {best_f1:.4f}")
    logger.info("=" * 60)

    return best_params, search_results, param_metrics


if __name__ == "__main__":
    best_params, search_results, param_metrics = run_ecc_param_search()
