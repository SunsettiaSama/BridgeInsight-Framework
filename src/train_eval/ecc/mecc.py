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
from src.data_processer.preprocess.get_data_vib import (
    _load_dominant_freq_statistics,
    _calculate_freq_threshold,
)
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

# ===================== 全局配置与常量 =====================
CONFIG_PATH      = "config/train/datasets/annotation_dataset.yaml"
SAVE_DIR         = r"F:\Research\Vibration Characteristics In Cable Vibration\results\mecc_results"
RESULT_JSON_PATH = os.path.join(SAVE_DIR, 'mecc_search_results.json')

# 二分类标签：0=一般振动，1=涡激共振(VIV)
LABEL_IDS = [0, 1]
LABELS    = ['Normal', 'VIV']

# 固定参数（不参与搜索）
SIGMA_0 = 0.1

# 全量数据用于验证
VAL_RATIO = 0.01

# 强制重新搜索（True：忽略缓存；False：有缓存则直接绘图）
FORCE_RECOMPUTE = False

# 二参数联合搜索网格：k_viv × C_viv
PARAM_GRID = {
    'k_viv': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'C_viv': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

CMAP         = get_blue_color_map()
CMAP_CONTOUR = get_blue_color_map(style='gradient')


# ===================== 主频 95 分位阈值（来自统计，非超参数）=====================
def compute_freq_p95() -> float:
    stats = _load_dominant_freq_statistics()
    freq_p95 = _calculate_freq_threshold(stats)
    logger.info(f"主导模态 95% 分位阈值（来自统计）：{freq_p95:.4f} Hz")
    return freq_p95


# ===================== 数据集加载 =====================
def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    logger.info(f"加载数据集配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config_dict['auto_split'] = True
    config_dict['split_ratio'] = VAL_RATIO
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
    val_dataset   = dataset.get_val_dataset()
    logger.info(f"训练集大小：{len(train_dataset)}")
    logger.info(f"验证集大小：{len(val_dataset)}")

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_dl   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,         num_workers=num_workers)
    return train_dl, val_dl, dataset.get_num_classes()


def extract_data_from_dataloader(dataloader):
    features_list, labels_list = [], []
    with tqdm(total=len(dataloader), desc="提取数据") as pbar:
        for batch_data, batch_labels in dataloader:
            features_list.append(batch_data.numpy())
            labels_list.append(batch_labels.numpy())
            pbar.update(1)
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def filter_binary_samples(features: np.ndarray, labels: np.ndarray):
    mask = np.isin(labels, LABEL_IDS)
    kept = int(mask.sum())
    logger.info(f"二分类过滤：{len(labels)} → {kept} 个样本（保留 label 0/1）")
    return features[mask], labels[mask]


# ===================== MECC 分类核心 =====================
def mecc_classify(
    sig: np.ndarray,
    f0: float,
    k_viv: int,
    C_viv: float,
    freq_p95: float,
) -> int:
    sig = np.array(sig, dtype=np.float64)
    rms = np.sqrt(np.mean((sig - np.mean(sig)) ** 2))
    if rms < SIGMA_0:
        return 0

    fx, Pxx = signal.welch(sig, fs=50, nfft=512, nperseg=512, noverlap=256)
    E1      = np.max(Pxx)
    f_major = fx[np.argmax(Pxx)]

    # 主导模态低于 95% 分位阈值 → 随机振动，判为 Normal
    if f_major < freq_p95:
        return 0

    left    = f_major - k_viv * f0
    right   = f_major + k_viv * f0
    Pxx_out = Pxx[(fx < left) | (fx > right)]
    Ek      = np.max(Pxx_out) if len(Pxx_out) > 0 else 0.0
    mecc    = Ek / E1 if E1 != 0.0 else 1.0

    return 1 if mecc < C_viv else 0


# ===================== 二分类指标计算 =====================
def _compute_binary_metrics(y_true, y_pred) -> dict:
    precision_arr = precision_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    recall_arr    = recall_score(   y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    f1_arr        = f1_score(       y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)

    return {
        'per_class': {
            LABELS[i]: {
                'Precision': float(precision_arr[i]),
                'Recall':    float(recall_arr[i]),
                'F1':        float(f1_arr[i])
            }
            for i in range(len(LABELS))
        },
        'weighted': {
            'Precision': float(precision_score(y_true, y_pred, labels=LABEL_IDS, average='weighted', zero_division=0)),
            'Recall':    float(recall_score(   y_true, y_pred, labels=LABEL_IDS, average='weighted', zero_division=0)),
            'F1':        float(f1_score(       y_true, y_pred, labels=LABEL_IDS, average='weighted', zero_division=0))
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    }


def _calc_metrics(
    k_viv: int,
    C_viv: float,
    features: np.ndarray,
    labels: np.ndarray,
    f0: float,
    freq_p95: float,
):
    true_labels, pred_labels = [], []
    for data, label in zip(features, labels):
        sig  = data[:, 0] if data.ndim > 1 else data
        pred = mecc_classify(sig, f0, k_viv, C_viv, freq_p95)
        true_labels.append(int(label))
        pred_labels.append(pred)

    metrics = _compute_binary_metrics(true_labels, pred_labels)
    return metrics, metrics['weighted']['F1']


# ===================== 二参数联合搜索 =====================
def mecc_hyperparameter_search(
    features: np.ndarray,
    labels: np.ndarray,
    f0: float,
    freq_p95: float,
    param_grid: dict = None
):
    if param_grid is None:
        param_grid = PARAM_GRID

    k_viv_vals = param_grid['k_viv']
    C_viv_vals = param_grid['C_viv']
    total      = len(k_viv_vals) * len(C_viv_vals)

    logger.info("=" * 60)
    logger.info(f"开始 MECC 全数据集二分类二参数联合搜索，组合数：{total}")
    logger.info(f"sigma_0（固定）: {SIGMA_0}")
    logger.info(f"freq_p95（固定）: {freq_p95:.4f} Hz")
    logger.info(f"k_viv : {k_viv_vals}")
    logger.info(f"C_viv : {C_viv_vals}")
    logger.info(f"标签  : {LABEL_IDS} → {LABELS}")
    logger.info("=" * 60)

    search_results = []
    param_metrics  = {}
    best_f1        = 0.0
    best_params    = None

    with tqdm(total=total, desc="MECC联合搜索") as pbar:
        for k_viv in k_viv_vals:
            for C_viv in C_viv_vals:
                metrics, f1 = _calc_metrics(k_viv, C_viv, features, labels, f0, freq_p95)
                param_metrics[(k_viv, C_viv)] = metrics
                result = {
                    'params':    {'k_viv': k_viv, 'C_viv': C_viv},
                    'precision': metrics['weighted']['Precision'],
                    'recall':    metrics['weighted']['Recall'],
                    'f1':        f1
                }
                search_results.append(result)
                if f1 > best_f1:
                    best_f1     = f1
                    best_params = {'k_viv': k_viv, 'C_viv': C_viv}
                pbar.update(1)

    logger.info(f"搜索完成  最优参数：{best_params}  Weighted F1：{best_f1:.4f}")
    return best_params, search_results, param_metrics


# ===================== 绘图 =====================
def plot_mecc_f1_contour(
    search_results: list,
    param_grid: dict,
    ploter: PlotLib
):
    k_viv_vals = param_grid['k_viv']
    C_viv_vals = param_grid['C_viv']

    f1_matrix = np.full((len(k_viv_vals), len(C_viv_vals)), np.nan)
    for r in search_results:
        p  = r['params']
        ki = k_viv_vals.index(p['k_viv'])
        ci = C_viv_vals.index(p['C_viv'])
        f1_matrix[ki, ci] = r['f1']

    X, Y = np.meshgrid(C_viv_vals, k_viv_vals)

    best_idx         = np.nanargmax(f1_matrix)
    best_ki, best_ci = np.unravel_index(best_idx, f1_matrix.shape)
    best_k           = k_viv_vals[best_ki]
    best_C           = C_viv_vals[best_ci]
    best_f1          = f1_matrix[best_ki, best_ci]

    print(f"\n{'=' * 80}")
    print(f"【MECC - 最优参数】")
    print(f"{'=' * 80}")
    print(f"  k_viv   = {best_k}")
    print(f"  C_viv   = {best_C}")
    print(f"  sigma_0 = {SIGMA_0}  (固定)")
    print(f"  Best Weighted F1 = {best_f1:.4f}")
    print(f"{'=' * 80}\n")

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    cf = ax.contourf(X, Y, f1_matrix, cmap=CMAP_CONTOUR, levels=20, alpha=1.0)
    ax.contour(X, Y, f1_matrix, levels=20, colors='black', linewidths=0.5, alpha=0.3)

    ax.scatter([best_C], [best_k], color='red', s=80, zorder=5,
               label=f'Best F1={best_f1:.3f}')
    ax.legend(prop=ENG_FONT, loc='upper right')
    ax.set_xlabel(r"$C_{MECC}$", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel(r"$k_{MECC}$",                  labelpad=10, fontproperties=ENG_FONT)
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label('Weighted F1 Score', fontproperties=ENG_FONT)
    cbar.ax.tick_params(labelsize=SQUARE_FONT_SIZE - 4)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(ENG_FONT)

    plt.tight_layout()
    ploter.figs.append(fig)

    save_path = os.path.join(SAVE_DIR, 'mecc_contour_viv_params.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"等高线图已保存：{save_path}")
    return fig


def _plot_confusion_matrix(
    best_params: dict,
    f1_value: float,
    cm: np.ndarray,
    save_name: str,
    ploter: PlotLib
):
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=CMAP,
        xticklabels=LABELS, yticklabels=LABELS,
        ax=ax, cbar=False, annot_kws={'size': 20}
    )
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('True Label',      labelpad=10, fontproperties=ENG_FONT)
    param_str = f"k_viv={best_params['k_viv']}, C_viv={best_params['C_viv']}"
    ax.set_title(
        f"{param_str}\nWeighted F1: {f1_value:.3f}",
        pad=20, fontproperties=ENG_FONT
    )
    plt.tight_layout()
    ploter.figs.append(fig)

    save_path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"混淆矩阵已保存：{save_path}")


# ===================== 缓存读取 =====================
def load_mecc_cache():
    if not os.path.exists(RESULT_JSON_PATH):
        return None, None, None
    logger.info(f"发现缓存文件，直接加载：{RESULT_JSON_PATH}")
    with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best_params    = data['best_params']
    search_results = data['search_results']
    param_metrics  = {}
    for k_str, v in data['param_metrics'].items():
        k_viv_str, C_viv_str = k_str.split('_', 1)
        param_metrics[(int(k_viv_str), float(C_viv_str))] = {
            'weighted':         v['weighted'],
            'per_class':        v['per_class'],
            'confusion_matrix': np.array(v['confusion_matrix'])
        }
    logger.info(f"缓存加载完成，最优参数：{best_params}")
    return best_params, search_results, param_metrics


# ===================== 结果存储 =====================
def save_mecc_search_results(
    best_params: dict,
    search_results: list,
    param_metrics: dict,
    freq_p95: float
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    result_data = {
        'sigma_0':        SIGMA_0,
        'freq_p95':       freq_p95,
        'best_params':    best_params,
        'best_f1':        max(r['f1'] for r in search_results) if search_results else 0.0,
        'search_results': [
            {
                'params':    r['params'],
                'precision': r['precision'],
                'recall':    r['recall'],
                'f1':        r['f1']
            }
            for r in search_results
        ],
        'param_metrics': {
            f"{k[0]}_{k[1]}": {
                'weighted':         v['weighted'],
                'per_class':        v['per_class'],
                'confusion_matrix': v['confusion_matrix'].tolist()
            }
            for k, v in param_metrics.items()
        }
    }

    with open(RESULT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    logger.info(f"MECC搜索结果已保存：{RESULT_JSON_PATH}")


def write_result_file(
    best_params: dict,
    best_metrics: dict,
    search_results: list,
    dataset_size: int,
    freq_p95: float
):
    result_path = os.path.join(SAVE_DIR, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MECC参数搜索结果报告（全数据集二分类）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集配置: {CONFIG_PATH}\n")
        f.write(f"sigma_0={SIGMA_0}（固定，不参与搜索）\n")
        f.write(f"freq_p95={freq_p95:.4f} Hz（主导模态95%分位阈值，固定，不参与搜索）\n")
        f.write(f"有效样本数（label 0/1）: {dataset_size}\n")
        f.write(f"搜索网格: k_viv={PARAM_GRID['k_viv']}\n")
        f.write(f"         C_viv={PARAM_GRID['C_viv']}\n")
        f.write(f"搜索总组合数: {len(search_results)}\n")
        f.write("=" * 80 + "\n\n")

        f.write("【最优参数】\n")
        f.write(f"  k_viv = {best_params['k_viv']}\n")
        f.write(f"  C_viv = {best_params['C_viv']}\n\n")

        f.write("【最优参数详细指标】\n")
        f.write(f"加权Precision: {best_metrics['weighted']['Precision']:.4f}\n")
        f.write(f"加权Recall:    {best_metrics['weighted']['Recall']:.4f}\n")
        f.write(f"加权F1 Score:  {best_metrics['weighted']['F1']:.4f}\n")
        f.write("\n类别级指标:\n")
        for cls_name, cls_m in best_metrics['per_class'].items():
            f.write(f"  {cls_name}: P={cls_m['Precision']:.4f}  R={cls_m['Recall']:.4f}  F1={cls_m['F1']:.4f}\n")

        f.write("\n【搜索结果（按F1降序，TOP 10）】\n")
        f.write(f"{'k_viv':<8} {'C_viv':<8} {'加权P':<10} {'加权R':<10} {'加权F1':<10}\n")
        f.write("-" * 50 + "\n")
        sorted_results = sorted(search_results, key=lambda x: x['f1'], reverse=True)
        for r in sorted_results[:10]:
            p = r['params']
            f.write(
                f"{p['k_viv']:<8d} {p['C_viv']:<8.3f} "
                f"{r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}\n"
            )

    logger.info(f"结果文件已保存：{result_path}")


# ===================== 主流程 =====================
def run_mecc_param_search(
    config_path: str = CONFIG_PATH,
    param_grid: dict = None
):
    """主流程：优先从缓存加载 → 否则搜索 → 绘图"""
    if param_grid is None:
        param_grid = PARAM_GRID

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_params, search_results, param_metrics = (None, None, None) if FORCE_RECOMPUTE else load_mecc_cache()

    if best_params is None:
        logger.info("=" * 60)
        logger.info("开始 MECC 参数搜索（全数据集二分类，sigma_0 固定）")
        logger.info("=" * 60)

        # 1. 加载全量验证集
        dataset_config = load_dataset_config(config_path)
        _, val_dl, num_classes = create_dataloaders(dataset_config, batch_size=16)
        logger.info(f"数据集加载完成，类别数：{num_classes}")

        val_features, val_labels = extract_data_from_dataloader(val_dl)
        val_features, val_labels = filter_binary_samples(val_features, val_labels)
        dataset_size = len(val_labels)
        logger.info(f"验证集样本总数（label 0/1）：{dataset_size}")

        # 2. 初始化基频与主频阈值
        mount = Cal_Mount()
        inplane_modes, _ = mount.base_modes()
        f0       = inplane_modes[0]
        freq_p95 = compute_freq_p95()
        logger.info(f"拉索基频 f0 = {f0:.4f} Hz")
        logger.info(f"主导模态 95% 分位阈值 freq_p95 = {freq_p95:.4f} Hz")

        # 3. 二参数联合搜索
        best_params, search_results, param_metrics = mecc_hyperparameter_search(
            val_features, val_labels, f0, freq_p95, param_grid
        )

        # 4. 保存结果
        save_mecc_search_results(best_params, search_results, param_metrics, freq_p95)

        best_key     = (best_params['k_viv'], best_params['C_viv'])
        best_metrics = param_metrics[best_key]
        write_result_file(best_params, best_metrics, search_results, dataset_size, freq_p95)
    else:
        logger.info("使用缓存结果，跳过参数搜索直接绘图")

    best_key     = (best_params['k_viv'], best_params['C_viv'])
    best_metrics = param_metrics[best_key]
    best_f1      = best_metrics['weighted']['F1']

    # 5. 绘图
    ploter     = PlotLib()
    best_k_viv = best_params['k_viv']
    best_C_viv = best_params['C_viv']

    plot_mecc_f1_contour(search_results, param_grid, ploter)

    _plot_confusion_matrix(
        best_params, best_f1,
        best_metrics['confusion_matrix'],
        save_name=f"confusion_matrix_kv{best_k_viv}_Cv{best_C_viv}.png",
        ploter=ploter
    )

    ploter.show()

    logger.info("=" * 60)
    logger.info("MECC参数搜索完成")
    logger.info(f"  最优参数 : k_viv={best_k_viv}, C_viv={best_C_viv}")
    logger.info(f"  sigma_0  : {SIGMA_0}（固定）")
    logger.info(f"  加权F1   : {best_f1:.4f}")
    logger.info("=" * 60)

    return best_params, search_results, param_metrics


if __name__ == "__main__":
    run_mecc_param_search()
