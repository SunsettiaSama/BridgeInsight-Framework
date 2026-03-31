import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from src.identifier.cable_analysis_methods.mecc import Abnormal_Vibration_Filter
from src.data_processer.preprocess.get_data_vib import (
    _load_dominant_freq_statistics,
    _calculate_freq_threshold,
)
from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset, log_split_distribution
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

# 三分类标签：0=一般振动，1=涡激共振(VIV)，2=风雨振(RWIV)
ALL_LABEL_IDS = [0, 1, 2]
ALL_LABELS    = ['Normal', 'VIV', 'RWIV']

# 固定参数（不参与搜索）
SIGMA_0 = 0.1

# 四参数联合搜索网格：k_viv × C_viv × k_rwiv × C_rwiv
PARAM_GRID = {
    'k_viv':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'C_viv':  [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'k_rwiv': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'C_rwiv': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

CMAP         = get_blue_color_map()
CMAP_CONTOUR = get_blue_color_map(style='gradient')


# ===================== 频率阈值计算（来自统计，非超参数）=====================
def compute_freq_threshold(fallback: float = 5.0) -> float:
    stats = _load_dominant_freq_statistics()
    freq_threshold = _calculate_freq_threshold(stats)
    logger.info(f"高低频分割阈值（来自统计）：{freq_threshold:.4f} Hz")
    return freq_threshold


# ===================== 数据集加载 =====================
def load_dataset_config(config_path: str) -> AnnotationDatasetConfig:
    logger.info(f"加载数据集配置：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = AnnotationDatasetConfig(**config_dict)
    logger.info(f"数据集配置加载完成（split_ratio={config.split_ratio}, auto_split={config.auto_split}）")
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
    log_split_distribution(dataset)

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


# ===================== MECC 分类核心 =====================
def mecc_classify(
    sig: np.ndarray,
    f0: float,
    freq_threshold: float,
    k_viv: int,
    C_viv: float,
    k_rwiv: int,
    C_rwiv: float,
    classifier: Abnormal_Vibration_Filter
) -> int:
    sig = np.array(sig, dtype=np.float64)
    return classifier.classify_vibration(
        data=sig,
        f0=f0,
        sigma_0=SIGMA_0,
        k_viv=k_viv,
        C_viv=C_viv,
        k_rmw=k_rwiv,
        C_rmw=C_rwiv,
        freq_threshold=freq_threshold
    )


# ===================== 三分类指标计算 =====================
def _compute_3class_metrics(y_true, y_pred) -> dict:
    precision_arr = precision_score(y_true, y_pred, labels=ALL_LABEL_IDS, average=None, zero_division=0)
    recall_arr    = recall_score(   y_true, y_pred, labels=ALL_LABEL_IDS, average=None, zero_division=0)
    f1_arr        = f1_score(       y_true, y_pred, labels=ALL_LABEL_IDS, average=None, zero_division=0)

    return {
        'per_class': {
            ALL_LABELS[i]: {
                'Precision': float(precision_arr[i]),
                'Recall':    float(recall_arr[i]),
                'F1':        float(f1_arr[i])
            }
            for i in range(len(ALL_LABELS))
        },
        'weighted': {
            'Precision': float(precision_score(y_true, y_pred, labels=ALL_LABEL_IDS, average='weighted', zero_division=0)),
            'Recall':    float(recall_score(   y_true, y_pred, labels=ALL_LABEL_IDS, average='weighted', zero_division=0)),
            'F1':        float(f1_score(       y_true, y_pred, labels=ALL_LABEL_IDS, average='weighted', zero_division=0))
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=ALL_LABEL_IDS)
    }


def _calc_full_metrics(
    k_viv: int,
    C_viv: float,
    k_rwiv: int,
    C_rwiv: float,
    features: np.ndarray,
    labels: np.ndarray,
    f0: float,
    freq_threshold: float,
    classifier: Abnormal_Vibration_Filter
):
    true_labels, pred_labels = [], []
    for data, label in zip(features, labels):
        sig  = data[:, 0] if data.ndim > 1 else data
        pred = mecc_classify(sig, f0, freq_threshold, k_viv, C_viv, k_rwiv, C_rwiv, classifier)
        true_labels.append(int(label))
        pred_labels.append(pred)

    metrics = _compute_3class_metrics(true_labels, pred_labels)
    return metrics, metrics['weighted']['F1']


# ===================== 四参数联合搜索 =====================
def mecc_hyperparameter_search(
    features: np.ndarray,
    labels: np.ndarray,
    f0: float,
    freq_threshold: float,
    classifier: Abnormal_Vibration_Filter,
    param_grid: dict = None
):
    if param_grid is None:
        param_grid = PARAM_GRID

    k_viv_vals  = param_grid['k_viv']
    C_viv_vals  = param_grid['C_viv']
    k_rwiv_vals = param_grid['k_rwiv']
    C_rwiv_vals = param_grid['C_rwiv']
    total = len(k_viv_vals) * len(C_viv_vals) * len(k_rwiv_vals) * len(C_rwiv_vals)

    logger.info("=" * 60)
    logger.info(f"开始 MECC 全数据集三分类四参数联合搜索，组合数：{total}")
    logger.info(f"sigma_0（固定）: {SIGMA_0}")
    logger.info(f"k_viv   : {k_viv_vals}")
    logger.info(f"C_viv   : {C_viv_vals}")
    logger.info(f"k_rwiv  : {k_rwiv_vals}")
    logger.info(f"C_rwiv  : {C_rwiv_vals}")
    logger.info(f"标签    : {ALL_LABEL_IDS} → {ALL_LABELS}")
    logger.info("=" * 60)

    search_results = []
    param_metrics  = {}
    best_f1        = 0.0
    best_params    = None

    with tqdm(total=total, desc="MECC联合搜索") as pbar:
        for k_viv in k_viv_vals:
            for C_viv in C_viv_vals:
                for k_rwiv in k_rwiv_vals:
                    for C_rwiv in C_rwiv_vals:
                        metrics, f1 = _calc_full_metrics(
                            k_viv, C_viv, k_rwiv, C_rwiv,
                            features, labels, f0, freq_threshold, classifier
                        )
                        param_metrics[(k_viv, C_viv, k_rwiv, C_rwiv)] = metrics
                        result = {
                            'params':    {'k_viv': k_viv, 'C_viv': C_viv, 'k_rwiv': k_rwiv, 'C_rwiv': C_rwiv},
                            'precision': metrics['weighted']['Precision'],
                            'recall':    metrics['weighted']['Recall'],
                            'f1':        f1
                        }
                        search_results.append(result)
                        if f1 > best_f1:
                            best_f1     = f1
                            best_params = {'k_viv': k_viv, 'C_viv': C_viv, 'k_rwiv': k_rwiv, 'C_rwiv': C_rwiv}
                        pbar.update(1)

    logger.info(f"搜索完成  最优参数：{best_params}  Weighted F1：{best_f1:.4f}")
    return best_params, search_results, param_metrics


# ===================== 绘图 =====================
def _plot_contour_slice(
    search_results: list,
    x_key: str,
    x_vals: list,
    x_label: str,
    y_key: str,
    y_vals: list,
    y_label: str,
    fixed_params: dict,
    title_suffix: str,
    save_name: str,
    ploter: PlotLib
):
    f1_matrix = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in search_results:
        p = r['params']
        if any(abs(p[k] - v) > 1e-9 for k, v in fixed_params.items()):
            continue
        if p[x_key] not in x_vals or p[y_key] not in y_vals:
            continue
        xi = x_vals.index(p[x_key])
        yi = y_vals.index(p[y_key])
        f1_matrix[yi, xi] = r['f1']

    X, Y = np.meshgrid(x_vals, y_vals)

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    cf = ax.contourf(X, Y, f1_matrix, cmap=CMAP_CONTOUR, levels=20, alpha=1.0)
    ax.contour(X, Y, f1_matrix, levels=20, colors='black', linewidths=0.5, alpha=0.3)

    best_idx         = np.nanargmax(f1_matrix)
    best_yi, best_xi = np.unravel_index(best_idx, f1_matrix.shape)
    best_x           = x_vals[best_xi]
    best_y           = y_vals[best_yi]
    best_f1          = f1_matrix[best_yi, best_xi]

    ax.scatter([best_x], [best_y], color='red', s=80, zorder=5,
               label=f'Best F1={best_f1:.3f}')
    ax.legend(prop=ENG_FONT, loc='upper right')
    ax.set_xlabel(x_label, labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel(y_label, labelpad=10, fontproperties=ENG_FONT)
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label(f'Weighted F1 Score ({title_suffix})', fontproperties=ENG_FONT)

    plt.tight_layout()
    ploter.figs.append(fig)

    save_path = os.path.join(SAVE_DIR, save_name)
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
        xticklabels=ALL_LABELS, yticklabels=ALL_LABELS,
        ax=ax, cbar=False, annot_kws={'size': 18}
    )
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('True Label',      labelpad=10, fontproperties=ENG_FONT)
    param_str = (
        f"k_viv={best_params['k_viv']}, C_viv={best_params['C_viv']}, "
        f"k_rwiv={best_params['k_rwiv']}, C_rwiv={best_params['C_rwiv']}"
    )
    ax.set_title(
        f"{param_str}\nWeighted F1: {f1_value:.3f}",
        pad=20, fontproperties=ENG_FONT
    )
    plt.tight_layout()
    ploter.figs.append(fig)

    save_path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"混淆矩阵已保存：{save_path}")


# ===================== 结果存储 =====================
def save_mecc_search_results(
    freq_threshold: float,
    best_params: dict,
    search_results: list,
    param_metrics: dict
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    result_data = {
        'freq_threshold': freq_threshold,
        'sigma_0':        SIGMA_0,
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
            f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": {
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
    freq_threshold: float,
    best_params: dict,
    best_metrics: dict,
    search_results: list,
    dataset_size: int
):
    result_path = os.path.join(SAVE_DIR, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MECC参数搜索结果报告（全数据集三分类）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集配置: {CONFIG_PATH}\n")
        f.write(f"高低频分割阈值（统计结果）: {freq_threshold:.4f} Hz\n")
        f.write(f"sigma_0={SIGMA_0}（固定，不参与搜索）\n")
        f.write(f"有效样本数（label 0/1/2）: {dataset_size}\n")
        f.write(f"搜索网格: k_viv={PARAM_GRID['k_viv']}\n")
        f.write(f"         C_viv={PARAM_GRID['C_viv']}\n")
        f.write(f"         k_rwiv={PARAM_GRID['k_rwiv']}\n")
        f.write(f"         C_rwiv={PARAM_GRID['C_rwiv']}\n")
        f.write(f"搜索总组合数: {len(search_results)}\n")
        f.write("=" * 80 + "\n\n")

        f.write("【最优参数】\n")
        f.write(f"  k_viv  = {best_params['k_viv']}\n")
        f.write(f"  C_viv  = {best_params['C_viv']}\n")
        f.write(f"  k_rwiv = {best_params['k_rwiv']}\n")
        f.write(f"  C_rwiv = {best_params['C_rwiv']}\n\n")

        f.write("【最优参数详细指标】\n")
        f.write(f"加权Precision: {best_metrics['weighted']['Precision']:.4f}\n")
        f.write(f"加权Recall:    {best_metrics['weighted']['Recall']:.4f}\n")
        f.write(f"加权F1 Score:  {best_metrics['weighted']['F1']:.4f}\n")
        f.write("\n类别级指标:\n")
        for cls_name, cls_m in best_metrics['per_class'].items():
            f.write(f"  {cls_name}: P={cls_m['Precision']:.4f}  R={cls_m['Recall']:.4f}  F1={cls_m['F1']:.4f}\n")

        f.write("\n【搜索结果（按F1降序，TOP 10）】\n")
        f.write(f"{'k_viv':<8} {'C_viv':<8} {'k_rwiv':<8} {'C_rwiv':<8} {'加权P':<10} {'加权R':<10} {'加权F1':<10}\n")
        f.write("-" * 70 + "\n")
        sorted_results = sorted(search_results, key=lambda x: x['f1'], reverse=True)
        for r in sorted_results[:10]:
            p = r['params']
            f.write(
                f"{p['k_viv']:<8d} {p['C_viv']:<8.3f} {p['k_rwiv']:<8d} {p['C_rwiv']:<8.3f} "
                f"{r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}\n"
            )

    logger.info(f"结果文件已保存：{result_path}")


# ===================== 主流程 =====================
def run_mecc_param_search(
    config_path: str = CONFIG_PATH,
    param_grid: dict = None
):
    if param_grid is None:
        param_grid = PARAM_GRID

    os.makedirs(SAVE_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始 MECC 参数搜索（全数据集三分类，sigma_0 固定）")
    logger.info("=" * 60)

    # 1. 频率阈值（来自统计，非超参数）
    freq_threshold = compute_freq_threshold()

    # 2. 加载验证集（全部标签）
    dataset_config = load_dataset_config(config_path)
    _, val_dl, num_classes = create_dataloaders(dataset_config, batch_size=16)
    logger.info(f"数据集加载完成，类别数：{num_classes}")

    val_features, val_labels = extract_data_from_dataloader(val_dl)
    dataset_size = len(val_labels)
    logger.info(f"验证集样本总数（label 0/1/2）：{dataset_size}")

    # 3. 初始化分类器与基频
    classifier = Abnormal_Vibration_Filter(fs=50)
    mount      = Cal_Mount()
    inplane_modes, _ = mount.base_modes()
    f0 = inplane_modes[0]
    logger.info(f"拉索基频 f0 = {f0:.4f} Hz")

    # 4. 四参数联合搜索
    best_params, search_results, param_metrics = mecc_hyperparameter_search(
        val_features, val_labels, f0, freq_threshold, classifier, param_grid
    )

    # 5. 保存结果
    save_mecc_search_results(freq_threshold, best_params, search_results, param_metrics)

    best_key     = (best_params['k_viv'], best_params['C_viv'], best_params['k_rwiv'], best_params['C_rwiv'])
    best_metrics = param_metrics[best_key]
    best_f1      = best_metrics['weighted']['F1']

    write_result_file(freq_threshold, best_params, best_metrics, search_results, dataset_size)

    # 6. 绘图
    ploter        = PlotLib()
    best_k_viv    = best_params['k_viv']
    best_C_viv    = best_params['C_viv']
    best_k_rwiv   = best_params['k_rwiv']
    best_C_rwiv   = best_params['C_rwiv']

    # 截面1：VIV 参数空间（C_viv × k_viv），固定最优 RWIV 参数
    _plot_contour_slice(
        search_results,
        x_key='C_viv', x_vals=param_grid['C_viv'], x_label=r"$C_{VIV}$ (MECC threshold)",
        y_key='k_viv', y_vals=param_grid['k_viv'],  y_label=r"$k_{VIV}$",
        fixed_params={'k_rwiv': best_k_rwiv, 'C_rwiv': best_C_rwiv},
        title_suffix=f'k_rwiv={best_k_rwiv}, C_rwiv={best_C_rwiv}',
        save_name='mecc_contour_viv_params.png',
        ploter=ploter
    )

    # 截面2：RWIV 参数空间（C_rwiv × k_rwiv），固定最优 VIV 参数
    _plot_contour_slice(
        search_results,
        x_key='C_rwiv', x_vals=param_grid['C_rwiv'], x_label=r"$C_{RWIV}$ (MECC threshold)",
        y_key='k_rwiv', y_vals=param_grid['k_rwiv'],  y_label=r"$k_{RWIV}$",
        fixed_params={'k_viv': best_k_viv, 'C_viv': best_C_viv},
        title_suffix=f'k_viv={best_k_viv}, C_viv={best_C_viv}',
        save_name='mecc_contour_rwiv_params.png',
        ploter=ploter
    )

    # 三分类混淆矩阵
    _plot_confusion_matrix(
        best_params, best_f1,
        best_metrics['confusion_matrix'],
        save_name=f"confusion_matrix_kv{best_k_viv}_Cv{best_C_viv}_kr{best_k_rwiv}_Cr{best_C_rwiv}.png",
        ploter=ploter
    )

    ploter.show()

    logger.info("=" * 60)
    logger.info("MECC参数搜索完成")
    logger.info(f"  最优参数 : k_viv={best_k_viv}, C_viv={best_C_viv}, k_rwiv={best_k_rwiv}, C_rwiv={best_C_rwiv}")
    logger.info(f"  sigma_0  : {SIGMA_0}（固定）")
    logger.info(f"  加权F1   : {best_f1:.4f}")
    logger.info("=" * 60)

    return best_params, search_results, param_metrics


if __name__ == "__main__":
    run_mecc_param_search()
