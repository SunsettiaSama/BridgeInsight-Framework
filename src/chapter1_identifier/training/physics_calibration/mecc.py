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
from typing import Dict

from src.visualize_tools.utils import PlotLib
from src.identifier.cable_analysis_methods.base_mode_calculator import Cal_Mount, parse_mount_point_id
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
CONFIG_PATH         = "config/train/datasets/annotation_dataset.yaml"
CABLE_CONFIG_PATH   = "config/cables/cable_params.yaml"
SAVE_DIR               = r"F:\Research\Vibration Characteristics In Cable Vibration\results\mecc_results"
RESULT_SUMMARY_PATH    = os.path.join(SAVE_DIR, 'mecc_search_results_summary.json')

# 二分类标签：0=一般振动，1=涡激共振(VIV)
LABEL_IDS = [0, 1]
LABELS    = ['Normal', 'VIV']

# 搜参目标：Macro F1（两类 F1 等权平均，不随 Normal 样本数加权）
SEARCH_METRIC = 'macro_f1'

# 三组拉索测点独立搜参（测点编号 Cxx-yyy）
CABLE_GROUPS = {
    'C18': {
        'label': 'C18 柔性索',
        'mount_points': frozenset(['C18-101', 'C18-102']),
    },
    'C34-1xx': {
        'label': 'C34 北塔边跨',
        'mount_points': frozenset(['C34-101', 'C34-102']),
    },
    'C34-2xx_3xx': {
        'label': 'C34 跨中/南塔',
        'mount_points': frozenset(['C34-201', 'C34-202', 'C34-301', 'C34-302']),
    },
}

# WebUI 推送（需先启动 python -m src.visualize_tools.web_dashboard，或由 show_web 自动拉起）
PUSH_TO_WEBUI = True
WEBUI_PORT    = 5678
WEBUI_COLS    = 2

# 全量数据用于验证
VAL_RATIO = 0.01

# 强制重新搜索（True：忽略缓存；False：有缓存则直接绘图）
FORCE_RECOMPUTE = True

# 两阶段搜索：粗搜定位区域 → 细搜拉高 k / C 精度
ENABLE_FINE_SEARCH = True
FINE_K_RADIUS      = 1.0     # 细搜 k_viv 范围：best ± radius
FINE_K_STEP        = 0.1     # 细搜 k_viv 步长（可为小数）
FINE_C_RADIUS      = 0.20    # 细搜 C_viv 范围：best ± radius
FINE_C_STEP        = 0.005   # 细搜 C_viv 步长
FINE_SIGMA_RADIUS  = 0.05    # 细搜 sigma_0 范围：best ± radius
FINE_SIGMA_STEP    = 0.01    # 细搜 sigma_0 步长

SIGMA_0_MIN = 0.01
SIGMA_0_MAX = 0.30
K_VIV_MIN = 0.9   # 允许基频误差，临近峰剔除区间可略小于 1 倍 f0
K_VIV_MAX = 15.0
C_VIV_MIN = 0.05
C_VIV_MAX = 1.0

COARSE_SIGMA_STEP = 0.05
COARSE_K_STEP = 0.5
COARSE_C_STEP = 0.05

# 粗搜网格：sigma_0 / k / C 均为浮点步长
COARSE_PARAM_GRID = {
    'sigma_0': [round(v, 4) for v in np.arange(SIGMA_0_MIN, SIGMA_0_MAX + 1e-9, COARSE_SIGMA_STEP)],
    'k_viv': [round(v, 4) for v in np.arange(K_VIV_MIN, K_VIV_MAX + 1e-9, COARSE_K_STEP)],
    'C_viv': [round(v, 4) for v in np.arange(C_VIV_MIN, C_VIV_MAX + 1e-9, COARSE_C_STEP)],
}

# 兼容旧引用
PARAM_GRID = COARSE_PARAM_GRID

CMAP         = get_blue_color_map()
CMAP_CONTOUR = get_blue_color_map(style='gradient')


def _normalize_sigma_0(sigma_0) -> float:
    return round(float(sigma_0), 4)


def _normalize_k_viv(k_viv) -> float:
    return round(float(k_viv), 4)


def _normalize_c_viv(c_viv) -> float:
    return round(float(c_viv), 4)


def _param_key(sigma_0, k_viv, c_viv) -> tuple:
    return (
        _normalize_sigma_0(sigma_0),
        _normalize_k_viv(k_viv),
        _normalize_c_viv(c_viv),
    )


def _param_key_to_str(sigma_0, k_viv, c_viv) -> str:
    return (
        f"{_normalize_sigma_0(sigma_0):g}_"
        f"{_normalize_k_viv(k_viv):g}_"
        f"{_normalize_c_viv(c_viv):g}"
    )


def _parse_param_key(key_str: str) -> tuple:
    sigma_str, k_viv_str, c_viv_str = key_str.split('_', 2)
    return (
        _normalize_sigma_0(sigma_str),
        _normalize_k_viv(k_viv_str),
        _normalize_c_viv(c_viv_str),
    )


def _float_range(lo: float, hi: float, step: float) -> list:
    if step <= 0:
        raise ValueError(f"step 必须为正数，当前 step={step}")
    n = int(round((hi - lo) / step)) + 1
    return sorted({
        round(min(hi, lo + i * step), 4)
        for i in range(n)
        if lo - 1e-9 <= round(lo + i * step, 4) <= hi + 1e-9
    })


def _compute_search_score(metrics: dict) -> float:
    """Macro F1：Normal/VIV 的 F1 算术平均，缓解类别不平衡对 weighted F1 的支配。"""
    pc = metrics['per_class']
    return float((pc['Normal']['F1'] + pc['VIV']['F1']) / 2.0)


def _group_result_path(group_id: str) -> str:
    return os.path.join(SAVE_DIR, f'mecc_search_results_{group_id}.json')


def _mask_for_mount_points(sensor_ids: np.ndarray, mount_points) -> np.ndarray:
    return np.array([
        parse_mount_point_id(str(sid)) in mount_points
        for sid in sensor_ids
    ])


def _subset_by_group(
    features: np.ndarray,
    labels: np.ndarray,
    f0s: np.ndarray,
    sensor_ids: np.ndarray,
    group_id: str,
):
    cfg = CABLE_GROUPS[group_id]
    mask = _mask_for_mount_points(sensor_ids, cfg['mount_points'])
    n = int(mask.sum())
    if n == 0:
        raise ValueError(f"拉索组 {group_id} 无样本")
    logger.info(
        f"拉索组 [{group_id}] {cfg['label']}：N={n}  "
        f"Normal={int((labels[mask] == 0).sum())}  VIV={int((labels[mask] == 1).sum())}"
    )
    return features[mask], labels[mask], f0s[mask]


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
    # MECC 搜索只需按索引读验证集窗口，不需要 DL 训练用的全量预加载
    config_dict['enable_preload_cache'] = False
    config_dict['show_preload_progress'] = False
    config = AnnotationDatasetConfig(**config_dict)
    logger.info(
        "数据集配置加载完成（auto_split=True, split_ratio=%s, preload=off）",
        VAL_RATIO,
    )
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


def _sensor_id_from_annotation(anno: dict) -> str:
    sid = anno.get("sensor_id")
    if sid:
        return sid
    meta = anno.get("metadata") or {}
    sid = meta.get("sensor_id")
    if not sid:
        raise ValueError(f"标注缺少 sensor_id: {anno}")
    return sid


def _build_f0_cache(sensor_ids) -> Dict[str, float]:
    """按传感器解析测点编号（Cxx-yyy），从 config/cables/cable_params.yaml 计算面内一阶基频。"""
    cache: Dict[str, float] = {}
    for sid in sorted(sensor_ids):
        mount_point = parse_mount_point_id(sid)
        mount = Cal_Mount.from_sensor(sid)
        cache[sid] = mount.inplane_mode(1)
        logger.info(
            f"  {sid} → 测点 {mount_point} → f0 = {cache[sid]:.4f} Hz  ({CABLE_CONFIG_PATH})"
        )
    return cache


def extract_val_search_data(dataset: AnnotationDataset):
    """
    从验证集提取特征、标签及每样本基频。
    基频按样本 sensor_id 查 config/cables/cable_params.yaml，与生产 MECC 一致。
    """
    val_dataset = dataset.get_val_dataset()
    parent: AnnotationDataset = val_dataset.dataset

    sensor_ids = set()
    for orig_idx in val_dataset.indices:
        anno = parent._idx_to_annotation[orig_idx]
        if anno is None:
            continue
        sensor_ids.add(_sensor_id_from_annotation(anno))

    logger.info(
        f"验证集涉及 {len(sensor_ids)} 个传感器，基频来源：{CABLE_CONFIG_PATH}"
    )
    f0_cache = _build_f0_cache(sensor_ids)

    features_list, labels_list, f0_list, sensor_ids_list = [], [], [], []
    with tqdm(total=len(val_dataset), desc="提取验证集") as pbar:
        for i in range(len(val_dataset)):
            orig_idx = val_dataset.indices[i]
            anno = parent._idx_to_annotation[orig_idx]
            data, _ = val_dataset[i]
            sid = _sensor_id_from_annotation(anno)
            label = int(anno["class_id"])
            features_list.append(data.numpy())
            labels_list.append(label)
            f0_list.append(f0_cache[sid])
            sensor_ids_list.append(sid)
            pbar.update(1)

    features = np.stack(features_list, axis=0) if features_list else np.array([])
    labels   = np.array(labels_list)
    f0s      = np.array(f0_list, dtype=np.float64)
    sensor_ids = np.array(sensor_ids_list)
    return features, labels, f0s, sensor_ids


def filter_binary_samples(
    features: np.ndarray,
    labels: np.ndarray,
    f0s: np.ndarray,
    sensor_ids: np.ndarray,
):
    mask = np.isin(labels, LABEL_IDS)
    kept = int(mask.sum())
    logger.info(f"二分类过滤：{len(labels)} → {kept} 个样本（保留 label 0/1）")
    return features[mask], labels[mask], f0s[mask], sensor_ids[mask]


# ===================== MECC 分类核心 =====================
def mecc_classify(
    sig: np.ndarray,
    f0: float,
    k_viv: float,
    C_viv: float,
    freq_p95: float,
    sigma_0: float,
) -> int:
    sig = np.array(sig, dtype=np.float64)
    rms = np.sqrt(np.mean((sig - np.mean(sig)) ** 2))
    if rms < sigma_0:
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
        'macro': {
            'Precision': float(precision_score(y_true, y_pred, labels=LABEL_IDS, average='macro', zero_division=0)),
            'Recall':    float(recall_score(   y_true, y_pred, labels=LABEL_IDS, average='macro', zero_division=0)),
            'F1':        float(f1_score(       y_true, y_pred, labels=LABEL_IDS, average='macro', zero_division=0))
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    }


def _calc_metrics(
    sigma_0: float,
    k_viv: float,
    C_viv: float,
    features: np.ndarray,
    labels: np.ndarray,
    f0s: np.ndarray,
    freq_p95: float,
):
    true_labels, pred_labels = [], []
    for data, label, f0 in zip(features, labels, f0s):
        sig  = data[:, 0] if data.ndim > 1 else data
        pred = mecc_classify(sig, f0, k_viv, C_viv, freq_p95, sigma_0)
        true_labels.append(int(label))
        pred_labels.append(pred)

    metrics = _compute_binary_metrics(true_labels, pred_labels)
    return metrics, _compute_search_score(metrics)


def build_fine_param_grid(
    best_sigma: float,
    best_k: float,
    best_c: float,
    sigma_radius: float = FINE_SIGMA_RADIUS,
    sigma_step: float = FINE_SIGMA_STEP,
    k_radius: float = FINE_K_RADIUS,
    k_step: float = FINE_K_STEP,
    c_radius: float = FINE_C_RADIUS,
    c_step: float = FINE_C_STEP,
) -> dict:
    """在粗搜最优点邻域构建细搜网格（sigma / k / C 均支持浮点步长）。"""
    sigma_lo = max(SIGMA_0_MIN, round(best_sigma - sigma_radius, 4))
    sigma_hi = min(SIGMA_0_MAX, round(best_sigma + sigma_radius, 4))
    sigma_vals = _float_range(sigma_lo, sigma_hi, sigma_step)

    k_lo = max(K_VIV_MIN, round(best_k - k_radius, 4))
    k_hi = min(K_VIV_MAX, round(best_k + k_radius, 4))
    k_vals = _float_range(k_lo, k_hi, k_step)

    c_lo = max(C_VIV_MIN, round(best_c - c_radius, 4))
    c_hi = min(C_VIV_MAX, round(best_c + c_radius, 4))
    c_vals = _float_range(c_lo, c_hi, c_step)

    return {'sigma_0': sigma_vals, 'k_viv': k_vals, 'C_viv': c_vals}


def _format_param_grid_log(param_grid: dict) -> tuple:
    sigma_vals = param_grid['sigma_0']
    k_vals = param_grid['k_viv']
    c_vals = param_grid['C_viv']

    def _summarize(vals):
        if len(vals) <= 12:
            return str(vals)
        step = vals[1] - vals[0] if len(vals) > 1 else 0.0
        return f"[{vals[0]}, ..., {vals[-1]}] (n={len(vals)}, step≈{step:.4f})"

    return _summarize(sigma_vals), _summarize(k_vals), _summarize(c_vals)


def _select_best_params(param_metrics: dict) -> tuple:
    best_score  = -1.0
    best_params = None
    for (sigma_0, k_viv, c_viv), metrics in param_metrics.items():
        score = _compute_search_score(metrics)
        if score > best_score:
            best_score  = score
            best_params = {
                'sigma_0': _normalize_sigma_0(sigma_0),
                'k_viv':   _normalize_k_viv(k_viv),
                'C_viv':   _normalize_c_viv(c_viv),
            }
    return best_params, best_score


def _build_search_results_list(param_metrics: dict) -> list:
    results = []
    for (sigma_0, k_viv, c_viv), metrics in sorted(param_metrics.items()):
        w = metrics['weighted']
        results.append({
            'params': {
                'sigma_0': _normalize_sigma_0(sigma_0),
                'k_viv':   _normalize_k_viv(k_viv),
                'C_viv':   _normalize_c_viv(c_viv),
            },
            'precision':    w['Precision'],
            'recall':       w['Recall'],
            'weighted_f1':  w['F1'],
            'f1':           _compute_search_score(metrics),
        })
    return results


def grid_from_search_results(search_results: list, sigma_0: float = None) -> dict:
    if sigma_0 is not None:
        target_sigma = _normalize_sigma_0(sigma_0)
        filtered = [
            r for r in search_results
            if _normalize_sigma_0(r['params']['sigma_0']) == target_sigma
        ]
    else:
        filtered = search_results
    return {
        'sigma_0': sorted({r['params']['sigma_0'] for r in filtered}),
        'k_viv':   sorted({r['params']['k_viv'] for r in filtered}),
        'C_viv':   sorted({r['params']['C_viv'] for r in filtered}),
    }


# ===================== 三参数联合搜索 =====================
def mecc_hyperparameter_search(
    features: np.ndarray,
    labels: np.ndarray,
    f0s: np.ndarray,
    freq_p95: float,
    param_grid: dict = None,
    stage_name: str = "MECC联合搜索",
    param_metrics: dict = None,
):
    if param_grid is None:
        param_grid = COARSE_PARAM_GRID

    sigma_vals = param_grid['sigma_0']
    k_viv_vals = param_grid['k_viv']
    C_viv_vals = param_grid['C_viv']
    total      = len(sigma_vals) * len(k_viv_vals) * len(C_viv_vals)

    sigma_log, k_log, c_log = _format_param_grid_log(param_grid)
    logger.info("=" * 60)
    logger.info(f"{stage_name}，组合数：{total}")
    logger.info(f"freq_p95（固定）: {freq_p95:.4f} Hz")
    logger.info(f"基频 f0 : 按样本 sensor_id → {CABLE_CONFIG_PATH}")
    logger.info(f"sigma_0 : {sigma_log}")
    logger.info(f"k_viv   : {k_log}")
    logger.info(f"C_viv   : {c_log}")
    logger.info(f"标签    : {LABEL_IDS} → {LABELS}")
    logger.info("=" * 60)

    if param_metrics is None:
        param_metrics = {}

    with tqdm(total=total, desc=stage_name) as pbar:
        for sigma_0 in sigma_vals:
            for k_viv in k_viv_vals:
                for C_viv in C_viv_vals:
                    metrics, _ = _calc_metrics(
                        sigma_0, k_viv, C_viv,
                        features, labels, f0s, freq_p95,
                    )
                    param_metrics[_param_key(sigma_0, k_viv, C_viv)] = metrics
                    pbar.update(1)

    best_params, best_score = _select_best_params(param_metrics)
    search_results = _build_search_results_list(param_metrics)
    logger.info(f"{stage_name}完成  当前最优：{best_params}  Macro F1：{best_score:.4f}")
    return best_params, search_results, param_metrics


def run_two_stage_mecc_search(
    features: np.ndarray,
    labels: np.ndarray,
    f0s: np.ndarray,
    freq_p95: float,
    coarse_grid: dict = None,
    enable_fine: bool = ENABLE_FINE_SEARCH,
) -> tuple:
    """粗搜 + 邻域细搜，合并结果后重新选优。"""
    coarse_grid = coarse_grid or COARSE_PARAM_GRID
    param_metrics = {}

    _, _, param_metrics = mecc_hyperparameter_search(
        features, labels, f0s, freq_p95,
        param_grid=coarse_grid,
        stage_name="MECC粗搜",
        param_metrics=param_metrics,
    )
    coarse_best, coarse_score = _select_best_params(param_metrics)
    logger.info(f"粗搜最优：{coarse_best}  Macro F1={coarse_score:.4f}")

    if enable_fine:
        fine_grid = build_fine_param_grid(
            coarse_best['sigma_0'],
            coarse_best['k_viv'],
            coarse_best['C_viv'],
        )
        _, _, param_metrics = mecc_hyperparameter_search(
            features, labels, f0s, freq_p95,
            param_grid=fine_grid,
            stage_name="MECC细搜",
            param_metrics=param_metrics,
        )

    best_params, best_score = _select_best_params(param_metrics)
    search_results = _build_search_results_list(param_metrics)
    logger.info(
        f"两阶段搜索完成：共 {len(param_metrics)} 组 | "
        f"最优 {best_params}  Macro F1={best_score:.4f}"
    )
    return best_params, search_results, param_metrics


# ===================== 绘图 =====================
def plot_mecc_f1_contour(
    search_results: list,
    best_params: dict,
    ploter: PlotLib,
    param_grid: dict = None,
    group_id: str = None,
):
    best_sigma = best_params['sigma_0']
    if param_grid is None:
        param_grid = grid_from_search_results(search_results, sigma_0=best_sigma)

    k_viv_vals = param_grid['k_viv']
    C_viv_vals = param_grid['C_viv']

    f1_lookup = {
        _param_key(r['params']['sigma_0'], r['params']['k_viv'], r['params']['C_viv']): r['f1']
        for r in search_results
    }

    f1_matrix = np.full((len(k_viv_vals), len(C_viv_vals)), np.nan)
    for ki, k in enumerate(k_viv_vals):
        for ci, c in enumerate(C_viv_vals):
            f1_matrix[ki, ci] = f1_lookup.get(_param_key(best_sigma, k, c), np.nan)

    X, Y = np.meshgrid(C_viv_vals, k_viv_vals)

    best_k = best_params['k_viv']
    best_C = best_params['C_viv']
    best_f1 = f1_lookup.get(
        _param_key(best_sigma, best_params['k_viv'], best_params['C_viv']),
        np.nanmax(f1_matrix),
    )

    print(f"\n{'=' * 80}")
    print(f"【MECC - 最优参数】{group_id or ''}")
    print(f"{'=' * 80}")
    print(f"  sigma_0 = {best_sigma}")
    print(f"  k_viv   = {best_k}")
    print(f"  C_viv   = {best_C}")
    print(f"  Best Macro F1 = {best_f1:.4f}")
    print(f"  （等高线固定 sigma_0={best_sigma}，展示 k–C 平面）")
    print(f"{'=' * 80}\n")

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    cf = ax.contourf(X, Y, f1_matrix, cmap=CMAP_CONTOUR, levels=20, alpha=1.0)
    ax.contour(X, Y, f1_matrix, levels=20, colors='black', linewidths=0.5, alpha=0.3)

    ax.scatter([best_C], [best_k], color='red', s=80, zorder=5,
               label=f'Best Macro F1={best_f1:.3f}')
    ax.legend(prop=ENG_FONT, loc='upper right')
    ax.set_xlabel(r"$C_{MECC}$", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel(r"$k_{MECC}$",                  labelpad=10, fontproperties=ENG_FONT)
    ax.tick_params(axis='both', labelsize=SQUARE_FONT_SIZE - 4)

    cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label('Macro F1 Score', fontproperties=ENG_FONT)
    cbar.ax.tick_params(labelsize=SQUARE_FONT_SIZE - 4)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(ENG_FONT)

    plt.tight_layout()
    ploter.figs.append(fig)

    suffix = f"_{group_id}" if group_id else ""
    save_path = os.path.join(SAVE_DIR, f'mecc_contour_viv_params{suffix}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"等高线图已保存：{save_path}")
    return fig


def _plot_confusion_matrix(
    best_params: dict,
    macro_f1: float,
    cm: np.ndarray,
    save_name: str,
    ploter: PlotLib,
    group_label: str = None,
):
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=CMAP,
        xticklabels=LABELS, yticklabels=LABELS,
        ax=ax, cbar=False, annot_kws={'size': 20}
    )
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('True Label',      labelpad=10, fontproperties=ENG_FONT)
    param_str = (
        f"sigma_0={best_params['sigma_0']}, "
        f"k_viv={best_params['k_viv']}, C_viv={best_params['C_viv']}"
    )
    title_prefix = f"{group_label}\n" if group_label else ""
    ax.set_title(
        f"{title_prefix}{param_str}\nMacro F1: {macro_f1:.3f}",
        pad=20, fontproperties=ENG_FONT
    )
    plt.tight_layout()
    ploter.figs.append(fig)

    save_path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"混淆矩阵已保存：{save_path}")


# ===================== 缓存读取 =====================
def _all_group_caches_exist() -> bool:
    return all(os.path.exists(_group_result_path(gid)) for gid in CABLE_GROUPS)


def load_mecc_group_cache(group_id: str):
    path = _group_result_path(group_id)
    if not os.path.exists(path):
        return None, None, None, None
    logger.info(f"发现缓存文件，直接加载：{path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best_params = {
        'sigma_0': _normalize_sigma_0(data['best_params']['sigma_0']),
        'k_viv':   _normalize_k_viv(data['best_params']['k_viv']),
        'C_viv':   _normalize_c_viv(data['best_params']['C_viv']),
    }
    search_results = data['search_results']
    param_metrics  = {}
    for k_str, v in data['param_metrics'].items():
        entry = {
            'weighted':         v['weighted'],
            'per_class':        v['per_class'],
            'confusion_matrix': np.array(v['confusion_matrix']),
        }
        if 'macro' in v:
            entry['macro'] = v['macro']
        param_metrics[_parse_param_key(k_str)] = entry
    freq_p95 = float(data['freq_p95'])
    logger.info(f"缓存加载完成 [{group_id}]，最优参数：{best_params}")
    return best_params, search_results, param_metrics, freq_p95


# ===================== 结果存储 =====================
def save_mecc_group_results(
    group_id: str,
    best_params: dict,
    search_results: list,
    param_metrics: dict,
    freq_p95: float,
    dataset_size: int,
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    cfg = CABLE_GROUPS[group_id]

    result_data = {
        'group_id':          group_id,
        'group_label':       cfg['label'],
        'mount_points':      sorted(cfg['mount_points']),
        'search_metric':     SEARCH_METRIC,
        'cable_config_path': CABLE_CONFIG_PATH,
        'freq_p95':          freq_p95,
        'dataset_size':      dataset_size,
        'best_params':       best_params,
        'best_f1':           max(r['f1'] for r in search_results) if search_results else 0.0,
        'search_results': [
            {
                'params':      r['params'],
                'precision':   r['precision'],
                'recall':      r['recall'],
                'f1':          r['f1'],
                'weighted_f1': r.get('weighted_f1', r['f1']),
            }
            for r in search_results
        ],
        'param_metrics': {
            _param_key_to_str(k[0], k[1], k[2]): {
                'weighted':         v['weighted'],
                'macro':            v.get('macro', {
                    'F1': _compute_search_score(v),
                }),
                'per_class':        v['per_class'],
                'confusion_matrix': v['confusion_matrix'].tolist(),
            }
            for k, v in param_metrics.items()
        },
    }

    path = _group_result_path(group_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    logger.info(f"MECC搜索结果已保存：{path}")


def save_mecc_summary(all_group_results: dict, freq_p95: float):
    summary = {
        'search_metric': SEARCH_METRIC,
        'freq_p95':      freq_p95,
        'groups':        all_group_results,
        'generated_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULT_SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    logger.info(f"MECC汇总结果已保存：{RESULT_SUMMARY_PATH}")


def write_result_file(
    group_id: str,
    group_label: str,
    best_params: dict,
    best_metrics: dict,
    search_results: list,
    dataset_size: int,
    freq_p95: float,
):
    result_path = os.path.join(SAVE_DIR, f'result_{group_id}.txt')
    macro_f1 = _compute_search_score(best_metrics)
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MECC参数搜索结果报告 — {group_label} ({group_id})\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"搜参目标: {SEARCH_METRIC}（Macro F1，两类等权）\n")
        f.write(f"数据集配置: {CONFIG_PATH}\n")
        f.write(f"基频配置: {CABLE_CONFIG_PATH}（按 sensor_id 查表）\n")
        f.write(f"freq_p95={freq_p95:.4f} Hz（主导模态95%分位阈值，固定，不参与搜索）\n")
        f.write(f"有效样本数（label 0/1）: {dataset_size}\n")
        f.write(
            f"粗搜网格: sigma_0 step={COARSE_SIGMA_STEP}, range=[{SIGMA_0_MIN}, {SIGMA_0_MAX}]\n"
        )
        f.write(f"         k_viv step={COARSE_K_STEP}, range=[{K_VIV_MIN}, {K_VIV_MAX}]\n")
        f.write(
            f"         C_viv step={COARSE_C_STEP}, range=[{C_VIV_MIN}, {C_VIV_MAX}]\n"
        )
        if ENABLE_FINE_SEARCH:
            f.write(
                f"细搜配置: sigma±{FINE_SIGMA_RADIUS} step={FINE_SIGMA_STEP}, "
                f"k±{FINE_K_RADIUS} step={FINE_K_STEP}, "
                f"C±{FINE_C_RADIUS} step={FINE_C_STEP}\n"
            )
        f.write(f"搜索总组合数: {len(search_results)}\n")
        f.write("=" * 80 + "\n\n")

        f.write("【最优参数】\n")
        f.write(f"  sigma_0 = {best_params['sigma_0']}\n")
        f.write(f"  k_viv   = {best_params['k_viv']}\n")
        f.write(f"  C_viv   = {best_params['C_viv']}\n\n")

        f.write("【最优参数详细指标】\n")
        f.write(f"Macro F1:      {macro_f1:.4f}\n")
        f.write(f"加权Precision: {best_metrics['weighted']['Precision']:.4f}\n")
        f.write(f"加权Recall:    {best_metrics['weighted']['Recall']:.4f}\n")
        f.write(f"加权F1 Score:  {best_metrics['weighted']['F1']:.4f}\n")
        f.write("\n类别级指标:\n")
        for cls_name, cls_m in best_metrics['per_class'].items():
            f.write(f"  {cls_name}: P={cls_m['Precision']:.4f}  R={cls_m['Recall']:.4f}  F1={cls_m['F1']:.4f}\n")

        f.write("\n【搜索结果（按 Macro F1 降序，TOP 10）】\n")
        f.write(
            f"{'sigma_0':<8} {'k_viv':<8} {'C_viv':<8} "
            f"{'MacroF1':<10} {'加权F1':<10}\n"
        )
        f.write("-" * 54 + "\n")
        sorted_results = sorted(search_results, key=lambda x: x['f1'], reverse=True)
        for r in sorted_results[:10]:
            p = r['params']
            wf1 = r.get('weighted_f1', r['f1'])
            f.write(
                f"{p['sigma_0']:<8g} {p['k_viv']:<8g} {p['C_viv']:<8.3f} "
                f"{r['f1']:<10.4f} {wf1:<10.4f}\n"
            )

    logger.info(f"结果文件已保存：{result_path}")


def _push_group_figs_to_webui(
    contour_fig,
    cm_fig,
    group_id: str,
    group_label: str,
    webui_page: str,
):
    from src.visualize_tools.web_dashboard import push as web_push

    web_push(
        contour_fig,
        page=webui_page,
        slot=0,
        title=f"{group_label} — k–C 等高线 (Macro F1)",
        port=WEBUI_PORT,
        page_cols=WEBUI_COLS,
    )
    web_push(
        cm_fig,
        page=webui_page,
        slot=1,
        title=f"{group_label} — 混淆矩阵",
        port=WEBUI_PORT,
        page_cols=WEBUI_COLS,
    )
    logger.info(f"WebUI 已推送：page={webui_page!r}  group={group_id}")


def _plot_group_results(
    group_id: str,
    group_label: str,
    best_params: dict,
    search_results: list,
    param_metrics: dict,
    webui_page: str = None,
):
    best_key     = _param_key(
        best_params['sigma_0'], best_params['k_viv'], best_params['C_viv'],
    )
    best_metrics = param_metrics[best_key]
    macro_f1     = _compute_search_score(best_metrics)
    best_sigma   = best_params['sigma_0']
    best_k_viv   = best_params['k_viv']
    best_C_viv   = best_params['C_viv']

    ploter = PlotLib()
    fine_plot_grid = build_fine_param_grid(best_sigma, best_k_viv, best_C_viv)
    contour_fig = plot_mecc_f1_contour(
        search_results, best_params, ploter,
        param_grid=fine_plot_grid, group_id=group_id,
    )
    _plot_confusion_matrix(
        best_params, macro_f1,
        best_metrics['confusion_matrix'],
        save_name=(
            f"confusion_matrix_{group_id}_"
            f"s{best_sigma:g}_kv{best_k_viv:g}_Cv{best_C_viv:g}.png"
        ),
        ploter=ploter,
        group_label=group_label,
    )
    cm_fig = ploter.figs[-1]

    if PUSH_TO_WEBUI and webui_page:
        _push_group_figs_to_webui(contour_fig, cm_fig, group_id, group_label, webui_page)

    logger.info(
        f"[{group_id}] 最优 sigma_0={best_sigma}, k_viv={best_k_viv}, "
        f"C_viv={best_C_viv}  Macro F1={macro_f1:.4f}"
    )
    return best_params, macro_f1


# ===================== 主流程 =====================
def run_mecc_param_search(
    config_path: str = CONFIG_PATH,
    coarse_grid: dict = None,
    enable_fine: bool = ENABLE_FINE_SEARCH,
):
    """主流程：三组拉索独立搜参（Macro F1）→ 绘图 → WebUI 推送"""
    if coarse_grid is None:
        coarse_grid = COARSE_PARAM_GRID

    os.makedirs(SAVE_DIR, exist_ok=True)
    webui_ts   = datetime.now().strftime('%Y%m%d_%H%M')
    webui_page = f"MECC搜参 {webui_ts}"

    if PUSH_TO_WEBUI:
        from src.visualize_tools.web_dashboard import _get_or_start_dashboard
        _get_or_start_dashboard(
            port=WEBUI_PORT, cols=WEBUI_COLS, max_rounds=20,
        )
        logger.info(f"WebUI 已就绪：http://localhost:{WEBUI_PORT}  page={webui_page!r}")

    any_need_search = FORCE_RECOMPUTE or any(
        not os.path.exists(_group_result_path(gid)) for gid in CABLE_GROUPS
    )
    val_features = val_labels = val_f0s = val_sensor_ids = None
    freq_p95 = None

    if any_need_search:
        logger.info("=" * 60)
        logger.info("开始 MECC 三组拉索独立搜参（目标：Macro F1）")
        logger.info("=" * 60)

        dataset_config = load_dataset_config(config_path)
        dataset = AnnotationDataset(dataset_config)
        logger.info(f"数据集加载完成，类别数：{dataset.get_num_classes()}")

        val_features, val_labels, val_f0s, val_sensor_ids = extract_val_search_data(dataset)
        val_features, val_labels, val_f0s, val_sensor_ids = filter_binary_samples(
            val_features, val_labels, val_f0s, val_sensor_ids,
        )
        logger.info(
            f"验证集样本总数（label 0/1）：{len(val_labels)}  "
            f"Normal={(val_labels == 0).sum()}  VIV={(val_labels == 1).sum()}"
        )
        logger.info(
            f"基频 f0 范围：[{val_f0s.min():.4f}, {val_f0s.max():.4f}] Hz "
            f"（unique={len(np.unique(val_f0s))}）"
        )

        freq_p95 = compute_freq_p95()
        logger.info(f"主导模态 95% 分位阈值 freq_p95 = {freq_p95:.4f} Hz")

    all_group_results = {}

    for group_id, cfg in CABLE_GROUPS.items():
        group_label = cfg['label']
        logger.info("-" * 60)
        logger.info(f"拉索组 [{group_id}] {group_label}")

        if FORCE_RECOMPUTE or not os.path.exists(_group_result_path(group_id)):
            gf, gl, gf0 = _subset_by_group(
                val_features, val_labels, val_f0s, val_sensor_ids, group_id,
            )
            best_params, search_results, param_metrics = run_two_stage_mecc_search(
                gf, gl, gf0, freq_p95,
                coarse_grid=coarse_grid,
                enable_fine=enable_fine,
            )
            best_key     = _param_key(
                best_params['sigma_0'], best_params['k_viv'], best_params['C_viv'],
            )
            best_metrics = param_metrics[best_key]
            save_mecc_group_results(
                group_id, best_params, search_results, param_metrics,
                freq_p95, len(gl),
            )
            write_result_file(
                group_id, group_label, best_params, best_metrics,
                search_results, len(gl), freq_p95,
            )
        else:
            logger.info(f"使用缓存 [{group_id}]，跳过搜参")
            best_params, search_results, param_metrics, freq_p95 = load_mecc_group_cache(group_id)

        best_params, macro_f1 = _plot_group_results(
            group_id, group_label, best_params, search_results, param_metrics,
            webui_page=webui_page if PUSH_TO_WEBUI else None,
        )
        all_group_results[group_id] = {
            'label':       group_label,
            'best_params': best_params,
            'macro_f1':    macro_f1,
        }

    if freq_p95 is None and all_group_results:
        _, _, _, freq_p95 = load_mecc_group_cache(next(iter(CABLE_GROUPS)))
    if freq_p95 is not None:
        save_mecc_summary(all_group_results, freq_p95)

    logger.info("=" * 60)
    logger.info("MECC 三组拉索搜参完成")
    for gid, info in all_group_results.items():
        bp = info['best_params']
        logger.info(
            f"  [{gid}] sigma_0={bp['sigma_0']}, k_viv={bp['k_viv']}, "
            f"C_viv={bp['C_viv']}  Macro F1={info['macro_f1']:.4f}"
        )
    logger.info("=" * 60)

    return all_group_results


if __name__ == "__main__":
    run_mecc_param_search()
