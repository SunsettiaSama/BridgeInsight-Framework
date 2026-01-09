# ===================== 统一导入所有依赖库 =====================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import os
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import concurrent.futures
import pickle
import logging  # ✅ 新增：导入logging模块
# 注意：若自定义模块路径报错，请根据实际项目结构调整
from ..NN.datasets.VIVImgDataset import MyDataset
from ..visualize_tools.utils import PlotLib
from ..method.base_mode_calculator import Cal_Mount
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing



# ===================== 全局配置与常量定义 =====================
# 数据集路径（抽离为常量，便于修改）
DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\data\\'
IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\img\\'
# 结果保存路径
SAVE_DIR = r"F:\Research\Vibration Characteristics In Cable Vibration\outputs\mecc_pareto_result"
RESULT_NPZ_PATH = os.path.join(SAVE_DIR, 'pareto_search_results.npz')
RESULT_TXT_PATH = os.path.join(SAVE_DIR, 'result.txt')
CONTOUR_PLOT_PATH = os.path.join(SAVE_DIR, 'mecc_pareto_contour_plot.png')
METRICS_PKL_PATH = os.path.join(SAVE_DIR, 'metrics_dict.pkl')  # 保存完整metrics字典

# ✅ 重新配置logger（确保SAVE_DIR已定义）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SAVE_DIR, 'param_search.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 标签映射（剔除过渡态，仅保留有效类别）
LABEL_MAP = {0: 'Normal Vibration', 2: 'VIV'}
LABELS = list(LABEL_MAP.values())  # ['Normal Vibration', 'VIV']
LABEL_IDS = [0, 2]  # 对应标签的数字ID

# 帕累托最优验证的参数网格
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
THRESHOLD_VALUES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 调试用参数（需要时取消注释）
# DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\data\\'
# IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\img\\'
# K_VALUES = [1, 2]
# THRESHOLD_VALUES = [0.1, 0.2]

# 秒级切分区间
INTERVAL = (45, 46)
SECOND_INTERVAL = (10, 20)

# ===================== 工具函数定义 =====================
def create_custom_gray_cmap():
    """创建从深灰色到纯白色的自定义颜色映射"""
    colors = ['#808080', '#FFFFFF']
    cmap = LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
    return cmap

# 初始化自定义色图和字体配置
CMAP = create_custom_gray_cmap()
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False
ENG_FONT = FontProperties(family='Times New Roman', size=20)
CN_FONT = FontProperties(family='SimHei', size=20)

# ===================== 数据集类定义 =====================
class ECC_Dataset(MyDataset):
    """MECC方法专用数据集类，加载振动加速度数据与标签（过滤过渡态）"""
    def __init__(self, data_dir, img_dir):
        super().__init__(data_dir=data_dir, img_dir=img_dir)
        self._filter_transition_samples()
    
    def _filter_transition_samples(self):
        """过滤所有标签为1（过渡态）的样本，仅保留0/Normal、2/VIV"""
        valid_paths = []
        for path in self.paths:
            mat_data = loadmat(path)
            label_data_pair = [(k, v) for k, v in mat_data.items() if isinstance(v, np.ndarray)]
            if not label_data_pair:
                continue
            label, _ = label_data_pair[0]
            if int(label) in LABEL_MAP.keys():
                valid_paths.append(path)
        self.paths = valid_paths
        if len(self.paths) == 0:
            raise ValueError("过滤后无有效样本（无Normal/VIV类数据）")
    
    def __getitem__(self, index):
        """重载获取样本逻辑：返回(振动数据, 标签)，仅含0/2"""
        path = self.paths[index]
        mat_data = loadmat(path)
        label_data_pair = [(k, v) for k, v in mat_data.items() if isinstance(v, np.ndarray)]
        if not label_data_pair:
            raise ValueError(f"样本{path}未找到有效标签/数据")
        label, data = label_data_pair[0]
        label_int = int(label)
        if label_int not in LABEL_MAP.keys():
            raise ValueError(f"样本{path}标签{label_int}非有效类别（仅0/2）")
        return data, label_int

# ===================== MECC分类核心函数 =====================
def mecc_classify(fxtopk, pxxtopk, k, threshold, base_freq):
    """MECC分类函数：区分Normal/VIV两类（全局统一版本）"""
    try:
        fxtopk = np.array(fxtopk, dtype=np.float64)
        pxxtopk = np.array(pxxtopk, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"fxtopk/pxxtopk类型错误：{e}")
    
    if k == 0 or len(fxtopk) <= 1 or len(pxxtopk) <= 1:
        return 0
    
    f0 = fxtopk[0]
    freq_lower = f0 - k * base_freq
    freq_upper = f0 + k * base_freq
    
    mask_outside = (fxtopk < freq_lower) | (fxtopk > freq_upper)
    pxxtopk_outside = pxxtopk[mask_outside]
    
    if len(pxxtopk_outside) == 0:
        return 2
    max_pxx_outside = np.max(pxxtopk_outside)
    energy_ratio = max_pxx_outside / pxxtopk[0]
    return 2 if energy_ratio < threshold else 0

# ===================== 指标计算函数 =====================
def calculate_metrics(y_true, y_pred, labels=LABEL_IDS):
    """计算F1/Precision/Recall（类别级+全局加权）（全局统一版本）"""
    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    
    metrics = {
        'per_class': {
            LABELS[i]: {'Precision': precision_per_class[i], 'Recall': recall_per_class[i], 'F1': f1_per_class[i]}
            for i in range(len(LABELS))
        },
        'weighted': {'Precision': precision_weighted, 'Recall': recall_weighted, 'F1': f1_weighted},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels)
    }
    return metrics

# ===================== 子进程专用计算函数（修复版） =====================
def calculate_single_param_combination(args):
    """
    子进程专用计算函数（使用全局统一的分类/指标函数，避免重定义）
    args: (k, threshold, data_dir, img_dir)
    """
    # ✅ 强化：子进程完全禁用日志和打印（核心修改2）
    import logging
    logging.getLogger().setLevel(logging.CRITICAL + 1)  # 禁用所有日志
    import warnings
    warnings.filterwarnings('ignore')
    import sys
    sys.stdout = open(os.devnull, 'w')  # 重定向stdout到空
    sys.stderr = open(os.devnull, 'w')  # 重定向stderr到空
    
    k, threshold, data_dir, img_dir = args
    try:
        # 子进程独立初始化数据集（使用全局定义的ECC_Dataset）
        dataset = ECC_Dataset(data_dir=data_dir, img_dir=img_dir)
        mount = Cal_Mount()
        base_freq = mount.multi_nums

        true_labels = []
        pred_labels = []
        
        # MECC核心计算逻辑（使用全局mecc_classify）
        for data, label in dataset:
            true_labels.append(label)
            fx, pxxden = signal.welch(data[0, :], fs=50, nfft=65536, nperseg=2048, noverlap=1)
            fxtopk, pxxtopk, _, _ = mount.peaks(fx, pxxden, return_intervals=True)
            pred_label = mecc_classify(fxtopk, pxxtopk, k, threshold, base_freq)
            pred_labels.append(pred_label)
        
        # 计算指标（使用全局calculate_metrics）
        metrics = calculate_metrics(true_labels, pred_labels)
        return (k, threshold), metrics, metrics['weighted']['F1']
    
    except Exception as e:
        # ✅ 移除：子进程内的print语句（核心修改3）
        # 仅返回异常状态，主进程统一处理
        return (k, threshold), None, np.nan
    finally:
        # 恢复stdout/stderr（可选）
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

# ===================== 多进程参数搜索函数（带详细日志） =====================
def multi_process_param_search(data_dir, img_dir, k_values, threshold_values, max_workers=8):
    """
    修复版多进程参数搜索函数（添加详细日志）
    参数：
        data_dir: 数据集路径
        img_dir: 图片路径
        k_values: k值列表
        threshold_values: 阈值列表
        max_workers: 最大进程数（建议4-8）
    返回：
        f1_matrix, metrics_dict
    """
    # 强制限制进程数（Windows下避免资源耗尽）
    max_workers = min(max_workers, 8)
    
    # ========== 详细日志打印（替换为logger.info） ==========
    logger.info("\n" + "="*80)
    logger.info("📋 多进程参数搜索配置详情")
    logger.info("="*80)
    logger.info(f"CPU核心数: {os.cpu_count()} | 启用进程数: {max_workers}")
    logger.info(f"k值范围: [{min(k_values)}, {max(k_values)}] | 数量: {len(k_values)}")
    logger.info(f"阈值范围: [{min(threshold_values)}, {max(threshold_values)}] | 数量: {len(threshold_values)}")
    total_combinations = len(k_values) * len(threshold_values)
    logger.info(f"总参数组合数: {total_combinations}")
    logger.info(f"数据集路径: {data_dir}")
    logger.info(f"图片路径: {img_dir}")
    logger.info("="*80 + "\n")

    # 构建参数组合
    param_combinations = [(k, threshold, data_dir, img_dir) 
                         for k in k_values 
                         for threshold in threshold_values]
    
    f1_matrix = np.zeros((len(k_values), len(threshold_values)))
    metrics_dict = {}
    
    # Windows下强制使用spawn启动方式
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # 提交任务
        futures = {
            executor.submit(calculate_single_param_combination, args): args 
            for args in param_combinations
        }
        
        # 进度条（主进程保留，仅用于可视化进度）
        with tqdm(total=len(futures), desc="🔍 多进程参数搜索", ncols=120) as pbar:
            completed = 0
            failed = 0
            for future in as_completed(futures):
                try:
                    (k, threshold), metrics, f1 = future.result(timeout=300)  # 5分钟超时
                    # 填充结果矩阵
                    k_idx = k_values.index(k)
                    th_idx = threshold_values.index(threshold)
                    f1_matrix[k_idx, th_idx] = f1
                    metrics_dict[(k, threshold)] = metrics
                    
                    completed += 1
                    # 每完成10个组合打印进度（替换为logger.info）
                    if completed % 10 == 0:
                        logger.info(f"✅ 已完成 {completed}/{len(futures)} 组合 (失败: {failed})")
                        
                except Exception as e:
                    args = futures[future]
                    k, threshold = args[0], args[1]
                    logger.error(f"⚠️ 组合(k={k}, threshold={threshold})超时/失败: {str(e)}")
                    metrics_dict[(k, threshold)] = None
                    failed += 1
                pbar.update(1)
    
    # 搜索完成日志（替换为logger.info）
    logger.info(f"\n📊 搜索完成统计: 总组合{len(futures)} | 成功{completed} | 失败{failed}")
    valid_f1 = [v for v in f1_matrix.flatten() if not np.isnan(v)]
    if valid_f1:
        logger.info(f"F1分数范围: [{np.min(valid_f1):.4f}, {np.max(valid_f1):.4f}] | 平均值: {np.mean(valid_f1):.4f}")
    
    return f1_matrix, metrics_dict

# ===================== 结果可视化函数（主进程执行） =====================
def plot_pareto_contour(k_values, threshold_values, f1_matrix, ploter):
    """封装MECC双参数帕累托等高线图绘制逻辑（主进程执行）"""
    logger.info("\n🎨 开始绘制等高线图...")
    fig, ax = plt.subplots(figsize=(12, 8))
    X, Y = np.meshgrid(threshold_values, k_values)
    
    # 绘制等高线
    contour_plot = ax.contourf(
        X, Y, f1_matrix,
        cmap=CMAP,
        levels=20,
        alpha=1.0
    )
    ax.contour(X, Y, f1_matrix, levels=20, colors='black', linewidths=0.5, alpha=0.3)
    
    # 设置标签和刻度
    ax.set_xlabel(r"$C'_{MECC}$", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel(r'$k$', labelpad=10, fontproperties=ENG_FONT)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.01, 0.1)], fontproperties=ENG_FONT)
    k_min = min(k_values)
    k_max = max(k_values)
    ax.set_yticks(np.arange(k_min, k_max+1, 2))
    ax.set_yticklabels([f'{int(x)}' for x in np.arange(k_min, k_max+1, 2)], fontproperties=ENG_FONT)
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # 颜色条
    cbar = plt.colorbar(contour_plot, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', fontproperties=ENG_FONT)
    plt.tight_layout()
    ploter.figs.append(fig)  # ✅ 保留：添加到ploter
    
    # 保存图片
    fig.savefig(CONTOUR_PLOT_PATH, dpi=300, bbox_inches='tight')
    logger.info(f"✅ 等高线图已保存至：{CONTOUR_PLOT_PATH}")
    
    # ✅ 移除：plt.show()（核心修改4）
    # 改为在主进程末尾调用ploter.show()
    return fig

def plot_mecc_confusion_matrix(cm, k, threshold, f1_score_val, ploter, save_dir):
    """封装MECC混淆矩阵绘制与保存逻辑"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=CMAP,
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        cbar=False,
        annot_kws={'size': 20}
    )
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('True Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_title(f'Confusion Matrix (k={k}, MECC₀={threshold})\nWeighted F1: {f1_score_val:.3f}', pad=20, fontproperties=ENG_FONT)
    plt.tight_layout()
    ploter.figs.append(fig)  # ✅ 保留：添加到ploter
    
    cm_save_path = os.path.join(save_dir, f'confusion_matrix_k{k}_th{threshold:.2f}.png')
    fig.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ 混淆矩阵已保存至：{cm_save_path}")

# ===================== 结果文件读写函数 =====================
def load_mecc_results():
    """读取已保存的MECC帕累托搜索结果"""
    if not os.path.exists(RESULT_NPZ_PATH):
        return None, None, None, None, None, None
    try: 
        data = np.load(RESULT_NPZ_PATH, allow_pickle=True)
        k_values = data['K_VALUES']
        threshold_values = data['THRESHOLD_VALUES']
        f1_matrix = data['f1_matrix']
        best_params = data['best_params'].item()
        best_metrics = data['best_metrics'].item()
        
        # 加载metrics字典
        if os.path.exists(METRICS_PKL_PATH):
            with open(METRICS_PKL_PATH, 'rb') as f:
                metrics_dict = pickle.load(f)
        else:
            metrics_dict = None
        
        logger.info(f"✅ 加载历史结果 - k数量：{len(k_values)}, 阈值数量：{len(threshold_values)}")
        return k_values, threshold_values, f1_matrix, best_params, best_metrics, metrics_dict
    except Exception as e:
        logger.error(f"❌ 加载历史结果失败：{e}")
        return None, None, None, None, None, None

def write_result_file(save_dir, k_values, threshold_values, f1_matrix, metrics_dict, dataset_size):
    """完善的结果写入逻辑（包含详细参数信息）"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, 'result.txt')
    
    # 扁平化参数组合并筛选有效值
    f1_flat = f1_matrix.flatten()
    param_pairs = [(k_values[i], threshold_values[j]) for i in range(len(k_values)) for j in range(len(threshold_values))]
    valid_pairs = [(k, th, f1) for (k, th), f1 in zip(param_pairs, f1_flat) 
                  if not np.isnan(f1) and metrics_dict.get((k, th)) is not None]
    
    # 按F1分数降序排序
    valid_pairs_sorted = sorted(valid_pairs, key=lambda x: x[2], reverse=True)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        # 基础信息
        f.write("="*80 + "\n")
        f.write(f"MECC双参数（k+threshold）搜索结果报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集路径: {DATA_DIR}\n")
        f.write(f"有效样本数量: {dataset_size}\n")
        f.write(f"搜索k值列表: {k_values}\n")
        f.write(f"搜索阈值列表: {threshold_values}\n")
        f.write(f"总参数组合数: {len(k_values) * len(threshold_values)}\n")
        f.write(f"有效计算组合数: {len(valid_pairs)}\n")
        f.write("="*80 + "\n\n")
        
        # 最优结果
        if len(valid_pairs_sorted) > 0:
            best_k, best_th, best_f1 = valid_pairs_sorted[0]
            best_metrics = metrics_dict[(best_k, best_th)]
            f.write("【最优参数结果（F1最高）】\n")
            f.write(f"k值: {best_k}\n")
            f.write(f"阈值: {best_th}\n")
            f.write(f"加权Precision (P): {best_metrics['weighted']['Precision']:.4f}\n")
            f.write(f"加权Recall (R): {best_metrics['weighted']['Recall']:.4f}\n")
            f.write(f"加权F1 Score: {best_metrics['weighted']['F1']:.4f}\n")
            f.write("="*80 + "\n\n")
            
            # 前十最优结果
            f.write("【前十最优参数组合（按F1分数降序）】\n")
            f.write(f"{'排名':<6} {'k值':<6} {'阈值':<8} {'P':<10} {'R':<10} {'F1':<10}\n")
            f.write("-"*60 + "\n")
            
            top_n = min(10, len(valid_pairs_sorted))
            for idx, (k, th, f1) in enumerate(valid_pairs_sorted[:top_n], 1):
                metrics = metrics_dict[(k, th)]
                p = metrics['weighted']['Precision']
                r = metrics['weighted']['Recall']
                f1_val = metrics['weighted']['F1']
                f.write(f"{idx:<6} {k:<6} {th:<8.2f} {p:<10.4f} {r:<10.4f} {f1_val:<10.4f}\n")
        else:
            f.write("未找到有效参数组合！\n")
    
    logger.info(f"✅ 结果文件已保存至：{result_path}")

# ===================== 主执行函数（完整修复版） =====================
def run_pareto_verification_and_plot():
    """
    修复版主函数：
    1. 使用全局路径常量
    2. 正确处理结果保存/加载
    3. 主进程绘制等高线图
    4. 正确返回结果
    """
    # 1. 初始化配置（使用全局常量，避免路径覆盖）
    logger.info("="*80)
    logger.info("开始执行MECC帕累托最优验证/绘图流程...")
    logger.info("="*80)
    
    # 创建保存目录（解决路径不存在问题）
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. 加载历史结果
    k_vals, th_vals, f1_mat, best_params, best_metrics, metrics_dict = load_mecc_results()
    
    # 3. 若无历史结果则执行多进程搜索
    if f1_mat is None:
        logger.info("❌ 未检测到历史结果文件，启动多进程参数搜索...")
        
        # 执行多进程搜索
        f1_mat, metrics_dict = multi_process_param_search(
            data_dir=DATA_DIR,
            img_dir=IMG_DIR,
            k_values=K_VALUES,
            threshold_values=THRESHOLD_VALUES,
            max_workers=8
        )
        k_vals = K_VALUES
        th_vals = THRESHOLD_VALUES
        
        # 找到最优参数
        f1_flat = f1_mat.flatten()
        param_pairs = [(k_vals[i], th_vals[j]) for i in range(len(k_vals)) for j in range(len(th_vals))]
        valid_pairs = [(k, th, f1) for (k, th), f1 in zip(param_pairs, f1_flat) 
                      if not np.isnan(f1) and metrics_dict.get((k, th)) is not None]
        
        if valid_pairs:
            best_k, best_th, best_f1 = sorted(valid_pairs, key=lambda x: x[2], reverse=True)[0]
            best_params = (best_k, best_th)
            best_metrics = metrics_dict[best_params]
        else:
            best_params = None
            best_metrics = None
        
        # 4. 保存结果（修复字段缺失问题）
        # 保存npz文件（包含所有必要字段）
        np.savez(
            RESULT_NPZ_PATH,
            K_VALUES=k_vals,
            THRESHOLD_VALUES=th_vals,
            f1_matrix=f1_mat,
            best_params=best_params,
            best_metrics=best_metrics
        )
        # 保存metrics字典
        with open(METRICS_PKL_PATH, 'wb') as f:
            pickle.dump(metrics_dict, f)
        logger.info(f"✅ 结果已保存至：{RESULT_NPZ_PATH}")
        
        # 5. 写入结果文本
        dataset = ECC_Dataset(DATA_DIR, IMG_DIR)
        write_result_file(SAVE_DIR, k_vals, th_vals, f1_mat, metrics_dict, len(dataset.paths))
    else:
        logger.info("✅ 已加载历史结果，跳过参数搜索")
    
    # 6. 主进程绘制等高线图（本轮修改）
    ploter = None  # 初始化ploter
    if f1_mat is not None and len(f1_mat) > 0:
        ploter = PlotLib()  # 初始化绘图工具
        plot_pareto_contour(k_vals, th_vals, f1_mat, ploter)
        
        # 绘制最优参数的混淆矩阵
        if best_params and best_metrics:
            best_k, best_th = best_params
            cm = best_metrics['confusion_matrix']
            plot_mecc_confusion_matrix(cm, best_k, best_th, best_metrics['weighted']['F1'], ploter, SAVE_DIR)
    
    # ✅ 新增：主进程末尾调用ploter.show()（核心修改5）
    if ploter and hasattr(ploter, 'show') and len(ploter.figs) > 0:
        logger.info("\n🖼️ 显示所有绘制的图像...")
        ploter.show()
    
    logger.info("\n🎉 所有流程执行完成！")
    return f1_mat, k_vals, th_vals, metrics_dict

# ===================== 执行主流程 =====================
if __name__ == "__main__":
    # Windows多进程必需
    multiprocessing.freeze_support()
    
    # 执行主流程并获取结果
    f1_matrix, k_list, threshold_list, metrics_dict = run_pareto_verification_and_plot()