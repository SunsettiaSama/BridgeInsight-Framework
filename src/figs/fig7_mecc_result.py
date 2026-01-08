# ===================== 统一导入所有依赖库 =====================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
# 注意：若自定义模块路径报错，请根据实际项目结构调整
from ..NN.datasets.VIVImgDataset import MyDataset
from ..visualize_tools.utils import PlotLib
from ..method.base_mode_calculator import Cal_Mount
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
# --- 修改点 0: 新增os导入（用于路径创建） ---
import os
# --- 修改点 1: 导入 LinearSegmentedColormap 用于创建自定义渐变 ---
from matplotlib.colors import LinearSegmentedColormap
# --- 新增点 0: 导入datetime用于记录时间（增强结果文件可读性） ---
from datetime import datetime

# ===================== 全局配置与常量定义 =====================
# 数据集路径（抽离为常量，便于修改）
DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\data\\'
IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\img\\'
# --- 修改点 2: 新增帕累托结果保存路径（核心需求1） ---
SAVE_DIR = r"F:\Research\Vibration Characteristics In Cable Vibration\outputs\pareto_result"
# --- 新增点 1: 定义结果文件路径常量（统一管理，避免硬编码） ---
RESULT_NPZ_PATH = os.path.join(SAVE_DIR, 'pareto_search_results.npz')
RESULT_TXT_PATH = os.path.join(SAVE_DIR, 'result.txt')
CONTOUR_PLOT_PATH = os.path.join(SAVE_DIR, 'mecc_pareto_contour_plot.png')

# 标签映射（剔除过渡态，仅保留有效类别）
LABEL_MAP = {0: 'Normal Vibration', 2: 'VIV'}
LABELS = list(LABEL_MAP.values())  # ['Normal Vibration', 'VIV']
LABEL_IDS = [0, 2]  # 对应标签的数字ID

# ===================== 核心：最优参数定义（重点） =====================
BEST_K = 5  # 最优k值
BEST_THRESHOLD = 0.1  # 最优MECC_0阈值

# --- 新增点 1: 帕累托最优验证的参数网格 ---
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
THRESHOLD_VALUES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 调试用参数（需要时取消注释）
# DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\data\\'
# IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\img\\'
# K_VALUES = [1, 2]
# THRESHOLD_VALUES = [0.1, 0.2]

# --- 新增点 2: 新增秒级切分区间 (来自参考代码1) ---
INTERVAL = (45, 46)
SECOND_INTERVAL = (10, 20)

# --- 修改点 2: 创建自定义灰度 colormap (新实现) ---
def create_custom_gray_cmap():
    """
    创建一个从深灰色到纯白色的自定义颜色映射。
    - 低端（低值）: 中度灰色 #808080
    - 高端（高值）: 纯白色 #FFFFFF
    """
    # 定义颜色节点，从深灰到纯白
    colors = ['#808080', '#FFFFFF']
    # 创建一个具有256个颜色级别的线性分段颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_gray', colors, N=256)
    return cmap

CMAP = create_custom_gray_cmap()

# 绘图配置
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 新增点 4: 定义中、英文的 FontProperties 对象 (来自参考代码1) ---
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
        for path in tqdm(self.paths, desc="过滤过渡态样本"):
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
        print(f"样本过滤完成：保留{len(self.paths)}个有效样本（剔除过渡态）")
    
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
    """
    MECC分类函数：区分Normal/VIV两类
    """
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

# ===================== 核心：指标计算函数 =====================
def calculate_metrics(y_true, y_pred, labels=LABEL_IDS):
    """
    计算F1/Precision/Recall（类别级+全局加权）
    """
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

# ===================== 核心：最优参数评估函数 =====================
def evaluate_best_params():
    """
    基于最优参数（k=5, MECC_0=0.1）计算并输出F1/Precision/Recall
    """
    print("="*60)
    print("开始评估最优参数：k={}, MECC_0={}".format(BEST_K, BEST_THRESHOLD))
    print("="*60)
    dataset = ECC_Dataset(data_dir=DATA_DIR, img_dir=IMG_DIR)
    mount = Cal_Mount()
    base_freq = mount.multi_nums

    true_labels = []
    pred_labels = []
    with tqdm(total=len(dataset), desc="最优参数推理") as pbar:
        for data, label in dataset:
            true_labels.append(label)
            fx, pxxden = signal.welch(data[0, :], fs=50, nfft=65536, nperseg=2048, noverlap=1)
            fxtopk, pxxtopk, _, _ = mount.peaks(fx, pxxden, return_intervals=True)
            pred_label = mecc_classify(fxtopk, pxxtopk, BEST_K, BEST_THRESHOLD, base_freq)
            pred_labels.append(pred_label)
            pbar.update(1)

    metrics = calculate_metrics(true_labels, pred_labels)

    print("\n===== 最优参数指标结果（类别级） =====")
    for cls_name, cls_metrics in metrics['per_class'].items():
        print(f"\n【{cls_name}】")
        print(f"  Precision: {cls_metrics['Precision']:.3f}")
        print(f"  Recall:    {cls_metrics['Recall']:.3f}")
        print(f"  F1 Score:  {cls_metrics['F1']:.3f}")
    
    print("\n===== 最优参数指标结果（全局加权） =====")
    print(f"Weighted Precision: {metrics['weighted']['Precision']:.3f}")
    print(f"Weighted Recall:    {metrics['weighted']['Recall']:.3f}")
    print(f"Weighted F1 Score:  {metrics['weighted']['F1']:.3f}")
    
    print("\n===== 混淆矩阵 =====")
    cm = metrics['confusion_matrix']
    print(f"          预测：{LABELS[0]}  {LABELS[1]}")
    print(f"真实：{LABELS[0]}  {cm[0,0]}      {cm[0,1]}")
    print(f"真实：{LABELS[1]}  {cm[1,0]}      {cm[1,1]}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=CMAP, xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar=False, annot_kws={'size': 20})
    ax.set_xlabel('Predicted Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('True Label', labelpad=10, fontproperties=ENG_FONT)
    ax.set_title(f'Confusion Matrix (k={BEST_K}, MECC₀={BEST_THRESHOLD})\nWeighted F1: {metrics["weighted"]["F1"]:.3f}', pad=20, fontproperties=ENG_FONT)
    plt.tight_layout()
    plt.show()

    return metrics

def plot_pareto_contour(k_values, threshold_values, f1_matrix, ploter):
    """
    封装MECC双参数帕累托等高线图绘制逻辑
    参数：
        k_values: k值列表
        threshold_values: 阈值列表
        f1_matrix: F1分数矩阵 (len(k) × len(threshold))
        ploter: PlotLib实例
    返回：
        fig: 绘制好的图实例
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    # 构建网格数据（用于等高线绘制）
    X, Y = np.meshgrid(threshold_values, k_values)
    # 绘制等高线填充图（灰度），避免数字标注
    contour_plot = ax.contourf(
        X, Y, f1_matrix,
        cmap=CMAP,  # 保留原有自定义灰度映射
        levels=20,  # 等高线层级（越多越精细）
        alpha=1.0   # 不透明，保持原有视觉效果
    )
    # 添加等高线轮廓线（增强层次感，可选）
    ax.contour(X, Y, f1_matrix, levels=20, colors='black', linewidths=0.5, alpha=0.3)
    # 保留原有标签配置
    ax.set_xlabel(r"$C'_{MECC}$", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel(r'$k$', labelpad=10, fontproperties=ENG_FONT)
    
    # ===================== 核心修改：均匀设置横纵坐标刻度 =====================
    # 1. 横坐标（阈值）：按固定步长生成均匀刻度（0到1，步长0.1），适配阈值范围[0,1]
    ax.set_xticks(np.arange(0, 1.01, 0.1))  # 生成0,0.1,0.2,...,1.0的均匀刻度
    ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.01, 0.1)], fontproperties=ENG_FONT)
    
    # 2. 纵坐标（k值）：按固定步长生成均匀刻度（基于k值的最小/最大值，步长2）
    k_min = min(k_values)
    k_max = max(k_values)
    ax.set_yticks(np.arange(k_min, k_max+1, 2))  # 生成k_min, k_min+2,...,k_max的均匀刻度
    ax.set_yticklabels([f'{int(x)}' for x in np.arange(k_min, k_max+1, 2)], fontproperties=ENG_FONT)
    
    # 3. 可选：调整刻度标签显示，避免重叠（增强可读性）
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # ==========================================================================
    
    # 添加颜色条（保留原有配置）
    cbar = plt.colorbar(contour_plot, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', fontproperties=ENG_FONT)
    plt.tight_layout()
    ploter.figs.append(fig)
    
    # 保存等高线图
    fig.savefig(CONTOUR_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"帕累托等高线图已保存至：{CONTOUR_PLOT_PATH}")
    return fig

# --- 新增点 3: 封装MECC混淆矩阵绘制函数 ---
def plot_mecc_confusion_matrix(cm, k, threshold, f1_score_val, ploter, save_dir):
    """
    封装MECC混淆矩阵绘制与保存逻辑
    参数：
        cm: 混淆矩阵
        k: k值
        threshold: 阈值
        f1_score_val: 对应F1分数
        ploter: PlotLib实例
        save_dir: 保存目录
    """
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
    ploter.figs.append(fig)
    
    # 保存混淆矩阵图
    cm_save_path = os.path.join(save_dir, f'confusion_matrix_k{k}_th{threshold}.png')
    fig.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存至：{cm_save_path}")

# ===================== 新增：历史结果读取函数 =====================
# --- 新增点 4: 定义读取MECC结果文件的函数 ---
def load_mecc_results():
    """
    读取已保存的MECC帕累托搜索结果
    返回：
        k_values, threshold_values, f1_matrix, best_k, best_threshold, best_cm
    """
    if not os.path.exists(RESULT_NPZ_PATH):
        return None, None, None, None, None, None
    
    # 读取npz文件
    data = np.load(RESULT_NPZ_PATH, allow_pickle=True)
    k_values = data['K_VALUES']
    threshold_values = data['THRESHOLD_VALUES']
    f1_matrix = data['f1_matrix']
    best_k = data['best_k']
    best_threshold = data['best_threshold']
    best_cm = data['best_cm']
    
    print(f"已找到历史结果文件，读取k数量：{len(k_values)}, 阈值数量：{len(threshold_values)}")
    return k_values, threshold_values, f1_matrix, best_k, best_threshold, best_cm

# ===================== 新增：结果文本报告生成函数 =====================
# --- 新增点 5: 定义MECC结果写入函数 ---
def write_result_file(save_dir, k_values, threshold_values, f1_matrix, best_k, best_threshold, best_metrics, dataset_size):
    """
    将MECC双参数搜索结果写入result.txt文件
    参数：
        save_dir: 保存目录
        k_values: k值列表
        threshold_values: 阈值列表
        f1_matrix: F1分数矩阵
        best_k: 最优k值
        best_threshold: 最优阈值
        best_metrics: 最优参数对应的metrics
        dataset_size: 数据集样本数量
    """
    result_path = os.path.join(save_dir, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        # 1. 写入基础信息
        f.write("="*80 + "\n")
        f.write(f"MECC双参数（k+threshold）搜索结果报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集路径: {DATA_DIR}\n")
        f.write(f"有效样本数量: {dataset_size}\n")
        f.write(f"搜索k值列表: {k_values}\n")
        f.write(f"搜索阈值列表: {threshold_values}\n")
        f.write(f"最优参数: k={best_k}, threshold={best_threshold}\n")
        f.write("="*80 + "\n\n")

        # 2. 写入最优参数详细结果
        f.write("【最优参数详细指标】\n")
        f.write(f"k值: {best_k}, 阈值: {best_threshold}\n")
        f.write(f"加权Precision (P): {best_metrics['weighted']['Precision']:.4f}\n")
        f.write(f"加权Recall (R): {best_metrics['weighted']['Recall']:.4f}\n")
        f.write(f"加权F1 Score: {best_metrics['weighted']['F1']:.4f}\n")
        f.write("\n类别级指标:\n")
        for cls_name, cls_metrics in best_metrics['per_class'].items():
            f.write(f"  {cls_name}:\n")
            f.write(f"    Precision: {cls_metrics['Precision']:.4f}\n")
            f.write(f"    Recall:    {cls_metrics['Recall']:.4f}\n")
            f.write(f"    F1 Score:  {cls_metrics['F1']:.4f}\n")
        f.write(f"\n混淆矩阵:\n")
        cm = best_metrics['confusion_matrix']
        f.write(f"          预测：{LABELS[0]}  {LABELS[1]}\n")
        f.write(f"真实：{LABELS[0]}  {cm[0,0]}      {cm[0,1]}\n")
        f.write(f"真实：{LABELS[1]}  {cm[1,0]}      {cm[1,1]}\n")
        f.write("="*80 + "\n\n")

        # 3. 写入所有参数组合的F1分数（前10个最优组合）
        f.write("【Top10 最优参数组合（按F1分数排序）】\n")
        # 扁平化F1矩阵并排序
        f1_flat = f1_matrix.flatten()
        param_pairs = [(k_values[i], threshold_values[j]) for i in range(len(k_values)) for j in range(len(threshold_values))]
        # 过滤nan值并排序
        valid_pairs = [(k, th, f1) for (k, th), f1 in zip(param_pairs, f1_flat) if not np.isnan(f1)]
        valid_pairs_sorted = sorted(valid_pairs, key=lambda x: x[2], reverse=True)[:10]
        
        f.write(f"{'排名':<6} {'k值':<6} {'阈值':<8} {'F1分数':<12}\n")
        f.write("-"*40 + "\n")
        for idx, (k, th, f1) in enumerate(valid_pairs_sorted, 1):
            f.write(f"{idx:<6} {k:<6} {th:<8.2f} {f1:<12.4f}\n")
    
    print(f"\n结果文件已保存至：{result_path}")

# ===================== 重构：参数指标计算函数 =====================
# --- 修改点 3: 重构calculate_metrics_for_params，返回完整metrics ---
def calculate_metrics_for_params(k, threshold, dataset, mount):
    """
    计算指定MECC参数下的完整指标（P/R/F1+混淆矩阵）
    返回：完整metrics字典、加权F1
    """
    true_labels = []
    pred_labels = []
    base_freq = mount.multi_nums

    with tqdm(total=len(dataset), desc=f"计算 k={k}, 阈值={threshold}", leave=False) as pbar:
        for data, label in dataset:
            true_labels.append(label)
            fx, pxxden = signal.welch(data[0, :], fs=50, nfft=65536, nperseg=2048, noverlap=1)
            fxtopk, pxxtopk, _, _ = mount.peaks(fx, pxxden, return_intervals=True)
            pred_label = mecc_classify(fxtopk, pxxtopk, k, threshold, base_freq)
            pred_labels.append(pred_label)
            pbar.update(1)
    
    # 计算完整metrics（替代原仅返回f1和cm）
    metrics = calculate_metrics(true_labels, pred_labels)
    return metrics, metrics['weighted']['F1']

# ===================== 重构：主流程函数（核心修改） =====================
def run_pareto_verification_and_plot():
    """
    主流程：帕累托最优验证 + 等高线图 + 结果保存 + 最优参数混淆矩阵
    新增逻辑：
        1. 检查是否有历史结果文件
        2. 有则直接读取并绘图；无则执行参数搜索后绘图
    """
    print("="*60)
    print("开始执行MECC帕累托最优验证/绘图流程...")
    print("="*60)
    
    # --- 修改点 4: 创建保存目录（保留） ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 修改点 5: 初始化PlotLib ---
    ploter = PlotLib()
    
    # 第一步：检查是否有历史结果文件
    k_values, threshold_values, f1_matrix, best_k, best_threshold, best_cm = load_mecc_results()
    
    if k_values is not None and f1_matrix is not None:
        # 有历史结果：直接调用封装的画图函数
        print("✅ 检测到历史结果文件，跳过参数搜索，直接绘图...")
        
        # 绘制帕累托等高线图
        plot_pareto_contour(k_values, threshold_values, f1_matrix, ploter)
        
        # 读取result.txt获取最优阈值的详细信息（可选）
        if os.path.exists(RESULT_TXT_PATH):
            with open(RESULT_TXT_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                print("\n📜 历史最优结果摘要：")
                # 简单提取最优参数和F1
                for line in content.split('\n'):
                    if "最优参数:" in line:
                        print(line.strip())
                    if "加权F1 Score:" in line and "最优参数详细指标" in content.split('\n')[content.split('\n').index(line)-2]:
                        print(line.strip())
        
        # 显示图形
        ploter.show()
        return f1_matrix, k_values, threshold_values, None
    
    # 无历史结果：执行完整参数搜索流程
    print("❌ 未检测到历史结果文件，执行双参数搜索...")
    dataset = ECC_Dataset(data_dir=DATA_DIR, img_dir=IMG_DIR)
    mount = Cal_Mount()
    dataset_size = len(dataset)  # 记录数据集大小

    f1_matrix = np.zeros((len(K_VALUES), len(THRESHOLD_VALUES)))
    metrics_dict = {}  # 替换原cm_dict，存储完整metrics: {(k,th): metrics}
    total_combinations = len(K_VALUES) * len(THRESHOLD_VALUES)
    
    with tqdm(total=total_combinations, desc="总进度") as pbar_total:
        for i, k in enumerate(K_VALUES):
            for j, threshold in enumerate(THRESHOLD_VALUES):
                try:
                    # --- 修改点 6: 调用重构后的指标计算函数 ---
                    metrics, f1 = calculate_metrics_for_params(k, threshold, dataset, mount)
                    f1_matrix[i, j] = f1
                    metrics_dict[(k, threshold)] = metrics
                except Exception as e:
                    print(f"\n参数组合(k={k}, threshold={threshold})计算失败：{str(e)}")
                    f1_matrix[i, j] = np.nan
                    metrics_dict[(k, threshold)] = None
                pbar_total.update(1)

    # --- 修改点 7: 调用封装的等高线图绘制函数 ---
    plot_pareto_contour(K_VALUES, THRESHOLD_VALUES, f1_matrix, ploter)

    # --- 修改点 8: 保存MECC搜索结果（保留原有逻辑，补充注释） ---
    np.savez(
        RESULT_NPZ_PATH,
        f1_matrix=f1_matrix,
        K_VALUES=K_VALUES,
        THRESHOLD_VALUES=THRESHOLD_VALUES,
        best_k=BEST_K,
        best_threshold=BEST_THRESHOLD,
        best_cm=metrics_dict.get((BEST_K, BEST_THRESHOLD), {}).get('confusion_matrix', np.array([]))
    )
    print(f"\nMECC帕累托搜索结果已保存至：{RESULT_NPZ_PATH}")

    # 筛选最优参数并绘制混淆矩阵
    f1_threshold = 0.9
    high_perf_mask = f1_matrix >= f1_threshold
    high_perf_indices = np.argwhere(high_perf_mask)
    high_perf_params = [(K_VALUES[i], THRESHOLD_VALUES[j]) for i, j in high_perf_indices]
    
    print(f"\n===== 高F1值参数组合筛选结果（F1≥{f1_threshold}） =====")
    if len(high_perf_params) > 0:
        print(f"共找到{len(high_perf_params)}个最优参数组合：")
        best_k = high_perf_params[0][0]
        best_threshold = high_perf_params[0][1]
        best_metrics = metrics_dict[(best_k, best_threshold)]
        for idx, (k, threshold) in enumerate(high_perf_params):
            f1_value = f1_matrix[K_VALUES.index(k), THRESHOLD_VALUES.index(threshold)]
            cm = metrics_dict[(k, threshold)]['confusion_matrix']
            print(f"{idx+1}. k={k}, threshold={threshold}, F1={f1_value:.3f}")
            if cm is not None:
                # --- 修改点 9: 调用封装的混淆矩阵绘制函数 ---
                plot_mecc_confusion_matrix(cm, k, threshold, f1_value, ploter, SAVE_DIR)
    else:
        max_f1_idx = np.nanargmax(f1_matrix)
        max_i, max_j = np.unravel_index(max_f1_idx, f1_matrix.shape)
        best_k = K_VALUES[max_i]
        best_threshold = THRESHOLD_VALUES[max_j]
        best_f1 = f1_matrix[max_i, max_j]
        best_metrics = metrics_dict[(best_k, best_threshold)]
        print(f"未找到F1≥{f1_threshold}的参数组合，当前最高F1值为：{best_f1:.3f}（k={best_k}, threshold={best_threshold}）")
        if best_metrics is not None:
            cm = best_metrics['confusion_matrix']
            # --- 修改点 10: 调用封装的混淆矩阵绘制函数 ---
            plot_mecc_confusion_matrix(cm, best_k, best_threshold, best_f1, ploter, SAVE_DIR)

    # --- 修改点 11: 写入result.txt文件 ---
    write_result_file(SAVE_DIR, K_VALUES, THRESHOLD_VALUES, f1_matrix, best_k, best_threshold, best_metrics, dataset_size)

    ploter.show()
    return f1_matrix, K_VALUES, THRESHOLD_VALUES, metrics_dict

# ===================== 执行主流程 =====================
if __name__ == "__main__":
    # 优先执行最优参数评估（核心）
    best_metrics = evaluate_best_params()
    
    # 如需运行帕累托最优验证，取消下方注释
    # f1_matrix, k_list, threshold_list, metrics_dict = run_pareto_verification_and_plot()