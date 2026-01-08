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
# --- 修改点 1: 导入 LinearSegmentedColormap 用于创建自定义渐变 ---
from matplotlib.colors import LinearSegmentedColormap

# ===================== 全局配置与常量定义 =====================
# 数据集路径（抽离为常量，便于修改）
DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\data\\'
IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\img\\'

# 标签映射（剔除过渡态，仅保留有效类别）
LABEL_MAP = {0: 'Normal Vibration', 2: 'VIV'}
LABELS = list(LABEL_MAP.values())  # ['Normal Vibration', 'VIV']
LABEL_IDS = [0, 2]  # 对应标签的数字ID

# ===================== 核心：最优参数定义（重点） =====================
BEST_K = 5  # 最优k值
BEST_THRESHOLD = 0.1  # 最优MECC_0阈值

# --- 新增点 1: 帕累托最优验证的参数网格 ---
K_VALUES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
THRESHOLD_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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

# ===================== 新增：帕累托最优验证相关函数 (来自参考代码3) =====================

def calculate_metrics_for_params(k, threshold, dataset, mount):
    """
    计算指定参数下的F1分数和混淆矩阵
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
    
    f1 = f1_score(true_labels, pred_labels, average='weighted', labels=LABEL_IDS, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_IDS)
    return f1, cm

def plot_confusion_matrix(cm, k, threshold, f1_score_val):
    """
    绘制混淆矩阵热力图
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
    return fig

def run_pareto_verification_and_plot():
    """
    主流程：帕累托最优验证 + 热力图 + 最优参数混淆矩阵
    """
    print("="*60)
    print("开始执行帕累托最优验证...")
    print("="*60)
    dataset = ECC_Dataset(data_dir=DATA_DIR, img_dir=IMG_DIR)
    mount = Cal_Mount()
    ploter = PlotLib()

    f1_matrix = np.zeros((len(K_VALUES), len(THRESHOLD_VALUES)))
    cm_dict = {}
    total_combinations = len(K_VALUES) * len(THRESHOLD_VALUES)
    
    with tqdm(total=total_combinations, desc="总进度") as pbar_total:
        for i, k in enumerate(K_VALUES):
            for j, threshold in enumerate(THRESHOLD_VALUES):
                try:
                    f1, cm = calculate_metrics_for_params(k, threshold, dataset, mount)
                    f1_matrix[i, j] = f1
                    cm_dict[(k, threshold)] = cm
                except Exception as e:
                    print(f"\n参数组合(k={k}, threshold={threshold})计算失败：{str(e)}")
                    f1_matrix[i, j] = np.nan
                    cm_dict[(k, threshold)] = None
                pbar_total.update(1)

    # 绘制参数-F1热力图
    fig, ax = plt.subplots(figsize=(16, 8))
    f1_matrix_reversed = np.flipud(f1_matrix)
    k_labels_reversed = K_VALUES[::-1]
    
    sns.heatmap(
        f1_matrix_reversed,
        annot=True,
        fmt='.2f',
        cmap=CMAP,
        xticklabels=THRESHOLD_VALUES,
        yticklabels=k_labels_reversed,
        ax=ax,
        cbar_kws={'label': 'F1 Score', 'shrink': 0.8},
        mask=np.isnan(f1_matrix_reversed),
        annot_kws={'size': 12} # 为了防止标注重叠，适当减小字号
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"$C'_{MECC}$", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel(r'$k$', labelpad=10, fontproperties=ENG_FONT)
    plt.tight_layout()
    ploter.figs.append(fig)

    # 筛选最优参数并绘制混淆矩阵
    f1_threshold = 0.9
    high_perf_mask = f1_matrix >= f1_threshold
    high_perf_indices = np.argwhere(high_perf_mask)
    high_perf_params = [(K_VALUES[i], THRESHOLD_VALUES[j]) for i, j in high_perf_indices]
    
    print(f"\n===== 高F1值参数组合筛选结果（F1≥{f1_threshold}） =====")
    if len(high_perf_params) > 0:
        print(f"共找到{len(high_perf_params)}个最优参数组合：")
        for idx, (k, threshold) in enumerate(high_perf_params):
            f1_value = f1_matrix[K_VALUES.index(k), THRESHOLD_VALUES.index(threshold)]
            cm = cm_dict[(k, threshold)]
            print(f"{idx+1}. k={k}, threshold={threshold}, F1={f1_value:.3f}")
            if cm is not None:
                cm_fig = plot_confusion_matrix(cm, k, threshold, f1_value)
                ploter.figs.append(cm_fig)
    else:
        max_f1_idx = np.nanargmax(f1_matrix)
        max_i, max_j = np.unravel_index(max_f1_idx, f1_matrix.shape)
        best_k = K_VALUES[max_i]
        best_threshold = THRESHOLD_VALUES[max_j]
        best_f1 = f1_matrix[max_i, max_j]
        best_cm = cm_dict[(best_k, best_threshold)]
        print(f"未找到F1≥{f1_threshold}的参数组合，当前最高F1值为：{best_f1:.3f}（k={best_k}, threshold={best_threshold}）")
        if best_cm is not None:
            cm_fig = plot_confusion_matrix(best_cm, best_k, best_threshold, best_f1)
            ploter.figs.append(cm_fig)

    ploter.show()
    return f1_matrix, K_VALUES, THRESHOLD_VALUES, cm_dict


# ===================== 执行主流程 =====================
if __name__ == "__main__":
    # 优先执行最优参数评估（核心）
    best_metrics = evaluate_best_params()
    
    # 如需运行帕累托最优验证，取消下方注释
    # f1_matrix, k_list, threshold_list, cm_dict = run_pareto_verification_and_plot()