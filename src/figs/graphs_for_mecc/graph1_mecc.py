# ===================== 统一导入所有依赖库 =====================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import signal
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from src.deep_learning_module.datasets.VIVImgDataset import MyDataset

# 自定义模块（需确保路径/依赖正确）
from src.visualize_tools.utils import PlotLib   # 假设自定义类在utils中，可根据实际路径调整
from src.method.base_mode_calculator import Cal_Mount

# ===================== 全局配置与常量定义 =====================
# 数据集路径（抽离为常量，便于修改）
DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\data\\'
IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\img\\'

# 标签映射（剔除过渡态，仅保留有效类别）
LABEL_MAP = {0: 'Normal Vibration', 2: 'VIV'}
LABELS = list(LABEL_MAP.values())  # ['Normal Vibration', 'VIV']

# 帕累托最优验证的参数网格（可根据需求扩展）
K_VALUES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]  # MECC的k值候选
THRESHOLD_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 原硬编码0.2的阈值候选

# 调试用参数（注释掉，正式运行启用上方）
# K_VALUES = [0, 1]
# THRESHOLD_VALUES = [0, 0.1]
# DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\data\\'
# IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\img\\'

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===================== 数据集类定义 =====================
class ECC_Dataset(MyDataset):
    """MECC方法专用数据集类，加载振动加速度数据与标签（过滤过渡态）"""
    def __init__(self, data_dir, img_dir):
        super().__init__(data_dir=data_dir, img_dir=img_dir)
        # 预处理：过滤掉过渡态样本路径（label=1）
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
            if int(label) in LABEL_MAP.keys():  # 仅保留0/2标签样本
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
        # 二次校验：确保标签为0/2（防止过滤遗漏）
        if label_int not in LABEL_MAP.keys():
            raise ValueError(f"样本{path}标签{label_int}非有效类别（仅0/2）")
        return data, label_int

def mecc_classify(fxtopk, pxxtopk, k, threshold, base_freq):
    """
    重构后的MECC方法：仅区分Normal/VIV两类（完全剔除过渡态）
    :param fxtopk: 频谱峰值频率序列（列表/数组）
    :param pxxtopk: 频谱峰值功率序列（列表/数组）
    :param k: MECC的频率区间参数
    :param threshold: 能量集中系数阈值
    :param base_freq: 基础倍频（来自Cal_Mount）
    :return: 分类标签（0:Normal 2:VIV）
    """
    # 强制转换为numpy数组
    try:
        fxtopk = np.array(fxtopk, dtype=np.float64)
        pxxtopk = np.array(pxxtopk, dtype=np.float64)
    except Exception as e:
        raise TypeError(f"fxtopk/pxxtopk类型错误：{e}")
    
    # 边界条件处理
    if k == 0 or len(fxtopk) <= 1 or len(pxxtopk) <= 1:
        return 0
    
    # 定义k倍基频的过滤区间
    f0 = fxtopk[0]
    freq_lower = f0 - k * base_freq
    freq_upper = f0 + k * base_freq
    
    # Mask过滤区间外峰值
    mask_outside = (fxtopk < freq_lower) | (fxtopk > freq_upper)
    pxxtopk_outside = pxxtopk[mask_outside]
    
    # 分类判定
    if len(pxxtopk_outside) == 0:
        return 2
    max_pxx_outside = np.max(pxxtopk_outside)
    energy_ratio = max_pxx_outside / pxxtopk[0]
    return 2 if energy_ratio < threshold else 0

# ===================== F1计算+混淆矩阵生成 =====================
def calculate_metrics_for_params(k, threshold, dataset, mount):
    """
    计算指定参数下的F1分数+混淆矩阵（仅Normal/VIV两类）
    :return: 加权F1分数、混淆矩阵、真实标签列表、预测标签列表
    """
    true_labels = []
    pred_labels = []
    base_freq = mount.multi_nums

    with tqdm(total=len(dataset), desc=f"计算 k={k}, 阈值={threshold}", leave=False) as pbar:
        for data, label in dataset:
            # 记录真实标签（仅0/2）
            true_labels.append(label)
            
            # 频谱分析
            fx, pxxden = signal.welch(
                data[0, :], 
                fs=50, 
                nfft=65536,
                nperseg=2048, 
                noverlap=1
            )
            
            # 提取频谱峰值
            fxtopk, pxxtopk, _, _ = mount.peaks(fx, pxxden, return_intervals=True)
            
            # MECC分类预测
            pred_label = mecc_classify(fxtopk, pxxtopk, k, threshold, base_freq)
            pred_labels.append(pred_label)
            
            pbar.update(1)
    
    # 计算加权F1（仅针对0/2类）
    f1 = f1_score(true_labels, pred_labels, average='weighted', labels=[0,2])
    # 生成混淆矩阵（按LABELS顺序：Normal(0)、VIV(2)）
    cm = confusion_matrix(true_labels, pred_labels, labels=[0,2])
    return f1, cm, true_labels, pred_labels

# ===================== 混淆矩阵可视化 =====================
def plot_confusion_matrix(cm, k, threshold, f1_score_val):
    """
    绘制混淆矩阵热力图（仅Normal/VIV两类）
    :param cm: 混淆矩阵（2x2）
    :param k: 当前MECC参数k
    :param threshold: 当前MECC阈值
    :param f1_score_val: 该参数组合的F1分数
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # 绘制混淆矩阵
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',  # 整数标注（样本数量）
        cmap='Blues',
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        cbar=False,
        annot_kws={'size': plt.rcParams['font.size']}
    )
    # 标签与标题
    ax.set_xlabel('Predicted Label', labelpad=10)
    ax.set_ylabel('True Label', labelpad=10)
    plt.tight_layout()
    return fig

# ===================== 帕累托最优验证+混淆矩阵绘制 =====================
def run_pareto_verification_and_plot():
    """主流程：帕累托最优验证 + 热力图 + 最优参数混淆矩阵"""
    # 1. 初始化数据集与工具类
    print("初始化数据集...")
    dataset = ECC_Dataset(data_dir=DATA_DIR, img_dir=IMG_DIR)
    mount = Cal_Mount()
    ploter = PlotLib()

    # 2. 遍历参数网格，计算F1和混淆矩阵
    print(f"\n开始遍历参数网格（k={len(K_VALUES)}个值，阈值={len(THRESHOLD_VALUES)}个值）...")
    f1_matrix = np.zeros((len(K_VALUES), len(THRESHOLD_VALUES)))
    cm_dict = {}  # 存储各参数组合的混淆矩阵：(k, threshold) → cm
    total_combinations = len(K_VALUES) * len(THRESHOLD_VALUES)
    
    with tqdm(total=total_combinations, desc="总进度") as pbar_total:
        for i, k in enumerate(K_VALUES):
            for j, threshold in enumerate(THRESHOLD_VALUES):
                try:
                    f1, cm, _, _ = calculate_metrics_for_params(k, threshold, dataset, mount)
                    f1_matrix[i, j] = f1
                    cm_dict[(k, threshold)] = cm
                except Exception as e:
                    print(f"\n参数组合(k={k}, threshold={threshold})计算失败：{str(e)}")
                    f1_matrix[i, j] = np.nan
                    cm_dict[(k, threshold)] = None
                pbar_total.update(1)

    # 3. 绘制参数-F1热力图
    fig, ax = plt.subplots(figsize=(12, 9))
    f1_matrix_reversed = np.flipud(f1_matrix)
    k_labels_reversed = K_VALUES[::-1]
    
    sns.heatmap(
        f1_matrix_reversed,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=THRESHOLD_VALUES,
        yticklabels=k_labels_reversed,
        ax=ax,
        cbar_kws={'label': 'Weighted F1 Score', 'shrink': 0.8},
        mask=np.isnan(f1_matrix_reversed),
        annot_kws={'size': plt.rcParams['font.size']}
    )
    ax.invert_yaxis()
    ax.set_xlabel(r'$MECC_0$', labelpad=10)
    ax.set_ylabel(r'$k$', labelpad=10)
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    plt.tight_layout()
    ploter.figs.append(fig)

    # 4. 筛选最优参数并绘制混淆矩阵
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
            # 绘制最优参数的混淆矩阵
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
        # 绘制最高F1参数的混淆矩阵
        if best_cm is not None:
            cm_fig = plot_confusion_matrix(best_cm, best_k, best_threshold, best_f1)
            ploter.figs.append(cm_fig)

    # 5. 展示所有图表
    ploter.show()
    return f1_matrix, K_VALUES, THRESHOLD_VALUES, cm_dict

# ===================== 执行主流程 =====================
if __name__ == "__main__":
    f1_matrix, k_list, threshold_list, cm_dict = run_pareto_verification_and_plot()