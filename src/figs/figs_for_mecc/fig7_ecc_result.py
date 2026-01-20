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
# --- 新增点 0: 导入datetime用于记录时间（可选，增强结果文件可读性） ---
from datetime import datetime

# ===================== 全局配置与常量定义 =====================
# 数据集路径（抽离为常量，便于修改）
DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\data\\'
IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\NN\datasets\dev\img\\'
# --- 修改点 2: 新增帕累托结果保存路径（核心需求1） ---
SAVE_DIR = r"F:\Research\Vibration Characteristics In Cable Vibration\outputs\ecc_pareto_result"
# --- 新增点 5: 定义结果文件路径常量 ---
RESULT_NPZ_PATH = os.path.join(SAVE_DIR, 'ecc_search_results.npz')
RESULT_TXT_PATH = os.path.join(SAVE_DIR, 'result.txt')
F1_PLOT_PATH = os.path.join(SAVE_DIR, 'ecc_threshold_f1_plot.png')

# 标签映射（剔除过渡态，仅保留有效类别）
LABEL_MAP = {0: 'Normal Vibration', 2: 'VIV'}
LABELS = list(LABEL_MAP.values())  # ['Normal Vibration', 'VIV']
LABEL_IDS = [0, 2]  # 对应标签的数字ID

# ===================== 核心：最优参数定义（重点） =====================
# --- 修改点 3: 删除K相关参数（ECC仅需threshold） ---
BEST_THRESHOLD = 0.1  # 最优ECC阈值

# --- 新增点 1: ECC参数搜索网格（仅保留threshold） ---
THRESHOLD_VALUES = [0, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# 调试用参数（需要时取消注释）
# DATA_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\data\\'
# IMG_DIR = r'F:\Research\Vibration Characteristics In Cable Vibration\data\test_NN_data\img\\'
# THRESHOLD_VALUES = [0.1, 0.2]

# --- 新增点 2: 新增秒级切分区间 (来自参考代码1) ---
INTERVAL = (45, 46)
SECOND_INTERVAL = (10, 20)

# --- 修改点 2: 创建自定义灰度 colormap (保留原有逻辑，不修改) ---
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

# 绘图配置（完全保留原有配置，不修改）
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = 26
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# --- 新增点 4: 定义中、英文的 FontProperties 对象 (保留原有逻辑) ---
ENG_FONT = FontProperties(family='Times New Roman', size=26)
CN_FONT = FontProperties(family='SimHei', size=26)

# ===================== 数据集类定义（完全保留，不修改） =====================
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

# ===================== ECC分类核心函数（修正问题，保留核心逻辑） =====================
def ecc_classify(fxtopk, pxxtopk, threshold):
    """
    ECC分类函数：区分Normal/VIV两类
    核心逻辑：PSD次大值 / PSD最大值 < 阈值 → 判定为VIV(2)，否则为Normal(0)
    参数：
        fxtopk: 峰值频率数组
        pxxtopk: 峰值功率谱密度数组
        threshold: 判定阈值
    返回：
        label: 0（Normal）/ 2（VIV）
    """
    try:
        fxtopk = np.array(fxtopk, dtype=np.float64)
        pxxtopk = np.array(pxxtopk, dtype=np.float64)
        
        # 异常处理：峰值数量不足2时，默认判定为Normal
        if len(pxxtopk) < 2:
            return 0
        
        # 排序后取次大值和最大值（升序排序，最后两位是最大、次大）
        pxxtopk_sorted = sorted(pxxtopk)
        second_max = pxxtopk_sorted[-2]
        max_val = pxxtopk_sorted[-1]
        
        # 核心判定逻辑：次大/最大 < 阈值 → VIV(2)，否则Normal(0)
        return 2 if (second_max / max_val) < threshold else 0
    
    except Exception as e:
        print(f"ECC分类出错：{e}")
        return 0  # 出错时默认返回Normal

# ===================== 核心：指标计算函数（完全保留，不修改） =====================
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

# ===================== 核心：最优参数评估函数（删除K相关逻辑） =====================
def evaluate_best_params():
    """
    基于最优阈值计算并输出F1/Precision/Recall（ECC无k参数）
    """
    print("="*60)
    print(f"开始评估最优参数：ECC阈值={BEST_THRESHOLD}")
    print("="*60)
    dataset = ECC_Dataset(data_dir=DATA_DIR, img_dir=IMG_DIR)
    mount = Cal_Mount()

    true_labels = []
    pred_labels = []
    with tqdm(total=len(dataset), desc="最优参数推理") as pbar:
        for data, label in dataset:
            true_labels.append(label)
            fx, pxxden = signal.welch(data[0, :], fs=50, nfft=65536, nperseg=2048, noverlap=1)
            fxtopk, pxxtopk, _, _ = mount.peaks(fx, pxxden, return_intervals=True)
            pred_label = ecc_classify(fxtopk, pxxtopk, BEST_THRESHOLD)
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

    plt.tight_layout()
    plt.show()

    return metrics

# ===================== 封装的画图逻辑函数（新增核心） =====================
# --- 新增点 6: 封装F1-阈值曲线绘制函数 ---
def plot_ecc_f1_curve(threshold_values, f1_scores, ploter):
    """
    封装ECC阈值-F1曲线绘制逻辑
    参数：
        threshold_values: 阈值列表
        f1_scores: 对应F1分数数组
        ploter: PlotLib实例
    返回：
        fig: 绘制好的图实例
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # 绘制折线（用原有灰度系的深灰色，保持显示配置）
    ax.plot(threshold_values, f1_scores, color='#808080', linewidth=3, marker='o', markersize=8)
    # 填充曲线下方区域（灰度渐变，匹配原有CMAP）
    ax.fill_between(threshold_values, f1_scores, alpha=0.5, color='#808080')
    
    # 保留原有显示配置（字体、字号）
    ax.set_xlabel(r"$C'_{ECC}$", labelpad=10, fontproperties=ENG_FONT)
    ax.set_ylabel('F1 Score', labelpad=10, fontproperties=ENG_FONT)
    ax.set_xticks(threshold_values[::2])  # 间隔显示阈值，避免拥挤
    ax.set_ylim(0, 1.0)  # F1分数范围0-1
    ax.grid(True, alpha=0.3)  # 轻微网格，不影响视觉
    plt.tight_layout()
    ploter.figs.append(fig)
    
    # 保存图片
    fig.savefig(F1_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"F1-阈值曲线图已保存至：{F1_PLOT_PATH}")
    return fig

# --- 新增点 7: 封装混淆矩阵绘制函数（增强版） ---
def plot_ecc_confusion_matrix(threshold, f1_value, cm, ploter, save_dir):
    """
    封装混淆矩阵绘制与保存逻辑
    参数：
        threshold: 阈值
        f1_value: 对应F1分数
        cm: 混淆矩阵
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
    ax.set_title(f'Confusion Matrix (ECC阈值={threshold})\nWeighted F1: {f1_value:.3f}', pad=20, fontproperties=ENG_FONT)
    plt.tight_layout()
    ploter.figs.append(fig)
    
    # 保存混淆矩阵图
    cm_save_path = os.path.join(save_dir, f'confusion_matrix_th{threshold}.png')
    fig.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存至：{cm_save_path}")

# ===================== ECC参数搜索相关函数（重构） =====================
def calculate_metrics_for_params(threshold, dataset, mount):
    """
    计算指定ECC阈值下的完整指标（P/R/F1+混淆矩阵）
    返回：完整metrics字典、加权F1
    """
    true_labels = []
    pred_labels = []

    with tqdm(total=len(dataset), desc=f"计算 阈值={threshold}", leave=False) as pbar:
        for data, label in dataset:
            true_labels.append(label)
            fx, pxxden = signal.welch(data[0, :], fs=50, nfft=65536, nperseg=2048, noverlap=1)
            fxtopk, pxxtopk, _, _ = mount.peaks(fx, pxxden, return_intervals=True)
            pred_label = ecc_classify(fxtopk, pxxtopk, threshold)
            pred_labels.append(pred_label)
            pbar.update(1)
    
    # --- 新增点 1: 计算完整的P/R/F1指标 ---
    metrics = calculate_metrics(true_labels, pred_labels)
    return metrics, metrics['weighted']['F1']

# --- 新增点 2: 定义结果写入函数（保留，不修改） ---
def write_result_file(save_dir, threshold_metrics, best_threshold, best_metrics, dataset_size):
    """
    将ECC参数搜索结果写入result.txt文件
    参数：
        save_dir: 保存目录
        threshold_metrics: 字典，{threshold: metrics}
        best_threshold: 最优阈值
        best_metrics: 最优阈值对应的metrics
        dataset_size: 数据集样本数量
    """
    result_path = os.path.join(save_dir, 'result.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        # 1. 写入基础信息
        f.write("="*80 + "\n")
        f.write(f"ECC参数搜索结果报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集路径: {DATA_DIR}\n")
        f.write(f"有效样本数量: {dataset_size}\n")
        f.write(f"搜索阈值列表: {THRESHOLD_VALUES}\n")
        f.write(f"最优阈值: {best_threshold}\n")
        f.write("="*80 + "\n\n")

        # 2. 写入最优参数详细结果
        f.write("【最优阈值详细指标】\n")
        f.write(f"阈值: {best_threshold}\n")
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

        # 3. 写入所有阈值的详细结果
        f.write("【所有阈值详细搜索结果】\n")
        f.write(f"{'阈值':<8} {'加权P':<12} {'加权R':<12} {'加权F1':<12}\n")
        f.write("-"*50 + "\n")
        for threshold in sorted(threshold_metrics.keys()):
            metrics = threshold_metrics[threshold]
            p = metrics['weighted']['Precision']
            r = metrics['weighted']['Recall']
            f1 = metrics['weighted']['F1']
            f.write(f"{threshold:<8.2f} {p:<12.4f} {r:<12.4f} {f1:<12.4f}\n")
        
        # 4. 写入类别级详细结果（可选，增强可读性）
        f.write("\n【类别级详细指标（按阈值）】\n")
        for threshold in sorted(threshold_metrics.keys()):
            metrics = threshold_metrics[threshold]
            f.write(f"\n阈值 {threshold:.2f}:\n")
            for cls_name, cls_metrics in metrics['per_class'].items():
                f.write(f"  {cls_name}:\n")
                f.write(f"    Precision: {cls_metrics['Precision']:.4f}\n")
                f.write(f"    Recall:    {cls_metrics['Recall']:.4f}\n")
                f.write(f"    F1 Score:  {cls_metrics['F1']:.4f}\n")
    
    print(f"\n结果文件已保存至：{result_path}")

# --- 新增点 8: 定义读取结果文件的函数 ---
def load_ecc_results():
    """
    读取已保存的ECC参数搜索结果
    返回：
        threshold_values, f1_scores, best_threshold, best_cm
    """
    if not os.path.exists(RESULT_NPZ_PATH):
        return None, None, None, None
    
    # 读取npz文件
    data = np.load(RESULT_NPZ_PATH, allow_pickle=True)
    threshold_values = data['THRESHOLD_VALUES']
    f1_scores = data['f1_scores']
    best_threshold = data['best_threshold']
    best_cm = data['best_cm']
    
    print(f"已找到历史结果文件，读取阈值数量：{len(threshold_values)}")
    return threshold_values, f1_scores, best_threshold, best_cm

# ===================== 主流程函数（重构，核心修改） =====================
def run_ecc_param_search_and_plot():
    """
    主流程：ECC单参数（threshold）搜索 + F1曲线绘制 + 结果保存 + 结果文件写入
    核心逻辑：
        1. 检查是否有历史结果文件
        2. 有则直接读取并绘图；无则执行参数搜索后绘图
    """
    print("="*60)
    print("开始执行ECC参数搜索/绘图流程...")
    print("="*60)
    
    # --- 修改点 10: 创建保存目录（保留） ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 修改点 11: 初始化PlotLib ---
    ploter = PlotLib()
    
    # 第一步：检查是否有历史结果文件
    threshold_values, f1_scores, best_threshold, best_cm = load_ecc_results()
    
    if threshold_values is not None and f1_scores is not None:
        # 有历史结果：直接调用封装的画图函数
        print("✅ 检测到历史结果文件，跳过参数搜索，直接绘图...")
        
        # 绘制F1-阈值曲线
        plot_ecc_f1_curve(threshold_values, f1_scores, ploter)
        
        # 读取result.txt获取最优阈值的详细信息（可选，增强展示）
        if os.path.exists(RESULT_TXT_PATH):
            with open(RESULT_TXT_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                print("\n📜 历史最优结果摘要：")
                # 简单提取最优阈值和F1
                for line in content.split('\n'):
                    if "最优阈值:" in line:
                        print(line.strip())
                    if "加权F1 Score:" in line and "最优阈值详细指标" in content.split('\n')[content.split('\n').index(line)-2]:
                        print(line.strip())
        
        # 显示图形
        ploter.show()
        return f1_scores, threshold_values, None
    
    # 无历史结果：执行完整参数搜索流程
    print("❌ 未检测到历史结果文件，执行参数搜索...")
    dataset = ECC_Dataset(data_dir=DATA_DIR, img_dir=IMG_DIR)
    mount = Cal_Mount()
    dataset_size = len(dataset)  # 记录数据集大小

    # 初始化存储所有阈值指标的字典
    threshold_metrics = {}  # {threshold: metrics}
    f1_scores = np.zeros(len(THRESHOLD_VALUES))
    total_thresholds = len(THRESHOLD_VALUES)
    
    with tqdm(total=total_thresholds, desc="总进度") as pbar_total:
        for j, threshold in enumerate(THRESHOLD_VALUES):
            try:
                # 计算完整指标
                metrics, f1 = calculate_metrics_for_params(threshold, dataset, mount)
                f1_scores[j] = f1
                threshold_metrics[threshold] = metrics
            except Exception as e:
                print(f"\n阈值{threshold}计算失败：{str(e)}")
                f1_scores[j] = np.nan
                threshold_metrics[threshold] = None
            pbar_total.update(1)

    # --- 修改点 12: 调用封装的F1曲线绘制函数 ---
    plot_ecc_f1_curve(THRESHOLD_VALUES, f1_scores, ploter)

    # 保存ECC搜索结果
    np.savez(
        RESULT_NPZ_PATH,
        f1_scores=f1_scores,
        THRESHOLD_VALUES=THRESHOLD_VALUES,
        best_threshold=BEST_THRESHOLD,
        best_cm=threshold_metrics.get(BEST_THRESHOLD, {}).get('confusion_matrix', np.array([]))
    )
    print(f"\nECC参数搜索结果已保存至：{RESULT_NPZ_PATH}")

    # 筛选最优参数并绘制混淆矩阵
    f1_threshold = 0.9
    high_perf_mask = f1_scores >= f1_threshold
    high_perf_indices = np.argwhere(high_perf_mask)
    high_perf_thresholds = [THRESHOLD_VALUES[j] for j in high_perf_indices]
    
    print(f"\n===== 高F1值阈值筛选结果（F1≥{f1_threshold}） =====")
    if len(high_perf_thresholds) > 0:
        print(f"共找到{len(high_perf_thresholds)}个最优阈值：")
        best_threshold = high_perf_thresholds[0]
        best_metrics = threshold_metrics[best_threshold]
        for idx, threshold in enumerate(high_perf_thresholds):
            f1_value = f1_scores[THRESHOLD_VALUES.index(threshold)]
            cm = threshold_metrics[threshold]['confusion_matrix']
            print(f"{idx+1}. 阈值={threshold}, F1={f1_value:.3f}")
            if cm is not None:
                # --- 修改点 13: 调用封装的混淆矩阵绘制函数 ---
                plot_ecc_confusion_matrix(threshold, f1_value, cm, ploter, SAVE_DIR)
    else:
        max_f1_idx = np.nanargmax(f1_scores)
        best_threshold = THRESHOLD_VALUES[max_f1_idx]
        best_f1 = f1_scores[max_f1_idx]
        best_metrics = threshold_metrics[best_threshold]
        print(f"未找到F1≥{f1_threshold}的阈值，当前最高F1值为：{best_f1:.3f}（阈值={best_threshold}）")
        if best_metrics is not None:
            cm = best_metrics['confusion_matrix']
            # --- 修改点 14: 调用封装的混淆矩阵绘制函数 ---
            plot_ecc_confusion_matrix(best_threshold, best_f1, cm, ploter, SAVE_DIR)

    # 写入result.txt文件
    write_result_file(SAVE_DIR, threshold_metrics, best_threshold, best_metrics, dataset_size)

    # 显示图形
    ploter.show()
    return f1_scores, THRESHOLD_VALUES, threshold_metrics

# ===================== 执行主流程（修改为ECC逻辑） =====================
if __name__ == "__main__":
    # 优先执行最优参数评估（ECC无k）
    best_metrics = evaluate_best_params()
    
    # 运行ECC单参数搜索/绘图（取消注释执行）
    # f1_scores, threshold_list, threshold_metrics = run_ecc_param_search_and_plot()