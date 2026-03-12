# Autor@ 猫毛
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 项目路径设置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目内部模块
from src.data_processer.io_unpacker import UNPACK
from src.visualize_tools.utils import PlotLib
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow
from .config import FONT_SIZE, ENG_FONT, CN_FONT

# --------------- 全局绘图配置 ---------------
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.unicode_minus'] = False

# ==================== 本地绘图配置常量 ====================
# 直方图颜色配置（对齐 fig2_3_rms_statistics.py）
HISTOGRAM_COLOR = '#7895C1'        # 主频直方图颜色
HISTOGRAM_EDGE_COLOR = '#7895C1'   # 直方图边界颜色

# 坐标轴配置
GRID_COLOR = 'gray'
GRID_ALPHA = 0.4
GRID_LINEWIDTH = 0.5
GRID_LINESTYLE = '-'
AXIS_LINEWIDTH = 1.0

# 绘图尺寸
HISTOGRAM_FIG_SIZE = (10, 6)
DOUBLE_STACKED_FIG_SIZE = (12, 6)

# 双堆叠子图配置
DOUBLE_STACKED_WIDTH_RATIOS = [1, 2]  # 左右子图的宽度比例

# ==================== 多进程配置常量 ====================
NUM_WORKERS = 8  # 多进程的核心数目

# ==================== 配置类 ====================
class Config:
    FS = 50.0                      # 采样频率
    NFFT = 1024                     # FFT分辨率
    
    # 时间窗口配置
    SEGMENT_DURATION = 60          # 每个时间窗口的长度（秒）
    
    # 主频提取范围
    FREQ_MIN = 0                 # 最小频率 (Hz)
    FREQ_MAX = 25                  # 最大频率 (Hz)
    
    # 数据抽样配置
    SAMPLE_SIZE = None              # 抽样样本数目（均匀分布抽样），设为 None 时不抽样使用全部数据
    RANDOM_SEED = 42               # 随机种子
    
    # 直方图分箱数
    N_BINS = 100                    # 直方图的分箱数


# ==================== 单文件多进程处理函数 ====================
def process_single_file_extract_dominant_frequencies(file_path):
    """
    单文件处理工作函数，用于多进程
    在此函数中计算每个时间窗口的主频，及时释放原始数据
    
    参数：
        file_path: 振动数据文件路径
    
    返回：
        list: 该文件中所有时间窗口的主频列表
    """
    dominant_frequencies = []
    
    try:
        import numpy as np
        from scipy import signal
        from src.data_processer.io_unpacker import UNPACK
        
        unpacker = UNPACK(init_path=False)
        vibration_data = unpacker.VIC_DATA_Unpack(file_path)
        vibration_data = np.array(vibration_data)
        
        if len(vibration_data) == 0:
            return []
        
        segment_size = int(Config.SEGMENT_DURATION * Config.FS)
        
        # 对数据进行分段处理，计算每段的主频
        for i in range(0, len(vibration_data), segment_size):
            segment = vibration_data[i:i + segment_size]
            
            if len(segment) < 100:
                continue
            
            # 动态调整 NFFT，防止窗口小于 NFFT 导致的警告
            nfft = Config.NFFT
            if len(segment) < nfft:
                nfft = max(len(segment) // 2, 16)
            
            # 计算功率谱
            freqs, psd = signal.welch(
                segment,
                fs=Config.FS,
                nperseg=nfft,
                noverlap=nfft // 2,
                nfft=nfft
            )
            
            # 提取主频
            mask = (freqs >= Config.FREQ_MIN) & (freqs <= Config.FREQ_MAX)
            freqs_filtered = freqs[mask]
            psd_filtered = psd[mask]
            
            if len(psd_filtered) > 0:
                dominant_idx = np.argmax(psd_filtered)
                dominant_freq = freqs_filtered[dominant_idx]
                
                if not np.isnan(dominant_freq):
                    dominant_frequencies.append(dominant_freq)
        
        del vibration_data
        
    except Exception as e:
        pass
    
    return dominant_frequencies


# ==================== 数据获取函数 ====================
def get_all_dominant_frequencies_multiprocess():
    """
    使用多进程从工作流获取所有文件中的主频数据
    不加载原始数据到内存，只保存计算的主频
    
    返回：
        numpy.ndarray: 所有主频的数组
    """
    print("[获取数据] 运行振动工作流获取元数据...")
    metadata = run_vib_workflow(use_cache=True, force_recompute=False)
    
    print(f"✓ 从工作流获取 {len(metadata)} 条元数据记录")
    
    if not metadata:
        raise ValueError("无可用的元数据记录")
    
    # 提取所有文件路径
    all_files = [record['file_path'] for record in metadata]
    
    all_dominant_frequencies = []
    
    print(f"\n[处理数据] 使用多进程处理 {len(all_files)} 个文件...")
    
    # 使用多进程并行处理每个文件
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_file_extract_dominant_frequencies, fp): fp for fp in all_files}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(all_files), desc="多进程处理进度"):
            try:
                dominant_freqs = future.result()
                if dominant_freqs:
                    all_dominant_frequencies.extend(dominant_freqs)
            except Exception as e:
                pass
    
    all_dominant_frequencies = np.array(all_dominant_frequencies)
    print(f"✓ 共计算出 {len(all_dominant_frequencies)} 个主频样本")
    
    if len(all_dominant_frequencies) == 0:
        raise ValueError("未能计算任何主频数据")
    
    # 执行均匀分布抽样（如果 SAMPLE_SIZE 为 None，则不抽样）
    if Config.SAMPLE_SIZE is None:
        print(f"\n[抽样] 已禁用抽样，使用全部 {len(all_dominant_frequencies)} 个主频样本")
        sampled_frequencies = all_dominant_frequencies
    else:
        np.random.seed(Config.RANDOM_SEED)
        total_samples = len(all_dominant_frequencies)
        sample_indices = np.linspace(0, total_samples - 1, Config.SAMPLE_SIZE, dtype=int)
        sample_indices = np.unique(sample_indices)
        
        print(f"\n[抽样] 从 {total_samples} 个主频样本中均匀抽样 {len(sample_indices)} 个")
        
        sampled_frequencies = all_dominant_frequencies[sample_indices]
        
        print(f"✓ 最终抽样主频样本数: {len(sampled_frequencies)}")
    
    return sampled_frequencies


# ==================== 绘图函数 ====================
def plot_dominant_frequency_distribution(dominant_frequencies):
    """
    绘制主频的统计分布形态（直方图）
    绘图风格参考 fig2_3_rms_statistics.py
    
    参数：
        dominant_frequencies: 主频数组
    
    返回：
        fig: matplotlib figure对象
    """
    freq_min = np.min(dominant_frequencies)
    freq_max = np.max(dominant_frequencies)
    
    bins = np.linspace(freq_min, freq_max, Config.N_BINS + 1)
    
    fig, ax = plt.subplots(figsize=HISTOGRAM_FIG_SIZE)
    
    # 绘制直方图
    ax.hist(
        dominant_frequencies,
        bins=bins,
        color=HISTOGRAM_COLOR,
        edgecolor=HISTOGRAM_EDGE_COLOR,
        linewidth=0.8,
        alpha=1.0
    )
    
    # ==================== 坐标轴配置（参考 fig2_3_rms_statistics.py）====================
    # 隐藏上方和右方坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置坐标轴位置（从原点出发）
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', freq_min))
    ax.spines['left'].set_linewidth(AXIS_LINEWIDTH)
    ax.spines['bottom'].set_linewidth(AXIS_LINEWIDTH)
    
    # ==================== 标签和刻度配置 ====================
    ax.set_xlabel('频率 (Hz)', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks()], fontproperties=ENG_FONT)
    
    ax.set_ylabel('样本数量', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax.get_yticks()], fontproperties=ENG_FONT)
    
    # ==================== 网格线配置 ====================
    ax.set_axisbelow(True)
    ax.grid(axis='y', which='major', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    ax.grid(axis='x', alpha=GRID_ALPHA * 0.6, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    
    plt.tight_layout()
    
    return fig


def plot_dominant_frequency_double_stacked_subplots(dominant_frequencies):
    """
    绘制双堆叠子图（参考 fig2_3_rms_statistics.py）:
    左子图：从 FREQ_MIN 开始，包含样本中的95%分位数区间
    右子图：包含剩下的5%样本（高频异常部分）
    
    参数：
        dominant_frequencies: 主频数组
    
    返回：
        fig: matplotlib figure对象
    """
    freq_min = np.min(dominant_frequencies)
    freq_max = np.max(dominant_frequencies)
    
    if len(dominant_frequencies) == 0:
        return None
    
    # 计算95%分位数（用于划分左右子图）
    freq_p95 = np.percentile(dominant_frequencies, 95)
    
    left_subplot_min = freq_min
    left_subplot_max = freq_p95
    
    right_subplot_min = freq_p95
    right_subplot_max = freq_max
    
    interval_nums = 50
    left_interval_nums = interval_nums
    right_interval_nums = interval_nums * 2
    
    # 按范围筛选数据
    freq_left = dominant_frequencies[(dominant_frequencies >= left_subplot_min) & (dominant_frequencies <= left_subplot_max)]
    freq_right = dominant_frequencies[(dominant_frequencies > right_subplot_min) & (dominant_frequencies <= right_subplot_max)]
    
    fig = plt.figure(figsize=DOUBLE_STACKED_FIG_SIZE)
    gs = gridspec.GridSpec(1, 2, width_ratios=DOUBLE_STACKED_WIDTH_RATIOS)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # ======================== 左子图（95%以下） ========================
    bins_left = np.linspace(left_subplot_min, left_subplot_max, left_interval_nums + 1)
    # 计算左右对称的余量
    margin_left = (left_subplot_max - left_subplot_min) / 80
    x_start = left_subplot_min - margin_left
    x_end = left_subplot_max + margin_left
    
    freq_hist_left, _ = np.histogram(freq_left, bins=bins_left)
    ax1.bar(bins_left[:-1], freq_hist_left, width=np.diff(bins_left), 
            color=HISTOGRAM_COLOR, edgecolor=HISTOGRAM_EDGE_COLOR, linewidth=0.5, alpha=1.0)
    
    ax1.set_ylabel('样本数量', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    # 设置对称的范围
    ax1.set_xlim(x_start, x_end)
    
    auto_xticks_1 = ax1.get_xticks()
    # 过滤出在范围内的自动刻度
    valid_auto = auto_xticks_1[(auto_xticks_1 > left_subplot_min) & (auto_xticks_1 < left_subplot_max)]
    if len(auto_xticks_1) >= 2:
        default_interval = np.mean(np.diff(auto_xticks_1))
        # 防重叠：左侧检查
        if len(valid_auto) > 0 and (valid_auto[0] - left_subplot_min) < 0.8 * default_interval:
            valid_auto = valid_auto[1:]
        # 防重叠：右侧检查
        if len(valid_auto) > 0 and (left_subplot_max - valid_auto[-1]) < 0.8 * default_interval:
            valid_auto = valid_auto[:-1]
    
    new_xticks_1 = np.unique(np.concatenate([[left_subplot_min], valid_auto, [left_subplot_max]]))
    new_xticks_1 = np.sort(new_xticks_1)
    ax1.set_xticks(new_xticks_1)
    ax1.set_xticklabels([f'{x:.2f}' for x in new_xticks_1], fontproperties=ENG_FONT, rotation=45)
    
    ax1.set_yticklabels(ax1.get_yticks(), fontproperties=ENG_FONT)
    ax1.grid(axis='y', which='major', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    ax1.grid(axis='x', alpha=GRID_ALPHA * 0.6, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    ax1.set_axisbelow(True)
    
    # ======================== 右子图（95%以上） ========================
    bins_right = np.linspace(right_subplot_min, right_subplot_max, right_interval_nums + 1)
    # 计算左右对称的余量
    margin_right = (right_subplot_max - right_subplot_min) / 80
    right_subplot_x_start = right_subplot_min - margin_right
    right_subplot_x_end = right_subplot_max + margin_right
    
    freq_hist_right, _ = np.histogram(freq_right, bins=bins_right)
    ax2.bar(bins_right[:-1], freq_hist_right, width=np.diff(bins_right), 
            color=HISTOGRAM_COLOR, edgecolor=HISTOGRAM_EDGE_COLOR, linewidth=0.5, alpha=1.0, label='主频样本')
    
    # 设置对称的范围
    ax2.set_xlim(right_subplot_x_start, right_subplot_x_end)
    ax2.set_ylabel('样本数量', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    auto_xticks_2 = ax2.get_xticks()
    # 过滤出在范围内的自动刻度，并检查与两端的间隔
    valid_auto_2 = auto_xticks_2[(auto_xticks_2 > right_subplot_min) & (auto_xticks_2 < right_subplot_max)]
    if len(valid_auto_2) > 0 and len(auto_xticks_2) >= 2:
        default_interval_2 = np.mean(np.diff(auto_xticks_2))
        # 检查左侧重叠
        if (valid_auto_2[0] - right_subplot_min) < 0.8 * default_interval_2:
            valid_auto_2 = valid_auto_2[1:]
        # 检查右侧重叠
        if len(valid_auto_2) > 0 and (right_subplot_max - valid_auto_2[-1]) < 0.8 * default_interval_2:
            valid_auto_2 = valid_auto_2[:-1]
    
    new_xticks_2 = np.unique(np.concatenate([[right_subplot_min], valid_auto_2, [right_subplot_max]]))
    new_xticks_2 = np.sort(new_xticks_2)
    ax2.set_xticks(new_xticks_2)
    ax2.set_xticklabels([f'{x:.2f}' for x in new_xticks_2], fontproperties=ENG_FONT, rotation=45)
    
    ax2.set_yticklabels(ax2.get_yticks(), fontproperties=ENG_FONT)
    ax2.grid(axis='y', which='major', alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    ax2.grid(axis='x', alpha=GRID_ALPHA * 0.6, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)
    ax2.set_axisbelow(True)
    
    fig.text(0.5, 0.02, '频率 (Hz)', ha='center', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig




# ==================== 主分析函数 ====================
def main():
    """
    主分析函数：分析主频的统计分布
    """
    print("=" * 80)
    print("振动主频统计分布分析")
    print("=" * 80)
    
    # 步骤1: 多进程获取主频数据并抽样
    print("\n[步骤1] 多进程处理数据，计算主频...")
    dominant_frequencies = get_all_dominant_frequencies_multiprocess()
    
    # 步骤2: 绘制分布图
    print("\n[步骤2] 绘制分布图...")
    ploter = PlotLib()
    
    # 图1：单直方图
    print("\n开始绘制单直方图...")
    fig_distribution = plot_dominant_frequency_distribution(dominant_frequencies)
    ploter.figs.append(fig_distribution)
    
    # 图2：双堆叠子图（着重显示95%分位数的上下区间）
    print("\n开始绘制双堆叠子图...")
    fig_double_stacked = plot_dominant_frequency_double_stacked_subplots(dominant_frequencies)
    if fig_double_stacked:
        ploter.figs.append(fig_double_stacked)
    
    # 计算统计量
    freq_p95 = np.percentile(dominant_frequencies, 95)
    
    print("\n" + "=" * 80)
    print(f"✓ 分析完成")
    print(f"  - 分析的主频样本数: {len(dominant_frequencies)}")
    print(f"  - 主频范围: {np.min(dominant_frequencies):.4f} ~ {np.max(dominant_frequencies):.4f} Hz")
    print(f"  - 主频均值: {np.mean(dominant_frequencies):.4f} Hz")
    print(f"  - 主频标准差: {np.std(dominant_frequencies):.4f} Hz")
    print(f"  - 主频中位数: {np.median(dominant_frequencies):.4f} Hz")
    print(f"  - 主频95%分位数: {freq_p95:.4f} Hz")
    print("=" * 80 + "\n")
    
    ploter.show()


if __name__ == '__main__':
    main()
