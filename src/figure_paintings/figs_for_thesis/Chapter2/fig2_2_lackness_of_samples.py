from ....visualize_tools.utils import ChartApp, PlotLib
import matplotlib.pyplot as plt
import numpy as np
import os

from ....data_processer.io_unpacker import UNPACK
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 从配置文件导入绘图基础配置
from ..config import (
    FONT_SIZE, ENG_FONT, CN_FONT,
    BELOW_THRESHOLD_COLOR, ABOVE_THRESHOLD_COLOR,
    THRESHOLD_COLOR, N_BINS, TARGET_VIBRATION_SENSORS,
    FS
)

RESULT_SAVE_PATH =  r'F:\Research\Vibration Characteristics In Cable Vibration\results\lackness_statistics.txt'
ALL_VIBRATION_ROOT = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC"

# 常量硬编码
MISSING_RATE_THRESHOLD = 0.05  # 缺失率阈值

def process_single_file(file_path):
    """获取单个源文件的数据长度"""
    try:
        import numpy as np
        from ...data_processer.io_unpacker import UNPACK
        unpacker = UNPACK(init_path=False)
        vibration_data = unpacker.VIC_DATA_Unpack(file_path)
        return len(vibration_data)
    except Exception:
        return 0

# 绘制缺失比例直方图（单一线性坐标）
def plot_missing_ratio_histogram(missing_rates, font_size, ENG_FONT, CN_FONT):
    """
    绘制缺失比例直方图：
    在一副图中展示所有样本的缺失率分布
    """
    if len(missing_rates) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 7))
    valid_rates = missing_rates
    
    # 统计信息，确保绘图范围包含所有数据
    min_rate = np.min(valid_rates)
    max_rate = np.max(valid_rates)
    
    # 设定直方图的范围，默认 [0, 1]，但如果数据超限则扩大
    plot_min = min(0.0, min_rate)
    plot_max = max(1.0, max_rate)
    
    # 绘制直方图
    bins = np.linspace(plot_min, plot_max, N_BINS + 1)
    
    # 统计各区间
    ax.hist(valid_rates, bins=bins, color=BELOW_THRESHOLD_COLOR, 
             edgecolor='white', linewidth=0.5, alpha=0.8, label='样本缺失率分布')
    
    # 设置 Y 轴为对数坐标
    ax.set_yscale('log')
    
    # 绘制阈值线
    ax.axvline(MISSING_RATE_THRESHOLD, color=THRESHOLD_COLOR, linestyle='--', linewidth=2, 
                label=f'阈值 ({MISSING_RATE_THRESHOLD*100:.1f}%)')
    
    ax.set_xlabel('缺失率', fontproperties=CN_FONT)
    ax.set_ylabel('源文件数量', fontproperties=CN_FONT)
    
    # 设置 X 轴为百分比格式
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    
    # 设置字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(ENG_FONT)
        
    ax.legend(prop=CN_FONT, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_missing_ratio_dual_linear_subplots(missing_rates, font_size, ENG_FONT, CN_FONT):
    """
    绘制缺失比例双子图：
    左子图：缺失率 0~5% (线性坐标)
    右子图：缺失率 5%~95% (线性坐标)
    宽度比例 1:2，分箱密度比例 1:2
    """
    if len(missing_rates) == 0:
        return None

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    boundary = MISSING_RATE_THRESHOLD # 0.05
    left_rates = missing_rates[missing_rates <= boundary]
    right_rates = missing_rates[missing_rates > boundary]

    # 设定分箱数量
    # 左侧范围 0.05，右侧范围 0.95 (假设到1.0)
    # 密度 1:2 意味着右侧的分箱更密集
    n_bins_left = 20
    n_bins_right = 80 # 密度更高

    # 左子图 (包含可能小于0的部分)
    min_l = min(0.0, np.min(left_rates)) if len(left_rates) > 0 else 0.0
    ax1.hist(left_rates, bins=np.linspace(min_l, boundary, n_bins_left + 1), 
             color=BELOW_THRESHOLD_COLOR, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax1.set_title('低缺失率样本 (0-5%)', fontproperties=CN_FONT)
    ax1.set_ylabel('源文件数量', fontproperties=CN_FONT)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # 右子图 (5 - 100%)
    # 自动确定右侧上限
    max_r = max(1.0, np.max(right_rates)) if len(right_rates) > 0 else 1.0
    ax2.hist(right_rates, bins=np.linspace(boundary, max_r, n_bins_right + 1), 
             color=ABOVE_THRESHOLD_COLOR, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax2.set_title('高缺失率样本 (5%以上)', fontproperties=CN_FONT)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # 设置字体和网格
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(ENG_FONT)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlabel('缺失率', fontproperties=CN_FONT)

    plt.tight_layout()
    return fig

def Lackness_Of_Samples_Analysis():
    # 核心参数配置
    expected_length = int(FS * 60 * 60)  # 50Hz * 60s * 60m
    
    all_vibration_root = ALL_VIBRATION_ROOT
    ploter = PlotLib() 
    figs = []

    def get_all_vibration_files(root_dir, target_sensor_ids, suffix=".VIC"):
        vibration_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.upper().endswith(suffix.upper()):
                    if any(sensor_id in file for sensor_id in target_sensor_ids):
                        file_path = os.path.join(root, file)
                        vibration_files.append(file_path)
        return vibration_files

    all_vib_files = get_all_vibration_files(root_dir=all_vibration_root, target_sensor_ids=TARGET_VIBRATION_SENSORS)
    print(f"共获取所有振动文件数量：{len(all_vib_files)}")

    # 使用多进程并行获取数据
    actual_lengths = []
    print(f"开始并行处理文件并获取数据长度...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, fp): fp for fp in all_vib_files}
        for future in tqdm(as_completed(futures), total=len(all_vib_files), desc="数据长度获取"):
            try:
                file_len = future.result()
                # 即使长度为0也记录，作为100%缺失
                actual_lengths.append(file_len)
            except Exception as e:
                print(f"处理任务时出错: {e}")
                # 出错也记录为0长度
                actual_lengths.append(0)

    if not actual_lengths:
        print("警告：无有效样本数据")
        return

    # 计算缺失率：1 - (实际长度 / 预期长度)
    actual_lengths = np.array(actual_lengths)
    missing_rates = 1.0 - (actual_lengths / expected_length)
    
    # 打印统计信息
    total_samples = len(missing_rates)
    high_missing_samples = np.sum(missing_rates > MISSING_RATE_THRESHOLD)
    avg_missing_rate = np.mean(missing_rates)
    
    print("\n" + "="*50)
    print("           样本缺失率统计报告")
    print("="*50)
    print(f"处理源文件总数: {total_samples}")
    print(f"预期单文件长度: {expected_length} (50Hz * 60s * 60m)")
    print(f"平均缺失率: {avg_missing_rate*100:.2f}%")
    print(f"缺失率超过阈值 ({MISSING_RATE_THRESHOLD*100:.1f}%) 的样本数: {high_missing_samples}")
    print(f"超过阈值样本占比: {high_missing_samples/total_samples*100:.2f}%")
    print("="*50 + "\n")

    # 绘制缺失比例直方图
    print("开始绘制缺失率分布图...")
    fig = plot_missing_ratio_histogram(
        missing_rates=missing_rates,
        font_size=FONT_SIZE,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    
    # 验证绘图完整性
    hist_counts, _ = np.histogram(missing_rates, bins=np.linspace(min(0, np.min(missing_rates)), max(1, np.max(missing_rates)), N_BINS + 1))
    print(f"直方图统计样本总数: {np.sum(hist_counts)} / 原始文件总数: {total_samples}")
    
    boundary = MISSING_RATE_THRESHOLD
    left_rates = missing_rates[missing_rates <= boundary]
    right_rates = missing_rates[missing_rates > boundary]
    left_counts, _ = np.histogram(left_rates, bins=np.linspace(min(0.0, np.min(left_rates)) if len(left_rates)>0 else 0, boundary, 21))
    right_counts, _ = np.histogram(right_rates, bins=np.linspace(boundary, max(1.0, np.max(right_rates)) if len(right_rates)>0 else 1.0, 81))
    print(f"双子图统计样本总数: {np.sum(left_counts) + np.sum(right_counts)} / 原始文件总数: {total_samples}")
    
    if fig:
        figs.append(fig)
        plt.close(fig)

    # 绘制缺失比例双子图
    print("开始绘制缺失率双子图分布...")
    fig_dual = plot_missing_ratio_dual_linear_subplots(
        missing_rates=missing_rates,
        font_size=FONT_SIZE,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_dual:
        figs.append(fig_dual)
        plt.close(fig_dual)

    ploter.figs.extend(figs)
    ploter.show()

if __name__ == "__main__":
    Lackness_Of_Samples_Analysis()
