"""
================================================================================
文件依赖说明 (File Dependencies)
================================================================================
本文件依赖以下数据处理模块：

1. 数据获取模块 (Step 0):
   - src.data_processer.statistics.vibration_io_process.step0_get_vib_data
     └─> get_all_vibration_files() 获取所有振动文件路径
   
2. 数据筛选模块 (Step 1):
   - src.data_processer.statistics.vibration_io_process.step1_lackness_filter
     └─> run_lackness_filter() 执行缺失率筛选，过滤不符合要求的数据文件
     └─> 返回筛选后的文件路径列表和统计信息
   
3. 配置文件:
   - src.config.data_processer.statistics.vibration_io_process.config
     └─> 定义缺失率阈值 (MISSING_RATE_THRESHOLD) 和预期长度 (EXPECTED_LENGTH)
   
4. 数据解析:
   - src.data_processer.io_unpacker.UNPACK
     └─> 解析 .VIC 格式的振动数据文件

设计说明:
  本文件直接依赖 step1_lackness_filter 而非完整的 workflow，这样可以：
  - 确保数据来源稳定（只使用缺失率筛选后的数据）
  - 避免 workflow 后续新增步骤（step2, step3...）导致数据进一步筛选而不对齐
================================================================================
"""

from ....visualize_tools.utils import ChartApp, PlotLib
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec  

from ....data_processer.io_unpacker import UNPACK, DataManager
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 从配置文件导入绘图基础配置
from ..config import (
    FONT_SIZE, ENG_FONT, CN_FONT,
    NORMAL_VIB_COLOR, NORMAL_EDGE_COLOR,
    THRESHOLD_COLOR, N_BINS, TARGET_VIBRATION_SENSORS
)

# 导入数据处理模块
from ....data_processer.preprocess.vibration_io_process.step0_get_vib_data import get_all_vibration_files
from ....data_processer.preprocess.vibration_io_process.step1_lackness_filter import run_lackness_filter

RESULT_SAVE_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\results\rms_statistics.txt'

# 硬编码参数
FS = 50  # 振动信号采样频率
TIME_WINDOW = 60.0   # 计算RMS的时间窗口（秒）
# RMS_TRHESHOLD = 0.16 # RMS阈值已改为动态计算（95%分位值）

# 颜色配置（根据阈值区分）
BELOW_THRESHOLD_COLOR = '#8074C8'  # 小于均方根阈值的颜色
ABOVE_THRESHOLD_COLOR = '#7895C1'  # 大于均方根阈值的颜色


def process_single_file(file_path, window_size):
    """单文件处理工作函数，用于多进程"""
    try:
        import numpy as np
        from ....data_processer.io_unpacker import UNPACK
        unpacker = UNPACK(init_path=False)
        vibration_data = unpacker.VIC_DATA_Unpack(file_path)
        vibration_data = np.array(vibration_data)
        
        if len(vibration_data) == 0:
            return [], []
            
        rms_list = []
        lengths = []
        
        if len(vibration_data) >= window_size:
            for i in range(0, len(vibration_data) - window_size + 1, window_size):
                window_data = vibration_data[i:i+window_size]
                # 计算信号的均方根RMS
                rms_val = np.sqrt(np.mean(np.square(window_data)))
                if rms_val > 0:
                    rms_list.append(rms_val)
                    lengths.append(len(window_data))
        else:
            rms_val = np.sqrt(np.mean(np.square(vibration_data)))
            if rms_val > 0:
                rms_list.append(rms_val)
                lengths.append(len(vibration_data))
        
        return rms_list, lengths
    except Exception as e:
        # print(f"处理文件 {file_path} 出错: {e}")
        return [], []

# 绘制线性Y轴的RMS直方图函数
def plot_rms_hist_linear_y(random_vibration_rms, rms_threshold, n_bins, font_size, ENG_FONT, CN_FONT):
    """
    绘制RMS直方图（Y轴为线性直角坐标）
    将0~RMS_THRESHOLD作为一个区间，剩余区间按n_bins划分
    """
    all_valid_rms = np.array(random_vibration_rms) if len(random_vibration_rms) > 0 else np.array([])
    if len(all_valid_rms) == 0:
        return None
    
    bin_min = np.min(all_valid_rms)
    bin_max = np.max(all_valid_rms)
    
    # 创建bins：0~threshold作为一个bin，threshold~max按n_bins划分
    bins_below = [bin_min, rms_threshold]
    bins_above = np.linspace(rms_threshold, bin_max, n_bins + 1)
    bins = np.unique(np.concatenate([bins_below, bins_above]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 根据阈值分离数据
    below_threshold = random_vibration_rms[random_vibration_rms < rms_threshold]
    above_threshold = random_vibration_rms[random_vibration_rms >= rms_threshold]

    # 绘制小于阈值的样本
    if len(below_threshold) > 0:
        ax.hist(
            below_threshold,
            bins=bins,
            color=BELOW_THRESHOLD_COLOR,
            edgecolor=BELOW_THRESHOLD_COLOR,
            linewidth=0.8,
            label=f'小于阈值',
            alpha=1.0
        )
    
    # 绘制大于等于阈值的样本
    if len(above_threshold) > 0:
        ax.hist(
            above_threshold,
            bins=bins,
            color=ABOVE_THRESHOLD_COLOR,
            edgecolor=ABOVE_THRESHOLD_COLOR,
            linewidth=0.8,
            label=f'大于等于阈值',
            alpha=1.0
        )
    
    # 添加阈值垂直虚线
    ax.axvline(x=rms_threshold, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'均方根阈值$\sigma_0$')
    
    # 坐标轴配置
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', np.min(bins)))
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    ax.set_xlabel(r'均方根（$m/s^2$）', fontproperties=CN_FONT)
    ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks()], fontproperties=ENG_FONT)
    
    ax.set_yscale('linear')
    ax.set_ylabel('样本数量', fontproperties=CN_FONT)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax.get_yticks()], fontproperties=ENG_FONT)
    
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

# 封装RMS统计结果打印函数
def print_rms_statistics(rms_threshold, total_samples, total_below_threshold, total_above_threshold,
                         random_above_threshold, all_rms):
    """
    打印RMS统计结果
    """
    print("\n" + "="*60)
    print("                    RMS样本统计结果")
    print("="*60)
    
    print(f"1. 总样本数目：{total_samples}")
    
    if len(all_rms) > 0:
        rms_p95 = np.percentile(all_rms, 95)
        current_threshold_percentile = np.sum(all_rms < rms_threshold) / len(all_rms) * 100 if len(all_rms) >0 else 0.0
        print(f"2. RMS分位数统计：")
        print(f"   - 95%分位数对应的RMS值：{rms_p95:.4f} (m/s²)")
        print(f"   - 当前阈值({rms_threshold})对应的分位数：{current_threshold_percentile:.2f}%")
    else:
        print(f"2. RMS分位数统计：")
        print(f"   - 无有效RMS样本，无法计算95%分位数")
    
    below_ratio = (total_below_threshold / total_samples * 100) if total_samples > 0 else 0.0
    above_ratio = (total_above_threshold / total_samples * 100) if total_samples > 0 else 0.0
    print(f"3. 阈值（{rms_threshold}）分类统计：")
    print(f"   - 小于阈值样本数：{total_below_threshold}，占比：{below_ratio:.2f}%")
    print(f"   - 大于等于阈值样本数：{total_above_threshold}，占比：{above_ratio:.2f}%")
    
    print(f"4. 大于等于阈值样本的类型统计：")
    if total_above_threshold > 0:
        random_above_ratio = (random_above_threshold / total_above_threshold * 100)
        print(f"   - 一般振动（随机）样本数：{random_above_threshold}，占比：{random_above_ratio:.2f}%")
    else:
        print(f"   - 无大于等于阈值的样本，无需统计类型占比")
    print("="*60 + "\n")

# 双堆叠子图绘制函数
def plot_rms_double_stacked_subplots(random_vibration_rms, rms_threshold, font_size, ENG_FONT, CN_FONT):
    """
    绘制双堆叠子图：
    左子图：从rms_threshold开始，包含剔除阈值后剩下样本中的95%
    右子图：包含剩下的5%样本（x_max ~ 最大值）
    """
    samples_above_threshold = random_vibration_rms[random_vibration_rms >= rms_threshold]
    if len(samples_above_threshold) == 0:
        return None

    left_subplot_min = rms_threshold
    # 动态计算x_max：剔除阈值后，剩下样本中的95%分位值
    left_subplot_max = np.percentile(samples_above_threshold, 95)
    
    right_subplot_min = left_subplot_max
    right_subplot_max = np.max(random_vibration_rms)
    
    interval_nums = 50
    left_interval_nums = interval_nums
    right_interval_nums = interval_nums * 2

    random_left = random_vibration_rms[(random_vibration_rms >= left_subplot_min) & (random_vibration_rms <= left_subplot_max)]
    random_right = random_vibration_rms[(random_vibration_rms > right_subplot_min) & (random_vibration_rms <= right_subplot_max)]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # -------------------------- 左子图 --------------------------
    bins_left = np.linspace(left_subplot_min, left_subplot_max, left_interval_nums + 1)
    # 计算左右对称的余量
    margin_left = (left_subplot_max - left_subplot_min) / 80
    x_start = left_subplot_min - margin_left
    x_end = left_subplot_max + margin_left

    random_hist_left, _ = np.histogram(random_left, bins=bins_left)
    ax1.bar(bins_left[:-1], random_hist_left, width=np.diff(bins_left), 
            color=ABOVE_THRESHOLD_COLOR, edgecolor=ABOVE_THRESHOLD_COLOR, linewidth=0.5, alpha=1.0)

    ax1.set_ylabel('样本数量', fontproperties=CN_FONT)
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
    ax1.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # -------------------------- 右子图 --------------------------
    bins_right = np.linspace(right_subplot_min, right_subplot_max, right_interval_nums + 1) 
    # 计算左右对称的余量
    margin_right = (right_subplot_max - right_subplot_min) / 80
    right_subplot_x_start = right_subplot_min - margin_right
    right_subplot_x_end = right_subplot_max + margin_right

    random_hist_right, _ = np.histogram(random_right, bins=bins_right)
    ax2.bar(bins_right[:-1], random_hist_right, width=np.diff(bins_right), 
            color=ABOVE_THRESHOLD_COLOR, edgecolor=ABOVE_THRESHOLD_COLOR, linewidth=0.5, alpha=1.0, label='随机振动样本')
    
    # 设置对称的范围
    ax2.set_xlim(right_subplot_x_start, right_subplot_x_end)
    ax2.set_ylabel('样本数量', fontproperties=CN_FONT)  

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
    ax2.set_xticklabels([f'{x:.1f}' for x in new_xticks_2], fontproperties=ENG_FONT, rotation=45)

    ax2.set_yticklabels(ax2.get_yticks(), fontproperties=ENG_FONT)
    # ax2.legend(prop=CN_FONT, loc='upper right', frameon=True, fancybox=True, shadow=False)
    ax2.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)

    fig.text(0.5, 0.02, r'均方根（$m/s^2$）', ha='center', fontproperties=CN_FONT)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

def RMS_Statistics_Histogram():
    # 核心参数配置
    fs_vibration = FS
    time_window = TIME_WINDOW
    # rms_threshold 将在数据收集后通过95%分位值动态计算
    
    n_bins = N_BINS
    
    result_save_path = RESULT_SAVE_PATH

    ploter = PlotLib() 
    figs = []

    # ============================================================
    # 数据获取：使用 step1_lackness_filter 获取经过缺失率筛选的文件
    # ============================================================
    print("\n[Step 0] 获取所有振动文件...")
    all_files = get_all_vibration_files()
    print(f"✓ 获取到 {len(all_files)} 个振动文件")
    
    print("\n[Step 1] 执行缺失率筛选...")
    filtered_paths, statistics = run_lackness_filter(all_files)
    
    # 从筛选结果中提取符合目标传感器的文件
    all_vib_files = []
    for file_path in filtered_paths:
        # 筛选目标传感器的文件
        if any(sensor_id in file_path for sensor_id in TARGET_VIBRATION_SENSORS):
            all_vib_files.append(file_path)
    
    print(f"✓ 筛选后文件数量：{len(filtered_paths)}")
    print(f"✓ 符合目标传感器的文件数量：{len(all_vib_files)}")

    # 数据收集
    random_vibration_rms_list = []
    window_size = int(time_window * fs_vibration)

    # 使用多进程并行获取数据
    print(f"开始并行处理文件并计算RMS...")
    with ProcessPoolExecutor() as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_file, fp, window_size): fp for fp in all_vib_files}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(all_vib_files), desc="数据获取进度"):
            try:
                rms_res, len_res = future.result()
                if rms_res:
                    random_vibration_rms_list.extend(rms_res)
            except Exception as e:
                print(f"处理任务时出错: {e}")

    random_vibration_rms = np.array(random_vibration_rms_list)

    if len(random_vibration_rms) == 0:
        print("警告：无有效振动样本数据")
        return
    
    # 动态计算RMS阈值：采取统计上的95%分位值
    rms_threshold = np.percentile(random_vibration_rms, 95)
    print(f"\n" + "="*60)
    print(f"计算得到的动态RMS阈值 (95%分位值): {rms_threshold:.4f}")
    print("="*60 + "\n")

    print(f"随机振动样本数量：{len(random_vibration_rms)}")

    # 样本数量统计
    total_samples = len(random_vibration_rms)
    random_below_threshold = len(random_vibration_rms[random_vibration_rms < rms_threshold])
    random_above_threshold = len(random_vibration_rms[random_vibration_rms >= rms_threshold])
    total_below_threshold = random_below_threshold
    total_above_threshold = random_above_threshold

    # 保存统计结果到文件
    try:
        save_dir = os.path.dirname(result_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(result_save_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("RMS样本数量统计结果\n")
            f.write("="*50 + "\n")
            f.write(f"RMS阈值：{rms_threshold}\n")
            f.write(f"总样本数量：{total_samples}\n")
            f.write(f"小于阈值的总样本数量：{total_below_threshold}\n")
            f.write(f"大于等于阈值的总样本数量：{total_above_threshold}\n")
            f.write("-"*30 + "\n")
            f.write(f"随机振动 - 小于阈值：{random_below_threshold}\n")
            f.write(f"随机振动 - 大于等于阈值：{random_above_threshold}\n")
            f.write("="*50 + "\n")
        print(f"统计结果已保存至：{result_save_path}")
    except Exception as e:
        print(f"保存统计结果失败：{e}")

    all_rms = random_vibration_rms if len(random_vibration_rms) > 0 else np.array([])

    print_rms_statistics(
        rms_threshold=rms_threshold,
        total_samples=total_samples,
        total_below_threshold=total_below_threshold,
        total_above_threshold=total_above_threshold,
        random_above_threshold=random_above_threshold,
        all_rms=all_rms
    )

    print("\n开始绘制线性Y轴RMS直方图...")
    fig_linear_y = plot_rms_hist_linear_y(
        random_vibration_rms=random_vibration_rms,
        rms_threshold=rms_threshold,
        n_bins=n_bins,
        font_size=FONT_SIZE,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_linear_y:
        figs.append(fig_linear_y)
        plt.close(fig_linear_y)

    print("\n开始绘制双区间堆叠子图...")
    fig_double_stacked = plot_rms_double_stacked_subplots(
        random_vibration_rms=random_vibration_rms,
        rms_threshold=rms_threshold,
        font_size=FONT_SIZE,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_double_stacked:
        figs.append(fig_double_stacked)
        plt.close(fig_double_stacked)

    ploter.figs.extend(figs)

    ploter.show()
