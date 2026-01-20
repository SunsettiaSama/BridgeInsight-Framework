from ...visualize_tools.utils import ChartApp, PlotLib
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec  

from ...data_processer.data_processer_V0 import UNPACK, DataManager
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker

# 从配置文件导入绘图基础配置
from .config import (
    FONT_SIZE, ENG_FONT, CN_FONT,
    NORMAL_VIB_COLOR, NORMAL_EDGE_COLOR,
    THRESHOLD_COLOR, N_BINS
)

RESULT_SAVE_PATH =  r'F:\Research\Vibration Characteristics In Cable Vibration\results\rms_statistics.txt'
ALL_VIBRATION_ROOT = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC"

# 硬编码参数
FS = 50  # 振动信号采样频率
TIME_WINDOW = 60.0   # 计算RMS的时间窗口（秒）
RMS_TRHESHOLD = 0.16 # RMS阈值

# 颜色配置（根据阈值区分）
BELOW_THRESHOLD_COLOR = '#8074C8'  # 小于标准差阈值的颜色
ABOVE_THRESHOLD_COLOR = '#7895C1'  # 大于标准差阈值的颜色

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
    ax.axvline(x=rms_threshold, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'标准差阈值$\sigma_0$')
    
    # 坐标轴配置
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', np.min(bins)))
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    ax.set_xlabel(r'标准差（$m/s^2$）', fontproperties=CN_FONT)
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
    绘制双堆叠子图：左子图（0.16~0.5区间）、右子图（0.5~20区间）
    """
    left_subplot_min = rms_threshold
    left_subplot_max = 0.5
    
    right_subplot_max = 20.0
    interval_nums = 50

    left_interval_nums = interval_nums
    right_interval_nums = interval_nums * 2
    right_subplot_min = left_subplot_max

    random_left = random_vibration_rms[(random_vibration_rms >= left_subplot_min) & (random_vibration_rms <= left_subplot_max)]
    random_right = random_vibration_rms[(random_vibration_rms >= right_subplot_min) & (random_vibration_rms <= right_subplot_max)]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    bins_left = np.linspace(left_subplot_min, left_subplot_max, left_interval_nums + 1)
    x_start = rms_threshold - (left_subplot_max - left_subplot_min) / 81 

    random_hist_left, _ = np.histogram(random_left, bins=bins_left)
    ax1.bar(bins_left[:-1], random_hist_left, width=np.diff(bins_left), 
            color=ABOVE_THRESHOLD_COLOR, edgecolor=ABOVE_THRESHOLD_COLOR, linewidth=0.5, alpha=1.0)

    ax1.set_ylabel('样本数量', fontproperties=CN_FONT)
    ax1.set_xlim(x_start, left_subplot_max)

    auto_xticks_1 = ax1.get_xticks()
    new_xticks_1 = np.unique(np.concatenate([[left_subplot_min], auto_xticks_1, [left_subplot_max]]))
    new_xticks_1 = new_xticks_1[(new_xticks_1 >= left_subplot_min) & (new_xticks_1 <= left_subplot_max)]
    new_xticks_1 = np.sort(new_xticks_1)
    ax1.set_xticks(new_xticks_1)
    ax1.set_xticklabels([f'{x:.2f}' for x in new_xticks_1], fontproperties=ENG_FONT, rotation=45)

    ax1.set_xticklabels([f'{x:.2f}' for x in ax1.get_xticks()], fontproperties=ENG_FONT, rotation=45)
    ax1.set_yticklabels(ax1.get_yticks(), fontproperties=ENG_FONT)
    ax1.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    bins_right = np.linspace(right_subplot_min, right_subplot_max, right_interval_nums + 1) 
    right_subplot_x_start = right_subplot_min - (right_subplot_max - right_subplot_min) / (right_interval_nums + 1) 

    random_hist_right, _ = np.histogram(random_right, bins=bins_right)
    ax2.bar(bins_right[:-1], random_hist_right, width=np.diff(bins_right), 
            color=ABOVE_THRESHOLD_COLOR, edgecolor=ABOVE_THRESHOLD_COLOR, linewidth=0.5, alpha=1.0, label='随机振动样本')
    
    ax2.set_xlim(right_subplot_x_start, right_subplot_max)
    ax2.set_ylabel('样本数量', fontproperties=CN_FONT)  

    auto_xticks_2 = ax2.get_xticks()
    new_xticks_2 = np.unique(np.concatenate([[right_subplot_min], auto_xticks_2, [right_subplot_max]]))
    new_xticks_2 = new_xticks_2[(new_xticks_2 >= right_subplot_min) & (new_xticks_2 <= right_subplot_max)]
    new_xticks_2 = np.sort(new_xticks_2)
    ax2.set_xticks(new_xticks_2)
    ax2.set_xticklabels([f'{x:.1f}' for x in new_xticks_2], fontproperties=ENG_FONT, rotation=45)

    ax2.set_yticklabels(ax2.get_yticks(), fontproperties=ENG_FONT)
    ax2.legend(prop=CN_FONT, loc='upper right', frameon=True, fancybox=True, shadow=False)
    ax2.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)

    fig.text(0.5, 0.02, r'标准差（$m/s^2$）', ha='center', fontproperties=CN_FONT)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

# 绘制缺失比例直方图
def plot_missing_ratio_histogram(sequence_lengths, expected_length, n_bins, ENG_FONT, CN_FONT):
    """
    绘制缺失比例直方图（5%-95%分位）
    :param sequence_lengths: 样本长度列表
    :param expected_length: 期望长度（50*60=3000）
    :param n_bins: 分箱数
    """
    if len(sequence_lengths) == 0:
        return None
    
    # 计算缺失比例：当前样本长度/期望长度
    missing_ratios = np.array(sequence_lengths) / expected_length
    
    # 计算5%和95%分位数作为阈值
    ratio_5_percentile = np.percentile(missing_ratios, 5)
    ratio_95_percentile = np.percentile(missing_ratios, 95)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bin_min = np.min(missing_ratios)
    bin_max = np.max(missing_ratios)
    bins = np.linspace(bin_min, bin_max, n_bins + 1)
    
    # 根据5%和95%分位数分离数据
    below_5 = missing_ratios[missing_ratios < ratio_5_percentile]
    above_95 = missing_ratios[missing_ratios >= ratio_5_percentile]
    
    # 绘制小于5%分位数的样本
    if len(below_5) > 0:
        ax.hist(
            below_5,
            bins=bins,
            color=BELOW_THRESHOLD_COLOR,
            edgecolor=BELOW_THRESHOLD_COLOR,
            linewidth=0.8,
            label=f'5%以下样本',
            alpha=1.0
        )
    
    # 绘制大于等于5%分位数的样本
    if len(above_95) > 0:
        ax.hist(
            above_95,
            bins=bins,
            color=ABOVE_THRESHOLD_COLOR,
            edgecolor=ABOVE_THRESHOLD_COLOR,
            linewidth=0.8,
            label=f'95%以上样本',
            alpha=1.0
        )
    
    # 添加5%分位数垂直虚线
    ax.axvline(x=ratio_5_percentile, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'5%分位线')
    ax.text(
        ratio_5_percentile,
        ax.get_ylim()[1] * 0.95,
        f'5%: {ratio_5_percentile:.2f}',
        fontproperties=ENG_FONT,
        fontsize=FONT_SIZE,
        color=THRESHOLD_COLOR,
        ha='center',
        va='top'
    )
    
    # 添加95%分位数垂直虚线
    ax.axvline(x=ratio_95_percentile, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'95%分位线')
    ax.text(
        ratio_95_percentile,
        ax.get_ylim()[1] * 0.85,  # 调整y轴位置避免和5%文本重叠
        f'95%: {ratio_95_percentile:.2f}',
        fontproperties=ENG_FONT,
        fontsize=FONT_SIZE,
        color=THRESHOLD_COLOR,
        ha='center',
        va='top'
    )
    
    ax.set_xlabel('缺失比例', fontproperties=CN_FONT)
    ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks()], fontproperties=ENG_FONT)
    
    ax.set_ylabel('样本数量', fontproperties=CN_FONT)
    ax.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax.get_yticks()], fontproperties=ENG_FONT)
    
    # ========== 仅新增这一行：设置y轴为对数刻度 ==========
    ax.set_yscale('log')
    
    # 图例配置
    ax.legend(
        prop=CN_FONT,
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=False
    )
    
    ax.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

# 绘制缺失比例直方图
def plot_missing_ratio_histogram_logy(sequence_lengths, expected_length, n_bins, ENG_FONT, CN_FONT):
    """
    绘制缺失比例直方图（5%-95%分位）
    :param sequence_lengths: 样本长度列表
    :param expected_length: 期望长度（50*60=3000）
    :param n_bins: 分箱数
    """
    if len(sequence_lengths) == 0:
        return None
    
    # 计算缺失比例：当前样本长度/期望长度
    missing_ratios = np.array(sequence_lengths) / expected_length
    
    # 计算5%和95%分位数作为阈值
    ratio_5_percentile = np.percentile(missing_ratios, 5)
    ratio_95_percentile = np.percentile(missing_ratios, 95)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bin_min = np.min(missing_ratios)
    bin_max = np.max(missing_ratios)
    bins = np.linspace(bin_min, bin_max, n_bins + 1)
    
    # 根据5%和95%分位数分离数据
    below_5 = missing_ratios[missing_ratios < ratio_5_percentile]
    above_95 = missing_ratios[missing_ratios >= ratio_5_percentile]
    
    # 绘制小于5%分位数的样本
    if len(below_5) > 0:
        ax.hist(
            below_5,
            bins=bins,
            color=BELOW_THRESHOLD_COLOR,
            edgecolor=BELOW_THRESHOLD_COLOR,
            linewidth=0.8,
            label=f'5%以下样本',
            alpha=1.0
        )
    
    # 绘制大于等于5%分位数的样本
    if len(above_95) > 0:
        ax.hist(
            above_95,
            bins=bins,
            color=ABOVE_THRESHOLD_COLOR,
            edgecolor=ABOVE_THRESHOLD_COLOR,
            linewidth=0.8,
            label=f'95%以上样本',
            alpha=1.0
        )
    
    # 添加5%分位数垂直虚线
    ax.axvline(x=ratio_5_percentile, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'5%分位线')
    ax.text(
        ratio_5_percentile,
        ax.get_ylim()[1] * 0.95,
        f'5%: {ratio_5_percentile:.2f}',
        fontproperties=ENG_FONT,
        fontsize=FONT_SIZE,
        color=THRESHOLD_COLOR,
        ha='center',
        va='top'
    )
    
    # 添加95%分位数垂直虚线
    ax.axvline(x=ratio_95_percentile, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'95%分位线')
    ax.text(
        ratio_95_percentile,
        ax.get_ylim()[1] * 0.85,  # 调整y轴位置避免和5%文本重叠
        f'95%: {ratio_95_percentile:.2f}',
        fontproperties=ENG_FONT,
        fontsize=FONT_SIZE,
        color=THRESHOLD_COLOR,
        ha='center',
        va='top'
    )
    
    ax.set_xlabel('缺失比例', fontproperties=CN_FONT)
    ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks()], fontproperties=ENG_FONT)
    
    ax.set_ylabel('样本数量', fontproperties=CN_FONT)
    ax.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax.get_yticks()], fontproperties=ENG_FONT)
    
    # 图例配置
    ax.legend(
        prop=CN_FONT,
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=False
    )
    
    ax.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig



def RMS_Statistics_Histogram():
    # 核心参数配置
    fs_vibration = FS
    time_window = TIME_WINDOW
    rms_threshold = RMS_TRHESHOLD
    
    n_bins = N_BINS
    
    result_save_path = RESULT_SAVE_PATH

    target_sensors = [
        'ST-VIC-C18-102-01'
    ]

    all_vibration_root = ALL_VIBRATION_ROOT

    ploter = PlotLib() 
    unpacker = UNPACK(init_path = False)
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

    def calculate_rms(signal_data):
        """计算信号的均方根RMS"""
        if len(signal_data) == 0:
            return 0
        return np.sqrt(np.mean(np.square(signal_data)))

    # 数据收集
    random_vibration_rms_list = []
    sequence_lengths = []  # 收集窗口样本长度用于计算缺失比例
    expected_length = int(FS * TIME_WINDOW)  # 50 * 60 = 3000
    window_size = int(time_window * fs_vibration)

    all_vib_files = get_all_vibration_files(
        root_dir=all_vibration_root,
        target_sensor_ids=target_sensors
    )
    print(f"共获取所有振动文件数量：{len(all_vib_files)}")

    for file_path in all_vib_files:
        try:
            vibration_data = unpacker.VIC_DATA_Unpack(file_path)
            vibration_data = np.array(vibration_data)
        except Exception as e:
            print(f"解析振动文件失败：{file_path}，错误信息：{e}")
            continue
        
        if len(vibration_data) == 0:
            print(f"警告：{file_path} 无有效振动数据，跳过")
            continue
        
        if len(vibration_data) >= window_size:
            for i in range(0, len(vibration_data) - window_size + 1, window_size):
                window_data = vibration_data[i:i+window_size]
                rms_val = calculate_rms(window_data)
                
                if rms_val <= 0:
                    continue
                
                # 收集窗口样本长度
                sequence_lengths.append(len(window_data))
                random_vibration_rms_list.append(rms_val)
        else:
            rms_val = calculate_rms(vibration_data)
            if rms_val <= 0:
                continue
            
            # 收集窗口样本长度
            sequence_lengths.append(len(vibration_data))
            random_vibration_rms_list.append(rms_val)

    random_vibration_rms = np.array(random_vibration_rms_list)

    if len(random_vibration_rms) == 0:
        print("警告：无有效振动样本数据")
        return
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
    
    # 绘制缺失比例直方图
    print("\n开始绘制缺失比例直方图...")
    fig_missing_ratio = plot_missing_ratio_histogram(
        sequence_lengths=sequence_lengths,
        expected_length=expected_length,
        n_bins=50,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_missing_ratio:
        figs.append(fig_missing_ratio)
        plt.close(fig_missing_ratio)


    # 绘制缺失比例直方图
    print("\n开始绘制缺失比例直方图...")
    fig_missing_ratio = plot_missing_ratio_histogram_logy(
        sequence_lengths=sequence_lengths,
        expected_length=expected_length,
        n_bins=50,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_missing_ratio:
        figs.append(fig_missing_ratio)
        plt.close(fig_missing_ratio)

    ploter.figs.extend(figs)

    ploter.show()
