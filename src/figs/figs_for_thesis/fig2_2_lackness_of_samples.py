from ...visualize_tools.utils import ChartApp, PlotLib
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec  

from ...data_processer.io_unpacker import UNPACK
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 从配置文件导入绘图基础配置
from .config import (
    FONT_SIZE, ENG_FONT, CN_FONT,
    BELOW_THRESHOLD_COLOR, ABOVE_THRESHOLD_COLOR,
    THRESHOLD_COLOR, N_BINS
)

RESULT_SAVE_PATH =  r'F:\Research\Vibration Characteristics In Cable Vibration\results\lackness_statistics.txt'
ALL_VIBRATION_ROOT = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC"

# 硬编码参数
FS = 50  # 振动信号采样频率
TIME_WINDOW = 60.0   # 时间窗口（秒）

def process_single_file(file_path, window_size):
    """单文件处理工作函数，用于多进程"""
    try:
        import numpy as np
        from ...data_processer.io_unpacker import UNPACK
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
        return [], []

# 绘制缺失比例双堆叠子图
def plot_missing_ratio_double_stacked_subplots(sequence_lengths, expected_length, font_size, ENG_FONT, CN_FONT):
    """
    绘制缺失比例双堆叠子图：
    左子图：缺失严重样本 (数据完整度 0 ~ 95%)，展示其分布
    右子图：基本完整样本 (数据完整度 95% ~ 100%)，展示其密集分布
    """
    if len(sequence_lengths) == 0:
        return None
    
    missing_ratios = np.array(sequence_lengths) / expected_length
    
    # 设定显示与划分界限
    ratio_min_data = 0.0
    ratio_boundary = 0.95
    ratio_max_data = 1.0
    
    # 分离数据：左闭右开逻辑，确保完整样本在边界处被归类到右侧
    ratios_left = missing_ratios[missing_ratios < ratio_boundary]
    ratios_right = missing_ratios[missing_ratios >= ratio_boundary]
    
    fig = plt.figure(figsize=(12, 6))
    # 左侧宽度占比为2 (长尾区间)，右侧宽度占比为1 (密集区间)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    interval_nums = 50
    
    # -------------------------- 左子图 (严重缺失部分，0-95%) --------------------------
    bins_left = np.linspace(ratio_min_data, ratio_boundary, interval_nums + 1)
    # 计算步长，向外扩展一个区间
    step_left = (ratio_boundary - ratio_min_data) / interval_nums
    x_start_1 = ratio_min_data - step_left
    x_end_1 = ratio_boundary + step_left
    
    ax1.hist(ratios_left, bins=bins_left, color=ABOVE_THRESHOLD_COLOR, 
             edgecolor=ABOVE_THRESHOLD_COLOR, linewidth=0.5, alpha=1.0, label='缺失样本 (0-95%)')
    
    ax1.set_ylabel('样本数量', fontproperties=CN_FONT)
    ax1.set_xlim(x_start_1, x_end_1)
    
    auto_xticks_1 = ax1.get_xticks()
    valid_auto_1 = auto_xticks_1[(auto_xticks_1 > ratio_min_data) & (auto_xticks_1 < ratio_boundary)]
    if len(auto_xticks_1) >= 2:
        default_interval_1 = np.mean(np.diff(auto_xticks_1))
        if len(valid_auto_1) > 0 and (valid_auto_1[0] - ratio_min_data) < 0.8 * default_interval_1:
            valid_auto_1 = valid_auto_1[1:]
        if len(valid_auto_1) > 0 and (ratio_boundary - valid_auto_1[-1]) < 0.8 * default_interval_1:
            valid_auto_1 = valid_auto_1[:-1]
            
    new_xticks_1 = np.unique(np.concatenate([[ratio_min_data], valid_auto_1, [ratio_boundary]]))
    new_xticks_1 = np.sort(new_xticks_1)
    ax1.set_xticks(new_xticks_1)
    ax1.set_xticklabels([f'{x*100:.1f}%' for x in new_xticks_1], fontproperties=ENG_FONT, rotation=45)
    
    ax1.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax1.get_yticks()], fontproperties=ENG_FONT)
    ax1.legend(prop=CN_FONT, loc='upper left', frameon=True, fancybox=True, shadow=False)
    ax1.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # -------------------------- 右子图 (基本完整部分，95-100%) --------------------------
    bins_right = np.linspace(ratio_boundary, ratio_max_data, interval_nums // 2 + 1)
    # 计算步长，向外扩展一个区间
    step_right = (ratio_max_data - ratio_boundary) / (interval_nums // 2)
    x_start_2 = ratio_boundary - step_right
    x_end_2 = ratio_max_data + step_right
    
    ax2.hist(ratios_right, bins=bins_right, color=BELOW_THRESHOLD_COLOR, 
             edgecolor=BELOW_THRESHOLD_COLOR, linewidth=0.5, alpha=1.0, label='正常样本 (95-100%)')
    
    ax2.set_xlim(x_start_2, x_end_2)
    ax2.set_ylabel('样本数量', fontproperties=CN_FONT)
    
    auto_xticks_2 = ax2.get_xticks()
    valid_auto_2 = auto_xticks_2[(auto_xticks_2 > ratio_boundary) & (auto_xticks_2 < ratio_max_data)]
    if len(auto_xticks_2) >= 2:
        default_interval_2 = np.mean(np.diff(auto_xticks_2))
        if len(valid_auto_2) > 0 and (valid_auto_2[0] - ratio_boundary) < 0.8 * default_interval_2:
            valid_auto_2 = valid_auto_2[1:]
        if len(valid_auto_2) > 0 and (ratio_max_data - valid_auto_2[-1]) < 0.8 * default_interval_2:
            valid_auto_2 = valid_auto_2[:-1]
            
    new_xticks_2 = np.unique(np.concatenate([[ratio_boundary], valid_auto_2, [ratio_max_data]]))
    new_xticks_2 = np.sort(new_xticks_2)
    ax2.set_xticks(new_xticks_2)
    ax2.set_xticklabels([f'{x*100:.1f}%' for x in new_xticks_2], fontproperties=ENG_FONT, rotation=45)
    
    ax2.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax2.get_yticks()], fontproperties=ENG_FONT)
    ax2.legend(prop=CN_FONT, loc='upper left', frameon=True, fancybox=True, shadow=False)
    ax2.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    fig.text(0.5, 0.02, '完整比例', ha='center', fontproperties=CN_FONT)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

def Lackness_Of_Samples_Analysis():
    # 核心参数配置
    fs_vibration = FS
    time_window = TIME_WINDOW
    
    target_sensors = [
        'ST-VIC-C34-101-02', 'ST-VIC-C34-101-01', 'ST-VIC-C34-102-01', 'ST-VIC-C34-102-02',
        'ST-VIC-C18-101-01', 'ST-VIC-C18-101-02', 'ST-VIC-C18-102-01', 'ST-VIC-C18-102-02',
        'ST-VIC-C34-201-01', 'ST-VIC-C34-201-02', 'ST-VIC-C34-202-01', 'ST-VIC-C34-202-02',
        'ST-VIC-C34-301-01', 'ST-VIC-C34-301-02', 'ST-VIC-C34-302-01', 'ST-VIC-C34-302-02',
        'ST-VIC-C18-301-01', 'ST-VIC-C18-301-02', 'ST-VIC-C18-302-01', 'ST-VIC-C18-302-02'
    ]

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

    # 数据收集
    sequence_lengths = []  # 收集窗口样本长度用于计算缺失比例
    expected_length = int(FS * TIME_WINDOW)  # 50 * 60 = 3000
    window_size = int(time_window * fs_vibration)

    all_vib_files = get_all_vibration_files(root_dir=all_vibration_root, target_sensor_ids=target_sensors)
    print(f"共获取所有振动文件数量：{len(all_vib_files)}")

    # 使用多进程并行获取数据
    print(f"开始并行处理文件并收集长度...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, fp, window_size): fp for fp in all_vib_files}
        for future in tqdm(as_completed(futures), total=len(all_vib_files), desc="数据获取进度"):
            try:
                rms_res, len_res = future.result()
                if len_res:
                    sequence_lengths.extend(len_res)
            except Exception as e:
                print(f"处理任务时出错: {e}")

    if not sequence_lengths:
        print("警告：无有效样本数据")
        return

    # 绘制缺失比例直方图
    print("\n开始绘制缺失比例堆叠子图...")
    fig_double = plot_missing_ratio_double_stacked_subplots(
        sequence_lengths=sequence_lengths,
        expected_length=expected_length,
        font_size=FONT_SIZE,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_double:
        figs.append(fig_double)
        plt.close(fig_double)

    ploter.figs.extend(figs)
    ploter.show()

if __name__ == "__main__":
    Lackness_Of_Samples_Analysis()
