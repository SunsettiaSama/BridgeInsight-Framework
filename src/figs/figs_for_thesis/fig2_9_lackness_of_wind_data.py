"""
图2.6: 极端振动对应的风数据缺失率分布
基于含有极端窗口样本的风数据，检查每个风数据样本的缺失率并绘制分布图

功能特点：
1. 获取含有极端振动窗口的风数据（通过综合接口 get_data_pairs）
2. 检查每个风数据文件的缺失率
3. 绘制缺失率分布直方图（与 fig2_2_lackness_of_samples.py 相同的绘图逻辑）
4. 支持双子图展示：低缺失率和高缺失率样本分开显示
5. 使用 PlotLib 统一管理图表

缺失率定义：
- 无缺失: 100% (missing_rate = 0)
- 有缺失: 缺失部分越多，该部分比例逐渐下降
- 缺失率 = 1 - (actual_length / expected_length)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.statistics.wind_data_io_process.workflow import run as run_wind_workflow
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.io_unpacker import UNPACK

from src.visualize_tools.utils import PlotLib

from .config import (
    FONT_SIZE, ENG_FONT, CN_FONT,
    BELOW_THRESHOLD_COLOR, ABOVE_THRESHOLD_COLOR,
    THRESHOLD_COLOR, N_BINS
)

from src.config.sensor_config import WIND_FS

plt.style.use('default')
plt.rcParams['font.size'] = FONT_SIZE

# 常量硬编码
MISSING_RATE_THRESHOLD = 0.05  # 缺失率阈值
EXPECTED_WIND_LENGTH = int(WIND_FS * 60 * 60)  # 1Hz * 60s * 60m = 3600


def load_wind_data_raw(file_path):
    """
    加载原始风数据
    
    参数:
        file_path: 风数据文件路径
    
    返回:
        wind_velocity: 风速数组，失败返回 None
    """
    try:
        unpacker = UNPACK(init_path=False)
        wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(file_path)
        return np.array(wind_velocity)
    except Exception:
        return None


def process_single_wind_file(file_path):
    """
    计算单个风数据文件的缺失率
    
    参数:
        file_path: 风数据文件路径
    
    返回:
        缺失率（float），如果失败返回 -1
    """
    if not file_path or not os.path.exists(file_path):
        return -1
    
    wind_velocity = load_wind_data_raw(file_path)
    
    if wind_velocity is None:
        return -1
    
    actual_length = len(wind_velocity)
    missing_rate = 1.0 - (actual_length / EXPECTED_WIND_LENGTH)
    return missing_rate


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
             edgecolor='white', linewidth=0.5, alpha=0.8, label='风数据缺失率分布')
    
    # 设置 Y 轴为对数坐标
    ax.set_yscale('log')
    
    # 绘制阈值线
    ax.axvline(MISSING_RATE_THRESHOLD, color=THRESHOLD_COLOR, linestyle='--', linewidth=2, 
                label=f'阈值 ({MISSING_RATE_THRESHOLD*100:.1f}%)')
    
    ax.set_xlabel('缺失率', fontproperties=CN_FONT)
    ax.set_ylabel('风数据文件数量', fontproperties=CN_FONT)
    
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
    ax1.set_title('低缺失率风数据 (0-5%)', fontproperties=CN_FONT)
    ax1.set_ylabel('文件数量', fontproperties=CN_FONT)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # 右子图 (5 - 100%)
    # 自动确定右侧上限
    max_r = max(1.0, np.max(right_rates)) if len(right_rates) > 0 else 1.0
    ax2.hist(right_rates, bins=np.linspace(boundary, max_r, n_bins_right + 1), 
             color=ABOVE_THRESHOLD_COLOR, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax2.set_title('高缺失率风数据 (5%以上)', fontproperties=CN_FONT)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # 设置字体和网格
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(ENG_FONT)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlabel('缺失率', fontproperties=CN_FONT)

    plt.tight_layout()
    return fig


def Lackness_Of_Wind_Data_Analysis():
    print("="*80)
    print(" "*15 + "图2.6: 极端振动对应的风数据缺失率分布")
    print("="*80)
    
    ploter = PlotLib() 
    figs = []

    # Step 1: 运行振动数据工作流
    print("\n[Step 1] 运行振动数据工作流...")
    print("-"*80)
    vib_metadata = run_vib_workflow(use_cache=True)
    print(f"✓ 获取振动数据元数据: {len(vib_metadata)} 条")
    
    # Step 2: 运行风数据工作流（获取含有极端窗口的风数据）
    print("\n[Step 2] 运行风数据工作流（获取含有极端窗口的风数据）...")
    print("-"*80)
    wind_metadata = run_wind_workflow(
        vib_metadata=vib_metadata,
        use_cache=True,
        force_recompute=False,
        extreme_only=True
    )
    print(f"✓ 获取含有极端窗口的风数据元数据: {len(wind_metadata)} 条")
    
    if len(wind_metadata) == 0:
        print("警告：无可用的风数据，退出")
        return
    
    # Step 3: 计算风数据的缺失率
    print("\n[Step 3] 计算风数据缺失率（使用多进程）...")
    print("-"*80)
    
    missing_rates = []
    success_count = 0
    
    # 获取所有文件路径
    file_paths = [item.get('file_path') for item in wind_metadata]
    
    # 使用多进程并行处理文件
    print(f"开始并行处理 {len(file_paths)} 个风数据文件...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_wind_file, fp): i for i, fp in enumerate(file_paths)}
        for future in tqdm(as_completed(futures), total=len(file_paths), desc="缺失率计算"):
            missing_rate = future.result()
            if missing_rate >= 0:
                missing_rates.append(missing_rate)
                success_count += 1
    
    if len(missing_rates) == 0:
        print("警告：无有效的风数据样本")
        return
    
    missing_rates = np.array(missing_rates)
    
    # Step 4: 统计分析
    print("\n[Step 4] 统计分析...")
    print("-"*80)
    
    total_samples = len(missing_rates)
    failed_count = len(wind_metadata) - success_count
    high_missing_samples = np.sum(missing_rates > MISSING_RATE_THRESHOLD)
    avg_missing_rate = np.mean(missing_rates)
    
    print("\n" + "="*50)
    print("        风数据缺失率统计报告")
    print("="*50)
    print(f"处理风数据文件总数: {len(wind_metadata)}")
    print(f"成功处理文件数: {success_count}")
    print(f"失败/无效文件数: {failed_count}")
    print(f"预期单文件长度: {EXPECTED_WIND_LENGTH} (1Hz * 60s * 60m)")
    print(f"平均缺失率: {avg_missing_rate*100:.2f}%")
    print(f"缺失率超过阈值 ({MISSING_RATE_THRESHOLD*100:.1f}%) 的文件数: {high_missing_samples}")
    print(f"超过阈值文件占比: {high_missing_samples/total_samples*100:.2f}%")
    print("="*50 + "\n")
    
    # Step 5: 绘制缺失率分布图
    print("[Step 5] 绘制缺失率分布图...")
    print("-"*80)
    
    print("开始绘制缺失率分布图...")
    fig = plot_missing_ratio_histogram(
        missing_rates=missing_rates,
        font_size=FONT_SIZE,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    
    # 验证绘图完整性
    hist_counts, _ = np.histogram(missing_rates, bins=np.linspace(min(0, np.min(missing_rates)), max(1, np.max(missing_rates)), N_BINS + 1))
    print(f"直方图统计文件总数: {np.sum(hist_counts)} / 原始文件总数: {total_samples}")
    
    boundary = MISSING_RATE_THRESHOLD
    left_rates = missing_rates[missing_rates <= boundary]
    right_rates = missing_rates[missing_rates > boundary]
    left_counts, _ = np.histogram(left_rates, bins=np.linspace(min(0.0, np.min(left_rates)) if len(left_rates)>0 else 0, boundary, 21))
    right_counts, _ = np.histogram(right_rates, bins=np.linspace(boundary, max(1.0, np.max(right_rates)) if len(right_rates)>0 else 1.0, 81))
    print(f"双子图统计文件总数: {np.sum(left_counts) + np.sum(right_counts)} / 原始文件总数: {total_samples}")
    
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
    
    # 完成
    print("\n" + "="*80)
    print(" "*20 + "绘图处理完成")
    print("="*80)
    print(f"✓ 处理的风数据文件总数: {total_samples}")
    print(f"✓ 生成的图表数: {len(figs)}")
    print(f"✓ 数据来源：含有极端振动窗口的风数据")
    
    plt.close('all')
    ploter.show()


if __name__ == "__main__":
    Lackness_Of_Wind_Data_Analysis()
