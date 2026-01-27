from ...visualize_tools.utils import ChartApp, PlotLib
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec  

from ...data_processer.io_unpacker import UNPACK, DataManager
from ...data_processer.algorithms import isVIV
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker

plt.style.use('default')
font_size = 16

# ########################### 字体配置 ###########################
# 1. 全局默认字体设为Times New Roman（优先渲染英文，无中文乱码风险）
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimSun']
# 2. 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = font_size

# 3. 定义中英文字体对象
ENG_FONT = FontProperties(family='Times New Roman', size=font_size)
CN_FONT = FontProperties(
    family='SimSun', 
    size=font_size,
    # Windows系统无需指定fname，Linux/Mac需手动填写SimHei字体文件路径，示例：
    # fname='/usr/share/fonts/truetype/simhei/SimHei.ttf'
)

RESULT_SAVE_PATH =  r'F:\Research\Vibration Characteristics In Cable Vibration\results\rms_statistics.txt'
ALL_VIBRATION_ROOT = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC"

# ###################################################################


# 硬编码参数
FS = 50  # 振动信号采样频率
TIME_WINDOW = 60.0   # 计算RMS的时间窗口（秒）
RMS_TRHESHOLD = 0.16 # RMS阈值（VIV双重筛选+随机振动高低区分）

# 可视化相关
N_BINS = 100  # 高粒度分箱

# MECC判定参数组合
MECC0 = 0.02
K = 11

NORMAL_VIB_COLOR = '#D0D0D0'
NORMAL_EDGE_COLOR = '#A0A0A0'
VIV_VIB_COLOR = '#606060'
VIV_EDGE_COLOR = '#303030'
THRESHOLD_COLOR = '#202020'   # 深灰色
VIV_ALPHA = 0.6  # VIV样本透明度（区分于随机振动的不透明）

# 绘制对数Y轴的RMS直方图函数（原有逻辑完整保留）
def plot_rms_hist_log_y(random_vibration_rms, viv_rms, rms_threshold, n_bins, font_size, ENG_FONT, CN_FONT, viv_alpha):
    """
    绘制RMS直方图（Y轴为对数坐标）
    :param random_vibration_rms: 随机振动RMS数组
    :param viv_rms: VIV振动RMS数组
    :param rms_threshold: RMS阈值
    :param n_bins: 分箱数
    :param font_size: 字体大小
    :param ENG_FONT: 英文字体配置
    :param CN_FONT: 中文字体配置
    :param viv_alpha: VIV样本透明度
    :return: fig: 绘制好的matplotlib figure对象
    """
    # 1. 确定直方图分箱范围（包含两组样本）
    all_valid_rms = []
    if len(random_vibration_rms) > 0:
        all_valid_rms.extend(random_vibration_rms)
    if len(viv_rms) > 0:
        all_valid_rms.extend(viv_rms)
    all_valid_rms = np.array(all_valid_rms)
    bin_min = np.min(all_valid_rms)
    bin_max = np.max(all_valid_rms)
    bins = np.linspace(bin_min, bin_max, n_bins + 1)

    # 2. 创建画布（与原配置一致）
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. 绘制随机振动样本直方图（先绘制，作为底层）
    if len(random_vibration_rms) > 0:
        ax.hist(
            random_vibration_rms,
            bins=bins,
            color=NORMAL_VIB_COLOR,
            edgecolor= NORMAL_EDGE_COLOR, # 使用一个较深灰色作为边框
            linewidth=0.8,
            label=f'随机振动',
            alpha=1.0  # 随机振动不透明
        )

    # 4. 绘制VIV样本直方图（后绘制，叠加在随机振动上层，带透明度）
    if len(viv_rms) > 0:
        # 分离VIV的低/高RMS（此处VIV已过全局RMS阈值，可按需细分，也可直接绘制整体）
        viv_high_rms = viv_rms[viv_rms >= rms_threshold]
        
        # VIV-高RMS（核心VIV样本）
        if len(viv_high_rms) > 0:
            ax.hist(
                viv_high_rms,
                bins=bins,
                color=VIV_VIB_COLOR,
                edgecolor=VIV_EDGE_COLOR,
                linestyle='--',
                linewidth=1.0,
                label=f'涡激共振',
                alpha=viv_alpha
            )

    # 5. 添加阈值垂直虚线
    ax.axvline(x=rms_threshold, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'标准差阈值$\sigma_0$')
    # 基础标注：在X轴附近标注RMS阈值文本（仅保留在第一张图）
    ax.text(
        rms_threshold,  # X坐标：阈值位置
        ax.get_ylim()[0] * 1.1,  # Y坐标：X轴上方少许，避免遮挡
        f'RMS Threshold: {rms_threshold}',  # 标注文本
        fontproperties=ENG_FONT,  # 使用英文字体
        fontsize=font_size,
        color=THRESHOLD_COLOR,
        ha='center',  # 水平居中对齐阈值
        va='bottom'   # 垂直底部对齐（贴近X轴）
    )

    # 6. 坐标轴配置（原对数坐标逻辑）
    # X轴：RMS
    ax.set_xlabel(r'标准差（$m/s^2$）', fontproperties=CN_FONT)
    ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks()], fontproperties=ENG_FONT)
    
    # Y轴：样本数量（对数坐标）
    ax.set_yscale('log')
    ax.set_ylabel('样本数量', fontproperties=CN_FONT)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticklabels(ax.get_yticks(), fontproperties=ENG_FONT)

    # 7. 图例配置
    ax.legend(
        prop=CN_FONT,
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=False
    )

    # 8. 网格配置
    ax.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax.grid(axis='y', which='minor', alpha=0.2, linestyle='--', linewidth=0.3)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # 9. 调整布局
    plt.tight_layout()
    
    return fig

# 绘制线性Y轴（直角坐标）的RMS直方图函数（原有逻辑完整保留，新增笛卡尔坐标样式）
def plot_rms_hist_linear_y(random_vibration_rms, viv_rms, rms_threshold, n_bins, font_size, ENG_FONT, CN_FONT):
    """
    绘制RMS直方图（Y轴为线性直角坐标），样式调整为笛卡尔坐标系（仅保留左下边框）
    :param random_vibration_rms: 随机振动RMS数组
    :param viv_rms: VIV振动RMS数组
    :param rms_threshold: RMS阈值
    :param n_bins: 分箱数
    :param font_size: 字体大小
    :param ENG_FONT: 英文字体配置
    :param CN_FONT: 中文字体配置
    :return: fig: 绘制好的matplotlib figure对象
    """
    # 1. 确定直方图分箱范围（与原逻辑一致）
    all_valid_rms = []
    if len(random_vibration_rms) > 0:
        all_valid_rms.extend(random_vibration_rms)
    if len(viv_rms) > 0:
        all_valid_rms.extend(viv_rms)
    all_valid_rms = np.array(all_valid_rms)
    bin_min = np.min(all_valid_rms)
    bin_max = np.max(all_valid_rms)
    bins = np.linspace(bin_min, bin_max, n_bins + 1)
    
    # 2. 创建画布（与原配置一致）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 3. 绘制随机振动样本直方图（与原逻辑一致）
    if len(random_vibration_rms) > 0:
        ax.hist(
            random_vibration_rms,
            bins=bins,
            color=NORMAL_VIB_COLOR,
            edgecolor= NORMAL_EDGE_COLOR,
            linewidth=0.8,
            label=f'随机振动',
            alpha=1.0
        )
    
    # 4. 绘制VIV样本直方图（与原逻辑一致）
    if len(viv_rms) > 0:
        viv_high_rms = viv_rms[viv_rms >= rms_threshold]
        if len(viv_high_rms) > 0:
            ax.hist(
                viv_high_rms,
                bins=bins,
                color=VIV_VIB_COLOR,
                edgecolor=VIV_EDGE_COLOR,
                linestyle='--',
                linewidth=1.0,
                label=f'涡激共振',
                alpha=0.6
            )
    
    # 5. 添加阈值垂直虚线
    ax.axvline(x=rms_threshold, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.8, label=r'标准差阈值$\sigma_0$')
    
    # 6. 坐标轴配置
    # ========== 核心修改：笛卡尔坐标样式（去除上/右边框） ==========
    # 隐藏上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 调整X轴和Y轴的位置（让轴线交汇，更贴近笛卡尔坐标）
    ax.spines['bottom'].set_position(('data', 0))  # X轴对齐Y=0
    ax.spines['left'].set_position(('data', np.min(bins)))  # Y轴对齐X=数据最小值
    # 确保轴线样式清晰（可选：调整边框宽度，增强笛卡尔视觉）
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # X轴配置（与原逻辑一致，仅保留格式）
    ax.set_xlabel(r'标准差（$m/s^2$）', fontproperties=CN_FONT)
    ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks()], fontproperties=ENG_FONT)
    
    # Y轴配置（线性直角坐标）
    ax.set_yscale('linear')
    ax.set_ylabel('样本数量', fontproperties=CN_FONT)
    # 线性坐标刻度格式化
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax.get_yticks()], fontproperties=ENG_FONT)
    
    # 7. 图例配置（与原逻辑一致，注释保留）
    # ax.legend(
    #     prop=CN_FONT,
    #     loc='upper right',
    #     frameon=True,
    #     fancybox=True,
    #     shadow=False
    # )
    
    # 8. 网格配置（与原逻辑一致，笛卡尔风格下网格仍保留可读性）
    # ax.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    # ax.grid(axis='y', which='minor', alpha=0.2, linestyle='--', linewidth=0.3)
    # ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 9. 调整布局（与原逻辑一致）
    plt.tight_layout()
    
    return fig

# 绘制X>阈值的线性Y轴RMS直方图函数（原有逻辑完整保留）
def plot_rms_hist_linear_y_above_threshold(random_vibration_rms, viv_rms, rms_threshold, n_bins, font_size, ENG_FONT, CN_FONT):
    """
    绘制RMS直方图（X>阈值部分 + Y轴线性直角坐标）
    :param random_vibration_rms: 随机振动RMS数组
    :param viv_rms: VIV振动RMS数组
    :param rms_threshold: RMS阈值
    :param n_bins: 分箱数
    :param font_size: 字体大小
    :param ENG_FONT: 英文字体配置
    :param CN_FONT: 中文字体配置
    :return: fig: 绘制好的matplotlib figure对象
    """
    # 截取X大于阈值的部分
    random_vibration_rms_above = random_vibration_rms[random_vibration_rms > rms_threshold]
    viv_rms_above = viv_rms[viv_rms > rms_threshold]
    
    # 有效性校验
    if len(random_vibration_rms_above) == 0 and len(viv_rms_above) == 0:
        print("警告：无X>阈值的RMS样本数据")
        return None
    
    # 1. 确定直方图分箱范围（仅针对X>阈值部分）
    all_valid_rms_above = []
    if len(random_vibration_rms_above) > 0:
        all_valid_rms_above.extend(random_vibration_rms_above)
    if len(viv_rms_above) > 0:
        all_valid_rms_above.extend(viv_rms_above)
    all_valid_rms_above = np.array(all_valid_rms_above)
    bin_min = rms_threshold
    bin_max = np.max(all_valid_rms_above)
    bins = np.linspace(bin_min, bin_max, n_bins + 1)
    
    # 2. 创建画布（与原配置一致）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 3. 绘制随机振动样本直方图（仅X>阈值部分）
    if len(random_vibration_rms_above) > 0:
        ax.hist(
            random_vibration_rms_above,
            bins=bins,
            color=NORMAL_VIB_COLOR,
            edgecolor= NORMAL_EDGE_COLOR,
            linewidth=0.8,
            label=f'随机振动（X>{rms_threshold}）',
            alpha=1.0
        )
    
    # 4. 绘制VIV样本直方图（仅X>阈值部分）
    if len(viv_rms_above) > 0:
        ax.hist(
            viv_rms_above,
            bins=bins,
            color=VIV_VIB_COLOR,
            edgecolor=VIV_EDGE_COLOR,
            linestyle='--',
            linewidth=1.0,
            label=f'涡激共振（X>{rms_threshold}）',
            alpha=0.6
        )
    
    ax.set_xlim(left=rms_threshold)

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticklabels([f'{int(x)}' if x.is_integer() else f'{x:.1f}' for x in ax.get_yticks()], fontproperties=ENG_FONT)
    
    # 8. 网格配置（与原逻辑一致）
    ax.grid(axis='y', which='major', alpha=0.5, linestyle='-', linewidth=0.5)
    ax.grid(axis='y', which='minor', alpha=0.2, linestyle='--', linewidth=0.3)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 9. 调整布局（与原逻辑一致）
    plt.tight_layout()
    
    return fig

# 新增：封装RMS统计结果打印函数（核心新增逻辑）
def print_rms_statistics(rms_threshold, total_samples, total_below_threshold, total_above_threshold,
                         random_above_threshold, viv_above_threshold, all_rms):
    """
    打印RMS统计结果，包含总样本数、95%分位数、阈值分类统计、大于阈值样本的类型占比
    :param rms_threshold: RMS阈值
    :param total_samples: 总样本数目
    :param total_below_threshold: 小于阈值的样本数
    :param total_above_threshold: 大于等于阈值的样本数
    :param random_above_threshold: 大于阈值的随机振动（一般）样本数
    :param viv_above_threshold: 大于阈值的VIV样本数
    :param all_rms: 所有样本的RMS数组（np.array），用于计算95%分位数
    """
    print("\n" + "="*60)
    print("                    RMS样本统计结果")
    print("="*60)
    
    # 1. 总样本数目
    print(f"1. 总样本数目：{total_samples}")
    
    # 新增：计算并打印95%分位数对应的RMS值（处理空数组边界情况）
    if len(all_rms) > 0:
        rms_p95 = np.percentile(all_rms, 95)  # 计算95%分位数
        # 补充：计算当前手动阈值对应的分位数（反向验证，可选但实用）
        current_threshold_percentile = np.sum(all_rms < rms_threshold) / len(all_rms) * 100 if len(all_rms) >0 else 0.0
        print(f"2. RMS分位数统计：")
        print(f"   - 95%分位数对应的RMS值：{rms_p95:.4f} (m/s²)")
        print(f"   - 当前阈值({rms_threshold})对应的分位数：{current_threshold_percentile:.2f}%")
    else:
        print(f"2. RMS分位数统计：")
        print(f"   - 无有效RMS样本，无法计算95%分位数")
    
    # 3. 大于/小于阈值的样本数、占比百分比（处理除零错误）
    below_ratio = (total_below_threshold / total_samples * 100) if total_samples > 0 else 0.0
    above_ratio = (total_above_threshold / total_samples * 100) if total_samples > 0 else 0.0
    print(f"3. 阈值（{rms_threshold}）分类统计：")
    print(f"   - 小于阈值样本数：{total_below_threshold}，占比：{below_ratio:.2f}%")
    print(f"   - 大于等于阈值样本数：{total_above_threshold}，占比：{above_ratio:.2f}%")
    
    # 4. 大于阈值样本中的类型统计（处理除零错误）
    print(f"4. 大于等于阈值样本的类型统计：")
    if total_above_threshold > 0:
        random_above_ratio = (random_above_threshold / total_above_threshold * 100)
        viv_above_ratio = (viv_above_threshold / total_above_threshold * 100)
        print(f"   - 一般振动（随机）样本数：{random_above_threshold}，占比：{random_above_ratio:.2f}%")
        print(f"   - VIV样本数：{viv_above_threshold}，占比：{viv_above_ratio:.2f}%")
    else:
        print(f"   - 无大于等于阈值的样本，无需统计类型占比")
    print("="*60 + "\n")


# -------------------------- 核心修改：双堆叠子图绘制函数 --------------------------
def plot_rms_double_stacked_subplots(random_vibration_rms, viv_rms, rms_threshold, font_size, ENG_FONT, CN_FONT):
    """
    绘制双堆叠子图：左子图（0.16~0.5区间）、右子图（0.5~20区间）
    核心调整：统一x/y轴标签，x标签放在整个图表最底部
    """
    # 1. 定义区间范围（原逻辑完全保留）
    left_subplot_min = rms_threshold  # 0.16
    left_subplot_max = 0.5
    
    right_subplot_max = 20.0
    interval_nums = 50

    left_interval_nums = interval_nums
    right_interval_nums = interval_nums * 2
    right_subplot_min = left_subplot_max

    # 2. 筛选对应区间的样本（原逻辑完全保留）
    random_left = random_vibration_rms[(random_vibration_rms >= left_subplot_min) & (random_vibration_rms <= left_subplot_max)]
    viv_left = viv_rms[(viv_rms >= left_subplot_min) & (viv_rms <= left_subplot_max)]
    random_right = random_vibration_rms[(random_vibration_rms >= right_subplot_min) & (random_vibration_rms <= right_subplot_max)]
    viv_right = viv_rms[(viv_rms >= right_subplot_min) & (viv_rms <= right_subplot_max)]

    # 3. 创建画布（原逻辑完全保留）
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])  # 左侧占1/3，右侧占2/3
    ax1 = fig.add_subplot(gs[0, 0])  # 左子图
    ax2 = fig.add_subplot(gs[0, 1])  # 右子图（独立纵轴）

    # -------------------------- 左子图：0.16~0.5（原逻辑完全保留） --------------------------
    bins_left = np.linspace(left_subplot_min, left_subplot_max, left_interval_nums + 1)
    x_start = rms_threshold - (left_subplot_max - left_subplot_min) / 81 

    random_hist_left, _ = np.histogram(random_left, bins=bins_left)
    viv_hist_left, _ = np.histogram(viv_left, bins=bins_left)
    ax1.bar(bins_left[:-1], random_hist_left, width=np.diff(bins_left), 
            color=NORMAL_VIB_COLOR, edgecolor=NORMAL_EDGE_COLOR, linewidth=0.5, alpha=1.0)
    ax1.bar(bins_left[:-1], viv_hist_left, width=np.diff(bins_left), bottom=random_hist_left,
            color=VIV_VIB_COLOR, edgecolor=VIV_EDGE_COLOR, linewidth=0.5, alpha=VIV_ALPHA)

    # 坐标轴配置（仅移除左子图xlabel，保留ylabel和其他逻辑）
    # ========== 修改1：移除左子图单独的xlabel（改为全局统一） ==========
    # ax1.set_xlabel(r'标准差（$m/s^2$）', fontproperties=CN_FONT)  # 注释/删除该行
    ax1.set_ylabel('样本数量', fontproperties=CN_FONT)  # 保留ylabel（统一y标签）
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

    # -------------------------- 右子图：0.5~20（原逻辑完全保留） --------------------------
    bins_right = np.linspace(right_subplot_min, right_subplot_max, right_interval_nums + 1) 
    right_subplot_x_start = right_subplot_min - (right_subplot_max - right_subplot_min) / (right_interval_nums + 1) 

    random_hist_right, _ = np.histogram(random_right, bins=bins_right)
    viv_hist_right, _ = np.histogram(viv_right, bins=bins_right)
    ax2.bar(bins_right[:-1], random_hist_right, width=np.diff(bins_right), 
            color=NORMAL_VIB_COLOR, edgecolor=NORMAL_EDGE_COLOR, linewidth=0.5, alpha=1.0, label='随机振动样本')
    ax2.bar(bins_right[:-1], viv_hist_right, width=np.diff(bins_right), bottom=random_hist_right,
            color=VIV_VIB_COLOR, edgecolor=VIV_EDGE_COLOR, linewidth=0.5, alpha=VIV_ALPHA, label='涡激共振样本')
    
    # 坐标轴配置（新增右子图ylabel，保留其他逻辑）
    ax2.set_xlim(right_subplot_x_start, right_subplot_max)
    # ========== 修改2：给右子图添加统一的ylabel（样本数量） ==========
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

    # ========== 修改3：在整个图表最底部添加统一的xlabel（标准差+单位） ==========
    fig.text(0.5, 0.02, r'标准差（$m/s^2$）', ha='center', fontproperties=CN_FONT)

    # 整体布局调整（原逻辑完全保留，注意调整布局避免label被遮挡）
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 微调rect，给底部xlabel留空间
    return fig

def RMS_Statistics_Histogram():
    # ########################### 核心参数配置 ###########################
    # RMS计算相关
    fs_vibration = FS  # 振动信号采样频率
    time_window = TIME_WINDOW  # 计算RMS的时间窗口（秒）
    rms_threshold = RMS_TRHESHOLD # RMS阈值（VIV双重筛选+随机振动高低区分）
    
    # 可视化相关
    n_bins = N_BINS  # 高粒度分箱
    viv_alpha = VIV_ALPHA  # VIV样本透明度（区分于随机振动的不透明）
    
    # 结果保存路径配置
    result_save_path = RESULT_SAVE_PATH

    # ###################################################################
    # 目标传感器
    target_sensors = [
        'ST-VIC-C18-102-01'   # 对应振动传感器ID
    ]

    # 路径配置
    all_vibration_root = ALL_VIBRATION_ROOT  # 所有振动数据根目录

    ploter = PlotLib() 
    unpacker = UNPACK(init_path = False)
    figs = []  # 存储RMS统计直方图，用于tk交互

    # ------------------- 递归遍历获取所有振动文件路径 -------------------
    def get_all_vibration_files(root_dir, target_sensor_ids, suffix=".VIC"):
        """
        递归遍历目录下所有振动文件
        :param root_dir: 根目录
        :param target_sensor_ids: 目标传感器ID列表（用于筛选文件）
        :param suffix: 振动文件后缀（.VIC/.vic）
        :return: 符合条件的振动文件路径列表
        """
        vibration_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # 筛选后缀匹配 + 包含目标传感器ID的文件
                if file.upper().endswith(suffix.upper()):
                    if any(sensor_id in file for sensor_id in target_sensor_ids):
                        file_path = os.path.join(root, file)
                        vibration_files.append(file_path)
        return vibration_files
    # -------------------------------------------------------------------

    # ------------------- 核心工具函数 -------------------
    def calculate_rms(signal_data):
        """计算信号的均方根RMS"""
        if len(signal_data) == 0:
            return 0
        return np.sqrt(np.mean(np.square(signal_data)))

    # ------------------- 单样本VIV识别函数 -------------------
    def is_viv_single_sample(sample_data, fs, **kwargs):
        """
        单样本VIV识别接口（MECC方法，仅返回布尔值）
        :param sample_data: 单个振动样本序列（np.array，长度需匹配窗口大小）
        :param fs: 采样频率（MECC方法所需）
        :param kwargs: 其他MECC自定义参数（如阶数、判别阈值等）
        :return: bool: True=是VIV，False=不是VIV
        """
        is_viv = isVIV(sample_data, f0times = K, mecc0 = MECC0)        
        return is_viv
    # -------------------------------------------------------------------

    # ########################### 第一步：数据分离 - 随机振动 vs VIV ###########################
    # 数据存储：明确分离随机振动和VIV样本
    random_vibration_rms_list = []  # 随机振动样本（非VIV）
    viv_rms_list = []               # VIV样本（MECC识别 + RMS阈值筛选）
    window_size = int(time_window * fs_vibration)  # 窗口大小（样本点数）

    # 1. 获取所有振动文件路径
    all_vib_files = get_all_vibration_files(
        root_dir=all_vibration_root,
        target_sensor_ids=target_sensors
    )
    print(f"共获取所有振动文件数量：{len(all_vib_files)}")

    # 2. 逐个解析文件并处理（数据分离核心逻辑）
    for file_path in all_vib_files:
        try:
            # 解析振动数据
            vibration_data = unpacker.VIC_DATA_Unpack(file_path)
            vibration_data = np.array(vibration_data)
        except Exception as e:
            print(f"解析振动文件失败：{file_path}，错误信息：{e}")
            continue
        
        # 数据清洗
        if len(vibration_data) == 0:
            print(f"警告：{file_path} 无有效振动数据，跳过")
            continue
        
        # 按时间窗口分段处理（逐个样本判断，实现数据分离）
        if len(vibration_data) >= window_size:
            for i in range(0, len(vibration_data) - window_size + 1, window_size):
                window_data = vibration_data[i:i+window_size]  # 单个窗口样本
                rms_val = calculate_rms(window_data)
                
                # 仅保留有效RMS样本（rms>0）
                if rms_val <= 0:
                    continue
                
                # VIV样本：双重筛选（MECC识别 + RMS阈值≥rms_threshold）
                is_viv = is_viv_single_sample(window_data, fs_vibration)
                if is_viv and rms_val >= rms_threshold:
                    viv_rms_list.append(rms_val)
                # 随机振动样本：非VIV 或 VIV但RMS低于阈值
                else:
                    random_vibration_rms_list.append(rms_val)
        else:
            # 数据长度不足窗口大小时处理
            rms_val = calculate_rms(vibration_data)
            if rms_val <= 0:
                continue
            
            # VIV样本：双重筛选
            is_viv = is_viv_single_sample(vibration_data, fs_vibration)
            if is_viv and rms_val >= rms_threshold:
                viv_rms_list.append(rms_val)
            # 随机振动样本
            else:
                random_vibration_rms_list.append(rms_val)

    # 转换为numpy数组
    random_vibration_rms = np.array(random_vibration_rms_list)
    viv_rms = np.array(viv_rms_list)

    # 有效性校验
    if len(random_vibration_rms) == 0 and len(viv_rms) == 0:
        print("警告：无有效振动样本数据")
        return
    print(f"随机振动样本数量：{len(random_vibration_rms)}")
    print(f"VIV样本数量（MECC+RMS阈值筛选）：{len(viv_rms)}")

    # 样本数量统计
    # 1. 基础统计
    total_samples = len(random_vibration_rms) + len(viv_rms)  # 总样本数
    # 2. 随机振动样本统计
    random_below_threshold = len(random_vibration_rms[random_vibration_rms < rms_threshold])
    random_above_threshold = len(random_vibration_rms[random_vibration_rms >= rms_threshold])
    # 3. 涡激共振样本统计
    viv_below_threshold = len(viv_rms[viv_rms < rms_threshold]) if len(viv_rms) > 0 else 0
    viv_above_threshold = len(viv_rms[viv_rms >= rms_threshold]) if len(viv_rms) > 0 else 0
    # 4. 按阈值汇总
    total_below_threshold = random_below_threshold + viv_below_threshold
    total_above_threshold = random_above_threshold + viv_above_threshold

    # 保存统计结果到文件
    try:
        # 创建保存目录（如果不存在）
        save_dir = os.path.dirname(result_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 写入统计信息
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
            f.write("-"*30 + "\n")
            f.write(f"涡激共振 - 小于阈值：{viv_below_threshold}\n")
            f.write(f"涡激共振 - 大于等于阈值：{viv_above_threshold}\n")
            f.write("="*50 + "\n")
        print(f"统计结果已保存至：{result_save_path}")
    except Exception as e:
        print(f"保存统计结果失败：{e}")

    # 主函数中先构建all_rms数组（原有代码中已有相关变量，直接组合即可）
    all_rms = np.concatenate([random_vibration_rms, viv_rms]) if (len(random_vibration_rms) + len(viv_rms)) >0 else np.array([])

    # 调用打印函数时传入all_rms
    print_rms_statistics(
        rms_threshold=rms_threshold,
        total_samples=total_samples,
        total_below_threshold=total_below_threshold,
        total_above_threshold=total_above_threshold,
        random_above_threshold=random_above_threshold,
        viv_above_threshold=viv_above_threshold,
        all_rms=all_rms  # 新增传入的参数
    )

    # ########################### 第二步：调用封装函数绘制所有直方图 ###########################
    # 绘制对数Y轴RMS直方图
    print("\n开始绘制对数Y轴RMS直方图...")
    fig_log_y = plot_rms_hist_log_y(
        random_vibration_rms=random_vibration_rms,
        viv_rms=viv_rms,
        rms_threshold=rms_threshold,
        n_bins=n_bins,
        font_size=font_size,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT,
        viv_alpha=viv_alpha
    )
    if fig_log_y:
        figs.append(fig_log_y)
        plt.close(fig_log_y)

    # 绘制线性Y轴直方图
    print("\n开始绘制线性Y轴RMS直方图...")
    fig_linear_y = plot_rms_hist_linear_y(
        random_vibration_rms=random_vibration_rms,
        viv_rms=viv_rms,
        rms_threshold=rms_threshold,
        n_bins=n_bins,
        font_size=font_size,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_linear_y:
        figs.append(fig_linear_y)
        plt.close(fig_linear_y)
    
    # 绘制X>阈值的线性Y轴直方图
    print("\n开始绘制X>阈值的线性Y轴直方图...")
    fig_linear_y_above = plot_rms_hist_linear_y_above_threshold(
        random_vibration_rms=random_vibration_rms,
        viv_rms=viv_rms,
        rms_threshold=rms_threshold,
        n_bins=n_bins,
        font_size=font_size,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_linear_y_above:
        figs.append(fig_linear_y_above)
        plt.close(fig_linear_y_above)
    
    # -------------------------- 新增：调用双堆叠子图绘制函数 --------------------------
    print("\n开始绘制双区间堆叠子图...")
    fig_double_stacked = plot_rms_double_stacked_subplots(
        random_vibration_rms=random_vibration_rms,
        viv_rms=viv_rms,
        rms_threshold=rms_threshold,
        font_size=font_size,
        ENG_FONT=ENG_FONT,
        CN_FONT=CN_FONT
    )
    if fig_double_stacked:
        figs.append(fig_double_stacked)
        plt.close(fig_double_stacked)
    
    # 将所有fig添加到ploter
    ploter.figs.extend(figs)

    # ########################### 展示图表 ###########################
    ploter.show()

# 改动备注：
# 1. 移除了所有原有修改标注（✅），保持代码整洁
# 2. 新增函数print_rms_statistics：封装RMS统计结果打印逻辑，包含：
#    - 总样本数目
#    - 大于/小于阈值的样本数及占比百分比（保留2位小数）
#    - 大于阈值样本中一般振动/VIV样本的数目及占比百分比
# 3. 在主函数RMS_Statistics_Histogram中，保存统计文件后调用新增的打印函数
# 4. 所有原有导入逻辑、执行逻辑、绘图逻辑均未修改
# 5. 打印函数中处理了除零错误，避免统计时出现除以零的异常
# -------------------------- 新增改动备注 --------------------------
# 6. 新增函数plot_rms_double_stacked_subplots：实现双区间堆叠子图绘制，核心配置：
#    - 左子图：0.16~2.5区间，50个细粒度bins，标注阈值，无图例
#    - 右子图：2.5~20区间，20个粗粒度bins，显示图例，堆叠样式（随机在下，VIV在上）
#    - 两个子图在一个fig中左右排列，共享Y轴刻度
# 7. 在主函数中调用该新函数，将生成的fig加入figs列表，参与后续展示
# 8. 新函数中严格遵循要求：
#    - Threshold开始位置为0.16（非0）
#    - 堆叠型展示（随机振动底层，VIV振动上层）
#    - 0~2.5区间粒度增大（50个bins）
#    - label仅显示在2.5~20的右子图上
#    - 两个子图放在一个fig中统一显示（左右布局）