"""
风数据样本检查工具
简单的风速、风向、风攻角图像检查
依赖 unpacker 和 visualize_tools
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.visualize_tools.utils import PlotLib

# 导入绘图配置
from src.figs.figs_for_thesis.config import ENG_FONT, CN_FONT, FONT_SIZE, REC_FIG_SIZE

# 配置 matplotlib
plt.style.use('default')
plt.rcParams['font.size'] = FONT_SIZE


# 风速检查路径
WIND_FILE_PATH = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\UAN\09\10\ST-UAN-G02-002-01_110000.UAN"
    




def check_wind_sample(file_path):
    """
    检查并绘制风数据样本
    
    参数:
        file_path: 风数据文件路径（.UAN格式）
    
    功能:
        1. 加载原始风数据（不修改）
        2. 绘制三张图：风速、风向、风攻角
        3. 记录数据点数量和基本统计信息
    """
    print("="*80)
    print("风数据样本检查工具")
    print("="*80)
    
    # 检查文件存在性
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n✓ 文件路径: {file_path}")
    print(f"✓ 文件名: {os.path.basename(file_path)}")
    
    # 加载数据
    print("\n[步骤1] 加载原始风数据...")
    try:
        unpacker = UNPACK(init_path=False)
        wind_velocity, wind_direction, wind_attack_angle = unpacker.Wind_Data_Unpack(file_path)
        
        # 转换为numpy数组
        wind_velocity = np.array(wind_velocity)
        wind_direction = np.array(wind_direction)
        wind_attack_angle = np.array(wind_attack_angle)
        
        print(f"✓ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 数据统计
    print("\n[步骤2] 数据统计...")
    n_samples = len(wind_velocity)
    print(f"✓ 数据点总数: {n_samples}")
    print(f"\n风速统计:")
    print(f"  - 最小值: {wind_velocity.min():.2f} m/s")
    print(f"  - 最大值: {wind_velocity.max():.2f} m/s")
    print(f"  - 平均值: {wind_velocity.mean():.2f} m/s")
    print(f"  - 标准差: {wind_velocity.std():.2f} m/s")
    
    print(f"\n风向统计:")
    print(f"  - 最小值: {wind_direction.min():.2f} °")
    print(f"  - 最大值: {wind_direction.max():.2f} °")
    print(f"  - 平均值: {wind_direction.mean():.2f} °")
    print(f"  - 标准差: {wind_direction.std():.2f} °")
    
    print(f"\n风攻角统计:")
    print(f"  - 最小值: {wind_attack_angle.min():.2f} °")
    print(f"  - 最大值: {wind_attack_angle.max():.2f} °")
    print(f"  - 平均值: {wind_attack_angle.mean():.2f} °")
    print(f"  - 标准差: {wind_attack_angle.std():.2f} °")
    
    # 绘制图表
    print("\n[步骤3] 绘制图表...")
    
    # 时间轴（假设采样频率为1Hz）
    time_axis = np.arange(n_samples)
    
    # 图1：风速时程曲线
    print("  绘制风速时程曲线...")
    fig1 = plt.figure(figsize=REC_FIG_SIZE)
    ax1 = fig1.add_subplot(111)
    ax1.plot(time_axis, wind_velocity, color='#2E86AB', linewidth=1.0, alpha=0.8)
    ax1.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax1.set_ylabel('风速 (m/s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax1.set_title(f'风速时程曲线 ({n_samples} 个数据点)', 
                  fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax1.set_xlim(0, n_samples)
    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.tight_layout()
    
    # 图2：风向时程曲线
    print("  绘制风向时程曲线...")
    fig2 = plt.figure(figsize=REC_FIG_SIZE)
    ax2 = fig2.add_subplot(111)
    ax2.plot(time_axis, wind_direction, color='#A23B72', linewidth=1.0, alpha=0.8)
    ax2.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax2.set_ylabel('风向 (°)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax2.set_title(f'风向时程曲线 ({n_samples} 个数据点)', 
                  fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax2.set_xlim(0, n_samples)
    ax2.set_ylim(-10, 370)
    ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.tight_layout()
    
    # 图3：风攻角时程曲线
    print("  绘制风攻角时程曲线...")
    fig3 = plt.figure(figsize=REC_FIG_SIZE)
    ax3 = fig3.add_subplot(111)
    ax3.plot(time_axis, wind_attack_angle, color='#F18F01', linewidth=1.0, alpha=0.8)
    ax3.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax3.set_ylabel('风攻角 (°)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax3.set_title(f'风攻角时程曲线 ({n_samples} 个数据点)', 
                  fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax3.set_xlim(0, n_samples)
    ax3.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.tight_layout()
    
    # 使用 PlotLib 展示图表
    print("\n[步骤4] 准备展示图表...")
    ploter = PlotLib()
    ploter.figs.append(fig1)
    ploter.figs.append(fig2)
    ploter.figs.append(fig3)
    
    print(f"✓ 已准备 {len(ploter.figs)} 张图表")
    print("\n" + "="*80)
    print("图表展示中...")
    print("="*80)
    
    plt.close('all')
    ploter.show()


if __name__ == "__main__":
    # 硬编码的风数据文件路径
    # 请根据实际情况修改此路径

    check_wind_sample(WIND_FILE_PATH)
