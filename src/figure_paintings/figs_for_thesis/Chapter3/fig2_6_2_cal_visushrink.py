import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from scipy import stats

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.visualize_tools.utils import PlotLib
from ..config import ENG_FONT, CN_FONT, FONT_SIZE
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow


# ==================== 常量配置 ====================
class Config:
    # 采样频率
    FS = 50.0
    
    # 极端窗口配置
    WINDOW_SIZE = 3000  # 60秒窗口 @ 50Hz
    
    # 数据截取配置
    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 10
    
    # 小波去噪配置
    WAVELET_TYPE = 'coif2'
    WAVELET_LEVEL = 5
    
    # 随机抽样配置
    NUM_SAMPLES_TO_PLOT = 1    # 仅分析一个样本
    NUM_WINDOWS_TO_PLOT = 1
    RANDOM_SEED = 42


# ==================== 数据获取函数 ====================
def get_sample_data():
    """获取一个样本用于分析"""
    print("[获取数据] 运行振动工作流获取元数据...")
    metadata = run_vib_workflow(use_cache=True, force_recompute=False)
    
    print(f"✓ 从工作流获取 {len(metadata)} 条元数据记录")
    
    records_with_extreme = [
        item for item in metadata 
        if len(item.get('extreme_rms_indices', [])) > 0
    ]
    print(f"✓ 其中包含极端窗口的记录：{len(records_with_extreme)} 条")
    
    if not records_with_extreme:
        raise ValueError("无包含极端窗口的记录")
    
    np.random.seed(Config.RANDOM_SEED)
    record = records_with_extreme[np.random.randint(len(records_with_extreme))]
    
    file_path = record['file_path']
    sensor_id = record['sensor_id']
    extreme_indices = record.get('extreme_rms_indices', [])
    
    unpacker = UNPACK(init_path=False)
    vibration_data = unpacker.VIC_DATA_Unpack(file_path)
    vibration_data = np.array(vibration_data)
    
    window_idx = extreme_indices[0]
    start_sample = window_idx * Config.WINDOW_SIZE
    end_sample = (window_idx + 1) * Config.WINDOW_SIZE
    
    window_data = vibration_data[start_sample:end_sample]
    
    # 截取指定时间段
    trim_start_idx = int(Config.TRIM_START_SECOND * Config.FS)
    trim_end_idx = int(Config.TRIM_END_SECOND * Config.FS)
    
    return window_data[trim_start_idx:trim_end_idx], sensor_id


# ==================== 小波分解与阈值分析 ====================
def analyze_wavelet_decomposition(signal_data):
    """
    分析小波分解的各层系数及其对应的阈值
    
    参数：
        signal_data: 原始信号
    
    返回：
        analysis_results: 包含分解信息的字典
    """
    print("\n" + "="*80)
    print("小波分解与阈值分析")
    print("="*80)
    
    # 执行小波分解
    print(f"\n[小波分解] 使用 {Config.WAVELET_TYPE} 小波，分解层数：{Config.WAVELET_LEVEL}")
    wavelet = Config.WAVELET_TYPE
    level = Config.WAVELET_LEVEL
    
    # 多层小波分解
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    # 分离近似系数和细节系数
    cA = coeffs[0]  # 近似系数
    cD_list = coeffs[1:]  # 细节系数列表
    
    print(f"✓ 分解完成")
    print(f"  - 近似系数 cA{level} 长度: {len(cA)}")
    for i, cD in enumerate(cD_list, 1):
        print(f"  - 细节系数 cD{level-i+1} 长度: {len(cD)}")
    
    # 计算每层的阈值
    analysis_data = []
    
    print(f"\n[阈值计算] 使用 sqtwolog 方法")
    print(f"  公式: lambda = sigma * sqrt(2 * log(N))")
    print(f"  其中 sigma 为噪声标准差估计，N 为信号长度\n")
    
    # 对每个细节系数层计算阈值
    for i, cD in enumerate(cD_list):
        level_num = level - i
        length = len(cD)
        
        # 估计噪声标准差（使用最细节系数的中位数绝对偏差法）
        sigma = np.median(np.abs(cD - np.median(cD))) / 0.6745
        
        # 计算 sqtwolog 阈值
        threshold_sqtwolog = sigma * np.sqrt(2 * np.log(length))
        
        # 计算系数的能量（方差）
        energy = np.var(cD)
        
        # 计算硬阈值后的能量抑制率
        thresholded = pywt.threshold(cD, threshold_sqtwolog, mode='hard')
        energy_suppressed = np.var(thresholded)
        suppression_rate = (1 - energy_suppressed / (energy + 1e-10)) * 100
        
        # 统计超过阈值的系数数量
        above_threshold = np.sum(np.abs(cD) > threshold_sqtwolog)
        below_threshold = np.sum(np.abs(cD) <= threshold_sqtwolog)
        
        analysis_data.append({
            '分解层': f'cD{level_num}',
            '系数长度': length,
            '噪声标准差σ': sigma,
            '阈值λ': threshold_sqtwolog,
            'λ/σ比值': threshold_sqtwolog / sigma,
            '系数能量': energy,
            '超阈值系数数': above_threshold,
            '低于阈值系数数': below_threshold,
            '低阈比例(%)': (below_threshold / length) * 100,
            '能量抑制率(%)': suppression_rate
        })
    
    # 近似系数不进行阈值处理
    cA_energy = np.var(cA)
    analysis_data.insert(0, {
        '分解层': f'cA{level}',
        '系数长度': len(cA),
        '噪声标准差σ': '-',
        '阈值λ': '保留',
        'λ/σ比值': '-',
        '系数能量': cA_energy,
        '超阈值系数数': '-',
        '低于阈值系数数': '-',
        '低阈比例(%)': '-',
        '能量抑制率(%)': '-'
    })
    
    return analysis_data, cD_list, cA, signal_data


# ==================== 表格绘制 ====================
def draw_analysis_table(analysis_data, sensor_id):
    """
    绘制分析结果表格
    
    参数：
        analysis_data: 分析结果列表
        sensor_id: 传感器ID
    """
    # 创建 DataFrame
    df = pd.DataFrame(analysis_data)
    
    # 创建图表
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['分解层', '系数长度', 'σ (标准差)', 'λ (阈值)', 'λ/σ', '能量', 
               '超阈系数', '低阈系数', '低阈比(%)', '抑制率(%)']
    
    for idx, row in df.iterrows():
        table_row = [
            row['分解层'],
            f"{int(row['系数长度'])}",
            f"{row['噪声标准差σ']:.4f}" if isinstance(row['噪声标准差σ'], float) else str(row['噪声标准差σ']),
            f"{row['阈值λ']:.4f}" if isinstance(row['阈值λ'], float) else str(row['阈值λ']),
            f"{row['λ/σ比值']:.4f}" if isinstance(row['λ/σ比值'], float) else str(row['λ/σ比值']),
            f"{row['系数能量']:.6f}",
            f"{int(row['超阈值系数数'])}" if isinstance(row['超阈值系数数'], (int, np.integer)) else str(row['超阈值系数数']),
            f"{int(row['低于阈值系数数'])}" if isinstance(row['低于阈值系数数'], (int, np.integer)) else str(row['低于阈值系数数']),
            f"{row['低阈比例(%)']:.2f}" if isinstance(row['低阈比例(%)'], float) else str(row['低阈比例(%)']),
            f"{row['能量抑制率(%)']:.2f}" if isinstance(row['能量抑制率(%)'], float) else str(row['能量抑制率(%)'])
        ]
        table_data.append(table_row)
    
    # 绘制表格
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.08, 0.08, 0.10, 0.10]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontproperties=CN_FONT)
    
    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        if i == 1:  # 近似系数行
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#E8F5E9')
        else:  # 细节系数行
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F5F5F5')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')
    
    fig.suptitle(
        f"{sensor_id} - Visushrink 阈值分析表\n"
        f"小波基: {Config.WAVELET_TYPE}, 分解层数: {Config.WAVELET_LEVEL}",
        fontproperties=CN_FONT, fontsize=FONT_SIZE, fontweight='bold', y=0.98
    )
    
    # 添加说明文本
    explanation = (
        "关键指标说明：\n"
        "• σ: 使用中位数绝对偏差法估计的噪声标准差\n"
        "• λ: Visushrink 阈值 = σ × √(2ln(N))，其中 N 为系数长度\n"
        "• λ/σ: 阈值与标准差的比值，反映了阈值的严格程度\n"
        "• 低阈比(%): 被阈值抑制的系数比例，越高表示高频越被抑制\n"
        "• 抑制率(%): 应用硬阈值后系数能量的下降百分比"
    )
    
    fig.text(0.05, -0.05, explanation, fontproperties=CN_FONT, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    return fig


# ==================== 主函数 ====================
def main():
    """主函数"""
    try:
        print("="*80)
        print("Visushrink 阈值抑制原理分析")
        print("="*80)
        
        print("\n[步骤1] 获取样本数据...")
        signal_data, sensor_id = get_sample_data()
        print(f"✓ 成功获取样本数据（传感器: {sensor_id}, 长度: {len(signal_data)}）")
        
        print("\n[步骤2] 执行小波分解与阈值分析...")
        analysis_data, cD_list, cA, original_signal = analyze_wavelet_decomposition(signal_data)
        
        print("\n[步骤3] 绘制分析表格...")
        ploter = PlotLib()
        fig = draw_analysis_table(analysis_data, sensor_id)
        ploter.figs.append(fig)
        
        print("\n" + "="*80)
        print("✓ 分析完成")
        print("="*80 + "\n")
        
        # 打印总结信息
        print("[分析总结]")
        print(f"  - 小波基: {Config.WAVELET_TYPE}")
        print(f"  - 分解层数: {Config.WAVELET_LEVEL}")
        print(f"  - 信号长度: {len(original_signal)}")
        
        df = pd.DataFrame(analysis_data)
        cd_rows = df[df['分解层'] != f'cA{Config.WAVELET_LEVEL}']
        print(f"\n  高频抑制特性：")
        for idx, row in cd_rows.iterrows():
            print(f"    {row['分解层']}: 低阈比例 {row['低阈比例(%)']:.1f}%, 能量抑制率 {row['能量抑制率(%)']:.1f}%")
        
        print(f"\n  结论：")
        print(f"    sqtwolog 阈值通过 √(2ln(N)) 因子进行缩放")
        print(f"    高频系数（细节系数）长度较小，导致阈值相对较小")
        print(f"    这使得高频成分更容易被阈值抑制，从而实现去噪效果")
        
        ploter.show()
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        raise


if __name__ == "__main__":
    main()
