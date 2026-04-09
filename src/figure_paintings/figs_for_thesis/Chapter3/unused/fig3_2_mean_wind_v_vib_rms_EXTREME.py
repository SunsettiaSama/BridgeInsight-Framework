"""
图2.7: 平均风速与振动RMS的关系分析
绘制平均风速（横坐标）与振动加速度RMS（纵坐标）的散点图
显示面内和面外两个方向的传感器数据

研究目的：
- 对比同一根拉索在面内和面外两个方向的风-振动响应特性
- 分析风速与不同方向振动的相关性
- 揭示拉索运动的方向特异性

数据处理流程：
1. 从工作流获取极端振动对应的风数据和振动文件路径
2. 使用滑动窗口计算振动RMS
3. 使用滑动窗口计算风速平均值
4. 绘制散点图，显示面内/面外两种方向的数据对比
"""

# --------------- 模块导入 ---------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据处理工作流
from src.data_processer.preprocess.workflow import get_data_pairs

# 导入可视化工具
from src.visualize_tools.utils import PlotLib

# 导入传感器配置
from src.config.sensor_config import (
    WIND_SENSOR_NAMES,
    WIND_FS,
    WIND_VALID_THRESHOLD,
    VIB_TO_WIND_SENSOR_MAP,
)

# 导入绘图配置
from .config import ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE


# --------------- 全局配置 ---------------
plt.style.use('default')
plt.rcParams['font.size'] = FONT_SIZE

# 选定的传感器组合
SELECTED_VIB_SENSORS = {
    'edge_span_inplane': 'ST-VIC-C34-101-01',      # 北索塔边跨最远
    'edge_span_outplane': 'ST-VIC-C34-101-02',     # 北索塔边跨最远
    'mid_span_inplane': 'ST-VIC-C34-201-01',       # 北索塔跨中最远
    'mid_span_outplane': 'ST-VIC-C34-201-02',      # 北索塔跨中最远
}

ENABLE_MULTI_PROCESS = True
ENABLE_EXTREME_WINDOW = True

# 风传感器配置
# 注意：这里需要根据实际配置修改风传感器ID
WIND_SENSOR_ID = 'ST-UAN-G04-001-01'
DURATION_MINUTES = 1

# --------------- 工具函数 ---------------
def calculate_mean(sequence):
    """
    计算序列的平均值
    
    参数:
        sequence: 可迭代序列
    
    返回:
        float: 序列的平均值
    """
    sequence = np.asarray(sequence)
    return float(np.mean(sequence))


def calculate_rms(sequence):
    """
    计算序列的均方根（Root Mean Square）
    
    参数:
        sequence: 可迭代序列
    
    返回:
        float: 序列的均方根
    """
    sequence = np.asarray(sequence)
    return float(np.sqrt(np.mean(np.square(sequence))))


def _process_single_data_pair(data_pair):
    """
    处理单个数据对（用于多进程处理）
    
    参数:
        data_pair: get_data_pairs返回的单个Dict项
                   - segmented_windows可能是：
                     1. 列表 [(vib_segment, (wind_speed, wind_direction, wind_angle)), ...]
                     2. 元组 (vib_data, (wind_speed, wind_direction, wind_angle))（原始数据未切分）
                     3. None或空列表（无有效数据）
        
    返回:
        list: [(mean_wind, rms_vib), ...] 格式的数据列表
    """
    try:
        segmented_windows = data_pair.get('segmented_windows')
        if not segmented_windows:
            return []
        
        results = []
        
        # 兼容两种格式：列表（切分后）或元组（原始数据）
        if isinstance(segmented_windows, list):
            # 列表格式：已切分的数据对列表
            windows_to_process = segmented_windows
        elif isinstance(segmented_windows, tuple) and len(segmented_windows) == 2:
            # 元组格式：原始数据（vib_data, (wind_speed, wind_direction, wind_angle))
            windows_to_process = [segmented_windows]
        else:
            # 其他格式：无法处理
            return []
        
        for vib_segment, wind_segment in windows_to_process:
            wind_speed, wind_direction, wind_angle = wind_segment
            
            mean_wind = calculate_mean(wind_speed)
            rms_vib = calculate_rms(vib_segment)
            
            results.append((mean_wind, rms_vib))
        
        return results
    except Exception as e:
        print(f"处理数据对时出错: {e}")
        return []


def process_data_pairs(data_pairs, use_multiprocess=ENABLE_MULTI_PROCESS):
    """
    处理get_data_pairs返回的结果，计算平均风速和振动RMS
    
    参数:
        data_pairs: get_data_pairs返回的数据对列表
        use_multiprocess: 是否使用多进程处理
    
    返回:
        tuple: (wind_speeds, vib_rms_values)
            - wind_speeds: 平均风速列表
            - vib_rms_values: 对应的振动RMS列表
    """
    print("\n[处理阶段] 计算风速和振动RMS...")
    print(f"  数据对总数: {len(data_pairs)}")
    print(f"  多进程模式: {'启用' if use_multiprocess else '禁用'}")
    
    all_results = []
    
    if use_multiprocess:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_single_data_pair, dp) for dp in data_pairs]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
                result = future.result()
                all_results.extend(result)
    else:
        for data_pair in tqdm(data_pairs, desc="处理进度"):
            result = _process_single_data_pair(data_pair)
            all_results.extend(result)
    
    if not all_results:
        print("  警告: 未获得有效数据")
        return [], []
    
    wind_speeds = [item[0] for item in all_results]
    vib_rms_values = [item[1] for item in all_results]
    
    print(f"  ✓ 处理完成: {len(all_results)} 个窗口数据点")
    
    return wind_speeds, vib_rms_values


def plot_wind_vs_vibration(wind_speeds, vib_inplane, vib_outplane, 
                           location_name='', output_path=None):
    """
    绘制风速与振动RMS的关系图
    
    参数:
        wind_speeds: 风速数据
        vib_inplane: 面内振动RMS
        vib_outplane: 面外振动RMS
        location_name: 位置名称（用于标题）
        output_path: 输出文件路径（可选）
    
    返回:
        fig, ax: matplotlib figure和axis对象
    """
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    
    # 确保数据长度一致
    min_len = min(len(wind_speeds), len(vib_inplane), len(vib_outplane))
    wind_speeds = wind_speeds[:min_len]
    vib_inplane = vib_inplane[:min_len]
    vib_outplane = vib_outplane[:min_len]
    
    # 绘制散点图
    ax.scatter(wind_speeds, vib_inplane, 
              color='#1f77b4', s=80, alpha=0.6, 
              edgecolors='#1f77b4', linewidth=1.5,
              label='面内方向', marker='o')
    
    ax.scatter(wind_speeds, vib_outplane, 
              color='#ff7f0e', s=80, alpha=0.6, 
              edgecolors='#ff7f0e', linewidth=1.5,
              label='面外方向', marker='s')
    
    # 设置标签
    ax.set_xlabel('平均风速 （m/s）', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('加速度均方根 （m/s²）', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    
    # 设置标题
    # title = f'风速与振动RMS关系 - {location_name}\n(Wind Speed vs Vibration RMS - {location_name})'
    # ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=20)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置图例
    ax.legend(loc='upper left', fontsize=FONT_SIZE-2, framealpha=0.95,
             prop=CN_FONT if location_name else ENG_FONT)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（可选）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存至: {output_path}")
    
    return fig, ax


def main_interface(wind_sensor_id, vib_inplane_id, vib_outplane_id,
                   enable_extreme_window=False, window_duration_minutes=None,
                   use_multiprocess=ENABLE_MULTI_PROCESS, use_data_multiprocess=True,
                   output_path=None):
    """
    主接口：获取并处理面内和面外数据，返回绘图所需的数据
    
    参数:
        wind_sensor_id: 风传感器ID
        vib_inplane_id: 面内振动传感器ID
        vib_outplane_id: 面外振动传感器ID
        enable_extreme_window: 是否启用极端窗口筛选
        window_duration_minutes: 窗口时长（分钟）
        use_multiprocess: 获取数据时是否使用多进程
        use_data_multiprocess: 处理数据时是否使用多进程
        output_path: 输出图表路径（可选）
    
    返回:
        tuple: (fig, ax)
            - fig: matplotlib figure对象
            - ax: matplotlib axis对象
    """
    print("\n" + "="*80)
    print("主接口：获取、处理并绘制风速与振动数据")
    print("="*80)
    print(f"  风传感器ID: {wind_sensor_id}")
    print(f"  面内振动传感器ID: {vib_inplane_id}")
    print(f"  面外振动传感器ID: {vib_outplane_id}")
    print(f"  极端窗口模式: {enable_extreme_window}")
    print(f"  窗口时长: {window_duration_minutes} 分钟" if window_duration_minutes else "  窗口时长: 不切分")
    
    # ===== 处理面内数据 =====
    print("\n[面内数据处理]")
    print("-"*80)
    
    print("\n[获取阶段] 调用get_data_pairs获取面内数据...")
    inplane_data_pairs = get_data_pairs(
        wind_sensor_id=wind_sensor_id,
        vib_sensor_id=vib_inplane_id,
        use_multiprocess=use_multiprocess,
        enable_extreme_window=enable_extreme_window,
        window_duration_minutes=window_duration_minutes
    )
    
    if not inplane_data_pairs:
        print("  错误: 面内数据获取失败")
        return None, None
    
    print(f"  ✓ 获取成功: {len(inplane_data_pairs)} 个数据对")
    
    print("\n[处理阶段] 处理面内数据...")
    wind_speeds_inplane, vib_inplane = process_data_pairs(inplane_data_pairs, use_multiprocess=use_data_multiprocess)
    
    if not wind_speeds_inplane:
        print("  错误: 面内数据处理失败")
        return None, None
    
    # ===== 处理面外数据 =====
    print("\n[面外数据处理]")
    print("-"*80)
    
    print("\n[获取阶段] 调用get_data_pairs获取面外数据...")
    outplane_data_pairs = get_data_pairs(
        wind_sensor_id=wind_sensor_id,
        vib_sensor_id=vib_outplane_id,
        use_multiprocess=use_multiprocess,
        enable_extreme_window=enable_extreme_window,
        window_duration_minutes=window_duration_minutes
    )
    
    if not outplane_data_pairs:
        print("  错误: 面外数据获取失败")
        return None, None
    
    print(f"  ✓ 获取成功: {len(outplane_data_pairs)} 个数据对")
    
    print("\n[处理阶段] 处理面外数据...")
    wind_speeds_outplane, vib_outplane = process_data_pairs(outplane_data_pairs, use_multiprocess=use_data_multiprocess)
    
    if not wind_speeds_outplane:
        print("  错误: 面外数据处理失败")
        return None, None
    
    # ===== 绘制图表 =====
    print("\n[绘图阶段]")
    print("-"*80)
    print("\n调用plot_wind_vs_vibration绘制图表...")
    
    location_name = vib_inplane_id
    fig, ax = plot_wind_vs_vibration(
        wind_speeds_inplane, vib_inplane, vib_outplane,
        location_name=location_name,
        output_path=output_path
    )
    
    print("\n" + "="*80)
    print("处理完成")
    print("="*80 + "\n")
    
    return fig, ax


# --------------- 主要外部调用接口 ---------------
def generate_figures(wind_sensor_id=WIND_SENSOR_ID,
                     enable_extreme_window=ENABLE_EXTREME_WINDOW,
                     window_duration_minutes=1,
                     use_multiprocess=ENABLE_MULTI_PROCESS,
                     use_data_multiprocess=True,
                     show_plots=True):
    """
    主要外部调用接口：生成风速与振动RMS关系的所有图表
    
    参数:
        wind_sensor_id: 风传感器ID
        enable_extreme_window: 是否启用极端窗口筛选
        window_duration_minutes: 窗口时长（分钟）
        use_multiprocess: 获取数据时是否使用多进程
        use_data_multiprocess: 处理数据时是否使用多进程
        show_plots: 是否显示图表（通过PlotLib GUI）
    
    返回:
        PlotLib: 包含所有生成图表的PlotLib实例
    """
    print("\n" + "="*80)
    print("主程序：生成风速与振动RMS关系图表")
    print("="*80)
    
    # 创建PlotLib实例
    plot_lib = PlotLib()
    
    # 遍历SELECTED_VIB_SENSORS中的传感器组合
    for location_key, (inplane_sensor, outplane_sensor) in [
        ('edge_span', (SELECTED_VIB_SENSORS['edge_span_inplane'], 
                       SELECTED_VIB_SENSORS['edge_span_outplane'])),
        ('mid_span', (SELECTED_VIB_SENSORS['mid_span_inplane'], 
                      SELECTED_VIB_SENSORS['mid_span_outplane']))
    ]:
        print(f"\n{'='*80}")
        print(f"处理位置: {location_key}")
        print(f"{'='*80}")
        
        # 调用主接口处理数据并绘图
        fig, ax = main_interface(
            wind_sensor_id=wind_sensor_id,
            vib_inplane_id=inplane_sensor,
            vib_outplane_id=outplane_sensor,
            enable_extreme_window=enable_extreme_window,
            window_duration_minutes=window_duration_minutes,
            use_multiprocess=use_multiprocess,
            use_data_multiprocess=use_data_multiprocess,
            output_path=None
        )
        
        if fig is not None and ax is not None:
            # 将图表添加到PlotLib
            plot_lib.figs.append(fig)
            print(f"✓ {location_key} 图表已添加到PlotLib")
        else:
            print(f"✗ {location_key} 处理失败，跳过")
    
    print("\n" + "="*80)
    print(f"总共生成了 {len(plot_lib.figs)} 张图表")
    print("="*80 + "\n")
    
    # 显示图表
    if show_plots:
        print("启动PlotLib GUI查看图表...\n")
        plot_lib.show()
    
    return plot_lib


# --------------- 主程序入口 ---------------
if __name__ == '__main__':
    # 生成图表
    plot_lib = generate_figures(
        wind_sensor_id=WIND_SENSOR_ID,
        enable_extreme_window=ENABLE_EXTREME_WINDOW,
        window_duration_minutes=DURATION_MINUTES,
        use_multiprocess=ENABLE_MULTI_PROCESS,
        use_data_multiprocess=True,
        show_plots=True
    )

