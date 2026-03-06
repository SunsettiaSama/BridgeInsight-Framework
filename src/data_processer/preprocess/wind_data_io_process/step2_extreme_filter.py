"""
Step 2: 筛选极端振动对应的风数据样本

基于振动数据的 extreme_rms_indices，提取对应时间窗口的风数据样本
"""

import os
import sys
import numpy as np
from collections import defaultdict

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从配置文件导入常量
from src.config.data_processer.preprocess.wind_data_io_process.config import FS as WIND_FS
from src.config.sensor_config import VIB_TO_WIND_SENSOR_MAP, DEFAULT_WIND_SENSORS

# 硬编码参数（与振动数据保持一致）
VIB_FS = 50  # 振动信号采样频率
VIB_TIME_WINDOW = 60.0  # 振动数据计算RMS的时间窗口（秒）
WIND_TIME_WINDOW = 60.0  # 风数据时间窗口（秒）


def build_timestamp_index(metadata_list):
    """
    构建时间戳到元数据的索引映射
    
    参数:
        metadata_list: 元数据列表，每项包含 sensor_id, month, day, hour
    
    返回:
        timestamp_map: {(sensor_id, month, day, hour): metadata_item}
    """
    timestamp_map = {}
    for item in metadata_list:
        key = (
            item.get('sensor_id'),
            item.get('month'),
            item.get('day'),
            item.get('hour')
        )
        timestamp_map[key] = item
    return timestamp_map


def get_wind_sensor_from_vib_sensor(vib_sensor_id):
    """
    根据振动传感器ID推断对应的风传感器ID
    
    从 sensor_config.py 导入映射关系
    
    参数:
        vib_sensor_id: 振动传感器ID
    
    返回:
        wind_sensor_ids: 对应的风传感器ID列表
    """
    # 从配置文件导入的映射关系
    return VIB_TO_WIND_SENSOR_MAP.get(vib_sensor_id, DEFAULT_WIND_SENSORS)


def run_extreme_filter(wind_metadata, vib_metadata, logger=None):
    """
    根据振动数据的极端值索引，筛选对应的风数据样本
    
    参数:
        wind_metadata: 风数据元数据列表，每项包含:
            - sensor_id: 传感器ID
            - month, day, hour: 时间戳
            - file_path: 文件路径
        vib_metadata: 振动数据元数据列表，每项包含:
            - sensor_id: 传感器ID
            - month, day, hour: 时间戳
            - file_path: 文件路径
            - extreme_rms_indices: 极端振动的窗口索引列表
        logger: 可选的日志记录器
    
    返回:
        filtered_metadata: 筛选后的风数据元数据列表，每项额外包含:
            - extreme_time_ranges: 极端振动对应的时间范围列表 [(start_sec, end_sec), ...]
        statistics: 统计信息字典
    """
    def log_message(msg):
        """统一的日志输出函数"""
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    log_message(f"开始极端振动对应的风数据筛选...")
    log_message(f"振动数据元数据数量: {len(vib_metadata)}")
    log_message(f"风数据元数据数量: {len(wind_metadata)}")
    
    # 构建风数据的时间戳索引
    wind_timestamp_map = {}
    for wind_item in wind_metadata:
        key = (
            wind_item.get('sensor_id'),
            wind_item.get('month'),
            wind_item.get('day'),
            wind_item.get('hour')
        )
        wind_timestamp_map[key] = wind_item
    
    # 筛选出包含极端振动的风数据
    filtered_metadata = []
    total_extreme_samples = 0
    total_extreme_time_ranges = 0
    sensor_extreme_counts = defaultdict(int)  # 统计各传感器的极端样本数
    
    for vib_item in vib_metadata:
        # 获取极端振动索引
        extreme_indices = vib_item.get('extreme_rms_indices', [])
        if len(extreme_indices) == 0:
            continue
        
        # 获取振动传感器信息
        vib_sensor_id = vib_item.get('sensor_id')
        month = vib_item.get('month')
        day = vib_item.get('day')
        hour = vib_item.get('hour')
        
        # 推断对应的风传感器
        wind_sensor_ids = get_wind_sensor_from_vib_sensor(vib_sensor_id)
        
        # 查找对应的风数据文件
        for wind_sensor_id in wind_sensor_ids:
            wind_key = (wind_sensor_id, month, day, hour)
            if wind_key in wind_timestamp_map:
                wind_item = wind_timestamp_map[wind_key].copy()
                
                # 计算极端振动对应的时间范围（秒）
                # 振动数据的窗口索引 * 窗口时长 = 时间范围
                extreme_time_ranges = []
                for idx in extreme_indices:
                    start_sec = idx * VIB_TIME_WINDOW
                    end_sec = (idx + 1) * VIB_TIME_WINDOW
                    extreme_time_ranges.append((start_sec, end_sec))
                
                # 添加极端时间范围信息
                wind_item['extreme_time_ranges'] = extreme_time_ranges
                wind_item['vib_sensor_id'] = vib_sensor_id  # 记录对应的振动传感器
                wind_item['extreme_window_count'] = len(extreme_indices)  # 极端窗口数量
                
                filtered_metadata.append(wind_item)
                total_extreme_samples += 1
                total_extreme_time_ranges += len(extreme_indices)
                sensor_extreme_counts[wind_sensor_id] += 1
    
    # 打印详细统计信息
    log_message("\n" + "="*70)
    log_message("                极端振动对应风数据筛选报告")
    log_message("="*70)
    log_message(f"1. 输入统计：")
    log_message(f"   - 振动数据文件数: {len(vib_metadata)}")
    log_message(f"   - 风数据文件数: {len(wind_metadata)}")
    log_message(f"   - 包含极端振动的振动文件数: {sum(1 for v in vib_metadata if len(v.get('extreme_rms_indices', [])) > 0)}")
    log_message(f"\n2. 筛选结果：")
    log_message(f"   - 筛选出的风数据文件数: {total_extreme_samples}")
    log_message(f"   - 总极端时间窗口数: {total_extreme_time_ranges}")
    log_message(f"   - 平均每个文件的极端窗口数: {total_extreme_time_ranges/total_extreme_samples:.2f}" if total_extreme_samples > 0 else "   - 平均每个文件的极端窗口数: 0")
    
    if len(sensor_extreme_counts) > 0:
        log_message(f"\n3. 各风传感器极端样本统计：")
        for sensor_id, count in sorted(sensor_extreme_counts.items()):
            log_message(f"   - {sensor_id}: {count} 个文件")
    
    log_message("="*70 + "\n")
    
    log_message(f"✓ 极端振动对应风数据筛选完成")
    log_message(f"✓ 筛选出 {total_extreme_samples} 个包含极端振动时段的风数据文件\n")
    
    # 组装统计信息
    statistics = {
        'total_extreme_samples': total_extreme_samples,
        'total_extreme_time_ranges': total_extreme_time_ranges,
        'sensor_extreme_counts': dict(sensor_extreme_counts),
        'avg_extreme_windows_per_file': total_extreme_time_ranges / total_extreme_samples if total_extreme_samples > 0 else 0
    }
    
    return filtered_metadata, statistics


if __name__ == "__main__":
    # 测试接口
    from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow
    from step0_get_wind_data import get_all_wind_files
    from step1_timestamp_align import run_timestamp_align
    from src.data_processer.io_unpacker import parse_path_metadata
    
    print("="*80)
    print(" "*20 + "Step 2: 极端振动对应风数据筛选测试")
    print("="*80)
    
    # 1. 运行振动数据工作流
    print("\n[前置步骤] 运行振动数据工作流...")
    print("-"*80)
    vib_metadata = run_vib_workflow(use_cache=True)
    print(f"✓ 获取振动数据元数据: {len(vib_metadata)} 条")
    
    # 2. 获取并对齐风数据
    print("\n[Step 0-1] 获取并对齐风数据...")
    print("-"*80)
    all_wind_files = get_all_wind_files()
    aligned_paths, stats = run_timestamp_align(all_wind_files, vib_metadata)
    wind_metadata = parse_path_metadata(aligned_paths)
    for i, file_path in enumerate(aligned_paths):
        wind_metadata[i]['file_path'] = file_path
    print(f"✓ 获取对齐后的风数据: {len(wind_metadata)} 条")
    
    # 3. 运行极端值筛选
    print("\n[Step 2] 执行极端振动对应风数据筛选...")
    print("-"*80)
    filtered_metadata, filter_stats = run_extreme_filter(wind_metadata, vib_metadata)
    
    print(f"\n✓ 筛选完成！")
    print(f"✓ 筛选出 {filter_stats['total_extreme_samples']} 个包含极端振动的风数据文件")
    print(f"✓ 总极端时间窗口数: {filter_stats['total_extreme_time_ranges']}")
    
    # 显示示例
    if len(filtered_metadata) > 0:
        print("\n[示例] 前3个极端振动对应的风数据文件：")
        for i, item in enumerate(filtered_metadata[:3], 1):
            print(f"\n{i}. 文件: {item.get('file_path', 'N/A')}")
            print(f"   风传感器: {item.get('sensor_id', 'N/A')}")
            print(f"   对应振动传感器: {item.get('vib_sensor_id', 'N/A')}")
            print(f"   时间: {item.get('month')}/{item.get('day')} {item.get('hour')}:00")
            print(f"   极端窗口数: {item.get('extreme_window_count', 0)}")
            print(f"   极端时间范围: {item.get('extreme_time_ranges', [])[:3]}...")  # 只显示前3个
