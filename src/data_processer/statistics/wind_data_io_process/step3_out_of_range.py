"""
Step 3: 查询越界点数据

基于 Step 2 的极端振动时间范围，查询具体的越界风速/风向点
"""

import os
import sys
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, Manager
from tqdm import tqdm

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从配置文件导入常量
from src.config.data_processer.statistics.wind_data_io_process.config import (
    FS as WIND_FS,
    OUT_OF_RANGE_VELOCITY_MIN,
    OUT_OF_RANGE_VELOCITY_MAX,
    OUT_OF_RANGE_DIRECTION_MIN,
    OUT_OF_RANGE_DIRECTION_MAX,
    OUT_OF_RANGE_ATTACK_ANGLE_MIN,
    OUT_OF_RANGE_ATTACK_ANGLE_MAX,
)
from src.config.sensor_config import WIND_VALID_THRESHOLD
from src.data_processer.io_unpacker import UNPACK


def check_out_of_range(data, min_val, max_val):
    """
    检查数据是否超出指定范围
    
    参数:
        data: 数据数组
        min_val: 最小值（包含）
        max_val: 最大值（包含）
    
    返回:
        mask: 布尔掩码，True 表示越界点
    """
    return (data < min_val) | (data > max_val)


def extract_out_of_range_indices(data, min_val, max_val):
    """
    提取越界点的索引位置
    
    参数:
        data: 数据数组
        min_val: 最小值
        max_val: 最大值
    
    返回:
        indices: 越界点的索引数组
        out_of_range_values: 越界点的值数组
    """
    out_of_range_mask = check_out_of_range(data, min_val, max_val)
    indices = np.where(out_of_range_mask)[0]
    out_of_range_values = data[out_of_range_mask]
    return indices, out_of_range_values


def load_wind_data(file_path):
    """
    加载原始风数据
    
    参数:
        file_path: 风数据文件路径
    
    返回:
        wind_velocity: 风速数组
        wind_direction: 风向数组
        wind_attack_angle: 风攻角数组
    """
    unpacker = UNPACK(init_path=False)
    wind_velocity, wind_direction, wind_attack_angle = unpacker.Wind_Data_Unpack(file_path)
    return (np.array(wind_velocity), np.array(wind_direction), np.array(wind_attack_angle))


def _process_metadata_item(args):
    """
    处理单个元数据项的辅助函数，用于多进程
    
    参数:
        args: 包含 (metadata_item, min_vel, max_vel, min_dir, max_dir) 的元组
    
    返回:
        处理结果元组
    """
    metadata_item, min_vel, max_vel, min_dir, max_dir = args
    
    file_path = metadata_item.get('file_path')
    sensor_id = metadata_item.get('sensor_id')
    extreme_time_ranges = metadata_item.get('extreme_time_ranges', [])
    
    if not file_path or len(extreme_time_ranges) == 0:
        return None, {}, {}
    
    try:
        wind_velocity, wind_direction, wind_attack_angle = load_wind_data(file_path)
    except:
        return None, {}, {}
    
    if wind_velocity is None:
        return None, {}, {}
    
    out_of_range_windows = []
    vel_count = 0
    dir_count = 0
    ang_count = 0
    
    for window_idx, (start_sec, end_sec) in enumerate(extreme_time_ranges):
        start_idx = int(start_sec * WIND_FS)
        end_idx = int(end_sec * WIND_FS)
        
        start_idx = max(0, start_idx)
        end_idx = min(len(wind_velocity), end_idx)
        
        if start_idx < end_idx:
            window_vel = wind_velocity[start_idx:end_idx]
            window_dir = wind_direction[start_idx:end_idx]
            window_ang = wind_attack_angle[start_idx:end_idx]
            
            vel_indices, vel_values = extract_out_of_range_indices(window_vel, min_vel, max_vel)
            dir_indices, dir_values = extract_out_of_range_indices(window_dir, min_dir, max_dir)
            ang_indices, ang_values = extract_out_of_range_indices(window_ang, OUT_OF_RANGE_ATTACK_ANGLE_MIN, OUT_OF_RANGE_ATTACK_ANGLE_MAX)
            
            window_info = {
                'window_idx': window_idx,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'vel_out_of_range': {
                    'count': len(vel_indices),
                    'indices': vel_indices.tolist(),
                    'values': vel_values.tolist(),
                    'absolute_indices': (vel_indices + start_idx).tolist()
                },
                'dir_out_of_range': {
                    'count': len(dir_indices),
                    'indices': dir_indices.tolist(),
                    'values': dir_values.tolist(),
                    'absolute_indices': (dir_indices + start_idx).tolist()
                },
                'ang_out_of_range': {
                    'count': len(ang_indices),
                    'indices': ang_indices.tolist(),
                    'values': ang_values.tolist(),
                    'absolute_indices': (ang_indices + start_idx).tolist()
                }
            }
            
            out_of_range_windows.append(window_info)
            vel_count += len(vel_indices)
            dir_count += len(dir_indices)
            ang_count += len(ang_indices)
    
    if len(out_of_range_windows) > 0:
        new_metadata = metadata_item.copy()
        new_metadata['out_of_range_windows'] = out_of_range_windows
        
        sensor_counts = {
            'vel': {sensor_id: vel_count},
            'dir': {sensor_id: dir_count},
            'ang': {sensor_id: ang_count}
        }
        
        return new_metadata, {'vel': vel_count, 'dir': dir_count, 'ang': ang_count}, sensor_counts
    
    return None, {}, {}


def run_out_of_range_query(filtered_metadata, min_vel=OUT_OF_RANGE_VELOCITY_MIN, max_vel=OUT_OF_RANGE_VELOCITY_MAX, min_dir=OUT_OF_RANGE_DIRECTION_MIN, max_dir=OUT_OF_RANGE_DIRECTION_MAX, logger=None, num_processes=None):
    """
    根据 Step 2 的筛选结果，查询越界风速/风向点的具体信息
    
    参数:
        filtered_metadata: Step 2 的输出，包含 extreme_time_ranges 的元数据列表
        min_vel: 风速下界 (m/s)
        max_vel: 风速上界 (m/s)
        min_dir: 风向下界 (°)
        max_dir: 风向上界 (°)
        logger: 可选的日志记录器
        num_processes: 进程数，默认为 None（使用 CPU 核心数）
    
    返回:
        out_of_range_metadata: 添加越界点信息的元数据列表
        statistics: 统计信息字典
    """
    def log_message(msg):
        """统一的日志输出函数"""
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    log_message(f"开始查询越界点数据...")
    log_message(f"风速范围: [{min_vel}, {max_vel}] m/s")
    log_message(f"风向范围: [{min_dir}, {max_dir}] °")
    log_message(f"输入元数据数量: {len(filtered_metadata)}")
    
    # 准备多进程输入数据
    process_args = [
        (item, min_vel, max_vel, min_dir, max_dir) 
        for item in filtered_metadata
    ]
    
    # 使用多进程处理数据
    out_of_range_metadata = []
    total_out_of_range_vel = 0
    total_out_of_range_dir = 0
    total_out_of_range_ang = 0
    sensor_vel_counts = defaultdict(int)
    sensor_dir_counts = defaultdict(int)
    sensor_ang_counts = defaultdict(int)
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_metadata_item, process_args),
            total=len(process_args),
            desc="处理元数据项",
            unit="项"
        ))
    
    for metadata_result, counts, sensor_counts in results:
        if metadata_result is not None:
            out_of_range_metadata.append(metadata_result)
            if counts:
                total_out_of_range_vel += counts.get('vel', 0)
                total_out_of_range_dir += counts.get('dir', 0)
                total_out_of_range_ang += counts.get('ang', 0)
            
            if sensor_counts:
                for sensor_id, vel_count in sensor_counts.get('vel', {}).items():
                    sensor_vel_counts[sensor_id] += vel_count
                for sensor_id, dir_count in sensor_counts.get('dir', {}).items():
                    sensor_dir_counts[sensor_id] += dir_count
                for sensor_id, ang_count in sensor_counts.get('ang', {}).items():
                    sensor_ang_counts[sensor_id] += ang_count
    
    # 打印详细统计信息
    log_message("\n" + "="*70)
    log_message("                越界点查询报告")
    log_message("="*70)
    log_message(f"1. 输入统计：")
    log_message(f"   - 总元数据项数: {len(filtered_metadata)}")
    log_message(f"   - 有越界点的元数据项: {len(out_of_range_metadata)}")
    
    log_message(f"\n2. 越界点统计：")
    log_message(f"   - 总风速越界点数: {total_out_of_range_vel}")
    log_message(f"   - 总风向越界点数: {total_out_of_range_dir}")
    log_message(f"   - 总风攻角越界点数: {total_out_of_range_ang}")
    
    if len(sensor_vel_counts) > 0:
        log_message(f"\n3. 各传感器风速越界统计：")
        for sensor_id in sorted(sensor_vel_counts.keys()):
            log_message(f"   - {sensor_id}: {sensor_vel_counts[sensor_id]} 个点")
    
    if len(sensor_dir_counts) > 0:
        log_message(f"\n4. 各传感器风向越界统计：")
        for sensor_id in sorted(sensor_dir_counts.keys()):
            log_message(f"   - {sensor_id}: {sensor_dir_counts[sensor_id]} 个点")
    
    if len(sensor_ang_counts) > 0:
        log_message(f"\n5. 各传感器风攻角越界统计：")
        for sensor_id in sorted(sensor_ang_counts.keys()):
            log_message(f"   - {sensor_id}: {sensor_ang_counts[sensor_id]} 个点")
    
    log_message("="*70 + "\n")
    
    log_message(f"✓ 越界点查询完成")
    log_message(f"✓ 查询到 {len(out_of_range_metadata)} 个包含越界点的数据文件\n")
    
    # 组装统计信息
    statistics = {
        'total_out_of_range_vel': total_out_of_range_vel,
        'total_out_of_range_dir': total_out_of_range_dir,
        'total_out_of_range_ang': total_out_of_range_ang,
        'items_with_out_of_range': len(out_of_range_metadata),
        'sensor_vel_counts': dict(sensor_vel_counts),
        'sensor_dir_counts': dict(sensor_dir_counts),
        'sensor_ang_counts': dict(sensor_ang_counts)
    }
    
    return out_of_range_metadata, statistics


if __name__ == "__main__":
    # 测试接口
    from src.data_processer.statistics.wind_data_io_process.step0_get_wind_data import get_all_wind_files
    from src.data_processer.statistics.wind_data_io_process.step1_timestamp_align import run_timestamp_align
    from src.data_processer.statistics.wind_data_io_process.step2_extreme_filter import run_extreme_filter
    from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow
    from src.data_processer.io_unpacker import parse_path_metadata
    
    print("="*80)
    print(" "*20 + "Step 3: 越界点查询测试")
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
    print(f"✓ 筛选出 {filter_stats['total_extreme_samples']} 个包含极端振动的风数据文件")
    
    # 4. 运行越界点查询
    print("\n[Step 3] 执行越界点查询...")
    print("-"*80)
    out_of_range_metadata, out_of_range_stats = run_out_of_range_query(
        filtered_metadata,
        min_vel=OUT_OF_RANGE_VELOCITY_MIN, max_vel=OUT_OF_RANGE_VELOCITY_MAX,
        min_dir=OUT_OF_RANGE_DIRECTION_MIN, max_dir=OUT_OF_RANGE_DIRECTION_MAX
    )
    
    print(f"\n✓ 查询完成！")
    print(f"✓ 风速越界点总数: {out_of_range_stats['total_out_of_range_vel']}")
    print(f"✓ 风向越界点总数: {out_of_range_stats['total_out_of_range_dir']}")
    
    # 显示示例
    if len(out_of_range_metadata) > 0:
        print("\n[示例] 前3个包含越界点的数据文件：")
        for i, item in enumerate(out_of_range_metadata[:3], 1):
            print(f"\n{i}. 文件: {item.get('file_path', 'N/A')}")
            print(f"   传感器: {item.get('sensor_id', 'N/A')}")
            print(f"   包含越界窗口数: {len(item.get('out_of_range_windows', []))}")
            
            # 显示第一个窗口的越界信息
            if len(item.get('out_of_range_windows', [])) > 0:
                window = item['out_of_range_windows'][0]
                print(f"   第一个窗口 ({window['start_sec']}s - {window['end_sec']}s):")
                print(f"     - 风速越界点: {window['vel_out_of_range']['count']} 个")
                print(f"     - 风向越界点: {window['dir_out_of_range']['count']} 个")
                print(f"     - 风攻角越界点: {window['ang_out_of_range']['count']} 个")
