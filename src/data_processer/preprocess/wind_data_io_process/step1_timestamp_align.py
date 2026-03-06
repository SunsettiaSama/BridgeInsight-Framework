import os
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入路径解析工具
from src.data_processer.io_unpacker import parse_path_str


def extract_timestamp_from_path(file_path):
    """
    从文件路径中提取时间戳信息
    
    参数:
        file_path: 文件路径
    
    返回:
        timestamp_tuple: (month, day, hour) 元组，如果解析失败返回 None
    """
    parsed = parse_path_str(file_path)
    if parsed and 'month' in parsed and 'day' in parsed and 'hour' in parsed:
        return (parsed['month'], parsed['day'], parsed['hour'])
    return None


def run_timestamp_align(wind_file_paths, vib_metadata, logger=None):
    """
    根据振动数据元数据的时间戳，筛选对应的风数据文件
    
    参数:
        wind_file_paths: 所有风数据文件路径列表
        vib_metadata: 振动数据元数据列表，每项包含 month, day, hour 等字段
        logger: 可选的日志记录器
    
    返回:
        aligned_paths: 对齐后的风数据文件路径列表
        statistics: 统计信息字典
            - 'vib_timestamps': 振动数据的时间戳集合
            - 'matched_count': 匹配成功的数量
            - 'total_wind_files': 总风数据文件数
    """
    def log_message(msg):
        """统一的日志输出函数"""
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    log_message(f"开始时间戳对齐筛选...")
    log_message(f"振动数据元数据数量: {len(vib_metadata)}")
    log_message(f"风数据文件总数: {len(wind_file_paths)}")
    
    # 从振动元数据中提取所有时间戳
    vib_timestamps = set()
    for item in vib_metadata:
        month = item.get('month')
        day = item.get('day')
        hour = item.get('hour')
        if month and day and hour:
            vib_timestamps.add((month, day, hour))
    
    log_message(f"振动数据时间戳数量: {len(vib_timestamps)}")
    
    # 筛选匹配的风数据文件
    aligned_paths = []
    matched_timestamps = set()
    
    for wind_path in wind_file_paths:
        wind_timestamp = extract_timestamp_from_path(wind_path)
        if wind_timestamp and wind_timestamp in vib_timestamps:
            aligned_paths.append(wind_path)
            matched_timestamps.add(wind_timestamp)
    
    # 打印详细统计信息
    log_message("\n" + "="*70)
    log_message("                    时间戳对齐筛选报告")
    log_message("="*70)
    log_message(f"1. 输入统计：")
    log_message(f"   - 振动数据元数据数: {len(vib_metadata)}")
    log_message(f"   - 振动数据时间戳数: {len(vib_timestamps)}")
    log_message(f"   - 风数据文件总数: {len(wind_file_paths)}")
    log_message(f"\n2. 匹配结果：")
    log_message(f"   - 匹配成功的风数据文件: {len(aligned_paths)}")
    log_message(f"   - 匹配成功的时间戳数: {len(matched_timestamps)}")
    log_message(f"   - 匹配率: {len(aligned_paths)/len(wind_file_paths)*100:.2f}%" if len(wind_file_paths) > 0 else "   - 匹配率: 0%")
    log_message(f"\n3. 时间戳覆盖率：")
    log_message(f"   - 有对应风数据的时间戳: {len(matched_timestamps)} / {len(vib_timestamps)}")
    coverage = len(matched_timestamps) / len(vib_timestamps) * 100 if len(vib_timestamps) > 0 else 0
    log_message(f"   - 覆盖率: {coverage:.2f}%")
    
    # 未匹配的时间戳
    unmatched_timestamps = vib_timestamps - matched_timestamps
    if len(unmatched_timestamps) > 0 and len(unmatched_timestamps) <= 10:
        log_message(f"\n4. 未匹配的时间戳（示例）：")
        for i, ts in enumerate(list(unmatched_timestamps)[:10], 1):
            log_message(f"   {i}. 月:{ts[0]} 日:{ts[1]} 时:{ts[2]}")
    elif len(unmatched_timestamps) > 10:
        log_message(f"\n4. 未匹配的时间戳: {len(unmatched_timestamps)} 个")
    
    log_message("="*70 + "\n")
    
    log_message(f"✓ 时间戳对齐完成")
    log_message(f"✓ 保留 {len(aligned_paths)} 个风数据文件\n")
    
    # 组装统计信息
    statistics = {
        'vib_timestamps': vib_timestamps,
        'matched_timestamps': matched_timestamps,
        'matched_count': len(aligned_paths),
        'total_wind_files': len(wind_file_paths),
        'coverage': coverage
    }
    
    return aligned_paths, statistics


if __name__ == "__main__":
    # 测试接口
    from step0_get_wind_data import get_all_wind_files
    
    # 模拟振动数据元数据
    mock_vib_metadata = [
        {'sensor_id': 'ST-VIC-C34-101-01', 'month': '09', 'day': '01', 'hour': '12'},
        {'sensor_id': 'ST-VIC-C34-101-01', 'month': '09', 'day': '01', 'hour': '13'},
        {'sensor_id': 'ST-VIC-C34-101-01', 'month': '09', 'day': '02', 'hour': '10'},
    ]
    
    print("步骤0：获取所有风数据文件...")
    all_wind_files = get_all_wind_files()
    
    print(f"\n步骤1：执行时间戳对齐筛选...")
    aligned_paths, stats = run_timestamp_align(all_wind_files, mock_vib_metadata)
    
    print(f"\n✓ 对齐完成！")
    print(f"✓ 从 {stats['total_wind_files']} 个文件中筛选出 {stats['matched_count']} 个匹配的文件")
    
    # 显示前几个匹配的文件
    if aligned_paths:
        print("\n匹配的风数据文件示例（前3个）：")
        for i, fp in enumerate(aligned_paths[:3], 1):
            print(f"{i}. {fp}")
