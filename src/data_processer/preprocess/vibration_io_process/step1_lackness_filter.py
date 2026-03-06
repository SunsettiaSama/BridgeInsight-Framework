import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的解析工具
from src.data_processer.io_unpacker import UNPACK

# 从配置文件导入常量
from src.config.data_processer.preprocess.vibration_io_process.config import (
    MISSING_RATE_THRESHOLD,
    EXPECTED_LENGTH
)

def get_file_length(file_path):
    """获取单个源文件的数据长度"""
    try:
        unpacker = UNPACK(init_path=False)
        vibration_data = unpacker.VIC_DATA_Unpack(file_path)
        return len(vibration_data)
    except Exception:
        return 0

def run_lackness_filter(all_file_paths, threshold=MISSING_RATE_THRESHOLD, expected_length=EXPECTED_LENGTH, logger=None):
    """
    对文件列表进行缺失率筛选
    
    参数:
        all_file_paths: 所有文件路径列表
        threshold: 缺失率阈值（默认 0.05，即 5%）
        expected_length: 预期单文件长度（默认 50Hz * 1小时）
        logger: 可选的日志记录器，如果提供则使用 logger.log() 代替 print()
    
    返回:
        filtered_paths: 筛选后的文件路径列表
        statistics: 统计信息字典
            - 'all_lengths': 所有文件的实际长度数组
            - 'all_missing_rates': 所有文件的缺失率数组
            - 'filtered_indices': 筛选通过的文件索引列表
    """
    def log_message(msg):
        """统一的日志输出函数"""
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    log_message(f"开始对 {len(all_file_paths)} 个文件进行缺失率筛选 (阈值: {threshold*100:.1f}%)...")
    
    # 收集所有文件的实际长度
    all_lengths = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(get_file_length, fp): fp for fp in all_file_paths}
        for future in tqdm(as_completed(futures), total=len(all_file_paths), desc="缺失率计算中"):
            fp = futures[future]
            try:
                actual_len = future.result()
                all_lengths.append(actual_len)
            except Exception as e:
                print(f"处理文件 {fp} 时出错: {e}")
                all_lengths.append(0)
    
    # 计算缺失率
    all_lengths = np.array(all_lengths)
    all_missing_rates = 1.0 - (all_lengths / expected_length)
    
    # 打印详细统计信息
    total_samples = len(all_missing_rates)
    high_missing_samples = np.sum(all_missing_rates > threshold)
    low_missing_samples = np.sum(all_missing_rates <= threshold)
    avg_missing_rate = np.mean(all_missing_rates)
    
    log_message("\n" + "="*70)
    log_message("                    样本缺失率筛选报告")
    log_message("="*70)
    log_message(f"1. 筛选参数：")
    log_message(f"   - 缺失率阈值: {threshold*100:.1f}%")
    log_message(f"   - 预期单文件长度: {expected_length} (50Hz * 60s * 60m)")
    log_message(f"\n2. 处理统计（总计: {total_samples} 个文件）:")
    log_message(f"   - 平均缺失率: {avg_missing_rate*100:.2f}%")
    log_message(f"   - 符合条件的文件 (≤{threshold*100:.1f}%): {low_missing_samples} ({low_missing_samples/total_samples*100:.2f}%)")
    log_message(f"   - 不符合条件的文件 (>{threshold*100:.1f}%): {high_missing_samples} ({high_missing_samples/total_samples*100:.2f}%)")
    log_message("="*70 + "\n")
    
    # 筛选符合条件的文件路径
    filtered_paths = []
    filtered_indices = []
    for i, fp in enumerate(all_file_paths):
        if all_missing_rates[i] <= threshold:
            filtered_paths.append(fp)
            filtered_indices.append(i)
    
    log_message(f"✓ 筛选完成！")
    log_message(f"✓ 符合条件的文件数: {len(filtered_paths)} / {len(all_file_paths)}\n")
    
    # 返回筛选后的路径和统计信息
    statistics = {
        'all_lengths': all_lengths,
        'all_missing_rates': all_missing_rates,
        'filtered_indices': filtered_indices
    }
    
    return filtered_paths, statistics


if __name__ == "__main__":
    # 测试接口
    from step0_get_vib_data import get_all_vibration_files
    
    print("步骤1：获取所有振动文件...")
    all_vib_files = get_all_vibration_files()
    
    print(f"\n步骤2：执行缺失率筛选...")
    filtered_paths, stats = run_lackness_filter(all_vib_files)
    
    print(f"✓ 筛选完成！")
    print(f"✓ 返回 {len(filtered_paths)} 个筛选后的文件路径")
