import numpy as np
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import signal
from tqdm import tqdm
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK

FS = 50.0
SEGMENT_DURATION = 60
FREQ_MIN = 0
FREQ_MAX = 25
NFFT = 1024


def process_single_file_dominant_frequencies(file_path):
    """
    单文件处理工作函数，用于多进程
    在此函数中计算每个时间窗口的主频
    
    参数：
        file_path: 振动数据文件路径
    
    返回：
        list: 该文件中所有时间窗口的主频列表
    
    异常：
        任何处理过程中的异常会自然抛出，由调用端处理
    """
    dominant_frequencies = []
    
    unpacker = UNPACK(init_path=False)
    vibration_data = unpacker.VIC_DATA_Unpack(file_path)
    vibration_data = np.array(vibration_data)
    
    if len(vibration_data) == 0:
        return []
    
    segment_size = int(SEGMENT_DURATION * FS)

    # 只迭代完整 segment，与 RMS 计算保持一致（不含末尾不足一窗口的碎片）
    for i in range(0, len(vibration_data) - segment_size + 1, segment_size):
        segment = vibration_data[i:i + segment_size]

        if len(segment) < 100:
            continue
        
        nfft = NFFT
        if len(segment) < nfft:
            nfft = max(len(segment) // 2, 16)
        
        freqs, psd = signal.welch(
            segment,
            fs=FS,
            nperseg=nfft,
            noverlap=nfft // 2,
            nfft=nfft
        )
        
        mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
        freqs_filtered = freqs[mask]
        psd_filtered = psd[mask]
        
        if len(psd_filtered) > 0:
            dominant_idx = np.argmax(psd_filtered)
            dominant_freq = freqs_filtered[dominant_idx]
            
            if not np.isnan(dominant_freq):
                dominant_frequencies.append(float(dominant_freq))
    
    del vibration_data
    
    return dominant_frequencies


def run_dominant_freq_statistics(file_paths, logger=None):
    """
    对文件列表进行主频分布统计分析，计算95%分位数
    
    参数:
        file_paths: 文件路径列表
        logger: 可选的日志记录器
    
    返回:
        statistics: 统计信息字典
            - 'per_file_frequencies': 每个文件的主频列表（与 file_paths 顺序对应）
            - 'all_dominant_frequencies': 所有主频的平铺列表
            - 'freq_p95': 主频的95%分位数
            - 'freq_stats': 包含 min, max, mean, std, median 的统计信息
    """
    def log_message(msg):
        """统一的日志输出函数"""
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    log_message(f"开始对 {len(file_paths)} 个文件进行主频分布统计...")
    
    file_order_map = {}   # {file_idx: List[float]}
    failed_files = []
    
    log_message(f"开始并行处理文件并计算主频...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file_dominant_frequencies, fp): i
                  for i, fp in enumerate(file_paths)}
        
        for future in tqdm(as_completed(futures), total=len(file_paths), 
                          desc="主频计算进度"):
            file_idx = futures[future]
            file_path = file_paths[file_idx]
            try:
                dominant_freqs = future.result()
                file_order_map[file_idx] = dominant_freqs if dominant_freqs else []
            except Exception as e:
                log_message(f"\n✗ 文件处理失败: {file_path}")
                log_message(f"  错误类型: {type(e).__name__}")
                log_message(f"  错误信息: {str(e)}")
                failed_files.append((file_path, str(e)))
                file_order_map[file_idx] = []
    
    if failed_files:
        log_message(f"\n⚠ 共有 {len(failed_files)} 个文件处理失败：")
        for fp, error in failed_files:
            log_message(f"  - {fp}: {error}")

    # 按原始顺序重建每文件主频列表
    per_file_frequencies = [file_order_map.get(i, []) for i in range(len(file_paths))]
    all_dominant_frequencies = np.array([f for freqs in per_file_frequencies for f in freqs])
    
    if len(all_dominant_frequencies) == 0:
        log_message("警告：无有效主频数据")
        return None
    
    freq_p95 = np.percentile(all_dominant_frequencies, 95)
    
    freq_stats = {
        'min': float(np.min(all_dominant_frequencies)),
        'max': float(np.max(all_dominant_frequencies)),
        'mean': float(np.mean(all_dominant_frequencies)),
        'std': float(np.std(all_dominant_frequencies)),
        'median': float(np.median(all_dominant_frequencies)),
        'p95': float(freq_p95),
        'total_samples': int(len(all_dominant_frequencies))
    }
    
    log_message("\n" + "="*70)
    log_message("              主频分布统计报告")
    log_message("="*70)
    log_message(f"1. 基本统计（所有主频样本）：")
    log_message(f"   - 总样本数: {freq_stats['total_samples']}")
    log_message(f"   - 频率范围: {freq_stats['min']:.4f} ~ {freq_stats['max']:.4f} Hz")
    log_message(f"   - 平均主频: {freq_stats['mean']:.4f} Hz")
    log_message(f"   - 标准差: {freq_stats['std']:.4f} Hz")
    log_message(f"   - 中位数: {freq_stats['median']:.4f} Hz")
    log_message(f"\n2. 极端主频阈值（95%分位数）：")
    log_message(f"   - 主频95%分位值: {freq_stats['p95']:.4f} Hz")
    log_message("="*70 + "\n")
    
    log_message(f"✓ 主频分布统计分析完成")
    log_message(f"✓ 已计算 {len(all_dominant_frequencies)} 个主频样本\n")
    
    statistics = {
        'per_file_frequencies': per_file_frequencies,
        'all_dominant_frequencies': all_dominant_frequencies.tolist(),
        'freq_p95': float(freq_p95),
        'freq_stats': freq_stats
    }
    
    return statistics


def save_dominant_freq_results(statistics, save_path, logger=None):
    """
    保存主频统计结果到 JSON 文件
    
    参数:
        statistics: 统计信息字典
        save_path: 保存路径
        logger: 可选的日志记录器
    """
    def log_message(msg):
        """统一的日志输出函数"""
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=4, ensure_ascii=False)
    
    log_message(f"✓ 主频统计结果已保存至: {save_path}")


if __name__ == "__main__":
    from step0_get_vib_data import get_all_vibration_files
    from step1_lackness_filter import run_lackness_filter
    
    print("步骤0：获取所有振动文件...")
    all_files = get_all_vibration_files()
    
    print(f"\n步骤1：执行缺失率筛选...")
    filtered_paths, stats = run_lackness_filter(all_files)
    
    print(f"\n步骤3：执行主频分布统计...")
    freq_stats = run_dominant_freq_statistics(filtered_paths)
    
    if freq_stats:
        print(f"\n✓ 主频统计完成！")
        print(f"✓ 95%分位值: {freq_stats['freq_p95']:.4f} Hz")
        print(f"✓ 样本数: {freq_stats['freq_stats']['total_samples']}")
