import os
import sys
import json
import numpy as np
from datetime import datetime
from io import StringIO

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入各步骤模块
from src.data_processer.preprocess.vibration_io_process.step0_get_vib_data import get_all_vibration_files
from src.data_processer.preprocess.vibration_io_process.step1_lackness_filter import run_lackness_filter
from src.data_processer.preprocess.vibration_io_process.step2_rms_statistics import run_rms_statistics
from src.data_processer.preprocess.vibration_io_process.step3_dominant_freq_statistics import (
    run_dominant_freq_statistics, save_dominant_freq_results
)
from src.data_processer.io_unpacker import parse_path_metadata

# 从配置文件导入常量
from src.config.data_processer.preprocess.vibration_io_process.config import (
    MISSING_RATE_THRESHOLD,
    EXPECTED_LENGTH,
    FILTER_RESULT_PATH,
    WORKFLOW_CACHE_PATH,
    DOMINANT_FREQ_STATISTICS_RESULT_PATH
)


# 全局报告收集器
class ReportCollector:
    """收集工作流执行过程中的所有输出信息和处理参数"""
    def __init__(self):
        self.logs = []
        self.process_params = {}  # 存储处理过程中的关键参数
    
    def log(self, message):
        """记录消息并打印"""
        print(message)
        self.logs.append(message)
    
    def set_param(self, key, value):
        """设置处理参数"""
        self.process_params[key] = value
    
    def get_param(self, key, default=None):
        """获取处理参数"""
        return self.process_params.get(key, default)
    
    def get_report(self):
        """获取完整报告"""
        return "\n".join(self.logs)
    
    def get_params(self):
        """获取所有处理参数"""
        return self.process_params.copy()


# 全局报告收集器实例
report_collector = ReportCollector()


def save_workflow_cache(metadata, process_params, cache_path=WORKFLOW_CACHE_PATH):
    """
    保存工作流结果到缓存文件
    
    参数:
        metadata: 纯净的元数据列表
        process_params: 处理参数字典
        cache_path: 缓存文件路径
    """
    # 组装缓存数据
    cache_data = {
        'metadata': metadata,
        'process_params': process_params,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 确保目录存在
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # 保存为 JSON
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=4, ensure_ascii=False)
    
    print(f"✓ 工作流缓存已保存至: {cache_path}")


def load_workflow_cache(cache_path=WORKFLOW_CACHE_PATH):
    """
    从缓存文件读取工作流结果
    
    参数:
        cache_path: 缓存文件路径
    
    返回:
        metadata: 元数据列表，如果文件不存在返回 None
        process_params: 处理参数字典，如果文件不存在返回 None
    """
    if not os.path.exists(cache_path):
        return None, None
    
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    metadata = cache_data.get('metadata', [])
    process_params = cache_data.get('process_params', {})
    
    print(f"✓ 从缓存读取工作流结果")
    print(f"  - 缓存时间: {cache_data.get('timestamp', 'N/A')}")
    print(f"  - 缺失率阈值: {process_params.get('missing_rate_threshold', 'N/A')}")
    print(f"  - 筛选后文件数: {len(metadata)}")
    
    return metadata, process_params


def run(threshold=MISSING_RATE_THRESHOLD, 
        expected_length=EXPECTED_LENGTH,
        save_path=FILTER_RESULT_PATH,
        cache_path=WORKFLOW_CACHE_PATH,
        use_cache=True,
        force_recompute=False):
    """
    振动数据处理完整工作流
    
    工作流程：
        Step 0: 获取所有振动文件路径
        Step 1: 缺失率筛选
        Step 2: RMS统计分析与极端振动识别
    
    参数:
        threshold: 缺失率阈值（默认 0.05，即 5%）
        expected_length: 预期单文件长度（默认 50Hz * 1小时）
        save_path: 结果保存路径
        cache_path: 缓存文件路径
        use_cache: 是否使用缓存（默认 True）
        force_recompute: 是否强制重新计算（默认 False）
    
    返回:
        metadata: 纯净的元数据列表，每项包含完整的样本信息：
            - sensor_id: 传感器ID
            - month: 月份
            - day: 日期
            - hour: 小时
            - file_path: 文件路径
            - actual_length: 实际数据长度
            - missing_rate: 缺失率
            - extreme_rms_indices: 极端RMS振动的窗口索引列表
    """
    # 重置报告收集器
    global report_collector
    report_collector = ReportCollector()

    # 从配置中读取用于 fingerprint 的附加字段
    from src.config.data_processer.preprocess.vibration_io_process.config import (
        ALL_VIBRATION_ROOT, TARGET_VIBRATION_SENSORS, NFFT
    )
    current_fingerprint = {
        'missing_rate_threshold': threshold,
        'expected_length':        expected_length,
        'all_vibration_root':     ALL_VIBRATION_ROOT,
        'target_sensors':         sorted(TARGET_VIBRATION_SENSORS),
        'nfft':                   NFFT,
    }

    # 记录处理参数
    for k, v in current_fingerprint.items():
        report_collector.set_param(k, v)
    
    # 尝试从缓存读取
    if use_cache and not force_recompute:
        cached_metadata, cached_params = load_workflow_cache(cache_path)
        if cached_metadata is not None:
            # 检查缓存的参数是否与当前 fingerprint 完全一致
            mismatch = {
                k: {'cached': cached_params.get(k), 'current': v}
                for k, v in current_fingerprint.items()
                if cached_params.get(k) != v
            }
            if not mismatch:
                report_collector.log("✓ 使用缓存结果（参数匹配）\n")
                for key, value in cached_params.items():
                    report_collector.set_param(key, value)
                return cached_metadata
            else:
                report_collector.log(f"⚠ 缓存参数不匹配，将重新计算... 差异字段：{mismatch}")
    
    if force_recompute:
        report_collector.log("⚠ 强制重新计算模式\n")
    
    # ============================================================
    # Step 0: 获取所有振动文件路径
    # ============================================================
    report_collector.log("="*80)
    report_collector.log(" " * 25 + "振动数据处理工作流")
    report_collector.log("="*80)
    report_collector.log("\n[Step 0] 获取所有振动文件...")
    report_collector.log("-"*80)
    
    all_file_paths = get_all_vibration_files()
    
    # 记录处理参数
    report_collector.set_param('total_files', len(all_file_paths))
    report_collector.log(f"✓ 获取到 {len(all_file_paths)} 个振动文件")
    
    # ============================================================
    # Step 1: 缺失率筛选
    # ============================================================
    report_collector.log("\n[Step 1] 执行缺失率筛选...")
    report_collector.log("-"*80)
    
    filtered_paths, statistics = run_lackness_filter(
        all_file_paths=all_file_paths,
        threshold=threshold,
        expected_length=expected_length,
        logger=report_collector
    )
    
    # 记录处理参数
    report_collector.set_param('filtered_files', len(filtered_paths))
    
    # ============================================================
    # Step 2: RMS统计分析与极端振动识别
    # ============================================================
    report_collector.log("\n[Step 2] 执行RMS统计分析与极端振动识别...")
    report_collector.log("-"*80)
    
    rms_file_paths, rms_statistics = run_rms_statistics(
        file_paths=filtered_paths,
        logger=report_collector
    )
    
    # 记录RMS阈值参数
    if rms_statistics:
        report_collector.set_param('rms_threshold_95', rms_statistics['rms_threshold_95'])
    
    # ============================================================
    # Step 3: 主频分布统计分析
    # ============================================================
    report_collector.log("\n[Step 3] 执行主频分布统计分析...")
    report_collector.log("-"*80)
    
    freq_statistics = run_dominant_freq_statistics(
        file_paths=filtered_paths,
        logger=report_collector
    )
    
    # 记录主频统计参数
    if freq_statistics:
        report_collector.set_param('freq_p95', freq_statistics['freq_p95'])
        report_collector.set_param('freq_stats', freq_statistics['freq_stats'])
        # 保存主频统计结果
        save_dominant_freq_results(
            statistics=freq_statistics,
            save_path=DOMINANT_FREQ_STATISTICS_RESULT_PATH,
            logger=report_collector
        )
    
    # ============================================================
    # 构建扁平化元数据
    # ============================================================
    report_collector.log("\n[数据处理] 构建元数据...")
    report_collector.log("-"*80)
    
    # 直接解析筛选后的文件路径为扁平化元数据
    metadata = parse_path_metadata(filtered_paths)
    
    # 为元数据添加完整的样本信息
    for i, idx in enumerate(statistics['filtered_indices']):
        metadata[i]['file_path'] = filtered_paths[i]
        metadata[i]['actual_length'] = int(statistics['all_lengths'][idx])
        metadata[i]['missing_rate'] = float(statistics['all_missing_rates'][idx])
        
        # 添加 RMS 逐窗口值及极端振动索引
        if rms_statistics:
            rms_arr = rms_statistics['all_file_rms'][i]
            metadata[i]['rms_per_window'] = [float(v) for v in rms_arr]
            metadata[i]['extreme_rms_indices'] = rms_statistics['extreme_indices'][i]
        
        # 添加主频逐窗口值
        if freq_statistics and 'per_file_frequencies' in freq_statistics:
            metadata[i]['dominant_freq_per_window'] = freq_statistics['per_file_frequencies'][i]
    
    # 记录包含极端振动的文件数
    if rms_statistics:
        files_with_extreme = sum(1 for item in metadata if len(item.get('extreme_rms_indices', [])) > 0)
        report_collector.set_param('files_with_extreme_vibration', files_with_extreme)
    
    # 保存元数据到文件
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    report_collector.log(f"✓ 元数据构建完成 ({len(metadata)} 个文件)")
    report_collector.log(f"✓ 结果已保存至: {save_path}")
    
    # ============================================================
    # 工作流总结
    # ============================================================
    report_collector.log("\n" + "="*80)
    report_collector.log(" " * 30 + "工作流完成")
    report_collector.log("="*80)
    report_collector.log(f"✓ 原始文件总数: {len(all_file_paths)}")
    report_collector.log(f"✓ 筛选后文件数: {len(metadata)}")
    report_collector.log(f"✓ 筛选通过率: {len(metadata)/len(all_file_paths)*100:.2f}%")
    
    if rms_statistics:
        report_collector.log(f"✓ RMS 95%分位值: {rms_statistics['rms_threshold_95']:.4f} m/s²")
    
    if freq_statistics:
        report_collector.log(f"✓ 主频 95%分位值: {freq_statistics['freq_p95']:.4f} Hz")
        report_collector.log(f"✓ 主频样本总数: {freq_statistics['freq_stats']['total_samples']}")
    
    report_collector.log("="*80 + "\n")
    
    # 保存工作流缓存和报告
    if use_cache:
        save_workflow_cache(metadata, report_collector.get_params(), cache_path)
        # 保存报告到文本文件
        report_path = cache_path.replace('.json', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_collector.get_report())
        report_collector.log(f"✓ 工作流报告已保存至: {report_path}")
    
    return metadata


if __name__ == "__main__":
    # 运行完整工作流（默认使用缓存）
    # 如果需要强制重新计算，设置 force_recompute=True
    metadata = run(
        use_cache=True,           # 使用缓存
        force_recompute=False     # 如果缓存存在且参数匹配，使用缓存
    )
    
    # 可选：显示一些示例结果
    print("\n[示例] 前3个筛选后的文件元数据：")
    for i, item in enumerate(metadata[:3], 1):
        print(f"\n{i}. 文件路径: {item.get('file_path', 'N/A')}")
        print(f"   传感器: {item.get('sensor_id', 'N/A')}")
        print(f"   时间: {item.get('month', 'N/A')}/{item.get('day', 'N/A')} {item.get('hour', 'N/A')}:00")
        print(f"   实际长度: {item.get('actual_length', 'N/A')}")
        print(f"   缺失率: {item.get('missing_rate', 0)*100:.2f}%")
        extreme_indices = item.get('extreme_rms_indices', [])
        print(f"   极端振动窗口数: {len(extreme_indices)}")
    
    # 显示处理参数摘要
    print("\n[处理参数摘要]")
    print(f"  原始文件总数: {report_collector.get_param('total_files', 'N/A')}")
    print(f"  筛选后文件数: {report_collector.get_param('filtered_files', 'N/A')}")
    rms_threshold = report_collector.get_param('rms_threshold_95')
    if rms_threshold:
        print(f"  RMS 95%分位值阈值: {rms_threshold:.4f} m/s²")
        files_with_extreme = report_collector.get_param('files_with_extreme_vibration', 0)
        print(f"  包含极端振动的文件数: {files_with_extreme} / {len(metadata)}")
    
    freq_p95 = report_collector.get_param('freq_p95')
    if freq_p95:
        print(f"  主频 95%分位值: {freq_p95:.4f} Hz")
        freq_stats = report_collector.get_param('freq_stats')
        if freq_stats:
            print(f"  主频样本总数: {freq_stats['total_samples']}")
            print(f"  主频范围: {freq_stats['min']:.4f} ~ {freq_stats['max']:.4f} Hz")
            print(f"  主频均值: {freq_stats['mean']:.4f} Hz")
    
    # 示例：强制重新计算
    # metadata = run(force_recompute=True)
