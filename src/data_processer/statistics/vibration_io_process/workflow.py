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
from src.data_processer.statistics.vibration_io_process.step0_get_vib_data import get_all_vibration_files
from src.data_processer.statistics.vibration_io_process.step1_lackness_filter import run_lackness_filter
from src.data_processer.io_unpacker import parse_path_metadata

# 从配置文件导入常量
from src.config.data_processer.statistics.vibration_io_process.config import (
    MISSING_RATE_THRESHOLD,
    EXPECTED_LENGTH,
    FILTER_RESULT_PATH,
    WORKFLOW_CACHE_PATH
)


# 全局报告收集器
class ReportCollector:
    """收集工作流执行过程中的所有输出信息"""
    def __init__(self):
        self.logs = []
    
    def log(self, message):
        """记录消息并打印"""
        print(message)
        self.logs.append(message)
    
    def get_report(self):
        """获取完整报告"""
        return "\n".join(self.logs)


# 全局报告收集器实例
report_collector = ReportCollector()


def save_workflow_cache(results, cache_path=WORKFLOW_CACHE_PATH):
    """
    保存工作流结果到缓存文件
    
    参数:
        results: 工作流结果字典
        cache_path: 缓存文件路径
    """
    # 添加时间戳
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['threshold'] = MISSING_RATE_THRESHOLD
    results['expected_length'] = EXPECTED_LENGTH
    
    # 确保目录存在
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    # 保存为 JSON
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✓ 工作流缓存已保存至: {cache_path}")


def load_workflow_cache(cache_path=WORKFLOW_CACHE_PATH):
    """
    从缓存文件读取工作流结果
    
    参数:
        cache_path: 缓存文件路径
    
    返回:
        results: 工作流结果字典，如果文件不存在返回 None
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"✓ 从缓存读取工作流结果")
        print(f"  - 缓存时间: {results.get('timestamp', 'N/A')}")
        print(f"  - 缺失率阈值: {results.get('threshold', 'N/A')*100:.1f}%")
        print(f"  - 筛选后文件数: {len(results.get('file_paths', []))}")
        
        return results
    except Exception as e:
        print(f"警告：读取缓存文件时出错: {e}")
        return None


def run_vibration_data_workflow(threshold=MISSING_RATE_THRESHOLD, 
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
    
    参数:
        threshold: 缺失率阈值（默认 0.05，即 5%）
        expected_length: 预期单文件长度（默认 50Hz * 1小时）
        save_path: 结果保存路径
        cache_path: 缓存文件路径
        use_cache: 是否使用缓存（默认 True）
        force_recompute: 是否强制重新计算（默认 False）
    
    返回:
        results: 包含结果的字典
            - 'file_paths': 筛选后的文件路径列表
            - 'metadata': 扁平化元数据列表
    """
    # 重置报告收集器
    global report_collector
    report_collector = ReportCollector()
    
    # 尝试从缓存读取
    if use_cache and not force_recompute:
        cached_results = load_workflow_cache(cache_path)
        if cached_results is not None:
            # 检查缓存的参数是否匹配
            if (cached_results.get('threshold') == threshold and 
                cached_results.get('expected_length') == expected_length):
                report_collector.log("✓ 使用缓存结果（参数匹配）\n")
                return cached_results
            else:
                report_collector.log("⚠ 缓存参数不匹配，将重新计算...")
    
    if force_recompute:
        report_collector.log("⚠ 强制重新计算模式\n")
    
    results = {}
    
    # ============================================================
    # Step 0: 获取所有振动文件路径
    # ============================================================
    report_collector.log("="*80)
    report_collector.log(" " * 25 + "振动数据处理工作流")
    report_collector.log("="*80)
    report_collector.log("\n[Step 0] 获取所有振动文件...")
    report_collector.log("-"*80)
    
    all_file_paths = get_all_vibration_files()
    
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
    
    # ============================================================
    # 构建扁平化元数据
    # ============================================================
    report_collector.log("\n[数据处理] 构建元数据...")
    report_collector.log("-"*80)
    
    # 直接解析筛选后的文件路径为扁平化元数据
    metadata = parse_path_metadata(filtered_paths)
    
    # 为元数据添加缺失率信息
    for i, idx in enumerate(statistics['filtered_indices']):
        metadata[i]['actual_length'] = int(statistics['all_lengths'][idx])
        metadata[i]['missing_rate'] = float(statistics['all_missing_rates'][idx])
    
    # 保存元数据到文件
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    report_collector.log(f"✓ 元数据构建完成 ({len(metadata)} 个文件)")
    report_collector.log(f"✓ 结果已保存至: {save_path}")
    
    # 组装最终结果
    results['file_paths'] = filtered_paths
    results['metadata'] = metadata
    
    # ============================================================
    # 工作流总结
    # ============================================================
    report_collector.log("\n" + "="*80)
    report_collector.log(" " * 30 + "工作流完成")
    report_collector.log("="*80)
    report_collector.log(f"✓ 原始文件总数: {len(all_file_paths)}")
    report_collector.log(f"✓ 筛选后文件数: {len(filtered_paths)}")
    report_collector.log(f"✓ 筛选通过率: {len(filtered_paths)/len(all_file_paths)*100:.2f}%")
    report_collector.log("="*80 + "\n")
    
    # 保存工作流缓存和报告
    if use_cache:
        save_workflow_cache(results, cache_path)
        # 保存报告到文本文件
        report_path = cache_path.replace('.json', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_collector.get_report())
        report_collector.log(f"✓ 工作流报告已保存至: {report_path}")
    
    return results


if __name__ == "__main__":
    # 运行完整工作流（默认使用缓存）
    # 如果需要强制重新计算，设置 force_recompute=True
    results = run_vibration_data_workflow(
        use_cache=True,           # 使用缓存
        force_recompute=False     # 如果缓存存在且参数匹配，使用缓存
    )
    
    # 可选：显示一些示例结果
    print("\n[示例] 前3个筛选后的文件元数据：")
    for i, item in enumerate(results['metadata'][:3], 1):
        print(f"\n{i}. 传感器: {item.get('sensor_id', 'N/A')}")
        print(f"   时间: {item.get('month', 'N/A')}/{item.get('day', 'N/A')} {item.get('hour', 'N/A')}:00")
        print(f"   实际长度: {item.get('actual_length', 'N/A')}")
        print(f"   缺失率: {item.get('missing_rate', 0)*100:.2f}%")
    
    # 示例：强制重新计算
    # results = run_vibration_data_workflow(force_recompute=True)
