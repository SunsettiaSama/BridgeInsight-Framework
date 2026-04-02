import os
import sys
import json
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入各步骤模块
from src.data_processer.preprocess.wind_data_io_process.step0_get_wind_data import get_all_wind_files
from src.data_processer.preprocess.wind_data_io_process.step1_timestamp_align import run_timestamp_align
from src.data_processer.preprocess.wind_data_io_process.step2_extreme_filter import run_extreme_filter
from src.data_processer.preprocess.wind_data_io_process.step3_out_of_range import run_out_of_range_query
from src.data_processer.io_unpacker import parse_path_metadata

# 从配置文件导入常量
from src.config.data_processer.preprocess.wind_data_io_process.config import (
    FILTER_RESULT_PATH,
    WORKFLOW_CACHE_PATH
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
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        metadata = cache_data.get('metadata', [])
        process_params = cache_data.get('process_params', {})
        
        print(f"✓ 从缓存读取工作流结果")
        print(f"  - 缓存时间: {cache_data.get('timestamp', 'N/A')}")
        print(f"  - 对齐后文件数: {len(metadata)}")
        
        return metadata, process_params
    except Exception as e:
        print(f"警告：读取缓存文件时出错: {e}")
        return None, None


def run(vib_metadata,
        save_path=FILTER_RESULT_PATH,
        cache_path=WORKFLOW_CACHE_PATH,
        use_cache=True,
        force_recompute=False,
        extreme_only=False):
    """
    风数据处理完整工作流
    
    工作流程：
        Step 0: 获取所有风数据文件路径
        Step 1: 根据振动数据元数据进行时间戳对齐筛选
        Step 2: (可选) 筛选出极端振动对应的风数据样本
        Step 3: (可选) 查询极端振动时间范围内的越界点信息
    
    参数:
        vib_metadata: 振动数据元数据列表（来自振动数据workflow）
        save_path: 结果保存路径
        cache_path: 缓存文件路径
        use_cache: 是否使用缓存（默认 True）
        force_recompute: 是否强制重新计算（默认 False）
        extreme_only: 是否只返回极端振动对应的风数据（默认 False）
    
    返回:
        metadata: 纯净的元数据列表，每项包含完整的样本信息：
            - sensor_id: 传感器ID
            - month: 月份
            - day: 日期
            - hour: 小时
            - file_path: 文件路径
            - extreme_time_ranges: (如果 extreme_only=True) 极端振动时间范围
            - out_of_range_windows: (如果 extreme_only=True) 越界点详细信息
    """
    # 重置报告收集器
    global report_collector
    report_collector = ReportCollector()
    
    # 记录处理参数
    report_collector.set_param('vib_metadata_count', len(vib_metadata))
    report_collector.set_param('extreme_only', extreme_only)
    
    # 尝试从缓存读取
    if use_cache and not force_recompute:
        cached_metadata, cached_params = load_workflow_cache(cache_path)
        if cached_metadata is not None:
            # 检查缓存的参数是否匹配（基于振动元数据数量和extreme_only标志）
            if (cached_params.get('vib_metadata_count') == len(vib_metadata) and
                cached_params.get('extreme_only') == extreme_only):
                report_collector.log("✓ 使用缓存结果（参数匹配）\n")
                # 恢复处理参数
                for key, value in cached_params.items():
                    report_collector.set_param(key, value)
                return cached_metadata
            else:
                report_collector.log("⚠ 缓存参数不匹配，将重新计算...")
    
    if force_recompute:
        report_collector.log("⚠ 强制重新计算模式\n")
    
    # ============================================================
    # Step 0: 获取所有风数据文件路径
    # ============================================================
    report_collector.log("="*80)
    report_collector.log(" " * 25 + "风数据处理工作流")
    report_collector.log("="*80)
    report_collector.log("\n[Step 0] 获取所有风数据文件...")
    report_collector.log("-"*80)
    
    all_wind_files = get_all_wind_files()
    
    # 记录处理参数
    report_collector.set_param('total_wind_files', len(all_wind_files))
    report_collector.log(f"✓ 获取到 {len(all_wind_files)} 个风数据文件")
    
    # ============================================================
    # Step 1: 时间戳对齐筛选
    # ============================================================
    report_collector.log("\n[Step 1] 执行时间戳对齐筛选...")
    report_collector.log("-"*80)
    
    aligned_paths, statistics = run_timestamp_align(
        wind_file_paths=all_wind_files,
        vib_metadata=vib_metadata,
        logger=report_collector
    )
    
    # 记录处理参数
    report_collector.set_param('aligned_files', len(aligned_paths))
    report_collector.set_param('coverage', statistics.get('coverage', 0))
    
    # ============================================================
    # 构建扁平化元数据
    # ============================================================
    report_collector.log("\n[数据处理] 构建元数据...")
    report_collector.log("-"*80)
    
    # 直接解析对齐后的文件路径为扁平化元数据
    metadata = parse_path_metadata(aligned_paths)
    
    # 为元数据添加文件路径
    for i, file_path in enumerate(aligned_paths):
        metadata[i]['file_path'] = file_path
    
    # ============================================================
    # Step 2: (可选) 极端振动对应的风数据筛选
    # ============================================================
    if extreme_only:
        report_collector.log("\n[Step 2] 筛选极端振动对应的风数据...")
        report_collector.log("-"*80)
        
        metadata, extreme_stats = run_extreme_filter(
            wind_metadata=metadata,
            vib_metadata=vib_metadata,
            logger=report_collector
        )
        
        # 记录处理参数
        report_collector.set_param('extreme_samples', extreme_stats.get('total_extreme_samples', 0))
        report_collector.set_param('extreme_time_ranges', extreme_stats.get('total_extreme_time_ranges', 0))
        
        # ============================================================
        # Step 3: (可选) 查询极端振动时间范围内的越界点信息
        # ============================================================
        report_collector.log("\n[Step 3] 查询越界点信息...")
        report_collector.log("-"*80)
        
        metadata, out_of_range_stats = run_out_of_range_query(
            filtered_metadata=metadata,
            min_vel=0, max_vel=70,
            min_dir=0, max_dir=360,
            logger=report_collector
        )
        
        # 记录处理参数
        report_collector.set_param('out_of_range_vel', out_of_range_stats.get('total_out_of_range_vel', 0))
        report_collector.set_param('out_of_range_dir', out_of_range_stats.get('total_out_of_range_dir', 0))
        report_collector.set_param('out_of_range_ang', out_of_range_stats.get('total_out_of_range_ang', 0))
        report_collector.set_param('items_with_out_of_range', out_of_range_stats.get('items_with_out_of_range', 0))
    
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
    report_collector.log(f"✓ 原始风数据文件总数: {len(all_wind_files)}")
    report_collector.log(f"✓ 对齐后文件数: {len(aligned_paths)}")
    if extreme_only:
        report_collector.log(f"✓ 极端振动对应的风数据文件数: {report_collector.get_param('extreme_samples', 0)}")
        report_collector.log(f"✓ 极端时间窗口总数: {report_collector.get_param('extreme_time_ranges', 0)}")
        report_collector.log(f"✓ 包含越界点的文件数: {report_collector.get_param('items_with_out_of_range', 0)}")
        report_collector.log(f"✓ 风速越界点总数: {report_collector.get_param('out_of_range_vel', 0)}")
        report_collector.log(f"✓ 风向越界点总数: {report_collector.get_param('out_of_range_dir', 0)}")
        report_collector.log(f"✓ 风攻角越界点总数: {report_collector.get_param('out_of_range_ang', 0)}")
    else:
        report_collector.log(f"✓ 最终输出文件数: {len(metadata)}")
    report_collector.log(f"✓ 对齐率: {len(aligned_paths)/len(all_wind_files)*100:.2f}%" if len(all_wind_files) > 0 else "✓ 对齐率: 0%")
    report_collector.log(f"✓ 时间戳覆盖率: {statistics.get('coverage', 0):.2f}%")
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
    # 需要先运行振动数据 workflow 获取 vib_metadata
    from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow
    
    print("="*80)
    print(" " * 20 + "风数据处理工作流 - 完整示例")
    print("="*80)
    
    # 1. 运行振动数据工作流
    print("\n[前置步骤] 运行振动数据工作流...")
    print("-"*80)
    vib_metadata = run_vib_workflow(use_cache=True)
    
    print(f"\n✓ 获取振动数据元数据: {len(vib_metadata)} 条")
    
    # 2. 运行风数据工作流
    print("\n" + "="*80)
    print(" " * 20 + "开始风数据处理工作流")
    print("="*80)
    
    wind_metadata = run(
        vib_metadata=vib_metadata,
        use_cache=True,
        force_recompute=False
    )
    
    # 3. 显示示例结果
    print("\n[示例] 前3个对齐后的风数据文件元数据：")
    for i, item in enumerate(wind_metadata[:3], 1):
        print(f"\n{i}. 文件路径: {item.get('file_path', 'N/A')}")
        print(f"   传感器: {item.get('sensor_id', 'N/A')}")
        print(f"   时间: {item.get('month', 'N/A')}/{item.get('day', 'N/A')} {item.get('hour', 'N/A')}:00")
    
    # 4. 显示处理参数摘要
    print("\n[处理参数摘要]")
    print(f"  振动数据元数据数: {report_collector.get_param('vib_metadata_count', 'N/A')}")
    print(f"  原始风数据文件数: {report_collector.get_param('total_wind_files', 'N/A')}")
    print(f"  对齐后文件数: {report_collector.get_param('aligned_files', 'N/A')}")
    print(f"  时间戳覆盖率: {report_collector.get_param('coverage', 0):.2f}%")
