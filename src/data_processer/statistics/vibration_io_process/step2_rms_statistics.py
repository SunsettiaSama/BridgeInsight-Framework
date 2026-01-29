import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的解析工具
from src.data_processer.io_unpacker import UNPACK

# 硬编码参数
FS = 50  # 振动信号采样频率
TIME_WINDOW = 60.0  # 计算RMS的时间窗口（秒）


def process_single_file_rms(file_path, window_size):
    """
    单文件RMS计算工作函数，用于多进程
    
    参数:
        file_path: 文件路径
        window_size: 时间窗口大小（采样点数）
    
    返回:
        rms_list: RMS值列表 (numpy array)
    """
    try:
        import numpy as np
        from src.data_processer.io_unpacker import UNPACK
        unpacker = UNPACK(init_path=False)
        vibration_data = unpacker.VIC_DATA_Unpack(file_path)
        vibration_data = np.array(vibration_data)
        
        if len(vibration_data) == 0:
            return np.array([])
            
        rms_list = []
        
        if len(vibration_data) >= window_size:
            for i in range(0, len(vibration_data) - window_size + 1, window_size):
                window_data = vibration_data[i:i+window_size]
                # 计算信号的均方根RMS
                rms_val = np.sqrt(np.mean(np.square(window_data)))
                if rms_val > 0:
                    rms_list.append(rms_val)
        else:
            rms_val = np.sqrt(np.mean(np.square(vibration_data)))
            if rms_val > 0:
                rms_list.append(rms_val)
        
        return np.array(rms_list)
    except Exception as e:
        return np.array([])


def run_rms_statistics(file_paths, 
                       fs=FS, 
                       time_window=TIME_WINDOW,
                       logger=None):
    """
    对文件列表进行RMS统计分析，识别极端振动
    
    参数:
        file_paths: 文件路径列表
        fs: 采样频率（默认 50Hz）
        time_window: 时间窗口（默认 60秒）
        logger: 可选的日志记录器
    
    返回:
        file_paths: 原样返回文件路径列表
        statistics: 统计信息字典
            - 'all_file_rms': 每个文件的RMS数组列表
            - 'extreme_indices': 每个文件中超过95%阈值的窗口索引列表
            - 'rms_threshold_95': 95%分位值阈值
    """
    def log_message(msg):
        """统一的日志输出函数"""
        if logger:
            logger.log(msg)
        else:
            print(msg)
    
    log_message(f"开始对 {len(file_paths)} 个文件进行RMS统计分析...")
    
    # 计算窗口大小
    window_size = int(time_window * fs)
    
    # 收集每个文件的RMS数据
    all_file_rms = []
    file_order_map = {}  # 维护文件顺序映射
    
    # 使用多进程并行获取数据
    log_message(f"开始并行处理文件并计算RMS...")
    with ProcessPoolExecutor() as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_file_rms, fp, window_size): i 
                  for i, fp in enumerate(file_paths)}
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(file_paths), 
                          desc="RMS计算进度"):
            try:
                file_idx = futures[future]
                rms_array = future.result()
                file_order_map[file_idx] = rms_array
            except Exception as e:
                log_message(f"处理任务时出错: {e}")
                file_order_map[futures[future]] = np.array([])
    
    # 按原始顺序恢复
    all_file_rms = [file_order_map[i] for i in range(len(file_paths))]
    
    # 收集所有RMS值用于计算全局阈值
    all_rms_values = np.concatenate([rms for rms in all_file_rms if len(rms) > 0])
    
    if len(all_rms_values) == 0:
        log_message("警告：无有效振动样本数据")
        return file_paths, None
    
    # 动态计算RMS阈值：采取统计上的95%分位值
    rms_threshold_95 = np.percentile(all_rms_values, 95)
    
    # 统计信息
    total_samples = len(all_rms_values)
    below_threshold = np.sum(all_rms_values < rms_threshold_95)
    above_threshold = np.sum(all_rms_values >= rms_threshold_95)
    
    # 计算每个文件中超过阈值的窗口索引
    extreme_indices = []
    total_extreme_files = 0
    for rms_array in all_file_rms:
        if len(rms_array) > 0:
            indices = np.where(rms_array >= rms_threshold_95)[0].tolist()
            extreme_indices.append(indices)
            if len(indices) > 0:
                total_extreme_files += 1
        else:
            extreme_indices.append([])
    
    # 打印详细统计信息
    log_message("\n" + "="*70)
    log_message("                    RMS极端振动识别报告")
    log_message("="*70)
    log_message(f"1. 基本统计（所有样本）：")
    log_message(f"   - 总样本数: {total_samples}")
    log_message(f"   - 平均RMS: {np.mean(all_rms_values):.4f} m/s²")
    log_message(f"   - 标准差: {np.std(all_rms_values):.4f} m/s²")
    log_message(f"   - 最小值: {np.min(all_rms_values):.4f} m/s²")
    log_message(f"   - 最大值: {np.max(all_rms_values):.4f} m/s²")
    log_message(f"\n2. 极端振动阈值（95%分位值）：")
    log_message(f"   - RMS阈值: {rms_threshold_95:.4f} m/s²")
    log_message(f"\n3. 极端振动识别统计：")
    log_message(f"   - 小于阈值样本数: {below_threshold} ({below_threshold/total_samples*100:.2f}%)")
    log_message(f"   - 大于等于阈值样本数: {above_threshold} ({above_threshold/total_samples*100:.2f}%)")
    log_message(f"   - 包含极端振动的文件数: {total_extreme_files} / {len(file_paths)} ({total_extreme_files/len(file_paths)*100:.2f}%)")
    log_message("="*70 + "\n")
    
    log_message(f"✓ RMS统计分析完成")
    log_message(f"✓ 已识别 {len(file_paths)} 个文件中的极端振动窗口\n")
    
    # 组装统计信息（只保留必要的索引数据）
    statistics = {
        'all_file_rms': all_file_rms,  # 每个文件的RMS数组
        'extreme_indices': extreme_indices,  # 每个文件的极端振动窗口索引
        'rms_threshold_95': rms_threshold_95  # 全局阈值
    }
    
    return file_paths, statistics


if __name__ == "__main__":
    # 测试接口
    from step0_get_vib_data import get_all_vibration_files
    from step1_lackness_filter import run_lackness_filter
    
    print("步骤0：获取所有振动文件...")
    all_files = get_all_vibration_files()
    
    print(f"\n步骤1：执行缺失率筛选...")
    filtered_paths, stats = run_lackness_filter(all_files)
    
    print(f"\n步骤2：执行RMS统计分析...")
    file_paths, rms_stats = run_rms_statistics(filtered_paths)
    
    if rms_stats:
        print(f"\n✓ RMS统计完成！")
        print(f"✓ 95%分位值阈值: {rms_stats['rms_threshold_95']:.4f} m/s²")
        print(f"✓ 返回文件数: {len(file_paths)}")
        
        # 统计有极端振动的文件
        files_with_extreme = sum(1 for indices in rms_stats['extreme_indices'] if len(indices) > 0)
        print(f"✓ 包含极端振动的文件: {files_with_extreme} / {len(file_paths)}")
