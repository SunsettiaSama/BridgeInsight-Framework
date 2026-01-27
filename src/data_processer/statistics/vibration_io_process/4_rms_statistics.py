import os
import json
import numpy as np
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

# 添加项目根目录到 sys.path 以支持 src.xxx 形式的导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import parse_path_metadata

# 1. 常量硬编码
JSON_SAVE_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\results\statistics\rms_statistics.json'
JSON_EXTREME_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\results\statistics\rms_statistics_extreme.json'
ALL_VIBRATION_ROOT = r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC"
FS = 50
TIME_WINDOW = 60.0
TARGET_SENSORS = [
    'ST-VIC-C34-101-02', 'ST-VIC-C34-101-01',
    'ST-VIC-C34-102-01', 'ST-VIC-C34-102-02',
    'ST-VIC-C18-101-01', 'ST-VIC-C18-101-02',
    'ST-VIC-C18-102-01', 'ST-VIC-C18-102-02',
    'ST-VIC-C34-201-01', 'ST-VIC-C34-201-02',
    'ST-VIC-C34-202-01', 'ST-VIC-C34-202-02',
    'ST-VIC-C34-301-01', 'ST-VIC-C34-301-02',
    'ST-VIC-C34-302-01', 'ST-VIC-C34-302-02',
    'ST-VIC-C18-301-01', 'ST-VIC-C18-301-02',
    'ST-VIC-C18-302-01', 'ST-VIC-C18-302-02'
]

def load_rms_metadata(json_path=JSON_SAVE_PATH):
    """
    封装函数：从 JSON 中读取结果文件并返回为元数据列表
    
    参数:
        json_path: JSON 文件路径
            - 默认为 JSON_SAVE_PATH (P95 结果, Top 5%)
            - 可传入 JSON_EXTREME_PATH (极端结果, Top 0.25%)
    
    返回:
        元数据列表，每个元素包含 path, indices 等字段
    """
    if not os.path.exists(json_path):
        print(f"错误：文件 {json_path} 不存在。")
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def vic_data_unpack_local(file_path):
    """
    本地 VIC 数据解析实现，确保环境兼容性
    """
    split_str = '_'
    strings = os.path.split(file_path)
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            prefix = strings[-1].split(split_str)[0].encode('utf-8') if split_str else strings[-1].encode('utf-8')
            idx_in_data = data.index(prefix) + len(prefix) + 1
            count = (len(data) - idx_in_data) // 4
            if count <= 0: return np.array([])
            return np.array(struct.unpack("f" * count, data[idx_in_data:]))
    except Exception:
        return np.array([])

def process_single_file(file_path, window_size):
    """
    单文件处理：计算各窗口 RMS 及其索引
    """
    try:
        vibration_data = vic_data_unpack_local(file_path)
        if len(vibration_data) == 0: return file_path, []
        rms_info = []
        if len(vibration_data) >= window_size:
            for i in range(0, len(vibration_data) - window_size + 1, window_size):
                rms_val = np.sqrt(np.mean(np.square(vibration_data[i:i+window_size])))
                if rms_val > 0: rms_info.append((i, float(rms_val)))
        else:
            rms_val = np.sqrt(np.mean(np.square(vibration_data)))
            if rms_val > 0: rms_info.append((0, float(rms_val)))
        return file_path, rms_info
    except Exception:
        return file_path, []

def print_statistics(total_files, file_rms_results, all_rms, threshold_p95, final_threshold):
    """
    详细打印统计结果
    """
    all_rms = np.array(all_rms)
    total_samples = len(all_rms)
    
    # 样本级统计
    samples_above_p95 = np.sum(all_rms >= threshold_p95)
    samples_below_p95 = np.sum(all_rms < threshold_p95)
    samples_extreme = np.sum(all_rms >= final_threshold)
    
    # 文件级统计
    files_with_p95 = 0
    files_with_extreme = 0
    files_all_below_p95 = 0
    for fp, rms_info in file_rms_results.items():
        rmss = [item[1] for item in rms_info]
        if any(r >= threshold_p95 for r in rmss):
            files_with_p95 += 1
        if any(r >= final_threshold for r in rmss):
            files_with_extreme += 1
        if all(r < threshold_p95 for r in rmss):
            files_all_below_p95 += 1

    print("\n" + "="*70)
    print("                    RMS 统计详细报告")
    print("="*70)
    print(f"1. 阈值参数：")
    print(f"   - P95 阈值 (高振动): {threshold_p95:.6f} m/s²  [保存至主文件]")
    print(f"   - 极端阈值 (Top 0.25%): {final_threshold:.6f} m/s²  [另存至极端文件]")
    print(f"\n2. 样本统计 (总计: {total_samples} 个窗口):")
    print(f"   - P95 以上样本 (Top 5%): {samples_above_p95} ({samples_above_p95/total_samples*100:.2f}%)")
    print(f"   - P95 以下样本: {samples_below_p95} ({samples_below_p95/total_samples*100:.2f}%)")
    print(f"   - 极端振动样本 (Top 0.25%): {samples_extreme} ({samples_extreme/total_samples*100:.4f}%)")
    print(f"\n3. 文件统计 (总计: {total_files} 个文件):")
    print(f"   - 包含 P95 以上振动的文件: {files_with_p95}")
    print(f"   - 包含极端振动的文件: {files_with_extreme}")
    print(f"   - 所有窗口均低于 P95 的文件: {files_all_below_p95}")
    print("="*70 + "\n")

def main():
    window_size = int(TIME_WINDOW * FS)
    vibration_files = []
    for root, _, files in os.walk(ALL_VIBRATION_ROOT):
        for file in files:
            if file.upper().endswith(".VIC") and any(sid in file for sid in TARGET_SENSORS):
                vibration_files.append(os.path.join(root, file))
    
    file_rms_results = {}
    all_rms_list = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, fp, window_size): fp for fp in vibration_files}
        for future in tqdm(as_completed(futures), total=len(vibration_files), desc="计算进度"):
            fp, rms_info = future.result()
            if rms_info:
                file_rms_results[fp] = rms_info
                all_rms_list.extend([item[1] for item in rms_info])
    
    if not all_rms_list: return
    
    # 逻辑对齐：双重 P95 筛选
    threshold_p95 = np.percentile(all_rms_list, 95)
    samples_above_p95 = [r for r in all_rms_list if r >= threshold_p95]
    final_threshold = np.percentile(samples_above_p95, 95)

    # 打印详细统计
    print_statistics(len(vibration_files), file_rms_results, all_rms_list, threshold_p95, final_threshold)

    # 1. 构建并保存 P95 元数据（top 5%，主文件）
    metadata_p95 = []
    for fp, rms_info in file_rms_results.items():
        indices = [int(idx) for idx, rms in rms_info if rms >= threshold_p95]
        if indices: 
            metadata_p95.append({"path": fp, "indices": indices})
    
    # 完善元数据逻辑：嵌入 parse_path_metadata
    metadata_p95 = parse_path_metadata(metadata_p95)
    
    os.makedirs(os.path.dirname(JSON_SAVE_PATH), exist_ok=True)
    with open(JSON_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_p95, f, indent=4, ensure_ascii=False)
    
    print(f"✓ P95 元数据（Top 5%）已保存至：{JSON_SAVE_PATH}")
    print(f"  - 包含 {len(metadata_p95)} 个文件的高振动事件")
    
    # 2. 构建并保存极端振动元数据（top 0.25%，另存文件）
    metadata_extreme = []
    for fp, rms_info in file_rms_results.items():
        indices = [int(idx) for idx, rms in rms_info if rms >= final_threshold]
        if indices:
            metadata_extreme.append({"path": fp, "indices": indices})
    
    # 完善元数据逻辑：嵌入 parse_path_metadata
    metadata_extreme = parse_path_metadata(metadata_extreme)
    
    with open(JSON_EXTREME_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_extreme, f, indent=4, ensure_ascii=False)
    
    print(f"✓ 极端振动元数据（Top 0.25%）已保存至：{JSON_EXTREME_PATH}")
    print(f"  - 包含 {len(metadata_extreme)} 个文件的极端振动事件\n")

if __name__ == "__main__":
    main()