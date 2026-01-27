import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入必要的解析工具
from src.data_processer.io_unpacker import UNPACK, parse_path_metadata

# 常量定义
MISSING_RATE_THRESHOLD = 0.05
FILTER_RESULT_PATH = r'F:\Research\Vibration Characteristics In Cable Vibration\docs\data_processer\statistics\files_after_lackness_filter.json'
FS = 50
EXPECTED_LENGTH = int(FS * 60 * 60)  # 50Hz * 1小时

def get_file_length(file_path):
    """获取单个源文件的数据长度"""
    try:
        unpacker = UNPACK(init_path=False)
        vibration_data = unpacker.VIC_DATA_Unpack(file_path)
        return len(vibration_data)
    except Exception:
        return 0

def run_lackness_filter(all_file_paths, threshold=MISSING_RATE_THRESHOLD, expected_length=EXPECTED_LENGTH):
    """
    对文件列表进行缺失率筛选，并保存符合条件的文件路径
    """
    print(f"开始对 {len(all_file_paths)} 个文件进行缺失率筛选 (阈值: {threshold*100:.1f}%)...")
    
    filtered_results = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(get_file_length, fp): fp for fp in all_file_paths}
        for future in tqdm(as_completed(futures), total=len(all_file_paths), desc="缺失率筛选中"):
            fp = futures[future]
            try:
                actual_len = future.result()
                missing_rate = 1.0 - (actual_len / expected_length)
                
                if missing_rate <= threshold:
                    filtered_results.append({
                        "path": fp,
                        "actual_length": int(actual_len),
                        "missing_rate": float(missing_rate)
                    })
            except Exception as e:
                print(f"处理文件 {fp} 时出错: {e}")

    # 完善元数据
    filtered_results = parse_path_metadata(filtered_results)
    
    # 保存结果
    save_dir = os.path.dirname(FILTER_RESULT_PATH)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    with open(FILTER_RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n筛选完成！")
    print(f"符合条件的文件数: {len(filtered_results)} / {len(all_file_paths)}")
    print(f"结果已保存至: {FILTER_RESULT_PATH}")
    
    return filtered_results

if __name__ == "__main__":
    # 示例运行脚本
    from src.figs.figs_for_thesis.config import TARGET_VIBRATION_SENSORS
    from src.figs.figs_for_thesis.fig2_2_lackness_of_samples import ALL_VIBRATION_ROOT
    
    def get_all_vibration_files(root_dir, target_sensor_ids, suffix=".VIC"):
        vibration_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.upper().endswith(suffix.upper()):
                    if any(sensor_id in file for sensor_id in target_sensor_ids):
                        file_path = os.path.join(root, file)
                        vibration_files.append(file_path)
        return vibration_files

    all_vib_files = get_all_vibration_files(ALL_VIBRATION_ROOT, TARGET_VIBRATION_SENSORS)
    run_lackness_filter(all_vib_files)
