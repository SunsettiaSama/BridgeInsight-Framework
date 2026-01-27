import os
import sys

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从配置文件导入常量
from src.config.data_processer.statistics.vibration_io_process.config import (
    ALL_VIBRATION_ROOT,
    VIBRATION_FILE_SUFFIX,
    TARGET_VIBRATION_SENSORS
)


def get_all_vibration_files(root_dir=ALL_VIBRATION_ROOT, 
                             target_sensor_ids=TARGET_VIBRATION_SENSORS, 
                             suffix=VIBRATION_FILE_SUFFIX):
    """
    获取指定目录下所有符合条件的振动文件路径
    
    参数:
        root_dir: 数据根目录
        target_sensor_ids: 目标传感器ID列表
        suffix: 文件后缀名（默认 .VIC）
    
    返回:
        vibration_files: 文件路径列表
    """
    vibration_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.upper().endswith(suffix.upper()):
                if any(sensor_id in file for sensor_id in target_sensor_ids):
                    file_path = os.path.join(root, file)
                    vibration_files.append(file_path)
    return vibration_files


if __name__ == "__main__":
    # 测试接口
    all_files = get_all_vibration_files()
    print(f"共获取所有振动文件数量：{len(all_files)}")
    
    # 显示前5个文件路径作为示例
    if all_files:
        print("\n文件路径示例（前5个）：")
        for i, fp in enumerate(all_files[:5], 1):
            print(f"{i}. {fp}")
