# --------------- 模块导入 ---------------
import os
import sys
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processer.io_unpacker import UNPACK
from src.config.io_config import WIND_DATA_ROOT

# 导入子模块工作流接口
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.preprocess.wind_data_io_process.workflow import run as run_wind_workflow

# --------------- 采样频率和窗口配置 ---------------
_VIB_FS = 50
_WIND_FS_CONFIG = 1
_VIB_TIME_WINDOW = 60.0
_WIND_TIME_WINDOW = 60.0
_VIB_WINDOW_SIZE = int(_VIB_FS * _VIB_TIME_WINDOW)
_WIND_WINDOW_SIZE = int(_WIND_FS_CONFIG * _WIND_TIME_WINDOW)


# --------------- 私有辅助函数 ---------------
def _get_processed_metadata(use_vib_cache=True, use_wind_cache=True, 
                            vib_force_recompute=False, wind_force_recompute=False):
    """
    内部函数：获取经过处理的元数据（集成振动和风数据工作流）
    """
    # 运行振动数据工作流
    vib_metadata = run_vib_workflow(
        use_cache=use_vib_cache,
        force_recompute=vib_force_recompute
    )
    
    # 运行风数据工作流
    wind_metadata = run_wind_workflow(
        vib_metadata = vib_metadata, 
        use_cache=use_wind_cache,
        force_recompute=wind_force_recompute
    )
    
    return vib_metadata, wind_metadata


def _segment_windows_by_duration(metadata_item, data_dict, window_duration_minutes, vib_fs=_VIB_FS, wind_fs=_WIND_FS_CONFIG):
    """
    内部函数：按指定时间长度切分原始数据为固定时长的窗口
    """
    try:
        if data_dict is None:
            return []
        
        vib_data = data_dict.get('vib')
        wind_speed = data_dict.get('wind_speed')
        wind_direction = data_dict.get('wind_direction')
        wind_angle = data_dict.get('wind_angle')
        
        if (vib_data is None or wind_speed is None or 
            wind_direction is None or wind_angle is None):
            return []
        
        vib_window_size = int(vib_fs * window_duration_minutes * 60)
        if vib_window_size <= 0 or len(vib_data) < vib_window_size:
            return []
        
        segmented_pairs = []
        freq_ratio = vib_fs / wind_fs
        
        for start_idx in range(0, len(vib_data) - vib_window_size + 1, vib_window_size):
            vib_start = start_idx
            vib_end = start_idx + vib_window_size
            
            vib_segment = vib_data[vib_start:vib_end]
            
            wind_start = int(vib_start / freq_ratio)
            wind_end = int(vib_end / freq_ratio)
            wind_end = min(wind_end, len(wind_speed))
            
            wind_speed_seg = wind_speed[wind_start:wind_end]
            wind_direction_seg = wind_direction[wind_start:wind_end]
            wind_angle_seg = wind_angle[wind_start:wind_end]
            
            if (len(vib_segment) > 0 and len(wind_speed_seg) > 0 and
                len(wind_direction_seg) > 0 and len(wind_angle_seg) > 0):
                wind_segment = (wind_speed_seg, wind_direction_seg, wind_angle_seg)
                segmented_pairs.append((vib_segment, wind_segment))
        
        return segmented_pairs
    
    except Exception:
        return []


def _segment_extreme_wind_windows(metadata_item, data_dict, vib_fs=_VIB_FS, wind_fs=_WIND_FS_CONFIG, vib_window_size=_VIB_WINDOW_SIZE):
    """
    内部函数：基于元数据中的极端振动索引，切分极端风速窗口数据
    """
    try:
        if data_dict is None:
            return []
        
        extreme_indices = metadata_item.get('extreme_rms_indices', [])
        if len(extreme_indices) == 0:
            return []
        
        vib_data = data_dict.get('vib')
        wind_speed = data_dict.get('wind_speed')
        wind_direction = data_dict.get('wind_direction')
        wind_angle = data_dict.get('wind_angle')
        
        if (vib_data is None or wind_speed is None or 
            wind_direction is None or wind_angle is None):
            return []
        
        segmented_pairs = []
        freq_ratio = vib_fs / wind_fs
        
        for extreme_idx in extreme_indices:
            vib_start = extreme_idx
            vib_end = extreme_idx + vib_window_size
            
            if vib_start < 0 or vib_end > len(vib_data):
                continue
            
            vib_segment = vib_data[vib_start:vib_end]
            
            wind_start = int(vib_start / freq_ratio)
            wind_end = int(vib_end / freq_ratio)
            wind_end = min(wind_end, len(wind_speed))
            
            wind_speed_seg = wind_speed[wind_start:wind_end]
            wind_direction_seg = wind_direction[wind_start:wind_end]
            wind_angle_seg = wind_angle[wind_start:wind_end]
            
            if (len(vib_segment) > 0 and len(wind_speed_seg) > 0 and
                len(wind_direction_seg) > 0 and len(wind_angle_seg) > 0):
                wind_segment = (wind_speed_seg, wind_direction_seg, wind_angle_seg)
                segmented_pairs.append((vib_segment, wind_segment))
        
        return segmented_pairs
    
    except Exception:
        return []


def _process_single_vib_item(args):
    """
    内部函数：处理单个振动元数据项（用于多进程）
    
    参数:
        args: 元组，包含 (vib_item, wind_sensor_id, enable_extreme_window, window_duration_minutes)
    
    返回:
        处理结果字典
    """
    try:
        vib_item, wind_sensor_id, enable_extreme_window, window_duration_minutes = args
        
        raw_data = _load_vibration_and_wind_data(vib_item, wind_sensor_id)
        
        segmented_windows = raw_data
        if raw_data is not None:
            if enable_extreme_window:
                segmented_windows = _segment_extreme_wind_windows(vib_item, raw_data)
            elif window_duration_minutes is not None:
                segmented_windows = _segment_windows_by_duration(vib_item, raw_data, window_duration_minutes)
        
        return {
            'vib_metadata': vib_item,
            'segmented_windows': segmented_windows
        }
    
    except Exception as e:
        return {
            'vib_metadata': args[0],
            'segmented_windows': []
        }


def _load_vibration_and_wind_data(vib_metadata_item, wind_sensor_id):
    """
    内部函数：从单个振动metadata解析并读取对应的振动和风原始数据
    """
    try:
        unpacker = UNPACK(init_path=False)
        vib_file_path = vib_metadata_item.get('file_path')
        
        # 读取振动数据
        vibration_data = unpacker.VIC_DATA_Unpack(vib_file_path)
        
        if vibration_data is None or len(vibration_data) == 0:
            return None
        
        vibration_data = np.array(vibration_data)
        
        # 根据vib_metadata构造风数据文件路径
        month = vib_metadata_item.get('month')
        day = vib_metadata_item.get('day')
        hour = vib_metadata_item.get('hour')
        
        wind_file_path = os.path.join(WIND_DATA_ROOT, str(month).zfill(2), str(day).zfill(2), 
                                      f"{wind_sensor_id}_{str(hour).zfill(2)}")
        
        # 寻找对应的风数据文件（可能有不同的扩展名）
        wind_dir = os.path.dirname(wind_file_path)
        wind_files = []
        
        if os.path.exists(wind_dir):
            for fname in os.listdir(wind_dir):
                if fname.startswith(os.path.basename(wind_file_path)):
                    wind_files.append(os.path.join(wind_dir, fname))
        
        if len(wind_files) == 0:
            return None
        
        # 选择第一个匹配的文件
        wind_file_path = wind_files[0]
        
        # 读取风数据
        wind_data = unpacker.Wind_Data_Unpack(wind_file_path)
        
        # 返回包含所有原始数据的字典
        return {
            'vib': vibration_data,
            'wind_speed': np.array(wind_data[0]),
            'wind_direction': np.array(wind_data[1]),
            'wind_angle': np.array(wind_data[2])
        }
    
    except Exception:
        return None


# --------------- 主入口函数 ---------------
def get_data_pairs(wind_sensor_id, vib_sensor_id=None, use_multiprocess=True, 
                   enable_extreme_window=False, window_duration_minutes=None,
                   use_vib_cache=True, use_wind_cache=True):
    """
    批量提取振动和风数据对的主入口函数
    
    该函数一次性集成了子模块工作流和数据处理，无需预先加载元数据。
    
    参数:
        wind_sensor_id: 风传感器ID字符串（如 'ST-UAN-G04-001-01'）
        vib_sensor_id: 振动传感器ID过滤（字符串或None）
                       若指定则仅处理该传感器的数据
        use_multiprocess: 是否使用多进程处理（默认False）
        enable_extreme_window: 是否进行极端窗口筛选（默认False）
        window_duration_minutes: 窗口时长（分钟），当enable_extreme_window=False时有效
                                 若为None则返回完整原始数据不切分
        use_vib_cache: 是否使用振动数据缓存（默认True）
        use_wind_cache: 是否使用风数据缓存（默认True）
    
    返回:
        list: 数据对列表，每一项为字典，包含：
            {
                'vib_metadata': 对应的振动元数据,
                'segment_config': 切分配置字典，包含：
                    - 'vib_sensor_id': 振动传感器ID
                    - 'wind_sensor_id': 风传感器ID
                    - 'enable_extreme_window': 是否使用极端窗口筛选
                    - 'window_duration_minutes': 窗口时长(仅在常规切分时有效)
                    - 'vib_fs': 振动采样频率
                    - 'wind_fs': 风速采样频率
                'segmented_windows': 切分后的数据对列表
                    [(vib_segment, (wind_speed, wind_direction, wind_angle)), ...]
            }
        
        若无有效数据，返回空列表
    """
    print("="*80)
    print(" "*15 + "批量提取振动和风数据对（集成工作流）")
    print("="*80)
    
    # Step 1: 获取处理后的元数据
    print("\n[Step 1] 获取处理后的元数据...")
    print("-"*80)
    vib_metadata, wind_metadata = _get_processed_metadata(
        use_vib_cache=use_vib_cache,
        use_wind_cache=use_wind_cache
    )
    print(f"✓ 获取到 {len(vib_metadata)} 条振动元数据")
    print(f"✓ 获取到 {len(wind_metadata)} 条风数据元数据")
    
    # Step 2: 传感器筛选
    filtered_metadata_list = vib_metadata
    if vib_sensor_id is not None:
        filtered_metadata_list = [item for item in vib_metadata 
                                  if item.get('sensor_id') == vib_sensor_id]
        print(f"\n[Step 2] 传感器筛选")
        print("-"*80)
        print(f"  - 原始元数据数: {len(vib_metadata)}")
        print(f"  - 振动传感器ID过滤: {vib_sensor_id}")
        print(f"  - 风传感器ID: {wind_sensor_id}")
        print(f"  - 筛选后元数据数: {len(filtered_metadata_list)}")
        
        if len(filtered_metadata_list) == 0:
            print(f"\n  警告: 未找到与 '{vib_sensor_id}' 匹配的元数据")
            print("\n" + "="*80 + "\n")
            return []
    
    # Step 3: 数据处理配置
    print(f"\n[Step 3] 数据处理配置")
    print("-"*80)
    print(f"  - 待处理元数据数: {len(filtered_metadata_list)}")
    print(f"  - 多进程模式: {'启用' if use_multiprocess else '禁用'}")
    print(f"  - 极端窗口筛选: {'启用' if enable_extreme_window else '禁用'}")
    if not enable_extreme_window and window_duration_minutes is not None:
        print(f"  - 常规窗口切分: 启用 ({window_duration_minutes} 分钟)")
    else:
        print(f"  - 常规窗口切分: 禁用")
    
    data_pairs = []
    
    print(f"\n[Step 4] 处理进度")
    print("-"*80)
    
    # 构建segment_config
    segment_config = {
        'vib_sensor_id': vib_sensor_id,
        'wind_sensor_id': wind_sensor_id,
        'enable_extreme_window': enable_extreme_window,
        'window_duration_minutes': window_duration_minutes,
        'vib_fs': _VIB_FS,
        'wind_fs': _WIND_FS_CONFIG
    }
    
    # 单进程处理
    if not use_multiprocess:
        for idx, vib_item in enumerate(tqdm(filtered_metadata_list, desc="提取数据对")):
            raw_data = _load_vibration_and_wind_data(vib_item, wind_sensor_id)
            
            segmented_windows = raw_data
            if raw_data is not None:
                if enable_extreme_window:
                    segmented_windows = _segment_extreme_wind_windows(vib_item, raw_data)
                elif window_duration_minutes is not None:
                    segmented_windows = _segment_windows_by_duration(vib_item, raw_data, window_duration_minutes)
            
            data_pairs.append({
                'vib_metadata': vib_item,
                'segment_config': segment_config,
                'segmented_windows': segmented_windows
            })
    else:
        # 多进程处理
        max_workers = min(4, len(filtered_metadata_list))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 构建任务参数列表
            task_args = [
                (vib_item, wind_sensor_id, enable_extreme_window, window_duration_minutes)
                for vib_item in filtered_metadata_list
            ]
            
            # 提交所有任务
            futures = {
                executor.submit(_process_single_vib_item, args): args[0]
                for args in task_args
            }
            
            # 处理完成的任务
            for future in tqdm(as_completed(futures), total=len(futures), desc="提取数据对（多进程）"):
                try:
                    result = future.result()
                    data_pairs.append({
                        'vib_metadata': result['vib_metadata'],
                        'segment_config': segment_config,
                        'segmented_windows': result['segmented_windows']
                    })
                except Exception as e:
                    vib_item = futures[future]
                    data_pairs.append({
                        'vib_metadata': vib_item,
                        'segment_config': segment_config,
                        'segmented_windows': []
                    })
    
    # 统计结果
    successful_segmented = sum(1 for pair in data_pairs if pair.get('segmented_windows') and len(pair['segmented_windows']) > 0)
    total_windows = sum(len(pair['segmented_windows']) for pair in data_pairs if pair.get('segmented_windows'))
    failed_count = sum(1 for pair in data_pairs if not pair.get('segmented_windows') or len(pair['segmented_windows']) == 0)
    
    print(f"\n[处理结果统计]")
    print("-"*80)
    print(f"✓ 总处理数: {len(data_pairs)}")
    print(f"✓ 成功切分: {successful_segmented}")
    print(f"✗ 切分失败或无数据: {failed_count}")
    if total_windows > 0:
        print(f"✓ 总窗口数: {total_windows}")
    
    if len(data_pairs) > 0:
        success_rate = (successful_segmented / len(data_pairs) * 100) if successful_segmented > 0 else 0
        print(f"✓ 成功率: {success_rate:.2f}%")
    
    print("\n" + "="*80 + "\n")
    
    return data_pairs
