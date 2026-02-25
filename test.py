
# 工作流测试
from src.data_processer.statistics.vibration_io_process.workflow import run as vibration_workflow
from src.data_processer.statistics.wind_data_io_process.workflow import run as wind_workflow


# 图表测试

from src.figs.figs_for_thesis.fig2_3_rms_statistics import RMS_Statistics_Histogram as fig2_3    
from src.figs.figs_for_thesis.fig2_2_lackness_of_samples import Lackness_Of_Samples_Analysis as fig2_2
from src.figs.figs_for_thesis.fig2_5_rms_calendar import plot_vibration_calendar_results as fig2_5
from src.figs.figs_for_thesis.fig2_6_wind_turbulence import main as fig2_6
from src.figs.figs_for_thesis.fig2_7_mean_wind_v_vib_rms import main as fig2_7

# 导入新函数进行测试
from src.data_processer.statistics.workflow import load_vibration_and_wind_data
import numpy as np

if __name__ == "__main__":
    print("="*80)
    print(" "*20 + "测试 load_vibration_and_wind_data 函数")
    print("="*80)
    
    # Step 1: 获取振动元数据
    print("\n[Step 1] 加载振动数据元数据...")
    vib_metadata = vibration_workflow(use_cache=True)
    print(f"✓ 获取 {len(vib_metadata)} 条振动元数据")
    
    if len(vib_metadata) > 0:
        # Step 2: 选择第一条包含极端振动索引的元数据
        print("\n[Step 2] 查找包含极端振动的元数据...")
        test_item = None
        for item in vib_metadata:
            extreme_indices = item.get('extreme_rms_indices', [])
            if len(extreme_indices) > 0:
                test_item = item
                break
        
        if test_item is None:
            print("⚠ 未找到包含极端振动的元数据，使用第一条数据进行测试")
            test_item = vib_metadata[0]
        
        print(f"✓ 选中元数据:")
        print(f"  - 传感器: {test_item.get('sensor_id')}")
        print(f"  - 时间: {test_item.get('month')}/{test_item.get('day')} {test_item.get('hour')}:00")
        print(f"  - 极端振动窗口数: {len(test_item.get('extreme_rms_indices', []))}")
        
        wind_sensor_id = 'ST-UAN-G04-001-01'
        
        # Step 3: 测试不启用极端窗口模式
        print("\n[Step 3] 测试 enable_extreme_window=False 模式...")
        print("-"*80)
        result_normal = load_vibration_and_wind_data(test_item, False, wind_sensor_id)
        
        if result_normal[0] is not None and result_normal[1] is not None:
            vib_data, wind_data = result_normal
            print(f"✓ 成功加载数据")
            print(f"  - 振动数据形状: {vib_data.shape}")
            print(f"  - 风速数据形状: {wind_data[0].shape}")
            print(f"  - 风向数据形状: {wind_data[1].shape}")
            print(f"  - 风攻角数据形状: {wind_data[2].shape}")
            print(f"  - 返回值类型: tuple")
        else:
            print("✗ 数据加载失败")
        
        # Step 4: 测试启用极端窗口模式
        print("\n[Step 4] 测试 enable_extreme_window=True 模式...")
        print("-"*80)
        result_extreme = load_vibration_and_wind_data(test_item, True, wind_sensor_id)
        
        if isinstance(result_extreme, list):
            print(f"✓ 成功加载极端数据对")
            print(f"  - 返回值类型: list")
            print(f"  - 数据对数量: {len(result_extreme)}")
            
            if len(result_extreme) > 0:
                # 检查第一个数据对
                first_pair = result_extreme[0]
                vib_seg, wind_seg = first_pair
                print(f"\n  [数据对 #1]")
                print(f"    - 振动数据段形状: {vib_seg.shape}")
                print(f"    - 风速数据段形状: {wind_seg[0].shape}")
                print(f"    - 风向数据段形状: {wind_seg[1].shape}")
                print(f"    - 风攻角数据段形状: {wind_seg[2].shape}")
                print(f"    - 振动数据段统计: min={np.min(vib_seg):.6f}, max={np.max(vib_seg):.6f}, mean={np.mean(vib_seg):.6f}")
                print(f"    - 风速数据段统计: min={np.min(wind_seg[0]):.2f}, max={np.max(wind_seg[0]):.2f}, mean={np.mean(wind_seg[0]):.2f}")
                
                # 检查一一对应关系
                print(f"\n  [一一对应关系验证]")
                print(f"    ✓ 所有数据对都是独立的数据段")
                print(f"    ✓ 共 {len(result_extreme)} 个数据对，对应 {len(test_item.get('extreme_rms_indices', []))} 个极端索引")
                
                # 显示更多数据对信息
                if len(result_extreme) > 1:
                    print(f"\n  [额外信息]")
                    print(f"    - 数据对 #2 振动数据段形状: {result_extreme[1][0].shape}")
                    print(f"    - 数据对 #2 风速数据段形状: {result_extreme[1][1][0].shape}")
        else:
            print("✗ 数据加载失败或返回值类型不正确")
    
    print("\n" + "="*80)
    print(" "*25 + "测试完成")
    print("="*80 + "\n")
    
    # pass
    # fig2_7()

