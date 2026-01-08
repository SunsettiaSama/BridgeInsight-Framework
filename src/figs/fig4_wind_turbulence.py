# --------------- 模块导入（去重+分类整理，补充字体相关导入）---------------
# 自定义库
from ..visualize_tools.utils import ChartApp, PlotLib
from ..data_processer.data_processer_V0 import UNPACK, DataManager

# 第三方库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties  # 新增：导入字体管理类

# --------------- 全局绘图配置（统一设置，优化字体配置）---------------
plt.style.use('default')
font_size = 12  # 统一字体大小，便于维护

# 1. 调整字体优先级：英文优先用Times New Roman，中文用SimHei
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 2. 定义中英文字体对象（便于单独指定文本字体）
ENG_FONT = FontProperties(family='Times New Roman', size=font_size)
CN_FONT = FontProperties(
    family='SimHei',
    size=font_size,
    # Linux/Mac系统需手动指定SimHei字体路径，Windows无需修改
    # fname='/usr/share/fonts/truetype/simhei/SimHei.ttf'
)

CMAP_NORMAL = plt.cm.gist_yarg 
CMAP_VIV = plt.cm.gist_yarg
# --------------- 工具函数（单独实现紊流度计算+提取重复逻辑）---------------
def calculate_turbulence_intensity(wind_speed_group):
    """
    【单独逻辑实现】紊流度TI计算：TI = (风速标准差 / 平均风速) × 100%
    处理异常情况：样本数不足2（无法算标准差）、平均风速接近0（避免除零）
    Args:
        wind_speed_group: 某一分箱内的风速样本列表（np.array）
    Returns:
        ti_value: 紊流度值（百分比形式，异常情况返回0）
    """
    # 异常值处理
    if len(wind_speed_group) < 2:
        return 0.0
    u_mean = np.mean(wind_speed_group)
    if u_mean <= 1e-6:  # 避免除以零
        return 0.0
    
    # 正常计算紊流度
    u_std = np.std(wind_speed_group)
    ti_value = (u_std / u_mean) * 100  # 转换为百分比
    return round(ti_value, 2)  # 保留2位小数，提升可读性

def wind_data_cleaning(wind_velocities, wind_directions, wind_angles=None):
    """
    风数据清洗：过滤无效风速（<0.1m/s）+ 格式转换
    Args:
        wind_velocities: 原始风速数据
        wind_directions: 原始风向数据
        wind_angles: 原始风角度数据（可选）
    Returns:
        清洗后的风速、风向、风角度（若传入）
    """
    # 转换为numpy数组
    wind_velocities = np.array(wind_velocities)
    wind_directions = np.array(wind_directions)
    if wind_angles is not None:
        wind_angles = np.array(wind_angles)
    
    # 过滤无效风速
    valid_mask = wind_velocities > 0.1
    wind_velocities = wind_velocities[valid_mask]
    wind_directions = wind_directions[valid_mask]
    if wind_angles is not None:
        wind_angles = wind_angles[valid_mask]
    
    return (wind_velocities, wind_directions, wind_angles) if wind_angles is not None else (wind_velocities, wind_directions)

def correct_wind_direction(wind_directions, correction_val):
    """
    风向修正：统一修正逻辑，归一化到0-360度
    Args:
        wind_directions: 原始风向数据（np.array）
        correction_val: 修正值（如360、180）
    Returns:
        修正后的风向数据
    """
    wind_directions = correction_val - wind_directions  # 方向修正
    wind_directions = np.mod(wind_directions, 360)      # 归一化到0-360度
    return wind_directions

def plot_wind_rose(theta, counts, ti_values, axis_of_bridge, bin_step, cmap):
    """
    绘制风玫瑰图（紊流度颜色映射，count归一化表示占比）
    Args:
        theta: 分箱起始角度（弧度）
        counts: 每个分箱的原始样本数
        ti_values: 每个分箱的紊流度（用于颜色映射）
        axis_of_bridge: 桥轴线角度（度）
        bin_step: 分箱步长（度）
    Returns:
        fig, ax: 绘图对象
    """
    # 创建极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # ------------- 修改4：count归一化，转为样本占比百分比 -------------
    counts = np.array(counts)
    total_count = counts.sum()
    if total_count > 0:
        counts_normalized = (counts / total_count) * 100  # 归一化为百分比
    else:
        counts_normalized = counts  # 避免除以零
    # ----------------------------------------------------------------

    # 绘制柱状图（归一化后的占比作为半径，表示样本百分比）
    bars = ax.bar(
        theta, counts_normalized, 
        width=np.deg2rad(bin_step), 
        bottom=0.0, 
        alpha=0.8, 
        align='edge'
    )

    # 设置极坐标轴方向（0度朝北，顺时针）
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # ------------- 修改1：颜色映射改为紊流度，对齐色系 -------------
    # 紊流度颜色归一化配置
    norm = plt.Normalize(min(ti_values), max(ti_values))
    # 为柱体分配对应紊流度的颜色
    for bar, ti in zip(bars, ti_values):
        bar.set_facecolor(cmap(norm(ti)))
    
    # 颜色条配置：标注为“紊流度”，缩小放在右侧，避免喧宾夺主
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, 
        orientation='vertical', 
        label='紊流度(%)', 
        pad=0.05,  # 缩小与图的间距
        shrink=0.8  # 缩小颜色条尺寸
    )
    cbar.set_label('紊流度(%)', fontproperties=CN_FONT)  # 中文标签用SimHei
    cbar.ax.tick_params(labelsize=font_size)
    # ----------------------------------------------------------------

    # 图像美化
    # 1. 地理坐标标签（英文，显式指定Times New Roman）
    x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticklabels(x_ticks, fontproperties=ENG_FONT)

    # ------------- 修改3：桥轴线改成中文 -------------
    y_max = np.max(counts_normalized) if len(counts_normalized) > 0 else 1
    bridge_theta1 = np.deg2rad(axis_of_bridge)
    bridge_theta2 = np.deg2rad(axis_of_bridge + 180)
    ax.plot([bridge_theta1, bridge_theta1], [0, y_max * 1.1], color='red', linestyle='--')
    ax.plot([bridge_theta2, bridge_theta2], [0, y_max * 1.1], color='red', linestyle='--')
    ax.annotate('桥轴线', xy=(bridge_theta1, y_max * 0.9), 
                ha='center', va='bottom', fontproperties=CN_FONT)
    # ----------------------------------------------------------------

    # ------------- 修改4：y轴刻度显示百分比，对应归一化后的count -------------
    # 调整y轴刻度间隔，适配百分比范围

    # 优化1：细粒度间隔计算（按y_max的1/10切分，最小间隔2，避免过小）
    y_tick_interval = max(2, round(y_max / 5))  # 切分更细，最小间隔2（可改为1进一步加密）

    # 优化2：绑定定位器，按细粒度间隔生成刻度
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))

    ax.set_ylim(0, y_max * 1.1)
    # 刻度标签显示百分比符号，用Times New Roman
    ax.set_yticklabels([f"{int(num)}%" for num in ax.get_yticks()], fontproperties=ENG_FONT)
    ax.yaxis.set_label_coords(-0.1, 1.1)
    ax.set_rlabel_position(270)  # 180+90，左侧显示刻度
    # ----------------------------------------------------------------

    # ------------- 修改2：去掉所有标题及顶部内容 -------------
    # 删除原有标题相关代码，不设置任何顶部文本
    # ----------------------------------------------------------------

    return fig, ax

# --------------- 主函数：VIV风玫瑰图绘制 ---------------
def TI_Wind_Rose_Map():
    # 核心参数配置（统一管理）
    config = {
        "interval_nums": 36,                # 角度分箱数量
        "axis_of_bridge": 10.6,             # 桥轴线角度（度）
        "time_interval": 1,                 # VIV时段截取长度（分钟）
        "fs": 1,                            # 风速采样频率（Hz）
        "target_viv_sensor_id": 'ST-VIC-C18-102-01',  # VIV传感器ID
        "VIV_excel_path": r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\Img\VIV.xlsx',
        "wind_dirs": [r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September"],
        "sensor_ids": ["跨中桥面上游"]
    }

    # 目标传感器配置
    target_sensors = ['ST-UAN-G04-001-01']  # 跨中桥面上游
    wind_dir_correction = {'ST-UAN-G04-001-01': 360}  # 风向修正映射

    # 初始化工具类
    ploter = PlotLib()
    manager = DataManager()
    unpacker = UNPACK()
    figs = []  # 存储玫瑰图对象
    bin_step = int(360 / config["interval_nums"])  # 分箱步长
    bins = np.arange(0, 360 + bin_step, bin_step)  # 角度分箱区间

    # --------------- 第一部分：VIV发生时段的风玫瑰图 ---------------
    # 读取VIV数据表格
    df_viv = pd.read_excel(config["VIV_excel_path"])

    for idx, sensor_id in enumerate(target_sensors):
        viv_wind_dir_deg = np.array([])  # 风向（度，未修正）
        viv_wind_vel = np.array([])      # 风速（m/s）

        # 遍历VIV记录，匹配对应风数据
        for row in df_viv.values:
            viv_path, viv_time, plane = row
            # 匹配风传感器路径
            wind_paths = unpacker.VIC_Path_2_WindPath(VICpath=viv_path, wind_sensor_ids=[sensor_id])
            if len(wind_paths) != 1:
                continue

            # 解析风数据
            wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(wind_paths[0])
            wind_velocity = np.array(wind_velocity)
            wind_direction = np.array(wind_direction)
            
            # 截取VIV发生时段数据（防止索引越界）
            start_idx = viv_time
            end_idx = viv_time + config["time_interval"] * 60 * config["fs"]
            end_idx = min(end_idx, len(wind_velocity))
            if start_idx >= end_idx:
                continue
            
            # 数据清洗：过滤无效风速
            vel_slice = wind_velocity[start_idx:end_idx]
            dir_slice = wind_direction[start_idx:end_idx]
            valid_mask = vel_slice > 0.1
            vel_valid = vel_slice[valid_mask]
            dir_valid = dir_slice[valid_mask]
            
            if len(vel_valid) == 0:
                continue
            
            # 拼接数据
            viv_wind_dir_deg = np.hstack((viv_wind_dir_deg, dir_valid))
            viv_wind_vel = np.hstack((viv_wind_vel, vel_valid))

        # 检查有效数据
        if len(viv_wind_dir_deg) == 0:
            print(f"警告：传感器 {sensor_id} 无VIV发生时的有效风数据")
            continue

        # 风向修正
        correction_val = wind_dir_correction[sensor_id]
        viv_wind_dir_deg = correct_wind_direction(viv_wind_dir_deg, correction_val)

        # 角度分箱
        digitized = np.digitize(viv_wind_dir_deg, bins)
        grouped_speeds = [viv_wind_vel[digitized == i] for i in range(1, len(bins))]

        # 统计量计算
        counts = [len(speeds) for speeds in grouped_speeds]  # 原始样本数
        ti_values = [calculate_turbulence_intensity(speeds) for speeds in grouped_speeds]  # 紊流度

        # 绘制风玫瑰图（移除sensor_name参数，无需标题）
        theta = np.deg2rad(bins[:-1])
        fig, _ = plot_wind_rose(
            theta=theta,
            counts=counts,
            ti_values=ti_values,
            axis_of_bridge=config["axis_of_bridge"],
            bin_step=bin_step, 
            cmap = CMAP_VIV
        )
        figs.append(fig)

    # --------------- 第二部分：整体风数据的风玫瑰图 ---------------
    for wind_dir in config["wind_dirs"]:
        for sensor_id in config["sensor_ids"]:
            # 获取风数据
            wind_velocities, wind_directions, wind_angles = manager.get_wind_data_from_root(
                wind_dir, mode="interval", sensor_id=sensor_id
            )

            # 数据清洗
            wind_velocities, wind_directions, wind_angles = wind_data_cleaning(wind_velocities, wind_directions, wind_angles)
            if len(wind_velocities) == 0:
                print(f"警告：{sensor_id} 在 {wind_dir} 下无有效风数据")
                continue

            # 风向修正
            wind_directions = correct_wind_direction(wind_directions, 360)

            # 角度分箱
            digitized = np.digitize(wind_directions, bins)
            grouped_speeds = [wind_velocities[digitized == i] for i in range(1, len(bins))]

            # 统计量计算
            counts = [len(speeds) for speeds in grouped_speeds]  # 原始样本数
            ti_values = [calculate_turbulence_intensity(speeds) for speeds in grouped_speeds]  # 紊流度

            # 绘制风玫瑰图
            theta = np.deg2rad(bins[:-1])
            fig, _ = plot_wind_rose(
                theta=theta,
                counts=counts,
                ti_values=ti_values,
                axis_of_bridge=config["axis_of_bridge"],
                bin_step=bin_step, 
                cmap = CMAP_NORMAL
            )
            figs.append(fig)

    # 汇总图表并展示
    ploter.figs.extend(figs)
    plt.close('all')  # 释放内存
    ploter.show()