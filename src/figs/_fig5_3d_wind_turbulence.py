
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from collections import deque
import os
import struct
import re
import pandas as pd
import matplotlib.animation as animation
from pathlib import Path
from scipy.io import savemat, loadmat
from tkinter import Tk, Label, Button, Entry, StringVar, filedialog, messagebox
from PIL import Image, ImageTk
import copy
import random
from sklearn.cluster import KMeans, DBSCAN

import datetime


plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['font.size'] = 22


from ..data_processer.data_processer_V0 import *
from ..visualize_tools.utils import *


def Wind_Turbulence_wVelocity_wDirection():
    import matplotlib
    import tkinter as tk
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from scipy.optimize import curve_fit

    # ===================== 颜色配置字典（统一管理所有配色/样式） =====================
    COLOR_CONFIG = {
        'cmap_main': 'coolwarm',          # 主配色方案
        'viv_scatter': 'pink',            # VIV样本散点颜色
        'max_freq_contour': 'darkred',    # 最大频数区间轮廓颜色
        'max_freq_surface': 'red',        # 最大频数区间曲面片颜色
        'tk_background': '#f0f0f0',     # TKinter窗口背景色
        'max_freq_surface_alpha': 0.2,    # 曲面片透明度（样式关联）
        'max_freq_contour_linewidth': 2   # 轮廓线宽（样式关联）
    }

    # ===================== 1. 对齐玫瑰图：核心参数统一 =====================
    interval_nums = 36  # 与玫瑰图一致的区间数量

    # 传感器列表：与玫瑰图完全一致（保留北索塔塔顶+跨中上游，可按需注释）
    target_sensors = [
        'ST-UAN-T01-003-01',  # 北索塔塔顶
        'ST-UAN-G04-001-01'   # 跨中桥面上游
    ]
    sensor_names = [
        'Wind Turbulence (Top Of North Pylon)',
        'Wind Turbulence (Upstream of Mid-Span)'
    ]
    # 风向修正规则：严格对齐玫瑰图（区分传感器）
    wind_dir_correction = {
        'ST-UAN-T01-003-01': 180,  # 北索塔塔顶：180 - 原始风向
        'ST-UAN-G04-001-01': 360   # 跨中桥面上游：360 - 原始风向
    }

    # 路径配置：与玫瑰图完全对齐
    VIV_excel_path = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\Img\VIV.xlsx'
    wind_data_root = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\UAN\\'
    time_interval = 1  # 分钟，与玫瑰图一致的时间截取长度
    fs = 1  # 采样频率1Hz，与玫瑰图一致
    target_viv_sensor_id = 'ST-VIC-C18-102-01'  # VIV传感器ID（与玫瑰图一致）
    valid_wind_speed_thresh = 0.1  # 有效风速阈值（与玫瑰图一致：>0.1m/s）

    # ===================== 2. 初始化工具类与数据容器 =====================
    unpacker = UNPACK()
    ploter = PlotLib()
    figs = []  # 存储图表（3D紊流度图）

    # ===================== 3. 处理「非VIV」风场数据（全量风数据） =====================
    # 按传感器分组存储：修正后的风向、平均风速、紊流度
    normal_data = {
        sensor: {'dir': np.array([]), 'vel': np.array([]), 'ti': np.array([])} 
        for sensor in target_sensors
    }

    # 遍历每个目标传感器
    for sensor_id in target_sensors:
        # 匹配该传感器的所有风数据文件路径
        all_wind_paths = unpacker.File_Read_Paths(wind_data_root)
        sensor_wind_paths = unpacker.File_Match_Sensor_Path(pattern=sensor_id, paths=all_wind_paths)
        
        if len(sensor_wind_paths) == 0:
            print(f"警告：传感器{sensor_id}未匹配到风数据文件")
            continue

        # 遍历每个风数据文件
        for file_path in sensor_wind_paths:
            # 按1分钟切片解析风数据（与玫瑰图时间粒度一致）
            wind_velocity_slice, wind_direction_slice, _ = unpacker.File_Detach_Data(
                file_path, time_interval=time_interval, mode='UAN'
            )

            # 遍历每个1分钟切片
            for vel_slice, dir_slice in zip(wind_velocity_slice, wind_direction_slice):
                vel_arr = np.array(vel_slice)
                dir_arr = np.array(dir_slice)

                # 过滤无效风速（与玫瑰图一致：>0.1m/s）
                valid_mask = vel_arr > valid_wind_speed_thresh
                if not np.any(valid_mask):
                    continue
                vel_valid = vel_arr[valid_mask]
                dir_valid = dir_arr[valid_mask]

                # 计算该时段的平均风速、脉动风速均方根、紊流度（TI）
                vel_mean = np.mean(vel_valid)
                vel_rms = np.sqrt(np.mean((vel_valid - vel_mean) ** 2))
                ti = vel_rms / vel_mean if vel_mean != 0 else 0  # 避免除零

                # 风向修正（与玫瑰图一致：先修正，再归一化到0-360度）
                # 修正：调整计算顺序（原始风向 - 修正值），避免风向反转
                correction_val = wind_dir_correction[sensor_id]
                dir_corrected = correction_val - dir_valid  # 符号正确（修正值减原始）
                dir_corrected = np.mod(dir_corrected, 360)
                dir_mean = np.mean(dir_corrected)  # 修正后的时段平均风向

                # 拼接数据
                normal_data[sensor_id]['dir'] = np.hstack((normal_data[sensor_id]['dir'], dir_mean))
                normal_data[sensor_id]['vel'] = np.hstack((normal_data[sensor_id]['vel'], vel_mean))
                normal_data[sensor_id]['ti'] = np.hstack((normal_data[sensor_id]['ti'], ti))

    # ===================== 4. 处理「VIV发生时」的风场数据（对齐玫瑰图逻辑） =====================
    # 按传感器分组存储VIV时段数据
    viv_data = {
        sensor: {'dir': np.array([]), 'vel': np.array([]), 'ti': np.array([])} 
        for sensor in target_sensors
    }

    # 读取VIV数据表格（与玫瑰图路径一致）
    df_viv = pd.read_excel(VIV_excel_path)

    # 遍历每个目标传感器
    for sensor_id in target_sensors:
        # 遍历每条VIV记录
        for row in df_viv.values:
            viv_path, viv_time, plane = row  # 与玫瑰图VIV数据解析逻辑一致
            # 匹配VIV对应的风传感器路径（与玫瑰图一致）
            wind_paths = unpacker.VIC_Path_2_WindPath(VICpath=viv_path, wind_sensor_ids=[sensor_id])
            if len(wind_paths) != 1:
                continue

            # 解析风数据（与玫瑰图时间截取逻辑一致：VIV发生时段1min内）
            wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(wind_paths[0])
            wind_velocity = np.array(wind_velocity)
            wind_direction = np.array(wind_direction)

            # 截取VIV时段数据（防止索引越界，与玫瑰图一致）
            start_idx = viv_time
            end_idx = viv_time + time_interval * 60 * fs
            end_idx = min(end_idx, len(wind_velocity))
            if start_idx >= end_idx:
                continue

            # 提取时段数据并过滤无效风速（与玫瑰图一致）
            vel_slice = wind_velocity[start_idx:end_idx]
            dir_slice = wind_direction[start_idx:end_idx]
            valid_mask = vel_slice > valid_wind_speed_thresh
            if not np.any(valid_mask):
                continue
            vel_valid = vel_slice[valid_mask]
            dir_valid = dir_slice[valid_mask]

            # 计算紊流度（TI）
            vel_mean = np.mean(vel_valid)
            vel_rms = np.sqrt(np.mean((vel_valid - vel_mean) ** 2))
            ti = vel_rms / vel_mean if vel_mean != 0 else 0

            # 风向修正（与玫瑰图完全一致）
            # 修正：调整计算顺序（原始风向 - 修正值）
            correction_val = wind_dir_correction[sensor_id]
            dir_corrected = correction_val - dir_valid
            dir_corrected = np.mod(dir_corrected, 360)
            dir_mean = np.mean(dir_corrected)

            # 拼接VIV时段数据
            viv_data[sensor_id]['dir'] = np.hstack((viv_data[sensor_id]['dir'], dir_mean))
            viv_data[sensor_id]['vel'] = np.hstack((viv_data[sensor_id]['vel'], vel_mean))
            viv_data[sensor_id]['ti'] = np.hstack((viv_data[sensor_id]['ti'], ti))

    # ===================== 6. 可视化：3D紊流度散点图（柱坐标/极坐标，保留拟合曲线） =====================
    for idx, sensor_id in enumerate(target_sensors):
        if len(normal_data[sensor_id]['dir']) == 0:
            print(f"警告：传感器{sensor_id}无有效非VIV风数据，跳过3D绘图")
            continue
        
        # 【核心修正】保留柱坐标：theta为修正后风向转弧度，rho为风速，z为紊流度
        theta_rad = np.deg2rad(normal_data[sensor_id]['dir'])  # 柱坐标-角度（弧度）
        rho = normal_data[sensor_id]['vel']                    # 柱坐标-半径（风速）
        z = normal_data[sensor_id]['ti']                       # 柱坐标-高度（紊流度）

        # 3D散点图（普通样本）- 恢复柱坐标入参（theta/rho/z）
        fig, ax = ploter.scatter_3d(
            theta=theta_rad,    # 柱坐标角度（弧度）
            rho=rho,            # 柱坐标半径（风速）
            z=z,                # 柱坐标高度（紊流度）
            legend='Normal Samples',
            title=f'{sensor_names[idx]} - 3D Turbulence Distribution'
        )

        # 叠加VIV样本（柱坐标）
        if len(viv_data[sensor_id]['dir']) > 0:
            theta_viv_rad = np.deg2rad(viv_data[sensor_id]['dir'])  # VIV样本角度（弧度）
            rho_viv = viv_data[sensor_id]['vel']                    # VIV样本风速
            z_viv = viv_data[sensor_id]['ti']                       # VIV样本紊流度

            fig, ax = ploter.scatter_3d(
                theta=theta_viv_rad,
                rho=rho_viv,
                z=z_viv,
                fig=fig, ax=ax,
                color=COLOR_CONFIG['viv_scatter'],
                legend='VIV Samples',
                s=30,
                marker='^'
            )

        # 替换原拟合曲线逻辑：计算VIV风向频数最大区间，并在z=0平面绘制扇形标记
        if len(viv_data[sensor_id]['dir']) > 0:
            # 过滤有效VIV风向数据
            viv_dir_valid = viv_data[sensor_id]['dir'][np.isfinite(viv_data[sensor_id]['dir'])]
            if len(viv_dir_valid) == 0:
                print(f"警告：传感器{sensor_id}无有效VIV风向数据，跳过扇形标记")
                continue

            # 步骤1：按玫瑰图分箱规则（36区间）对VIV风向分箱，计算各区间频数
            bin_step = 360 / interval_nums  # 每个区间的角度宽度（10°）
            bins = np.arange(0, 360 + bin_step, bin_step)  # 分箱边界：0,10,20,...,360
            digitized = np.digitize(viv_dir_valid, bins)  # 匹配每个风向的区间索引
            counts = np.bincount(digitized, minlength=len(bins))[:len(bins)-1]  # 各区间频数

            # 步骤2：找到频数最大的区间
            max_count_idx = np.argmax(counts)
            # 该区间的角度范围（起始/结束）
            theta_start = bins[max_count_idx]  # 区间起始角度（度）
            theta_end = bins[max_count_idx + 1]  # 区间结束角度（度）
            # 转换为弧度（适配绘图）
            theta_start_rad = np.deg2rad(theta_start)
            theta_end_rad = np.deg2rad(theta_end)

            # 步骤3：在z=0平面绘制「频数最大区间」的轮廓+曲面片（替代填充，避免3D报错）
            # 扇形半径：取VIV最大风速的1.1倍，保证覆盖数据范围
            fan_radius = np.max(viv_data[sensor_id]['vel'][np.isfinite(viv_data[sensor_id]['vel'])]) * 1.1

            # ---------------------- 1. 绘制扇形轮廓（边框：两条半径 + 弧边） ----------------------
            # 生成弧边的角度序列
            theta_arc = np.linspace(theta_start_rad, theta_end_rad, 100)
            # 弧边坐标
            x_arc = fan_radius * np.cos(theta_arc)
            y_arc = fan_radius * np.sin(theta_arc)
            z_arc = np.zeros_like(x_arc)
            # 第一条半径（原点→弧起始点）
            x_r1 = np.array([0, fan_radius * np.cos(theta_start_rad)])
            y_r1 = np.array([0, fan_radius * np.sin(theta_start_rad)])
            z_r1 = np.zeros_like(x_r1)
            # 第二条半径（原点→弧结束点）
            x_r2 = np.array([0, fan_radius * np.cos(theta_end_rad)])
            y_r2 = np.array([0, fan_radius * np.sin(theta_end_rad)])
            z_r2 = np.zeros_like(x_r2)

            # 绘制轮廓（深红色粗线，突出标记）
            ax.plot(x_arc, y_arc, z_arc, COLOR_CONFIG['max_freq_contour'],  linewidth=2, label='Max Frequency Interval (VIV)')
            ax.plot(x_r1, y_r1, z_r1, COLOR_CONFIG['max_freq_contour'],  linewidth=2)
            ax.plot(x_r2, y_r2, z_r2, COLOR_CONFIG['max_freq_contour'],  linewidth=2)

            # ---------------------- 2. 绘制薄曲面片（替代填充，3D下更稳定） ----------------------
            # 生成极坐标网格（角度：区间范围，半径：0→fan_radius）
            theta_grid = np.linspace(theta_start_rad, theta_end_rad, 50)
            r_grid = np.linspace(0, fan_radius, 20)
            theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)
            # 转换为笛卡尔坐标
            x_mesh = r_mesh * np.cos(theta_mesh)
            y_mesh = r_mesh * np.sin(theta_mesh)
            z_mesh = np.zeros_like(x_mesh)  # 固定z=0

            # 绘制曲面片（半透明红色，无填充报错风险）
            ax.plot_surface(
                x_mesh, y_mesh, z_mesh,
                color=COLOR_CONFIG['max_freq_surface'], 
                alpha=COLOR_CONFIG['max_freq_surface_alpha'], 
                linewidth=0,  # 关闭曲面网格线
                antialiased=True
            )

            ax.legend()

        figs.append(fig)

    # ===================== 7. TKinter交互展示（与玫瑰图一致） =====================
    root = tk.Tk()
    root.title("3D Wind Turbulence Distribution (Aligned with Rose Map)")
    root.configure(bg=COLOR_CONFIG['tk_background'])
    app = ChartApp(root, figs)
    root.mainloop()

    return
