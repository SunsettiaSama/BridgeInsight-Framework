from ..visualize_tools.utils import ChartApp, PlotLib
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

from ..data_processer.data_processer_V0 import UNPACK, DataManager
from matplotlib.font_manager import FontProperties

import datetime
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.style.use('default')
font_size = 12

# ########################### 字体配置修改 ###########################
# 1. 全局默认字体设为Times New Roman（优先渲染英文，无中文乱码风险）
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
# 2. 解决负号显示为方块的问题（必加配置）
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = font_size

# 3. 定义中英文字体对象（保持你的原有定义，补充字体路径兼容性）
ENG_FONT = FontProperties(family='Times New Roman', size=font_size)
CN_FONT = FontProperties(
    family='SimHei', 
    size=font_size,
    # Windows系统无需指定fname，Linux/Mac需手动填写SimHei字体文件路径，示例：
    # fname='/usr/share/fonts/truetype/simhei/SimHei.ttf'
)

CMAP = plt.cm.gist_yarg

# ###################################################################

def Wind_Rose_Map():
    # 1. 对齐参考代码格式：核心参数定义
    interval_nums = 36  # 区间数量（参考代码统一用36）
    normalize = True    # 归一化开关（参考代码默认开启）
    axis_of_bridge = 10.6  # 桥轴线角度（参考代码固定值）
    
    # 目标传感器：北索塔塔顶、跨中桥面上游（对应参考代码的传感器命名）
    target_sensors = [
        # 'ST-UAN-T01-003-01',  # 北索塔塔顶
        'ST-UAN-G04-001-01'   # 跨中桥面上游
    ]
    sensor_names = [
        # 'VIV Wind Distribution (Top Of North Pylon)',
        'VIV Wind Distribution (Upstream of Mid-Span)'
    ]
    # 风向修正规则（参考代码区分传感器）
    wind_dir_correction = {
        # 'ST-UAN-T01-003-01': 180,  # 北索塔塔顶：180 - 原始风向
        'ST-UAN-G04-001-01': 360   # 跨中桥面上游：360 - 原始风向
    }
    
    # 路径配置
    VIV_excel_path = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\Img\VIV.xlsx'
    # wind_data_root = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\UAN\\'

    ploter = PlotLib() 
    manager = DataManager()

    # 初始化工具类
    unpacker = UNPACK()
    figs = []  # 存储VIV相关玫瑰图，用于tk交互

    # 读取VIV数据表格（核心：仅处理VIV发生时的风数据）
    df_viv = pd.read_excel(VIV_excel_path)
    time_interval = 1  # 分钟，取VIV发生时段1min内的风数据
    fs = 1  # 采样频率1Hz
    
    # 遍历每个目标风传感器（对齐参考代码循环结构）
    for idx, sensor_id in enumerate(target_sensors):
        # 存储该传感器下VIV发生时的风数据（保留角度值，先不转弧度）
        viv_wind_dir_deg = np.array([])  # 风向（角度，未修正）
        viv_wind_vel = np.array([])      # 风速（m/s）

        # 遍历VIV记录，匹配对应风数据
        for row in df_viv.values:
            viv_path, viv_time, plane = row
            # 从VIV路径匹配对应风传感器路径
            wind_paths = unpacker.VIC_Path_2_WindPath(VICpath=viv_path, wind_sensor_ids=[sensor_id])
            # 仅处理匹配到的、且属于目标VIV传感器的记录
            if len(wind_paths) == 1:
                # 解析风数据（保留原始角度，不先转弧度）
                wind_velocity, wind_direction, _ = unpacker.Wind_Data_Unpack(wind_paths[0])
                wind_velocity = np.array(wind_velocity)
                wind_direction = np.array(wind_direction)
                
                # 截取VIV发生时段的风数据（防止索引越界）
                start_idx = viv_time
                end_idx = viv_time + time_interval * 60 * fs
                end_idx = min(end_idx, len(wind_velocity))  # 简化越界判断
                if start_idx >= end_idx:
                    continue
                
                # 提取有效时段数据 + 过滤无效风速（参考代码：风速>0.1m/s为有效）
                vel_slice = wind_velocity[start_idx:end_idx]
                dir_slice = wind_direction[start_idx:end_idx]
                valid_mask = vel_slice > 0.1
                vel_valid = vel_slice[valid_mask]
                dir_valid = dir_slice[valid_mask]
                
                if len(vel_valid) == 0:
                    continue
                
                # 拼接数据（保留角度格式）
                viv_wind_dir_deg = np.hstack((viv_wind_dir_deg, dir_valid))
                viv_wind_vel = np.hstack((viv_wind_vel, vel_valid))

        # 检查是否有有效VIV风数据
        if len(viv_wind_dir_deg) == 0:
            print(f"警告：{sensor_names[idx]} 无VIV发生时的有效风数据")
            continue

        # 2. 对齐参考代码：风向修正（区分传感器）
        # 参考代码逻辑：正北指向桥轴线方向，先修正再归一化到0-360
        correction_val = wind_dir_correction[sensor_id]
        viv_wind_dir_deg = correction_val - viv_wind_dir_deg  # 方向修正
        viv_wind_dir_deg = np.mod(viv_wind_dir_deg, 360)     # 归一化到0-360度

        # 3. 对齐参考代码：分箱逻辑（角度分箱，非弧度）
        bin_step = int(360 / interval_nums)
        bins = np.arange(0, 360 + bin_step, bin_step)  # 参考代码分箱方式
        digitized = np.digitize(viv_wind_dir_deg, bins)  # 匹配分箱索引
        
        # 按分箱分组风速数据
        grouped_speeds = [viv_wind_vel[digitized == i] for i in range(1, len(bins))]
        # 计算每个分箱的样本数（玫瑰图半径）
        counts = [len(speeds) for speeds in grouped_speeds]
        
        # 对齐参考代码：归一化（normalize=True）
        if normalize:
            counts = np.array(counts) / np.sum(counts)  # 归一化到0-1
        
        # 计算每个分箱的平均风速（用于颜色映射）
        average_speeds = [np.mean(speeds) if len(speeds) > 0 else 0 for speeds in grouped_speeds]

        # 4. 对齐参考代码：绘制极坐标玫瑰图
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # 设置风向区间的角度（弧度）和宽度（参考代码格式）
        theta = np.deg2rad(bins[:-1])  # 每个区间的起始角度（转弧度）
        width = np.deg2rad(bin_step)   # 每个扇区的宽度（转弧度）

        # 绘制柱状图（参考代码样式：skyblue初始色，alpha=0.8，align='edge'）
        bars = ax.bar(theta, counts, width=width, bottom=0.0, color='skyblue', alpha=0.8, align='edge')

        # 对齐参考代码：极坐标轴方向（0度朝北，顺时针）
        ax.set_theta_zero_location('N')      # 0度朝北
        ax.set_theta_direction(-1)           # 顺时针方向

        # 5. 对齐参考代码：颜色映射（viridis色板，循环赋值）
        
        cmap = CMAP  # 类viridis的渐变色板（替代原Reds，解决过浅问题）
        norm = plt.Normalize(min(average_speeds), max(average_speeds))
        # 循环赋值颜色（参考代码zip方式）
        for r, bar, color in zip(average_speeds, bars, cmap(norm(average_speeds))):
            bar.set_facecolor(color)

        # 对齐参考代码：添加颜色条（vertical，pad=0.1）
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # ########################### 字体修改1：颜色条中文标签指定中文字体 ###########################
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='风速 (m/s)', pad=0.1)
        cbar.set_label('风速 (m/s)', fontproperties=CN_FONT)  # 中文标签用SimHei
        cbar.ax.tick_params(labelsize=font_size)  # 颜色条刻度字体（英文/数字自动用Times New Roman）
        # ###########################################################################################

        # 6. 对齐参考代码：图像处理/美化（核心格式对齐）
        # 地理坐标标签（N/NE/E/SE/S/SW/W/NW）- 英文自动用Times New Roman
        x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ax.set_xticklabels(x_ticks, fontproperties=ENG_FONT)  # 显式指定英文标签字体（可选，全局已默认）

        # 桥轴线绘制（参考代码：axis_of_bridge=10.6度，红色虚线）
        y_max = np.max(counts)
        ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, y_max * 1.1], color='red', linestyle='--')
        ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, y_max * 1.1], color='red', linestyle='--')
        # ########################### 字体修改2：中文注释指定中文字体 ###########################
        ax.annotate('桥轴线', xy=(np.deg2rad(axis_of_bridge), y_max * 0.9), 
                    ha='center', va='bottom', fontproperties=CN_FONT)
        # #########################################################################################
        
        # 降低y轴刻度密度（参考代码：MultipleLocator）
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.round(y_max * 0.25, 2)))
        # 调整y轴最大值（略微大于柱状图最大值）
        ax.set_ylim(0, y_max * 1.1)
        # y轴刻度转百分比（参考代码格式）
        ax.set_yticks(np.hstack([0, ax.get_yticks()[1:]]))
        # ########################### 字体修改3：y轴刻度标签（英文/数字）指定英文字体 ###########################
        ax.set_yticklabels([str(int(np.round(max(0, num * 100)))) + "%" for num in ax.get_yticks()], 
                           fontproperties=ENG_FONT)
        # ###########################################################################################
        # 调整y轴标签位置（左侧）
        ax.yaxis.set_label_coords(-0.1, 1.1)
        # 调整y轴位置到玫瑰图左侧（参考代码：180+90）
        ax.set_rlabel_position(180 + 90)

        # 设置图表标题（参考代码样式，若启用需指定对应字体）
        # if sensor_names[idx]:
        #     ax.set_title(sensor_names[idx], va='bottom', fontproperties=ENG_FONT)  # 英文标题用Times New Roman

        # 存储图表
        figs.append(fig)
        

    ploter.figs.extend(figs)
    plt.close()  # 释放内存

    # 区间数量
    interval_nums = 36
    # 归一化
    normalize = True
    sensor_ids = ["跨中桥面上游"]
    wind_dirs = [
                # r"F:\Research\Vibration Characteristics In Cable Vibration\data\2023September\UAN", 
                r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September"
                    ]

    for wind_dir in wind_dirs:
        for sensor_id in sensor_ids:
            wind_velocities, wind_directions, wind_angles = manager.get_wind_data_from_root(wind_dir, mode = "interval", sensor_id = sensor_id)
            # 对象转换
            wind_velocities, wind_directions, wind_angles = np.array(wind_velocities), np.array(wind_directions), np.array(wind_angles)
            
            # 做一个清洗工作，认为风速 < 0.1 m/s的均为无效风速
            indices = wind_velocities > 0.1
            wind_velocities = np.array(wind_velocities[indices])
            wind_directions = np.array(wind_directions[indices])
            wind_angles = np.array(wind_angles[indices])

            wind_directions = np.array(wind_directions)
            # 风向进行方向修正，正北指向桥轴线方向
            wind_directions = 360 - wind_directions # 顺时针修正
            # 将风速的方向归一化到0-360之间
            wind_directions = np.mod(wind_directions, 360)
            # wind_directions = wind_directions + 180 + 10.6 # 以南方桥轴线方向为初始方向
            wind_velocities = np.array(wind_velocities)

            bins = np.arange(0, 360 + int(360 / interval_nums), int(360 / interval_nums))  # 每18度一个区间（20个区间）
            digitized = np.digitize(wind_directions, bins)
            
            grouped_speeds = [wind_velocities[digitized == i] for i in range(1, len(bins))] # List[List]，其中，内部的list长度为该区间内风速的个数
            counts = [len(speeds) for speeds in grouped_speeds]
            if normalize: 
                counts = np.array(counts) / np.sum(counts)

            # 计算每个区间的平均风速
            average_speeds = [np.mean(speeds) if len(speeds) > 0 else 0 for speeds in grouped_speeds]

            # 创建极坐标图
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            # 设置风向区间的角度（弧度）和宽度
            theta = np.deg2rad(bins[:-1])  # 每个区间的起始角度
            width = np.deg2rad(int(360 / interval_nums))

            # 绘制柱状图
            bars = ax.bar(theta, counts, width=width, bottom=0.0, color='skyblue', alpha=0.8, align = 'edge')

            # 设置极坐标轴方向（0度朝北，顺时针方向）
            ax.set_theta_zero_location('N')      # 0度朝北
            ax.set_theta_direction(-1)           # 顺时针方向

            # 添加颜色映射（可选）
            cmap = CMAP
            norm = plt.Normalize(min(average_speeds), max(average_speeds))
            for r, bar, color in zip(average_speeds, bars, cmap(norm(average_speeds))):
                bar.set_facecolor(color)

            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # ########################### 字体修改4：第二部分颜色条中文标签指定中文字体 ###########################
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='风速 (m/s)', pad = 0.1)
            cbar.set_label('风速 (m/s)', fontproperties=CN_FONT)
            cbar.ax.tick_params(labelsize=font_size)
            # ###########################################################################################

            # 设置标题（若启用需指定字体）
            # ax.set_title(next(titles), va='bottom', fontproperties=ENG_FONT)

            # ############################
            # 图像处理，美化图像部分
            # ############################
            # 地理坐标（英文标签指定英文字体）
            x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            ax.set_xticklabels(x_ticks, fontproperties=ENG_FONT)

            # 桥轴
            y_max = np.max(counts)
            # ax.set_ylim(0, y_max)
            axis_of_bridge = 10.6  # degree
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, y_max * 1.1], color='red', linestyle='--')
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, y_max * 1.1], color='red', linestyle='--')
            # ########################### 字体修改5：第二部分中文注释指定中文字体 ###########################
            ax.annotate('桥轴线', xy=(np.deg2rad(axis_of_bridge), y_max * 0.9), 
                        ha='center', va='bottom', fontproperties=CN_FONT)
            # ###########################################################################################
            
            # 降低y轴刻度密度
            ax.yaxis.set_major_locator(plt.MultipleLocator(np.round(y_max * 0.25, 2)))
            # 调整y轴最大值略微大于柱状图的最大值
            ax.set_ylim(0, y_max * 1.1)
            # ########################### 字体修改6：修复原有传参错误 + 指定英文字体 ###########################
            ax.set_yticks(np.hstack([0, ax.get_yticks()[1: ]]))
            ax.set_yticklabels([str(int(np.round(max(0, num * 100)))) + "%" for num in ax.get_yticks()], 
                               fontproperties=ENG_FONT)
            # ###########################################################################################
            # 调整y轴标签为左侧
            ax.yaxis.set_label_coords(-0.1, 1.1)

            # 调整y轴位置到玫瑰图左侧
            ax.set_rlabel_position(180 + 90)

            ploter.figs.append(fig)


    ploter.show()