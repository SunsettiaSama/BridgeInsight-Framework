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
from typing import List
import datetime
from .data_processer.io_unpacker import DataManager



# =================================================所有解析方法都在这================================================

class UNPACK():

    """
    数据解析方法都在这里了
        
    """
    def __init__(self, init_path = True):
        self.VIC_root = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC'
        
        self.wind_root = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\UAN'
        if init_path:
            self.VIC_Paths_from_root = self.File_Read_Paths(self.VIC_root)
            self.wind_path_lis = self.File_Read_Paths(self.wind_root)
        return 
    
    def _read_subfolders(self, root, relative_path):
        '''
        读取路径下的所有子文件夹，生成一个列表并返回
        
        
        '''
        # 读取路径，即所有子文件夹的路径，得到所有的文件名称
        # 维护一个列表，存储所有子文件夹的路径
        subfolders = deque()
        for item in os.listdir(root):
            # 递归，每次循环，full_path改变
            full_path = os.path.join(root, item)
            # 若是文件夹，则执行下面代码，将新的相对路径添加到列表中
            if os.path.isdir(full_path):

                if relative_path == '':
                    # 新的相对路径是原来的相对路径加上新的路径
                    new_relative_path = os.path.join(root, item)
                else:
                    new_relative_path = os.path.join(relative_path, item)

                subfolders.append(new_relative_path)
                # 循环调用，妙
                # 递归调用方法，能遍历该目录下的所有文件夹
                subfolders.extend(self._read_subfolders(full_path, new_relative_path))

        return subfolders
            
    def _read_files_in_subfolders(self, root, subfolders):
        # 维护文件列表
        files = deque()
        for item in subfolders:
            # 遍历子文件夹，读取子文件夹下的文件
            root = os.path.join(root, item)
            for file in os.listdir(root):

                file_path = os.path.join(root, file).replace('\\', '/')
                if os.path.isfile(file_path):
                    files.append(file_path)
        return files

    def File_Read_Paths(self, root):
        '''
        传入根目录，返回根目录下所有文件的文件路径
        注意，这里返回的结果都是正斜杠
        '''
        subfolders = self._read_subfolders(root, relative_path = '')
        file_paths_lis = self._read_files_in_subfolders(root, subfolders)

        return file_paths_lis

    def File_Match_Sensor_Path(self, pattern, paths):
        """
        匹配路径列表中的路径。
        注：这一组机制在所有文件中通用
        
        :param pattern: 要匹配的字符串或正则表达式模式
        :param paths: 路径列表
        :return: 匹配的路径列表
        """
        matched_paths = deque()
        for path in paths:
            if pattern == os.path.split(path)[1].split('_')[0]:
                matched_paths.append(path)
        return list(matched_paths)
    
    def File_Match_Pattern(self, pattern, paths):
        """
        更通用的字符串匹配方法
        注：这一组机制在所有文件中通用
        
        :param pattern: 要匹配的字符串或正则表达式模式
        :param paths: list, 路径列表
        :return: 匹配的路径列表
        """
        matched_paths = deque()
        for path in paths:
            if pattern in path:
                matched_paths.append(path)
        return list(matched_paths)

    def File_Detach_Data(self, path, time_interval = 10, mode = 'VIC'):
        """
        params:
        time_interval: 默认10分钟
        """
        match mode:
            case 'VIC':
                fs = 50
                data = self.VIC_DATA_Unpack(path)
                point_nums = time_interval * 60 * fs
                data_lis = []
                for i in range(int(60 / time_interval)):
                    if len(data[i * point_nums:( i + 1 )* point_nums]) == point_nums:
                        data_lis.append(np.array(data[i * point_nums:( i + 1 )* point_nums]))
                return np.array(data_lis)
                
            case 'UAN':
                fs = 1
                wind_velocity, wind_direction, wind_Angle = self.Wind_Data_Unpack(path)
                point_nums = time_interval * 60 * fs

                wind_velocity_slice = []
                wind_direction_slice = []
                wind_Angle_slice = []

                for i in range(int(60 / time_interval)):
                    wind_velocity_slice.append(np.array(wind_velocity[i * point_nums:( i + 1 )* point_nums]))
                    wind_direction_slice.append(np.array(wind_direction[i * point_nums:( i + 1 )* point_nums]))
                    wind_Angle_slice.append(np.array(wind_Angle[i * point_nums:( i + 1 )* point_nums]))
                
                return wind_velocity_slice, wind_direction_slice, wind_Angle_slice

    def Wind_Data_Unpack(self, fname) -> List[List]: 
        with open(fname, "rb") as f:
            f_content = f.read()
            try:
                a=f_content.decode("GB18030")
                a=re.findall(r"\d+\.\d*\,\d+\.\d*\,\d+\.\d*",a)
            except ValueError:
                a=str(f_content)
                a=re.findall(r"\d+\.\d*\,\d+\.\d*\,\d+\.\d*",a)
            speed,direction,angle=[],[],[]
            for j in a:
                m=j.split(",")
                speed.append(float(m[0]))
                direction.append(float(m[1]))
                angle.append(round(float(m[2])-60.0,1))   #是否是减去60？
            # 风速风向风攻角，ppt上有图，可以参考
            dat=[speed,direction,angle]
        return dat
    
    def VIC_Path_2_WindPath(self, 
                            VICpath, 
                            wind_sensor_ids = ['ST-UAN-T01-003-01'],
                            ) -> list:
        """
        匹配VIC的路径到对应的风速路径
        注意，此时的excel 应当由两列组成：
        {
        'path': []
        'time': []
        }
        """
        wind_path = []
        for wind_sensor_id in wind_sensor_ids:
            
            month, day, file_name = Path(VICpath).parts[6:9]
            VIC_sensor_id, hour = file_name.split('_')
            hour = hour[:2]
            
            pattern = os.path.join(self.wind_root, month, day, wind_sensor_id) + '_' + hour
            pattern = pattern.replace('\\', '/')
            matched_paths = self.File_Match_Pattern(pattern, paths = self.wind_path_lis)
            wind_path.extend(matched_paths)
            
        # 去除重复元素
        cache = []
        for item in wind_path:
            if item not in cache:
                cache.append(item)
        wind_path = cache

        return wind_path
    
    def VIC_Path_Lis(self, root = 'F:/Research/My_Thesis/Data/苏通/VIC/'):
        unpacker = UNPACK()
        VIC_Path_lis = unpacker.File_Read_Paths(root)

        return VIC_Path_lis

    def VIC_path_Inplane_Outplane(self, path):
        '''
        实现面内外的路径转化
        path可反斜杠可正斜杠
        '''
        
        sensor_id = Path(path).parts[-1].split('_')[0]

        # 上下游拉索加速度散点图
        VIC_Name_Lis_Up_Down_Stream_In_Plane = [
            'ST-VIC-C18-101-01', 
            'ST-VIC-C18-102-01', # 面内
            'ST-VIC-C18-401-01', 
            'ST-VIC-C18-402-01',
            'ST-VIC-C18-501-01', 
            'ST-VIC-C18-502-01',

        ]

        VIC_Name_Lis_Up_Down_Stream_Out_Plane = [
            'ST-VIC-C18-101-02', 
            'ST-VIC-C18-102-02', # 面外
            'ST-VIC-C18-401-02', 
            'ST-VIC-C18-402-02',
            'ST-VIC-C18-501-02', 
            'ST-VIC-C18-502-02',
        ]
        if sensor_id in VIC_Name_Lis_Up_Down_Stream_In_Plane:
            in_or_out = 'Inplane'
            counter_id = VIC_Name_Lis_Up_Down_Stream_Out_Plane[VIC_Name_Lis_Up_Down_Stream_In_Plane.index(sensor_id)]

        elif sensor_id in VIC_Name_Lis_Up_Down_Stream_Out_Plane:
            in_or_out = 'Outplane'
            counter_id = VIC_Name_Lis_Up_Down_Stream_In_Plane[VIC_Name_Lis_Up_Down_Stream_Out_Plane.index(sensor_id)]
        

        replaced_string = str(Path(path).parent) + '\\' + counter_id
        replaced_string = replaced_string.replace('\\', '/') + '_' + Path(path).parts[-1].split('_')[1][:2]
        replaced_path = self.File_Match_Pattern(pattern = replaced_string, paths = self.VIC_Paths_from_root)

        return replaced_path
    
    def VIC_path_UpStream_DownStream(self, path):

        Upstream = [
            'ST-VIC-C18-101-01',
            'ST-VIC-C18-401-01',
            'ST-VIC-C18-501-01',
            'ST-VIC-C18-101-02',
            'ST-VIC-C18-401-02',
            'ST-VIC-C18-501-02',
        ]

        Downstream = [
            'ST-VIC-C18-102-01', # 面内
            'ST-VIC-C18-402-01',
            'ST-VIC-C18-502-01',
            'ST-VIC-C18-102-02', # 面外
            'ST-VIC-C18-402-02',
            'ST-VIC-C18-502-02',
        ]
        sensor_id = Path(path).parts[-1].split('_')[0]

        if sensor_id in Upstream:
            Up_Or_Down = 'Upstream'
            counter_id = Downstream[Upstream.index(sensor_id)]

        elif sensor_id in Downstream:
            Up_Or_Down = 'Downstream'
            counter_id = Upstream[Downstream.index(sensor_id)]
        

        replaced_string = str(Path(path).parent) + '\\' + counter_id
        replaced_string = replaced_string.replace('\\', '/') + '_' + Path(path).parts[-1].split('_')[1][:2]
        replaced_path = self.File_Match_Pattern(pattern = replaced_string, paths = self.VIC_Paths_from_root)

        return replaced_path, Up_Or_Down

    def VIC_DATA_Unpack(self, file_path):
        '''
        引入两种解析数据的方法，一种是正向计数（Count up），另一种为反向计数
            count up: 即寻找文件名字符串索引，对其后续编码进行解析
            count down: 反向使用offset，从文件末尾，以长度为标准60 * 60 * 1 * 50 * 4 = 720000的后720000个字符串进行解析
        Input:
            file_path: str, 文件路径
        Output: 
            floats: np.ndarray, 解析后的数据

        '''
        # 文件名分离字符串
        split_str = '_'

        strings = os.path.split(file_path)
        # 假定在字符串后面解析是正确的
        floats = np.array([])

        try:
            with open(file_path, "rb") as f: #振动数据读取  50Hz
                # 非泛用机制：将字符串拆开，取前者寻找。例如ST-VIC-C18-101-01_170000.VIC，拆成['ST-VIC-C18-101-01', '_170000.VIC']
                data = f.read()
                if split_str:
                    # 为什么要encode('utf-8')来着？
                    # 答：将其转化成utf-8的字节序列，然后再在string中寻找相应值
                    string = strings[-1].split(split_str)[0].encode('utf-8')
                else:
                    string = strings[-1].encode('utf-8')
                idx_in_data = data.index(string) + len(string) + 1
                # 加一个字符串筛选机制
                float = struct.unpack("f" * ((len(data)-idx_in_data)//4), data[idx_in_data:])
                floats = np.hstack([floats, float])
        except:
            pass
        return floats 

    def VIC_RMS(self, path, time_interval = 10):
        """
        



        """
        rmss = deque()
        data = self.VIC_DATA_Unpack(path)
        for i in range(int(60 / time_interval)):
            data_i = data[i * time_interval: (i + 1) * time_interval]

            mean_value = np.mean(data_i)
            rms = np.sqrt(np.mean(np.square(data_i - mean_value)))
            rmss.append(rms)

        return list(rmss)
    
    def PNG_Path_2_VIC_Data(self, PNG_path, time_interval = 1):
        """
        给定PNG的路径，返回对应的响应时程数据
        
        
        """

        month, day, hour, minute = Path(PNG_path).parts[-1].split('_')[:-1]
        sensor_id = Path(PNG_path).parts[-1].split('_')[-1].split('Response')[0]
        VIC_path_pattern = os.path.join(self.VIC_root, month[1:], day[1:], sensor_id + '_' + hour[1:]).replace('\\', '/').replace('//', '/')
        paths = self.File_Match_Pattern(VIC_path_pattern, self.VIC_Paths_from_root)
        if len(paths) == 0:
            return np.array([])
        
        data_slice = self.File_Detach_Data(path = paths[0], time_interval = time_interval, mode = 'VIC')
        return data_slice[int(minute[2:])]

    def PNG_Path_2_VIC_Path(self, PNG_path) -> str:
        """
        给定PNG的路径，返回对应的VIC文件路径
        
        
        """
        month, day, time_str, time_time = Path(PNG_path).parts[-1].split('_')[:-1]
        sensor_id = Path(PNG_path).parts[-1].split('_')[-1].split('Response')[0]
        VIC_path_pattern = os.path.join(self.VIC_root, month[1:], day[1:], sensor_id + '_' + time_str[1:]).replace('\\', '/').replace('//', '/')
        paths = self.File_Match_Pattern(VIC_path_pattern, self.VIC_Paths_from_root)

        if len(paths) == 1:
            return (paths[0], time_time)

    def UAN_yield_wind_Data(self, time_interval = 1):
        """
        生成器，可直接用for循环从该函数中提取风速、风向和风攻角，
        最后的返回对象为文件路径和对应时间索引，可直接使用
        Example:
            for wind_velocity, wind_direction, wind_angle, (wind_path, i) in UAN_yield_wind_Data():
                print(wind_velocity, wind_direction, wind_angle, (wind_path, i))
        
        """
        for wind_path in self.wind_path_lis:
            wind_velocity_slice, wind_direction_slice, wind_angle_slice = self.File_Detach_Data(path = wind_path, mode = 'UAN', time_interval = 1)
            for i in range(int(60 / time_interval)):
                yield wind_velocity_slice[i], wind_direction_slice[i], wind_angle_slice[i], (wind_path, i)

    def VIC_yield_wind_Data(self, time_interval = 1):
        """
        生成器，可直接用for循环从该函数中提取响应时程，
        最后的返回对象为文件路径和对应时间索引，可直接使用
        Example:
            for Response_Time_series, (wind_path, i) in VIC_yield_wind_Data():
                print(Response_Time_series, (wind_path, i))
        
        """
        for VIC_path in self.VIC_Paths_from_root:
            response_slice = self.File_Detach_Data(path = VIC_path, time_interval = 1)
            for i in range(int(60 / time_interval)):
                yield response_slice[i], (VIC_path, i)

    def displacement_calculation(self, data, base_modes, fs = 50, window = 100):
        """
        注意，该函数计算轨迹时，返回的轨迹shape为（window / 2，len(data) - window / 2）
        会减少掉一个窗口的长度
        
        """
        from scipy.fft import fft, fftfreq, fftshift
        # 引入时间窗口
        y_lis = []
        for i in np.arange(start = window / 2, stop = len(data) - window / 2):
            data_i = data[int(i - window / 2): int(i + window / 2)]
            Y = fftshift(fft(data_i))
            T = 1 / fs
            L = len(data_i)
            frequencies = fftfreq(L, T)[ : int(L/2)]
            # 应该不需要归一化？
            magnitude_spectrum = 2.0 / L * np.abs(Y[ : int(L/2)])
            phase_spectrum = np.angle(Y[ : int(L/2)])
        
            trajectory_magnitude = np.array([0])
            trajectory_phase = np.array([0])

            # 需要找到基频附近的相位和幅度
            base_modes = np.array(base_modes)
            base_modes = base_modes[base_modes < (fs / 2)]
            freqs = []
            for base_mode in base_modes:
                index_of_base_mode = np.argmin(np.abs(frequencies - base_mode))
                magnitude = magnitude_spectrum[index_of_base_mode]
                phase = phase_spectrum[index_of_base_mode]

                trajectory_magnitude = np.hstack((trajectory_magnitude, magnitude))
                trajectory_phase = np.hstack((trajectory_phase, phase))

                base_mode_i = frequencies[index_of_base_mode]
                freqs.append(base_mode_i)

            trajectory_magnitude = trajectory_magnitude[1:]
            trajectory_phase = trajectory_phase[1:]
            
            time = i
            # 注意这里，重构的是位移，而不是加速度
            # 所以不能对加速度直接进行分解后，代入（那样只是简单重构），采用离散积分试一下吧
            yi = np.sum( ((- trajectory_magnitude) / np.power(base_modes, 2)) * np.sin(time * base_modes + trajectory_phase))
            y_lis.append(yi)
        
        return np.array(y_lis)

    # 假设起始点为（0，0），求解位移
    def sinc_integrate(self, time_series, time_nums, fs = 50):
        """
        加密倍数
        """
        length = len(time_series)

        
        # 确定需要插值的点
        time_index = np.arange(0, length)
        time_axis = time_index / fs

        time_series_integrated = np.array([0])
        for index in time_index:
            # 去头去尾
            interval = np.linspace(index, index + 1, 2 + (time_nums - 1))[1:-1]
            time_series_integrated = np.hstack((time_series_integrated, time_series[index]))

            # 中间值的插值
            for n in interval:
                # 对齐时间轴
                n = n / fs
                # sinc 插值
                time_series_integrated_i = np.sum(time_series * (np.sin(np.pi * (n - time_axis))) / (np.pi * (n - time_axis)))
                time_series_integrated = np.hstack((time_series_integrated, time_series_integrated_i))
            
        
        time_series_integrated = time_series_integrated[1:]

        return time_series_integrated





