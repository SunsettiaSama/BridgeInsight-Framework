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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
class ChartApp:

    """
    示例代码如下
    figs = []

    # 示例图表1
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [4, 5, 6])
    ax1.set_title("Chart 1")
    figs.append(fig1)

    # 示例图表2
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [6, 5, 4])
    ax2.set_title("Chart 2")
    figs.append(fig2)

    # 示例图表3
    fig3, ax3 = plt.subplots()
    ax3.plot([1, 2, 3], [7, 8, 9])
    ax3.set_title("Chart 3")
    figs.append(fig3)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()
    
    """
    def __init__(self, root, figs):
        self.root = root
        self.figs = figs
        self.current_fig_index = 0

        # 创建一个Frame来放置图表
        self.chart_frame = tk.Frame(root)
        self.chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 创建一个FigureCanvasTkAgg对象
        self.canvas = FigureCanvasTkAgg(self.figs[self.current_fig_index], master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 创建按钮
        self.prev_button = ttk.Button(root, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_button = ttk.Button(root, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.save_button = ttk.Button(root, text="Save", command=self.save_current_chart)
        self.save_button.pack(side=tk.BOTTOM, padx=5, pady=5)

    def show_previous(self):
        self.current_fig_index = (self.current_fig_index - 1) % len(self.figs)
        self.update_chart()

    def show_next(self):
        self.current_fig_index = (self.current_fig_index + 1) % len(self.figs)
        self.update_chart()

    def update_chart(self):
        # 清除当前画布
        self.canvas.get_tk_widget().destroy()
        # 重新创建画布
        self.canvas = FigureCanvasTkAgg(self.figs[self.current_fig_index], master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def save_current_chart(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            self.figs[self.current_fig_index].savefig(file_path)
            print(f"Chart saved to {file_path}")

# =================================================所有解析方法都在这================================================

class UNPACK():

    """
    数据解析方法都在这里了
        
    """
    def __init__(self, init_path = True):
        self.VIC_root = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\VIC\\'
        
        self.wind_root = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\UAN\\'
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

    def Wind_Data_Unpack(self, fname): 
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

# ===============================================画图模块=========================================================
class PlotLib():

    def __init__(self):
        self.figs = []
        return 

    def plot(self, y, x = None, title = None, 
             xlabel = None, 
             ylabel = None,
             color = None, 
             legend = None,
             xlim = None, 
             ylim = None, 
             dpi = None, 
             fig = None, 
             style = None, 
             alpha = None, 
             ax = None, 
             add_fig = True, ):

        if not fig:
            fig = plt.figure()

        if dpi:
            fig.dpi = dpi
        
        if not ax:
            ax = fig.add_subplot(111)

        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)


        if xlim:
            ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)

        ax.grid(True)

        if x is not None:
            ax.plot(x, y, 
            label = legend if legend else None, 
            color = color if color else 'skyblue', 
            linestyle = style if style else None, 
            alpha = alpha if alpha else 0.8)
        else:
            ax.plot(y, 
            label = legend if legend else None, 
            color = color if color else 'skyblue', 
            linestyle = style if style else None,
            alpha = alpha if alpha else 0.8)

        if add_fig:
            self.figs.append(fig)

        return fig, ax

    def plots(self,
        data_lis,
        color = None,
        alpha = 0.7,
        titles = None, 
        labels = None,
        xlims = None,
        ylims = None,
        coordinate_systems = None,
        add_fig = True, 
            ):


        num_plots = len(data_lis)
        fig, axes = plt.subplots(num_plots)

        if num_plots == 1:
            axes = [axes]

        if not coordinate_systems:
            lines= [ax.plot([], [], color = color if color else 'skyblue', alpha = alpha)[0] for ax in axes]

        if coordinate_systems != None:
            lines = []
            for k, coordinate_system in enumerate(coordinate_systems):
                if coordinate_system == 'linear':
                    lines.append(axes[k].plot([], [], color = color if color else 'skyblue', alpha = alpha)[0])
                elif coordinate_system == 'loglog':
                    lines.append(axes[k].loglog([], [], color = color if color else 'skyblue', alpha = alpha)[0])
                elif coordinate_system == 'semilogy':
                    lines.append(axes[k].semilogy([], [], color = color if color else 'skyblue', alpha = alpha)[0])

        for i in range(len(axes)):
            if not labels is None:
                axes[i].set_xlabel(labels[i][0])
                axes[i].set_ylabel(labels[i][1])
                # 增大子图之间的间距
                fig.subplots_adjust(hspace=0.5)


            if not xlims is None:
                axes[i].set_xlim(xlims[i])
            else:
                axes[i].set_xlim(
                    (np.min(data_lis[i][0]), np.max(data_lis[i][0]))
                )

            if not titles is None:
                axes[i].set_title(titles[i])
                fig.subplots_adjust(hspace=0.5)

            if not ylims is None:
                axes[i].set_ylim(ylims[i])
            else:
                axes[i].set_ylim(
                    (np.min(data_lis[i][1])* 1.1, np.max(data_lis[i][1]* 1.1))
                )

            lines[i].set_data(data_lis[i][0], data_lis[i][1])


        if add_fig:
            self.figs.append(fig)
        return fig, axes

    def scatters(self,
        data_lis,
        color = None,
        alpha = 0.7,
        titles = None, 
        labels = None,
        xlims = None,
        ylims = None,
        add_fig = True, 
        s = 0.8
            ):

        num_plots = len(data_lis)
        fig, axes = plt.subplots(num_plots)

        if num_plots == 1:
            axes = [axes]

        for i in range(len(axes)):
            if not labels is None:
                axes[i].set_xlabel(labels[i][0])
                axes[i].set_ylabel(labels[i][1])
                # 增大子图之间的间距
                fig.subplots_adjust(hspace=0.5)

            if not xlims is None:
                axes[i].set_xlim(xlims[i])
            else:
                axes[i].set_xlim(
                    (np.min(data_lis[i][0]), np.max(data_lis[i][0]))
                )

            if not titles is None:
                axes[i].set_title(titles[i])
                fig.subplots_adjust(hspace=0.5)

            if not ylims is None:
                axes[i].set_ylim(ylims[i])
            else:
                axes[i].set_ylim(
                    (np.min(data_lis[i][1])* 1.1, np.max(data_lis[i][1]* 1.1))
                )

            axes[i].scatter(data_lis[i][0], data_lis[i][1], color = color if color else 'skyblue', alpha = alpha, s = s)


        if add_fig:
            self.figs.append(fig)
        return fig, axes


    def loglog(self, time_series, x = None, 
                            title = 'Time_Series', 
                            xlabel = 'Time',
                            ylabel = 'Value', 
                            xlim = None, ylim = None, 
                            dpi = None, 
                            add_fig = True):

        if not fig:
            fig = plt.figure()

        if dpi:
            fig.dpi = dpi
        
        if not ax:
            ax = fig.add_subplot(111)

        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)


        if xlim:
            ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)
        


        ax.grid(True)
        if x is not None:
            ax.loglog(x, time_series)
        else:
            ax.loglog(time_series)
        
        if add_fig:
            self.figs.append(fig)

        return fig, ax

    def rose_hist(self, x, title = 'Wind Direction Distribution', 
                  separate_bins = 20,
                  density = True,
                  color = None, 
                  fig = None, 
                  ax = None, 
                  y_label = "Percent Frequency (%)",
                  geographic_orientation = False, 
                  bridge_axis = True,
                  dpi = None, 
                  add_fig = True
                  ):
        
        if fig == None:
            fig = plt.figure()
        if ax == None:
            ax = fig.add_subplot(111, polar = True)

        if dpi:
            fig.dpi = dpi
        
        n, bins, patches = ax.hist(x, 
                                bins = separate_bins, 
                                density = density, 
                                color = color if color else 'skyblue',
                                )
        y_max = ax.get_ylim()[1]
        # 设置0度位置在北方
        ax.set_theta_zero_location('N')
        # 手动设置每个条形的边框颜色为透明
        for patch in patches:
            patch.set_edgecolor('none')  # 去掉边框
            
        ax.xaxis.grid(True)
        ax.yaxis.grid(ls = '--')

        ax.set_title(title)
        # ax.set_ylabel(ylable)
        if geographic_orientation:
            x_ticks = ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE']
            # ax.set_xticks(np.linspace(0, 2 * np.pi, len(x_ticks), endpoint=False), labels = x_ticks)
            ax.set_xticklabels(x_ticks)

        ax.tick_params(axis='y', labelleft=True, left=True)

        if bridge_axis == True:
            ax.text(-0.12, 0.5, y_label, rotation=90, transform=ax.transAxes, va="center", ha="center")
            # 桥梁中轴线
            axis_of_bridge = 360 - 10.6 # degree
            max_freq = y_max
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, max_freq], color =  'pink')
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, max_freq], color =  'pink')
            ax.annotate('Bridge axis', xy = (np.deg2rad(axis_of_bridge), max_freq* 0.9), ha = 'center', va = 'bottom')
        
        if add_fig:
            self.figs.append(fig)

        return fig, ax
    
    def rose_scatter(self, theta, rho, 
                title = None, 
                xlabel = None,
                ylabel = None, 
                xlim = None, 
                ylim = None,
                dpi = None, 
                fig = None, 
                ax = None, 
                s = None, 
                marker = None,
                legend = None,
                color = None,
                colorbar = None, 
                cmap = None, 
                c_data = None,
                color_label = None,
                geographic_orientation = True,
                bridge_axis = True, 
                norm = None, 
                alpha = 0.8, 
                add_fig = True
                ):
        if not fig:
            fig = plt.figure()

        if dpi:
            fig.dpi = dpi
        
        if not ax:
            ax = fig.add_subplot(111, polar = True)
            # 设置0度位置在北方
            ax.set_theta_zero_location('N')
            ax.xaxis.grid(True)
            ax.yaxis.grid(ls = '--')

        if not cmap:
            ax.scatter(theta, rho, 
                    color = color if color else 'skyblue',
                    s = s if s else 1, 
                    alpha = alpha, 
                    label = legend if legend else None, 
                    marker = marker if marker else 'o')
        else:
            import matplotlib
            from matplotlib.colors import LinearSegmentedColormap

            # 定义颜色映射并添加 alpha 通道
            base_cmap = matplotlib.colormaps[cmap]
            color_list = base_cmap(np.arange(base_cmap.N))
            # 添加 alpha 通道，这里我们将 alpha 设定为线性变化从 0 到 1
            color_list[:, -1] = alpha

            # 创建新的颜色映射
            custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', color_list)

            sc = ax.scatter(theta, rho,
                        cmap = custom_cmap,
                        c = c_data, 
                        s = s if s else 1, 
                        label = legend if legend else None, 
                        marker = marker if marker else 'o',
                        norm = norm if norm else None,
                        add_fig = True, 
                        )
            if colorbar:
                cbar = fig.colorbar(sc, ax = ax, label = color_label)
                cbar.ax.tick_params(labelsize=10)
            

        if xlim:
            ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)
        
        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_yticklabels([])
            ax.text(-0.12, 0.5, ylabel, rotation=90, transform=ax.transAxes, va="center", ha="center")


        # ax.set_ylabel(ylable)
        if geographic_orientation:
            x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            ax.set_xticklabels(x_ticks)

        # 逆时针
        ax.set_theta_direction(-1)

        # # 百分比
        # lin_space = np.arange(0, y_lim, step = 0.05)
        # yticks = lin_space

        # # 在 x = 45° 的位置添加 y 轴刻度值标签
        # ylabel_position = np.deg2rad(360 - 45) 
        # for y_tick in yticks:
        #     ax.text(ylabel_position, y_tick, f'{y_tick * 100:.0f}', va="center", ha="left")

        if bridge_axis == True:
            # 桥梁中轴线
            axis_of_bridge = 10.6 # degree
            max_freq = np.max(rho)
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, max_freq], color =  'pink')
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, max_freq], color =  'pink')
            ax.annotate('Bridge axis', xy = (np.deg2rad(axis_of_bridge), max_freq - 0.5), ha = 'center', va = 'bottom')
        
        if add_fig:
            self.figs.append(fig)
        
        return fig, ax

    def scatter(self, x, y, 
                title = None, 
                xlabel = None,
                ylabel = None, 
                xlim = None, 
                ylim = None,
                dpi = None, 
                fig = None, 
                ax = None, 
                marker = None,
                color = None,
                legend = None,
                s = None,
                alpha = 0.8, 
                add_fig = True
                ):
        if fig == None:
            fig = plt.figure()

        if dpi:
            fig.dpi = dpi
        
        if ax == None:
            ax = fig.add_subplot(111)

        ax.scatter(x, y, 
                color = color if color else 'skyblue',
                s = s if s else 1, 
                alpha = alpha, 
                marker = marker if marker else 'o',
                label = legend if legend else None)
        if xlim:
            ax.set_xlim(xlim)#

        if ylim:
            ax.set_ylim(ylim)
        
        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)

        if add_fig:
            self.figs.append(fig)

        return fig, ax
    
    def scatter_3d(self, theta, rho, z, 
                title = None, 
                xlabel = None,
                ylabel = None, 
                zlabel = None,
                xlim = None, 
                ylim = None,
                dpi = None, 
                fig = None, 
                ax = None, 
                marker = None,
                color = None,
                edgecolor = None,
                legend = None,
                s = None,
                alpha = 0.5, 
                add_fig = True
                ):
        '''
        rho: ndarray, shape (n,)
        theta: ndarray, shape (n,)
        z: ndarray, shape (n,)
        
        
        '''
        x = rho * np.cos(theta - np.pi / 2)
        y = rho * np.sin(theta- np.pi / 2)
        z = z

        if fig == None:
            fig = plt.figure()

        if dpi:
            fig.dpi = dpi
        
        if ax == None:
            ax = fig.add_subplot(111, projection = '3d')
            ax.set_axis_off()


            from mpl_toolkits.mplot3d import Axes3D, art3d
            from mpl_toolkits.mplot3d.art3d import Path3DCollection
            from matplotlib.patches import Circle
            font_size = 10

            # 用来画柱坐标的圈圈
            degs = np.linspace(0, 360, 360 * 5)
            circle_nums = 4
            # 添加自定义刻度和标签
            for i in range(1, circle_nums):
                x_xy_axis = np.max(rho[np.isfinite(rho)]) * i / circle_nums * np.cos(np.deg2rad(degs)) * 1.1
                y_xy_axis = np.max(rho[np.isfinite(rho)]) * i / circle_nums * np.sin(np.deg2rad(degs)) * 1.1
                z_xy_axis = np.zeros(len(x_xy_axis))
                ax.plot(x_xy_axis, y_xy_axis, z_xy_axis, zdir='z', color = 'grey', linestyle = '--', linewidth = 0.5) 

            # 默认正北为0°
            # 添加方向角
            strings = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            dir_rho = np.max(rho[np.isfinite(rho)]) * 1.2
            dir_theta = np.array([90, 45, 0, -45, -90, -135, -180, -225])
            x_xy_ticks = dir_rho * np.cos(np.deg2rad(dir_theta))
            y_xy_ticks = dir_rho * np.sin(np.deg2rad(dir_theta))
            z_xy_ticks = np.zeros(len(dir_theta))
            for i in range(len(dir_theta)):
                theta_tick = dir_theta[i]
                x_tick = x_xy_ticks[i]
                y_tick = y_xy_ticks[i]
                z_tick = z_xy_ticks[i]
                ax.text(x = x_tick, y = y_tick, z = z_tick, s = strings[i], fontsize = font_size)


            for i in range(len(dir_theta)):
                x_tick = (0, dir_rho * np.cos(np.deg2rad(dir_theta[i])))
                y_tick = (0, dir_rho * np.sin(np.deg2rad(dir_theta[i])))
                z_tick = (0, 0)
                ax.plot(x_tick, 
                        y_tick, 
                        z_tick, 
                        color = 'grey', 
                        linewidth = 0.5)
                
            # 在SW处画风速标记
            degs = np.linspace(0, 360, 360 * 5)
            circle_nums = 4
            SW = 'SW'
            wind_ticks = [0, 10, 20, 30, 40, 50]
            for i in range(0, circle_nums):
                x_SW_tick = np.max(rho[np.isfinite(rho)]) * i / circle_nums * np.cos(np.deg2rad(dir_theta[strings.index(SW)])) * 1.1
                y_SW_tick = np.max(rho[np.isfinite(rho)]) * i / circle_nums * np.sin(np.deg2rad(dir_theta[strings.index(SW)])) * 1.1
                z_SW_tick = 0
                ax.text(x = x_SW_tick, y = y_SW_tick, z = z_SW_tick, s = wind_ticks[i], fontsize = font_size)

            # 用来画柱坐标的z轴及其刻度
            z_ticks = np.linspace(np.min(z[np.isfinite(z)]), np.max(z[np.isfinite(z)]), 5)
            z_axis_x = np.zeros(len(z_ticks))
            z_axis_y = np.zeros(len(z_ticks))
            z_axis_z = z_ticks
            ax.plot(z_axis_x, z_axis_y, z_axis_z, color = 'grey', linewidth = 0.5)
            for i in range(len(z_ticks)):
                z_tick = z_ticks[i]
                ax.text(x = 0, y = 0, z = z_tick, s = f'{z_tick:.1f}', fontsize = font_size)

            ax.grid(False)
            ax.set_axis_off()

        if color:
            ax.scatter(x, y, z,
                    color = color,
                    s = s if s else 1, 
                    alpha = alpha, 
                    marker = marker if marker else 'o',
                    label = legend if legend else None, 
                    edgecolors = edgecolor if edgecolor else 'none')
        else:
            ax.scatter(x, y, z,
                       color = 'skyblue',
                        s = s if s else 1, 
                        alpha = alpha, 
                        marker = marker if marker else 'o',
                        label = legend if legend else None, 
                        edgecolors = edgecolor if edgecolor else 'none')





        # from scipy.spatial import ConvexHull
        # # 用来画xy上的投影，此时z归为0.5倍
        # x_proj = x
        # y_proj = y
        # z_proj = (np.min(z) - (np.max(z) + np.min(z))* 0.2)* np.ones(len(y_proj))
        # ax.scatter(x_proj, y_proj, z_proj, color = 'pink', marker = 'X', s = 1)
        
        # z_proj = z
        # np.random.seed(seed = 1)
        # index = np.where(x_proj == np.random.choice(x_proj))

        # x_proj_i, y_proj_i, z_proj_i = x_proj[index], y_proj[index], z_proj[index]
        # ax.scatter(
        #     x_proj_i, 
        #     y_proj_i, 
        #     0, 
        #     color = 'pink', marker = 'X', s = 5)

        # ax.plot(
        #     np.hstack((x_proj_i, x_proj_i)), 
        #     np.hstack((y_proj_i, y_proj_i)), 
        #     np.hstack((z_proj_i, 0)), 
        #     color = 'pink', marker = 'X')



        # hull = ConvexHull(np.vstack([x_proj, y_proj]).T)
        # # hull.simplices内部，是一个列表，内部包含了所有两个点的索引
        # for index in hull.vertices:
        #     x_proj_i = 
        #     y_proj_i = np.hstack((y_proj[index], y_proj[index]))
        #     z_proj_i = np.hstack((z_proj[index], np.array([0])))

        #     ax.plot(x_proj_i, y_proj_i, z_proj_i, color = 'pink')

        # z_hull = np.zeros(len(hull.vertices))
        # ax.plot(hull.points[hull.vertices, 0], hull.points[hull.vertices, 1], z_hull, color = 'pink')
        

        # 自动缩放到合适的视角
        # ax.autoscale_view()
        fig.subplots_adjust(left=0, right=1, bottom=-1, top=2)




        if xlim:
            ax.set_xlim(xlim)#

        if ylim:
            ax.set_ylim(ylim)
        
        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)

        if zlabel:
            ax.set_zlabel(ylabel)

        if add_fig:
            self.figs.append(fig)

        return fig, ax
    
    def hist(self, x, 
             bins = 50, # 直方图的箱数
            #  range = None, # 直方图值域范围
                density = True, # 是否选择归一化
                title = None, 
                xlabel = None,
                ylabel = None, 
                dpi = None, 
                fig = None, 
                ax = None, 
                color = None,
                legend = None,
                alpha = 0.7, 
                add_fig = True
                ):
        
        if not fig:
            fig = plt.figure()

        if dpi:
            fig.dpi = dpi
        
        if not ax:
            ax = fig.add_subplot(111)

        ax.hist(x, bins = bins,
                color = color if color else 'skyblue',
                alpha = alpha, 
                density = density,
                label = legend if legend else None)
        
        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)
        
        if add_fig:
            self.figs.append(fig)

        return fig, ax
    
    def animate(self, data_lis,
                    color = None,
                    alpha = 0.7,
                    title = None, 
                    labels = None,
                    xlims = None,
                    ylims = None,
                    fps = None,
                    save_path=None, 

                    dynamic_adjust = [(True, False), (False, False)]):

        """
        data: list, [(x_data, y_data), 
                (x_data, y_data), 
                (x_data, y_data), ...]
        x_data: nd.array, shape == (Frame, x_data)
        dynamic_adjust: [(x_adjust, y_adjust), (x_adjust, y_adjust), ...]
            example: [(True, False), (False, False)]

        frame_data: ndarray, 数组形状为(K, n)，第一个为帧数（比如每秒1帧），第二个为数据
        
        
        """
        num_plots = len(data_lis)
        fig, axes = plt.subplots(num_plots)

        if num_plots == 1:
            axes = [axes]

        lines= [ax.plot([], [], color = color if color else 'skyblue', alpha = alpha)[0] for ax in axes]

        for i in range(len(axes)):
            if not labels is None:
                axes[i].set_xlabel(labels[i][0])
                axes[i].set_ylabel(labels[i][1])
                # 增大子图之间的间距
                fig.subplots_adjust(hspace=0.5)


            if not xlims is None:
                axes[i].set_xlim(xlims[i])
            else:
                axes[i].set_xlim(
                    (np.min(data_lis[i][0]), np.max(data_lis[i][0]))
                )

            if not title is None:
                axes[i].set_title(title[i])
                fig.subplots_adjust(hspace=0.5)

            if not ylims is None:
                axes[i].set_ylim(ylims[i])
            else:
                axes[i].set_ylim(
                    (np.min(data_lis[i][1]), np.max(data_lis[i][1]))
                )
            
        # 初始化函数
        def init():
            for line in lines:
                line.set_data([], [])
                return line,
        # 更新函数
        def update(frame):
            for i, (x, y) in enumerate(data_lis):
                
                y_frame = y[frame, :]  # 假设每个帧的数据点数相同
                x_frame = x[frame, :]  # 假设每个帧的数据点数相同

                if x_frame.shape != y_frame.shape:
                    raise ValueError('x shape and y shape should be same!') 

                for x_adjust, y_adjust in dynamic_adjust:

                    if x_adjust:
                        min_x, max_x = min(x_frame), max(x_frame)
                        if min_x == max_x:  # 防止x轴范围相同
                            min_x -= 0.1
                            max_x += 0.1
                        axes[i].set_xlim(min_x, max_x)

                    if y_adjust:
                        min_y, max_y = min(y_frame), max(y_frame)
                        if min_y == max_y:  # 防止y轴范围相同
                            min_y -= 0.1
                            max_y += 0.1
                        axes[i].set_ylim(min_y, max_y)
                    
                lines[i].set_data(x_frame, y_frame)

            return lines

        ani = animation.FuncAnimation(fig, update, frames=range(data_lis[0][0].shape[0]), init_func=init, blit=True)
        if save_path:
            extention = os.path.splitext(save_path)[1]

            if extention == '.gif':
                ani.save(save_path, writer=animation.PillowWriter(fps= fps if fps else 3))

            if extention == '.mp4':
                ani.save(save_path, writer='ffmpeg', fps=fps if fps else 3)
        plt.close()

        return ani

    def show_sample(self, data, fs = 50, nperseg = 256, 
                    add_fig = True, scatter = False):
        
        # 查看样本
        fx, pxxden = signal.welch(data, fs = fs, nfft = 65536, 
                                    nperseg = nperseg, noverlap = 1)
        
        times = np.arange(len(data)) / fs
        data_lis = [
            (times, data),
            (fx, pxxden) 
            ]
        
        titles = [
            f'Time Series',
            'Power Spectral Density'
        ]

        labels = [
            ('Time(s)', 'Acceleration $(m/s^2)$'),
            ('Frequency(hz)', 'PSD')
        ]
        if scatter:
            fig, axes = self.scatters(data_lis, 
                                    titles = titles, 
                                    labels = labels, 
                                    add_fig = False, 
                                    )
        else:
            fig, axes = self.plots(data_lis, 
                                    titles = titles, 
                                    labels = labels, 
                                    add_fig = False, 
                                    )
        if add_fig:
            self.figs.append(fig)

        return fig, axes

    def show(self):
        '''
        显示所有记录的图像
        
        '''
        tk = Tk()
        app = ChartApp(tk, self.figs)
        tk.mainloop()
        return 
    


class Abnormal_Vibration_Filter():


    def __init__(self):


        return 


    def VIV_Filter(self, data, f0 = 1, f0times = 2):
        """
        利用MECC准则，筛选VIV
        1. 振动加速度的最大值应大于0.01
        2. 单峰准则k，f0：k为能量峰值底部的区间长度，f0为基频
        3. 高阶控制n, 应当为基频的3阶及以上
        """

        rms = np.sqrt(np.mean((data - np.mean(data)) ** 2))
        # 准则4 RMS控制
        if rms < 0.3:
            return False
        

        fx, Pxx_den = signal.welch(data, fs = 50, 
                                   nfft = 512,
                                   nperseg = 512,
                                   noverlap = 256)

        # 准则一：振幅控制
        if np.max(data) < 0.01:
            return False
        E1 = np.max(Pxx_den)

        index = np.where(Pxx_den == E1)
        # 主导模态
        f_major = fx[index]

        # 准则三：主导模态
        if f_major < 3 * f0:
            return False

        
        f_major_left, f_major_right = f_major - f0times * f0, f_major + f0times * f0
        # 主导模态 2倍基频以内不管
        # 2倍基频以外寻找最大值
        Pxx_den_left = Pxx_den[fx < f_major_left]
        Pxx_den_right = Pxx_den[fx > f_major_right]
        
        Ek = np.sort(np.hstack((Pxx_den_left, Pxx_den_right)))[-1]

        # 准则4：rms阈值
        rms = np.mean(np.square(data - np.mean(data)))
        if rms < 0.03:
            return False

        # 准则二：单峰
        if Ek / E1 < 0.1:
            return True

class _Propose_Data_GUI:
    def __init__(self, root, save_excel_root, 
                 default_img_root = r"F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\Sampling_10000\\", 
                 image_list = []):
        self.root = root
        self.root.title("Image Viewer")
        self.root.bind('<Return>', self.next_image)

        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Control-s>', self.save_to_excel)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.current_image_index = 0
        self.data_dict = {}
        self.image_list = image_list

        # GUI setup
        self.label = Label(root)
        self.label.grid(row=1, column=1)

        # 全选并且放到光标的末尾
        self.entry_text = StringVar()
        self.entry = Entry(root, textvariable=self.entry_text)
        self.entry.grid(row=2, column=1, padx=30, pady=10)
        self.entry.focus_set()
        self.entry.select_range(0, tk.END)

        # 添加一个状态标签
        self.status_label = Label(root, text="", fg="green")
        self.status_label.grid(row=3, column=1, padx=30, pady=10)

        Button(root, text="Next", command=self.next_image).grid(row=3, column=2, padx=10, pady=10)
        Button(root, text="Previous", command=self.prev_image).grid(row=3, column=0, padx=10, pady=10)
        Button(root, text="Save", command=self.save_to_excel).grid(row=0, column=2, sticky="ne")

        # 绑定StringVar的追踪功能，以监控输入变化
        self.entry_text.trace_add('write', self.update_data_and_feedback)


        # 设置默认路径
        self.default_folder = default_img_root 
        # 
        self.save_excel = save_excel_root if save_excel_root else 'output.xlsx'
        # 如果没有图像列表，则选择文件夹并收集图像
        if not self.image_list:
            self.folder_selected = filedialog.askdirectory(
                title="请选择包含图片的文件夹",
                initialdir=self.default_folder or os.path.expanduser("~")
            )

            # 再次尝试加载已有的数据，以防有部分图片已经被处理过
            self.load_existing_data()

        # Show the first image and set its prefill y
        if self.image_list:
            self.show_image()

    def load_existing_data(self):
        """尝试加载现有的output.xlsx文件，并过滤已完成的图片"""
        
        self.data_dict = dict()

        try:
            df = pd.read_excel(self.save_excel)
            self.image_list_from_excel = list(df['File Path'])
            self.data_dict_from_excel = dict(zip(df['File Path'], df['User Input']))
            
            # 重置当前索引为0，因为我们只关心未完成的图片
            self.current_image_index = 0
            
        except FileNotFoundError:
            # 如果文件不存在，则初始化为空
            self.image_list_from_excel = []
            self.data_dict_from_excel = {}
        except Exception as e:
            messagebox.showerror("加载失败", f"加载{self.save_excel}时出现错误: {str(e)}")
            self.image_list_from_excel = []
            self.data_dict_from_excel = {}
        
        self.collect_images(self.folder_selected)

        self.image_list = [item for item in self.image_list if item not in self.image_list_from_excel]
        self.data_dict = self.data_dict_from_excel
        # dict不需要清洗，只需要添加就行
        
        if not self.image_list:
            messagebox.showinfo("信息", "所有图片都已经完成了输入。")

    def collect_images(self, folder_path):
            for file_name in os.listdir(folder_path):
                self.image_list.append(os.path.join(folder_path, file_name))


    def show_image(self):
        if not self.image_list:
            return
        img = Image.open(self.image_list[self.current_image_index])
        img.thumbnail((800, 600))
        photo = ImageTk.PhotoImage(img)
        self.label.config(image=photo)
        self.label.image = photo
        
        # 设置预填值到输入框
        self.entry_text.set('0')

    def update_data_and_feedback(self, *args):
        if self.image_list and self.current_image_index < len(self.image_list):
            # 更新当前图片对应的用户输入
            current_image_path = self.image_list[self.current_image_index]
            new_input = self.entry_text.get()
            self.data_dict[current_image_path] = new_input
            
            # 提供即时反馈
            self.status_label.config(text=f"已获取输入: {new_input}", fg="green")

    def next_image(self, event=None):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.show_image()

    def prev_image(self, event=None):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image()

    def save_to_excel(self):
        try:
            df = pd.DataFrame(list(self.data_dict.items()), columns=['File Path', 'User Input'])
            df.to_excel(self.save_excel, index=False)
            messagebox.showinfo("保存成功", f"数据已成功保存到{self.save_excel}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存过程中出现错误: {str(e)}")

    def on_closing(self):
        if messagebox.askokcancel("退出", "您想要退出吗？未保存的数据将会丢失。"):
            self.root.destroy()

class Data_Process():


    def __init__(self):
        return

    
    def Proposal_Samples(self, 
                         save_excel_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\test.xlsx', ):
        root = Tk()
        app = _Propose_Data_GUI(root, save_excel_path)
        root.mainloop()
        return 
    

    def __call__(self, *args, **kwds):
        self.Proposal_Samples()

class Cal_Mount():
    """
    该类用以计算峰值，并返回峰值对应的最大值
    默认参数
    mount = Cal_Mount(length = 576.9147, 
                      force = 8046000, 
                      alphac = 20.338908862 * np.pi / 180, 
                      E = 2.1e11, # Pa的强度
                      A = 0.024092, 
                      m = 100.8)

    print(mount.base_mode()[1])

    """
    def __init__(self, 
                 length = 576.9147, 
                 force = 8046000, 
                 alphac = 20.338908862 * np.pi / 180, 
                 A = 0.024092, 
                 m = 100.8, 
                E = 2.1e11):

        # 几何参数
        self.alphac = alphac    # 弧度制
        self.A = A
        self.length = length

        # 力学参数
        self.E = E
        self.force = force
        self.m = m
        self.multi_nums = self.base_modes()[1][-1] - self.base_modes()[1][-2]


        return 
    
    def inplane_mode(self, n):

        # 奇数
        # 奇数对应正对称振动
        if n % 2 == 1:
            lambda_ = self.E * self.A * np.square((self.m * 9.8 * np.cos(self.alphac))) / (self.force ** 3)
            if n == 1:
                zeta = 0.0017 * np.power(lambda_, 2) + 0.1254 * lambda_ + 3.1444
            elif n == 3:
                zeta = 0.0053 * lambda_ + 9.4239
            else:
                zeta = n * np.pi
                
            fn = (zeta / (2 * np.pi * self.length)) * np.sqrt(self.force / self.m)

            return fn
        

        # 偶数
        # 偶数对应反对称振动
        elif n % 2 == 0:
            fn = (n / (2 * self.length)) * np.sqrt(self.force / self.m)

            return fn
        
    def outplane_mode(self, n):

        fn = (n / (2 * self.length)) * np.sqrt(self.force / self.m)

        return fn

    def base_modes(self, max_freq = 25):
        """
        调用该方法，传回小于max_freq的基频

        return inplane_modes, outplane_modes
        """
        
        inplane_modes = [0]
        outplane_modes = [0]

        # 面内
        while inplane_modes[-1] < max_freq:
            inplane_modes.append(self.inplane_mode(len(inplane_modes) + 1))


        # 面外
        while outplane_modes[-1] < max_freq:
            outplane_modes.append(self.outplane_mode(len(outplane_modes) + 1))
        
        self.inplane_modes = inplane_modes[1: ]
        self.outplane_modes = outplane_modes[1: ]

        return self.inplane_modes, self.outplane_modes
    
    def peaks(self, fx, pxxden, return_intervals = False):
        """
        返回能找到的前50阶振动模态
        return fx_peaks, pxxden_peaks

        if return_intervals = True:
            return fx_peaks, pxxden_peaks, fx_intervals, pxxden_intervals
        """
        
        # 临近模态区间范围
        interval_length = 0.2

        fx_peaks = []
        pxxden_peaks = []
        
        # 稍微兼容一下
        if len(fx.shape) == 2:
            fx = fx[0, :]
        if len(pxxden.shape) == 2:
            pxxden = pxxden[0, :]

        fxi = fx
        pxxdeni = pxxden
        
        if return_intervals:
            fx_intervals = []   
            pxxden_intervals = []

        pxxden_max_max = np.max(pxxden)
        max_value = pxxden_max_max
        while len(fxi) != 0:
            
            # 最大值
            max_value = np.max(pxxdeni)
            max_index = np.argwhere(pxxdeni == max_value)
            fx_max = fxi[max_index].item()

            # 获取该数值区间下的最大值
            fx_peaks.append(fx_max)
            pxxden_peaks.append(max_value)

            # 若需要返回区间
            if return_intervals:
                
                pxxden_intervali = pxxdeni[fxi < fx_max + self.multi_nums * interval_length]
                fx_intervali = fxi[fxi < fx_max + self.multi_nums * interval_length]

                pxxden_intervali = pxxden_intervali[fx_intervali > fx_max - self.multi_nums * interval_length]
                fx_intervali = fx_intervali[fx_intervali > fx_max - self.multi_nums * interval_length]

                fx_intervals.append(fx_intervali)
                pxxden_intervals.append(pxxden_intervali)


            # 删除区间
            # 左区间
            pxxden_left = pxxdeni[fxi < fx_max - self.multi_nums * interval_length]
            fx_left = fxi[fxi < fx_max - self.multi_nums * interval_length]

            # 右区间
            pxxden_right = pxxdeni[fxi > fx_max + self.multi_nums * interval_length]
            fx_right = fxi[fxi > fx_max + self.multi_nums * interval_length]

            pxxdeni = np.concatenate((pxxden_left, pxxden_right))
            fxi = np.concatenate((fx_left, fx_right))

        if return_intervals:
            return fx_peaks, pxxden_peaks, fx_intervals, pxxden_intervals
        else:
                return fx_peaks, pxxden_peaks






def In_Out_Plane_Figure():

    # Up_Stream_Sensor_In_Plane = []
    # Down_Stream_Sensor_In_Plane = []
    root = 'F:/Research/My_Thesis/Data/苏通/VIC/'
    unpacker = UNPACK()
    VIC_Path_lis = unpacker.File_Read_Paths(root)


    # 上下游拉索加速度散点图
    VIC_Name_Lis_Up_Down_Stream_In_Plane = [
        ('ST-VIC-C18-101-01', 'ST-VIC-C18-102-01'), # 面内
        ('ST-VIC-C18-401-01', 'ST-VIC-C18-402-01'),
        ('ST-VIC-C18-501-01', 'ST-VIC-C18-502-01'),
        

    ]

    VIC_Name_Lis_Up_Down_Stream_Out_Plane = [
        ('ST-VIC-C18-101-02', 'ST-VIC-C18-102-02'), # 面外
        ('ST-VIC-C18-401-02', 'ST-VIC-C18-402-02'),
        ('ST-VIC-C18-501-02', 'ST-VIC-C18-502-02'),
    ]
    Up_Stream_pathlis = []
    Down_Stream_pathlis = [] 

    Abnormal_fil = Abnormal_Vibration_Filter()

    for Up_Stream_Sensor_In_Plane, Down_Stream_Sensor_In_Plane in VIC_Name_Lis_Up_Down_Stream_In_Plane:

        Up_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Up_Stream_Sensor_In_Plane, VIC_Path_lis)
        Down_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Down_Stream_Sensor_In_Plane, VIC_Path_lis)

        for i in range(min(len(Up_Stream_pathlis_raw), 
                        len(Down_Stream_pathlis_raw))):

            if os.path.split(Up_Stream_pathlis_raw[i])[0] == os.path.split(Down_Stream_pathlis_raw[i])[0]:
                if os.path.split(Up_Stream_pathlis_raw[i])[1].split('_')[1][0:2] == os.path.split(Down_Stream_pathlis_raw[i])[1].split('_')[1][0:2]:
                    Up_Stream_pathlis.append(Up_Stream_pathlis_raw[i])
                    Down_Stream_pathlis.append(Down_Stream_pathlis_raw[i])

    Up_Stream_Data = np.array([0])
    Down_Stream_Data = np.array([0])

    for i in range(len(Up_Stream_pathlis)):
        up_stream_fname = Up_Stream_pathlis[i]
        down_stream_fname = Down_Stream_pathlis[i]

        up_stream_data_slice = unpacker.File_Detach_Data(up_stream_fname, time_interval = 3)
        down_stream_data_slice = unpacker.File_Detach_Data(down_stream_fname, time_interval = 3)

        if len(up_stream_data_slice) == 0:
            continue

        if len(down_stream_data_slice) == 0:
            continue
        
        if up_stream_data_slice.shape[0] == 20 and down_stream_data_slice.shape[0] == 20:
            up_rms_slice = np.mean(np.square(up_stream_data_slice - np.mean(up_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)
            down_rms_slice = np.mean(np.square(down_stream_data_slice - np.mean(down_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)


            Up_Stream_Data = np.hstack((Up_Stream_Data, np.squeeze(up_rms_slice)))
            Down_Stream_Data = np.hstack((Down_Stream_Data, np.squeeze(down_rms_slice)))

    Up_Stream_Data = Up_Stream_Data[1:]
    Down_Stream_Data = Down_Stream_Data[1:]

    cache = Up_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    cache = Down_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    ploter = PlotLib()
    fig, ax = ploter.scatter(Up_Stream_Data, Down_Stream_Data, 
                             xlim = (0, 0.02), 
                             ylim = (0, 0.02), 
                             legend = 'In Plane Acceleration RMS'
                             )


    Up_Stream_pathlis = []
    Down_Stream_pathlis = [] 

    for Up_Stream_Sensor_In_Plane, Down_Stream_Sensor_In_Plane in VIC_Name_Lis_Up_Down_Stream_Out_Plane:

        Up_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Up_Stream_Sensor_In_Plane, VIC_Path_lis)
        Down_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Down_Stream_Sensor_In_Plane, VIC_Path_lis)

        for i in range(min(len(Up_Stream_pathlis_raw), 
                        len(Down_Stream_pathlis_raw))):

            if os.path.split(Up_Stream_pathlis_raw[i])[0] == os.path.split(Down_Stream_pathlis_raw[i])[0]:
                if os.path.split(Up_Stream_pathlis_raw[i])[1].split('_')[1][0:2] == os.path.split(Down_Stream_pathlis_raw[i])[1].split('_')[1][0:2]:
                    Up_Stream_pathlis.append(Up_Stream_pathlis_raw[i])
                    Down_Stream_pathlis.append(Down_Stream_pathlis_raw[i])

    Up_Stream_Data = np.array([0])
    Down_Stream_Data = np.array([0])

    for i in range(len(Up_Stream_pathlis)):
        up_stream_fname = Up_Stream_pathlis[i]
        down_stream_fname = Down_Stream_pathlis[i]

        up_stream_data_slice = unpacker.File_Detach_Data(up_stream_fname, time_interval = 3)
        down_stream_data_slice = unpacker.File_Detach_Data(down_stream_fname, time_interval = 3)

        if len(up_stream_data_slice) == 0:
            continue

        if len(down_stream_data_slice) == 0:
            continue
        
        if up_stream_data_slice.shape[0] == 20 and down_stream_data_slice.shape[0] == 20:
            up_rms_slice = np.mean(np.square(up_stream_data_slice - np.mean(up_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)
            down_rms_slice = np.mean(np.square(down_stream_data_slice - np.mean(down_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)

            Up_Stream_Data = np.hstack((Up_Stream_Data, np.squeeze(up_rms_slice)))
            Down_Stream_Data = np.hstack((Down_Stream_Data, np.squeeze(down_rms_slice)))

    Up_Stream_Data = Up_Stream_Data[1:]
    Down_Stream_Data = Down_Stream_Data[1:]

    cache = Up_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    cache = Down_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    ploter = PlotLib()
    fig, ax = ploter.scatter(Up_Stream_Data, Down_Stream_Data, 
                             title = 'Acceleration RMS Distribution',
                             xlabel = r'UpStream Acceleration $(m/s^2)$',
                             ylabel = r'DownStream Acceleration $(m/s^2)$', 
                             fig = fig,
                             ax = ax, 
                             color = 'pink', 
                             legend = 'Out Plane Acceleration RMS')
    


    ax.legend()
    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()

    return fig

def PSD_ECC():
    unpacker = UNPACK()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()


    # 上下游拉索加速度散点图
    VIC_Name_Lis_Up_Down_Stream_In_Plane = [
        'ST-VIC-C18-101-01', 
        'ST-VIC-C18-102-01', # 面内
        # 'ST-VIC-C18-401-01', 
        # 'ST-VIC-C18-402-01',
        # 'ST-VIC-C18-501-01', 
        # 'ST-VIC-C18-502-01',

    ]

    VIC_Name_Lis_Up_Down_Stream_Out_Plane = [
        'ST-VIC-C18-101-02', 
        'ST-VIC-C18-102-02', # 面外
        # 'ST-VIC-C18-401-02', 
        # 'ST-VIC-C18-402-02',
        # 'ST-VIC-C18-501-02', 
        # 'ST-VIC-C18-502-02',
    ]

    datax = np.array([0])
    datay = np.array([0])
    dataz = np.array([0])
    for sensor in VIC_Name_Lis_Up_Down_Stream_In_Plane:
        path_lis = unpacker.File_Match_Sensor_Path(sensor, VIC_Path_Lis)
        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            for data in data_lis:
                fx, Pxx_den = signal.welch(data, fs = 50, nfft = 512, 
                                           nperseg = 512, noverlap = 256)
                Pxx_den_max = np.sort(Pxx_den)[-10:]

                e1 = Pxx_den_max[-1]
                e2 = Pxx_den_max[-2]
                e3 = Pxx_den_max[-3]

                e2_e1 = e2 / e1
                e3_e1 = e3 / e1



                datax = np.hstack((datax, e2_e1))
                datay = np.hstack((datay, e3_e1))
                dataz = np.hstack((dataz, e2_e1))

    
    ploter = PlotLib()
    fig, ax = ploter.scatter(datax, datay, 
                             title = 'Energy Density Distribution',
                             xlabel = r'$e_2/e_1$',
                             ylabel = r'$e_3/e_1$', 
                             legend = 'In Plane',
                             xlim = (0, 1),
                             ylim = (0, 1),)
    ax.invert_xaxis()
    fig_hist, ax_hist = ploter.hist(dataz,
                             title = r'$e_2/e_1$ Distribution',
                             xlabel = r'$e_2/e_1$',
                             legend = 'In Plane',
    )
    ax_hist.invert_xaxis()

    datax = np.array([0])
    datay = np.array([0])
    dataz = np.array([0])
    for sensor in VIC_Name_Lis_Up_Down_Stream_Out_Plane:
        path_lis = unpacker.File_Match_Sensor_Path(sensor, VIC_Path_Lis)
        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            for data in data_lis:
                fx, Pxx_den = signal.welch(data, fs = 50, nfft = 512, 
                                           nperseg = 512, noverlap = 256)
                Pxx_den_max = np.sort(Pxx_den)[-10:]

                e1 = Pxx_den_max[-1]
                e2 = Pxx_den_max[-2]
                e3 = Pxx_den_max[-3]

                e2_e1 = e2 / e1

                e3_e1 = e3 / e1

                datax = np.hstack((datax, e2_e1))
                datay = np.hstack((datay, e3_e1))
                dataz = np.hstack((dataz, e2_e1))


    fig, ax = ploter.scatter(datax, datay, 
                             fig = fig, 
                             ax = ax,
                             legend = 'Out Plane',
                             color = 'pink')

    fig_hist, ax_hist = ploter.hist(dataz,
                             fig = fig_hist,
                             ax = ax_hist,
                             title = r'$e_2/e_1$ Distribution',
                             color = 'pink',
                             legend = 'Out Plane',
    )

    
    # 画出边界线
    x = np.linspace(0.2, 1, 100)
    # 这里是可以手动调整的两个点，可以依照自己想要的输入
    x1, y1 = 1, 0.50
    x2, y2 = -0.5, 0
    k, b = np.asarray(np.dot(np.linalg.inv(np.asmatrix(np.array([[x1, 1], [x2, 1]]))), np.asmatrix(np.array([[y1], [y2]]))))
    y = k * x + b
    # y <= k * x + b 为该图的下半区域，记得这点
    ax.plot(x, y, color = '#B7B7EB', linestyle = '--', label = f'Boundary: $e_3 = {k[0]:.2f}e_2 + {b[0]:.2f}e_1$')


    ax.legend()
    ax_hist.legend()
    scatter_fig = fig
    hist_fig = fig_hist

    return scatter_fig, hist_fig

def VIV_Filter():

    unpacker = UNPACK()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()


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


    # 取基频的经验值为1Hz
    f0 = 1 # Hz
    fs = 50
    Abnormal_Vibration_File_Lis_Inplane = []
    Abnormal_Vibration_File_Lis_Outplane = []

    Abnormal_Vibration_file = Abnormal_Vibration_Filter()

    for i in range(len(VIC_Name_Lis_Up_Down_Stream_In_Plane)):
        sensor_Inplane = VIC_Name_Lis_Up_Down_Stream_In_Plane[i]
        
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Inplane, VIC_Path_Lis)
        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            for data in data_lis:
                if Abnormal_Vibration_file.VIV_Filter(data, f0 = 1):
                    Abnormal_Vibration_File_Lis_Inplane.append(path)
                    continue

    for i in range(len(VIC_Name_Lis_Up_Down_Stream_Out_Plane)):
        sensor_Outplane = VIC_Name_Lis_Up_Down_Stream_Out_Plane[i]
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Outplane, VIC_Path_Lis)

        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            for data in data_lis:
                if Abnormal_Vibration_file.VIV_Filter(data, f0 = 1):
                    Abnormal_Vibration_File_Lis_Inplane.append(path)
                    continue
    df_Inplane = {
        'Inplane':Abnormal_Vibration_File_Lis_Inplane,
        
    }
    df_Inplane = pd.DataFrame(df_Inplane)
    df_Inplane.to_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Inplane_VIV.xlsx')

    df_Outplane = {
        'Outplane':Abnormal_Vibration_File_Lis_Outplane,
    }

    df_Outplane = pd.DataFrame(df_Outplane)
    df_Inplane.to_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Outplane_VIV.xlsx')
    

    return 

def VIV_statistic_Ek_E1():

    unpacker = UNPACK()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()

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

    Path_With_Time_Intervals = []

    # 取基频的经验值为1Hz
    f0 = 1 # Hz
    fs = 50

    ploter = PlotLib()
    ab_fil = Abnormal_Vibration_Filter()
    for i in range(len(VIC_Name_Lis_Up_Down_Stream_In_Plane)):
        sensor_Inplane = VIC_Name_Lis_Up_Down_Stream_In_Plane[i]
        
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Inplane, VIC_Path_Lis)
        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            for time, data in enumerate(data_lis):
                if ab_fil.VIV_Filter(data):
                    fx, Pxx_den = signal.welch(data, fs = 50, nfft = 512, 
                                            nperseg = 512, noverlap = 256)
                    if len(Pxx_den) == 0:
                        continue
                    E1 = np.max(Pxx_den)

                    index = np.where(Pxx_den == E1)
                    # 主导模态
                    f_major = fx[index]

                    f_major_left, f_major_right = f_major - 0.5 * f0, f_major + 0.5 * f0
                    # 主导模态 2倍基频以内不管
                    # 2倍基频以外寻找最大值
                    Pxx_den_left = Pxx_den[fx < f_major_left]
                    Pxx_den_right = Pxx_den[fx > f_major_right]

                    # 提取出Ek/E1的具体状态
                    # 主导模态应该大于10倍基频
                    x = np.arange(len(data)) / fs
                    data_lis = [
                        (x, data),
                        (fx, Pxx_den)
                        ]
                    
                    title = [
                        f'Time Series',
                        'Power Spectral Density'
                    ]

                    labels = [
                        ('Time(s)', 'Acceleration $(m/s^2)$'),
                        ('Frequency(hz)', 'PSD')
                    ]

                    Path_With_Time_Intervals.append((path, time, 'Inplane'))

                    save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\test\\'
                    
                    fx_left = fx[fx < f_major_left]
                    fx_right = fx[fx > f_major_right]
                    

                    Pxx_den_left = Pxx_den[fx < f_major_left]
                    Pxx_den_right = Pxx_den[fx > f_major_right]

                    file = Path(path)
                    Responsetitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + file.parts[-1][:-4].split('_')[1][:2] + '_TT' +  str(time) + '_' + file.parts[-1][:-4].split('_')[0] + 'Response'
                    save_path = save_root + Responsetitle + '.png'

                    fig, axes = ploter.plots(data_lis, titles = title, labels = labels)
                    fig, axes[1] = ploter.plot(Pxx_den_left, x = fx_left, fig = fig, ax = axes[1], color = 'pink')
                    fig, axes[1] = ploter.plot(Pxx_den_right, x = fx_right, fig = fig, ax = axes[1], color = 'pink')

                    fig.savefig(save_path)
                    plt.close()

    for i in range(len(VIC_Name_Lis_Up_Down_Stream_Out_Plane)):
        sensor_Outplane = VIC_Name_Lis_Up_Down_Stream_Out_Plane[i]
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Outplane, VIC_Path_Lis)

        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            for time, data in enumerate(data_lis):
                if ab_fil.VIV_Filter(data):
                    fx, Pxx_den = signal.welch(data, fs = 50, nfft = 512, 
                                            nperseg = 512, noverlap = 256)

                    E1 = np.max(Pxx_den)

                    index = np.where(Pxx_den == E1)
                    # 主导模态
                    f_major = fx[index]

                    f_major_left, f_major_right = f_major - 0.5 * f0, f_major + 0.5 * f0
                    # 主导模态 2倍基频以内不管
                    # 2倍基频以外寻找最大值
                    Pxx_den_left = Pxx_den[fx < f_major_left]
                    Pxx_den_right = Pxx_den[fx > f_major_right]
                    if len(Pxx_den) == 0:
                        continue
                    Ek = np.sort(np.hstack((Pxx_den_left, Pxx_den_right)))[-1]

                    # 提取出Ek/E1的具体状态
                    # 主导模态应该大于10倍基频
                    x = np.arange(len(data))
                    data_lis = [
                        (x, data),
                        (fx, Pxx_den)
                        ]
                    
                    title = [
                        f'Time Series Ek/E1',
                        'Power Spectral Density'
                    ]

                    labels = [
                        ('Time(s)', 'Acceleration $(m/s^2)$'),
                        ('Frequency(hz)', 'PSD')
                    ]

                    Path_With_Time_Intervals.append((path, time, 'Outplane'))
                    save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\test\\'
                    
                    fx_left = fx[fx < f_major_left]
                    fx_right = fx[fx > f_major_right]
                    

                    Pxx_den_left = Pxx_den[fx < f_major_left]
                    Pxx_den_right = Pxx_den[fx > f_major_right]

                    
                    file = Path(path)
                    Responsetitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + str(time) + '_TT' + file.parts[-1][:-4].split('_')[1][:2] + '_' + file.parts[-1][:-4].split('_')[0] + 'Response' 
                    save_path = save_root + Responsetitle + '.png'

                    fig, axes = ploter.plots(data_lis, titles = title, labels = labels)
                    fig, axes[1] = ploter.plot(Pxx_den_left, x = fx_left, fig = fig, ax = axes[1], color = 'pink')
                    fig, axes[1] = ploter.plot(Pxx_den_right, x = fx_right, fig = fig, ax = axes[1], color = 'pink')

                    fig.savefig(save_path)
                    plt.close()

    VIV_Occurence = {
        'path': [path for path, _, __  in Path_With_Time_Intervals],
        'time': [time for _, time, __  in Path_With_Time_Intervals],
        'Plane':[plane for _, __, plane  in Path_With_Time_Intervals],
    }
    df = pd.DataFrame(VIV_Occurence)
    df.to_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\VIV.xlsx')

    return 

def Wind_Rose_Map():

    root = r'F:\Research\My_Thesis\Data\苏通\UAN'
    unpacker = UNPACK()
    file_paths_lis = unpacker.File_Read_Paths(root)

    wind_velocities_RMS_lis = deque()
    wind_directions_lis = np.array([])


    for file_path in file_paths_lis:
        wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(file_path)

        # 取风速平均值，风向可不管
        # average_wind_velocity = np.mean(wind_velocity)
        wind_direction = wind_direction

        # wind_velocities_RMS_lis.append(average_wind_velocity)
        wind_directions_lis = np.hstack((wind_directions_lis, np.deg2rad(wind_direction)))

    # 如果风向数据大于360，则抛弃
    wind_directions_lis = wind_directions_lis[wind_directions_lis < 360]
    ploter = PlotLib()
    fig, ax = ploter.rose_hist(wind_directions_lis, separate_bins = 30)

    # 调整风向图的各个显示参数
    # 设置刻度标签
    x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticklabels(x_ticks)

    # 逆时针
    ax.set_theta_direction(-1)
    
    # 百分比
    yticks = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    ylabel = [str(int(i)) for i in np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]) * 100]
    ax.set_yticks(ticks = yticks, labels = ylabel)
    ax.set_yticklabels([])

    # 在 x = 45° 的位置添加 y 轴刻度值标签
    ylabel_position = np.deg2rad(360 - 45) 
    for y_tick in yticks:
        ax.text(ylabel_position, y_tick, f'{y_tick * 100:.0f}', va="center", ha="left")

    ax.text(-0.12, 0.5, "Percent Frequency (%)", rotation=90, transform=ax.transAxes, va="center", ha="center")


    # 桥梁中轴线
    axis_of_bridge = 10.6 # degree
    max_freq = np.deg2rad(max(wind_directions_lis))
    ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, 0.35], color =  'pink')
    ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, 0.35], color =  'pink')
    ax.annotate('Bridge axis', xy = (np.deg2rad(axis_of_bridge), 0.3), ha = 'center', va = 'bottom')
    return fig 

def VIV_Extract():

    df = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Inplane_VIV.xlsx')
    ploter = PlotLib()
    unpacker = UNPACK()
    fs = 50
    save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\Abnormal_Vibration_Ek_E1_0.1_0.5f0_Outplane\\'
    time_interval = 1

    x_slice = np.ones((1, time_interval * 60 * fs))
    for i in range(int(60 / time_interval)):
        x_data = np.arange(i* time_interval * 60 * fs, (i + 1) * time_interval * 60 * fs) / fs
        x_slice = np.vstack((x_slice, x_data))
    x_slice = x_slice[1:]
    for index in range(len(df)):
        In_Plane_File_Name = df['Inplane'][index]

        file = Path(In_Plane_File_Name)
        Responsetitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + file.parts[-1][:-4].split('_')[1][:2] + '_' + file.parts[-1][:-4].split('_')[0] + 'Response' 
        save_path = save_root + Responsetitle + '.mp4'

        data_slice = unpacker.File_Detach_Data(In_Plane_File_Name, time_interval = time_interval)

        nfft = 512
        PSD_slice = np.zeros((1, int(nfft / 2 + 1)))

        fx_slice = np.ones((1, int(nfft / 2 + 1)))

        for data_i in data_slice:
            fx, Pxx_Den = signal.welch(data_i, fs = 50, nfft = 512, nperseg = 512, noverlap = 256)
            PSD_slice = np.vstack((PSD_slice, Pxx_Den))
            fx_slice = np.vstack((fx_slice, fx))
        
        fx_slice = fx_slice[1:]
        PSD_slice = PSD_slice[1:]

        
        PSDtitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + file.parts[-1][:-4].split('_')[1][:2] + '_' + file.parts[-1][:-4].split('_')[0] + 'PSD'
        save_path = save_root + PSDtitle + '.mp4'

        data_lis = [
            (x_slice, data_slice),
            (fx_slice, PSD_slice)
        ]

        ploter.animate(data_lis, save_path = save_path,
                    title = ['Time Series Acceleration', 'Power Spectral Density'], 
                    labels = [('Time(s)', 'Acceleration $(m/s^2)$'), ('Frequency(Hz)', 'Power Spectral Density')])

    for index in range(len(df)):
        In_Plane_File_Name = df['Outplane'][index]

        file = Path(In_Plane_File_Name)
        Responsetitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + file.parts[-1][:-4].split('_')[1][:2] + '_' + file.parts[-1][:-4].split('_')[0] + 'Response' 
        save_path = save_root + Responsetitle + '.mp4'

        data_slice = unpacker.File_Detach_Data(In_Plane_File_Name, time_interval = time_interval)

        nfft = 512
        PSD_slice = np.zeros((1, int(nfft / 2 + 1)))

        fx_slice = np.ones((1, int(nfft / 2 + 1)))

        for data_i in data_slice:
            fx, Pxx_Den = signal.welch(data_i, fs = 50, nfft = 512, nperseg = 512, noverlap = 256)
            PSD_slice = np.vstack((PSD_slice, Pxx_Den))
            fx_slice = np.vstack((fx_slice, fx))
        
        fx_slice = fx_slice[1:]
        PSD_slice = PSD_slice[1:]

        
        PSDtitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + file.parts[-1][:-4].split('_')[1][:2] + '_' + file.parts[-1][:-4].split('_')[0] + 'PSD'
        save_path = save_root + PSDtitle + '.mp4'

        data_lis = [
            (x_slice, data_slice),
            (fx_slice, PSD_slice)
        ]

        ploter.animate(data_lis, save_path = save_path,
                    title = ['Time Series Acceleration', 'Power Spectral Density'], 
                    labels = [('Time(s)', 'Acceleration $(m/s^2)$'), ('Frequency(Hz)', 'Power Spectral Density')])
    
    return 

def In_Out_Plane_Figure_With_VIV():

    # Up_Stream_Sensor_In_Plane = []
    # Down_Stream_Sensor_In_Plane = []
    root = 'F:/Research/My_Thesis/Data/苏通/VIC/'
    unpacker = UNPACK()
    VIC_Path_lis = unpacker.File_Read_Paths(root)


    # 上下游拉索加速度散点图
    VIC_Name_Lis_Up_Down_Stream_In_Plane = [
        ('ST-VIC-C18-101-01', 'ST-VIC-C18-102-01'), # 面内
        ('ST-VIC-C18-401-01', 'ST-VIC-C18-402-01'),
        ('ST-VIC-C18-501-01', 'ST-VIC-C18-502-01'),
        

    ]

    VIC_Name_Lis_Up_Down_Stream_Out_Plane = [
        ('ST-VIC-C18-101-02', 'ST-VIC-C18-102-02'), # 面外
        ('ST-VIC-C18-401-02', 'ST-VIC-C18-402-02'),
        ('ST-VIC-C18-501-02', 'ST-VIC-C18-502-02'),
    ]
    Up_Stream_pathlis = []
    Down_Stream_pathlis = [] 

    Abnormal_fil = Abnormal_Vibration_Filter()

    for Up_Stream_Sensor_In_Plane, Down_Stream_Sensor_In_Plane in VIC_Name_Lis_Up_Down_Stream_In_Plane:

        Up_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Up_Stream_Sensor_In_Plane, VIC_Path_lis)
        Down_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Down_Stream_Sensor_In_Plane, VIC_Path_lis)

        for i in range(min(len(Up_Stream_pathlis_raw), 
                        len(Down_Stream_pathlis_raw))):

            if os.path.split(Up_Stream_pathlis_raw[i])[0] == os.path.split(Down_Stream_pathlis_raw[i])[0]:
                if os.path.split(Up_Stream_pathlis_raw[i])[1].split('_')[1][0:2] == os.path.split(Down_Stream_pathlis_raw[i])[1].split('_')[1][0:2]:
                    Up_Stream_pathlis.append(Up_Stream_pathlis_raw[i])
                    Down_Stream_pathlis.append(Down_Stream_pathlis_raw[i])

    Up_Stream_Data = np.array([0])
    Down_Stream_Data = np.array([0])

    for i in range(len(Up_Stream_pathlis)):
        up_stream_fname = Up_Stream_pathlis[i]
        down_stream_fname = Down_Stream_pathlis[i]

        up_stream_data_slice = unpacker.File_Detach_Data(up_stream_fname, time_interval = 3)
        down_stream_data_slice = unpacker.File_Detach_Data(down_stream_fname, time_interval = 3)

        if len(up_stream_data_slice) == 0:
            continue

        if len(down_stream_data_slice) == 0:
            continue
        
        if up_stream_data_slice.shape[0] == 20 and down_stream_data_slice.shape[0] == 20:
            up_rms_slice = np.mean(np.square(up_stream_data_slice - np.mean(up_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)
            down_rms_slice = np.mean(np.square(down_stream_data_slice - np.mean(down_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)


            Up_Stream_Data = np.hstack((Up_Stream_Data, np.squeeze(up_rms_slice)))
            Down_Stream_Data = np.hstack((Down_Stream_Data, np.squeeze(down_rms_slice)))

    Up_Stream_Data = Up_Stream_Data[1:]
    Down_Stream_Data = Down_Stream_Data[1:]

    cache = Up_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    cache = Down_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    ploter = PlotLib()
    fig, ax = ploter.scatter(Up_Stream_Data, Down_Stream_Data, 
                             xlim = (-0.1, 4), 
                             ylim = (-0.1, 4), 
                             legend = 'InPlane Acceleration RMS'
                             )


    Up_Stream_pathlis = []
    Down_Stream_pathlis = [] 

    for Up_Stream_Sensor_In_Plane, Down_Stream_Sensor_In_Plane in VIC_Name_Lis_Up_Down_Stream_Out_Plane:

        Up_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Up_Stream_Sensor_In_Plane, VIC_Path_lis)
        Down_Stream_pathlis_raw = unpacker.File_Match_Sensor_Path(Down_Stream_Sensor_In_Plane, VIC_Path_lis)

        for i in range(min(len(Up_Stream_pathlis_raw), 
                        len(Down_Stream_pathlis_raw))):

            if os.path.split(Up_Stream_pathlis_raw[i])[0] == os.path.split(Down_Stream_pathlis_raw[i])[0]:
                if os.path.split(Up_Stream_pathlis_raw[i])[1].split('_')[1][0:2] == os.path.split(Down_Stream_pathlis_raw[i])[1].split('_')[1][0:2]:
                    Up_Stream_pathlis.append(Up_Stream_pathlis_raw[i])
                    Down_Stream_pathlis.append(Down_Stream_pathlis_raw[i])

    Up_Stream_Data = np.array([0])
    Down_Stream_Data = np.array([0])

    for i in range(len(Up_Stream_pathlis)):
        up_stream_fname = Up_Stream_pathlis[i]
        down_stream_fname = Down_Stream_pathlis[i]

        up_stream_data_slice = unpacker.File_Detach_Data(up_stream_fname, time_interval = 3)
        down_stream_data_slice = unpacker.File_Detach_Data(down_stream_fname, time_interval = 3)

        if len(up_stream_data_slice) == 0:
            continue

        if len(down_stream_data_slice) == 0:
            continue
        
        if up_stream_data_slice.shape[0] == 20 and down_stream_data_slice.shape[0] == 20:
            up_rms_slice = np.mean(np.square(up_stream_data_slice - np.mean(up_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)
            down_rms_slice = np.mean(np.square(down_stream_data_slice - np.mean(down_stream_data_slice, axis = 1)[:, np.newaxis]), axis = 1)

            Up_Stream_Data = np.hstack((Up_Stream_Data, np.squeeze(up_rms_slice)))
            Down_Stream_Data = np.hstack((Down_Stream_Data, np.squeeze(down_rms_slice)))

    Up_Stream_Data = Up_Stream_Data[1:]
    Down_Stream_Data = Down_Stream_Data[1:]

    cache = Up_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    cache = Down_Stream_Data
    Up_Stream_Data = Up_Stream_Data[cache > 0.0002]
    Down_Stream_Data = Down_Stream_Data[cache > 0.0002]

    ploter = PlotLib()
    fig, ax = ploter.scatter(Up_Stream_Data, Down_Stream_Data, 
                             title = 'Acceleration RMS Distribution',
                             xlabel = r'UpStream Acceleration $(m/s^2)$',
                             ylabel = r'DownStream Acceleration $(m/s^2)$', 
                             fig = fig,
                             ax = ax, 
                             color = 'pink', 
                             legend = 'OutPlane Acceleration RMS')

    df = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIV.xlsx')
    VIV_Inplane_Data_Up_Stream = []
    VIV_Inplane_Data_Down_Stream = []

    VIV_Outplane_Data_Up_Stream = []
    VIV_Outplane_Data_Down_Stream = []

    for path, time ,plane in df.values:
        time_interval = 1
        counter_VICpath, Up_Or_Down = unpacker.VIC_path_UpStream_DownStream(path)
        if len(counter_VICpath) == 1:
            Response_Data = unpacker.File_Detach_Data(path, time_interval = 1)[time: time + time_interval]
            Counter_Response_Data= unpacker.File_Detach_Data(counter_VICpath[0], time_interval = 1)[time: time + time_interval]

            Response_Data_rms = np.mean(np.square(Response_Data - np.mean(Response_Data)))
            Counter_Response_Data_rms = np.mean(np.square(Counter_Response_Data - np.mean(Counter_Response_Data)))

            if plane == 'Inplane':
                if Up_Or_Down == 'Upstream':
                    VIV_Inplane_Data_Up_Stream.append(Response_Data_rms)
                    VIV_Inplane_Data_Down_Stream.append(Counter_Response_Data_rms)

                elif Up_Or_Down == 'Downstream':
                    VIV_Inplane_Data_Up_Stream.append(Counter_Response_Data_rms)
                    VIV_Inplane_Data_Down_Stream.append(Response_Data_rms)


            if plane == 'Outplane':
                if Up_Or_Down == 'Upstream':
                    VIV_Outplane_Data_Up_Stream.append(Response_Data_rms)
                    VIV_Outplane_Data_Down_Stream.append(Counter_Response_Data_rms)

                elif Up_Or_Down == 'Downstream':
                    VIV_Outplane_Data_Up_Stream.append(Counter_Response_Data_rms)
                    VIV_Outplane_Data_Down_Stream.append(Response_Data_rms)

    VIV_Inplane_Data_Up_Stream = np.array(VIV_Inplane_Data_Up_Stream)
    VIV_Inplane_Data_Down_Stream = np.array(VIV_Inplane_Data_Down_Stream)

    VIV_Outplane_Data_Up_Stream = np.array(VIV_Outplane_Data_Up_Stream)
    VIV_Outplane_Data_Down_Stream = np.array(VIV_Outplane_Data_Down_Stream)

    fig, ax = ploter.scatter(VIV_Inplane_Data_Up_Stream, VIV_Inplane_Data_Down_Stream, 
                             fig = fig,
                             ax = ax, 
                             color = '#8074CB', 
                             marker = '^',
                             s = 1,
                             legend = 'InPlane Acceleration RMS (VIV)')
    
    fig, ax = ploter.scatter(VIV_Outplane_Data_Up_Stream, VIV_Outplane_Data_Down_Stream, 
                             fig = fig,
                             ax = ax, 
                             color = '#E3625D', 
                             marker = '^',
                             s = 1,
                             legend = 'OutPlane Acceleration RMS (VIV)')
    ax.legend()
    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()

def VIV_Wind_Rose_Map():
    # 1. 对齐参考代码格式：核心参数定义
    interval_nums = 36  # 区间数量（参考代码统一用36）
    normalize = True    # 归一化开关（参考代码默认开启）
    axis_of_bridge = 10.6  # 桥轴线角度（参考代码固定值）
    
    # 目标传感器：北索塔塔顶、跨中桥面上游（对应参考代码的传感器命名）
    target_sensors = [
        'ST-UAN-T01-003-01',  # 北索塔塔顶
        'ST-UAN-G04-001-01'   # 跨中桥面上游
    ]
    sensor_names = [
        'VIV Wind Distribution (Top Of North Pylon)',
        'VIV Wind Distribution (Upstream of Mid-Span)'
    ]
    # 风向修正规则（参考代码区分传感器）
    wind_dir_correction = {
        'ST-UAN-T01-003-01': 180,  # 北索塔塔顶：180 - 原始风向
        'ST-UAN-G04-001-01': 360   # 跨中桥面上游：360 - 原始风向
    }
    
    # 路径配置
    VIV_excel_path = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\Img\VIV.xlsx'
    wind_data_root = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\UAN\\'
    
    # 初始化工具类
    unpacker = UNPACK()
    figs = []  # 存储VIV相关玫瑰图，用于tk交互

    # 读取VIV数据表格（核心：仅处理VIV发生时的风数据）
    df_viv = pd.read_excel(VIV_excel_path)
    time_interval = 1  # 分钟，取VIV发生时段1min内的风数据
    fs = 1  # 采样频率1Hz
    target_sensor_id = 'ST-VIC-C18-102-01'  # VIV传感器ID

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
        cmap = plt.cm.plasma  # 类viridis的渐变色板（替代原Reds，解决过浅问题）
        norm = plt.Normalize(min(average_speeds), max(average_speeds))
        # 循环赋值颜色（参考代码zip方式）
        for r, bar, color in zip(average_speeds, bars, cmap(norm(average_speeds))):
            bar.set_facecolor(color)

        # 对齐参考代码：添加颜色条（vertical，pad=0.1）
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation='vertical', label='Average Wind Speed (m/s)', pad=0.1)

        # 6. 对齐参考代码：图像处理/美化（核心格式对齐）
        # 地理坐标标签（N/NE/E/SE/S/SW/W/NW）
        x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ax.set_xticklabels(x_ticks)

        # 桥轴线绘制（参考代码：axis_of_bridge=10.6度，红色虚线）
        y_max = np.max(counts)
        ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, y_max * 1.1], color='red', linestyle='--')
        ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, y_max * 1.1], color='red', linestyle='--')
        ax.annotate('Bridge Axis', xy=(np.deg2rad(axis_of_bridge), y_max * 0.9), ha='center', va='bottom')
        
        # 降低y轴刻度密度（参考代码：MultipleLocator）
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.round(y_max * 0.25, 2)))
        # 调整y轴最大值（略微大于柱状图最大值）
        ax.set_ylim(0, y_max * 1.1)
        # y轴刻度转百分比（参考代码格式）
        ax.set_yticks(np.hstack([0, ax.get_yticks()[1:]]))
        ax.set_yticklabels([str(int(np.round(max(0, num * 100)))) + "%" for num in ax.get_yticks()])
        # 调整y轴标签位置（左侧）
        ax.yaxis.set_label_coords(-0.1, 1.1)
        # 调整y轴位置到玫瑰图左侧（参考代码：180+90）
        ax.set_rlabel_position(180 + 90)

        # 设置图表标题（参考代码样式）
        ax.set_title(sensor_names[idx], va='bottom')

        # 存储图表
        figs.append(fig)
        plt.close()  # 释放内存

    # 7. 保留tkinter交互逻辑（原代码核心逻辑）
    root = tk.Tk()
    root.title("VIV Wind Rose Chart (Aligned with Reference Format)")
    root.configure(bg='#f0f0f0')
    # 实例化ChartApp展示图表（确保ChartApp支持多图展示）
    app = ChartApp(root, figs)
    root.mainloop()

    return

def Wind_Rose_Map_With_VIV():
    sensor = [
              'ST-UAN-T01-003-01', # 北索塔塔顶
              'ST-UAN-T02-003-01', # 南索塔塔顶
              'ST-UAN-G04-001-01', # 跨中桥面上游
              'ST-UAN-G04-002-01', # 跨中桥面下游
               ]

    name = [
        'Wind Direction Distribution (Top Of North Pylon)',
        'Wind Direction Distribution (Top Of South Pylon)',
        'Wind Direction Distribution (Upstream of Mid-Span)',
        'Wind Direction Distribution (DownStream of Mid-Span)'
    ]
    VIV_excel = r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\Img\VIV.xlsx'
    df = pd.read_excel(VIV_excel)
    
    unpacker = UNPACK()
    path = r'F:\Research\Vibration Characteristics In Cable Vibration\data\2024September\SuTong\UAN\\'
    paths = unpacker.File_Read_Paths(path)
    figs = []
    for k in range(len(sensor)):

        file_paths_lis = unpacker.File_Match_Sensor_Path(pattern = sensor[k], paths = paths)

        print(np.mean(unpacker.Wind_Data_Unpack(file_paths_lis[0])[0]))

        wind_direction_lis = np.array([0])
        wind_velocity_lis = np.array([0])
        for file_path in file_paths_lis:
            # 60 * 60s 每秒的风速
            wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(file_path)
            if len(wind_direction) == len(wind_velocity):

                wind_direction_lis = np.hstack((wind_direction_lis, np.deg2rad(wind_direction)))
                wind_velocity_lis = np.hstack((wind_velocity_lis, wind_velocity))
            else:
                min_length = np.max(len(wind_velocity), len(wind_direction))
                wind_direction_lis = np.hstack((wind_direction_lis, np.deg2rad(wind_direction[:min_length])))
                wind_velocity_lis = np.hstack((wind_velocity_lis, np.deg2rad(wind_velocity[:min_length])))


        # wind_direction_lis = 2 * np.pi - wind_direction_lis[1:]
        # wind_velocity_lis = 2 * np.pi - wind_velocity_lis[1:]
        wind_direction_lis = wind_direction_lis[1:]
        wind_velocity_lis = wind_velocity_lis[1:]
        # 获取VIV的风速分布图

        ploter = PlotLib()
        fig, ax = ploter.rose_hist(wind_direction_lis, 
                                   title = name[k],
                                   separate_bins = 50,
                                   geographic_orientation = True, 
                                   bridge_axis = True)

        

        # ax.legend()
        figs.append(fig)

        # fig, ax = ploter.scatter(x = wind_direction_lis, 
        #                          y = wind_velocity_lis,
        #                          xlabel = 'Mean Wind Direction (rad)',
        #                          ylabel = 'Mean Wind Velocity $(m/s)$',
        #                         title = name[k],)
        
        # figs.append(fig)

        plt.close()

    VIV_wind_velocity_lis = []
    VIV_wind_direction_lis = []
    df = pd.read_excel(r'F:\Research\Vibration Characteristics In Cable Vibration\之前的代码\之前的结果\Img\VIV.xlsx')
    time_interval = 1 # min
    fs = 1
    # 做了传感器筛选
    # 多个传感器不能相互干扰
    sensor_id = 'ST-VIC-C18-102-01'

    for path, time ,plane in df.values:
        VICpath = path
        Wind_path = unpacker.VIC_Path_2_WindPath(VICpath = VICpath, wind_sensor_ids = [sensor[k]])
        if len(Wind_path) == 1 and Path(path).parts[-1].split('_')[0] == sensor_id: 
            wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(Wind_path[0])

            VIV_wind_velocity_mean = np.mean(wind_velocity[time : time + time_interval * 60 * fs])
            VIV_wind_direction_mean = np.deg2rad(np.mean(wind_direction[time : time + time_interval * 60 * fs]))
            VIV_wind_velocity_lis.append(VIV_wind_velocity_mean)
            VIV_wind_direction_lis.append(VIV_wind_direction_mean)

    # 风速
    VIV_wind_velocity_lis = np.array(VIV_wind_velocity_lis)
    fig, ax = ploter.hist(VIV_wind_velocity_lis, 
                        bins = 20,
                        density = True, 
                        title = 'Mean Wind Velocity Speed Distribution',
                        xlabel = 'Mean Wind Velocity $(m/s)$',
                        ylabel = 'Frequency Count',
                        color = 'pink')
            
    figs.append(fig)

    # 风向
    VIV_wind_direction_lis = np.array(VIV_wind_direction_lis)
    fig, ax = ploter.rose_hist(VIV_wind_direction_lis, 
                                separate_bins = 20,
                                density = True, 
                                color = 'pink', 
                                geographic_orientation = True,
                                bridge_axis = True)
            
    figs.append(fig)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

    return 

def Wind_Turbulence_with_VIV():

    sensor = [
              'ST-UAN-T01-003-01', # 北索塔塔顶
            #   'ST-UAN-T02-003-01', # 南索塔塔顶
              'ST-UAN-G04-001-01', # 跨中桥面上游
            #   'ST-UAN-G04-002-01', # 跨中桥面下游
               ]

    name = [
        'Wind Direction Distribution (Top Of North Pylon)',
        # 'Wind Direction Distribution (Top Of South Pylon)',
        'Wind Direction Distribution (Upstream of Mid-Span)',
        # 'Wind Direction Distribution (DownStream of Mid-Span)'
    ]

    unpacker = UNPACK()
    path = r'F:\Research\My_Thesis\Data\苏通\UAN\\'
    paths = unpacker.File_Read_Paths(path)
    figs = []
    time_interval = 1
    fs = 1
    num_points = time_interval * fs * 60 # s
    ploter = PlotLib()

    for k in range(len(sensor)):

        file_paths_lis = unpacker.File_Match_Sensor_Path(pattern = sensor[k], paths = paths)

        wind_velocity_lis_mean = np.array([0])
        wind_TI_lis = np.array([0])
        for file_path in file_paths_lis:
            # # 60 * 60s 每秒的风速
            # wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(file_path)
            for wind_velocity in unpacker.File_Detach_Data(file_path, time_interval = 1, mode = 'UAN')[0]:
            

                wind_velocity_mean = np.mean(wind_velocity)
                wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
                Wind_TI = wind_velocity_rms / wind_velocity_mean

                # 平均值和紊流度
                wind_velocity_lis_mean = np.hstack((wind_velocity_lis_mean, wind_velocity_mean))
                wind_TI_lis = np.hstack((wind_TI_lis, Wind_TI))


        wind_velocity_lis_mean = wind_velocity_lis_mean[1:]
        wind_TI_lis = wind_TI_lis[1:]

        fig, ax = ploter.scatter(
            x = wind_velocity_lis_mean, 
            y = wind_TI_lis, 
            legend = 'Normal Sample',
        )

        if k == 0:
            VIV_Mean = np.array([0])
            VIV_Turbulence = np.array([0])
            df = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIV.xlsx')
            time_interval = 1 # min

            for path, time ,plane in df.values:
                VICpath = path
                Wind_path = unpacker.VIC_Path_2_WindPath(VICpath = VICpath, wind_sensor_ids = [sensor[k]])
                if len(Wind_path) == 1:
                    wind_velocity_slice = np.array(unpacker.File_Detach_Data(file_path, time_interval = 1, mode = 'UAN')[0][:-1])

                    wind_velocity_mean = np.mean(wind_velocity_slice, axis = 1)
                    wind_velocity_rms = np.sqrt(np.mean((wind_velocity_slice - wind_velocity_mean[:, np.newaxis]) ** 2, axis=1))
                    Wind_TI = wind_velocity_rms / wind_velocity_mean

                    # 平均值和紊流度
                    VIV_Mean = np.hstack((VIV_Mean, wind_velocity_mean))
                    VIV_Turbulence = np.hstack((VIV_Turbulence, Wind_TI))

            VIV_Mean = VIV_Mean[1:]
            VIV_Turbulence = VIV_Turbulence[1:]

            fig, ax = ploter.scatter(
                x = VIV_Mean, 
                y = VIV_Turbulence, 
                title = 'Wind Turbulence Distribution',
                xlabel = 'Mean Wind Velocity $(m/s)$',
                ylabel = 'Wind Turbulence',
                color = 'pink',
                legend = 'VIV Sample',
                fig = fig, 
                ax = ax,
                ylim = (0, 0.5),
                marker = '^',
                s = 2,

            )

            ax.legend()
            figs.append(fig)
    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()


    return 

def Response_RMS_and_Wind_Velocity():
    unpacker = UNPACK()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()

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

    sensor = [
              'ST-UAN-T01-003-01', # 北索塔塔顶
              'ST-UAN-T02-003-01', # 南索塔塔顶
              'ST-UAN-G04-001-01', # 跨中桥面上游
              'ST-UAN-G04-002-01', # 跨中桥面下游
               ]

    # 取基频的经验值为1Hz
    f0 = 1 # Hz
    fs = 50
    ploter = PlotLib()

    data_x = []
    data_y = []
    for i in range(len(VIC_Name_Lis_Up_Down_Stream_In_Plane)):
        sensor_Inplane = VIC_Name_Lis_Up_Down_Stream_In_Plane[i]
        
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Inplane, VIC_Path_Lis)
        for path in path_lis:
            # 默认使用北索塔的风速
            wind_path = unpacker.VIC_Path_2_WindPath(VICpath = path)
            if len(wind_path) == 0:
                continue
            Response_data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            Wind_Velocity_lis = unpacker.File_Detach_Data(path = wind_path[0], time_interval = 1, mode = 'UAN')[0]
            
            for i in range(np.min((len(Response_data_lis), len(Wind_Velocity_lis)))):
                Response_data = Response_data_lis[i]
                wind_velocity_mean = np.mean(Wind_Velocity_lis[i])

                Response_rms = np.sqrt(np.mean((Response_data - np.mean(Response_data)) ** 2))

                data_x.append(wind_velocity_mean)
                data_y.append(Response_rms)

    fig, ax = ploter.scatter(
        x = data_x, 
        y = data_y, 
        title = 'RMS Distribution',
        xlabel = 'Mean Wind Velocity $(m/s)$',
        ylabel = 'Acceleration RMS',
        legend = 'Inplane',
    )

    data_x = []
    data_y = []
    for i in range(len(VIC_Name_Lis_Up_Down_Stream_Out_Plane)):
        sensor_Inplane = VIC_Name_Lis_Up_Down_Stream_Out_Plane[i]
        
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Inplane, VIC_Path_Lis)
        for path in path_lis:
            # 默认使用北索塔的风速
            wind_path = unpacker.VIC_Path_2_WindPath(VICpath = path)
            if len(wind_path) == 0:
                continue
            Response_data_lis = unpacker.File_Detach_Data(path, time_interval = 1)
            Wind_Velocity_lis = unpacker.File_Detach_Data(path = wind_path[0], time_interval = 1, mode = 'UAN')[0]
            
            for i in range(np.min((len(Response_data_lis), len(Wind_Velocity_lis)))):
                Response_data = Response_data_lis[i]
                wind_velocity_mean = np.mean(Wind_Velocity_lis[i])

                Response_rms = np.sqrt(np.mean((Response_data - np.mean(Response_data))))

                data_x.append(wind_velocity_mean)
                data_y.append(Response_rms)

    fig, ax = ploter.scatter(
        x = data_x, 
        y = data_y, 
        fig = fig,
        ax = ax,
        legend = 'Outplane',
        color = 'pink'
    )

    ax.legend()
    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()

    return 

def main():
    figs = []
    

    # figs.append(In_Out_Plane_Figure())
    # figs.extend(PSD_ECC())
    # figs.append(Wind_Rose_Map())

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

    return 
    
def f_major_statistic():
    # 统计涡激振动主导模态
    unpacker = UNPACK()
    ploter = PlotLib()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()

    df = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIV.xlsx')
    time_interval = 1
    fs = 50

    f_majors = np.array([0])
    # 统计涡激的主导模态
    for path, time, In_Out in df.values:
        Res_Acceleration = unpacker.File_Detach_Data(path, time_interval = 1)[time : time + time_interval]
        fx, Pxx_Den = signal.welch(Res_Acceleration, fs = 50, nfft = 512, nperseg = 512, noverlap = 256)
        Pxx_Den = np.squeeze(Pxx_Den)
        Pxx_Den_max = np.max(Pxx_Den)
        f_major = fx[Pxx_Den == Pxx_Den_max]

        f_majors = np.hstack((f_majors, f_major))
    
    f_majors = f_majors[1:]
    

    fig, ax = ploter.hist(
                        x = f_majors,
                        bins = 100, 
                        density = True,
                        title = 'Dominant Mode Statistical Plot',
                        xlabel = 'Frequency(Hz)',
                        ylabel = 'Frequency Count',
                        color = 'pink',
                    )

    

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()

def VIV_RMS_Statistic_Filter():
    # 
    unpacker = UNPACK()
    ploter = PlotLib()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()

    df = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIV.xlsx')
    time_interval = 1
    fs = 50

    RMSs = np.array([0])
    figs = []
    for path, time, In_Out in df.values:
        Res_Acceleration = unpacker.File_Detach_Data(path, time_interval = 1)[time]
        rms = np.sqrt(np.mean((Res_Acceleration - np.mean(Res_Acceleration)) ** 2))
        RMSs = np.hstack((RMSs, rms))

        if rms > 0.3:
            Res_Acceleration = Res_Acceleration
            time = np.arange(len(Res_Acceleration)) / 60 / fs

            fx, Pxx_Den = signal.welch(Res_Acceleration, fs = 50, nfft = 512, nperseg = 512, noverlap = 256)
            

            data = [
                (time, Res_Acceleration),
                (fx, Pxx_Den)
            ]

                    
            titles = [
                f'Time Series',
                'Power Spectral Density'
            ]

            labels = [
                ('Time(s)', 'Acceleration $(m/s^2)$'),
                ('Frequency(hz)', 'PSD')
            ]

            fig_i, ax_i = ploter.plots(
                data_lis = data,
                titles = titles,
                labels = labels,
            )

            figs.append(fig_i)
            plt.close()
    
    RMSs = RMSs[1:]
    

    fig, ax = ploter.hist(
                        x = RMSs,
                        bins = 100, 
                        density = True,
                        title = 'RMS Statistical Plot',
                        xlabel = 'RMS',
                        ylabel = 'Frequency Count',
                    )

    figs.append(fig)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

def MECC_Filter():


    # Up_Stream_Sensor_In_Plane = []
    # Down_Stream_Sensor_In_Plane = []
    root = 'F:/Research/My_Thesis/Data/苏通/VIC/'
    unpacker = UNPACK()
    VIC_Path_lis = unpacker.File_Read_Paths(root)
    filt = Abnormal_Vibration_Filter()

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
    VIV_Path_Lis = []
    time_interval = 1
    ploter = PlotLib()
    fs = 50
    figs = []

    for path in VIC_Path_lis:
        data_slice = unpacker.File_Detach_Data(path, time_interval = time_interval, mode = 'VIC')

        in_or_out = None

        sensor_id = Path(path).parts[-1].split('_')[0]
        if sensor_id in VIC_Name_Lis_Up_Down_Stream_In_Plane:
            in_or_out = 'Inplane'  
        elif sensor_id in VIC_Name_Lis_Up_Down_Stream_Out_Plane:
            in_or_out = 'Outplane'  

        for i, data in enumerate(data_slice):
            if filt.VIV_Filter(data) and in_or_out != None:
                
                time = i * time_interval
                VIV_Path_Lis.append((path, time, in_or_out))

                Res_Acceleration = data
                time = np.arange(len(Res_Acceleration)) / 60 / fs
                fx, Pxx_Den = signal.welch(Res_Acceleration, fs = 50, nfft = 512, nperseg = 512, noverlap = 256)
                

                data = [
                    (time, Res_Acceleration),
                    (fx, Pxx_Den)
                ]

                    
                titles = [
                    f'Time Series',
                    'Power Spectral Density'
                ]

                labels = [
                    ('Time(s)', 'Acceleration $(m/s^2)$'),
                    ('Frequency(hz)', 'PSD')
                ]

                fig_i, ax_i = ploter.plots(
                    data_lis = data,
                    titles = titles,
                    labels = labels,
                )

                figs.append(fig_i)
                plt.close()
    
    df = pd.DataFrame(VIV_Path_Lis, columns = ['path', 'time', 'Plane'])
    df.to_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\test.xlsx')

    figs.append(figs)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

    return 

def Wind_Velocity_Hist():

    sensor = [
              'ST-UAN-T01-003-01', # 北索塔塔顶
              'ST-UAN-T02-003-01', # 南索塔塔顶
              'ST-UAN-G04-001-01', # 跨中桥面上游
              'ST-UAN-G04-002-01', # 跨中桥面下游
               ]

    unpacker = UNPACK()
    path = r'F:\Research\My_Thesis\Data\苏通\UAN\\'
    paths = unpacker.File_Read_Paths(path)
    figs = []
    time_interval = 1
    fs = 1
    num_points = time_interval * fs * 60 # s
    ploter = PlotLib()

    for k in range(len(sensor)):

        file_paths_lis = unpacker.File_Match_Sensor_Path(pattern = sensor[k], paths = paths)

        wind_velocity_lis_mean = np.array([0])
        wind_TI_lis = np.array([0])
        for file_path in file_paths_lis:
            # # 60 * 60s 每秒的风速
            # wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(file_path)
            for wind_velocity in unpacker.File_Detach_Data(file_path, time_interval = 1, mode = 'UAN')[0]:
            

                wind_velocity_mean = np.mean(wind_velocity)
                wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
                Wind_TI = wind_velocity_rms / wind_velocity_mean

                # 平均值和紊流度
                wind_velocity_lis_mean = np.hstack((wind_velocity_lis_mean, wind_velocity_mean))
                wind_TI_lis = np.hstack((wind_TI_lis, Wind_TI))


        wind_velocity_lis_mean = wind_velocity_lis_mean[1:]
        wind_TI_lis = wind_TI_lis[1:]

        fig, ax = ploter.hist(
            x = wind_velocity_lis_mean, 
            bins = 50, 
            title = 'Wind Velocity Distribution',
            xlabel = 'Wind Velocity (m/s)',
            ylabel = 'Frequency Count',
            # xlim = (0, 30)
        )

        figs.append(fig)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

def Wind_Turbulence_Scatter():

    sensor = [
              'ST-UAN-T01-003-01', # 北索塔塔顶
              'ST-UAN-T02-003-01', # 南索塔塔顶
              'ST-UAN-G04-001-01', # 跨中桥面上游
               ]
    append_sensor = [
        'ST-UAN-G02-002-01',
        'ST-UAN-G02-002-02',
        'ST-UAN-G04-002-01', # 跨中桥面下游
    ]
    legend = [None, None, 'UpStream Samples' 'DownStream Samples' ]
    unpacker = UNPACK()
    path = r'F:\Research\My_Thesis\Data\苏通\UAN\\'
    paths = unpacker.File_Read_Paths(path)
    figs = []
    time_interval = 1
    fs = 1
    num_points = time_interval * fs * 60 # s
    ploter = PlotLib()
    figs = []
    for k in range(len(sensor)):

        file_paths_lis = unpacker.File_Match_Sensor_Path(pattern = sensor[k], paths = paths)

        wind_velocity_lis_mean = np.array([0])
        wind_TI_lis = np.array([0])
        for file_path in file_paths_lis:
            # # 60 * 60s 每秒的风速
            # wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(file_path)
            for wind_velocity in unpacker.File_Detach_Data(file_path, time_interval = 1, mode = 'UAN')[0]:
            

                wind_velocity_mean = np.mean(wind_velocity)
                wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
                Wind_TI = wind_velocity_rms / wind_velocity_mean

                # 平均值和紊流度
                wind_velocity_lis_mean = np.hstack((wind_velocity_lis_mean, wind_velocity_mean))
                wind_TI_lis = np.hstack((wind_TI_lis, Wind_TI))


        wind_velocity_lis_mean = wind_velocity_lis_mean[1:]
        wind_TI_lis = wind_TI_lis[1:]

        if not k == 2:
            fig, ax = ploter.scatter(
                x = wind_velocity_lis_mean, 
                y = wind_TI_lis,
                title = 'Wind Turbulence Intensity Distribution',
                xlabel = 'Mean Wind Velocity (m/s)',
                ylabel = 'Wind Turbulence Intensity',
                ylim = (-0.02, 0.5),
                xlim = (-1, 30),
            )
            
            figs.append(fig)
        else:
            
            wind_velocity_lis_mean_raw = wind_velocity_lis_mean
            wind_TI_lis_raw = wind_TI_lis
            for i in range(len(append_sensor)):

                file_paths_lis = unpacker.File_Match_Sensor_Path(pattern = append_sensor[i], paths = paths)

                wind_velocity_lis_mean = np.array([0])
                wind_TI_lis = np.array([0])
                for file_path in file_paths_lis:
                    # # 60 * 60s 每秒的风速
                    # wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(file_path)
                    for wind_velocity in unpacker.File_Detach_Data(file_path, time_interval = 1, mode = 'UAN')[0]:
            
                        wind_velocity_mean = np.mean(wind_velocity)
                        wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
                        Wind_TI = wind_velocity_rms / wind_velocity_mean

                        # 平均值和紊流度
                        wind_velocity_lis_mean = np.hstack((wind_velocity_lis_mean, wind_velocity_mean))
                        wind_TI_lis = np.hstack((wind_TI_lis, Wind_TI))

                    wind_velocity_lis_mean = wind_velocity_lis_mean[1:]
                    wind_TI_lis = wind_TI_lis[1:]


                fig, ax = ploter.scatter(
                    x = wind_velocity_lis_mean_raw, 
                    y = wind_TI_lis_raw,
                    title = 'Wind Turbulence Intensity Distribution',
                    xlabel = 'Mean Wind Velocity (m/s)',
                    ylabel = 'Wind Turbulence Intensity',
                    ylim = (-0.02, 0.5),
                    xlim = (-1, 30),
                    legend = 'UpStream Wind',
                )

                fig, ax = ploter.scatter(
                    x = wind_velocity_lis_mean_raw, 
                    y = wind_TI_lis_raw,
                    title = 'Wind Turbulence Intensity Distribution',
                    xlabel = 'Mean Wind Velocity (m/s)',
                    ylabel = 'Wind Turbulence Intensity',
                    ylim = (-0.02, 0.5),
                    xlim = (-1, 30),
                    legend = f'Append Sensor: {append_sensor[i]}',
                    fig = fig ,
                    ax = ax ,
                    color = 'pink' 
                )

                ax.legend()

                figs.append(fig)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

def Small_f_major_Abnormal_Vibration():

    # Up_Stream_Sensor_In_Plane = []
    # Down_Stream_Sensor_In_Plane = []
    root = 'F:/Research/My_Thesis/Data/苏通/VIC/'
    unpacker = UNPACK()
    VIC_Path_lis = unpacker.File_Read_Paths(root)
    filt = Abnormal_Vibration_Filter()

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
    VIV_Path_Lis = []
    time_interval = 1
    ploter = PlotLib()
    fs = 50
    figs = []

    for path in VIC_Path_lis:
        data_slice = unpacker.File_Detach_Data(path, time_interval = time_interval, mode = 'VIC')

        in_or_out = None

        sensor_id = Path(path).parts[-1].split('_')[0]
        if sensor_id in VIC_Name_Lis_Up_Down_Stream_In_Plane:
            in_or_out = 'Inplane'  
        elif sensor_id in VIC_Name_Lis_Up_Down_Stream_Out_Plane:
            in_or_out = 'Outplane'  

        for i, data in enumerate(data_slice):
            fx, Pxx_den = signal.welch(data, fs = 50, 
                                    nfft = 512,
                                    nperseg = 512,
                                    noverlap = 256)

            # # 准则一：振幅控制
            # if np.max(data) < 0.01:
            #     return False
            E1 = np.max(Pxx_den)

            if E1 < 0.5:
                
                time = i * time_interval
                VIV_Path_Lis.append((path, time, in_or_out))

                Res_Acceleration = data
                time = np.arange(len(Res_Acceleration)) / 60 / fs
                fx, Pxx_Den = signal.welch(Res_Acceleration, fs = 50, nfft = 512, nperseg = 512, noverlap = 256)
                

                data = [
                    (time, Res_Acceleration),
                    (fx, Pxx_Den)
                ]

                    
                titles = [
                    f'Time Series',
                    'Power Spectral Density'
                ]

                labels = [
                    ('Time(s)', 'Acceleration $(m/s^2)$'),
                    ('Frequency(hz)', 'PSD')
                ]

                fig_i, ax_i = ploter.plots(
                    data_lis = data,
                    titles = titles,
                    labels = labels,
                )

                figs.append(fig_i)
                plt.close()
    
    figs.append(figs)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()
    return 

def Uniform_Sampling_Data():
    
    unpacker = UNPACK()
    ploter = PlotLib()
    unpacker = UNPACK()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()
    total_samples = 1000
    time_interval = 1
    
    path_with_time = deque()

    save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\test\\'
    for path in VIC_Path_Lis:
        for time in range(int(60 / time_interval)):
            path_with_time.append((path, time))

    path_with_time = list(path_with_time)

    samples = deque()

    # 循环直到我们有15000个样本
    while len(samples) < 15000:
        # 随机选择一个元素并添加到结果列表中
        samples.append(random.choice(path_with_time))

    samples = list(samples)


    fs = 50
    for path, time in samples:
        response_time_series = unpacker.VIC_DATA_Unpack(path)[time * fs * 60: (time + time_interval) * fs * 60]
        if len(response_time_series) < time_interval * fs * 60:
            continue
        fx, Pxx_den = signal.welch(response_time_series, fs = 50, nfft = 512, 
                                nperseg = 512, noverlap = 256)

        x = np.arange(len(response_time_series)) / fs # 秒作为单位画图
        data_lis = [
            (x, response_time_series),
            (fx, Pxx_den)
            ]
        
        titles = [
            'Time Series',
            'Power Spectral Density'
        ]

        labels = [
            ('Time(s)', 'Acceleration $(m/s^2)$'),
            ('Frequency(hz)', 'PSD')
        ]

        file = Path(path)
        Responsetitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + file.parts[-1][:-4].split('_')[1][:2] + '_' + file.parts[-1][:-4].split('_')[0] + 'Response' 
        save_path = save_root + Responsetitle + '.png'

        fig, axes = ploter.plots(data_lis, titles = titles, labels = labels)

        fig.savefig(save_path)
        plt.close()

def Normal_Vibration_Velocity_RMS():
    from scipy.stats import norm

    ploter = PlotLib()
    raw_data_x = np.load(r'C:\Users\admin\Desktop\datax.npy')
    raw_data_y = np.load(r'C:\Users\admin\Desktop\datay.npy')
    figs = []
    save_root = r'C:\Users\admin\Desktop\test\\'
    # for file_num in range(10):
    file_num = 3
    coeff = 0.2 / ((23 + 2 * 3) ** 2)
    fat = 0.03
    parabola_X = np.arange(0, 30, 0.001)

    parabola_Y = coeff * np.square(parabola_X )
    fig, ax = ploter.plot(
        x = parabola_X, 
        y = parabola_Y, 
        color = 'pink',
        legend = f'k = {coeff}',
        style = '--', 
    )

    data_x = np.array(raw_data_x)
    data_y = np.array(raw_data_y)


    cache_x = data_x
    cache_y = data_y
    data_x = data_x[cache_y < (coeff * np.square(cache_x) + fat)]
    data_y = data_y[cache_y <( coeff * np.square(cache_x) + fat)]


    cache_x = data_x
    cache_y = data_y
    data_x = data_x[cache_y > (coeff * np.square(cache_x) - fat)]
    data_y = data_y[cache_y > (coeff * np.square(cache_x) - fat)]


    fig, ax = ploter.scatter(
        x = data_x, 
        y = data_y, 
        fig = fig, 
        ax = ax,
        title = 'RMS Distribution',
        xlabel = 'Mean Wind Velocity $(m/s)$',
        ylabel = 'Acceleration RMS',
        xlim = (0, 30),
        ylim = (0, 0.5)
    )

    # 给定三个阈值，然后来划到图上

    x_lis = np.array([10, 15, 20])
    y_lis = coeff * np.square(x_lis)
    ylim = np.arange(0, 0.5, 0.1)

    # y_ticks = np.hstack((y_lis, ylim))

    # ax.set_yticks(y_ticks, np.round(y_ticks, 2))

    used_twice_x = data_x
    used_twice_y = data_y
    for i in range(len(x_lis)):
        x = x_lis[i]
        y = y_lis[i]

        # line1
        line1_x = np.arange(0, x, step = 0.01)
        line1_y = np.ones(shape = line1_x.shape) * y

        fig, ax = ploter.plot(
            x = line1_x, 
            y = line1_y, 
            fig = fig, 
            ax = ax,
            color = 'pink', 
            style = '--', 
        )

        # line2
        
        line2_y = np.arange(0, y, step = 0.01)
        line2_x = np.ones(shape = line2_y.shape) * x

        fig, ax = ploter.plot(
            x = line2_x, 
            y = line2_y, 
            fig = fig, 
            ax = ax,
            color = 'pink', 
            style = '--', 
        )
        ax.annotate(f'({x}, {y:.2f})', xy = (x, y), 
        xytext = (x + 0.1, y + 0.01),
        )

    ax.grid(False)
    figs.append(fig)
    raw_data_x_used_once = raw_data_x
    raw_data_y_used_once = raw_data_y

    for i in range(len(x_lis)):

        x = x_lis[i]
        y = y_lis[i]

        data_x = np.array(raw_data_x_used_once)
        data_y = np.array(raw_data_y_used_once)
    
        # 正态拟合
        cache_x = data_x
        cache_y = data_y
        data_x = data_x[cache_x < x + fat]
        data_y = data_y[cache_x < x + fat]

        cache_x = data_x
        cache_y = data_y
        data_x = data_x[cache_x > x - fat]
        data_y = data_y[cache_x > x - fat]

        # 仅有y数据服从正态分布

        norm_samples = data_y
        mean, std_dev = norm.fit(norm_samples)

        Normal_X = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
        Normal_Y = norm.pdf(Normal_X, mean, std_dev)

        final_x = Normal_Y + x
        final_y = Normal_X + y

        fig_i, ax_i = ploter.plot(
            x = final_x, 
            y = final_y, 
            xlabel = f'Wind Velocity = ${x} (m/s)$',
            ylabel = 'Acceleration RMS',
            color = 'pink', 
            title = f'$miu$ = {mean:.2f}, $sigma$ = {std_dev:.2f}',
        )

        fig_i, ax_i = ploter.scatter(
            x = data_x, 
            y = data_y, 
            fig = fig_i, 
            ax = ax_i,
        )

        ax_i.grid(False)

        figs.append(fig_i)





    # 正态拟合

    
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()



    return 

def PSD_Integral_Analysis():

    root = 'F:/Research/My_Thesis/Data/苏通/VIC/'
    unpacker = UNPACK()
    VIC_Path_lis = unpacker.File_Read_Paths(root)
    filt = Abnormal_Vibration_Filter()

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
    VIV_Path_Lis = []
    time_interval = 1
    ploter = PlotLib()
    fs = 50
    figs = []

    from scipy import integrate

    PSD_Integrals_Inplane = deque()
    PSD_Integrals_Outplane = deque()

    for path in VIC_Path_lis:
        data_slice = unpacker.File_Detach_Data(path, time_interval = time_interval, mode = 'VIC')
        in_or_out = None

        sensor_id = Path(path).parts[-1].split('_')[0]
        if sensor_id in VIC_Name_Lis_Up_Down_Stream_In_Plane:
            in_or_out = 'Inplane'  
        elif sensor_id in VIC_Name_Lis_Up_Down_Stream_Out_Plane:
            in_or_out = 'Outplane'  

        for i, data in enumerate(data_slice):
            # 筛选机制
            if len(data) == fs * time_interval * 60:
                fx, Pxx_den = signal.welch(data, fs = 50, 
                                        nfft = 512,
                                        nperseg = 512,
                                        noverlap = 256)
                
                
                # 不应该有负数，但为什么有负数呢？
                # 函数调用问题，它的x，y是反过来的，和我们定义的抽象函数一样
                PSD_Integral = integrate.simpson(x = fx, y = Pxx_den)

                if in_or_out == 'Inplane':
                    PSD_Integrals_Inplane.append(PSD_Integral)
                else:
                    PSD_Integrals_Outplane.append(PSD_Integral)

    PSD_Integrals_Inplane = np.array(PSD_Integrals_Inplane)
    PSD_Integrals_Inplane = PSD_Integrals_Inplane

    fig1, ax = ploter.hist(PSD_Integrals_Inplane, 
                            bins = 100, 
                            title = 'PSD Integral Distribution of Inplane', 
                            xlabel = 'Area', 
                            ylabel = 'Frequency')
    

    PSD_Integrals_Inplane = PSD_Integrals_Inplane[PSD_Integrals_Inplane < 25]
    figs.append(fig1)

    fig2, ax = ploter.hist(PSD_Integrals_Inplane, 
                            bins = 100, 
                            title = 'PSD Integral Distribution of Inplane', 
                            xlabel = 'Area', 
                            ylabel = 'Frequency')
    figs.append(fig2)

    PSD_Integrals_Inplane = PSD_Integrals_Inplane[PSD_Integrals_Inplane < 5]

    fig3, ax = ploter.hist(PSD_Integrals_Inplane, 
                            bins = 100, 
                            title = 'PSD Integral Distribution of Inplane', 
                            xlabel = 'Area', 
                            ylabel = 'Frequency')
    figs.append(fig3)


    PSD_Integrals_Inplane = PSD_Integrals_Inplane[PSD_Integrals_Inplane < 0.5]

    fig4, ax = ploter.hist(PSD_Integrals_Inplane, 
                            bins = 100, 
                            title = 'PSD Integral Distribution of Inplane', 
                            xlabel = 'Area', 
                            ylabel = 'Frequency')
    figs.append(fig4) 
    plt.close()
    
    PSD_Integrals_Inplane = PSD_Integrals_Inplane[PSD_Integrals_Inplane < 0.1]

    fig4, ax = ploter.hist(PSD_Integrals_Inplane, 
                            bins = 100, 
                            title = 'PSD Integral Distribution of Inplane', 
                            xlabel = 'Area', 
                            ylabel = 'Frequency')
    figs.append(fig4) 
    plt.close()


    PSD_Integrals_Inplane = PSD_Integrals_Inplane[PSD_Integrals_Inplane < 0.04]

    fig4, ax = ploter.hist(PSD_Integrals_Inplane, 
                            bins = 100, 
                            title = 'PSD Integral Distribution of Inplane', 
                            xlabel = 'Area', 
                            ylabel = 'Frequency')
    figs.append(fig4) 


    PSD_Integrals_Inplane = PSD_Integrals_Inplane[PSD_Integrals_Inplane < 0.025]
    fig4, ax = ploter.hist(PSD_Integrals_Inplane, 
                            bins = 100, 
                            title = 'PSD Integral Distribution of Inplane', 
                            xlabel = 'Area', 
                            ylabel = 'Frequency')
    figs.append(fig4) 
    
    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()
    return 

def Wind_Turbulence_wVelocity_wDirection():
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    sensors = [
            #   'ST-UAN-T01-003-01', # 北索塔塔顶
            #   'ST-UAN-T02-003-01', # 南索塔塔顶
              'ST-UAN-G04-001-01', # 跨中桥面上游
            #   'ST-UAN-G04-002-01', # 跨中桥面下游
               ]

    name = [
        # 'Wind Direction Distribution (Top Of North Pylon)',
        # 'Wind Direction Distribution (Top Of South Pylon)',
        'Wind Direction Distribution (Upstream of Mid-Span)',
        # 'Wind Direction Distribution (DownStream of Mid-Span)'
    ]

    unpacker = UNPACK()
    path = r'F:\Research\My_Thesis\Data\苏通\UAN\\'
    paths = unpacker.File_Read_Paths(path)
    figs = []
    time_interval = 1
    fs = 1
    num_points = time_interval * fs * 60 # s
    ploter = PlotLib()
    cmap = 'coolwarm'

    for k in range(len(sensors)):
        
        file_paths_lis = unpacker.File_Match_Sensor_Path(pattern = sensors[k], paths = paths)

        wind_directions = np.array([0])
        wind_velocities = np.array([0])
        wind_TIs = np.array([0])
        for file_path in file_paths_lis:
            # # 60 * 60s 每秒的风速
            # wind_velocity, wind_direction, wind_Angle = unpacker.Wind_Data_Unpack(file_path)
            wind_velocity_slice, wind_direction_slice, _ = unpacker.File_Detach_Data(file_path, time_interval = 1, mode = 'UAN')

            for i in range(int(60 / time_interval)):

                wind_velocity = np.array(wind_velocity_slice[i])
                wind_direction = np.array(wind_direction_slice[i]) 

                wind_direction_mean = np.mean(wind_direction)
                wind_velocity_mean = np.mean(wind_velocity)
                wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
                Wind_TI = wind_velocity_rms / wind_velocity_mean

                wind_directions = np.hstack((wind_directions, wind_direction_mean))
                wind_velocities = np.hstack((wind_velocities, wind_velocity_mean))
                wind_TIs = np.hstack((wind_TIs, Wind_TI))

        wind_directions = wind_directions[1:]
        wind_velocities = wind_velocities[1:]
        wind_TIs = wind_TIs[1:]

        # 筛选风速小于30
        # wind_directions = wind_directions[wind_velocities < 30]
        # wind_TIs = wind_TIs[wind_velocities < 30]
        # wind_velocities = wind_velocities[wind_velocities < 30]

    df_VIV = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\VIV.xlsx')
    wind_directions_VIV = np.array([0])
    wind_velocities_VIV = np.array([0])
    wind_TIs_VIV = np.array([0])

    for k in range(len(sensors)):

        for _, path, time, plane in df_VIV.values:
            wind_path = unpacker.VIC_Path_2_WindPath(path, wind_sensor_ids = sensors)
            if len(wind_path) == 0:
                continue

            wind_velocity_slice, wind_direction_slice, _ = unpacker.File_Detach_Data(wind_path[0], time_interval = 1, mode = 'UAN')

            wind_velocity = np.array(wind_velocity_slice[time])
            wind_direction = np.array(wind_direction_slice[time]) 

            wind_velocity_mean = np.mean(wind_velocity)
            wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
            Wind_TI = wind_velocity_rms / wind_velocity_mean

            wind_directions_VIV = np.hstack((wind_directions_VIV, wind_direction_mean))
            wind_velocities_VIV = np.hstack((wind_velocities_VIV, wind_velocity_mean))
            wind_TIs_VIV = np.hstack((wind_TIs_VIV, Wind_TI))

    wind_directions_VIV = wind_directions_VIV[1:]
    wind_velocities_VIV = wind_velocities_VIV[1:]
    wind_TIs_VIV = wind_TIs_VIV[1:]
    # 2D效果图
    fig, ax = ploter.rose_scatter(
        theta = np.deg2rad(360 - wind_directions),
        rho = wind_velocities, 
        title = 'Wind Turbulence Distribution',
        cmap = cmap, 
        c_data = wind_TIs,
        alpha = 0.5, 
        colorbar = True, 
        color_label = 'Wind Turbulence',
        # legend = 'Normal Samples', 
        bridge_axis = True
    )



    # fig, ax = ploter.rose_scatter(
    #     theta = np.deg2rad(360 - wind_directions_VIV), 
    #     rho = wind_velocities_VIV,
    #     cmap = cmap, 
    #     c_data = wind_TIs_VIV,
    #     fig = fig, 
    #     ax = ax,
    #     alpha = 0.5, 
    #     s = 10, 
    #     legend = 'VIV Samples', 
    #     marker = 'X'
    #     )

    # ax.legend()
    figs.append(fig)


    # 3D效果图
    fig, ax = ploter.scatter_3d(
        theta = np.deg2rad(360 - wind_directions), 
        rho = wind_velocities,
        z = wind_TIs,
        legend = 'Normal Samples', 
    )

    fig, ax = ploter.scatter_3d(
        theta = np.deg2rad(360 - wind_directions_VIV), 
        rho = wind_velocities_VIV,
        z = wind_TIs_VIV,
        fig = fig, 
        ax = ax, 
        color = 'pink', 
        legend = 'VIV Samples', 
        s = 10, 
        marker = '^'
    )

    theta_mean_VIV = np.mean(360 - wind_directions_VIV)
    rho_max_VIV = np.max(wind_velocities_VIV[np.isfinite(wind_velocities_VIV)])

    # 拟合曲线，画出
    from scipy.optimize import curve_fit
    def inv_func(x, k):
        return k / x
    # 使用 curve_fit 进行拟合，popt 是最优拟合参数，pcov 是协方差矩阵
    popt, pcov = curve_fit(inv_func, wind_velocities_VIV[np.isfinite(wind_velocities_VIV)], wind_TIs_VIV[np.isfinite(wind_velocities_VIV)])

    # popt 包含了拟合得到的最佳参数 k
    k_optimal = popt[0]

    surf_rho = np.linspace(0, rho_max_VIV, 300)
    surf_z = inv_func(surf_rho, k = k_optimal)
    # 用最大值稍微修正一下
    surf_rho = surf_rho[surf_z < 0.9]
    surf_z = surf_z[surf_z < 0.9]

    surf_z = surf_z[surf_rho > np.min(np.isfinite(surf_rho))]
    surf_rho = surf_rho[surf_rho > np.min(np.isfinite(surf_rho))]

    surf_theta = np.deg2rad(theta_mean_VIV * np.ones(surf_rho.shape)) - np.pi / 2

    # 曲线的投影线
    ax.plot(surf_rho * np.cos(surf_theta), 
            surf_rho * np.sin(surf_theta), 
            surf_z, 
            color = 'red', 
            label = 'Fitted Curve')
    
    proj_x = surf_rho * np.cos(surf_theta)
    proj_y = surf_rho * np.sin(surf_theta)
    proj_z = np.zeros(shape = proj_x.shape)

    ax.plot(proj_x, 
            proj_y, 
            proj_z, 
            color = 'orange', 
            linestyle = '--',
            )

    ax.legend()
    figs.append(fig)

    # 创建主窗口
    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

def Vibration_RMS_Statistic():
    unpacker = UNPACK()
    VIC_Path_Lis = unpacker.VIC_Path_Lis()

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

    Path_With_Time_Intervals = []

    # 取基频的经验值为1Hz
    f0 = 1 # Hz
    fs = 50

    ploter = PlotLib()
    ab_fil = Abnormal_Vibration_Filter()
    time_interval = 1
    rmss = deque()
    for i in range(len(VIC_Name_Lis_Up_Down_Stream_In_Plane)):
        sensor_Inplane = VIC_Name_Lis_Up_Down_Stream_In_Plane[i]
        
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Inplane, VIC_Path_Lis)
        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = time_interval)
            for time, data in enumerate(data_lis):
                if len(data) == 60 * fs * time_interval:
                    data_mean = np.mean(data)
                    rms = np.sqrt(np.mean((data - data_mean) ** 2))
                    rmss.append(rms)



    for i in range(len(VIC_Name_Lis_Up_Down_Stream_Out_Plane)):
        sensor_Outplane = VIC_Name_Lis_Up_Down_Stream_Out_Plane[i]
        path_lis = unpacker.File_Match_Sensor_Path(sensor_Outplane, VIC_Path_Lis)

        for path in path_lis:
            data_lis = unpacker.File_Detach_Data(path, time_interval = time_interval)
            for time, data in enumerate(data_lis):
                if len(data) == 60 * fs * time_interval:
                    data_mean = np.mean(data)
                    rms = np.sqrt(np.mean((data - data_mean) ** 2))
                    rmss.append(rms)
    figs = []
    rmss = np.array(rmss)
    fig1, ax = ploter.hist(x = rmss, bins = 100, 
                          xlabel = 'RMS', ylabel = 'Frequency', 
                          title = 'RMS distribution', 
                          )
    figs.append(fig1)


    fig2, ax = ploter.hist(x = rmss[rmss < 10], bins = 100, 
                          xlabel = 'RMS', ylabel = 'Frequency', 
                          title = 'RMS distribution', 
                          )
    figs.append(fig2)

    fig3, ax = ploter.hist(x = rmss[rmss < 2.5], bins = 100, 
                          xlabel = 'RMS', ylabel = 'Frequency', 
                          title = 'RMS distribution', 
                          )
    figs.append(fig3)

    fig4, ax = ploter.hist(x = rmss[rmss < 1], bins = 100, 
                          xlabel = 'RMS', ylabel = 'Frequency', 
                          title = 'RMS distribution', 
                          )
    figs.append(fig4)

    fig5, ax = ploter.hist(x = rmss[rmss < 0.5], bins = 100, 
                          xlabel = 'RMS', ylabel = 'Frequency', 
                          title = 'RMS distribution', 
                          )
    figs.append(fig5)

    fig6, ax = ploter.hist(x = rmss[rmss < 0.25], bins = 100, 
                          xlabel = 'RMS', ylabel = 'Frequency', 
                          title = 'RMS distribution', 
                          )
    figs.append(fig6)

    fig6, ax = ploter.hist(x = rmss[rmss < 0.2], bins = 100, 
                          xlabel = 'RMS', ylabel = 'Frequency', 
                          title = 'RMS distribution', 
                          )
    figs.append(fig6)


    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, figs)

    # 运行主循环
    root.mainloop()

    return 

def wVIV_save_cable_vib_at_angle_tur_():

    wind_sensors = ['ST-UAN-G04-001-01']
    unpacker = UNPACK()
    file_paths = unpacker.VIC_Path_Lis()

    unpacker = UNPACK()
    path = r'F:\Research\My_Thesis\Data\苏通\UAN\\'
    time_interval = 1
    
    figs = []

    # ------------------------------------------------------------获取阈值--------------------------------------------------------------------
    df_VIV = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\test1219.xlsx')
    wind_directions_VIV = np.array([0])
    wind_velocities_VIV = np.array([0])
    wind_TIs_VIV = np.array([0])
    wind_RMSs_VIV = np.array([0])
    VIC_root = r'F:\Research\My_Thesis\Data\苏通\VIC\\'
    VIC_total_paths = unpacker.VIC_Path_Lis()
    for file_path, user_input in df_VIV.values:
        if user_input != 1:
            continue
        month, day, time_str, time_time = Path(file_path).parts[-1].split('_')[:-1]
        sensor_id = Path(file_path).parts[-1].split('_')[-1].split('Response')[0]
        VIC_path_pattern = os.path.join(VIC_root, month[1:], day[1:], sensor_id + '_' + time_str[1:]).replace('\\', '/').replace('//', '/')
        paths = unpacker.File_Match_Pattern(VIC_path_pattern, VIC_total_paths)
        if len(paths) == 0:
            continue

        wind_path = unpacker.VIC_Path_2_WindPath(paths[0], wind_sensor_ids = wind_sensors)
        if len(wind_path) == 0:
            continue

        wind_velocity_slice, wind_direction_slice, _ = unpacker.File_Detach_Data(wind_path[0], time_interval = 1, mode = 'UAN')

        time_time = int(time_time[2:])
        wind_velocity = np.array(wind_velocity_slice[time_time])
        wind_direction = np.array(wind_direction_slice[time_time]) 

        wind_direction_mean = np.mean(wind_direction)
        wind_velocity_mean = np.mean(wind_velocity)
        wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
        Wind_TI = wind_velocity_rms / wind_velocity_mean

        wind_directions_VIV = np.hstack((wind_directions_VIV, wind_direction_mean))
        wind_velocities_VIV = np.hstack((wind_velocities_VIV, wind_velocity_mean))
        wind_TIs_VIV = np.hstack((wind_TIs_VIV, Wind_TI))
        wind_RMSs_VIV = np.hstack((wind_RMSs_VIV, wind_velocity_rms))

    wind_directions_VIV = wind_directions_VIV[1:]
    wind_velocities_VIV = wind_velocities_VIV[1:]
    wind_TIs_VIV = wind_TIs_VIV[1:]
    wind_RMSs_VIV = wind_RMSs_VIV[1:]
    
    # 洗掉NAN
    wind_velocities_VIV = wind_velocities_VIV[np.isfinite(wind_directions_VIV)]
    wind_TIs_VIV = wind_TIs_VIV[np.isfinite(wind_directions_VIV)]
    wind_RMSs_VIV = wind_RMSs_VIV[np.isfinite(wind_directions_VIV)]
    wind_directions_VIV = wind_directions_VIV[np.isfinite(wind_directions_VIV)]

    # 洗掉NAN
    wind_directions_VIV = wind_directions_VIV[np.isfinite(wind_velocities_VIV)]
    wind_TIs_VIV = wind_TIs_VIV[np.isfinite(wind_velocities_VIV)]
    wind_RMSs_VIV = wind_RMSs_VIV[np.isfinite(wind_velocities_VIV)]
    wind_velocities_VIV = wind_velocities_VIV[np.isfinite(wind_velocities_VIV)]

    # 得到三个近似的指标
    wind_directions_VIV_mean = np.mean(wind_directions_VIV)
    wind_velocities_VIV_mean = np.mean(wind_velocities_VIV)
    wind_RMSs_VIV_mean = np.mean(wind_RMSs_VIV)
    wind_TIs_VIV_mean = np.mean(wind_TIs_VIV)

    # 筛选用到的函数
    def fil_data(num, interval):
        '''
        注意，该筛选有头有尾
        array: ndarray, 输入数据
        interval: 筛选的间隔
        '''
        return num > interval[0] and num < interval[1]
    
    # 风速、风向、紊流度、风的RMS，都约束一下
    wind_directions_VIV_interval = (wind_directions_VIV_mean - 5, wind_directions_VIV_mean + 5)
    wind_TIs_VIV_mean_interval = (0, wind_TIs_VIV_mean)
    wind_RMSs_VIV_mean_interval = (wind_RMSs_VIV_mean * 0.9, wind_RMSs_VIV_mean * 10)
    wind_velocities_VIV_interval = (wind_velocities_VIV_mean * 0.95, wind_velocities_VIV_mean * 1.05)
    
    # wind_TIs_VIV_mean_interval = (0.02, 0.06)
    # wind_velocities_VIV_interval = (8, 12)

    VIC_sensor = 'ST-VIC-C18-102-01'
    VIC_paths = unpacker.File_Match_Sensor_Path(VIC_sensor, file_paths)
    Satisfied_VIC_paths = deque()
    # 读取VIC数据，筛选在上面那个区间里头的值
    for VIC_path in VIC_paths:
        wind_path = unpacker.VIC_Path_2_WindPath(VIC_path, wind_sensor_ids = wind_sensors)
        if len(wind_path) == 0:
            continue
        wind_path = wind_path[0]
        wind_data_slice = unpacker.File_Detach_Data(wind_path, time_interval = time_interval, mode = 'UAN')
        for i in range(int(60 / time_interval)):

            wind_velocity, wind_direction, wind_angle = wind_data_slice[0][i], wind_data_slice[1][i], wind_data_slice[2][i]
            wind_velocity = np.array(wind_velocity)
            wind_direction = np.array(wind_direction) 

            wind_direction_mean = np.mean(wind_direction)
            wind_velocity_mean = np.mean(wind_velocity)
            wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
            Wind_TI = wind_velocity_rms / wind_velocity_mean


            # 两个约束条件
            if fil_data(wind_direction_mean, wind_directions_VIV_interval
                        ) and fil_data(Wind_TI, wind_TIs_VIV_mean_interval
                        ) and fil_data(wind_velocity_mean, wind_velocities_VIV_interval
                        ) and fil_data(wind_velocity_rms, wind_RMSs_VIV_mean_interval):
                Satisfied_VIC_paths.append((VIC_path, i * time_interval))
    
    # 提取符合要求的数据
    fs = 50
    save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIV_Special_DIrections_TI_Samples\\'
    ploter = PlotLib()
    for VIC_path, time in Satisfied_VIC_paths:   
        response_time_series = unpacker.VIC_DATA_Unpack(VIC_path)[time * fs * 60: (time + time_interval) * fs * 60]
        fx, Pxx_den = signal.welch(response_time_series, fs = 50, nfft = 512, 
                                nperseg = 512, noverlap = 256)

        x = np.arange(len(response_time_series)) / fs # 秒作为单位画图
        data_lis = [
            (x, response_time_series),
            (fx, Pxx_den) 
            ]
        
        titles = [
            'Time Series',
            'Power Spectral Density'
        ]

        labels = [
            ('Time(s)', 'Acceleration $(m/s^2)$'),
            ('Frequency(hz)', 'PSD')
        ]

        file = Path(VIC_path)
        Responsetitle = 'M' + file.parts[-3] + '_D' + file.parts[-2] + '_T' + file.parts[-1][:-4].split('_')[1][:2] + '_TT' + str(time) + '_' + file.parts[-1][:-4].split('_')[0] + 'Response' 
        save_path = save_root + Responsetitle + '.png'
        if len(data_lis) == 0:
            continue
        fig, axes = ploter.plots(data_lis, titles = titles, labels = labels)
        fig.savefig(save_path)
        plt.close()


    return 

def VIV_wind_Turbulence_distribution():
    VIC_sensor = 'ST-VIC-C18-102-01'
    wind_sensors = ['ST-UAN-G04-001-01']
    unpacker = UNPACK()

    unpacker = UNPACK()
    time_interval = 1
    
    figs = []

    # ------------------------------------------------------------获取阈值--------------------------------------------------------------------
    df_VIV = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\test1219.xlsx')
    wind_directions_VIV = np.array([0])
    wind_velocities_VIV = np.array([0])
    wind_TIs_VIV = np.array([0])
    VIC_root = r'F:\Research\My_Thesis\Data\苏通\VIC\\'
    VIC_total_paths = unpacker.VIC_Path_Lis()
    for file_path, user_input in df_VIV.values:
        if user_input != 1:
            continue
        month, day, time_str, time_time = Path(file_path).parts[-1].split('_')[:-1]
        sensor_id = Path(file_path).parts[-1].split('_')[-1].split('Response')[0]
        VIC_path_pattern = os.path.join(VIC_root, month[1:], day[1:], sensor_id + '_' + time_str[1:]).replace('\\', '/').replace('//', '/')
        paths = unpacker.File_Match_Pattern(VIC_path_pattern, VIC_total_paths)
        if len(paths) == 0:
            continue
        data_slice = unpacker.File_Detach_Data(path = paths[0], time_interval = time_interval, mode = 'VIC')
        times = deque()

        wind_path = unpacker.VIC_Path_2_WindPath(paths[0], wind_sensor_ids = wind_sensors)
        if len(wind_path) == 0:
            continue

        wind_velocity_slice, wind_direction_slice, _ = unpacker.File_Detach_Data(wind_path[0], time_interval = 1, mode = 'UAN')

        time_time = int(time_time[2:])
        wind_velocity = np.array(wind_velocity_slice[time_time])
        wind_direction = np.array(wind_direction_slice[time_time]) 
        wind_direction_mean = np.mean(wind_direction)
        wind_velocity_mean = np.mean(wind_velocity)
        wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
        Wind_TI = wind_velocity_rms / wind_velocity_mean

        wind_directions_VIV = np.hstack((wind_directions_VIV, wind_direction_mean))
        wind_velocities_VIV = np.hstack((wind_velocities_VIV, wind_velocity_mean))
        wind_TIs_VIV = np.hstack((wind_TIs_VIV, Wind_TI))

    wind_directions_VIV = wind_directions_VIV[1:]
    wind_velocities_VIV = wind_velocities_VIV[1:]
    wind_TIs_VIV = wind_TIs_VIV[1:]
    
    # 洗掉NAN
    wind_velocities_VIV = wind_velocities_VIV[np.isfinite(wind_directions_VIV)]
    wind_TIs_VIV = wind_TIs_VIV[np.isfinite(wind_directions_VIV)]
    wind_directions_VIV = wind_directions_VIV[np.isfinite(wind_directions_VIV)]

    # 洗掉NAN
    wind_directions_VIV = wind_directions_VIV[np.isfinite(wind_velocities_VIV)]
    wind_TIs_VIV = wind_TIs_VIV[np.isfinite(wind_velocities_VIV)]
    wind_velocities_VIV = wind_velocities_VIV[np.isfinite(wind_velocities_VIV)]


    ploter = PlotLib()
    fig, ax = ploter.scatter(
        x = wind_velocities_VIV,
        y = wind_TIs_VIV,
        xlabel = 'Wind Velocity $(m/s)$',
        ylabel = 'Turbulence',
        color = 'skyblue'
    )

    figs.append(fig)

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()


    return 

def VIV_wind_RMS_Distribution():
    VIC_sensor = 'ST-VIC-C18-102-01'
    wind_sensors = ['ST-UAN-G04-001-01']
    unpacker = UNPACK()

    unpacker = UNPACK()
    time_interval = 1
    
    figs = []

    # ------------------------------------------------------------获取阈值--------------------------------------------------------------------
    df_VIV = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\test1219.xlsx')
    wind_directions_VIV = np.array([0])
    wind_velocities_VIV = np.array([0])
    wind_TIs_VIV = np.array([0])
    wind_rmss_VIV = np.array([0])
    VIC_root = r'F:\Research\My_Thesis\Data\苏通\VIC\\'
    VIC_total_paths = unpacker.VIC_Path_Lis()
    for file_path, user_input in df_VIV.values:
        if user_input != 1:
            continue
        month, day, time_str, time_time = Path(file_path).parts[-1].split('_')[:-1]
        sensor_id = Path(file_path).parts[-1].split('_')[-1].split('Response')[0]
        VIC_path_pattern = os.path.join(VIC_root, month[1:], day[1:], sensor_id + '_' + time_str[1:]).replace('\\', '/').replace('//', '/')
        paths = unpacker.File_Match_Pattern(VIC_path_pattern, VIC_total_paths)
        if len(paths) == 0:
            continue
        data_slice = unpacker.File_Detach_Data(path = paths[0], time_interval = time_interval, mode = 'VIC')
        times = deque()

        wind_path = unpacker.VIC_Path_2_WindPath(paths[0], wind_sensor_ids = wind_sensors)
        if len(wind_path) == 0:
            continue

        wind_velocity_slice, wind_direction_slice, _ = unpacker.File_Detach_Data(wind_path[0], time_interval = 1, mode = 'UAN')

        time_time = int(time_time[2:])
        wind_velocity = np.array(wind_velocity_slice[time_time])
        wind_direction = np.array(wind_direction_slice[time_time]) 
        wind_direction_mean = np.mean(wind_direction)
        wind_velocity_mean = np.mean(wind_velocity)
        wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
        Wind_TI = wind_velocity_rms / wind_velocity_mean

        wind_directions_VIV = np.hstack((wind_directions_VIV, wind_direction_mean))
        wind_velocities_VIV = np.hstack((wind_velocities_VIV, wind_velocity_mean))
        wind_TIs_VIV = np.hstack((wind_TIs_VIV, Wind_TI))
        wind_rmss_VIV = np.hstack((wind_rmss_VIV, wind_velocity_rms))

    wind_directions_VIV = wind_directions_VIV[1:]
    wind_velocities_VIV = wind_velocities_VIV[1:]
    wind_TIs_VIV = wind_TIs_VIV[1:]
    wind_rmss_VIV = wind_rmss_VIV[1:]

    # 洗掉NAN
    wind_velocities_VIV = wind_velocities_VIV[np.isfinite(wind_directions_VIV)]
    wind_TIs_VIV = wind_TIs_VIV[np.isfinite(wind_directions_VIV)]
    wind_rmss_VIV = wind_rmss_VIV[np.isfinite(wind_directions_VIV)]
    wind_directions_VIV = wind_directions_VIV[np.isfinite(wind_directions_VIV)]
    

    # 洗掉NAN
    wind_directions_VIV = wind_directions_VIV[np.isfinite(wind_velocities_VIV)]
    wind_TIs_VIV = wind_TIs_VIV[np.isfinite(wind_velocities_VIV)]
    wind_rmss_VIV = wind_rmss_VIV[np.isfinite(wind_velocities_VIV)]
    wind_velocities_VIV = wind_velocities_VIV[np.isfinite(wind_velocities_VIV)]


    ploter = PlotLib()
    fig, ax = ploter.hist(
        x = wind_velocities_VIV,
        xlabel = 'RMSs$',
        ylabel = 'Frequency',
        color = 'skyblue'
    )

    figs.append(fig)

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()


    return 

def Normal_wind_Space_Directions_Distribution():

    unpacker = UNPACK()
    wind_paths = unpacker.wind_path_lis
    time_interval = 10
    fs = 1

    wind_directions = np.array([0])
    wind_velocities = np.array([0])
    wind_angles = np.array([0])

    for path in wind_paths:
        wind_velocity_slice, wind_direction_slice, wind_angles_slice = unpacker.File_Detach_Data(path, time_interval = time_interval, mode = 'UAN')
        
        for i in range(int(60 / time_interval)):

            wind_velocity, wind_direction, wind_angle = wind_velocity_slice[i], wind_direction_slice[i], wind_angles_slice[i]

            wind_direction_mean = np.mean(wind_direction)
            wind_velocity_mean = np.mean(wind_velocity)
            wind_angle_mean = np.mean(wind_angle)

            wind_directions = np.hstack((wind_directions, wind_direction_mean))
            wind_velocities = np.hstack((wind_velocities, wind_velocity_mean))
            wind_angles = np.hstack((wind_angles, wind_angle_mean))

    wind_directions = wind_directions[1:]
    wind_velocities = wind_velocities[1:]
    wind_angles = wind_angles[1:]
    
    # 洗掉NAN
    wind_velocities = wind_velocities[np.isfinite(wind_directions)]
    wind_angles = wind_angles[np.isfinite(wind_directions)]
    wind_directions = wind_directions[np.isfinite(wind_directions)]

    # 洗掉NAN
    wind_directions = wind_directions[np.isfinite(wind_velocities)]
    wind_angles = wind_angles[np.isfinite(wind_velocities)]
    wind_velocities = wind_velocities[np.isfinite(wind_velocities)]

    # 洗掉NAN
    wind_directions = wind_directions[np.isfinite(wind_angles)]
    wind_velocities = wind_velocities[np.isfinite(wind_angles)]
    wind_angles = wind_angles[np.isfinite(wind_angles)]

    # 平均风速应当小于60
    wind_directions = wind_directions[wind_velocities < 60]
    wind_angles = wind_angles[wind_velocities < 60]
    wind_velocities = wind_velocities[wind_velocities < 60]

    # 风向小于360
    wind_angles = wind_angles[wind_directions < 360]
    wind_velocities = wind_velocities[wind_directions < 360]
    wind_directions = wind_directions[wind_directions < 360]

    # 攻角应当在-90~90间
    wind_velocities = wind_velocities[wind_angles < 90]
    wind_directions = wind_directions[wind_angles < 90]
    wind_angles = wind_angles[wind_angles < 90]

    wind_velocities = wind_velocities[wind_angles > -90]
    wind_directions = wind_directions[wind_angles > -90]
    wind_angles = wind_angles[wind_angles > -90]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    # 正北修正
    # 设弧长为5, 从1开始（而不是零）
    R1 = 3
    R2 = 2.8
    X, Y, Z, U, V, W = [], [], [], [], [], []
    for i in range(len(wind_directions)):
        Angle = wind_angles[i]
        Direction = wind_directions[i]
        
        # 将风角转化成球坐标
        # 弧长设为0.5, 1
        phi = np.deg2rad(180 - Direction)
        theta = np.deg2rad(90 - Angle)

        x1 = R1 * np.sin(theta) * np.cos(phi)
        y1 = R1 * np.sin(theta) * np.sin(phi)
        z1 = R1 * np.cos(theta)

        vector_start = (x1, y1, z1)
        
        x2 = R2 * np.sin(theta) * np.cos(phi)
        y2 = R2 * np.sin(theta) * np.sin(phi)
        z2 = R2 * np.cos(theta)
        vector_end = (x2, y2, z2)
        
        X.append(vector_start[0])
        Y.append(vector_start[1])
        Z.append(vector_start[2])

        U.append(vector_end[0] - vector_start[0])
        V.append(vector_end[1] - vector_start[1])
        W.append(vector_end[2] - vector_start[2])


    ax.quiver(X = X, Y = Y, Z = Z, 
            U = U, V = V, W = W, 
            color = 'skyblue', label = 'Normal Samples')

    VIV_file = 'test1219.xlsx'
    sensor_id = 'ST-VIC-C18-102-01'
    wind_sensors = ['ST-UAN-G04-001-01']
    df_VIV = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\\' + VIV_file)
    X_VIV, Y_VIV, Z_VIV, U_VIV, V_VIV, W_VIV = [], [], [], [], [], []
    R1 = 1
    R2 = 0.3
    for PNG_path, user_input in df_VIV.values:

        VIC_path,  = unpacker.PNG_Path_2_VIC_Path(PNG_path)
        if VIC_path == None: 
            continue

        wind_path = unpacker.VIC_Path_2_WindPath(VICpath = VIC_path, wind_sensor_ids= wind_sensors)
        if len(wind_path) == 0:
            continue

        wind_velocity_slice, wind_direction_slice, wind_angles_slice = unpacker.File_Detach_Data(path = wind_path[0], time_interval = time_interval, mode = 'UAN')
        for i in range(int(60 / time_interval)):
            wind_velocity, wind_direction, wind_angle = wind_velocity_slice[i], wind_direction_slice[i], wind_angles_slice[i]

            Angle = np.mean(wind_angle)
            Direction = np.mean(wind_direction)

            # 将风角转化成球坐标
            # 弧长设为0.5, 1
            phi = np.deg2rad(180 - Direction)
            theta = np.deg2rad(90 - Angle)

            x1 = R1 * np.sin(theta) * np.cos(phi)
            y1 = R1 * np.sin(theta) * np.sin(phi)
            z1 = R1 * np.cos(theta)

            vector_start = (x1, y1, z1)
            
            x2 = R2 * np.sin(theta) * np.cos(phi)
            y2 = R2 * np.sin(theta) * np.sin(phi)
            z2 = R2 * np.cos(theta)
            vector_end = (x2, y2, z2)

            X_VIV.append(vector_start[0])
            Y_VIV.append(vector_start[1])
            Z_VIV.append(vector_start[2])

            U_VIV.append(vector_end[0] - vector_start[0])
            V_VIV.append(vector_end[1] - vector_start[1])
            W_VIV.append(vector_end[2] - vector_start[2])

    ax.quiver(X = X_VIV, Y = Y_VIV, Z = Z_VIV, 
            U = U_VIV, V = V_VIV, W = W_VIV, 
            color = 'pink', label = 'VIV Samples')

    font_size = 10
    # 用来画柱坐标的圈圈
    degs = np.linspace(0, 360, 360 * 5)
    circle_nums = 4
    # 添加自定义刻度和标签
    for i in range(1, circle_nums):
        x_xy_axis = R1 * i / circle_nums * np.cos(np.deg2rad(degs)) * 1.1
        y_xy_axis = R1 * i / circle_nums * np.sin(np.deg2rad(degs)) * 1.1
        z_xy_axis = np.zeros(len(x_xy_axis))
        ax.plot(x_xy_axis, y_xy_axis, z_xy_axis, zdir='z', color = 'grey', linestyle = '--', linewidth = 0.5) 

    # 默认正北为0°
    # 添加方向角
    strings = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    dir_rho = np.max(R1) * 1.2
    dir_theta = np.array([90, 45, 0, -45, -90, -135, -180, -225])
    x_xy_ticks = dir_rho * np.cos(np.deg2rad(dir_theta))
    y_xy_ticks = dir_rho * np.sin(np.deg2rad(dir_theta))
    z_xy_ticks = np.zeros(len(dir_theta))
    for i in range(len(dir_theta)):
        theta_tick = dir_theta[i]
        x_tick = x_xy_ticks[i]
        y_tick = y_xy_ticks[i]
        z_tick = z_xy_ticks[i]
        ax.text(x = x_tick, y = y_tick, z = z_tick, s = strings[i], fontsize = font_size)


    for i in range(len(dir_theta)):
        x_tick = (0, dir_rho * np.cos(np.deg2rad(dir_theta[i])))
        y_tick = (0, dir_rho * np.sin(np.deg2rad(dir_theta[i])))
        z_tick = (0, 0)
        ax.plot(x_tick, 
                y_tick, 
                z_tick, 
                color = 'grey', 
                linewidth = 0.5)
    
    ax.set_axis_off()
    ax.legend()
    x_focus, y_focus, z_focus = 0, 0, 0
    range_value = 1  # 可视范围的一半
    ax.set_xlim([x_focus - range_value, x_focus + range_value])
    ax.set_ylim([y_focus - range_value, y_focus + range_value])
    ax.set_zlim([z_focus - range_value, z_focus + range_value])

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()



    return 

def Response_Data_Within_Time():

    root = 'F:/Research/My_Thesis/Data/苏通/VIC/'
    unpacker = UNPACK()
    VIC_Path_lis = unpacker.File_Read_Paths(root)
    filt = Abnormal_Vibration_Filter()

    # 上下游拉索加速度散点图
    VIC_Name_Lis_Up_Down_Stream_In_Plane = [
        'ST-VIC-C18-101-01', 
        # 'ST-VIC-C18-102-01', # 面内
        # 'ST-VIC-C18-401-01', 
        # 'ST-VIC-C18-402-01',
        # 'ST-VIC-C18-501-01', 
        # 'ST-VIC-C18-502-01',

    ]

    VIC_Name_Lis_Up_Down_Stream_Out_Plane = [
        'ST-VIC-C18-101-02', 
    #     'ST-VIC-C18-102-02', # 面外
    #     'ST-VIC-C18-401-02', 
    #     'ST-VIC-C18-402-02',
    #     'ST-VIC-C18-501-02', 
    #     'ST-VIC-C18-502-02',
    ]

    sensor = VIC_Name_Lis_Up_Down_Stream_In_Plane[0]
    root = 'F:/Research/My_Thesis/Data/苏通/VIC/09/'
    day_num = 30
    time_num = 24
    VIC_path_Series = []
    for day in range(1, day_num + 1):
        for time in range(time_num):

            if time / 10 < 1:
                time_str = '0' + str(time)
            else:
                time_str = str(time)
            match_pattern = root + str(day) + '/' + sensor +  '_' + time_str
            VIC_path = unpacker.File_Match_Pattern(match_pattern, paths = VIC_Path_lis)
            if len(VIC_path) > 0:
                VIC_path_Series.append(VIC_path[0])

    response = np.array([0])
    for path in VIC_path_Series:
        data = unpacker.VIC_DATA_Unpack(path)
        response = np.hstack((response, data))

    response = response[1: ]

    ploter = PlotLib()
    fig, ax = ploter.plot(
        y = response, 
        x = np.arange(len(response)), 
        ylabel = 'Acceleration $(m/s^2)$', 
        xlabel = 'Time', 
        legend = 'Inplane',
        title = 'Time Series for September',
        alpha = 0.8,
    )
    # -------------------------------------------面外-----------------------------------------------
    sensor = VIC_Name_Lis_Up_Down_Stream_Out_Plane[0]
    root = 'F:/Research/My_Thesis/Data/苏通/VIC/09/'
    day_num = 30
    time_num = 24
    VIC_path_Series = []
    for day in range(1, day_num + 1):
        for time in range(time_num):

            if time / 10 < 1:
                time_str = '0' + str(time)
            else:
                time_str = str(time)
            match_pattern = root + str(day) + '/' + sensor +  '_' + time_str
            VIC_path = unpacker.File_Match_Pattern(match_pattern, paths = VIC_Path_lis)
            if len(VIC_path) > 0:
                VIC_path_Series.append(VIC_path[0])

    response = np.array([0])
    for path in VIC_path_Series:
        data = unpacker.VIC_DATA_Unpack(path)
        response = np.hstack((response, data))

    response = response[1: ]

    ploter = PlotLib()
    fig, ax = ploter.plot(
        y = response, 
        x = np.arange(len(response)), 
        fig = fig, 
        ax = ax, 
        legend = 'Outplane', 
        color = 'pink', 
        alpha = 0.8,
    )

    # 给坐标轴标上时间间隔
    # 9月一共30天
    total_days_num = 30
    day_points_num = len(response) / total_days_num
    day_step = 5

    indexes = []
    ticks = []
    date_start = datetime.datetime(2024, 9, 1)

    for day in np.arange(0, total_days_num + day_step, day_step):

        index = day_points_num * day
        time_delta = datetime.timedelta(days = int(day))
        tick = (date_start + time_delta).strftime("%m-%d")
        indexes.append(index)
        ticks.append(tick)

    ax.set_xticks(indexes, ticks)
    ax.legend()
    # 调整fig的大小
    fig.set_size_inches(20, 8)

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()

    return 

def Wind_Velocity_Data_Within_Time():


    unpacker = UNPACK()
    UAN_Path = unpacker.wind_path_lis

    wind_sensor_id = 'ST-UAN-G04-001-01'
    wind_path_lis = unpacker.File_Match_Pattern(wind_sensor_id, UAN_Path)

    day_num = 30
    time_num = 24
    UAN_adjusted_Series = []
    root = r'F:/Research/My_Thesis/Data/苏通/UAN/09/'
    for day in range(1, day_num + 1):
        for time in range(time_num):

            if time / 10 < 1:
                time_str = '0' + str(time)
            else:
                time_str = str(time)

            if day / 10 < 1:
                day_str = '0' + str(day)
            else:
                day_str = str(day)
            
            match_pattern = root + day_str + '/' + wind_sensor_id +  '_' + time_str
            UAN_path = unpacker.File_Match_Pattern(match_pattern, paths = wind_path_lis)
            if len(UAN_path) > 0:
                UAN_adjusted_Series.append(UAN_path[0])

    wind_velocities = np.array([0])
    for path in UAN_adjusted_Series:
        wind_velocity, wind_direction, wind_angle = unpacker.Wind_Data_Unpack(path)
        wind_velocities = np.hstack((wind_velocities, wind_velocity))
    
    wind_velocities = wind_velocities[1:]

    ploter = PlotLib()
    fig, ax = ploter.plot(
        y = wind_velocities, 
        x = np.arange(len(wind_velocities)), 
        ylabel = 'Wind Speed $(m)$', 
        xlabel = 'Time', 
        title = 'Wind Speed in September',
    )

    # 给坐标轴标上时间间隔
    # 9月一共30天
    total_days_num = 30
    day_points_num = len(wind_velocities) / total_days_num
    day_step = 5

    indexes = []
    ticks = []
    date_start = datetime.datetime(2024, 9, 1)

    for day in np.arange(0, total_days_num + day_step, day_step):

        index = day_points_num * day
        time_delta = datetime.timedelta(days = int(day))
        tick = (date_start + time_delta).strftime("%m-%d")
        indexes.append(index)
        ticks.append(tick)

    ax.set_xticks(indexes, ticks)

    # 调整fig的大小
    fig.set_size_inches(20, 8)

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()

    return 

def test():


    ploter = PlotLib()
    z = np.random.random(size = 1000) * 100
    rho = np.random.random(size = 1000) * 6
    theta = np.random.random(size = 1000) * 2 * np.pi

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fig, ax = ploter.rose_scatter(theta = theta, rho = rho, cmap = 'RdYlGn', c_data = z, color_label = 'Test')

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")

    # 创建应用
    app = ChartApp(root, [fig])

    # 运行主循环
    root.mainloop()


    return

def VIV_Trajectory_Fail():

    Inplane_VIV_png_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIVSamplesVI\M09_D04_T18_TT41_ST-VIC-C18-102-01Response.png'
    Outplane_VIV_png_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIVSamplesVI\M09_D04_T18_TT41_ST-VIC-C18-102-02Response.png'
    unpacker = UNPACK()
    Inplane_data = unpacker.PNG_Path_2_VIC_Data(Inplane_VIV_png_path)
    Outplane_data = unpacker.PNG_Path_2_VIC_Data(Outplane_VIV_png_path)
    
    ploter = PlotLib()
    data = [Inplane_data, Outplane_data]
    figs = []
    fs = 50
    for i in range(2):   
        response_time_series = data[i]
        fx, Pxx_den = signal.welch(response_time_series, fs = 50, nfft = 512, 
                                nperseg = 512, noverlap = 256)

        x = np.arange(len(response_time_series)) / fs # 秒作为单位画图
        data_lis = [
            (x, response_time_series),
            (fx, Pxx_den) 
            ]
        
        titles = [
            'Time Series',
            'Power Spectral Density'
        ]

        labels = [
            ('Time(s)', 'Acceleration $(m/s^2)$'),
            ('Frequency(hz)', 'PSD')
        ]


        fig, axes = ploter.plots(data_lis, titles = titles, labels = labels)
        figs.append(fig)
        plt.close()

    fs = 50
    times = np.arange(len(Inplane_data)) / fs

    print(Inplane_data.shape)
    ploter = PlotLib()

    X_velocity = []
    Y_velocity = []
    for step in range(len(Inplane_data)):
        integral_x = np.trapz(Inplane_data[0 : step], times[0 : step])
        integral_y = np.trapz(Outplane_data[0 : step], times[0 : step])
        X_velocity.append(integral_x)
        Y_velocity.append(integral_y)

    X_velocity = np.array(X_velocity)
    Y_velocity = np.array(Y_velocity)

    X_trajectory = []
    Y_trajectory = [] 

    for step in range(len(X_velocity)):
        integral_x = np.trapz(X_velocity[0 : step], times[0 : step])
        integral_y = np.trapz(Y_velocity[0 : step], times[0 : step])
        X_trajectory.append(integral_x)
        Y_trajectory.append(integral_y)

    X_trajectory = np.array(X_trajectory)
    Y_trajectory = np.array(Y_trajectory)

    fig, ax = ploter.plot(
        y = Y_trajectory, 
        x = X_trajectory, 
        ylabel = 'Displacement $(m)$', 
        xlabel = 'Displacement $(m)$', 
        title = 'Trajectory under VIV',
    )

    figs.append(fig)

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")
    app = ChartApp(root, figs)
    root.mainloop()



    return 

def VIV_mode_variation():

    length = 3
    Inplane_VIV_png_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIVSamplesVI\M09_D11_T13_TT26_ST-VIC-C18-102-01Response.png'
    # Outplane_VIV_png_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\VIVSamplesVI\M09_D04_T18_TT41_ST-VIC-C18-102-02Response.png'
    unpacker = UNPACK()
    Inplane_path = unpacker.PNG_Path_2_VIC_Path(Inplane_VIV_png_path)
    # Outplane_data = unpacker.PNG_Path_2_VIC_Data(Outplane_VIV_png_path)
    
    ploter = PlotLib()

    inplane_data_slice = unpacker.File_Detach_Data(Inplane_path[0], time_interval = 1)

    file_name = Path(Inplane_VIV_png_path).parts[-1]
    month, day, hours, minutes = file_name.split('_')[:-1]
    start_index = int(minutes[2:]) # min
    datas = np.array([0])

    fxs, Pxx_dens = [], []
    fs = 50
    for data in inplane_data_slice[start_index : start_index + length]:
        datas = np.hstack((datas, data))

        f, Pxx_den = signal.welch(data, fs = 50, nfft = 512, 
                                nperseg = 512, noverlap = 256)
        fxs.append(f)
        Pxx_dens.append(Pxx_den)

    datas = datas[1:]

    x = np.arange(len(datas)) / fs # 秒作为单位画图
    data_lis = [
        (x, datas),
        (np.array([0]), np.array([0])) 
        ]
    
    titles = [
        f'Time Series (2024-{month[1:]}-{day[1:]} {hours[1:]}:{minutes[2:]})',
        'Power Spectral Density'
    ]

    labels = [
        ('Time(s)', 'Acceleration $(m/s^2)$'),
        ('Frequency(hz)', 'PSD')
    ]

    fig, axes = ploter.plots(data_lis, 
                            #  titles = titles, 
                            #  labels = labels
                             )

    for i in range(length):
        fx = fxs[i]
        Pxx_den = Pxx_dens[i]
    
        fig, ax = ploter.plot(
            y = Pxx_den, 
            x = fx, 
            color = 'pink', 
            fig = fig, 
            ax = axes[1], 
        )

    # colors = [
    #     '#B7B7EB', 
    #     '#F09BA0',
    #     '#EAB883',
    #     '#9BBBE1', 
    #     '#9D9EA3',
    # ]

    colors = [
        'skyblue', 
        'deepskyblue',
        'blueviolet'
    ]

    lines = axes[1].get_lines()[1:]
    for i, line in enumerate(lines):
        line.set_color(colors[i])

    # 添加图例
    # labels = [f't = { i } ~ {i + 1} min' for i in range(length)]
    # axes[1].legend(lines, labels)
    axes[1].grid(False)
    axes[1].autoscale()
    axes[1].get_lines()[0].remove()

    # ---------------------------------------------聚焦部分，可删除------------------------------------------
    major_frequency = np.max(Pxx_dens[0])
    fs_center = fxs[0][np.where(Pxx_dens[0] == major_frequency)]
    axes[1].set_xlim(fs_center - 1, fs_center + 1)

    root = tk.Tk()
    root.title("Matplotlib Charts in Tkinter")
    app = ChartApp(root, [fig])
    root.mainloop()

    return 

def VIV_Trajectory_Fail2():

    df = pd.read_excel('./Ansys_Mode_Analysis/Base_Mode_of_MidSpan.xlsx')
    base_modes_Inplane = np.array(df['拉索(Hz)(面内)'])
    base_modes_Outplane = np.array(df['拉索(Hz)(面外)'])

    Inplane_VIV_png_path = './Img/VIVSamplesVI/M09_D04_T18_TT41_ST-VIC-C18-102-01Response.png'
    Outplane_VIV_png_path = './Img/VIVSamplesVI/M09_D04_T18_TT41_ST-VIC-C18-102-02Response.png'
    unpacker = UNPACK()
    Inplane_data = unpacker.PNG_Path_2_VIC_Data(Inplane_VIV_png_path)
    Outplane_data = unpacker.PNG_Path_2_VIC_Data(Outplane_VIV_png_path)
    window = 150
    Inplane_Trajectory = unpacker.displacement_calculation(Inplane_data, base_modes_Inplane, window = window)
    Outplane_Trajectory = unpacker.displacement_calculation(Outplane_data, base_modes_Outplane, window = window)

    # 需要找到一个单独的环形才对

    ploter = PlotLib()
    time_nums = 8
    fs = 50
    fs_ = fs * time_nums
    labels = []
    titles = []
    data_lis = []
    figs = []
    # 使用样条插值
    from scipy.interpolate import interp1d,splrep
    integrate_Inplane_curve = interp1d(np.arange(len(Inplane_data)) / fs, Inplane_data, kind='cubic')
    time_new = np.arange(len(Inplane_data) * time_nums)[:-1] / fs_
    
    integrate_Inplane_Data = integrate_Inplane_curve(time_new[time_new < (np.arange(len(Inplane_data)) / fs)[-1]])
    # 去除均值
    integrate_Inplane_Data = integrate_Inplane_Data - np.mean(integrate_Inplane_Data)

    # 查看样本
    fx, pxxden = signal.welch(integrate_Inplane_Data, fs = fs_, nfft = 65536, 
                                nperseg = 256, noverlap = 1)
    
    times = np.arange(len(integrate_Inplane_Data)) / fs_
    data_lis = [
        (times, integrate_Inplane_Data),
        (fx, pxxden) 
        ]
    
    titles = [
        f'Time Series',
        'Power Spectral Density'
    ]

    labels = [
        ('Time(s)', 'Acceleration $(m/s^2)$'),
        ('Frequency(hz)', 'PSD')
    ]

    fig, axes = ploter.plots(data_lis, 
                             titles = titles, 
                             labels = labels, 
                             coordinate_systems = ['linear', 'semilogy']
                             )
    figs.append(fig)

    fig, axes = ploter.plots(data_lis, 
                             titles = titles, 
                             labels = labels, 
                             )
    figs.append(fig)

    # 试一下积分曲线
    # 第一次积分，得到速度信号
    # 积分过程可以使用cumtrapz（累积积分）来实现数值积分
    velocityi = np.cumsum(integrate_Inplane_Data) / fs_
    # 去除一次项（拟合并去除线性趋势）
    p_velocityi = np.polyfit(times, velocityi, 4)  # 线性拟合
    velocityi -= np.polyval(p_velocityi, times)
    # 第二次积分，得到位移信号
    displacementi = np.cumsum(velocityi) / fs_
    # 去除一次项（拟合并去除线性趋势）
    p_displacementi = np.polyfit(times, displacementi, 4)  # 线性拟合
    displacementi -= np.polyval(p_displacementi, times)

    data_lis = [
        (times, integrate_Inplane_Data),
        (fx, pxxden),
        (times, displacementi),
        ]
    
    titles = [
        f'Time Series',
        'Power Spectral Density',
        'Displacement'
    ]

    labels = [
        ('Time(s)', 'Acceleration $(m/s^2)$'),
        ('Frequency(hz)', 'PSD'),
        ('Time(s)', 'Displacement $(m)$')
    ]

    fig, axes = ploter.plots(data_lis, 
                             titles = titles, 
                             labels = labels, 
                             )
    figs.append(fig)


    window = 100
    time_nums = 2
    Inplane_Trajectory_i = unpacker.displacement_calculation(integrate_Inplane_Data, 
                                                                base_modes_Inplane, 
                                                                window = window, 
                                                                fs = fs_)
    inplane_times = np.arange(len(Inplane_Trajectory_i)) / fs_
    y = Inplane_Trajectory_i
    # 画一下得到的面内振动图
    data_lis = [
        (times, integrate_Inplane_Data),
        (fx, pxxden),
        (inplane_times, Inplane_Trajectory_i),
        ]
    
    titles = [
        f'Time Series',
        'Power Spectral Density',
        'Displacement'
    ]

    labels = [
        ('Time(s)', 'Acceleration $(m/s^2)$'),
        ('Frequency(hz)', 'PSD'),
        ('Time(s)', 'Displacement $(m)$')
    ]

    fig, axes = ploter.plots(data_lis, 
                             titles = titles, 
                             labels = labels, 
                             )
    figs.append(fig)
    # 原始时域图像
    fig, ax = ploter.plot(y = integrate_Inplane_Data,
                            x =  times
                             )
    ax.set_xlim(10, 30)
    figs.append(fig)


    integrate_Outplane_curve = interp1d(np.arange(len(Outplane_data)) / fs, Outplane_data, kind='cubic')
    time_new = np.arange(len(Outplane_data) * time_nums)[:-1] / fs_
    
    integrate_Outplane_Data = integrate_Outplane_curve(time_new)
    # 去除均值
    integrate_Outplane_Data = integrate_Outplane_Data - np.mean(integrate_Outplane_Data)

    window = 100
    time_nums = 2
    Outplane_Trajectory_i = unpacker.displacement_calculation(integrate_Outplane_Data, 
                                                                base_modes_Inplane, 
                                                                window = window, 
                                                                fs = fs_)
    
    x = Outplane_Trajectory_i

    lengths = [10, 50, 100, 200, 500, len(x)]
    for length in lengths:
        fig, ax =  ploter.scatter(
            y = y[:length], 
            x = x[:length], 
            xlim = (-0.05, 0.05),
            ylim = (-1.8, 1.8),
            xlabel = 'Displacement(m)',
            ylabel = 'Displacement(m)', 
            title = 'Trajectory', 
                    )
        ax.grid(False)
        figs.append(fig)

    fig, ax =  ploter.scatter(
        y = y[:length], 
        x = x[:length], 
        # xlim = (-0.8, 0.8),
        # ylim = (-0.1, 0.1),
        xlabel = 'Displacement(m)',
        ylabel = 'Displacement(m)', 
        title = 'Trajectory', 
                )
    ax.grid(False)
    figs.append(fig)

    tk = Tk()
    app = ChartApp(tk, figs)
    tk.mainloop()

    return 

def VIV_Trajectory_fail3():
    df = pd.read_excel('./Ansys_Mode_Analysis/Base_Mode_of_MidSpan.xlsx')
    base_modes_Inplane = np.array(df['拉索(Hz)(面内)'])
    base_modes_Outplane = np.array(df['拉索(Hz)(面外)'])

    Inplane_VIV_png_path = './Img/VIVSamplesVI/M09_D16_T15_TT10_ST-VIC-C18-501-01Response.png'
    Outplane_VIV_png_path = './Img/VIVSamplesVI/M09_D16_T15_TT10_ST-VIC-C18-501-02Response.png'
    unpacker = UNPACK()
    Inplane_data = unpacker.PNG_Path_2_VIC_Data(Inplane_VIV_png_path)
    Outplane_data = unpacker.PNG_Path_2_VIC_Data(Outplane_VIV_png_path)
    
    # 筛选出具有面外振动的数据
    # root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\Samples\VIVSamplesVI\\'
    # for file_name in os.listdir(root):
    #     Inplane_path = root + file_name
    #     Inplane_data = unpacker.PNG_Path_2_VIC_Data(Inplane_path)
    #     Outplane_path = Inplane_path.replace('01Response', '02Response')
    #     Outplane_data = unpacker.PNG_Path_2_VIC_Data(Outplane_path)
    #     if len(Outplane_data) > 0:
    #         if np.mean(Outplane_data) > 0.01:
    #             print(Outplane_path)

    ploter = PlotLib()
    ploter.show_sample(Inplane_data)
    ploter.show_sample(Outplane_data)

    for i in range(10):

        ploter.plot(
            x = Outplane_data[:i * 10], 
            y = Inplane_data[:i * 10], 
            title = f'Step: {i * 10}'
        )
        plt.close()
 

    tk = Tk()
    app = ChartApp(tk, ploter.figs)
    tk.mainloop()


    return 

def VIV_Trajectory_fail4():

    Inplane_path = r'F:\Research\My_Thesis\Data\苏通\VIC\09\09\ST-VIC-C18-101-01_120000.VIC'
    Outplane_path = r'F:\Research\My_Thesis\Data\苏通\VIC\09\09\ST-VIC-C18-101-02_120000.VIC'
    unpacker = UNPACK()
    Inplane_data = unpacker.VIC_DATA_Unpack(Inplane_path)
    Outplane_data = unpacker.VIC_DATA_Unpack(Outplane_path)
    
    # 筛选出具有面外振动的数据
    # root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Img\Samples\VIVSamplesVI\\'
    # for file_name in os.listdir(root):
    #     Inplane_path = root + file_name
    #     Inplane_data = unpacker.PNG_Path_2_VIC_Data(Inplane_path)
    #     Outplane_path = Inplane_path.replace('01Response', '02Response')
    #     Outplane_data = unpacker.PNG_Path_2_VIC_Data(Outplane_path)
    #     if len(Outplane_data) > 0:
    #         if np.mean(Outplane_data) > 0.01:
    #             print(Outplane_path)

    ploter = PlotLib()
    ploter.show_sample(Inplane_data)
    ploter.show_sample(Outplane_data)

    for i in range(10):

        ploter.plot(
            x = Outplane_data[:i * 10], 
            y = Inplane_data[:i * 10], 
            title = f'Step: {i * 10}'
        )
        plt.close()
 

    tk = Tk()
    app = ChartApp(tk, ploter.figs)
    tk.mainloop()

def sampling_process__large_amplitude_samples():
    """
    以大于1为阈值，筛选出振动加速度大于1m/ss的值，找到VIV样本
    """
    # 上下游拉索加速度散点图
    sensors = [
        'ST-VIC-C18-101-01', 
        'ST-VIC-C18-102-01', # 面内
        'ST-VIC-C18-401-01', 
        'ST-VIC-C18-402-01',
        'ST-VIC-C18-501-01', 
        'ST-VIC-C18-502-01',

        # 'ST-VIC-C18-101-02', 
        # 'ST-VIC-C18-102-02', # 面外
        # 'ST-VIC-C18-401-02', 
        # 'ST-VIC-C18-402-02',
        # 'ST-VIC-C18-501-02', 
        # 'ST-VIC-C18-502-02',
    ]

    unpacker = UNPACK()
    ploter = PlotLib()
    all_files = unpacker.File_Read_Paths(r'F:\工作\XM_极端风致灾变\ST加速度传感器数据\VIC\\')
    i = 0 # 用于迭代保存的文件名称

    def ifile_name(i):
        if i < 10:
            file_name = '0000' + str(i) 
        
        elif i >= 10 and i < 100:
            file_name = '000' + str(i) 
            
        elif i >= 100 and i < 1000:
            file_name = '00' + str(i) 
        
        elif i >= 1000 and i < 10000:
            file_name = '0' + str(i) 

        elif i >= 10000 and i < 100000:
            file_name = str(i) 

        return file_name
    from scipy.io import savemat
    Train_save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\test\data\\'.replace('\\', '/')
    Dev_save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\2023\\'.replace('\\', '/')
    
    from tqdm import tqdm
    for sensor in sensors:
        file_lis = unpacker.File_Match_Sensor_Path(sensor, all_files)
        with tqdm(total = len(file_lis)) as pbar:
            for file in file_lis:
                data_slice = unpacker.File_Detach_Data(path = file, time_interval = 1, mode = 'VIC')
                for time, data in enumerate(data_slice):
                    if np.max(data) > 1 and np.max(data) < 2:

                        file_name = ifile_name(i)
                        savemat(file_name = Train_save_root + file_name + '.mat', mdict = {'data':data})

                        plt.close()
                        i += 1

                        # 这一部分很有可能找得到过渡阶段
                        continue
                
                pbar.update(1)



    return 

def read_samples():
    # 第一轮筛选结果
    processer = Data_Process()
    processer.Proposal_Samples(
        save_excel_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\2023\\sample_labels.xlsx', 
    )
    return 

def process_samples():
    import pandas as pd
    from PIL import Image
    from scipy.io import loadmat, savemat
    df = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\2023\\sample_labels.xlsx')
    
    normal_v_data_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\Normal_Vibration\data\\'
    normal_v_img_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\Normal_Vibration\img\\'

    VIV_data_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\VIV\data\\'
    VIV_img_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\VIV\img\\'

    data_root = r"F:/Research/My_Thesis/Vibration Characteristics In Cable Vibration/Samples/2023/File/"
    for file_path, user_input in zip(df['File Path'], df['User Input']):
        file_path = file_path.replace('\\', '/')
        
        # user_input为0，表示一般振动，添置到相应文件夹中
        img = Image.open(file_path)
        
        # file_path形如"F:/Research/My_Thesis/Vibration Characteristics In Cable Vibration/Samples/Train/Img\0291.png"
        # 找到文件名称
        file = Path(file_path).parts[-1]
        data_path = data_root + file[:-4] + '.mat'

        # 保存到对应位置
        # 后续2023年筛选的结果都由1xxxx开头，即1万开头，防止撞名
        if user_input == 0: # 一般振动
            img.save(normal_v_img_root +file)
            savemat(normal_v_data_root +file[:-4] + '.mat', {'data':loadmat(data_path)['data']})
        elif user_input == 1:
            img.save(VIV_img_root + '20' + file)
            savemat(VIV_data_root + '20' + file[:-4] + '.mat', {'data':loadmat(data_path)['data']})

        pass

    return 

def read_samples_twice():
    from tkinter import font
    # 第一轮筛选结果
    twice_save_excel_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Train\\Twice_sample_labels.xlsx'
    first_save_excel_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Train\\sample_labels.xlsx'
    df = pd.read_excel(first_save_excel_path)
    
    class twice_Propose_Data_GUI(_Propose_Data_GUI):
        def __init__(self, root):
            super().__init__(
                root = root, 
                save_excel_root = twice_save_excel_path, 
                image_list = list(df['File Path'])
            )

            self.text_lis = list(df['User Input'])
            self.entry_var = tk.StringVar(value = None)
            self.text_update = tk.Entry(root, 
                                        textvariable = self.entry_var, justify = 'center', 
                                        font = font.Font(family = 'Sim Hei', size = 16, weight = 'bold'))
            self.text_update.grid(row=0, column=1)


            self.update_text()
            return 
        
        def update_text(self):
            if self.current_image_index < len(self.image_list):
                # 插入列表中的下一个值到文本框
                if self.text_lis[self.current_image_index] == 0:
                    text = 'Normal Vibration(0)'
                elif self.text_lis[self.current_image_index] == 1:
                    text = 'VIV(1)'

                self.entry_var.set(text)
                self.entry_text.set(str(self.text_lis[self.current_image_index]))

        def next_image(self, event=None):
            if self.current_image_index < len(self.image_list) - 1:
                self.current_image_index += 1
                self.show_image()
                self.update_text()

        def prev_image(self, event=None):
            if self.current_image_index > 0:
                self.current_image_index -= 1
                self.show_image()
                self.update_text()

        def show_image(self):
            if not self.image_list:
                return
            img = Image.open(self.image_list[self.current_image_index])
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.label.config(image=photo)
            self.label.image = photo

    root = Tk()
    app = twice_Propose_Data_GUI(root)
    root.mainloop()

    return 

def process_samples_twice():
    import pandas as pd
    from PIL import Image
    from scipy.io import loadmat, savemat
    df = pd.read_excel(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Train\\Twice_sample_labels - 副本.xlsx')
    
    normal_v_data_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\Normal_Vibration\data\\'
    normal_v_img_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\Normal_Vibration\img\\'

    VIV_data_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\VIV\data\\'
    VIV_img_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\VIV\img\\'

    transition_data_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\transition\data\\'
    transition_img_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\transition\img\\'

    data_root = r"F:/Research/My_Thesis/Vibration Characteristics In Cable Vibration/Samples/Train/File/"
    for file_path, user_input in zip(df['File Path'], df['User Input']):
        file_path = file_path.replace('\\', '/')
        
        # user_input为0，表示一般振动，添置到相应文件夹中
        img = Image.open(file_path)
        

        # file_path形如"F:/Research/My_Thesis/Vibration Characteristics In Cable Vibration/Samples/Train/Img\0291.png"
        # 找到文件名称
        file = Path(file_path).parts[-1]
        data_path = data_root + file[:-4] + '.mat'

        # 保存到对应位置
        if user_input == 0: # 一般振动
            img.save(normal_v_img_root + file)
            savemat(normal_v_data_root + file[:-4] + '.mat', {'data':loadmat(data_path)['data']})
        elif user_input == 1:
            img.save(VIV_img_root + file)
            savemat(VIV_data_root + file[:-4] + '.mat', {'data':loadmat(data_path)['data']})
        elif user_input == 3:
            img.save(transition_img_root + file)
            savemat(transition_data_root + file[:-4] + '.mat', {'data':loadmat(data_path)['data']})
 
        pass

    return 

def Collect_All_Data():
    from scipy.io import savemat, loadmat
    sample_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\Vibrations\\'

    unpacker = UNPACK()

    all_sample_paths = unpacker.File_Read_Paths(sample_root)
    # 打乱列表
    import random
    def shuffle_list(lst):
        for i in range(len(lst) - 1, 0, -1):
            j = random.randint(0, i)
            lst[i], lst[j] = lst[j], lst[i]
        return lst
    all_sample_paths = shuffle_list(all_sample_paths)

    datasets_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\datasets\\'.replace('\\', '/')
    sample_nums = len(all_sample_paths) / 2
    index = 0
    for sample_path in all_sample_paths:
        # 确认标签，0为一般振动、1为过渡振动或不确定的振动、2为VIV
        match Path(sample_path).parts[-3]:
            case "Normal_Vibration":
                label = 0
            
            case "transition":
                label = 1
            
            case "VIV":
                label = 2

        # 确认是否为数据，如果不是，则跳出循环
        match Path(sample_path).parts[-2]:
            case "data":
                pass
            
            # 不使用copy image的方法，这样太费事了
            case "img":
                continue
            
        # 采用80-20开，放入训练集和测试集   
        # 训练集0.8、测试集0.2
        if index < sample_nums * 0.8:
            save_path = datasets_root + 'train/' + str(index) + '.mat'
            data = loadmat(sample_path)['data']
            # label:data的形式，采用str形式，可以避免type报错
            savemat(save_path, {str(label):data})
        else:
            save_path = datasets_root + 'dev/' + str(index) + '.mat'
            data = loadmat(sample_path)['data']
            # label:data的形式
            savemat(save_path, {str(label):data})

        save_path = datasets_root + 'all/' + str(index) + '.mat'
        data = loadmat(sample_path)['data']
        # label:data的形式
        savemat(save_path, {str(label):data})

        index += 1
        pass






    return 

def visualize_all_sample():
    from scipy.io import loadmat
    import os
    from torchvision.transforms import ToTensor
    from PIL import Image 
    unpacker = UNPACK()
    current_directory = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\test\data\\'
    with os.scandir(current_directory) as entries:
        paths = [entry.path for entry in entries if entry.is_file()]

    save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\test\img\\'
    ploter = PlotLib()
    for path in paths:
                                                                                            
        label, data = [(key, value) for key, value in loadmat(path).items() if isinstance(value, np.ndarray)][0]

        fig, ax = ploter.show_sample(data[0, :], add_fig = False)
        target_pixels = 512
        dpi = 100
        fig.set_size_inches(target_pixels / dpi, target_pixels / dpi)
        # test
        # fig.savefig(r'C:\Users\admin\Desktop\\' + 'test.png')
        # trans = ToTensor()
        # input_tensor = trans(Image.open(r'C:\Users\admin\Desktop\\' + 'test.png'))[:-1]
        # print(input_tensor.shape)

        img_name = Path(path).parts[-1][:-4] + '.png'
        fig.savefig(save_root + img_name)
        plt.close()
    

    return 

def check_img():
    from scipy.io import loadmat
    import os
    from torchvision.transforms import ToTensor
    from PIL import Image 
    unpacker = UNPACK()
    current_directory = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\img\\'
    with os.scandir(current_directory) as entries:
        paths = [entry.path for entry in entries if entry.is_file()]

    data_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\data\\'
    for path in paths:
        img = Image.open(path)
        tensor = ToTensor()(img)[:3]


        if tensor.shape[1] == 512:
            pass
        else:
            print('Warning, File <{path}> has wrong shape <{tensor.shape}>!')
            # file_name = Path(path).parts[-1][:-4]

            # label, data = [(key, value) for key, value in loadmat(path).items() if isinstance(value, np.ndarray)][0]

            # fig, ax = ploter.show_sample(data[0, :], add_fig = False)
            # target_pixels = 512
            # dpi = 100
            # fig.set_size_inches(target_pixels / dpi, target_pixels / dpi)
            # # test
            # # fig.savefig(r'C:\Users\admin\Desktop\\' + 'test.png')
            # # trans = ToTensor()
            # # input_tensor = trans(Image.open(r'C:\Users\admin\Desktop\\' + 'test.png'))[:-1]
            # # print(input_tensor.shape)

            # img_name = Path(path).parts[-1][:-4] + '.png'
            # fig.savefig(save_root + img_name)

    return 

def confusion_matrix():

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # 假设我们有以下的真实标签和预测标签
    y_true = ["cat", "dog", "cat", "cat", "dog", "rabbit"]
    y_pred = ["dog", "dog", "rabbit", "cat", "dog", "cat"]

    # 定义类别名称
    labels = ["Normal Vibration", "Transition", "VIV"]

    def plot_confusion_matrix(y_true, y_pred, labels, normalize=False):
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            # 归一化处理
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("显示归一化的混淆矩阵")
        else:
            print('显示未归一化的混淆矩阵')

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    # 调用函数，设置normalize为True或False
    plot_confusion_matrix(y_true, y_pred, labels, normalize=True)


    return 

def eval_model():
    
    from NN.EfficientViT.classification.model.efficientvit import EfficientViT
    from NN.datasets import MyDataset
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    Net = EfficientViT(img_size = 512, num_classes = 3).to(device)
    Net.load_state_dict(
        torch.load(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\models\V5\\ECA_Net_fold4.pth', map_location = device)
        )
    Net.eval()

    dataset = MyDataset(
        data_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\data\\', 
        img_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\img\\'
    )

    True_Label = []
    Predicted_Label = []
    labels = ['Normal Vibration', 'VIV']

    def int2label(num):
        if num == 0:
            label = 'Normal Vibration'
        elif num == 1:
            label = 'VIV'
        elif num == 2:
            label = 'VIV'
        return label
    
    VIV_lis = []
    with tqdm(total = len(dataset)) as pbar:
        for i, (img, label) in enumerate(dataset):

            img = img.to(device).unsqueeze(0)
            True_Label.append(int2label(label))
            predict_int = torch.argmax(Net(img))
            Predicted_Label.append(int2label(predict_int))

            pbar.update(1)

            
            if predict_int == 2:
                VIV_lis.append(img[0, :, :, :].cpu().numpy())
            
    # 绘制混淆矩阵
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    def plot_confusion_matrix(y_true, y_pred, labels, normalize=False):
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            # 归一化处理
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("显示归一化的混淆矩阵")
        else:
            print('显示未归一化的混淆矩阵')

        # 绘制混淆矩阵
        fig, ax = plt.subplots()
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Pastel2', xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        return fig, ax 

    
    fig, ax = plot_confusion_matrix(True_Label, Predicted_Label, labels = labels, normalize = True)
    ploter = PlotLib()
    ploter.figs.append(fig)

    # 绘制VIV真值结果
    for img_array in VIV_lis:
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(img_array, (1, 2, 0)))

        ax.grid(False)
        ax.axis('off')

        ploter.figs.append(fig)
        plt.close()


    ploter.show()

    return 

def predict_2023_all():
    from NN.EfficientViT.classification.model.efficientvit import EfficientViT
    from tqdm import tqdm
    from io import BytesIO
    import base64
    from PIL import Image
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader, Dataset

    class MyDataset(Dataset):
        def __init__(self, img_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\2024_09All_Samples\\'):
            self.pathlis = [img_root + entry.name for entry in os.scandir(img_root) if entry.is_file()]
            self.trans = ToTensor()
        def __len__(self):
            return len(self.pathlis)
        
        def __getitem__(self, idx):
            img = Image.open(self.pathlis[idx])
            img = self.trans((img))[:3]
            # 输出图的张量和路径
            return img, self.pathlis[idx]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Net = EfficientViT(img_size = 512, num_classes = 3).to(device)
    Net.load_state_dict(
        torch.load(r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\models\V4\\ECA_Net_fold4.pth', map_location = device)
        )
    Net.eval()

    # 初始化数据库
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = False)

    xlsx_path = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\2024_9VIV.xlsx'
    VIVs = []


    df = pd.DataFrame()
    df['VIV_Paths'] = ''

    with tqdm(total = len(dataloader)) as pbar:
        for nums, (batches, path) in enumerate(dataloader):
            batches = batches.to(device)
            imgpath = path

            # 预测
            predict_label = torch.argmax(Net(batches), dim = 1)
            for i in range(len(predict_label)):
                if predict_label[i] == 2:
                    df.loc[len(df)] = [imgpath[i]]

            pbar.update(1)
            # 记得保存啊
            if (nums + 1) % 100 == 0:
                df.to_excel(xlsx_path, index = False)

        df.to_excel(xlsx_path, index = False)
        print(f'Saved to <{xlsx_path}>... {nums} samples')



    return 

def visualize_all_2024_09_samples():
    from tqdm import tqdm

    unpacker = UNPACK()
    ploter = PlotLib()

    data_paths = unpacker.File_Read_Paths(r'F:\Research\My_Thesis\Data\苏通\VIC\\')
    # 还是太多了，采用单根拉索的数据来做吧 
    data_paths = unpacker.File_Match_Pattern(paths = data_paths, pattern = 'ST-VIC-C18-101-01')
    
    img_save_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\2024_09All_Samples\\'

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # 假设这是我们要加速的函数
    def draw_and_append(path):
        month, day, file_name = Path(path).parts[-3:]
        data_slice = unpacker.File_Detach_Data(path = path, time_interval = 1)
        for time, data in enumerate(data_slice):
            img_name = month + '-' + day + '-' + file_name[:-4] + '-' + str(time) + '.png'
            fig, ax = ploter.show_sample(data, add_fig = False)

            target_pixels = 512
            dpi = 100
            fig.set_size_inches(target_pixels / dpi, target_pixels / dpi)

            # 保存图片
            fig.savefig(img_save_root + img_name, dpi = dpi)
            plt.close()

        return 

    # 使用多线程执行函数
    def threaded_square(paths):
        results = []
        # 创建一个最大并发数为5的线程池
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有任务
            futures = {executor.submit(draw_and_append, path): path for path in paths}
            
            # 收集结果
            with tqdm(total=len(futures), desc="Drawing and appending", unit="image") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        print(f'Generated an exception: {exc}')
                    pbar.update(1)


        return results
    
    # 禁用图像
    import matplotlib
    matplotlib.use('Agg')
    # 调用函数
    results = threaded_square(data_paths)

    return 

def check_img():
    ploter = PlotLib()
    excel = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\Samples\\2024_9VIV.xlsx'
    paths = pd.read_excel(excel)['VIV_Paths'].tolist()
    for path in paths:
        img_array = plt.imread(path)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_array)

        ploter.figs.append(fig)

        plt.close()

    
    ploter.show()

    return 

def add_new_samples():
    from tkinter import font
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    class Add_Data_GUI(_Propose_Data_GUI):
        """
        该类的使用方法：
        1. 选取需要抽取的文件夹，比如9月份样本
        2. 修改需要添加样本的文件夹，比如训练样本，其中包含"500.mat"等类似的，以索引命名的文件
        3. 修改输出文件夹，比如原文件夹
        4. 每50轮标注后，程序自动保存，添加到输出文件夹中
        
        
        """
        def __init__(self, root: tk.Tk, 
                    extract_root = r'F:\Research\My_Thesis\Data\苏通\VIC\\', 
                    sample_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\data\\', 
                    to_folder = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\\test\\', 
                     ):

            # 设定随机生成图片时，内存中img_lis的长度
            self.img_lis_length = 50
            # 设置失效标签
            self.False_label = 'F'

            # 样本位置、保存样本的文件夹
            self.sample_root = sample_root
            self.to_folder = to_folder

            self.root = root
            self.root.title("Image Viewer")
            self.root.bind('<Return>', self.next_image)

            self.root.bind('<Left>', self.prev_image)
            self.root.bind('<Control-s>', self.save_to_excel)

            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            self.current_image_index = 0
            self.label_lis = []
            self.image_list = []
            self.data_lis = []
            # 图像列表
            self.unshown_fig = []
            # 修改后的索引列表
            self.need_refined_index = []

            # 全选并且放到光标的末尾
            self.entry_text = StringVar()
            self.entry = Entry(root, textvariable=self.entry_text)
            self.entry.grid(row=2, column=1, padx=30, pady=10)
            self.entry.focus_set()
            self.entry.select_range(0, tk.END)

            # 添加一个状态标签
            self.status_label = Label(root, text="", fg="green")
            self.status_label.grid(row=3, column=1, padx=30, pady=10)

            # 添加一个保存信息提醒栏
            self.save_print_label = Label(root, text="", fg="green")
            self.save_print_label.grid(row=0, column=1, padx=30, pady=10)


            Button(root, text="Next", command=self.next_image).grid(row=3, column=2, padx=10, pady=10)
            Button(root, text="Previous", command=self.prev_image).grid(row=3, column=0, padx=10, pady=10)
            # Button(root, text="Save", command=self.save_to_excel).grid(row=0, column=2, sticky="ne")

            # 绑定StringVar的追踪功能，以监控输入变化
            self.entry_text.trace_add('write', self.update_data_and_feedback)

            # 选择抽取的样本库
            self.default_folder = extract_root
            if not self.default_folder:
                self.folder_selected = filedialog.askdirectory(
                    title="请选择需要抽取的样本库",
                    initialdir=self.default_folder or os.path.expanduser("~")
                )
            else:
                self.folder_selected = extract_root
                

            # 随机抽样图片
            self.init_data_path(root = self.folder_selected)

            # 每次调用该方法，会在self.image_list中，生成50张图片
            self.visualize_random_data()

            # Show the first image and set its prefill y
            self.canvas = FigureCanvasTkAgg(figure = self.unshown_fig[self.current_image_index], master = root)
            self.show_image()

            return 


        def show_image(self):
            # 因为canvas与figure强绑定，所以在第二次开始显示图像的时候，就需要销毁canvas并且重新设定了
            if self.current_image_index > 0:
                self.canvas.get_tk_widget().destroy()
                
            # 画图
            # GUI setup
            self.canvas = FigureCanvasTkAgg(figure = self.unshown_fig[self.current_image_index], master = root)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=1, column=1)
            

            # 按照情况填充，之前有值和之前没有值
            if self.current_image_index < len(self.label_lis):
                self.entry_text.set(self.label_lis[self.current_image_index])
            elif self.current_image_index == len(self.label_lis):
                self.entry_text.set('F')

            if (self.current_image_index + 1) % self.img_lis_length == 0:
                self.save_data()


            self.entry.focus_set()


            # 求余还剩10个时更新列表
            # 这一功能是为了保证它不会炸，每次读写的时候图片数量是足够的
            if (self.current_image_index + 1) % self.img_lis_length == int(self.img_lis_length / 3 * 2):
                self.visualize_random_data()

        def save_data(self, 
                    *args):

            sample_saved_root = self.sample_root
            to_folder = self.to_folder

            # 输出保存信息
            print_text = f'\n已达{self.current_image_index}样本数，将数据保存到<{to_folder}>文件夹！'.replace('\\', '/').replace('//', '/')
            self.save_print_label.config(text = print_text)

            start_index = self.current_image_index - self.img_lis_length if self.current_image_index > self.img_lis_length else 0
            end_index = self.current_image_index
            # 取出需要保存的样本
            need_saved_img = self.unshown_fig[start_index : end_index]
            need_saved_data = self.data_lis[start_index : end_index]
            need_saved_label = self.label_lis[start_index: end_index]
            
            # 读取文件名称
            # 第一次读取完成即可，拿到最初的样本index
            if self.current_image_index == self.img_lis_length - 1:
                samples = sorted(os.listdir(sample_saved_root), key = lambda file_name: int(file_name[:-4])) if len(os.listdir(sample_saved_root)) > 1 else ["0.mat"]
                last_sample_name = samples[-1]
                self.start_save_index = int(last_sample_name[:-4])

            # 文件夹不存在，创建文件夹
            if not os.path.exists(to_folder + r'\img\\'):
                os.makedirs(to_folder + r'\img\\')
            
            if not os.path.exists(to_folder + r'\data\\'):
                os.makedirs(to_folder + r'\data\\')
            
            for i in range(len(need_saved_data)):
                save_index = self.start_save_index + start_index + i
                data = need_saved_data[i]
                label = need_saved_label[i]

                # 加一个样本抛弃机制，当用户输入为self.False_label时，则认为该样本无效，不保存
                if label == self.False_label:
                    continue


                fig = need_saved_img[i]

                # 保存图片
                target_pixels = 512
                dpi = 100
                default_figsize = fig.get_size_inches()

                fig.set_size_inches(target_pixels / dpi, target_pixels / dpi)
                fig.savefig(to_folder + r'\img\\' + str(save_index) + '.png')
                fig.set_size_inches(default_figsize)

                # 保存mat文件
                savemat(to_folder + r'\data\\' + str(save_index) + '.mat', mdict = {str(label): data})
            
            # 添加一个修改前文的机制，比如先前有一个值，打上label = 0后，需要修改成label = 1，依赖这个实现
            if len(self.need_refined_index) > 0:
                for index in self.need_refined_index:
                    save_index = self.start_save_index + index

                    need_saved_data = self.data_lis[index]
                    need_saved_label = self.label_lis[index]
                    # 加一个样本抛弃机制，当用户输入为9时，则认为该样本无效，不保存
                    if label == 9:
                        continue
                        
                    fig = need_saved_img[index]

                    # 保存图片
                    target_pixels = 512
                    dpi = 100
                    default_figsize = fig.get_size_inches()

                    fig.set_size_inches(target_pixels / dpi, target_pixels / dpi)
                    fig.savefig(to_folder + r'\img\\' + str(save_index) + '.png')
                    fig.set_size_inches(default_figsize)

                    # 这里修改了前文所保存的mat文件，但没有顺带更新fig
                    savemat(to_folder + r'\data\\' + str(save_index) + '.mat', mdict = {str(label): data})

            return 
        
        
        def update_data_and_feedback(self, *args):
            
            # 更新当前图片对应的用户输入
            new_input = self.entry_text.get()
            if self.current_image_index == len(self.label_lis):
                self.label_lis.append(new_input)
            elif self.current_image_index < len(self.label_lis):
                # 如果用户修改了该值，则更新列表
                if self.label_lis[self.current_image_index] != new_input:
                    self.label_lis[self.current_image_index] = new_input
                    self.need_refined_index.append(self.current_image_index)


            
            # 提供即时反馈
            self.status_label.config(text=f"已获取输入: {new_input}", fg="green")

        def visualize_random_data(self):
            # 抽取self.img_lis_length个样本
            uncheck_paths = self.random_choice()
            ploter = PlotLib()
            need_add_figs = []
            while len(need_add_figs) < 50:
                # 稍微做一下区分，区分普通样本mat文件和VIC文件
                for path in uncheck_paths:
                    match path[-4:]:
                        case '.VIC':
                            data_slice = self.unpacker.File_Detach_Data(path = path, time_interval = 1)
                            if len(data_slice) > 1: 
                                frame = np.random.choice(data_slice.shape[0])
                            else:
                                continue
                            
                            fig, ax = ploter.show_sample(data = data_slice[frame], add_fig = False)

                            # 只是一个占位符
                            need_add_figs.append('fig')

                            # fig_index记得更新
                            self.unshown_fig.append(fig)
                            # data_lis也一样，之后需要读取该列表来保存
                            self.data_lis.append(data_slice[frame])
                            plt.close()
                        
                        case '.mat':
                            label, data = [(key, value) for key, value in loadmat(path).items() if isinstance(value, np.ndarray)][0]

                            fig, ax = ploter.show_sample(data = data, add_fig = False)

                            # 只是一个占位符
                            need_add_figs.append('fig')
                            # fig_index记得更新
                            self.unshown_fig.append(fig)
                            # data_lis也一样，之后需要读取该列表来保存
                            self.data_lis.append(data_slice[frame])
                            plt.close()

            return 
        
        def next_image(self, event=None):
            if self.current_image_index < len(self.unshown_fig) - 1:
                self.current_image_index += 1
                self.show_image()

        def prev_image(self, event=None):
            if self.current_image_index > 0:
                self.current_image_index -= 1

                self.show_image()
                # 该机制可以看到之前的样本标注，标注的是哪一个
                self.entry_text.set(str(self.label_lis[self.current_image_index]))

        def read_datasets_index(self, root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\\'):
            """
            读取该datasets末尾索引值，记录下来，等之后写入的时候再用
            """

            return 
        
        def init_data_path(self, root = None):
            """
            初始化整体抽样数据来源路径
            """
            self.unpacker = UNPACK()
            if root == None:
                self.all_file_paths = self.unpacker.VIC_Paths_from_root
            else:
                self.all_file_paths = self.unpacker.File_Read_Paths(root)
            return 
        
        def random_choice(self):
            """
            随机抽样列表
            """
            uncheck_data_path = np.random.choice(a = self.all_file_paths, size = self.img_lis_length)

            return uncheck_data_path
        
    root = Tk()
    app = Add_Data_GUI(root, 
        extract_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\extract_root\\', 
        sample_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\\' + 'data\\', 
        to_folder = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\test\\', 
    )
    root.mainloop()

    return 

def datasets_description():

    datasets_root = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\data\\' 
    import os
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import numpy as np

    def analyze_mat_files(folder_path):
        """
        分析指定文件夹下所有.mat文件中的标签（keys），统计每种标签的数量并画出饼状图。
        
        :param folder_path: 包含.mat文件的文件夹路径
        """
        label_counts = {}

        def int2label(num):
            if num == 0:
                label = 'Normal Vibration'
            elif num == 1:
                label = 'Transition'
            elif num == 2:
                label = 'VIV'
            return label

        # 遍历文件夹下的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.mat'):
                file_path = os.path.join(folder_path, filename)
                mat_contents = sio.loadmat(file_path)
                
                # 假设每个.mat文件只有一个数据结构，其key即为标签
                for key in mat_contents.keys():
                    if key not in ['__globals__', '__header__', '__version__']:
                        key = int(key)
                        if int2label(key) in label_counts:
                            label_counts[int2label(key)] = label_counts[int2label(key)] + 1
                        else:
                            label_counts[int2label(key)] = 1
        
        # 准备画饼状图的数据
        labels = list(label_counts.keys())
        sizes = list(label_counts.values())

        colors = ['#ff9999','#66b3ff','#99ff99'] 

        # 绘制饼状图
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors = colors)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Sample Nums: {np.sum(sizes)}')
        plt.show()

    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 使用示例
    folder_path = datasets_root  # 替换为你的文件夹路径
    analyze_mat_files(folder_path)

    return 

def IsPeakRight(): 
    from NN.datasets import MyDataset
    import numpy as np
    class ECC_Dataset(MyDataset):
        def __init__(self, data_dir, img_dir):
            super().__init__(data_dir = data_dir, img_dir = img_dir)
        
        def __getitem__(self, index):
            # 获取label
            path = self.paths[index]
            label, data = [(key, value) for key, value in loadmat(path).items() if isinstance(value, np.ndarray)][0]
            output = int(label)

            return data, output
    
    # 仅验证集
    dataset = ECC_Dataset(
        data_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\data\\', 
        img_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\img\\'
    )
    data, label = dataset[0]

    ploter = PlotLib()

    fig, ax = ploter.show_sample(data[0, :], nperseg = 2048)
    mount = Cal_Mount()
    fs = 50
    nperseg = 2048

    fx, pxxden = signal.welch(data[0, :], fs = fs, nfft = 65536, 
                                nperseg = nperseg, noverlap = 1)
    
    fxtopk, pxxtopk, fxtopk_intervals, pxxtopk_intervals = mount.peaks(fx, pxxden, return_intervals = True)

    for i in range(len(fxtopk_intervals)):

        fxi = fxtopk_intervals[i]
        pxxi = pxxtopk_intervals[i]

        fig, ax = ploter.show_sample(data[0, :], nperseg = 2048)
        ax[1].plot(fxi, pxxi, 'pink')
        plt.close()

    ploter.show()


    # ok，完成峰值定义，详情见PPT
    # ==================== 下一步：完成ECC计算和MECC计算 =====================
    return 

def ECC_description():
    from NN.datasets import MyDataset
    import numpy as np
    class ECC_Dataset(MyDataset):
        def __init__(self, data_dir, img_dir):
            super().__init__(data_dir = data_dir, img_dir = img_dir)
        
        def __getitem__(self, index):
            # 获取label
            path = self.paths[index]
            label, data = [(key, value) for key, value in loadmat(path).items() if isinstance(value, np.ndarray)][0]
            output = int(label)

            return data, output
    
    # 仅验证集
    dataset = ECC_Dataset(
        data_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\data\\', 
        img_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\img\\'
    )

    True_Label = []
    Predicted_Label = []
    labels = ['Normal Vibration', 'Transition', 'VIV']

    def int2label(num):
        if num == 0:
            label = 'Normal Vibration'
        elif num == 1:
            label = 'Transition'
        elif num == 2:
            label = 'VIV'
        return label

    ploter = PlotLib()
    VIV_lis = []
    mount = Cal_Mount()
    from tqdm import tqdm
    
    # 绘制混淆矩阵
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    with tqdm(total = len(dataset)) as pbar:
        for i, (data, label) in enumerate(dataset):

            fx, pxxden = signal.welch(data[0, :], fs = 50, nfft = 65536, 
                                    nperseg = 2048, noverlap = 1)

            fxtopk, pxxtopk, fxtopk_intervals, pxxtopk_intervals = mount.peaks(fx, pxxden, return_intervals = True)
            # 降序
            e2_e1 = pxxtopk[1] / pxxtopk[0]
            True_Label.append(int2label(label))

            # 普通ECC
            # 判定，第二峰值/第一峰值小于0.1，此时为VIV、label为2
            if e2_e1 < 0.1:
                predict_int = 2
                fig, ax = ploter.show_sample(data[0, :], nperseg = 2048)
                plt.close()
            else:
                predict_int = 0

            Predicted_Label.append(int2label(predict_int))

            pbar.update(1)

            

    from sklearn.metrics import confusion_matrix
    def plot_confusion_matrix(y_true, y_pred, labels, normalize=False):
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            # 归一化处理
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("显示归一化的混淆矩阵")
        else:
            print('显示未归一化的混淆矩阵')

        # 绘制混淆矩阵
        fig, ax = plt.subplots()
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Pastel2', xticklabels=labels, yticklabels=labels)
        ax.set_title('ECC Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        return fig, ax 

    
    fig, ax = plot_confusion_matrix(True_Label, Predicted_Label, labels = labels, normalize = True)
    ploter.figs.append(fig)

    # 绘制VIV真值结果
    for img_array in VIV_lis:
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(img_array, (1, 2, 0)))

        ax.grid(False)
        ax.axis('off')

        ploter.figs.append(fig)
        plt.close()


    ploter.show()

    return 

def MECC_description(k = 3):
    from NN.datasets import MyDataset
    import numpy as np
    class ECC_Dataset(MyDataset):
        def __init__(self, data_dir, img_dir):
            super().__init__(data_dir = data_dir, img_dir = img_dir)
        
        def __getitem__(self, index):
            # 获取label
            path = self.paths[index]
            label, data = [(key, value) for key, value in loadmat(path).items() if isinstance(value, np.ndarray)][0]
            output = int(label)

            return data, output
    
    # 仅验证集
    dataset = ECC_Dataset(
        data_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\data\\', 
        img_dir = r'F:\Research\My_Thesis\Vibration Characteristics In Cable Vibration\NN\datasets\dev\img\\'
    )

    True_Label = []
    Predicted_Label = []
    labels = ['Normal Vibration', 'Transition', 'VIV']

    def int2label(num):
        if num == 0:
            label = 'Normal Vibration'
        elif num == 1:
            label = 'Transition'
        elif num == 2:
            label = 'VIV'
        return label

    mount = Cal_Mount()
    base_freq = mount.multi_nums

    def MECC(fxtopk, pxxtopk, k = 3, base_freq = base_freq):
        """
        判定一个频谱是否为VIV
        k控制topk判别效果
        """
        conditions = []
        multi_nums = (k - 1) / 2
        for i in range(1, k):
            conditions.append(fxtopk[i] < fxtopk[0] + multi_nums * base_freq and fxtopk[i] > fxtopk[0] - multi_nums * base_freq)
        
        conditions.append(pxxtopk[k] / pxxtopk[0] < 0.2)
        if all(conditions):
            return True
        else:
            return False



    VIV_lis = []
    ploter = PlotLib()
    justifier = Abnormal_Vibration_Filter()
    # 绘制混淆矩阵
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    with tqdm(total = len(dataset)) as pbar:
        for i, (data, label) in enumerate(dataset):

            fx, pxxden = signal.welch(data[0, :], fs = 50, nfft = 65536, 
                                    nperseg = 2048, noverlap = 1)

            fxtopk, pxxtopk, fxtopk_intervals, pxxtopk_intervals = mount.peaks(fx, pxxden, return_intervals = True)
            # 降序
            e2_e1 = pxxtopk[1] / pxxtopk[0]
            True_Label.append(int2label(label))

            if justifier.VIV_Filter(data = data[0, :], f0 = mount.multi_nums, f0times = 11):
                predict_int = 2
                fig, ax = ploter.show_sample(data[0, :], nperseg = 2048)
                plt.close()
            else:
                predict_int = 0

            Predicted_Label.append(int2label(predict_int))

            pbar.update(1)
    

            
    # 绘制混淆矩阵
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    def plot_confusion_matrix(y_true, y_pred, labels, normalize=False):
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if normalize:
            # 归一化处理
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("显示归一化的混淆矩阵")
        else:
            print('显示未归一化的混淆矩阵')

        # 绘制混淆矩阵
        fig, ax = plt.subplots()
        fmt = '.2f' if normalize else 'd'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Pastel2', xticklabels=labels, yticklabels=labels)
        ax.set_title('MECC Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        return fig, ax 

    
    fig, ax = plot_confusion_matrix(True_Label, Predicted_Label, labels = labels, normalize = True)
    ploter.figs.append(fig)

    # 绘制VIV真值结果
    for img_array in VIV_lis:
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(img_array, (1, 2, 0)))

        ax.grid(False)
        ax.axis('off')

        ploter.figs.append(fig)
        plt.close()


    ploter.show()
    return 



if __name__ == '__main__':
    MECC_description()