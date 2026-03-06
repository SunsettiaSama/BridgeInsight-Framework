import os
import sys
import json
import tkinter as tk
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, Frame, Toplevel, Radiobutton, FLAT, StringVar, filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from typing import Optional, Dict, List, Tuple
from collections import OrderedDict
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tkinter import Tk as TK_Root
from src.data_processer.io_unpacker import UNPACK
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow
from src.figs.figs_for_thesis.config import get_viridis_color_map

# ==================== 全局绘图配置 ====================
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FONT_SIZE = 11
LABEL_FONT_SIZE = 12
ENG_FONT = plt.matplotlib.font_manager.FontProperties(family='DejaVu Sans', size=LABEL_FONT_SIZE)
CN_FONT = plt.matplotlib.font_manager.FontProperties(family='SimHei', size=LABEL_FONT_SIZE)

FIG_SIZE = (12, 8)
NFFT = 2048
FS = 50.0
WINDOW_SIZE = 3000
FIGURE_CACHE_SIZE = 20  # 缓存最多20张图像

# ==================== 保存路径常量 ====================
DEFAULT_ANNOTATION_RESULT_PATH = os.path.join(
    os.path.dirname(__file__), 
    "../../annotation_results/annotation_results.json"
)


# ==================== 图像缓冲器 ====================
class FigureCache:
    """LRU 图像缓冲器，防止内存占用过大"""
    
    def __init__(self, max_size: int = FIGURE_CACHE_SIZE):
        self.max_size = max_size
        self.cache = OrderedDict()  # 维护LRU顺序
    
    def get(self, window_index: int) -> Optional[plt.Figure]:
        """获取缓存的图像"""
        if window_index in self.cache:
            # 移到最后（最近使用）
            self.cache.move_to_end(window_index)
            return self.cache[window_index]
        return None
    
    def put(self, window_index: int, fig: plt.Figure):
        """存储图像到缓冲"""
        if window_index in self.cache:
            self.cache.move_to_end(window_index)
        else:
            self.cache[window_index] = fig
            if len(self.cache) > self.max_size:
                # 删除最旧的（最不常用）图像
                oldest_idx, oldest_fig = self.cache.popitem(last=False)
                plt.close(oldest_fig)
    
    def clear(self):
        """清空缓冲"""
        for fig in self.cache.values():
            plt.close(fig)
        self.cache.clear()


# ==================== 数据提供者 ====================
class AnnotationDataProvider:
    """
    负责从工作流获取metadata，提取极端窗口数据
    """
    
    MODE_NORMAL = 'normal'
    MODE_EXTREME = 'extreme'
    MODE_SUPER_EXTREME = 'super_extreme'

    def __init__(self, use_cache: bool = True, force_recompute: bool = False, mode: str = MODE_EXTREME):
        self.use_cache = use_cache
        self.force_recompute = force_recompute
        self.unpacker = UNPACK(init_path=False)
        self.extreme_windows = []
        self.mode = mode
        
    def fetch_metadata_and_extreme_windows(self) -> List[Dict]:
        """根据筛选模式获取数据"""
        print(f"[步骤1] 获取metadata (模式: {self.mode})...")
        metadata = run_vib_workflow(use_cache=self.use_cache, force_recompute=self.force_recompute)
        
        if not metadata:
            raise ValueError("无法获取 metadata，工作流可能失败")
        
        print(f"✓ 获取到 {len(metadata)} 条 metadata 记录")
        
        if self.mode == self.MODE_NORMAL:
            windows = self._process_normal_mode(metadata)
        elif self.mode == self.MODE_EXTREME:
            windows = self._process_extreme_mode(metadata)
        elif self.mode == self.MODE_SUPER_EXTREME:
            windows = self._process_super_extreme_mode(metadata)
        else:
            raise ValueError(f"未知的筛选模式: {self.mode}")
        
        print(f"✓ 成功加载 {len(windows)} 个窗口")
        
        self.extreme_windows = windows
        return windows
    
    def _process_normal_mode(self, metadata: List[Dict]) -> List[Dict]:
        """正常模式：加载所有记录的所有数据"""
        print("\n[步骤2] 加载正常模式数据（所有窗口）...")
        all_windows = []
        
        for i, record in enumerate(metadata):
            file_path = record['file_path']
            sensor_id = record['sensor_id']
            time_str = f"{record['month']}/{record['day']} {record['hour']}:00"
            
            try:
                vibration_data = self.unpacker.VIC_DATA_Unpack(file_path)
                vibration_data = np.array(vibration_data)
                
                num_windows = len(vibration_data) // WINDOW_SIZE
                for window_idx in range(num_windows):
                    start_sample = window_idx * WINDOW_SIZE
                    end_sample = (window_idx + 1) * WINDOW_SIZE
                    
                    window_data = vibration_data[start_sample:end_sample]
                    window_info = {
                        'metadata': record,
                        'data': window_data,
                        'window_index': window_idx,
                        'sensor_id': sensor_id,
                        'time': time_str,
                        'file_path': file_path,
                        'mode': self.MODE_NORMAL
                    }
                    all_windows.append(window_info)
            except Exception as e:
                print(f"  ⚠ 加载失败 {sensor_id} {time_str}: {e}")
        
        return all_windows
    
    def _process_extreme_mode(self, metadata: List[Dict]) -> List[Dict]:
        """极端模式：只加载极端窗口的数据"""
        extreme_records = [m for m in metadata if len(m.get('extreme_rms_indices', [])) > 0]
        print(f"✓ 其中包含极端窗口的记录：{len(extreme_records)} 条")
        
        if not extreme_records:
            raise ValueError("无包含极端窗口的记录")
        
        print("\n[步骤2] 加载极端窗口数据...")
        return self._load_extreme_windows(extreme_records)
    
    def _process_super_extreme_mode(self, metadata: List[Dict]) -> List[Dict]:
        """超级极端模式：使用0.25%分位数据（待实现）"""
        print("\n⚠ 超级极端模式正在开发中，暂时使用极端模式")
        return self._process_extreme_mode(metadata)
    
    def _load_extreme_windows(self, extreme_records: List[Dict]) -> List[Dict]:
        """从极端记录中加载极端窗口数据"""
        all_extreme_windows = []
        
        for i, record in enumerate(extreme_records):
            file_path = record['file_path']
            extreme_indices = record['extreme_rms_indices']
            sensor_id = record['sensor_id']
            time_str = f"{record['month']}/{record['day']} {record['hour']}:00"
            
            try:
                vibration_data = self.unpacker.VIC_DATA_Unpack(file_path)
                vibration_data = np.array(vibration_data)
                
                for window_idx in extreme_indices:
                    start_sample = window_idx * WINDOW_SIZE
                    end_sample = (window_idx + 1) * WINDOW_SIZE
                    
                    if end_sample <= len(vibration_data):
                        window_data = vibration_data[start_sample:end_sample]
                        window_info = {
                            'metadata': record,
                            'data': window_data,
                            'window_index': window_idx,
                            'sensor_id': sensor_id,
                            'time': time_str,
                            'file_path': file_path,
                            'mode': self.MODE_EXTREME
                        }
                        all_extreme_windows.append(window_info)
            except Exception as e:
                print(f"  ⚠ 加载失败 {sensor_id} {time_str}: {e}")
        
        return all_extreme_windows
    
    def get_extreme_windows(self) -> List[Dict]:
        """获取已加载的极端窗口列表"""
        return self.extreme_windows
    
    def get_window_by_index(self, index: int) -> Optional[Dict]:
        """获取指定索引的极端窗口"""
        if 0 <= index < len(self.extreme_windows):
            return self.extreme_windows[index]
        return None


# ==================== 图像生成器 ====================
class AnnotationFigureGenerator:
    """负责生成标注用的时域和频域上下子图"""
    
    def __init__(self, fs: float = 50.0):
        self.fs = fs
    
    def generate_figure(self, window_info: Dict) -> Tuple[Optional[plt.Figure], Optional[str]]:
        """生成单个窗口的图像：左侧时域+频域，右侧频域变化"""
        try:
            data = window_info['data']
            sensor_id = window_info['sensor_id']
            time_str = window_info['time']
            window_idx = window_info['window_index']
            
            if len(data) == 0:
                return None, "窗口数据为空"
            
            fig = plt.figure(figsize=(18, 8))
            
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], wspace=0.3)
            ax_time = fig.add_subplot(gs[0, 0])
            ax_freq = fig.add_subplot(gs[1, 0])
            ax_freq_change = fig.add_subplot(gs[:, 1])
            
            self._plot_time_domain(ax_time, data, sensor_id, time_str, window_idx)
            self._plot_frequency_domain(ax_freq, data, sensor_id, time_str, window_idx)
            self._plot_frequency_evolution(ax_freq_change, data, sensor_id, time_str, window_idx)
            
            plt.tight_layout()
            
            return fig, None
            
        except Exception as e:
            return None, f"生成图像失败: {str(e)}"
    
    def _plot_time_domain(self, ax, data: np.ndarray, sensor_id: str, 
                          time_str: str, window_idx: int):
        """绘制时域波形图"""
        time_axis = np.arange(len(data)) / self.fs
        
        ax.plot(
            time_axis, 
            data, 
            color='#333333',
            linewidth=1.0
        )
        
        title = f"{sensor_id} @ {time_str} (窗口 {window_idx}) - 时域波形"
        ax.set_title(title, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        
        ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        
        ax.grid(True, color='gray', alpha=0.4, linewidth=0.5, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    
    def _plot_frequency_domain(self, ax, data: np.ndarray, sensor_id: str, 
                               time_str: str, window_idx: int):
        """绘制频域谱图"""
        f, psd = signal.welch(data, fs=self.fs, nperseg=int(NFFT/2), 
                              noverlap=int(NFFT/4), nfft=NFFT)
        
        freq_limit = 25
        mask = f <= freq_limit
        f_limited = f[mask]
        psd_limited = psd[mask]
        
        ax.plot(
            f_limited,
            psd_limited,
            color='#333333',
            linewidth=1.0
        )
        
        title = f"{sensor_id} @ {time_str} (窗口 {window_idx}) - 频域谱"
        ax.set_title(title, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        
        ax.set_xlabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('PSD $(m/s^2)^2/Hz$', labelpad=10, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        
        ax.grid(True, color='gray', alpha=0.4, linewidth=0.5, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        
        ax.set_xlim(0, freq_limit)
    
    def _plot_frequency_evolution(self, ax, data: np.ndarray, sensor_id: str, 
                                  time_str: str, window_idx: int):
        """绘制频域随时间变化的频带堆积图"""
        freq_limit = 25
        fs_int = int(self.fs)
        window_total = int(len(data) // fs_int)
        
        psd_list = []
        time_labels = []
        
        for i in range(window_total):
            start_idx = i * fs_int
            end_idx = (i + 1) * fs_int
            
            if end_idx <= len(data):
                segment = data[start_idx:end_idx]
                f, psd = signal.welch(segment, fs=self.fs, nperseg=int(self.fs * 0.8),
                                     noverlap=int(self.fs * 0.4), nfft=NFFT)
                
                mask = f <= freq_limit
                psd_limited = psd[mask]
                psd_list.append(psd_limited)
                time_labels.append(f"{i}s")
        
        if not psd_list:
            return
        
        spec_array = np.array(psd_list)
        
        f_limited = f[f <= freq_limit]
        
        cmap_gray = get_viridis_color_map(start_gray=0.2)

        im = ax.imshow(spec_array, aspect='auto', origin='lower', cmap=cmap_gray, 
                       extent=[0, freq_limit, 0, window_total],
                       interpolation='bilinear')
        
        title = f"{sensor_id} @ {time_str} (窗口 {window_idx}) - 频域演变"
        ax.set_title(title, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        
        ax.set_xlabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        
        ax.set_yticks(np.arange(0, window_total + 1, max(1, window_total // 10)))
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('功率谱密度 $(m/s^2)^2/Hz$', fontproperties=CN_FONT, fontsize=LABEL_FONT_SIZE)
        cbar.ax.tick_params(labelsize=FONT_SIZE)


# ==================== 用户界面 ====================
class AnnotationWindowGUI:
    """基于工作流数据的人工标注GUI"""
    
    def __init__(self, root, save_result_path: str = None):
        self.root = root
        self.root.title("振动数据标注系统")
        self.root.geometry("1800x1000")
        
        self.root.bind('<Return>', self.next_window)
        self.root.bind('<Right>', self.next_window)
        self.root.bind('<Left>', self.prev_window)
        self.root.bind('<Control-s>', self.save_results)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_window_index = 0
        self.annotation_data = {}
        self.extreme_windows = []
        self.filtered_indices = []
        
        self.save_result_path = save_result_path or DEFAULT_ANNOTATION_RESULT_PATH
        self.selected_mode = None
        self.rms_threshold = None
        self.amplitude_threshold = None
        self.date_start = None
        self.date_end = None
        self.sensor_ids = []
        
        self.data_provider = None
        self.figure_generator = AnnotationFigureGenerator(fs=FS)
        self.figure_cache = FigureCache(max_size=FIGURE_CACHE_SIZE)
        
        self._init_ui()
        
        self.root.after(100, self._show_mode_selection_dialog)
    
    def _init_ui(self):
        """初始化用户界面"""
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        top_frame = Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = Label(top_frame, text="加载中...", fg="blue", font=("Arial", 11, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        Button(top_frame, text="选择保存位置", command=self._select_save_path).pack(side=tk.RIGHT, padx=2)
        Button(top_frame, text="保存结果", command=self.save_results).pack(side=tk.RIGHT, padx=2)
        Button(top_frame, text="关闭", command=self.on_closing).pack(side=tk.RIGHT, padx=2)
        
        self.save_path_label = Label(top_frame, text="", fg="gray", font=("Arial", 9))
        self.save_path_label.pack(side=tk.RIGHT, padx=10)
        
        self.canvas_frame = Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        bottom_frame = Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)
        
        input_frame = Frame(bottom_frame)
        input_frame.pack(fill=tk.X, padx=2, pady=2)
        
        Label(input_frame, text="标注信息:", font=("Arial", 9), width=10).pack(side=tk.LEFT)
        
        self.entry_text = StringVar()
        self.entry = Entry(input_frame, textvariable=self.entry_text, font=("Arial", 9))
        self.entry.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        Button(input_frame, text="◀ (←)", command=self.prev_window, width=4).pack(side=tk.LEFT, padx=1)
        Button(input_frame, text="▶ (→)", command=self.next_window, width=4).pack(side=tk.LEFT, padx=1)
        
        self.entry_text.trace_add('write', self._on_annotation_changed)
        
        self._update_save_path_label()
    
    def _show_mode_selection_dialog(self):
        """显示模式选择对话框（作为主窗口的子窗口）"""
        dialog = Toplevel(self.root)
        dialog.title("选择筛选模式、日期范围和阈值")
        dialog.geometry("500x550")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 创建可滚动的框架
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        title_label = Label(scrollable_frame, text="请选择数据筛选模式", font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        mode_var = StringVar(value=AnnotationDataProvider.MODE_EXTREME)
        
        # 模式选择
        normal_frame = Frame(scrollable_frame, relief=FLAT, borderwidth=1)
        normal_frame.pack(fill=tk.X, padx=15, pady=5)
        Radiobutton(normal_frame, text="正常模式 - 加载所有60秒窗口", 
                      variable=mode_var, value=AnnotationDataProvider.MODE_NORMAL).pack(anchor=tk.W)
        Label(normal_frame, text="不做任何数据切分或处理", fg="gray", font=("Arial", 8)).pack(anchor=tk.W, padx=20)
        
        extreme_frame = Frame(scrollable_frame, relief=FLAT, borderwidth=1)
        extreme_frame.pack(fill=tk.X, padx=15, pady=5)
        Radiobutton(extreme_frame, text="极端模式 - 只加载极端窗口（推荐）", 
                      variable=mode_var, value=AnnotationDataProvider.MODE_EXTREME).pack(anchor=tk.W)
        Label(extreme_frame, text="使用极端振动窗口的结果", fg="gray", font=("Arial", 8)).pack(anchor=tk.W, padx=20)
        
        super_frame = Frame(scrollable_frame, relief=FLAT, borderwidth=1)
        super_frame.pack(fill=tk.X, padx=15, pady=5)
        Radiobutton(super_frame, text="超级极端模式 - 0.25%分位数据", 
                      variable=mode_var, value=AnnotationDataProvider.MODE_SUPER_EXTREME).pack(anchor=tk.W)
        Label(super_frame, text="待0.25%分位数据实现后启用", fg="gray", font=("Arial", 8)).pack(anchor=tk.W, padx=20)
        
        # 日期范围筛选
        Label(scrollable_frame, text="日期范围筛选（格式: MM/DD，留空则不筛选）", 
              font=("Arial", 10, "bold")).pack(pady=8, padx=15)
        
        date_frame = Frame(scrollable_frame)
        date_frame.pack(fill=tk.X, padx=15, pady=5)
        Label(date_frame, text="起始日期 (MM/DD):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        date_start_var = StringVar()
        Entry(date_frame, textvariable=date_start_var, width=12).pack(side=tk.LEFT, padx=5)
        Label(date_frame, text="结束日期 (MM/DD):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        date_end_var = StringVar()
        Entry(date_frame, textvariable=date_end_var, width=12).pack(side=tk.LEFT, padx=5)
        
        # 传感器筛选
        Label(scrollable_frame, text="传感器筛选（留空则不筛选）", 
              font=("Arial", 10, "bold")).pack(pady=8, padx=15)
        
        sensor_info = Label(scrollable_frame, 
              text="单个: ST-VIC-C18-101-01 | 多个用逗号分隔: ST-VIC-C18-101-01, ST-VIC-C18-102-01", 
              fg="gray", font=("Arial", 8), justify=tk.LEFT, wraplength=400)
        sensor_info.pack(pady=3, padx=15)
        
        sensor_frame = Frame(scrollable_frame)
        sensor_frame.pack(fill=tk.X, padx=15, pady=5)
        Label(sensor_frame, text="传感器ID:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        sensor_var = StringVar()
        Entry(sensor_frame, textvariable=sensor_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 阈值过滤
        Label(scrollable_frame, text="选择性阈值过滤（留空表示不使用）", 
              font=("Arial", 10, "bold")).pack(pady=8, padx=15)
        
        rms_frame = Frame(scrollable_frame)
        rms_frame.pack(fill=tk.X, padx=15, pady=5)
        Label(rms_frame, text="RMS阈值 (m/s²):", width=18, anchor=tk.W).pack(side=tk.LEFT)
        rms_var = StringVar()
        Entry(rms_frame, textvariable=rms_var, width=12).pack(side=tk.LEFT, padx=5)
        Label(rms_frame, text="小于此值的样本将被忽略", fg="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)
        
        amp_frame = Frame(scrollable_frame)
        amp_frame.pack(fill=tk.X, padx=15, pady=5)
        Label(amp_frame, text="最大振幅阈值 (m/s²):", width=18, anchor=tk.W).pack(side=tk.LEFT)
        amp_var = StringVar()
        Entry(amp_frame, textvariable=amp_var, width=12).pack(side=tk.LEFT, padx=5)
        Label(amp_frame, text="小于此值的样本将被忽略", fg="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)
        
        button_frame = Frame(scrollable_frame)
        button_frame.pack(pady=15)
        
        def on_confirm():
            self.selected_mode = mode_var.get()
            
            # 解析日期
            date_start_str = date_start_var.get().strip()
            date_end_str = date_end_var.get().strip()
            
            if date_start_str:
                try:
                    self.date_start = datetime.strptime(date_start_str, "%m/%d")
                except ValueError:
                    messagebox.showerror("输入错误", "起始日期格式错误，应为 MM/DD")
                    return
            
            if date_end_str:
                try:
                    self.date_end = datetime.strptime(date_end_str, "%m/%d")
                except ValueError:
                    messagebox.showerror("输入错误", "结束日期格式错误，应为 MM/DD")
                    return
            
            # 解析传感器ID
            sensor_str = sensor_var.get().strip()
            if sensor_str:
                self.sensor_ids = [s.strip() for s in sensor_str.split(',')]
                print(f"✓ 筛选传感器: {self.sensor_ids}")
            
            try:
                rms_str = rms_var.get().strip()
                self.rms_threshold = float(rms_str) if rms_str else None
            except ValueError:
                messagebox.showerror("输入错误", "RMS阈值必须是数字或留空")
                return
            
            try:
                amp_str = amp_var.get().strip()
                self.amplitude_threshold = float(amp_str) if amp_str else None
            except ValueError:
                messagebox.showerror("输入错误", "最大振幅阈值必须是数字或留空")
                return
            
            dialog.destroy()
            self._load_data()
        
        def on_cancel():
            dialog.destroy()
            self.on_closing()
        
        Button(button_frame, text="确认", command=on_confirm, width=10).pack(side=tk.LEFT, padx=5)
        Button(button_frame, text="取消", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _select_save_path(self):
        """打开文件对话框选择保存位置"""
        file_path = filedialog.asksaveasfilename(
            title="选择保存位置",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialfile="annotation_results.json"
        )
        
        if file_path:
            self.save_result_path = file_path
            self._update_save_path_label()
            messagebox.showinfo("成功", f"保存位置已更新:\n{file_path}")
    
    def _update_save_path_label(self):
        """更新保存路径标签"""
        if hasattr(self, 'save_path_label'):
            display_path = self.save_result_path
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            self.save_path_label.config(text=f"保存位置: {display_path}")
    
    def _load_data(self):
        """加载数据"""
        try:
            self.status_label.config(text="正在加载数据...", fg="blue")
            self.root.update()
            
            self.data_provider = AnnotationDataProvider(use_cache=True, mode=self.selected_mode)
            self.extreme_windows = self.data_provider.fetch_metadata_and_extreme_windows()
            
            if not self.extreme_windows:
                messagebox.showerror("错误", "没有找到任何窗口数据")
                self.on_closing()
                return
            
            self.status_label.config(text="正在应用阈值检查...", fg="blue")
            self.root.update()
            
            self._build_filtered_indices()
            
            self._load_existing_annotations()
            self.show_window()
            
        except Exception as e:
            messagebox.showerror("加载失败", f"加载数据失败: {str(e)}")
            self.on_closing()
    
    def _load_existing_annotations(self):
        """加载已有的标注结果"""
        if os.path.exists(self.save_result_path):
            try:
                with open(self.save_result_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    for item in results:
                        key = f"{item['file_path']}_{item['window_index']}"
                        annotation = item.get('annotation', '').strip()
                        # 只加载非空的标注
                        if annotation:
                            self.annotation_data[key] = annotation
                print(f"✓ 加载了 {len(self.annotation_data)} 条已有标注")
            except Exception as e:
                print(f"⚠ 加载已有标注失败: {e}")
    
    def _build_filtered_indices(self):
        """根据阈值、日期和传感器条件构建通过检查的窗口索引列表"""
        self.filtered_indices = []
        
        for i, window_info in enumerate(self.extreme_windows):
            if (self._check_window_threshold(window_info) and 
                self._check_window_date(window_info) and
                self._check_window_sensor(window_info)):
                self.filtered_indices.append(i)
        
        print(f"✓ 阈值检查完成: {len(self.filtered_indices)}/{len(self.extreme_windows)} 个窗口通过检查")
        
        if not self.filtered_indices:
            raise ValueError("没有任何窗口通过阈值、日期和传感器检查")
    
    def _check_window_threshold(self, window_info: Dict) -> bool:
        """检查窗口是否通过阈值检查"""
        data = window_info['data']
        
        if self.rms_threshold is not None:
            rms_value = np.sqrt(np.mean(np.square(data)))
            if rms_value < self.rms_threshold:
                return False
        
        if self.amplitude_threshold is not None:
            max_amplitude = np.max(np.abs(data))
            if max_amplitude < self.amplitude_threshold:
                return False
        
        return True
    
    def _check_window_date(self, window_info: Dict) -> bool:
        """检查窗口是否在指定日期范围内"""
        if self.date_start is None and self.date_end is None:
            return True
        
        # 从time字符串解析日期 (格式: "M/D HH:MM")
        try:
            time_str = window_info['time']
            month_day = time_str.split()[0]  # 获取 "M/D" 部分
            window_date = datetime.strptime(month_day, "%m/%d")
        except:
            return True
        
        # 检查日期范围
        if self.date_start is not None and window_date < self.date_start:
            return False
        
        if self.date_end is not None and window_date > self.date_end:
            return False
        
        return True
    
    def _check_window_sensor(self, window_info: Dict) -> bool:
        """检查窗口是否来自指定的传感器"""
        # 如果没有指定传感器，则所有传感器都通过
        if not self.sensor_ids:
            return True
        
        # 检查当前窗口的传感器ID是否在指定列表中
        sensor_id = window_info['sensor_id']
        return sensor_id in self.sensor_ids
    
    def show_window(self):
        """显示当前窗口（跳过不符合阈值的窗口）"""
        if not self.filtered_indices:
            return
        
        if self.current_window_index >= len(self.filtered_indices):
            self.current_window_index = len(self.filtered_indices) - 1
        
        actual_window_index = self.filtered_indices[self.current_window_index]
        window_info = self.extreme_windows[actual_window_index]
        
        # 先检查缓冲中是否有该图像
        fig = self.figure_cache.get(actual_window_index)
        
        if fig is None:
            # 如果缓冲中没有，生成新的图像
            fig, error_msg = self.figure_generator.generate_figure(window_info)
            
            if fig is None:
                self.status_label.config(text=f"图像生成失败: {error_msg}", fg="red")
                return
            
            # 存入缓冲
            self.figure_cache.put(actual_window_index, fig)
        
        # 清除旧的canvas
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # 显示图像
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        window_key = self._get_window_key(window_info)
        current_annotation = self.annotation_data.get(window_key, "")
        
        status_text = f"窗口 {self.current_window_index + 1}/{len(self.filtered_indices)} " \
                     f"(总计{len(self.extreme_windows)}) - {window_info['sensor_id']} @ {window_info['time']}"
        self.status_label.config(text=status_text, fg="green")
        
        self.entry_text.set(current_annotation)
        self.entry.focus_set()
    
    def _on_annotation_changed(self, *args):
        """当标注内容改变时调用"""
        if self.filtered_indices and self.current_window_index < len(self.filtered_indices):
            actual_window_index = self.filtered_indices[self.current_window_index]
            window_info = self.extreme_windows[actual_window_index]
            window_key = self._get_window_key(window_info)
            annotation = self.entry_text.get()
            self.annotation_data[window_key] = annotation
    
    def next_window(self, event=None):
        """显示下一个窗口"""
        if self.current_window_index < len(self.filtered_indices) - 1:
            self.current_window_index += 1
            self.show_window()
    
    def prev_window(self, event=None):
        """显示上一个窗口"""
        if self.current_window_index > 0:
            self.current_window_index -= 1
            self.show_window()
    
    def save_results(self, event=None):
        """保存标注结果，支持追加和覆盖"""
        try:
            # 确保保存路径的目录存在
            save_dir = os.path.dirname(self.save_result_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 1. 检查文件是否存在，加载已有数据
            existing_data = {}
            if os.path.exists(self.save_result_path):
                try:
                    with open(self.save_result_path, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                        # 将现有数据转为字典便于快速查找
                        existing_data = {
                            f"{item['file_path']}_{item['window_index']}": item 
                            for item in existing_results
                        }
                    print(f"✓ 加载了 {len(existing_data)} 条已有标注")
                except Exception as e:
                    print(f"⚠ 加载现有文件失败: {e}")
            
            # 2. 构建新的结果列表，只保存有意义的标注
            results = []
            new_count = 0
            updated_count = 0
            
            for window_info in self.extreme_windows:
                window_key = self._get_window_key(window_info)
                annotation = self.annotation_data.get(window_key, "").strip()
                
                # 只保存有明确意义的结果（非空且非空字符串）
                if annotation:
                    result_item = {
                        'metadata': window_info['metadata'],
                        'window_index': window_info['window_index'],
                        'sensor_id': window_info['sensor_id'],
                        'time': window_info['time'],
                        'file_path': window_info['file_path'],
                        'annotation': annotation
                    }
                    
                    if window_key in existing_data:
                        updated_count += 1
                    else:
                        new_count += 1
                    
                    results.append(result_item)
            
            # 3. 合并新旧数据（保留已有的数据）
            for key, item in existing_data.items():
                # 检查是否在新数据中
                if not any(f"{r['file_path']}_{r['window_index']}" == key for r in results):
                    results.append(item)
            
            # 4. 按照 file_path 和 window_index 排序，便于查看
            results.sort(key=lambda x: (x['file_path'], x['window_index']))
            
            # 5. 保存到文件
            with open(self.save_result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 6. 显示统计信息
            total_saved = len(results)
            messagebox.showinfo("保存成功", 
                              f"标注结果已保存\n"
                              f"新增标注: {new_count} 个\n"
                              f"更新标注: {updated_count} 个\n"
                              f"总保存数: {total_saved} 个\n"
                              f"保存路径: {self.save_result_path}")
            
            print(f"✓ 保存完成: 新增{new_count}个，更新{updated_count}个，总计{total_saved}个")
            
        except Exception as e:
            messagebox.showerror("保存失败", f"保存结果失败: {str(e)}")
            import traceback
            print(f"❌ 保存错误: {traceback.format_exc()}")
    
    def on_closing(self):
        """关闭窗口前的处理"""
        # 清理图像缓冲
        self.figure_cache.clear()
        
        if messagebox.askokcancel("确认退出", "确定要退出吗？未保存的标注将会丢失。"):
            self.root.destroy()
    
    @staticmethod
    def _get_window_key(window_info: Dict) -> str:
        """获取窗口的唯一标识符"""
        return f"{window_info['file_path']}_{window_info['window_index']}"


# ==================== 入口接口 ====================
class AnnotationGUI:
    """标注系统的入口类"""
    
    def __init__(self):
        pass
    
    def run(self, save_result_path: str = None):
        """启动标注GUI"""
        root = TK_Root()
        app = AnnotationWindowGUI(root, save_result_path)
        root.mainloop()
    
    def __call__(self, *args, **kwargs):
        self.run(**kwargs)


Data_Process = AnnotationGUI
