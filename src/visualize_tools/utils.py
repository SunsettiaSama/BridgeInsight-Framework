import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import re
import pandas as pd
import matplotlib.animation as animation
from tkinter import Tk, Label, Button, Entry, StringVar, filedialog, messagebox
from PIL import Image, ImageTk

GLOBAL_FONT_SIZE = 10
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = GLOBAL_FONT_SIZE



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

    def rose_hist(self, x, 
                  y_datas = None, 
                  title = 'Wind Direction Distribution', 
                  separate_bins = 20,
                  density = True,
                  color = None, 
                  fig = None, 
                  ax = None, 
                  y_label = "Percent Frequency (%)",
                  geographic_orientation = False, 
                  bridge_axis = True,
                  dpi = None, 
                  add_fig = True, 
                  log_y = False, 
                  top_n_density=3,  # 新增：标注多少个最高密度区间
                  highlight_max_speed=False  # 新增：是否标注最高平均风速区间
                  ):
        
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, polar=True)

        if dpi:
            fig.dpi = dpi

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        # 使用 numpy 计算直方图数据
        n, bins = np.histogram(
            x,
            bins=separate_bins,
            density=density
        )

        # 转换为极坐标参数
        theta_edges = np.deg2rad(bins)
        theta_centers = theta_edges[:-1] + np.diff(theta_edges) / 2
        width = np.diff(theta_edges)

        # 手动绘制条形
        bars = ax.bar(
            theta_centers,
            n,
            width=width,
            bottom=0.0,
            color=color if color else 'skyblue'
        )

        # 去掉边框
        for bar in bars:
            bar.set_edgecolor('none')

        # 自动缩放 Y 轴
        if log_y:
            # 对数坐标不能包含 0，设置最小值为 1e-6
            y_min = max(np.min(n[n > 0]) * 0.9, 1e-6)
            y_max = max(n) * 1.5
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)
        else:
            max_n = max(n) if n.size > 0 else 1.0
            y_max = max_n * 1.1
            ax.set_ylim(0, y_max)

        # 设置 0 度位置在北方
        ax.set_theta_zero_location('N')

        ax.xaxis.grid(True)
        ax.yaxis.grid(ls='--')

        ax.set_title(title)

        if geographic_orientation:
            x_ticks = ['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE']
            ax.set_xticklabels(x_ticks)

        ax.tick_params(axis='y', labelleft=True, left=True)

        if bridge_axis:
            ax.text(-0.12, 0.5, y_label, rotation=90, transform=ax.transAxes, va="center", ha="center")
            axis_of_bridge = 10.6  # degree
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, y_max], color='pink')
            ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, y_max], color='pink')
            ax.annotate('Bridge axis', xy=(np.deg2rad(axis_of_bridge), y_max * 0.9), ha='center', va='bottom')

        # 添加平均风速注释
        if y_datas is not None:
            bin_indices = np.digitize(x, bins) - 1  # -1 调整索引
            bin_means = []
            for i in range(len(bins) - 1):
                mask = (bin_indices == i)
                if np.any(mask):
                    bin_mean = np.mean(y_datas[mask])
                else:
                    bin_mean = np.nan  # 空 bin
                bin_means.append(bin_mean)

            # 标注：密度最高的 top_n_density 个区间
            if top_n_density > 0:
                valid_density_indices = np.argsort(n)[::-1][:top_n_density]
                for idx in valid_density_indices:
                    if not np.isnan(bin_means[idx]):
                        ax.text(
                            theta_centers[idx],
                            n[idx] * 1.15,  # 略高于柱形
                            f'{bin_means[idx]:.1f} m/s',
                            ha='center', va='center',
                            fontsize=GLOBAL_FONT_SIZE,
                            color='black'
                        )

            # 标注：最高平均风速区间
            if highlight_max_speed:
                max_speed_idx = np.nanargmax(bin_means)
                if not np.isnan(bin_means[max_speed_idx]):
                    ax.text(
                        theta_centers[max_speed_idx],
                        n[max_speed_idx] * 1.3,  # 更高位置，突出显示
                        f'{bin_means[max_speed_idx]:.1f} m/s',
                        ha='center', va='center',
                        fontsize=10,
                        color='red',
                        fontweight='bold'
                    )

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

