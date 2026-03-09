import numpy as np
import matplotlib.pyplot as plt
from ..visualize_tools.utils import PlotLib



class Fig4: 

    @ staticmethod
    def Wind_Turbulence_Scatter():
        
        # 横坐标为风速、纵坐标为紊流度
        wind_velocity_mean = np.mean(wind_velocity)
        wind_velocity_rms = np.sqrt(np.mean((wind_velocity - wind_velocity_mean) ** 2))
        Wind_TI = wind_velocity_rms / wind_velocity_mean

        # 平均值和紊流度
        wind_velocity_lis_mean = np.hstack((wind_velocity_lis_mean, wind_velocity_mean))
        wind_TI_lis = np.hstack((wind_TI_lis, Wind_TI))

        # 区间数量
        interval_nums = 36
        # 归一化
        normalize = True
        sensor_ids = ["跨中桥面上游"]
        wind_dirs = [r"F:\Research\Vibration Characteristics In Cable Vibration\data\2023September\UAN", 
                     r"F:\Research\Vibration Characteristics In Cable Vibration\data\2024September"]
        titles = iter(["202409 Mid-Span Upstream","202409 Mid-Span Upstream"])

        ploter = PlotLib() 
        manager = DataManager()
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
                cmap = plt.cm.viridis
                norm = plt.Normalize(min(average_speeds), max(average_speeds))
                for r, bar, color in zip(average_speeds, bars, cmap(norm(average_speeds))):
                    bar.set_facecolor(color)

                # 添加颜色条
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, orientation='vertical', label='Average Wind Speed (m/s)', pad = 0.1) # 调整颜色调和主图之间的长度

                # 设置标题
                # ax.set_title(next(titles), va='bottom')

                # ############################
                # 图像处理，美化图像部分
                # ############################
                # 地理坐标
                x_ticks = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                ax.set_xticklabels(x_ticks)

                # 桥轴
                y_max = np.max(counts)
                # ax.set_ylim(0, y_max)
                axis_of_bridge = 10.6  # degree
                ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge), [0, y_max * 1.1], color='red', linestyle='--')
                ax.plot(np.ones(2) * np.deg2rad(axis_of_bridge + 180), [0, y_max * 1.1], color='red', linestyle='--')
                ax.annotate('Bridge Axis', xy=(np.deg2rad(axis_of_bridge), y_max * 0.9), ha='center', va='bottom')
                
                # 降低y轴刻度密度
                ax.yaxis.set_major_locator(plt.MultipleLocator(np.round(y_max * 0.25, 2)))
                # 调整y轴最大值略微大于柱状图的最大值
                ax.set_ylim(0, y_max * 1.1)
                ax.set_yticks(np.hstack([0, ax.get_yticks()[1: ]]), [str(int(np.round(max(0, num * 100)))) + "%" for num in ax.get_yticks()])
                # 调整y轴标签为左侧
                ax.yaxis.set_label_coords(-0.1, 1.1)

                # 调整y轴位置到玫瑰图左侧
                ax.set_rlabel_position(180 + 90)

                ploter.figs.append(fig)


        ploter.show()


