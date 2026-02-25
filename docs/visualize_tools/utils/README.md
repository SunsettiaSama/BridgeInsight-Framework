# 可视化工具库 (utils.py) 使用文档

## 模块概述

`src/visualize_tools/utils.py` 是一个功能丰富的数据可视化模块，提供了 `PlotLib` 类作为主要的绘图接口。它支持多种图表类型（线图、散点图、玫瑰图、3D 散点图、直方图等），并通过灵活的 `fig`、`ax` 和 `show` 机制支持复杂的多子图绘制和交互式显示。

### 核心特性

- **统一的绘图接口**: `PlotLib` 类提供了一致的 API，适用于多种图表类型
- **灵活的图表管理**: 通过 `fig`/`ax` 参数支持在现有图表上绘制，或创建新图表
- **交互式显示**: `show()` 方法提供 Tkinter GUI 界面，支持图表切换、保存等功能
- **自动添加到管理器**: `add_fig` 参数控制图表是否自动添加到内部列表
- **支持多种坐标系**: 支持直角坐标系、极坐标系、3D 坐标系等
- **丰富的自定义选项**: 颜色、透明度、标记、图例等多种样式参数

---

## 关键概念

### fig/ax 机制

这是理解和有效使用 `PlotLib` 的核心。

#### 什么是 fig 和 ax？

- **fig** (Figure): Matplotlib 的顶级容器，代表整个图表窗口
- **ax** (Axes): 在 fig 内绘制图表的坐标系区域

#### 三种使用模式

##### 模式 1: 自动创建新图表（默认）

```python
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 如果不提供 fig 和 ax，会自动创建新的图表
fig, ax = lib.plot(
    y=[1, 2, 3, 4],
    title="New Figure"
)
```

**特点**:
- 最简单，无需管理 fig/ax
- 每次调用创建一个新图表
- 图表自动添加到 `lib.figs` 列表（通过 `add_fig=True`）

##### 模式 2: 在已有的 fig/ax 上绘制

```python
import matplotlib.pyplot as plt
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 创建一个 figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# 在这个 ax 上绘制多条线
fig, ax = lib.plot(
    y=[1, 2, 3, 4],
    x=[0, 1, 2, 3],
    color='blue',
    legend='Series 1',
    fig=fig,
    ax=ax,
    add_fig=False  # 不重复添加
)

fig, ax = lib.plot(
    y=[4, 3, 2, 1],
    x=[0, 1, 2, 3],
    color='red',
    legend='Series 2',
    fig=fig,
    ax=ax,
    add_fig=False
)

ax.legend()
lib.figs.append(fig)  # 手动添加一次
```

**特点**:
- 适合在同一坐标系上绘制多条数据
- 需要手动管理 fig/ax
- 通常设置 `add_fig=False` 避免重复添加

##### 模式 3: 子图绘制

```python
import matplotlib.pyplot as plt
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 创建 2x2 的子图布局
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 在每个子图上绘制
for i, ax in enumerate(axes.flat):
    lib.plot(
        y=np.random.rand(10),
        title=f"Subplot {i+1}",
        fig=fig,
        ax=ax,
        add_fig=False
    )

lib.figs.append(fig)
```

**特点**:
- 适合多个相关图表的同时展示
- 提高代码组织性和可读性

---

### show() 机制

`show()` 方法通过 Tkinter 提供交互式图表浏览界面。

#### 基本用法

```python
lib = PlotLib()

# 添加多个图表
for i in range(5):
    lib.plot(
        y=np.random.rand(100),
        title=f"Chart {i+1}"
    )

# 显示所有图表
lib.show()
```

#### show() 的功能

- **前后导航**: "Previous" 和 "Next" 按钮
- **保存图表**: "Save" 按钮，支持 PNG 和其他格式
- **实时显示**: 任何添加到 `self.figs` 的图表都可以浏览

#### 工作原理

`show()` 底层使用 `ChartApp` 类：

```python
def show(self):
    '''
    显示所有记录的图像
    '''
    tk = Tk()
    app = ChartApp(tk, self.figs)
    tk.mainloop()
    return
```

`ChartApp` 将 Matplotlib 图表集成到 Tkinter 窗口中，实现交互式浏览。

---

## 常用方法参考

### plot() - 基础线图

最常用的方法，用于绘制线图。

```python
fig, ax = lib.plot(
    y,                      # 必需：y 数据
    x=None,                 # 可选：x 数据，如果 None 则使用索引
    title=None,             # 可选：图表标题
    xlabel=None,            # 可选：x 轴标签
    ylabel=None,            # 可选：y 轴标签
    color=None,             # 可选：线条颜色，默认 'skyblue'
    legend=None,            # 可选：图例标签
    xlim=None,              # 可选：x 轴范围，tuple (min, max)
    ylim=None,              # 可选：y 轴范围，tuple (min, max)
    dpi=None,               # 可选：图表分辨率
    fig=None,               # 可选：现有 Figure 对象
    style=None,             # 可选：线条样式 ('-', '--', '-.', ':')
    alpha=None,             # 可选：透明度 (0-1)，默认 0.8
    ax=None,                # 可选：现有 Axes 对象
    add_fig=True            # 可选：是否添加到 self.figs
)
```

#### 示例

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 简单线图
time = np.linspace(0, 10, 100)
signal = np.sin(time)

fig, ax = lib.plot(
    y=signal,
    x=time,
    title='Sine Wave',
    xlabel='Time (s)',
    ylabel='Amplitude',
    color='blue',
    legend='sin(t)'
)

lib.show()
```

---

### scatter() - 散点图

```python
fig, ax = lib.scatter(
    x, y,                   # 必需：x, y 数据
    title=None,             # 可选：图表标题
    xlabel=None,            # 可选：x 轴标签
    ylabel=None,            # 可选：y 轴标签
    xlim=None,              # 可选：x 轴范围
    ylim=None,              # 可选：y 轴范围
    dpi=None,               # 可选：分辨率
    fig=None,               # 可选：现有 Figure
    ax=None,                # 可选：现有 Axes
    marker=None,            # 可选：标记形状 ('o', 's', '^', etc.)
    color=None,             # 可选：颜色
    legend=None,            # 可选：图例
    s=None,                 # 可选：标记大小，默认 1
    alpha=0.8,              # 可选：透明度
    add_fig=True            # 可选：是否添加到 self.figs
)
```

#### 示例

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 生成随机数据
x = np.random.randn(100)
y = np.random.randn(100)

fig, ax = lib.scatter(
    x=x, y=y,
    title='Random Scatter Plot',
    xlabel='X',
    ylabel='Y',
    s=20,
    color='red',
    alpha=0.6
)

lib.show()
```

---

### rose_hist() - 极坐标玫瑰图

用于显示风向分布等圆形数据。

```python
fig, ax = lib.rose_hist(
    x,                      # 必需：方向数据（度数），0-360
    y_datas=None,           # 可选：对应每个方向的标量值（如风速）
    title=None,             # 可选：标题
    separate_bins=20,       # 可选：分箱数
    density=True,           # 可选：是否归一化
    color=None,             # 可选：颜色
    fig=None,               # 可选：现有 Figure
    ax=None,                # 可选：现有 Axes
    y_label=None,           # 可选：径向轴标签
    geographic_orientation=False,  # 可选：是否显示地理方向（N, S, E, W）
    bridge_axis=True,       # 可选：是否显示桥梁中轴线（10.6°）
    dpi=None,               # 可选：分辨率
    add_fig=True,           # 可选：是否添加到 self.figs
    log_y=False,            # 可选：是否使用对数坐标
    top_n_density=3,        # 可选：标注最高密度的 N 个区间
    highlight_max_speed=False  # 可选：是否突出显示最高平均值区间
)
```

#### 示例

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 风向数据（0-360 度）
wind_direction = np.random.uniform(0, 360, 1000)

# 风速数据
wind_speed = np.random.uniform(0, 20, 1000)

fig, ax = lib.rose_hist(
    x=wind_direction,
    y_datas=wind_speed,
    title='Wind Direction Distribution',
    y_label='Frequency (%)',
    geographic_orientation=True,
    bridge_axis=True,
    top_n_density=3,
    highlight_max_speed=True
)

lib.show()
```

---

### plots() - 多个子图（线图）

一次创建多个行子图，每行一个数据。

```python
fig, axes = lib.plots(
    data_lis,               # 必需：数据列表 [(x1, y1), (x2, y2), ...]
    color=None,             # 可选：颜色
    alpha=0.7,              # 可选：透明度
    titles=None,            # 可选：标题列表
    labels=None,            # 可选：轴标签列表 [('x1', 'y1'), ('x2', 'y2'), ...]
    xlims=None,             # 可选：x 轴范围列表
    ylims=None,             # 可选：y 轴范围列表
    coordinate_systems=None,# 可选：坐标系列表 ['linear', 'loglog', 'semilogy']
    add_fig=True            # 可选：是否添加到 self.figs
)
```

#### 示例

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 准备数据
t = np.linspace(0, 10, 100)
data_lis = [
    (t, np.sin(t)),
    (t, np.cos(t)),
    (t, np.tan(t))
]

fig, axes = lib.plots(
    data_lis,
    titles=['sin(t)', 'cos(t)', 'tan(t)'],
    labels=[
        ('Time', 'Amplitude'),
        ('Time', 'Amplitude'),
        ('Time', 'Amplitude')
    ]
)

lib.show()
```

---

### scatters() - 多个子图（散点图）

```python
fig, axes = lib.scatters(
    data_lis,               # 必需：数据列表 [(x1, y1), (x2, y2), ...]
    color=None,             # 可选：颜色
    alpha=0.7,              # 可选：透明度
    titles=None,            # 可选：标题列表
    labels=None,            # 可选：轴标签列表
    xlims=None,             # 可选：x 轴范围列表
    ylims=None,             # 可选：y 轴范围列表
    add_fig=True,           # 可选：是否添加到 self.figs
    s=0.8                   # 可选：标记大小
)
```

---

### hist() - 直方图

```python
fig, ax = lib.hist(
    x,                      # 必需：数据
    bins=50,                # 可选：箱数
    density=True,           # 可选：是否归一化
    title=None,             # 可选：标题
    xlabel=None,            # 可选：x 轴标签
    ylabel=None,            # 可选：y 轴标签
    dpi=None,               # 可选：分辨率
    fig=None,               # 可选：现有 Figure
    ax=None,                # 可选：现有 Axes
    color=None,             # 可选：颜色
    legend=None,            # 可选：图例
    alpha=0.7,              # 可选：透明度
    add_fig=True            # 可选：是否添加到 self.figs
)
```

---

### scatter_3d() - 3D 散点图

支持柱坐标系的 3D 散点图，适合显示风向、距离和高度的关系。

```python
fig, ax = lib.scatter_3d(
    theta, rho, z,          # 必需：极角、径向、高度
    title=None,             # 可选：标题
    xlabel=None,            # 可选：x 轴标签
    ylabel=None,            # 可选：y 轴标签
    zlabel=None,            # 可选：z 轴标签
    xlim=None,              # 可选：x 轴范围
    ylim=None,              # 可选：y 轴范围
    dpi=None,               # 可选：分辨率
    fig=None,               # 可选：现有 Figure
    ax=None,                # 可选：现有 Axes
    marker=None,            # 可选：标记形状
    color=None,             # 可选：颜色
    edgecolor=None,         # 可选：边框颜色
    legend=None,            # 可选：图例
    s=None,                 # 可选：标记大小
    alpha=0.5,              # 可选：透明度
    add_fig=True            # 可选：是否添加到 self.figs
)
```

---

### rose_scatter() - 极坐标散点图

在极坐标上绘制散点，支持颜色映射。

```python
fig, ax = lib.rose_scatter(
    theta, rho,             # 必需：极角和径向值
    title=None,             # 可选：标题
    xlabel=None,            # 可选：x 轴标签
    ylabel=None,            # 可选：y 轴标签
    xlim=None,              # 可选：x 轴范围
    ylim=None,              # 可选：y 轴范围
    dpi=None,               # 可选：分辨率
    fig=None,               # 可选：现有 Figure
    ax=None,                # 可选：现有 Axes
    s=None,                 # 可选：标记大小，默认 1
    marker=None,            # 可选：标记形状
    legend=None,            # 可选：图例
    color=None,             # 可选：单一颜色
    colorbar=None,          # 可选：是否显示 colorbar
    cmap=None,              # 可选：颜色映射名称 ('viridis', 'plasma' 等)
    c_data=None,            # 可选：用于颜色映射的数据
    color_label=None,       # 可选：colorbar 标签
    geographic_orientation=True,  # 可选：是否显示地理方向
    bridge_axis=True,       # 可选：是否显示桥梁中轴线
    norm=None,              # 可选：Normalize 实例
    alpha=0.8,              # 可选：透明度
    add_fig=True            # 可选：是否添加到 self.figs
)
```

---

### animate() - 动画

生成动画（GIF 或 MP4）。

```python
ani = lib.animate(
    data_lis,               # 必需：数据列表 [(x_data, y_data), ...]
    color=None,             # 可选：颜色
    alpha=0.7,              # 可选：透明度
    title=None,             # 可选：标题列表
    labels=None,            # 可选：轴标签列表
    xlims=None,             # 可选：x 轴范围列表
    ylims=None,             # 可选：y 轴范围列表
    fps=None,               # 可选：帧率，默认 3
    save_path=None,         # 可选：保存路径（.gif 或 .mp4）
    dynamic_adjust=[(True, False), (False, False)]  # 可选：是否动态调整轴
)
```

#### 示例

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 准备动画数据：每帧一个 x 和 y 的完整数据
frames = 50
x_data = np.tile(np.linspace(0, 10, 100), (frames, 1))
y_data = np.array([np.sin(x_data[i] + i/10) for i in range(frames)])

data_lis = [(x_data, y_data)]

ani = lib.animate(
    data_lis,
    fps=10,
    save_path='animation.gif'
)
```

---

## 实际应用示例

### 示例 1: 多传感器数据对比

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 模拟多个传感器的时间序列数据
time = np.linspace(0, 100, 1000)
sensors = {
    'Sensor 1': np.sin(time) + np.random.normal(0, 0.1, 1000),
    'Sensor 2': np.cos(time) + np.random.normal(0, 0.1, 1000),
    'Sensor 3': np.sin(2*time) + np.random.normal(0, 0.1, 1000),
}

# 在同一图表上绘制所有传感器数据
fig = None
ax = None

for sensor_name, data in sensors.items():
    fig, ax = lib.plot(
        y=data,
        x=time,
        legend=sensor_name,
        fig=fig,
        ax=ax,
        add_fig=False,
        alpha=0.7
    )

if fig:
    ax.set_title('Multi-Sensor Comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    lib.figs.append(fig)

lib.show()
```

---

### 示例 2: 子图布局对比分析

```python
import numpy as np
import matplotlib.pyplot as plt
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 第一行：时域信号
signal1 = np.sin(np.linspace(0, 10, 100))
signal2 = np.cos(np.linspace(0, 10, 100))

lib.plot(y=signal1, title='Signal 1', ax=axes[0, 0], fig=fig, add_fig=False)
lib.plot(y=signal2, title='Signal 2', ax=axes[0, 1], fig=fig, add_fig=False)

# 第二行：对应的直方图
lib.hist(x=signal1, title='Histogram 1', ax=axes[1, 0], fig=fig, add_fig=False)
lib.hist(x=signal2, title='Histogram 2', ax=axes[1, 1], fig=fig, add_fig=False)

lib.figs.append(fig)
lib.show()
```

---

### 示例 3: 风速风向分析

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 模拟风数据
wind_directions = np.random.uniform(0, 360, 1000)
wind_speeds = np.random.exponential(5, 1000)  # 指数分布

# 玫瑰图：风向分布与平均风速
fig, ax = lib.rose_hist(
    x=wind_directions,
    y_datas=wind_speeds,
    title='Wind Distribution Analysis',
    y_label='Frequency (%)',
    geographic_orientation=True,
    top_n_density=5,
    highlight_max_speed=True
)

# 极坐标散点图：风向-风速关系
fig2, ax2 = lib.rose_scatter(
    theta=np.deg2rad(wind_directions),
    rho=wind_speeds,
    c_data=wind_speeds,
    cmap='viridis',
    colorbar=True,
    color_label='Wind Speed (m/s)',
    title='Wind Speed vs Direction',
    geographic_orientation=True
)

lib.show()
```

---

### 示例 4: 在已有图表上追加绘制

```python
import numpy as np
from src.visualize_tools.utils import PlotLib

lib = PlotLib()

# 创建一个新图表
time = np.linspace(0, 10, 100)
baseline = np.sin(time)

fig, ax = lib.plot(
    y=baseline,
    x=time,
    color='blue',
    legend='Baseline',
    title='Data Overlay Example'
)

# 在同一图表上添加新数据
noise = np.random.normal(0, 0.2, 100)
fig, ax = lib.plot(
    y=baseline + noise,
    x=time,
    color='red',
    legend='Noisy Signal',
    fig=fig,
    ax=ax,
    add_fig=False,
    alpha=0.5
)

ax.legend()
lib.show()
```

---

## 常见问题

### Q: 如何在同一图表上绘制多条线？

A: 使用 `fig` 和 `ax` 参数，设置 `add_fig=False`：

```python
fig, ax = lib.plot(y1, legend='Line 1')
fig, ax = lib.plot(y2, fig=fig, ax=ax, legend='Line 2', add_fig=False)
ax.legend()
lib.figs.append(fig)
```

---

### Q: 如何自定义图表大小和分辨率？

A: 通过 Matplotlib 的 `figure()` 或 `subplots()` 创建图表，然后传入 `fig` 和 `ax` 参数：

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 6), dpi=150)
ax = fig.add_subplot(111)

fig, ax = lib.plot(y=data, fig=fig, ax=ax)
lib.figs.append(fig)
```

---

### Q: 如何只显示某些图表，不显示全部？

A: 创建一个新的 `PlotLib` 实例或手动管理 `figs` 列表：

```python
lib = PlotLib()
lib.plot(...)  # 图表 1
lib.plot(...)  # 图表 2

# 只显示前 N 个
lib.figs = lib.figs[:N]
lib.show()
```

---

### Q: 能否在 Jupyter Notebook 中使用？

A: 可以，但不需要调用 `show()`。Matplotlib 会自动在 Notebook 中显示：

```python
%matplotlib inline
from src.visualize_tools.utils import PlotLib

lib = PlotLib()
fig, ax = lib.plot(y=[1,2,3,4])
# 图表会自动显示，无需调用 lib.show()
```

---

### Q: 如何保存图表？

A: 有两种方式：

1. **使用 `show()` GUI** 的 "Save" 按钮
2. **直接使用 Matplotlib**：

```python
fig.savefig('output.png', dpi=150, bbox_inches='tight')
fig.savefig('output.pdf', bbox_inches='tight')
```

---

### Q: 如何改变全局字体和样式？

A: 在模块开始处修改 Matplotlib 配置：

```python
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12
```

模块内置配置：

```python
GLOBAL_FONT_SIZE = 10
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = GLOBAL_FONT_SIZE
```

---

## 返回数据结构

所有绘图方法都返回 `(fig, ax)` 或 `(fig, axes)` 元组：

- **fig**: Matplotlib Figure 对象，代表整个图表窗口
- **ax** 或 **axes**: Matplotlib Axes 或 Axes 数组对象

这些对象可以进一步定制或传入其他方法。

---

## 模块依赖

- `matplotlib`: 绘图库
- `numpy`: 数值计算
- `pandas`: 数据处理
- `scipy.signal`: 信号处理
- `tkinter`: GUI 框架（用于 `show()` 和 `ChartApp`）
- `PIL`: 图像处理（用于 `_Propose_Data_GUI`）

---

## 版本信息

- **最后更新**: 2026-02-25
- **模块位置**: `src/visualize_tools/utils.py`
- **主要类**: `PlotLib`（绘图类）、`ChartApp`（交互显示）
- **其他类**: `_Propose_Data_GUI`（图像标注工具）

---

## 相关模块

- `matplotlib.pyplot`: 底层绘图接口
- `src.data_processer`: 数据处理模块

---

## 最佳实践

1. **使用 add_fig 控制添加时机**：只在最后一次绘制时添加
2. **合理使用 fig/ax 参数**：避免创建过多图表
3. **设置 alpha 提高可读性**：多条线重叠时
4. **使用 legend 标注数据**：便于理解
5. **保存高分辨率图表**：用于发表论文
