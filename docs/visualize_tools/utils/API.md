# API 参考

## PlotLib 类

主要的绘图类，所有绘图操作都通过这个类完成。

### 初始化

```python
lib = PlotLib()
```

**属性**:
- `figs`: list，存储所有创建的图表

---

## 方法列表

### plot() - 线图

```python
def plot(self, y, x=None, title=None, xlabel=None, ylabel=None,
         color=None, legend=None, xlim=None, ylim=None, dpi=None,
         fig=None, style=None, alpha=None, ax=None, add_fig=True)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| y | array-like | 必需 | y 轴数据 |
| x | array-like | None | x 轴数据，若 None 则使用索引 |
| title | str | None | 图表标题 |
| xlabel | str | None | x 轴标签 |
| ylabel | str | None | y 轴标签 |
| color | str | 'skyblue' | 线条颜色 |
| legend | str | None | 图例标签 |
| xlim | tuple | None | x 轴范围 (min, max) |
| ylim | tuple | None | y 轴范围 (min, max) |
| dpi | int | None | 图表分辨率 |
| fig | Figure | None | 现有 Figure 对象 |
| style | str | None | 线条样式 ('-', '--', '-.', ':') |
| alpha | float | 0.8 | 透明度 (0-1) |
| ax | Axes | None | 现有 Axes 对象 |
| add_fig | bool | True | 是否添加到 self.figs |

**返回**: `(fig, ax)` - Figure 和 Axes 对象

**示例**:

```python
# 基础用法
fig, ax = lib.plot(y=[1, 2, 3, 4])

# 带 x 轴和标签
fig, ax = lib.plot(
    y=[1, 2, 3, 4],
    x=[0, 1, 2, 3],
    title='My Chart',
    xlabel='X Axis',
    ylabel='Y Axis'
)

# 自定义样式
fig, ax = lib.plot(
    y=data,
    color='red',
    style='--',
    alpha=0.7,
    legend='Data'
)
```

---

### scatter() - 散点图

```python
def scatter(self, x, y, title=None, xlabel=None, ylabel=None,
            xlim=None, ylim=None, dpi=None, fig=None, ax=None,
            marker=None, color=None, legend=None, s=None,
            alpha=0.8, add_fig=True)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| x | array-like | 必需 | x 坐标 |
| y | array-like | 必需 | y 坐标 |
| title | str | None | 图表标题 |
| xlabel | str | None | x 轴标签 |
| ylabel | str | None | y 轴标签 |
| xlim | tuple | None | x 轴范围 |
| ylim | tuple | None | y 轴范围 |
| dpi | int | None | 分辨率 |
| fig | Figure | None | 现有 Figure |
| ax | Axes | None | 现有 Axes |
| marker | str | 'o' | 标记形状 ('o', 's', '^', 'v', etc.) |
| color | str | 'skyblue' | 颜色 |
| legend | str | None | 图例 |
| s | float | 1 | 标记大小 |
| alpha | float | 0.8 | 透明度 |
| add_fig | bool | True | 是否添加到 self.figs |

**返回**: `(fig, ax)`

---

### hist() - 直方图

```python
def hist(self, x, bins=50, density=True, title=None,
         xlabel=None, ylabel=None, dpi=None, fig=None,
         ax=None, color=None, legend=None, alpha=0.7, add_fig=True)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| x | array-like | 必需 | 数据 |
| bins | int | 50 | 箱数 |
| density | bool | True | 是否归一化 |
| title | str | None | 标题 |
| xlabel | str | None | x 轴标签 |
| ylabel | str | None | y 轴标签 |
| dpi | int | None | 分辨率 |
| fig | Figure | None | 现有 Figure |
| ax | Axes | None | 现有 Axes |
| color | str | 'skyblue' | 颜色 |
| legend | str | None | 图例 |
| alpha | float | 0.7 | 透明度 |
| add_fig | bool | True | 是否添加到 self.figs |

**返回**: `(fig, ax)`

---

### rose_hist() - 极坐标玫瑰图

```python
def rose_hist(self, x, y_datas=None, title=None, separate_bins=20,
              density=True, color=None, fig=None, ax=None,
              y_label=None, geographic_orientation=False,
              bridge_axis=True, dpi=None, add_fig=True,
              log_y=False, top_n_density=3, highlight_max_speed=False)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| x | array-like | 必需 | 方向数据 (0-360 度) |
| y_datas | array-like | None | 对应每个方向的标量值 |
| title | str | None | 标题 |
| separate_bins | int | 20 | 分箱数 |
| density | bool | True | 是否归一化 |
| color | str | 'skyblue' | 颜色 |
| fig | Figure | None | 现有 Figure |
| ax | Axes | None | 现有 Axes |
| y_label | str | "Percent Frequency (%)" | 径向轴标签 |
| geographic_orientation | bool | False | 是否显示方向标签 (N, S, E, W) |
| bridge_axis | bool | True | 是否显示桥梁中轴 |
| dpi | int | None | 分辨率 |
| add_fig | bool | True | 是否添加到 self.figs |
| log_y | bool | False | 是否使用对数坐标 |
| top_n_density | int | 3 | 标注最高密度的 N 个区间 |
| highlight_max_speed | bool | False | 是否突出最高平均值 |

**返回**: `(fig, ax)`

**示例**:

```python
import numpy as np

wind_dir = np.random.uniform(0, 360, 1000)
wind_speed = np.random.exponential(5, 1000)

fig, ax = lib.rose_hist(
    x=wind_dir,
    y_datas=wind_speed,
    geographic_orientation=True,
    top_n_density=5,
    highlight_max_speed=True
)
```

---

### rose_scatter() - 极坐标散点图

```python
def rose_scatter(self, theta, rho, title=None, xlabel=None,
                 ylabel=None, xlim=None, ylim=None, dpi=None,
                 fig=None, ax=None, s=None, marker=None,
                 legend=None, color=None, colorbar=None,
                 cmap=None, c_data=None, color_label=None,
                 geographic_orientation=True, bridge_axis=True,
                 norm=None, alpha=0.8, add_fig=True)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| theta | array-like | 必需 | 极角 (弧度或度数) |
| rho | array-like | 必需 | 径向值 |
| title | str | None | 标题 |
| xlabel | str | None | x 标签 |
| ylabel | str | None | y 标签 |
| xlim | tuple | None | x 范围 |
| ylim | tuple | None | y 范围 |
| dpi | int | None | 分辨率 |
| fig | Figure | None | 现有 Figure |
| ax | Axes | None | 现有 Axes |
| s | float | 1 | 标记大小 |
| marker | str | 'o' | 标记形状 |
| legend | str | None | 图例 |
| color | str | 'skyblue' | 颜色 |
| colorbar | bool | None | 是否显示 colorbar |
| cmap | str | None | 颜色映射 ('viridis', 'plasma', etc.) |
| c_data | array-like | None | 用于颜色映射的数据 |
| color_label | str | None | colorbar 标签 |
| geographic_orientation | bool | True | 是否显示地理方向 |
| bridge_axis | bool | True | 是否显示桥梁中轴 |
| norm | Normalize | None | 归一化方式 |
| alpha | float | 0.8 | 透明度 |
| add_fig | bool | True | 是否添加到 self.figs |

**返回**: `(fig, ax)`

---

### scatter_3d() - 3D 散点图

```python
def scatter_3d(self, theta, rho, z, title=None, xlabel=None,
               ylabel=None, zlabel=None, xlim=None, ylim=None,
               dpi=None, fig=None, ax=None, marker=None,
               color=None, edgecolor=None, legend=None, s=None,
               alpha=0.5, add_fig=True)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| theta | array-like | 必需 | 极角 |
| rho | array-like | 必需 | 径向值 |
| z | array-like | 必需 | 高度值 |
| title | str | None | 标题 |
| xlabel | str | None | x 标签 |
| ylabel | str | None | y 标签 |
| zlabel | str | None | z 标签 |
| xlim | tuple | None | x 范围 |
| ylim | tuple | None | y 范围 |
| dpi | int | None | 分辨率 |
| fig | Figure | None | 现有 Figure |
| ax | Axes3D | None | 现有 3D Axes |
| marker | str | 'o' | 标记形状 |
| color | str | 'skyblue' | 颜色 |
| edgecolor | str | 'none' | 边框颜色 |
| legend | str | None | 图例 |
| s | float | 1 | 标记大小 |
| alpha | float | 0.5 | 透明度 |
| add_fig | bool | True | 是否添加到 self.figs |

**返回**: `(fig, ax)`

**示例**:

```python
import numpy as np

theta = np.random.uniform(0, 2*np.pi, 1000)
rho = np.random.uniform(0, 50, 1000)
z = np.random.uniform(0, 100, 1000)

fig, ax = lib.scatter_3d(
    theta=theta,
    rho=rho,
    z=z,
    title='3D Scatter Plot'
)
```

---

### plots() - 多个子图（线图）

```python
def plots(self, data_lis, color=None, alpha=0.7, titles=None,
          labels=None, xlims=None, ylims=None,
          coordinate_systems=None, add_fig=True)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| data_lis | list | 必需 | 数据列表 `[(x1, y1), (x2, y2), ...]` |
| color | str | 'skyblue' | 颜色 |
| alpha | float | 0.7 | 透明度 |
| titles | list | None | 标题列表 |
| labels | list | None | 轴标签列表 `[('x1', 'y1'), ...]` |
| xlims | list | None | x 范围列表 |
| ylims | list | None | y 范围列表 |
| coordinate_systems | list | None | 坐标系列表 `['linear', 'loglog', 'semilogy']` |
| add_fig | bool | True | 是否添加到 self.figs |

**返回**: `(fig, axes)`

**示例**:

```python
import numpy as np

x = np.linspace(0, 10, 100)
data_lis = [
    (x, np.sin(x)),
    (x, np.cos(x)),
    (x, np.tan(x))
]

fig, axes = lib.plots(
    data_lis,
    titles=['sin(x)', 'cos(x)', 'tan(x)'],
    labels=[
        ('x', 'y'),
        ('x', 'y'),
        ('x', 'y')
    ]
)
```

---

### scatters() - 多个子图（散点图）

```python
def scatters(self, data_lis, color=None, alpha=0.7, titles=None,
             labels=None, xlims=None, ylims=None, add_fig=True, s=0.8)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| data_lis | list | 必需 | 数据列表 `[(x1, y1), (x2, y2), ...]` |
| color | str | 'skyblue' | 颜色 |
| alpha | float | 0.7 | 透明度 |
| titles | list | None | 标题列表 |
| labels | list | None | 轴标签列表 |
| xlims | list | None | x 范围列表 |
| ylims | list | None | y 范围列表 |
| add_fig | bool | True | 是否添加到 self.figs |
| s | float | 0.8 | 标记大小 |

**返回**: `(fig, axes)`

---

### show() - 交互式显示

```python
def show(self)
```

**参数**: 无

**返回**: None

**说明**: 启动 Tkinter 窗口显示 `self.figs` 中的所有图表

**功能**:
- **Previous**: 查看前一个图表
- **Next**: 查看后一个图表
- **Save**: 保存当前图表

**示例**:

```python
lib = PlotLib()
lib.plot(y=[1, 2, 3, 4])
lib.plot(y=[4, 3, 2, 1])
lib.show()  # 启动 GUI，可前后导航
```

---

### animate() - 动画

```python
def animate(self, data_lis, color=None, alpha=0.7, title=None,
            labels=None, xlims=None, ylims=None, fps=None,
            save_path=None, dynamic_adjust=[(True, False), (False, False)])
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| data_lis | list | 必需 | 动画数据 `[(x_data, y_data), ...]` |
| color | str | 'skyblue' | 颜色 |
| alpha | float | 0.7 | 透明度 |
| title | list | None | 标题列表 |
| labels | list | None | 轴标签列表 |
| xlims | list | None | x 范围列表 |
| ylims | list | None | y 范围列表 |
| fps | int | 3 | 帧率 |
| save_path | str | None | 保存路径 (.gif 或 .mp4) |
| dynamic_adjust | list | [(True, False), ...] | 是否动态调整轴 |

**返回**: Animation 对象

**说明**: 
- `x_data` 形状应为 `(frames, points)`
- `y_data` 形状应为 `(frames, points)`
- 如果指定 `save_path`，动画将被保存

**示例**:

```python
import numpy as np

frames = 50
t = np.linspace(0, 10, 100)
# 每帧的 t 数据
x_data = np.tile(t, (frames, 1))
# 每帧的 sin(t + offset) 数据
y_data = np.array([np.sin(x_data[i] + i*0.1) for i in range(frames)])

ani = lib.animate(
    data_lis=[(x_data, y_data)],
    fps=10,
    save_path='animation.gif'
)
```

---

### loglog() - 对数对数图

```python
def loglog(self, time_series, x=None, title='Time_Series',
           xlabel='Time', ylabel='Value', xlim=None, ylim=None,
           dpi=None, add_fig=True)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| time_series | array-like | 必需 | y 数据 |
| x | array-like | None | x 数据 |
| title | str | 'Time_Series' | 标题 |
| xlabel | str | 'Time' | x 轴标签 |
| ylabel | str | 'Value' | y 轴标签 |
| xlim | tuple | None | x 范围 |
| ylim | tuple | None | y 范围 |
| dpi | int | None | 分辨率 |
| add_fig | bool | True | 是否添加到 self.figs |

**返回**: `(fig, ax)`

---

### show_sample() - 快速显示信号样本

```python
def show_sample(self, data, fs=50, nperseg=256,
                add_fig=True, scatter=False)
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| data | array-like | 必需 | 时间序列数据 |
| fs | int | 50 | 采样频率 (Hz) |
| nperseg | int | 256 | 每个分段的长度 |
| add_fig | bool | True | 是否添加到 self.figs |
| scatter | bool | False | 是否使用散点图 |

**返回**: `(fig, axes)`

**说明**: 自动绘制时域和频域图表

---

## 常用常量

```python
GLOBAL_FONT_SIZE = 10  # 全局字体大小
```

---

## ChartApp 类

交互式图表浏览器（内部使用）。

```python
class ChartApp:
    def __init__(self, root, figs):
        # root: Tkinter 根窗口
        # figs: 图表列表
```

**功能**:
- 前后导航图表
- 保存当前图表
- Tkinter GUI 集成

---

## 数据结构规范

### 单条线数据

```python
y = [1, 2, 3, 4]  # 或 np.array([1, 2, 3, 4])
x = [0, 1, 2, 3]  # 如果不提供则使用索引
```

### 多条线数据

```python
data_lis = [
    (x1, y1),  # 第一个子图
    (x2, y2),  # 第二个子图
    ...
]
```

### 极坐标数据

```python
theta = np.array([0, 45, 90, 135, ...])  # 度数或弧度
rho = np.array([10, 15, 20, 25, ...])    # 径向值
```

### 3D 数据

```python
theta = np.array([...])  # 极角
rho = np.array([...])    # 径向
z = np.array([...])      # 高度
```

### 动画数据

```python
x_data = np.array([[...], [...], ...])  # (frames, points)
y_data = np.array([[...], [...], ...])  # (frames, points)
data_lis = [(x_data, y_data)]
```

---

## 错误处理

所有方法都返回 `(fig, ax)` 或 `(fig, axes)`，不抛出异常。

常见问题排查：

1. **图表不显示**: 检查 `lib.figs` 是否非空，调用 `lib.show()` 是否正确
2. **多重绘制失败**: 确保 `fig` 和 `ax` 参数正确，设置 `add_fig=False`
3. **保存失败**: 确保路径有效且有写入权限

---

## 版本信息

- **最后更新**: 2026-02-25
- **模块**: `src/visualize_tools/utils.py`
