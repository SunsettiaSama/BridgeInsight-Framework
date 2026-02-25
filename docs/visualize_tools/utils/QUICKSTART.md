# 快速入门

## 5 分钟快速开始

### 最简单的使用方式

```python
from src.visualize_tools.utils import PlotLib
import numpy as np

# 创建绘图工具
lib = PlotLib()

# 绘制一条线
y = [1, 2, 3, 4, 5]
fig, ax = lib.plot(y=y, title="Simple Plot")

# 显示图表
lib.show()
```

---

## 理解 fig/ax 机制（最重要）

### 什么时候需要理解 fig/ax？

当你需要在同一图表上绘制多条线或多个数据时。

### 最常见的场景

**场景 1: 绘制单个图表（不需要理解 fig/ax）**

```python
lib = PlotLib()
fig, ax = lib.plot(y=[1, 2, 3, 4])
lib.show()
```

**场景 2: 在同一图表上绘制多条线（需要理解 fig/ax）**

```python
lib = PlotLib()

# 第一条线
fig, ax = lib.plot(
    y=[1, 2, 3, 4],
    legend='Line 1',
    color='blue'
)

# 第二条线 - 传入第一次返回的 fig 和 ax
fig, ax = lib.plot(
    y=[4, 3, 2, 1],
    legend='Line 2',
    color='red',
    fig=fig,        # 重要：使用同一个 fig
    ax=ax,          # 重要：使用同一个 ax
    add_fig=False   # 重要：不重复添加
)

ax.legend()
lib.show()
```

**关键点**：
- 第一次调用：`lib.plot()` 会创建新的 `fig` 和 `ax`
- 后续调用：传入相同的 `fig` 和 `ax` 参数，在同一图表上添加数据
- 最后一次调用：设置 `add_fig=False`（或只最后手动 `lib.figs.append(fig)`）

---

## 通用工作流

### Step 1: 创建 PlotLib 实例

```python
from src.visualize_tools.utils import PlotLib

lib = PlotLib()
```

### Step 2: 调用绘图方法

```python
import numpy as np

# 准备数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制
fig, ax = lib.plot(
    y=y,
    x=x,
    title='Sine Wave',
    xlabel='Time (s)',
    ylabel='Amplitude',
    legend='sin(x)'
)
```

### Step 3: 显示图表

```python
lib.show()
```

---

## 常用图表类型

### 线图 (plot)

```python
lib.plot(y=[1,2,3,4], title='Simple Line')
```

### 散点图 (scatter)

```python
lib.scatter(x=[1,2,3,4], y=[1,4,9,16], title='Scatter')
```

### 直方图 (hist)

```python
data = np.random.randn(1000)
lib.hist(x=data, bins=30, title='Histogram')
```

### 玫瑰图 (rose_hist) - 用于风向分布

```python
wind_dir = np.random.uniform(0, 360, 500)
lib.rose_hist(x=wind_dir, title='Wind Distribution')
```

### 多子图 (plots)

```python
data_list = [
    (x, np.sin(x)),
    (x, np.cos(x)),
    (x, np.tan(x))
]
fig, axes = lib.plots(
    data_list,
    titles=['sin', 'cos', 'tan']
)
```

---

## 常见用例

### 用例 1: 对比两个信号

```python
from src.visualize_tools.utils import PlotLib
import numpy as np

lib = PlotLib()
x = np.linspace(0, 10, 100)

# 绘制信号 1
fig, ax = lib.plot(y=np.sin(x), legend='Signal 1', color='blue')

# 在同一图表上添加信号 2
fig, ax = lib.plot(
    y=np.cos(x),
    legend='Signal 2',
    color='red',
    fig=fig, ax=ax,
    add_fig=False
)

ax.legend()
lib.show()
```

### 用例 2: 显示多个相关的图表

```python
lib = PlotLib()

# 图表 1
lib.plot(y=np.random.randn(100), title='Chart 1')

# 图表 2
lib.plot(y=np.random.randn(100), title='Chart 2')

# 图表 3
lib.plot(y=np.random.randn(100), title='Chart 3')

# 使用 GUI 前后导航查看
lib.show()
```

### 用例 3: 风向风速分析

```python
lib = PlotLib()

wind_dir = np.random.uniform(0, 360, 1000)
wind_speed = np.random.exponential(5, 1000)

# 玫瑰图显示风向分布和平均风速
lib.rose_hist(
    x=wind_dir,
    y_datas=wind_speed,
    geographic_orientation=True,
    top_n_density=3,
    highlight_max_speed=True
)

lib.show()
```

### 用例 4: 4 个子图的完整流程

```python
import matplotlib.pyplot as plt
from src.visualize_tools.utils import PlotLib
import numpy as np

lib = PlotLib()

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 准备数据
x = np.linspace(0, 10, 100)
signals = [
    np.sin(x),
    np.cos(x),
    np.sin(2*x),
    np.cos(2*x)
]
titles = ['sin(x)', 'cos(x)', 'sin(2x)', 'cos(2x)']

# 在每个子图上绘制
for i, (ax, signal, title) in enumerate(zip(axes.flat, signals, titles)):
    lib.plot(y=signal, title=title, ax=ax, fig=fig, add_fig=False)

lib.figs.append(fig)
lib.show()
```

---

## show() 方法详解

### 什么是 show()?

`show()` 方法启动一个 Tkinter 窗口，让你可以交互式地浏览 `lib.figs` 中的所有图表。

### 功能按钮

- **Previous**: 查看前一个图表
- **Next**: 查看后一个图表
- **Save**: 保存当前图表为 PNG

### 使用示例

```python
lib = PlotLib()

# 添加多个图表
for i in range(5):
    lib.plot(y=np.random.randn(100), title=f'Chart {i+1}')

# 启动交互式浏览
lib.show()
```

---

## 参数速查表

| 方法 | 参数 | 说明 |
|------|------|------|
| plot | y, x | 必需：y 数据和可选 x 数据 |
| plot | title, xlabel, ylabel | 标签参数 |
| plot | color, alpha, style | 样式参数 |
| plot | xlim, ylim | 轴范围参数 |
| plot | fig, ax, add_fig | 管理参数 |
| scatter | x, y | 必需：x, y 数据 |
| scatter | marker, s | 标记参数 |
| hist | x, bins | 必需：数据和箱数 |
| rose_hist | x, y_datas | 必需：角度和可选的数值 |
| plots | data_lis | 必需：数据列表 |
| show | - | 无参数 |

---

## 常见错误及解决方案

### 错误 1: 多次添加同一图表

**症状**: `lib.show()` 时看到重复的图表

**原因**: 多次调用 `lib.plot()` 时都设置了 `add_fig=True`（默认值）

**解决**:
```python
# 错误做法
fig, ax = lib.plot(y1)  # add_fig=True（默认）
fig, ax = lib.plot(y2, fig=fig, ax=ax)  # 又一次 add_fig=True

# 正确做法
fig, ax = lib.plot(y1)
fig, ax = lib.plot(y2, fig=fig, ax=ax, add_fig=False)  # 只最后一次为 True
```

---

### 错误 2: 图表显示不出来

**症状**: `lib.show()` 启动窗口但看不到图表

**原因**: 没有调用 `lib.plot()` 或图表没有添加到 `lib.figs`

**解决**:
```python
# 检查是否有图表
print(len(lib.figs))  # 应该 > 0

# 手动添加
lib.figs.append(fig)

# 再调用 show
lib.show()
```

---

### 错误 3: fig/ax 参数传错了

**症状**: 新图表覆盖了旧图表，或图表显示错乱

**原因**: 没有正确传入 `fig` 和 `ax`，或忘记设置 `add_fig=False`

**解决**:
```python
# 错误：没有传 fig 和 ax
fig1, ax1 = lib.plot(y1)
fig2, ax2 = lib.plot(y2)  # 这会创建新图表，不会添加到 fig1

# 正确
fig1, ax1 = lib.plot(y1)
fig1, ax1 = lib.plot(y2, fig=fig1, ax=ax1, add_fig=False)
```

---

## 性能提示

### 大数据绘制

如果数据量很大（> 1,000,000 点），使用 `scatter` 而不是 `plot` 会更快。

```python
# 较慢（大数据）
lib.plot(y=huge_data)

# 较快（采样或使用 scatter）
lib.scatter(x=x[::100], y=huge_data[::100])  # 采样
```

### 多图表显示

如果有很多图表，可以分批显示：

```python
lib.figs = lib.figs[:10]  # 只显示前 10 个
lib.show()
```

---

## 下一步

- 查看 [完整使用文档](README.md) 了解所有参数和用例
- 查看 [API 参考](API.md) 获得详细的函数签名
- 探索示例代码学习高级用法

---

## 速记清单

✅ 单个图表 → 直接调用 `lib.plot()`  
✅ 多条线同一图 → 传入 `fig`, `ax`，设 `add_fig=False`  
✅ 多个子图 → 使用 `plt.subplots()`  
✅ 查看所有图表 → 调用 `lib.show()`  
✅ 保存图表 → 使用 `show()` GUI 或 `fig.savefig()`  
