# 可视化工具库 (visualize_tools) 文档

本目录包含 `src/visualize_tools/` 模块的完整使用文档。

## 快速导航

### 📖 文档目录

#### **utils.py** - 主要绘图模块
可视化工具库的核心模块，提供了 `PlotLib` 类和多种图表类型。

- **[README.md](utils/README.md)** - 完整参考文档
  - 模块概述和核心特性
  - 深度讲解 `fig/ax` 机制（最常用的接口）
  - 所有方法的详细说明
  - 实际应用示例
  - 常见问题解答

- **[QUICKSTART.md](utils/QUICKSTART.md)** - 快速入门指南
  - 5分钟快速开始
  - `fig/ax` 机制的直观理解
  - 常用图表类型速览
  - 实用代码片段
  - 参数速查表

- **[API.md](utils/API.md)** - 完整 API 参考
  - 所有方法的签名和参数
  - 详细的参数表格
  - 数据结构规范
  - 代码示例

- **[REFACTOR.md](utils/REFACTOR.md)** - 设计决策文档
  - `fig/ax` 机制的设计原理
  - 为什么有 `add_fig` 参数
  - 常见陷阱和最佳实践
  - 集成 `show()` 的工作原理

---

## 学习路径

### 第一次使用？

1. 先读 [QUICKSTART.md](utils/QUICKSTART.md) 的前两部分
2. 运行简单示例体验效果
3. 遇到 `fig/ax` 的问题，查看该文档的"理解 fig/ax 机制"部分

### 需要完整理解？

1. 从 [README.md](utils/README.md) 的"关键概念"章节开始
2. 学习三种使用模式（自动创建、追加绘制、子图）
3. 通过"实际应用示例"加深理解

### 查询特定方法？

- 使用 [API.md](utils/API.md) 查找方法签名和参数
- 或在 [README.md](utils/README.md) 的"常用方法参考"快速查找

### 遇到问题？

1. 查看 [README.md](utils/README.md) 的"常见问题"
2. 查看 [REFACTOR.md](utils/REFACTOR.md) 的"常见陷阱"
3. 检查 [QUICKSTART.md](utils/QUICKSTART.md) 中的错误排查表

---

## 关键概念速览

### fig 和 ax

- **fig** (Figure): 整个图表窗口
- **ax** (Axes): 在 fig 内绘制的坐标系区域

### 三种使用模式

| 模式 | 用途 | 参数 |
|------|------|------|
| **自动创建** | 单个独立图表 | 无需 `fig`, `ax` |
| **追加绘制** | 同图表多线对比 | 传入相同的 `fig`, `ax`, 设 `add_fig=False` |
| **子图布局** | 多个相关图表 | 用 `plt.subplots()` 创建，设 `add_fig=False` |

### show() 方法

启动 Tkinter GUI 交互式浏览 `lib.figs` 中的所有图表。

---

## 最常用的代码模板

### 模板 1: 单个图表（最简单）

```python
from src.visualize_tools.utils import PlotLib
import numpy as np

lib = PlotLib()
lib.plot(y=[1, 2, 3, 4], title='My Chart')
lib.show()
```

### 模板 2: 同图表多线对比

```python
lib = PlotLib()

fig, ax = lib.plot(y=[1, 2, 3, 4], legend='Line 1', color='blue')
fig, ax = lib.plot(y=[4, 3, 2, 1], legend='Line 2', color='red',
                   fig=fig, ax=ax, add_fig=False)

ax.legend()
lib.show()
```

### 模板 3: 多子图（2x2 布局）

```python
import matplotlib.pyplot as plt
from src.visualize_tools.utils import PlotLib

lib = PlotLib()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, ax in enumerate(axes.flat):
    lib.plot(y=np.random.randn(100), title=f'Plot {i}',
             ax=ax, fig=fig, add_fig=False)

lib.figs.append(fig)
lib.show()
```

---

## 快速参考

### 常用方法

- **plot()** - 线图
- **scatter()** - 散点图
- **hist()** - 直方图
- **rose_hist()** - 玫瑰图（风向分布）
- **plots()** - 多子图线图
- **scatters()** - 多子图散点图
- **scatter_3d()** - 3D 散点图
- **rose_scatter()** - 极坐标散点图
- **animate()** - 动画
- **show()** - 交互式显示

### 常用参数

| 参数 | 说明 |
|------|------|
| `y`, `x` | 必需的数据 |
| `title`, `xlabel`, `ylabel` | 标签 |
| `color`, `alpha`, `style` | 样式 |
| `fig`, `ax` | 现有的 Figure/Axes（追加绘制时） |
| `add_fig` | 是否自动添加到 `self.figs` |
| `xlim`, `ylim` | 轴范围 |
| `legend` | 图例标签 |

---

## 文件结构

```
docs/visualize_tools/
├── utils/
│   ├── README.md          # 完整参考文档
│   ├── QUICKSTART.md      # 快速入门
│   ├── API.md             # API 参考
│   ├── REFACTOR.md        # 设计决策
│   └── INDEX.md           # 本文件
```

---

## 模块信息

- **位置**: `src/visualize_tools/utils.py`
- **主要类**: `PlotLib`（绘图）、`ChartApp`（GUI 显示）
- **依赖**: matplotlib, numpy, pandas, tkinter, PIL
- **最后更新**: 2026-02-25

---

## 相关链接

- [统计数据工作流文档](../../../docs/statistics/workflow/README.md)
- [数据处理模块文档](../../../docs/data_processer/README.md)
