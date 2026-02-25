# visualize_tools 模块文档

可视化工具模块，提供各种数据绘制和图表展示的功能。

## 模块结构

```
src/visualize_tools/
├── utils.py          # 主要绘图工具（PlotLib 类）
└── __init__.py
```

## 快速开始

```python
from src.visualize_tools.utils import PlotLib

lib = PlotLib()
lib.plot(y=[1, 2, 3, 4], title='Simple Plot')
lib.show()
```

## 文档导航

### 📚 主要文档

- **[utils 模块完整文档](utils/INDEX.md)** - 从这里开始
  - 包含快速入门、完整参考和设计说明

### 📖 详细文档

如果你想深入了解 `utils.py` 模块：

- [README.md](utils/README.md) - 完整参考（包括 fig/ax 机制详解）
- [QUICKSTART.md](utils/QUICKSTART.md) - 快速入门指南
- [API.md](utils/API.md) - 完整 API 参考
- [REFACTOR.md](utils/REFACTOR.md) - 设计决策文档

## 核心概念

### PlotLib 类

主要的绘图工具类，支持多种图表类型和灵活的绘制选项。

**关键特性**:
- 统一的 API 接口
- 支持 `fig`/`ax` 参数进行高级绘制
- 自动图表管理
- 交互式 GUI 显示（`show()` 方法）

### 常用方法

- `plot()` - 线图
- `scatter()` - 散点图
- `hist()` - 直方图
- `rose_hist()` - 玫瑰图
- `show()` - 交互式显示

## 常见用途

### 绘制简单线图

```python
lib = PlotLib()
lib.plot(y=[1, 2, 3, 4], title='Data')
lib.show()
```

### 对比多条线

```python
lib = PlotLib()
fig, ax = lib.plot(y=[1, 2, 3, 4], legend='Line 1')
fig, ax = lib.plot(y=[4, 3, 2, 1], legend='Line 2',
                   fig=fig, ax=ax, add_fig=False)
ax.legend()
lib.show()
```

### 创建多子图

```python
import matplotlib.pyplot as plt

lib = PlotLib()
fig, axes = plt.subplots(2, 2)

for i, ax in enumerate(axes.flat):
    lib.plot(y=data[i], title=f'Plot {i}',
             ax=ax, fig=fig, add_fig=False)

lib.figs.append(fig)
lib.show()
```

## 学习建议

1. **新手**: 先读 [QUICKSTART.md](utils/QUICKSTART.md)
2. **进阶**: 查看 [README.md](utils/README.md) 的使用示例
3. **完全理解**: 阅读 [REFACTOR.md](utils/REFACTOR.md) 的设计原理

## 重要提示

### 关于 fig 和 ax 参数

这是 `PlotLib` 最强大且最常用的特性：

- **fig** (Figure): 整个图表窗口
- **ax** (Axes): 绘制区域

通过传入相同的 `fig` 和 `ax`，可以在同一图表上绘制多条线或多个数据集。

详见 [README.md](utils/README.md) 的"fig/ax 机制"章节。

### 关于 add_fig 参数

控制是否自动将图表添加到管理列表：

- 首次创建新 Figure: `add_fig=True`（默认）
- 追加绘制: `add_fig=False`
- 最后一次: `add_fig=False`，然后手动 `lib.figs.append(fig)`

详见 [REFACTOR.md](utils/REFACTOR.md) 的"add_fig 参数详解"。

## 常见问题

**Q: 多条线显示不出来？**  
A: 检查是否设置了 `ax.legend()` 并且设置了 `add_fig=False`

**Q: 图表显示重复？**  
A: 检查是否多次设置了 `add_fig=True`

**Q: 如何在 Jupyter 中使用？**  
A: 添加 `%matplotlib inline`，无需调用 `show()`

详见 [README.md](utils/README.md) 的"常见问题"。

## 版本信息

- **最后更新**: 2026-02-25
- **Python 版本**: 3.7+
- **依赖**: matplotlib, numpy, pandas, tkinter, PIL

---

## 相关文档

- [数据处理模块](../../data_processer/README.md)
- [统计工作流](../statistics/workflow/README.md)
