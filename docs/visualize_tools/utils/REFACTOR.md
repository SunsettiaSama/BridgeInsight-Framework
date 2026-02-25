# fig/ax 机制设计说明

## 设计目标

`PlotLib` 的 `fig` 和 `ax` 参数机制旨在解决以下问题：

1. **单图表场景的简洁性**: 不需要手动管理 `fig` 和 `ax`
2. **多元素场景的灵活性**: 支持在同一坐标系上绘制多条线或多个数据集
3. **多子图支持**: 与 Matplotlib 的 `subplots()` 无缝集成
4. **图表管理**: 自动跟踪创建的图表，方便后续显示或保存

---

## 核心机制

### 默认行为（无参数）

```python
fig, ax = lib.plot(y=[1, 2, 3, 4])
```

**发生的事**:
1. 创建新的 `Figure` 对象
2. 在 Figure 上创建新的 `Axes` 子图
3. 在 Axes 上绘制数据
4. 将 Figure 添加到 `self.figs`（因为 `add_fig=True`）

**使用场景**: 单个独立的图表

---

### 追加绘制（传入 fig/ax）

```python
fig, ax = lib.plot(y1, legend='Line 1')
fig, ax = lib.plot(y2, fig=fig, ax=ax, add_fig=False, legend='Line 2')
```

**发生的事**:
1. 第一次调用：创建 `fig` 和 `ax`，绘制，添加到 `self.figs`
2. 第二次调用：**使用现有的 `fig` 和 `ax`**，绘制新数据，**不重复添加**（`add_fig=False`）

**关键点**:
- 所有数据绘制在**同一个 Axes 上**
- 需要手动 `ax.legend()` 显示图例
- 必须设 `add_fig=False` 避免重复添加

**使用场景**: 对比多条线的关系

---

### 子图场景（与 plt.subplots() 配合）

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2)

for i, ax in enumerate(axes.flat):
    lib.plot(y=data[i], title=f'Chart {i}', ax=ax, fig=fig, add_fig=False)

lib.figs.append(fig)
```

**发生的事**:
1. 使用 `plt.subplots()` 创建预定义的 `fig` 和 `axes` 数组
2. 遍历每个 `ax`，在其上绘制不同的数据
3. 所有数据共享同一个 Figure（这是一个多子图的单一 Figure）
4. 手动添加 `fig` 到 `self.figs`

**关键点**:
- 最后设 `add_fig=False`，手动 `lib.figs.append(fig)`
- 每个 `ax` 是一个独立的绘制区域
- Figure 包含所有子图

**使用场景**: 相关数据的矩阵展示

---

## add_fig 参数详解

### 作用

控制是否自动将绘制结果添加到 `self.figs` 列表。

### 规则

| 调用方式 | add_fig 值 | 说明 |
|---------|-----------|------|
| 首次创建新 Figure | True（默认）| 创建并添加 |
| 追加绘制到已有 Figure | False | 避免重复添加 |
| 最后一次追加绘制 | False | 手动添加整个 Figure |

### 常见错误

**错误 1: 重复添加同一图表**

```python
# 错误：两次都 add_fig=True
fig, ax = lib.plot(y1)                           # 添加 fig（第 1 次）
fig, ax = lib.plot(y2, fig=fig, ax=ax)          # 又添加 fig（第 2 次）

# 结果：self.figs 中有两个相同的 fig
print(len(lib.figs))  # 2（应该是 1）
```

**正确做法**:

```python
fig, ax = lib.plot(y1)                           # add_fig=True（默认）
fig, ax = lib.plot(y2, fig=fig, ax=ax, add_fig=False)  # 不再添加
print(len(lib.figs))  # 1（正确）
```

**错误 2: 子图全部设 add_fig=True**

```python
# 错误
fig, axes = plt.subplots(2, 2)
lib.plot(y=data[0], ax=axes[0, 0], fig=fig, add_fig=True)  # +1
lib.plot(y=data[1], ax=axes[0, 1], fig=fig, add_fig=True)  # +1
lib.plot(y=data[2], ax=axes[1, 0], fig=fig, add_fig=True)  # +1
lib.plot(y=data[3], ax=axes[1, 1], fig=fig, add_fig=True)  # +1

print(len(lib.figs))  # 4（应该是 1）
```

**正确做法**:

```python
fig, axes = plt.subplots(2, 2)
lib.plot(y=data[0], ax=axes[0, 0], fig=fig, add_fig=False)
lib.plot(y=data[1], ax=axes[0, 1], fig=fig, add_fig=False)
lib.plot(y=data[2], ax=axes[1, 0], fig=fig, add_fig=False)
lib.plot(y=data[3], ax=axes[1, 1], fig=fig, add_fig=False)

lib.figs.append(fig)  # 手动添加一次
print(len(lib.figs))  # 1（正确）
```

---

## 设计决策

### 为什么有 fig/ax 参数？

**问题**: Matplotlib 的 API 很低级，每次绘制都需要显式管理 Figure 和 Axes

**解决**: `PlotLib` 提供了默认行为（自动创建），同时允许高级用户传入自己的 `fig`/`ax` 来实现更复杂的布局

### 为什么需要 add_fig 参数？

**问题**: 不同场景下，用户希望在不同的时机将图表添加到管理列表中

**解决**: 提供 `add_fig` 参数让用户显式控制何时添加，避免重复添加

### 为什么所有方法都返回 (fig, ax)?

**好处**:
1. 链式调用：`fig, ax = lib.plot(...); fig, ax = lib.plot(..., fig=fig, ax=ax)`
2. 一致性：所有方法返回相同的结构，容易学习
3. 灵活性：用户可以继续修改返回的 `fig` 和 `ax`

---

## 最佳实践

### 场景 1: 单图表（最简单）

```python
lib.plot(y=data, title='My Chart')
lib.show()
```

**要点**: 默认行为，无需考虑 `fig`/`ax`

---

### 场景 2: 同图表多线（对比）

```python
fig, ax = lib.plot(y=signal1, legend='Signal 1', color='blue')
fig, ax = lib.plot(y=signal2, fig=fig, ax=ax, legend='Signal 2', 
                   color='red', add_fig=False)
ax.legend()
lib.show()
```

**要点**:
- 传入 `fig` 和 `ax`
- 设 `add_fig=False`
- 调用 `ax.legend()` 显示图例

---

### 场景 3: 多子图（矩阵）

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, ax in enumerate(axes.flat):
    lib.plot(y=data_list[i], title=f'Plot {i}', 
             ax=ax, fig=fig, add_fig=False)

lib.figs.append(fig)
lib.show()
```

**要点**:
- 用 `plt.subplots()` 预创建布局
- 所有 `lib.plot()` 调用设 `add_fig=False`
- 最后手动 `lib.figs.append(fig)`

---

### 场景 4: 复杂布局

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0:2, 0:2])  # 左上大图
ax2 = fig.add_subplot(gs[0, 2])      # 右上小图
ax3 = fig.add_subplot(gs[1, 2])      # 右中小图
ax4 = fig.add_subplot(gs[2, :])      # 下方全宽

lib.plot(y=data1, title='Main', ax=ax1, fig=fig, add_fig=False)
lib.plot(y=data2, title='1', ax=ax2, fig=fig, add_fig=False)
lib.plot(y=data3, title='2', ax=ax3, fig=fig, add_fig=False)
lib.plot(y=data4, title='Bottom', ax=ax4, fig=fig, add_fig=False)

lib.figs.append(fig)
lib.show()
```

---

## 集成 show() 机制

### show() 的作用

`show()` 方法遍历 `self.figs` 中的所有图表，通过 Tkinter GUI 提供交互式浏览：

```python
def show(self):
    tk = Tk()
    app = ChartApp(tk, self.figs)  # 传入所有图表
    tk.mainloop()
    return
```

### 图表如何进入 self.figs？

两种方式：

1. **自动添加**（`add_fig=True`）：
   ```python
   lib.plot(y=data)  # 自动添加到 self.figs
   ```

2. **手动添加**（`add_fig=False`）：
   ```python
   fig, ax = lib.plot(y=data, add_fig=False)
   lib.figs.append(fig)  # 手动添加
   ```

### show() 何时调用？

在完成所有绘制并确认所有图表都在 `self.figs` 中后调用：

```python
lib = PlotLib()

# 添加多个图表
lib.plot(y=data1)
lib.plot(y=data2)

fig, ax = lib.plot(y=data3, add_fig=False)  # 需要手动添加
lib.figs.append(fig)

# 现在所有图表都在 self.figs 中
lib.show()  # 显示所有图表
```

---

## 总结

| 需求 | 操作 | 说明 |
|------|------|------|
| 单个图表 | `lib.plot(...)` | 默认行为，自动添加 |
| 多线同图 | 传入 `fig, ax`，设 `add_fig=False` | 需要 `ax.legend()` |
| 多子图 | 用 `plt.subplots()`，设 `add_fig=False`，最后手动添加 | 最灵活 |
| 显示所有 | `lib.show()` | 遍历 `self.figs` 显示 |

---

## 常见陷阱

🚫 **错误**: 在追加绘制时忘记设 `add_fig=False`  
✅ **正确**: 后续调用都设 `add_fig=False`

🚫 **错误**: 多次手动 `lib.figs.append(fig)`  
✅ **正确**: 只在最后追加绘制后添加一次

🚫 **错误**: 混淆 `fig` 和 `ax` 的概念  
✅ **正确**: `fig` 是容器，`ax` 是绘制区域

🚫 **错误**: 调用 `show()` 前忘记添加图表  
✅ **正确**: 确保所有图表都在 `self.figs` 中
