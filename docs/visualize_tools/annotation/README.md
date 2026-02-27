# 振动数据人工标注系统 (annotation.py) 使用文档

## 模块概述

`src/visualize_tools/annotation.py` 是一个完整的振动数据人工标注系统，集成了数据获取、图像生成、交互式标注和结果保存于一体。该系统能够自动从工作流获取极端振动窗口数据，生成时域和频域的上下子图，并提供灵活的多维度筛选机制和增量保存功能。

### 核心特性

- **自动化数据流**：从 `vibration_io_process` 工作流直接获取 metadata 和极端窗口数据
- **双轴图像显示**：时域波形（上）+ 频域谱（下）的组合显示
- **多维度筛选**：支持模式、日期范围、传感器ID、RMS 阈值、最大振幅阈值等多种筛选
- **智能图像缓冲**：LRU 缓冲机制防止大数据集内存溢出
- **增量保存**：支持数据追加和更新，只保存有意义的标注
- **Tkinter GUI**：提供友好的交互式标注界面，支持快捷键操作

---

## 快速开始

### 最简单的使用方式

```python
from src.visualize_tools.annotation import AnnotationGUI

# 创建标注系统实例
app = AnnotationGUI()

# 启动 GUI（会弹出模式选择对话框）
app.run()
```

### 执行示例

项目根目录下有测试脚本 `test_annotation.py`：

```bash
python test_annotation.py
```

---

## 工作流程

### 1. 启动应用

```python
app = AnnotationGUI()
app.run()
```

### 2. 配置对话框

系统会自动弹出一个配置对话框，用户需要设置：

#### 模式选择

- **正常模式**：加载所有60秒窗口（可能很多）
- **极端模式**（推荐）：只加载极端振动窗口
- **超级极端模式**：待0.25%分位数据实现后启用

#### 日期范围筛选（可选）

格式：`MM/DD`（例如 `09/15`）

- 起始日期：指定开始日期
- 结束日期：指定结束日期
- 留空则不筛选

#### 传感器筛选（可选）

支持单个或多个传感器：

- **单个**：`ST-VIC-C18-101-01`
- **多个**：`ST-VIC-C18-101-01, ST-VIC-C18-102-01, ST-VIC-C18-401-01`
- 留空则不筛选

#### 阈值过滤（可选）

- **RMS 阈值**（m/s²）：过滤 RMS 值小于此值的样本
- **最大振幅阈值**（m/s²）：过滤最大振幅小于此值的样本

### 3. 标注数据

主窗口会依次显示通过筛选的振动窗口：

- **上子图**：时域波形（加速度 vs 时间）
- **下子图**：频域谱（PSD vs 频率）
- **输入框**：输入标注（如 `0`、`1`、`2` 等标签）

### 4. 保存结果

按 `Ctrl+S` 或点击"保存结果"按钮保存标注数据。

---

## 详细功能说明

### 数据提供者 (AnnotationDataProvider)

负责从工作流获取数据。

```python
from src.visualize_tools.annotation import AnnotationDataProvider

provider = AnnotationDataProvider(use_cache=True, mode='extreme')
windows = provider.fetch_metadata_and_extreme_windows()

# 获取已加载的窗口
all_windows = provider.get_extreme_windows()

# 获取特定索引的窗口
window = provider.get_window_by_index(0)
```

#### 模式说明

| 模式 | 说明 |
|------|------|
| `MODE_NORMAL` | 加载所有60秒窗口 |
| `MODE_EXTREME` | 只加载极端RMS窗口（推荐） |
| `MODE_SUPER_EXTREME` | 等待0.25%分位数据实现 |

---

### 图像生成器 (AnnotationFigureGenerator)

负责生成时域+频域的上下子图。

```python
from src.visualize_tools.annotation import AnnotationFigureGenerator

generator = AnnotationFigureGenerator(fs=50.0)

# 生成图像
fig, error_msg = generator.generate_figure(window_info)

if fig is not None:
    # 显示图像或保存
    fig.savefig('output.png')
else:
    print(f"生成失败: {error_msg}")
```

#### 图像参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fs` | 50.0 | 采样频率（Hz） |
| `FIG_SIZE` | (12, 8) | 图像大小（英寸） |
| `NFFT` | 2048 | FFT 点数 |

---

### 图像缓冲 (FigureCache)

使用 LRU（最近最少使用）缓冲策略，防止大数据集内存溢出。

```python
from src.visualize_tools.annotation import FigureCache

cache = FigureCache(max_size=20)

# 存储图像
cache.put(window_idx, fig)

# 获取图像
fig = cache.get(window_idx)

# 清空缓冲
cache.clear()
```

#### 工作原理

- 最多缓冲 20 张图像
- 访问图像时自动移到最近使用位置
- 超过容量时自动删除最不常用的图像
- 防止 matplotlib 句柄占用过多内存

---

### 主界面 (AnnotationWindowGUI)

完整的标注界面实现。

#### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `→` 或 `Return` | 下一个窗口 |
| `←` | 上一个窗口 |
| `Ctrl+S` | 保存结果 |

#### 状态栏

显示当前标注进度：
```
窗口 15/250 (总计1000) - ST-VIC-C18-101-01 @ 9/15 01:00
```

---

## 筛选机制详解

### 1. 模式筛选

```python
# 在配置中选择：
# - 正常模式：所有窗口
# - 极端模式：仅极端窗口
# - 超级极端模式：待实现
```

### 2. 日期范围筛选

只显示在指定日期范围内的窗口：

```
起始日期: 09/01
结束日期: 09/30
```

### 3. 传感器筛选

只显示指定传感器的数据：

```
单个: ST-VIC-C18-101-01

多个: ST-VIC-C18-101-01, ST-VIC-C18-102-01, ST-VIC-C18-401-01
```

### 4. RMS 阈值

过滤 RMS 值小于阈值的样本：

```python
RMS值 = sqrt(mean(数据²))
```

### 5. 最大振幅阈值

过滤最大振幅小于阈值的样本：

```python
最大振幅 = max(|数据|)
```

### 组合筛选

所有筛选条件同时应用，只有全部通过的窗口才会显示。

---

## 保存机制

### 保存路径

默认保存路径由常量定义：

```python
DEFAULT_ANNOTATION_RESULT_PATH = 
    "../../annotation_results/annotation_results.json"
```

自动创建目录（如不存在）。

### 保存特性

1. **智能合并**：如果文件已存在，新数据与旧数据合并
2. **去重更新**：已有数据更新，新数据追加
3. **只保存有意义的标注**：
   - 跳过空字符串
   - 跳过未输入的窗口
   - 只保存明确的标签（如 `0`、`1`、`2`）

### 保存流程

```python
# 1. 检查文件是否存在
# 2. 加载已有数据
# 3. 构建新数据（仅包含非空标注）
# 4. 合并新旧数据
# 5. 按 file_path 和 window_index 排序
# 6. 保存到 JSON 文件
# 7. 显示统计信息
```

### 保存统计

保存后显示：

```
标注结果已保存
新增标注: 15 个
更新标注: 3 个
总保存数: 250 个
保存路径: ...
```

### 文件格式

```json
[
  {
    "metadata": {...},
    "window_index": 5,
    "sensor_id": "ST-VIC-C18-101-01",
    "time": "9/1 01:00",
    "file_path": "path/to/file.VIC",
    "annotation": "0"
  },
  ...
]
```

---

## 完整使用示例

### 示例 1：基础标注流程

```python
from src.visualize_tools.annotation import AnnotationGUI

# 创建实例
app = AnnotationGUI()

# 启动（会弹出配置对话框）
app.run()

# 配置对话框中设置：
# - 模式：极端模式
# - 日期：09/01 - 09/30
# - 传感器：ST-VIC-C18-101-01
# - RMS阈值: 0.002
# - 最大振幅阈值: 0.5

# 然后在主窗口中：
# - 按 ← / → 切换窗口
# - 在输入框输入标注（0, 1, 2 等）
# - 按 Ctrl+S 保存
```

### 示例 2：多个传感器标注

```python
from src.visualize_tools.annotation import AnnotationGUI

app = AnnotationGUI()
app.run()

# 在配置中设置：
# 传感器: ST-VIC-C18-101-01, ST-VIC-C18-102-01, ST-VIC-C18-401-01
```

### 示例 3：自定义保存路径

```python
from src.visualize_tools.annotation import AnnotationGUI

app = AnnotationGUI()

# 指定自定义保存路径
app.run(save_result_path="/path/to/custom_results.json")
```

---

## 配置常量

所有重要常量在文件顶部定义：

```python
# 绘图配置
FONT_SIZE = 12
LABEL_FONT_SIZE = 14
FIG_SIZE = (12, 8)
NFFT = 2048

# 采样参数
FS = 50.0              # 采样频率
WINDOW_SIZE = 3000     # 窗口点数（60秒 @ 50Hz）

# 缓冲配置
FIGURE_CACHE_SIZE = 20 # 最多缓冲20张图像

# 保存路径
DEFAULT_ANNOTATION_RESULT_PATH = 
    "../../annotation_results/annotation_results.json"
```

---

## 数据流向

```
工作流 (vibration_io_process)
    ↓
metadata + 极端窗口索引
    ↓
AnnotationDataProvider (获取数据)
    ↓
极端窗口数据
    ↓
AnnotationFigureGenerator (生成图像)
    ↓
时域 + 频域图像
    ↓
AnnotationWindowGUI (显示 + 标注)
    ↓
用户标注
    ↓
save_results() (保存)
    ↓
annotation_results.json
```

---

## 常见问题

### Q: 如何跳过不感兴趣的窗口？

A: 不输入标注（留空），保存时只会保存有明确标注的窗口。

---

### Q: 能否在标注途中修改阈值重新筛选？

A: 目前不支持。需要关闭并重新启动应用，选择新的阈值。

---

### Q: 如何查看已保存的标注？

A: 打开 `annotation_results.json` 文件（JSON 格式）。

```bash
# 查看文件
cat annotation_results.json

# 或用 Python 加载
import json
with open('annotation_results.json') as f:
    data = json.load(f)
    print(f"总共 {len(data)} 条标注")
```

---

### Q: 如何批量修改标注？

A: 关闭应用，直接编辑 JSON 文件，然后重新启动。系统会自动加载已有的标注。

---

### Q: 内存占用很大怎么办？

A: 系统已集成图像缓冲机制（最多20张）。如果仍然过大：

1. 使用 `NORMAL` 模式而非 `EXTREME` 模式
2. 增加 RMS 阈值过滤
3. 指定特定传感器而不是全部加载

---

### Q: 如何导出为其他格式（CSV、Excel）？

A: 标注结果保存为 JSON，可以手动转换：

```python
import json
import pandas as pd

with open('annotation_results.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv('annotations.csv', index=False)
df.to_excel('annotations.xlsx', index=False)
```

---

## 模块依赖

| 依赖 | 说明 |
|-----|------|
| `tkinter` | GUI 框架 |
| `matplotlib` | 图像生成 |
| `numpy` | 数值计算 |
| `scipy.signal` | 信号处理（Welch 方法） |
| `json` | 数据持久化 |
| `datetime` | 日期处理 |

---

## 性能参考

| 操作 | 耗时 |
|-----|------|
| 加载 1000 个窗口 | ~10-30秒 |
| 生成单张图像 | ~100-200ms |
| 显示图像（从缓冲） | <50ms |
| 显示图像（生成）| 100-200ms |
| 保存 1000 条标注 | ~1秒 |

---

## 架构设计

### 模块组件

```
annotation.py
├── FigureCache (LRU图像缓冲)
├── AnnotationDataProvider (数据获取)
├── AnnotationFigureGenerator (图像生成)
├── AnnotationWindowGUI (主界面)
└── AnnotationGUI (入口类)
```

### 设计原则

1. **分离关注点**：数据、图像、UI 分离
2. **内存优化**：使用 LRU 缓冲防止溢出
3. **增量保存**：支持多次启动和追加
4. **灵活筛选**：多维度组合条件
5. **用户友好**：快捷键 + 配置对话框

---

## 最佳实践

1. **使用极端模式**：减少窗口数量，加快处理
2. **设置合理阈值**：过滤掉不感兴趣的数据
3. **指定传感器**：逐个传感器标注，便于管理
4. **定期保存**：按 `Ctrl+S` 定期保存进度
5. **标注规范**：使用统一的标签体系（如 0/1/2）

---

## 版本信息

- **最后更新**: 2026-02-26
- **模块位置**: `src/visualize_tools/annotation.py`
- **文档位置**: `docs/visualize_tools/annotation/README.md`
- **测试脚本**: `test_annotation.py`

---

## 相关模块

- `src.data_processer.statistics.vibration_io_process`: 数据处理工作流
- `src.data_processer.io_unpacker`: VIC 文件读取
- `src.visualize_tools.utils`: 可视化工具库

---

## 反馈与改进

如有问题或改进建议，请参考以下地方：

1. 修改默认常量（文件顶部）
2. 扩展筛选条件（`_build_filtered_indices` 方法）
3. 自定义图像样式（`AnnotationFigureGenerator` 类）
