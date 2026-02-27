# 振动数据标注系统使用说明

## 概述

新的标注系统基于 `vibration_io_process` 工作流，能够自动获取极端振动窗口数据，并提供图形化的人工标注界面。

## 系统架构

### 核心模块

1. **annotation_data_provider.py** - 数据提供者
   - 从工作流获取metadata
   - 过滤并加载极端窗口数据
   - 提供数据访问接口

2. **annotation_figure_generator.py** - 图像生成器
   - 生成时域和频域上下子图
   - 使用Welch方法计算功率谱密度
   - 支持自定义绘图参数

3. **annotation_gui.py** - 用户界面
   - 提供交互式标注界面
   - 支持快捷键操作
   - 保存标注结果为JSON格式

## 使用方法

### 1. 基本启动

```python
from src.visualize_tools.annotation_gui import AnnotationGUI

# 启动标注系统
app = AnnotationGUI()
app.run(save_result_path="annotation_results.json")
```

### 2. 命令行启动

```bash
python test_annotation.py
```

## 操作说明

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `←` (Left) | 上一个窗口 |
| `→` (Right) 或 `Enter` | 下一个窗口 |
| `Ctrl+S` | 保存结果 |

### 界面元素

1. **顶部状态栏**
   - 显示当前窗口位置 (如 "窗口 1/150")
   - 显示传感器ID和时间信息

2. **中间图像区域**
   - 上子图：时域波形
   - 下子图：频域谱（0-25Hz）

3. **底部标注区**
   - 输入框：输入标注信息
   - 导航按钮：上一个/下一个窗口

## 数据格式

### 输入数据

工作流返回的metadata结构：
```python
{
    'sensor_id': str,              # 传感器ID
    'month': int,                  # 月份
    'day': int,                    # 日期
    'hour': int,                   # 小时
    'file_path': str,              # 文件路径
    'actual_length': int,          # 实际长度
    'missing_rate': float,         # 缺失率
    'extreme_rms_indices': list    # 极端窗口索引
}
```

### 输出数据 (annotation_results.json)

```json
[
  {
    "metadata": {...},           # 原始metadata
    "window_index": 5,
    "sensor_id": "ST-VIC-C18-101-01",
    "time": "9/1 01:00",
    "file_path": "path/to/file.VIC",
    "annotation": "正常振动"      # 用户标注
  },
  ...
]
```

## 常见配置

### 修改图像显示范围

在 `annotation_figure_generator.py` 中修改：

```python
# 修改频率范围
freq_limit = 25  # 更改为所需的Hz值

# 修改图像大小
FIG_SIZE = (12, 8)  # (宽, 高)

# 修改字体大小
LABEL_FONT_SIZE = 14
```

### 修改保存路径

```python
app.run(save_result_path="custom/path/results.json")
```

## 性能参考

- 加载数据：取决于文件数量和工作流缓存状态
- 图像生成：每个窗口约100-200ms
- 实时保存：每次保存 <100ms

## 常见问题

### Q: 如何断点续标？
A: 系统会自动加载已有的 `annotation_results.json` 文件，继续标注未完成的窗口。

### Q: 如何修改标注的窗口？
A: 直接在输入框中修改标注内容，系统会自动保存到内存中。

### Q: 如何重新开始？
A: 删除 `annotation_results.json` 文件，重新启动系统。

### Q: 频域图像中的单位是什么？
A: PSD单位为 $(m/s^2)^2/Hz$，表示功率谱密度。

## 技术细节

### 采样频率和窗口配置

```python
FS = 50.0              # 采样频率
WINDOW_SIZE = 3000     # 60秒窗口对应的采样点数
TIME_WINDOW = 60       # 时间窗口（秒）
```

### 频域分析参数

```python
NFFT = 2048           # FFT点数
使用Welch方法进行PSD计算
nperseg = NFFT/2      # 每个段的点数
noverlap = NFFT/4     # 重叠长度
```

## 扩展功能建议

1. 支持多个标注者的对比标注
2. 添加标注置信度选择
3. 支持导出为Excel格式
4. 添加自动标注建议功能
5. 支持图像缩放和平移

---

**最后更新**: 2026-02-26
**系统版本**: 1.0
