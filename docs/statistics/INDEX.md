# 统计数据处理模块（Statistics）- 文档导航

欢迎使用 `statistics` 统计数据处理模块的完整文档导航！

本模块是振动特性研究系统的核心数据处理管道，负责从原始振动和风速数据中提取、处理和统计特征信息。

---

## 📋 模块结构概览

```
src/data_processer/statistics/
├── workflow.py                           # 主入口：集成所有子模块工作流
├── vibration_io_process/                 # 振动数据处理子模块
│   ├── __init__.py
│   ├── workflow.py                       # 振动数据工作流
│   ├── step0_get_vib_data.py            # Step 0: 文件发现
│   ├── step1_lackness_filter.py          # Step 1: 缺失率筛选
│   └── step2_rms_statistics.py           # Step 2: RMS统计与极端识别
└── wind_data_io_process/                 # 风数据处理子模块
    ├── __init__.py
    ├── workflow.py                       # 风数据工作流
    ├── step0_get_wind_data.py           # Step 0: 文件发现
    ├── step1_timestamp_align.py          # Step 1: 时间戳对齐
    └── step2_extreme_filter.py           # Step 2: 极端振动对应风数据筛选
```

---

## 🎯 快速导航 - 按用途查找

### 我想快速开始使用数据处理

**→ 从这里开始：** [workflow.py 主入口](#主入口-workflowpy)

快速示例：
```python
from src.data_processer.statistics.workflow import get_data_pairs

# 一行代码获取处理后的振动和风数据对
data_pairs = get_data_pairs(wind_sensor_id='ST-UAN-G04-001-01')
```

---

### 我想了解振动数据处理过程

**→ 查看：** [vibration_io_process 子模块](#vibration_io_process-振动数据处理)

工作流：文件发现 → 缺失率筛选 → RMS统计与极端识别

---

### 我想了解风数据处理过程

**→ 查看：** [wind_data_io_process 子模块](#wind_data_io_process-风数据处理)

工作流：文件发现 → 时间戳对齐 → 极端振动对应风数据筛选

---

### 我需要完整的API参考

**→ 查看：** [API参考表](#api-参考表)

---

## 📚 完整模块索引

### 主入口 `workflow.py`

**位置：** `src/data_processer/statistics/workflow.py`

**功能：** 
- 集成所有子模块的完整工作流
- 提供统一的数据处理入口
- 支持灵活的数据切分和筛选策略
- 一次性返回对齐的振动和风数据对

**主要函数：**

| 函数 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `get_data_pairs()` | **核心函数** - 批量提取振动和风数据对 | wind_sensor_id, vib_sensor_id, 切分参数 | 数据对列表 |

**关键参数：**
- `wind_sensor_id`：风传感器ID（必需）
- `vib_sensor_id`：振动传感器ID（可选，None表示所有）
- `enable_extreme_window`：是否筛选极端振动窗口
- `window_duration_minutes`：常规窗口切分时长（分钟）
- `use_multiprocess`：是否使用多进程加速处理

**详细文档：** 📖 [workflow 完整指南](./workflow/README.md)

---

### vibration_io_process 振动数据处理

**位置：** `src/data_processer/statistics/vibration_io_process/`

**功能：** 处理振动数据文件，提取元数据，识别极端振动

**工作流程：**

#### **Step 0: 获取所有振动文件 (`step0_get_vib_data.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 从配置的目录递归收集所有振动数据文件 |
| 输入 | 数据根目录 + 目标传感器ID列表 |
| 输出 | 文件路径列表 |
| 文件格式 | `.VIC` 后缀的二进制振动数据文件 |
| 主函数 | `get_all_vibration_files()` |

**示例：**
```python
from src.data_processer.statistics.vibration_io_process.step0_get_vib_data import get_all_vibration_files

all_files = get_all_vibration_files()
print(f"发现 {len(all_files)} 个振动文件")
```

---

#### **Step 1: 缺失率筛选 (`step1_lackness_filter.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 基于数据缺失率过滤低质量文件，保留高质量数据 |
| 输入 | 文件路径列表 + 缺失率阈值（默认5%） |
| 输出 | 筛选后的文件路径 + 统计信息 |
| 缺失率计算 | `missing_rate = 1 - (actual_length / expected_length)` |
| 预期长度 | 180000 采样点（50Hz × 3600秒） |
| 主函数 | `run_lackness_filter()` |
| 处理方式 | 多进程并行计算 |

**示例：**
```python
from src.data_processer.statistics.vibration_io_process.step1_lackness_filter import run_lackness_filter

filtered_paths, stats = run_lackness_filter(all_files, threshold=0.05)
print(f"筛选后保留 {len(filtered_paths)} 个文件")
print(f"平均缺失率: {stats['all_missing_rates'].mean()*100:.2f}%")
```

**返回统计信息包含：**
- `all_lengths`：所有文件的实际采样点数
- `all_missing_rates`：所有文件的缺失率数组
- `filtered_indices`：筛选通过的文件索引

---

#### **Step 2: RMS统计与极端识别 (`step2_rms_statistics.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 计算RMS值，识别极端振动窗口 |
| 输入 | 文件路径列表 + 采样频率 + 时间窗口 |
| 输出 | 元数据列表 + RMS统计信息 |
| 采样频率 | 50 Hz（硬编码） |
| 时间窗口 | 60 秒（硬编码） |
| 极端判定 | RMS值 ≥ 95%分位数 |
| 主函数 | `run_rms_statistics()` |
| 处理方式 | 多进程并行计算 |

**极端窗口识别原理：**
- 将每个文件按60秒时间窗口切分
- 计算每个窗口的RMS值
- 对所有RMS值计算95%分位数
- 标记超过分位数的窗口索引

**示例：**
```python
from src.data_processer.statistics.vibration_io_process.step2_rms_statistics import run_rms_statistics

file_paths, stats = run_rms_statistics(filtered_paths)
rms_threshold = stats['rms_threshold_95']
print(f"极端振动RMS阈值: {rms_threshold:.4f}")

# 查看某个文件的极端窗口
extreme_indices = stats['extreme_indices'][0]
print(f"文件0包含 {len(extreme_indices)} 个极端窗口")
```

**返回统计信息包含：**
- `all_file_rms`：每个文件的RMS值数组列表
- `extreme_indices`：每个文件中极端窗口的索引列表
- `rms_threshold_95`：95%分位数阈值

---

#### **vibration_io_process 工作流 (`workflow.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 集成Step 0-2的完整工作流，返回纯净元数据 |
| 输入 | 无（内部调用各步骤） |
| 输出 | 处理后的元数据列表 |
| 缓存支持 | ✅ 支持结果缓存 |
| 主函数 | `run()` |

**返回的metadata结构：**
```python
{
    'sensor_id': str,              # 传感器ID
    'month': int,                  # 月份 (1-12)
    'day': int,                    # 日期 (1-31)
    'hour': int,                   # 小时 (0-23)
    'file_path': str,              # 源数据文件完整路径
    'actual_length': int,          # 实际采样点数
    'missing_rate': float,         # 缺失率 (0.0-1.0)
    'extreme_rms_indices': list    # 极端窗口索引 [0, 5, 12, ...]
}
```

**示例：**
```python
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow

vib_metadata = run_vib_workflow(use_cache=True)
print(f"处理完成，获得 {len(vib_metadata)} 条元数据")

# 找出包含极端振动的记录
extreme_records = [m for m in vib_metadata if len(m['extreme_rms_indices']) > 0]
print(f"包含极端振动的记录: {len(extreme_records)} 条")
```

**详细文档：** 📖 [vibration_io_process 完整指南](./vibration_io_process/README.md) · [API参考](./vibration_io_process/API.md)

---

### wind_data_io_process 风数据处理

**位置：** `src/data_processer/statistics/wind_data_io_process/`

**功能：** 处理风速数据文件，与振动数据进行时间戳对齐，筛选极端风条件数据

**工作流程：**

#### **Step 0: 获取所有风数据文件 (`step0_get_wind_data.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 从配置的目录递归收集所有风数据文件 |
| 输入 | 数据根目录 + 目标传感器ID列表 |
| 输出 | 文件路径列表 |
| 文件格式 | `.UAN` 后缀的二进制风数据文件 |
| 主函数 | `get_all_wind_files()` |

**示例：**
```python
from src.data_processer.statistics.wind_data_io_process.step0_get_wind_data import get_all_wind_files

all_wind_files = get_all_wind_files()
print(f"发现 {len(all_wind_files)} 个风数据文件")
```

---

#### **Step 1: 时间戳对齐 (`step1_timestamp_align.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 根据振动数据的时间戳，筛选对应时刻的风数据文件 |
| 输入 | 风数据文件列表 + 振动元数据列表 |
| 输出 | 对齐后的风数据文件路径 + 统计信息 |
| 匹配方式 | 基于 (月, 日, 时) 三元组的精确匹配 |
| 主函数 | `run_timestamp_align()` |

**对齐原理：**
- 从振动元数据中提取所有时间戳 (month, day, hour)
- 从风数据文件路径中解析时间戳
- 只保留与振动数据时间戳相匹配的风数据文件

**示例：**
```python
from src.data_processer.statistics.wind_data_io_process.step1_timestamp_align import run_timestamp_align

aligned_paths, stats = run_timestamp_align(all_wind_files, vib_metadata)
print(f"对齐后保留 {len(aligned_paths)} 个风数据文件")
print(f"时间戳覆盖率: {stats['coverage']:.2f}%")
```

**返回统计信息包含：**
- `vib_timestamps`：振动数据的时间戳集合
- `matched_timestamps`：匹配成功的时间戳集合
- `matched_count`：匹配成功的文件数
- `coverage`：覆盖率百分比

---

#### **Step 2: 极端振动对应风数据筛选 (`step2_extreme_filter.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 基于振动的极端窗口索引，筛选对应时间段的风数据 |
| 输入 | 风数据元数据 + 振动元数据（含极端索引） |
| 输出 | 包含极端振动时段的风数据元数据 |
| 映射关系 | 使用 `sensor_config.py` 中的振动-风传感器映射 |
| 主函数 | `run_extreme_filter()` |

**极端时间范围计算：**
- 极端窗口索引 × 60秒 = 时间范围起点
- 例：索引5 → 时间范围 300-360秒（5-6分钟位置）

**示例：**
```python
from src.data_processer.statistics.wind_data_io_process.step2_extreme_filter import run_extreme_filter

filtered_metadata, stats = run_extreme_filter(wind_metadata, vib_metadata)
print(f"筛选出 {len(filtered_metadata)} 个包含极端振动的风数据文件")
print(f"总极端时间窗口数: {stats['total_extreme_time_ranges']}")
```

**返回统计信息包含：**
- `total_extreme_samples`：包含极端振动的文件数
- `total_extreme_time_ranges`：总极端时间窗口数
- `sensor_extreme_counts`：各传感器的极端样本统计
- `avg_extreme_windows_per_file`：平均每个文件的极端窗口数

---

#### **wind_data_io_process 工作流 (`workflow.py`)**

| 项目 | 说明 |
|-----|------|
| 功能 | 集成Step 0-2的完整工作流，返回对齐的风数据元数据 |
| 输入 | 振动元数据 + 极端筛选选项 |
| 输出 | 对齐后的风数据元数据列表 |
| 缓存支持 | ✅ 支持结果缓存 |
| 主函数 | `run()` |

**返回的metadata结构：**
```python
{
    'sensor_id': str,              # 风传感器ID
    'month': int,                  # 月份
    'day': int,                    # 日期
    'hour': int,                   # 小时
    'file_path': str,              # 源数据文件路径
    # 以下字段仅在 extreme_only=True 时包含：
    'extreme_time_ranges': list,   # 极端振动时间范围 [(start_sec, end_sec), ...]
    'vib_sensor_id': str,          # 对应的振动传感器ID
    'extreme_window_count': int    # 极端窗口数量
}
```

**示例：**
```python
from src.data_processer.statistics.wind_data_io_process.workflow import run as run_wind_workflow

# 获取所有时间戳对齐的风数据
wind_metadata = run_wind_workflow(vib_metadata, extreme_only=False, use_cache=True)

# 或者只获取极端振动对应的风数据
extreme_wind_metadata = run_wind_workflow(vib_metadata, extreme_only=True, use_cache=True)
```

**详细文档：** 📖 [wind_data_io_process 完整指南](./wind_data_io_process/README.md)

---

## 📊 API 参考表

### 主入口函数

| 模块 | 函数 | 功能 | 返回类型 |
|------|------|------|---------|
| `workflow.py` | `get_data_pairs()` | 批量提取振动和风数据对 | List[Dict] |
| `vibration_io_process.workflow` | `run()` | 振动数据完整工作流 | List[Dict] |
| `wind_data_io_process.workflow` | `run()` | 风数据完整工作流 | List[Dict] |

### Step 函数

| 模块 | 函数 | 功能 | 返回 |
|------|------|------|------|
| `step0_get_vib_data.py` | `get_all_vibration_files()` | 获取所有振动文件 | List[str] |
| `step1_lackness_filter.py` | `run_lackness_filter()` | 缺失率筛选 | (List[str], Dict) |
| `step2_rms_statistics.py` | `run_rms_statistics()` | RMS统计与极端识别 | (List[str], Dict) |
| `step0_get_wind_data.py` | `get_all_wind_files()` | 获取所有风数据文件 | List[str] |
| `step1_timestamp_align.py` | `run_timestamp_align()` | 时间戳对齐 | (List[str], Dict) |
| `step2_extreme_filter.py` | `run_extreme_filter()` | 极端风数据筛选 | (List[Dict], Dict) |

---

## 🔄 数据流向图

```
┌─────────────────────────────────────────────────────────────────┐
│                    主入口: get_data_pairs()                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼────────┐         ┌───────▼────────┐
            │ 振动数据处理    │         │ 风数据处理      │
            │ vibration_io   │         │ wind_data_io   │
            └───────┬────────┘         └───────┬────────┘
                    │                           │
        ┌───────────┼───────────┐   ┌───────────┼───────────┐
        │           │           │   │           │           │
    ┌───▼──┐   ┌───▼──┐   ┌───▼──┐ │   ┌───▼──┐   ┌───▼──┐
    │Step0 │──▶│Step1 │──▶│Step2 │─┼──▶│Step0 │──▶│Step1 │
    │文件  │   │缺失  │   │RMS   │ │   │文件  │   │时间  │
    │发现  │   │率    │   │统计  │ │   │发现  │   │戳    │
    └──────┘   └──────┘   └──────┘ │   └──────┘   └──────┘
                    │              │           │
                    │              │       ┌───▼──┐
                    │              │       │Step2 │
                    │              │       │极端  │
                    │              │       │风    │
                    │              │       │筛选  │
                    │              │       └──────┘
                    │              │           │
                    └──────┬───────┴───────────┘
                           │
                   ┌───────▼────────┐
                   │  数据切分与对齐  │
                   │  segment_windows│
                   └───────┬────────┘
                           │
                   ┌───────▼────────┐
                   │  返回数据对列表  │
                   │  List[Dict]    │
                   └────────────────┘
```

---

## 💡 常见使用场景

### 场景 1: 获取高质量振动数据

```python
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib

# 获取经过缺失率筛选和RMS统计的振动元数据
vib_metadata = run_vib(threshold=0.05, use_cache=True)

# 找出高质量数据（缺失率<1%）
high_quality = [m for m in vib_metadata if m['missing_rate'] < 0.01]
print(f"高质量数据: {len(high_quality)} / {len(vib_metadata)}")
```

### 场景 2: 分析极端振动对应的风条件

```python
from src.data_processer.statistics.workflow import get_data_pairs

# 获取所有极端振动对应的振动和风数据对
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    enable_extreme_window=True,
    use_multiprocess=True
)

# 分析每个极端窗口
for pair in data_pairs:
    vib_metadata = pair['vib_metadata']
    if len(vib_metadata['extreme_rms_indices']) > 0:
        windows = pair['segmented_windows']
        print(f"传感器 {vib_metadata['sensor_id']} 的 {len(windows)} 个极端窗口")
```

### 场景 3: 按传感器统计分析

```python
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib
from collections import defaultdict

vib_metadata = run_vib(use_cache=True)

# 按传感器分组统计
sensor_stats = defaultdict(lambda: {'count': 0, 'extreme': 0})
for item in vib_metadata:
    sensor_id = item['sensor_id']
    sensor_stats[sensor_id]['count'] += 1
    if len(item['extreme_rms_indices']) > 0:
        sensor_stats[sensor_id]['extreme'] += 1

# 输出统计结果
for sensor_id, stats in sensor_stats.items():
    print(f"{sensor_id}: {stats['count']} 条，其中 {stats['extreme']} 条包含极端振动")
```

---

## 📖 详细文档清单

| 文档 | 位置 | 说明 | 推荐阅读对象 |
|------|------|------|-----------|
| **主工作流说明** | `workflow/README.md` | 集成工作流的完整指南 | 所有用户 |
| **振动处理指南** | `vibration_io_process/README.md` | 振动数据处理的详细说明 | 需要分析振动数据 |
| **风数据处理指南** | `wind_data_io_process/README.md` | 风数据处理的详细说明 | 需要分析风-振关联 |
| **振动API参考** | `vibration_io_process/API.md` | 函数签名和参数详解 | 开发者 |
| **索引导航** | **当前文件** | 模块总览和快速导航 | 新用户 |

---

## 🚀 快速开始

### 1. 最简单的使用方式

```python
# 一行代码获取处理后的数据
from src.data_processer.statistics.workflow import get_data_pairs

data_pairs = get_data_pairs(wind_sensor_id='ST-UAN-G04-001-01')
print(f"获取 {len(data_pairs)} 条数据对")
```

### 2. 分步骤使用各个子模块

```python
# Step 1: 处理振动数据
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib
vib_metadata = run_vib(use_cache=True)

# Step 2: 处理风数据
from src.data_processer.statistics.wind_data_io_process.workflow import run as run_wind
wind_metadata = run_wind(vib_metadata=vib_metadata, use_cache=True)

# Step 3: 进行分析...
```

---

## ⚙️ 配置信息

所有配置项目存储在：
- `src/config/data_processer/statistics/vibration_io_process/config.py`
- `src/config/data_processer/statistics/wind_data_io_process/config.py`

关键配置：
- **振动采样频率**：50 Hz
- **时间窗口**：60 秒
- **缺失率阈值**：5% (可配置)
- **极端振动判定**：95% 分位数

---

## 📞 相关资源

| 资源 | 位置 |
|------|------|
| 主配置文件 | `src/config/data_processer/statistics/` |
| 数据解包工具 | `src/data_processer/io_unpacker.py` |
| 传感器配置 | `src/config/sensor_config.py` |
| 缓存文件 | `cache/data_processer/statistics/` |

---

## 版本信息

- **模块版本**：statistics 1.0
- **文档版本**：1.0
- **最后更新**：2026-02-27
- **维护状态**：✅ 主动维护

---

**💡 提示**：首次使用建议按顺序阅读上面的"快速导航"部分，根据自己的需求找到合适的文档。
