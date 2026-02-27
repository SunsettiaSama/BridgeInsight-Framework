# 振动数据处理流程 (vibration_io_process) 使用文档

## 模块概述

`src/data_processer/statistics/vibration_io_process/` 是数据处理模块中用于振动数据输入输出处理的核心子模块。它负责从原始数据文件中提取、过滤、分析振动数据，并生成包含完整元数据信息的处理结果。

### 核心特性

- **三阶段处理流程**: 自动化的数据获取、缺失率筛选和RMS统计分析
- **元数据完整性**: 每条记录包含传感器信息、时间戳、数据质量和极端振动标记
- **高效并行处理**: 利用多进程加速文件读取和数据计算
- **缓存机制**: 支持工作流结果缓存，加速重复处理
- **极端振动识别**: 基于95%分位值的自适应阈值检测

---

## 工作流架构

```
vibration_io_process/
    ├── workflow.py                 # 主入口，协调三个步骤
    ├── step0_get_vib_data.py       # Step 0: 文件发现
    ├── step1_lackness_filter.py    # Step 1: 缺失率筛选
    └── step2_rms_statistics.py     # Step 2: RMS统计与极端识别
```

### 处理流程

```
[Step 0] 获取所有振动文件
    │
    └─→ 从配置的根目录递归查找所有 .VIC 格式的振动数据文件
    
[Step 1] 缺失率筛选
    │
    ├─→ 加载每个文件的原始数据
    ├─→ 计算实际数据长度和缺失率
    └─→ 保留缺失率 ≤ 阈值的文件
    
[Step 2] RMS统计分析
    │
    ├─→ 对筛选后的文件进行RMS计算（60秒窗口）
    ├─→ 计算全局95%分位值阈值
    └─→ 识别每个文件中的极端振动窗口
    
[数据处理] 元数据构建
    │
    ├─→ 解析文件路径获取时间和传感器信息
    ├─→ 补充数据质量和极端振动信息
    └─→ 返回完整的元数据列表
```

---

## 主入口函数

### `run()`

```python
def run(threshold=MISSING_RATE_THRESHOLD,
        expected_length=EXPECTED_LENGTH,
        save_path=FILTER_RESULT_PATH,
        cache_path=WORKFLOW_CACHE_PATH,
        use_cache=True,
        force_recompute=False)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.05 | 缺失率阈值（0-1），默认5% |
| `expected_length` | int | 180000 | 预期单文件长度（50Hz × 3600s） |
| `save_path` | str | 配置路径 | 元数据保存路径 |
| `cache_path` | str | 配置路径 | 缓存文件保存路径 |
| `use_cache` | bool | True | 是否使用缓存 |
| `force_recompute` | bool | False | 是否强制重新计算 |

#### 返回值

```python
List[Dict]: 元数据列表，每一项包含以下字段：
{
    'sensor_id': str,              # 传感器 ID（如 'ST-VIC-C18-101-01'）
    'month': int,                  # 数据月份（1-12）
    'day': int,                    # 数据日期（1-31）
    'hour': int,                   # 数据小时（0-23）
    'file_path': str,              # 源文件的完整路径
    'actual_length': int,          # 实际数据长度（采样点数）
    'missing_rate': float,         # 缺失率（0-1）
    'extreme_rms_indices': list    # 极端振动窗口索引列表
}
```

#### metadata 字段详解

**基本信息**
- `sensor_id`: 从文件路径解析获得，唯一标识一个物理传感器
- `month`, `day`, `hour`: 表示数据采集的时间，格式为 YYYY/MM/DD HH:00
- `file_path`: 数据文件的完整路径，用于后续加载原始数据

**数据质量指标**
- `actual_length`: 文件中实际存储的采样点数（不是时间）
- `missing_rate`: 数据缺失比例，计算方式：`1 - (actual_length / expected_length)`
  - 0.0 表示无缺失
  - 0.05 表示5%缺失（典型阈值）
  - > 0.05 表示超过筛选阈值，已被过滤

**极端振动信息**
- `extreme_rms_indices`: 包含超过95%分位值RMS阈值的时间窗口索引
  - 索引从 0 开始
  - 每个索引对应一个60秒的时间窗口
  - 例如：`[0, 1, 5]` 表示第0、1、5个窗口为极端振动
  - 空列表 `[]` 表示该文件无极端振动

---

## 使用示例

### 示例 1: 基础用法 - 使用缓存

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

# 首次运行（计算并缓存结果）
metadata = run(use_cache=True)

# 后续运行（直接使用缓存）
metadata = run(use_cache=True)
```

### 示例 2: 自定义筛选阈值

```python
# 更严格的筛选（1%缺失率）
metadata = run(threshold=0.01)

# 更宽松的筛选（10%缺失率）
metadata = run(threshold=0.10)
```

### 示例 3: 强制重新计算

```python
# 当原始数据更新时，强制重新计算
metadata = run(force_recompute=True)
```

### 示例 4: 处理返回的元数据

```python
metadata = run()

print(f"总文件数: {len(metadata)}")

for item in metadata:
    sensor = item['sensor_id']
    time_str = f"{item['month']}/{item['day']} {item['hour']:02d}:00"
    missing = item['missing_rate'] * 100
    extreme_count = len(item['extreme_rms_indices'])
    
    print(f"{sensor} @ {time_str}")
    print(f"  缺失率: {missing:.2f}%")
    print(f"  极端振动窗口数: {extreme_count}")
```

### 示例 5: 查找极端振动数据

```python
metadata = run()

# 找出包含极端振动的记录
extreme_records = [item for item in metadata 
                   if len(item['extreme_rms_indices']) > 0]

print(f"包含极端振动的记录: {len(extreme_records)} / {len(metadata)}")

for item in extreme_records[:5]:  # 显示前5条
    print(f"{item['sensor_id']}: {len(item['extreme_rms_indices'])} 个极端窗口")
```

---

## 各步骤详解

### Step 0: 获取振动文件

**函数**: `step0_get_vib_data.get_all_vibration_files()`

**功能**: 递归扫描配置的根目录，收集所有符合条件的振动数据文件路径

**工作原理**:
- 遍历 `ALL_VIBRATION_ROOT` 配置的目录树
- 查找所有后缀为 `.VIC` 的文件
- 过滤包含目标传感器ID的文件名

**返回**: 文件路径列表（无特定顺序）

### Step 1: 缺失率筛选

**函数**: `step1_lackness_filter.run_lackness_filter()`

**功能**: 评估每个文件的数据完整性，过滤不合格的样本

**工作原理**:
```
对每个文件：
  1. 读取原始数据，获取实际长度
  2. 计算缺失率 = 1 - (actual_length / expected_length)
  3. 若缺失率 ≤ threshold，则保留该文件
```

**统计输出示例**:
```
样本缺失率筛选报告
=======================================================================
1. 筛选参数：
   - 缺失率阈值: 5.0%
   - 预期单文件长度: 180000 (50Hz * 60s * 60m)

2. 处理统计（总计: 1000 个文件）:
   - 平均缺失率: 2.34%
   - 符合条件的文件 (≤5.0%): 950 (95.00%)
   - 不符合条件的文件 (>5.0%): 50 (5.00%)
```

### Step 2: RMS统计分析

**函数**: `step2_rms_statistics.run_rms_statistics()`

**功能**: 计算每个文件的RMS值，识别极端振动窗口

**工作原理**:
```
对每个筛选后的文件：
  1. 将数据分成60秒窗口（3000采样点）
  2. 计算每个窗口的RMS值：RMS = √(mean(signal²))
  3. 收集所有窗口的RMS值
  
全局处理：
  1. 计算所有RMS值的95%分位数作为阈值
  2. 识别每个文件中超过阈值的窗口
```

**统计输出示例**:
```
RMS极端振动识别报告
=======================================================================
1. 基本统计（所有样本）：
   - 总样本数: 15000
   - 平均RMS: 0.0234 m/s²
   - 标准差: 0.0156 m/s²
   - 最小值: 0.0001 m/s²
   - 最大值: 0.1234 m/s²

2. 极端振动阈值（95%分位值）：
   - RMS阈值: 0.0567 m/s²

3. 极端振动识别统计：
   - 小于阈值样本数: 14250 (95.00%)
   - 大于等于阈值样本数: 750 (5.00%)
   - 包含极端振动的文件数: 245 / 950 (25.79%)
```

---

## 缓存机制

### 缓存文件结构

工作流缓存存储在 `cache_path` 指定的位置（默认JSON格式）：

```json
{
    "metadata": [
        {
            "sensor_id": "ST-VIC-C18-101-01",
            "month": 3,
            "day": 15,
            "hour": 8,
            "file_path": "/data/vibration/2024/03/15/ST-VIC-C18-101-01.VIC",
            "actual_length": 178500,
            "missing_rate": 0.00833,
            "extreme_rms_indices": [5, 12, 23]
        },
        ...
    ],
    "process_params": {
        "missing_rate_threshold": 0.05,
        "expected_length": 180000,
        "total_files": 1000,
        "filtered_files": 950,
        "rms_threshold_95": 0.0567
    },
    "timestamp": "2026-02-26 14:30:45"
}
```

### 缓存验证机制

```python
# 缓存检查逻辑
if use_cache and cache_exists:
    if cached_params['missing_rate_threshold'] == current_threshold and \
       cached_params['expected_length'] == current_expected_length:
        # 参数匹配，使用缓存
        return cached_metadata
    else:
        # 参数不匹配，重新计算
        return run_full_workflow()
```

### 何时使用缓存

| 场景 | use_cache | force_recompute | 结果 |
|------|-----------|-----------------|------|
| 首次运行 | True | False | 计算并缓存 |
| 重复调用（无参数变化） | True | False | 使用缓存 |
| 参数变化 | True | False | 重新计算 |
| 强制更新 | True | True | 重新计算并覆盖缓存 |
| 禁用缓存 | False | - | 始终重新计算 |

---

## 配置参数

### 主要配置常量

位置: `src/config/data_processer/statistics/vibration_io_process/config.py`

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `MISSING_RATE_THRESHOLD` | 0.05 | 缺失率阈值 |
| `EXPECTED_LENGTH` | 180000 | 预期单文件长度 (50Hz × 3600s) |
| `ALL_VIBRATION_ROOT` | 配置路径 | 振动数据根目录 |
| `VIBRATION_FILE_SUFFIX` | '.VIC' | 振动文件后缀 |
| `TARGET_VIBRATION_SENSORS` | [...] | 目标传感器ID列表 |
| `FILTER_RESULT_PATH` | 配置路径 | 元数据保存路径 |
| `WORKFLOW_CACHE_PATH` | 配置路径 | 缓存文件路径 |

### 采样频率设置

| 参数 | 值 | 说明 |
|------|-----|------|
| `FS` | 50 Hz | 振动数据采样频率 |
| `TIME_WINDOW` | 60 s | RMS计算时间窗口 |
| `WINDOW_SIZE` | 3000 | 窗口采样点数 (50 × 60) |

---

## 性能特性

### 时间复杂度

- **Step 0**: O(n_files) - 线性扫描文件系统
- **Step 1**: O(n_files) - 并行处理，受CPU核心数限制
- **Step 2**: O(n_files × m_windows) - 并行RMS计算
- **总体**: 典型场景下约 5-10 分钟（1000个文件）

### 内存占用

- **元数据**: O(n_files) - 每条约 200 字节
- **缓存**: O(n_files) - JSON序列化存储
- **处理过程**: 内存占用随处理的并行度增加
- **建议**: ≥ 4GB RAM 用于 10000+ 文件

### 并行处理

- **Step 1**: 使用 `ProcessPoolExecutor` 并行读取文件
- **Step 2**: 使用 `ProcessPoolExecutor` 并行计算RMS
- **建议**: 设置 workers = CPU 核心数

---

## 常见问题

### Q1: 缺失率是如何计算的？

A: 缺失率 = `1 - (actual_length / expected_length)`
- 如果文件只有 176000 采样点，预期 180000：缺失率 = 1 - (176000/180000) = 0.0222 (2.22%)
- 缺失率 ≤ 阈值的文件才会保留

### Q2: 极端振动窗口索引有什么用？

A: 用于快速定位极端振动数据。例如 `extreme_rms_indices = [5, 12]` 表示：
- 第5个窗口（时间: 300-360秒）的RMS值超过95%阈值
- 第12个窗口（时间: 720-780秒）的RMS值超过95%阈值
可用于后续的详细分析或可视化

### Q3: 如何更新数据并重新处理？

A: 
```python
# 方法1: 强制重新计算
metadata = run(force_recompute=True)

# 方法2: 删除缓存文件后再运行
import os
os.remove(WORKFLOW_CACHE_PATH)
metadata = run(use_cache=True)
```

### Q4: 如何处理特定的传感器？

A: 直接在后续流程中过滤：
```python
metadata = run()
specific_sensor = [m for m in metadata if m['sensor_id'] == 'ST-VIC-C18-101-01']
```

### Q5: RMS值的单位是什么？

A: **m/s²**（加速度）。这是振动传感器的标准输出单位。

---

## 错误处理

### 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 返回空列表 | 无文件匹配 | 检查配置的根目录和传感器ID |
| 缓存参数不匹配 | 参数变化 | 设置 `force_recompute=True` |
| 内存溢出 | 文件过多 | 分批处理或增加系统内存 |
| 文件读取失败 | 文件损坏 | 检查数据文件完整性 |

### 调试模式

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)

metadata = run(force_recompute=True)
```

---

## 与其他模块的关系

### 下游模块

- **`workflow.py`**: 主统计工作流模块，将 `vibration_io_process` 与 `wind_data_io_process` 整合
- **`wind_data_io_process`**: 平行的风数据处理模块，结构相似

### 依赖模块

- **`src.data_processer.io_unpacker`**: 数据解包工具，用于读取原始 .VIC 文件
- **`src.config.data_processer.statistics.vibration_io_process.config`**: 配置管理

---

## 最佳实践

### 1. 缓存管理

```python
# 优先使用缓存，避免重复计算
metadata = run(use_cache=True, force_recompute=False)
```

### 2. 参数一致性

```python
# 保持参数一致以充分利用缓存
metadata1 = run(threshold=0.05)  # 计算并缓存
metadata2 = run(threshold=0.05)  # 使用缓存
```

### 3. 数据验证

```python
# 检查返回数据的质量
metadata = run()
assert len(metadata) > 0, "无有效数据"
assert all('extreme_rms_indices' in m for m in metadata), "数据不完整"
```

### 4. 异常处理

```python
# 虽然函数设计上不抛出异常，但应检查返回值
metadata = run()
if not metadata:
    print("处理失败，请检查配置和数据源")
```

---

## 版本信息

- **最后更新**: 2026-02-26
- **模块位置**: `src/data_processer/statistics/vibration_io_process/`
- **相关文档**: 
  - API 参考: [API.md](API.md)
  - 快速开始: [QUICKSTART.md](QUICKSTART.md)
  - 高级主题: [ADVANCED.md](ADVANCED.md)
