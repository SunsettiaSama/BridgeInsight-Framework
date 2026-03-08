# 风数据处理模块 (Wind Data I/O Process)

## 📋 概述

本模块负责风速、风向、风攻角数据的获取、时间戳对齐筛选和极端事件识别。与振动数据处理模块采用相同的架构设计，通过时间戳匹配确保风-振动数据的对应关系。

## 📁 模块结构

```
wind_data_io_process/
├── workflow.py                  # 完整工作流（主入口）
├── step0_get_wind_data.py       # Step 0: 获取风数据文件
├── step1_timestamp_align.py     # Step 1: 时间戳对齐筛选
├── step2_extreme_filter.py      # Step 2: 极端振动对应风数据筛选
├── __init__.py
└── README.md                    # 本文档
```

## 🎯 工作流程

### Step 0: 获取风数据文件

**功能**:
- 遍历风数据根目录（UAN目录）
- 根据目标风传感器ID列表筛选文件
- 匹配 `.UAN` 后缀的文件
- 返回符合条件的文件路径列表

**关键参数**:
- 采样频率: 1 Hz
- 文件后缀: `.UAN`
- 预期数据长度: 3600 点 (1Hz × 3600s)

### Step 1: 时间戳对齐筛选

**功能**:
- 从振动数据元数据提取所有时间戳 (month, day, hour)
- 从风数据文件路径提取时间戳
- 只保留与振动数据时间戳匹配的风数据文件
- 生成对齐统计报告

**输入**:
- `wind_file_paths`: 所有风数据文件路径列表
- `vib_metadata`: 振动数据元数据（包含时间戳信息）

**输出统计**:
```
输入统计：
  - 振动数据元数据数: N
  - 振动数据时间戳数: M
  - 风数据文件总数: K

匹配结果：
  - 匹配成功的风数据文件: X
  - 匹配成功的时间戳数: Y
  - 匹配率: X/K = Z%

时间戳覆盖率：
  - 有对应风数据的时间戳: Y/M
  - 覆盖率: Y/M × 100%
```

### Step 2: 极端振动对应风数据筛选 (可选)

**功能**:
- 根据振动数据的 `extreme_rms_indices` 找到极端振动对应的时间段
- 从风数据中提取这些时间段内的样本
- 为风数据添加极端时间范围标记
- 用于后续的风-极端振动关系分析

**关键逻辑**:
```
对于每个极端振动窗口索引:
  start_time = index × 60秒
  end_time = (index + 1) × 60秒
  → 找到对应时间段的风数据
```

**输出**:
```
每个筛选后的风数据项添加：
  - extreme_time_ranges: [(start_sec, end_sec), ...] 时间范围列表
  - vib_sensor_id: 对应的振动传感器ID
  - extreme_window_count: 极端窗口数量
```

## 🚀 完整工作流使用

### 使用示例

```python
from src.data_processer.statistics.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.statistics.wind_data_io_process.workflow import run as run_wind_workflow

# Step 1: 运行振动数据工作流
vib_metadata = run_vib_workflow(use_cache=True)
print(f"✓ 获取振动数据元数据: {len(vib_metadata)} 条")

# Step 2: 运行风数据工作流（获取所有对齐的风数据）
wind_metadata = run_wind_workflow(
    vib_metadata=vib_metadata,
    use_cache=True,
    extreme_only=False  # 获取所有对齐的风数据
)
print(f"✓ 获取对齐后的风数据: {len(wind_metadata)} 条")

# Step 3: 获取极端振动对应的风数据（可选）
wind_metadata_extreme = run_wind_workflow(
    vib_metadata=vib_metadata,
    use_cache=True,
    force_recompute=False,
    extreme_only=True  # 只获取极端振动对应的风数据
)
print(f"✓ 获取极端振动对应的风数据: {len(wind_metadata_extreme)} 条")

# 查看示例结果
for item in wind_metadata[:3]:
    print(f"\n文件: {item['file_path']}")
    print(f"  风传感器: {item['sensor_id']}")
    print(f"  时间: {item['month']}/{item['day']} {item['hour']}:00")
    if 'extreme_time_ranges' in item:
        print(f"  极端时间范围: {item['extreme_time_ranges'][:3]}...")
```

### 工作流输出元数据字段

基本字段（所有对齐的风数据）:

| 字段 | 类型 | 说明 |
|------|------|------|
| `sensor_id` | str | 风传感器ID |
| `month` | str | 月份 (e.g., '09') |
| `day` | str | 日期 (e.g., '01') |
| `hour` | str | 小时 (e.g., '12') |
| `file_path` | str | 完整文件路径 |

极端模式额外字段 (`extreme_only=True`):

| 字段 | 类型 | 说明 |
|------|------|------|
| `extreme_time_ranges` | list | 极端振动时间范围 [(start_sec, end_sec), ...] |
| `vib_sensor_id` | str | 对应的振动传感器ID |
| `extreme_window_count` | int | 极端窗口数量 |

## ⚙️ 配置说明

配置文件位置: `src/config/data_processer/statistics/wind_data_io_process/config.py`

### 主要配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FS` | 1 | 风速采样频率 (Hz) |
| `ALL_WIND_ROOT` | ... | 风数据根目录路径 |
| `WIND_FILE_SUFFIX` | ".UAN" | 风数据文件后缀 |

### 目标传感器列表

```python
TARGET_WIND_SENSORS = [
    'ST-UAN-G04-001-01',  # 跨中桥面上游
    'ST-UAN-G04-002-01',  # 跨中桥面下游
    'ST-UAN-G02-002-01',  # 北塔下游江侧桥塔与风障开槽处
    'ST-UAN-G02-002-02',  # 北塔下游江侧风障末尾处
    'ST-UAN-T01-003-01',  # 北索塔塔顶
    'ST-UAN-T02-003-01',  # 南索塔塔顶
]
```

## 📊 风数据文件格式

### 文件结构 (.UAN)

每个 `.UAN` 文件包含一小时的风速、风向、风攻角数据：
- **采样频率**: 1 Hz
- **数据长度**: 3600 个采样点（1小时）
- **数据内容**: 
  - 风速 (m/s)
  - 风向 (度, 0-360°)
  - 风攻角 (度)

### 文件命名规则

```
ST-UAN-G04-001-01_2024_09_01_12.UAN
├─ 传感器ID: ST-UAN-G04-001-01
├─ 年: 2024
├─ 月: 09
├─ 日: 01
└─ 小时: 12
```

## 💾 缓存机制

工作流支持两级缓存以加速重复运行:

1. **元数据缓存** (`metadata_cache.json`)
   - 存储对齐后的风数据元数据
   - 加速重复运行

2. **工作流报告** (`metadata_cache_report.txt`)
   - 记录处理过程和统计信息
   - 便于追踪和调试

### 缓存参数匹配

缓存只在以下条件下使用:
- 振动元数据数量匹配
- `extreme_only` 标志匹配
- `use_cache=True` 且 `force_recompute=False`

## 📈 与振动数据模块的对应关系

| 维度 | 振动数据模块 | 风数据模块 | 说明 |
|------|----------|--------|------|
| 模块名 | `vibration_io_process` | `wind_data_io_process` | 处理模块 |
| 文件后缀 | `.VIC` | `.UAN` | 数据格式 |
| 采样频率 | 50 Hz | 1 Hz | 采样密度 |
| 预期长度 | 180000 点 | 3600 点 | 单文件容量 |
| 步骤1 | 缺失率筛选 | 时间戳对齐 | 质量控制 |
| 步骤2 | RMS统计 | 极端筛选 | 事件识别 |

## 🔄 数据流向

```
┌──────────────────────────┐
│  Step 0: 获取风数据文件  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Step 1: 时间戳对齐筛选           │
│  输入: 风数据 + 振动数据元数据    │
│  输出: 对齐的风数据文件           │
└────────────┬─────────────────────┘
             │
             ▼ (构建元数据)
        ┌────────────┐
        │ 所有对齐   │
        │ 风数据     │
        └──┬───────┬─┘
           │       │
           │ (可选)│
           ▼       ▼
      ┌─────────────────────┐
      │ Step 2: 极端筛选     │
      │ 输出: 极端风-振动   │
      │       对应样本      │
      └─────────────────────┘
```

## 🛠️ 多进程优化

Step 1 时间戳对齐使用单进程（纯内存操作，无需多进程）
Step 2 极端筛选使用单进程（纯映射操作，计算量小）

## 📝 注意事项

1. **依赖关系**: 风数据工作流必须依赖振动数据工作流的输出 (`vib_metadata`)
2. **时间戳一致性**: 确保振动和风数据使用相同的时间戳格式 (month/day/hour)
3. **传感器映射**: 振动到风传感器的映射定义在 `sensor_config.py`
4. **缓存更新**: 当振动数据更新后，需重新运行风数据工作流或设置 `force_recompute=True`

## 🔗 相关文档

- [振动数据处理工作流文档](../vib_data_io_process/README.md)
- [数据解析模块](../../io_unpacker.py)
- [传感器配置](../../../../src/config/sensor_config.py)

---

**更新日期**: 2024-02-23  
**维护者**: 振动特性研究团队
