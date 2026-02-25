# 统计数据工作流 (Workflow) 使用文档

## 模块概述

`src/data_processer/statistics/workflow.py` 是数据处理模块的核心统计工作流，提供了一个统一的主入口函数 `get_data_pairs()`，用于批量提取、处理和切分振动和风数据。

### 核心特性

- **集成工作流**: 一次调用集成了振动和风数据的所有预处理步骤
- **无需预加载**: 自动加载和处理元数据，无需外部管理
- **灵活的数据切分**: 支持三种切分模式
- **内存优化**: 只保存必要的数据，不存储原始数据序列
- **传感器筛选**: 精确的传感器选择能力

---

## 主入口函数

### `get_data_pairs()`

```python
def get_data_pairs(wind_sensor_id, vib_sensor_id=None, use_multiprocess=False, 
                   enable_extreme_window=False, window_duration_minutes=None,
                   use_vib_cache=True, use_wind_cache=True)
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wind_sensor_id` | str | 必需 | 风传感器ID (e.g., `'ST-UAN-G04-001-01'`) |
| `vib_sensor_id` | str\|None | None | 振动传感器ID过滤，若指定则仅处理该传感器的数据 |
| `use_multiprocess` | bool | False | 是否使用多进程处理 |
| `enable_extreme_window` | bool | False | 是否进行极端窗口筛选 |
| `window_duration_minutes` | float\|None | None | 常规窗口时长（分钟），与 `enable_extreme_window` 互斥 |
| `use_vib_cache` | bool | True | 是否使用振动数据缓存 |
| `use_wind_cache` | bool | True | 是否使用风数据缓存 |

#### 返回值

```python
list: 数据对列表，每一项为字典，结构如下：
{
    'vib_metadata': {                    # 原始振动元数据
        'sensor_id': str,
        'month': int,
        'day': int,
        'hour': int,
        'file_path': str,
        'actual_length': int,
        'missing_rate': float,
        'extreme_rms_indices': list
    },
    'segment_config': {                  # 本次处理配置
        'vib_sensor_id': str|None,
        'wind_sensor_id': str,
        'enable_extreme_window': bool,
        'window_duration_minutes': float|None,
        'vib_fs': int,                   # 采样频率 50 Hz
        'wind_fs': int                   # 采样频率 1 Hz
    },
    'segmented_windows': [               # 切分后的数据窗口
        (vib_segment, (wind_speed, wind_direction, wind_angle)),
        ...
    ]
}
```

---

## 使用示例

### 示例 1: 基础用法 - 极端窗口切分

```python
from src.data_processer.statistics.workflow import get_data_pairs

# 提取极端风速窗口的振动-风数据对
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    enable_extreme_window=True
)

# 处理结果
for pair in data_pairs:
    metadata = pair['vib_metadata']
    print(f"传感器: {metadata['sensor_id']}, 时间: {metadata['month']}/{metadata['day']} {metadata['hour']}:00")
    print(f"窗口数: {len(pair['segmented_windows'])}")
    
    for vib_seg, (wind_speed_seg, wind_dir_seg, wind_angle_seg) in pair['segmented_windows']:
        print(f"  振动数据: {vib_seg.shape}, 风速数据: {wind_speed_seg.shape}")
```

### 示例 2: 常规时间窗口切分

```python
# 按 2 分钟的时间长度进行均匀切分
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-201-01',
    window_duration_minutes=2.0
)

# 处理数据
for pair in data_pairs:
    windows = pair['segmented_windows']
    print(f"总窗口数: {len(windows)}")
    
    # 进行分析
    for i, (vib_seg, wind_seg) in enumerate(windows):
        vib_rms = np.sqrt(np.mean(np.square(vib_seg)))
        wind_mean = np.mean(wind_seg[0])
        print(f"窗口 {i}: RMS={vib_rms:.4f}, 平均风速={wind_mean:.2f}")
```

### 示例 3: 多个传感器处理

```python
# 处理所有振动传感器
sensors = [
    'ST-VIC-C18-101-01',
    'ST-VIC-C18-101-02',
    'ST-VIC-C18-201-01',
    'ST-VIC-C18-201-02'
]

for sensor_id in sensors:
    data_pairs = get_data_pairs(
        wind_sensor_id='ST-UAN-G04-001-01',
        vib_sensor_id=sensor_id,
        window_duration_minutes=1.0
    )
    
    total_windows = sum(len(pair['segmented_windows']) for pair in data_pairs)
    print(f"{sensor_id}: {total_windows} 个数据窗口")
```

### 示例 4: 不使用缓存重新计算

```python
# 当数据更新时，强制重新计算
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    enable_extreme_window=True,
    use_vib_cache=False,
    use_wind_cache=False
)
```

---

## 数据流程图

```
get_data_pairs()
    │
    ├─ [Step 1] 获取处理后的元数据
    │   ├─ run_vib_workflow()  → 振动数据元数据
    │   └─ run_wind_workflow() → 风数据元数据
    │
    ├─ [Step 2] 传感器筛选
    │   └─ 根据 vib_sensor_id 过滤
    │
    ├─ [Step 3] 数据处理配置
    │   └─ 构建 segment_config
    │
    ├─ [Step 4] 批量处理
    │   ├─ _load_vibration_and_wind_data() → 加载原始数据
    │   ├─ _segment_extreme_wind_windows() 或
    │   └─ _segment_windows_by_duration() → 切分数据
    │
    └─ 返回数据对列表
```

---

## 采样频率信息

模块内部使用的采样频率配置：

| 数据类型 | 采样频率 | 窗口时长 | 窗口大小 |
|---------|---------|---------|----------|
| 振动数据 | 50 Hz | 60 秒 | 3000 采样点 |
| 风速数据 | 1 Hz | 60 秒 | 60 采样点 |
| 频率比 | 50:1 | - | - |

---

## 三种切分模式

### 模式 1: 极端窗口切分

**触发条件**: `enable_extreme_window=True`

**描述**: 基于 RMS 极端振动索引进行切分
- 从元数据的 `extreme_rms_indices` 中获取极端窗口位置
- 每个窗口大小为 3000 采样点（60秒）
- 返回所有极端条件下的数据对

**适用场景**: 分析极端振动和风速的对应关系

### 模式 2: 常规时间窗口切分

**触发条件**: `window_duration_minutes` 不为 None

**描述**: 按指定时间长度进行均匀切分
- 将完整数据分成若干个相等长度的窗口
- 窗口大小由参数指定（单位：分钟）
- 返回所有有效的数据对

**适用场景**: 时间序列分析、时频特性研究

### 模式 3: 完整原始数据

**触发条件**: 两个条件都不设置

**描述**: 返回完整的原始数据对
- 每条元数据对应一个完整的数据对
- `segmented_windows` 为空列表
- 需要外部自行处理原始数据

**适用场景**: 自定义数据处理、灵活分析

---

## 返回数据的处理

### 访问数据窗口

```python
for pair in data_pairs:
    # 元数据
    sensor = pair['vib_metadata']['sensor_id']
    time = f"{pair['vib_metadata']['month']}/{pair['vib_metadata']['day']}"
    
    # 配置
    config = pair['segment_config']
    vib_fs = config['vib_fs']  # 50 Hz
    wind_fs = config['wind_fs']  # 1 Hz
    
    # 数据窗口
    for vib_seg, (wind_speed_seg, wind_dir_seg, wind_angle_seg) in pair['segmented_windows']:
        # 使用数据进行分析
        pass
```

### 合并多个结果

```python
all_windows = []
for pair in data_pairs:
    all_windows.extend(pair['segmented_windows'])

print(f"总窗口数: {len(all_windows)}")
```

### 保存结果

```python
import pickle

# 保存数据对
with open('data_pairs.pkl', 'wb') as f:
    pickle.dump(data_pairs, f)

# 加载数据对
with open('data_pairs.pkl', 'rb') as f:
    data_pairs = pickle.load(f)
```

---

## 性能优化

### 使用缓存加速

```python
# 第一次调用（计算并缓存）
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    use_vib_cache=True,
    use_wind_cache=True
)

# 后续调用（使用缓存，快速返回）
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    use_vib_cache=True,
    use_wind_cache=True
)
```

### 传感器筛选的优势

```python
# 高效：仅处理特定传感器
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',  # 精确筛选
    window_duration_minutes=1.0
)
```

---

## 常见问题

### Q: 如何知道某个传感器是否有数据？

A: 查看返回的 `data_pairs` 长度和 `segmented_windows` 的长度。若都为 0，表示无数据。

### Q: 极端窗口和常规窗口可以同时使用吗？

A: 不可以。`enable_extreme_window=True` 时会忽略 `window_duration_minutes` 参数。

### Q: 如何控制输出信息的详细程度？

A: 函数总是输出详细的处理进度和统计信息。可以通过重定向 stdout 来控制显示。

### Q: 数据在内存中占用多少空间？

A: 仅存储切分后的数据窗口，不存储原始数据。具体占用取决于窗口数量。

---

## 相关模块

- `src.data_processer.statistics.vibration_io_process.workflow`: 振动数据工作流
- `src.data_processer.statistics.wind_data_io_process.workflow`: 风数据工作流
- `src.data_processer.io_unpacker`: 数据解包工具

---

## 版本信息

- 最后更新: 2026-02-25
- 模块位置: `src/data_processer/statistics/workflow.py`
