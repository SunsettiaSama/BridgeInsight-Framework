# 快速入门

## 5 分钟快速开始

### 最简单的使用方式

```python
from src.data_processer.statistics.workflow import get_data_pairs

# 提取数据
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    enable_extreme_window=True
)

# 处理结果
for pair in data_pairs:
    for vib_seg, (wind_speed, wind_dir, wind_angle) in pair['segmented_windows']:
        # 你的分析代码
        pass
```

---

## 完整示例

### 完整的数据分析流程

```python
import numpy as np
from src.data_processer.statistics.workflow import get_data_pairs

# 1. 获取数据
print("正在获取数据...")
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    window_duration_minutes=1.0
)

# 2. 统计总窗口数
total_windows = sum(len(pair['segmented_windows']) for pair in data_pairs)
print(f"总窗口数: {total_windows}")

# 3. 分析数据
print("\n开始分析...")
for pair in data_pairs:
    metadata = pair['vib_metadata']
    print(f"\n处理: {metadata['sensor_id']} @ {metadata['month']}/{metadata['day']} {metadata['hour']}:00")
    
    for i, (vib_seg, (wind_speed, wind_dir, wind_angle)) in enumerate(pair['segmented_windows']):
        # 计算振动 RMS
        vib_rms = np.sqrt(np.mean(np.square(vib_seg)))
        
        # 计算风速平均值
        wind_mean = np.mean(wind_speed)
        wind_std = np.std(wind_speed)
        
        print(f"  窗口 {i}: RMS={vib_rms:.4f}, 风速平均={wind_mean:.2f}±{wind_std:.2f}")
```

---

## 选择合适的切分模式

### 场景 1: 分析极端风速下的振动

使用极端窗口模式：

```python
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    enable_extreme_window=True
)
```

**特点**: 只获取极端风速条件下的数据对，数据量小，处理快

### 场景 2: 进行时频特性分析

使用常规窗口模式：

```python
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    window_duration_minutes=2.0
)
```

**特点**: 获取所有均匀切分的数据窗口，便于时序分析

### 场景 3: 自定义分析

不使用切分：

```python
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01'
)

# 自己处理原始数据
for pair in data_pairs:
    # segmented_windows 为空，需要自己实现
    pass
```

**特点**: 获取完整的原始数据，灵活性最高

---

## 处理多个传感器

```python
# 定义要处理的传感器
vibration_sensors = [
    'ST-VIC-C18-101-01',  # 边跨1/4跨 面内
    'ST-VIC-C18-101-02',  # 边跨1/4跨 面外
    'ST-VIC-C18-201-01',  # 跨中1/4跨 面内
    'ST-VIC-C18-201-02',  # 跨中1/4跨 面外
]

wind_sensor = 'ST-UAN-G04-001-01'

# 批量处理
results = {}
for vib_sensor in vibration_sensors:
    print(f"处理 {vib_sensor}...")
    results[vib_sensor] = get_data_pairs(
        wind_sensor_id=wind_sensor,
        vib_sensor_id=vib_sensor,
        window_duration_minutes=1.0
    )

# 统计结果
for sensor, data_pairs in results.items():
    total_windows = sum(len(p['segmented_windows']) for p in data_pairs)
    print(f"{sensor}: {total_windows} 个窗口")
```

---

## 数据保存和加载

### 保存结果

```python
import pickle
import json

# 方法 1: 使用 pickle（包含 numpy 数组）
with open('data_pairs.pkl', 'wb') as f:
    pickle.dump(data_pairs, f)

# 方法 2: 保存元数据到 JSON
metadata_only = [
    {
        'vib_metadata': pair['vib_metadata'],
        'segment_config': pair['segment_config'],
        'window_count': len(pair['segmented_windows'])
    }
    for pair in data_pairs
]

with open('metadata.json', 'w') as f:
    json.dump(metadata_only, f, indent=2)
```

### 加载结果

```python
import pickle

# 加载数据
with open('data_pairs.pkl', 'rb') as f:
    data_pairs = pickle.load(f)

# 继续使用
for pair in data_pairs:
    print(f"窗口数: {len(pair['segmented_windows'])}")
```

---

## 常见问题

### Q: 如何控制处理的数据量？

A: 使用 `vib_sensor_id` 参数精确指定传感器，或使用缓存加快处理。

### Q: 处理需要多长时间？

A: 取决于数据量和切分模式。使用缓存可显著加快处理。

### Q: 如何保存中间结果？

A: 使用 pickle 保存 `data_pairs`，或只保存元数据到 JSON。

### Q: 能否对结果进行二次筛选？

A: 可以，遍历 `data_pairs` 并过滤需要的元素。

---

## 下一步

- 查看 [完整 API 文档](API.md)
- 查看 [详细使用指南](README.md)
- 浏览 [示例代码](examples.py)（如果有）
