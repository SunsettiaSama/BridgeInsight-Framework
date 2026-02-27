# 高级主题

## Metadata 数据结构深度解析

### 核心概念

元数据（metadata）是对振动数据文件的完整描述，包含三大类信息：**时间信息**、**数据质量指标**和**振动特征标记**。

```python
{
    # 时间信息（来自文件路径解析）
    'sensor_id': str,              # 传感器标识
    'month': int,                  # 月份
    'day': int,                    # 日期
    'hour': int,                   # 小时
    
    # 文件信息（来自 Step 1）
    'file_path': str,              # 完整文件路径
    'actual_length': int,          # 实际采样点数
    'missing_rate': float,         # 缺失率
    
    # 振动特征（来自 Step 2）
    'extreme_rms_indices': list    # 极端RMS窗口索引
}
```

### 每个字段的详细说明

#### 1. 时间信息

**`sensor_id` (str)**
- 来源：从文件名或路径中解析
- 格式：`ST-{类型}-{位置}-{方向}-{编号}` 
  - 例：`ST-VIC-C18-101-01`
  - `VIC`: 振动传感器类型
  - `C18`: 塔高位置
  - `101`: 横截面位置
  - `01`: 传感器编号
- 用途：唯一标识一个物理传感器
- 唯一性：一个 `sensor_id` 对应同一位置的不同时间记录

**`month` (int)**
- 范围：1-12
- 含义：数据采集的月份
- 示例：3 表示3月

**`day` (int)**
- 范围：1-31（根据月份而定）
- 含义：数据采集的日期
- 示例：15 表示第15天

**`hour` (int)**
- 范围：0-23
- 含义：数据采集的小时（00:00-23:00）
- 示例：8 表示当天的第8个小时（08:00-08:59）

**时间标准化**
```python
# 时间范围
# 开始时间：2026年 month月 day日 hour:00:00
# 结束时间：2026年 month月 day日 hour:59:59
# 数据长度：恰好1小时（或接近1小时）

# 采样点数与时间的对应
actual_length = 180000  # 采样点
fs = 50  # Hz
duration = actual_length / fs  # = 3600 秒 = 1 小时
```

#### 2. 文件信息

**`file_path` (str)**
- 含义：源数据文件的完整绝对路径
- 格式：Windows 或 Unix 路径格式
- 用途：用于后续加载原始数据
- 例：`D:/data/vibration/2026/03/15/ST-VIC-C18-101-01_2026031508.VIC`

**`actual_length` (int)**
- 单位：采样点数（不是字节，不是秒）
- 范围：0-180000（对于1小时的数据）
- 计算：从解包器 `UNPACK.VIC_DATA_Unpack()` 返回
- 示例：
  - 180000：完整的1小时数据（50Hz × 3600s）
  - 178500：部分缺失（缺失 1500 点 = 30秒）
  - 0：文件损坏或无法读取

**`missing_rate` (float)**
- 单位：比例（0-1）
- 计算公式：`missing_rate = 1 - (actual_length / expected_length)`
- 其中 `expected_length = 180000`
- 范围：0.0 到 1.0
  - 0.0：无缺失（完整文件）
  - 0.05：5%缺失（90000点缺失）
  - 1.0：全部缺失（无有效数据）
- 阈值筛选：在 Step 1 中，`missing_rate <= threshold` 的文件被保留

**缺失率计算示例**
```python
# 例1：完整文件
actual_length = 180000
expected_length = 180000
missing_rate = 1 - (180000 / 180000) = 0.0

# 例2：缺失30秒
actual_length = 178500  # 180000 - 1500
expected_length = 180000
missing_rate = 1 - (178500 / 180000) = 0.00833 ≈ 0.83%

# 例3：缺失10%
actual_length = 162000
expected_length = 180000
missing_rate = 1 - (162000 / 180000) = 0.10 (10%)
```

#### 3. 振动特征

**`extreme_rms_indices` (list)**
- 单位：窗口索引（整数列表）
- 含义：包含极端振动的时间窗口位置
- 长度：可变（0 到 ~60 个元素）
- 示例：
  - `[]`：无极端振动
  - `[0, 5, 12]`：第0、5、12窗口为极端
  - `[15, 16, 17, 18, 19]`：连续5个窗口为极端

**窗口索引到时间的映射**
```python
# 时间窗口配置
WINDOW_SIZE = 3000  # 采样点
FS = 50  # Hz
WINDOW_DURATION = WINDOW_SIZE / FS  # = 60 秒

# 索引到时间映射
window_index = 5
start_time = window_index * 60  # = 300 秒 = 5分钟
end_time = (window_index + 1) * 60  # = 360 秒 = 6分钟
# 该窗口覆盖 5:00-5:59 分钟

# 索引到采样点映射
start_sample = window_index * 3000  # = 15000
end_sample = (window_index + 1) * 3000  # = 18000
# 该窗口包含采样点 [15000:18000]
```

**RMS阈值与极端识别**
```python
# Step 2 中的处理流程
1. 计算所有窗口的RMS值
   rms_values = [0.012, 0.015, ..., 0.089, ...]  # 所有RMS值

2. 计算全局95%分位数
   rms_threshold_95 = np.percentile(rms_values, 95)
   # 例：rms_threshold_95 = 0.0567 m/s²

3. 识别超过阈值的窗口
   extreme_rms_indices = np.where(rms_values >= 0.0567)[0].tolist()
   # 例：[5, 12, 23]

4. 结果含义
   # 窗口5、12、23的RMS值都在前5%（最高的振动水平）
```

---

## Metadata 的生成过程

### 生成流程图

```
原始文件
    │
    ├─ [Step 0] 文件发现
    │   └─ 获取所有 .VIC 文件路径
    │
    ├─ [Step 1] 缺失率筛选
    │   ├─ 读取文件：UNPACK.VIC_DATA_Unpack(file_path)
    │   ├─ 获取长度：actual_length = len(vibration_data)
    │   ├─ 计算缺失率：missing_rate = 1 - (actual_length / 180000)
    │   └─ 筛选：keep if missing_rate <= threshold
    │
    ├─ [Step 2] RMS 统计与极端识别
    │   ├─ 分窗：将数据分成 60秒 窗口
    │   ├─ 计算RMS：rms = √(mean(data²)) 对每个窗口
    │   ├─ 全局阈值：threshold_95 = percentile(all_rms, 95)
    │   └─ 识别极端：extreme_indices = where(rms >= threshold_95)
    │
    └─ [数据处理] 元数据构建
        ├─ 路径解析：parse_path_metadata(file_paths)
        │   └─ 提取：sensor_id, month, day, hour
        ├─ 补充字段：
        │   ├─ file_path
        │   ├─ actual_length (from Step 1)
        │   ├─ missing_rate (from Step 1)
        │   └─ extreme_rms_indices (from Step 2)
        └─ 输出：metadata list
```

### 代码流程示例

```python
# Step 0: 文件发现
all_files = get_all_vibration_files()
# all_files = ['path1.VIC', 'path2.VIC', ...]

# Step 1: 缺失率筛选
filtered_paths, stats = run_lackness_filter(all_files, threshold=0.05)
# 保留：缺失率 <= 5% 的文件

# Step 2: RMS 统计
file_paths, rms_stats = run_rms_statistics(filtered_paths)
# rms_stats['extreme_indices'] = [[5, 12], [0], [], ...]

# 元数据构建
metadata = parse_path_metadata(filtered_paths)  # 基础信息
for i, path in enumerate(filtered_paths):
    metadata[i]['file_path'] = path
    metadata[i]['actual_length'] = int(stats['all_lengths'][filtered_index[i]])
    metadata[i]['missing_rate'] = float(stats['all_missing_rates'][filtered_index[i]])
    metadata[i]['extreme_rms_indices'] = rms_stats['extreme_indices'][i]
```

---

## 常见查询模式

### 查询 1: 获取特定传感器的数据

```python
metadata = run()
sensor_id = 'ST-VIC-C18-101-01'
sensor_data = [m for m in metadata if m['sensor_id'] == sensor_id]
print(f"{sensor_id}: {len(sensor_data)} 条记录")
```

### 查询 2: 获取特定日期的数据

```python
metadata = run()
target_month, target_day = 3, 15
date_data = [m for m in metadata if m['month'] == target_month and m['day'] == target_day]
print(f"2026/3/15: {len(date_data)} 条记录")
```

### 查询 3: 找出数据质量较差的记录

```python
metadata = run()
# 缺失率 > 2% 的记录
poor_quality = [m for m in metadata if m['missing_rate'] > 0.02]
print(f"低质量记录: {len(poor_quality)}")

# 显示详情
for m in poor_quality:
    print(f"{m['sensor_id']}: 缺失率 {m['missing_rate']*100:.2f}%")
```

### 查询 4: 找出包含极端振动的记录

```python
metadata = run()
extreme_records = [m for m in metadata if len(m['extreme_rms_indices']) > 0]
print(f"包含极端振动的记录: {len(extreme_records)}")

# 按极端窗口数量排序
extreme_records_sorted = sorted(extreme_records, 
                               key=lambda x: len(x['extreme_rms_indices']), 
                               reverse=True)
for m in extreme_records_sorted[:10]:
    count = len(m['extreme_rms_indices'])
    print(f"{m['sensor_id']} @ {m['month']}/{m['day']} {m['hour']:02d}:00: {count} 个极端窗口")
```

### 查询 5: 统计分析

```python
import numpy as np
from collections import Counter

metadata = run()

# 传感器计数
sensors = [m['sensor_id'] for m in metadata]
sensor_counts = Counter(sensors)
print("传感器分布:")
for sensor, count in sensor_counts.most_common(5):
    print(f"  {sensor}: {count} 条记录")

# 缺失率统计
missing_rates = [m['missing_rate'] for m in metadata]
print(f"\n缺失率统计:")
print(f"  平均: {np.mean(missing_rates)*100:.2f}%")
print(f"  最小: {np.min(missing_rates)*100:.2f}%")
print(f"  最大: {np.max(missing_rates)*100:.2f}%")
print(f"  中位数: {np.median(missing_rates)*100:.2f}%")

# 极端振动统计
extreme_window_counts = [len(m['extreme_rms_indices']) for m in metadata]
print(f"\n极端窗口统计:")
print(f"  包含极端振动的记录: {np.sum(np.array(extreme_window_counts) > 0)}")
print(f"  总极端窗口数: {np.sum(extreme_window_counts)}")
print(f"  平均每条记录: {np.mean(extreme_window_counts):.2f} 个窗口")
```

---

## Metadata 的后续应用

### 应用 1: 数据预处理

```python
metadata = run()

# 筛选高质量数据
high_quality = [m for m in metadata if m['missing_rate'] < 0.01]
print(f"高质量数据: {len(high_quality)} / {len(metadata)}")

# 按时间排序
sorted_by_time = sorted(high_quality, 
                       key=lambda x: (x['month'], x['day'], x['hour']))

# 按传感器分组
by_sensor = {}
for m in sorted_by_time:
    sensor = m['sensor_id']
    if sensor not in by_sensor:
        by_sensor[sensor] = []
    by_sensor[sensor].append(m)
```

### 应用 2: 数据加载

```python
metadata = run()

# 选择一条记录
record = metadata[0]

# 加载原始数据
from src.data_processer.io_unpacker import UNPACK
unpacker = UNPACK(init_path=False)
vibration_data = unpacker.VIC_DATA_Unpack(record['file_path'])

# 数据长度验证
assert len(vibration_data) == record['actual_length']
print(f"✓ 数据加载成功: {len(vibration_data)} 采样点")

# 提取极端窗口
extreme_indices = record['extreme_rms_indices']
window_size = 3000  # 60秒 @ 50Hz
for idx in extreme_indices:
    start = idx * window_size
    end = (idx + 1) * window_size
    extreme_data = vibration_data[start:end]
    # 处理极端数据
```

### 应用 3: 可视化

```python
import matplotlib.pyplot as plt
from collections import Counter

metadata = run()

# 按日期分布统计
date_counts = Counter((m['month'], m['day']) for m in metadata)
dates = sorted(date_counts.keys())
counts = [date_counts[d] for d in dates]

plt.figure(figsize=(12, 4))
plt.bar(range(len(dates)), counts)
plt.xlabel('Date')
plt.ylabel('Record Count')
plt.title('Daily Distribution')
plt.show()

# 缺失率分布
import numpy as np
missing_rates = np.array([m['missing_rate'] for m in metadata]) * 100
plt.hist(missing_rates, bins=50)
plt.xlabel('Missing Rate (%)')
plt.ylabel('Count')
plt.title('Missing Rate Distribution')
plt.show()
```

---

## 缓存与持久化

### 缓存格式

缓存文件为 JSON 格式，包含完整的 metadata 列表和处理参数：

```json
{
    "metadata": [
        {
            "sensor_id": "...",
            "month": 3,
            "day": 15,
            "hour": 8,
            "file_path": "...",
            "actual_length": 180000,
            "missing_rate": 0.0,
            "extreme_rms_indices": [5, 12]
        }
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

### 手动加载缓存

```python
import json

cache_path = "path/to/cache.json"
with open(cache_path, 'r', encoding='utf-8') as f:
    cache_data = json.load(f)

metadata = cache_data['metadata']
params = cache_data['process_params']

print(f"缓存时间: {cache_data['timestamp']}")
print(f"元数据条数: {len(metadata)}")
print(f"处理参数: {params}")
```

---

## 注意事项

### 注意 1: 时间精度

```python
# 时间信息的粒度是 1 小时
# 不能精确到分钟或秒

# 错误用法
# "这个数据点在 3月15日 8时30分"

# 正确用法
# "这个数据对应 3月15日 8时这一小时内的所有数据"
```

### 注意 2: 缺失率解释

```python
# 缺失率是基于预期长度的相对值
# 不是指数据文件的大小，而是采样点数

# 例如：
actual_length = 178500  # 实际采样点数
expected_length = 180000  # 预期采样点数
missing_rate = 1 - (178500 / 180000) = 0.00833
# 表示相对于预期的 180000 点，缺失了 1500 点（1.5秒）
```

### 注意 3: 极端振动的统计性质

```python
# extreme_rms_indices 是基于全局统计的
# 不是绝对阈值，而是相对排名

# 95% 分位值意味着：
# - 5% 的窗口被标记为极端
# - 不同数据集的阈值可能不同
# - 高RMS窗口不一定代表故障，可能是正常的高风速条件
```

---

## 性能优化

### 优化 1: 缓存管理

```python
# 充分利用缓存，避免重复计算
metadata = run(use_cache=True)  # 如果缓存存在，快速返回

# 只在必要时重新计算
metadata = run(force_recompute=True)  # 强制更新
```

### 优化 2: 数据过滤

```python
# 及时过滤，减少后续处理量
metadata = run()

# 立即应用筛选
filtered = [m for m in metadata if m['missing_rate'] < 0.01]

# 减少内存占用
del metadata
```

### 优化 3: 批处理

```python
# 对大量数据使用分批处理
metadata = run()

batch_size = 100
for i in range(0, len(metadata), batch_size):
    batch = metadata[i:i+batch_size]
    process_batch(batch)
```

---

## 版本信息

- **最后更新**: 2026-02-26
- **适用版本**: vibration_io_process 1.0+
