# 快速开始

## 30秒快速入门

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

# 处理振动数据
metadata = run()

# 查看结果
print(f"处理了 {len(metadata)} 个文件")
for item in metadata[:3]:
    print(f"  {item['sensor_id']}: 缺失率 {item['missing_rate']*100:.1f}%")
```

---

## 安装与配置

### 前置条件

- Python 3.7+
- numpy
- tqdm
- 访问振动数据文件

### 配置检查

确保配置文件存在：
```
src/config/data_processer/statistics/vibration_io_process/config.py
```

检查关键配置：
```python
ALL_VIBRATION_ROOT          # 数据根目录
TARGET_VIBRATION_SENSORS    # 传感器列表
```

---

## 基础使用

### 示例 1: 首次运行

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

# 第一次运行会计算并缓存结果
metadata = run(use_cache=True)

print(f"✓ 处理完成")
print(f"  总文件数: {len(metadata)}")
print(f"  包含极端振动的文件: {sum(1 for m in metadata if len(m.get('extreme_rms_indices', [])) > 0)}")
```

### 示例 2: 使用缓存

```python
# 后续调用会直接使用缓存（快速返回）
metadata = run(use_cache=True)  # < 1秒
```

### 示例 3: 强制重新计算

```python
# 当需要更新时
metadata = run(force_recompute=True)
```

---

## 理解返回数据

### 元数据结构

每条元数据记录代表一个1小时的数据文件：

```python
metadata = run()

for item in metadata:
    # 基本信息
    print(item['sensor_id'])           # 传感器ID
    print(f"{item['month']}/{item['day']} {item['hour']}:00")  # 时间
    
    # 数据质量
    print(item['actual_length'])       # 数据长度（采样点）
    print(item['missing_rate'])        # 缺失率（0-1）
    
    # 极端振动
    print(item['extreme_rms_indices']) # 极端窗口列表 [5, 12, ...]
```

### 极端振动窗口

```python
# 例：[5, 12, 23]
# 表示第 5, 12, 23 个 60秒窗口的RMS值超过95%阈值

# 对应时间：
# 窗口5: 300-360秒 (5 × 60)
# 窗口12: 720-780秒 (12 × 60)
# 窗口23: 1380-1440秒 (23 × 60)
```

---

## 常见用途

### 用途 1: 数据质量检查

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

metadata = run()

# 检查缺失率分布
import numpy as np
missing_rates = [m['missing_rate'] for m in metadata]
print(f"平均缺失率: {np.mean(missing_rates)*100:.2f}%")
print(f"最大缺失率: {np.max(missing_rates)*100:.2f}%")

# 找出高缺失率文件
high_missing = [m for m in metadata if m['missing_rate'] > 0.02]
print(f"缺失率 > 2% 的文件: {len(high_missing)}")
```

### 用途 2: 极端振动分析

```python
metadata = run()

# 找出包含极端振动的文件
extreme_files = [m for m in metadata if len(m['extreme_rms_indices']) > 0]
print(f"包含极端振动的文件: {len(extreme_files)} / {len(metadata)}")

# 统计极端窗口数量
total_extreme_windows = sum(len(m['extreme_rms_indices']) for m in extreme_files)
print(f"总极端窗口数: {total_extreme_windows}")

# 分析特定传感器
sensor_id = 'ST-VIC-C18-101-01'
sensor_data = [m for m in metadata if m['sensor_id'] == sensor_id]
extreme_count = sum(len(m['extreme_rms_indices']) for m in sensor_data)
print(f"{sensor_id}: {extreme_count} 个极端窗口")
```

### 用途 3: 按传感器分析

```python
from collections import defaultdict

metadata = run()

# 按传感器分组
by_sensor = defaultdict(list)
for item in metadata:
    by_sensor[item['sensor_id']].append(item)

# 统计每个传感器
for sensor_id, items in by_sensor.items():
    total_files = len(items)
    extreme_files = sum(1 for m in items if len(m['extreme_rms_indices']) > 0)
    avg_missing = np.mean([m['missing_rate'] for m in items]) * 100
    print(f"{sensor_id}:")
    print(f"  文件数: {total_files}")
    print(f"  极端文件: {extreme_files}")
    print(f"  平均缺失率: {avg_missing:.2f}%")
```

### 用途 4: 时间序列分析

```python
metadata = run()

# 按时间排序
sorted_by_time = sorted(metadata, 
                       key=lambda x: (x['month'], x['day'], x['hour']))

# 看每天的趋势
from collections import Counter
by_date = defaultdict(list)
for item in sorted_by_time:
    date_key = f"{item['month']}/{item['day']}"
    by_date[date_key].append(item)

for date, items in sorted(by_date.items()):
    extreme_count = sum(len(m['extreme_rms_indices']) for m in items)
    print(f"{date}: {extreme_count} 个极端窗口 (共 {len(items)} 条记录)")
```

---

## 高级选项

### 自定义阈值

```python
# 严格筛选（缺失率 < 1%）
metadata = run(threshold=0.01)

# 宽松筛选（缺失率 < 10%）
metadata = run(threshold=0.10)
```

### 禁用缓存

```python
# 强制完全重新计算
metadata = run(use_cache=False)
```

### 指定保存位置

```python
# 自定义元数据保存位置
metadata = run(save_path='/custom/path/metadata.json')
```

---

## 典型工作流

### 完整分析流程

```python
import numpy as np
from collections import defaultdict
from src.data_processer.statistics.vibration_io_process.workflow import run

print("=" * 60)
print("振动数据分析工作流")
print("=" * 60)

# 1. 加载数据
print("\n[步骤1] 加载振动数据...")
metadata = run(use_cache=True)
print(f"✓ 加载完成：{len(metadata)} 个文件")

# 2. 质量检查
print("\n[步骤2] 数据质量检查...")
missing_rates = [m['missing_rate'] for m in metadata]
print(f"  平均缺失率: {np.mean(missing_rates)*100:.2f}%")
print(f"  最大缺失率: {np.max(missing_rates)*100:.2f}%")
print(f"  完整文件数: {np.sum(np.array(missing_rates) < 0.01)}")

# 3. 极端振动分析
print("\n[步骤3] 极端振动分析...")
extreme_files = [m for m in metadata if len(m['extreme_rms_indices']) > 0]
print(f"  包含极端振动的文件: {len(extreme_files)} / {len(metadata)}")
total_extreme = sum(len(m['extreme_rms_indices']) for m in extreme_files)
print(f"  总极端窗口数: {total_extreme}")

# 4. 按传感器统计
print("\n[步骤4] 按传感器统计...")
by_sensor = defaultdict(list)
for item in metadata:
    by_sensor[item['sensor_id']].append(item)

for sensor_id in sorted(by_sensor.keys())[:5]:  # 显示前5个
    items = by_sensor[sensor_id]
    extreme_count = sum(len(m['extreme_rms_indices']) for m in items)
    print(f"  {sensor_id}: {len(items)} 个文件, {extreme_count} 个极端窗口")

print("\n✓ 分析完成\n")
```

---

## 故障排查

### 问题 1: 返回空列表

```python
metadata = run()
if not metadata:
    print("❌ 无处理结果")
    # 检查：
    # 1. 数据根目录是否配置正确
    # 2. 数据文件是否存在
    # 3. 传感器ID是否匹配
```

### 问题 2: 处理太慢

```python
# 使用缓存加速
metadata = run(use_cache=True)  # 快速

# 如果还是慢，可能原因：
# 1. 首次运行需要计算（正常）
# 2. 数据量很大
# 3. 磁盘I/O慢
```

### 问题 3: 内存不足

```python
# 原因：处理文件过多
# 解决方案：
# 1. 增加系统内存
# 2. 分批处理
# 3. 检查 TARGET_VIBRATION_SENSORS 配置
```

---

## 下一步

- 查看 [完整 API 文档](API.md) 了解所有参数
- 查看 [详细说明](README.md) 了解工作原理
- 查看 [高级用法](ADVANCED.md) 了解深层特性

---

## 获取帮助

### 查看处理日志

```python
# 运行时会打印详细的处理日志
metadata = run(force_recompute=True)

# 日志包含：
# - 文件统计
# - 筛选结果
# - RMS分析
# - 最终摘要
```

### 检查缓存文件

缓存文件路径由配置指定，通常为JSON格式，可以直接查看内容。

---

## 常见参数搭配

| 场景 | 参数搭配 |
|------|---------|
| 首次运行 | `use_cache=True, force_recompute=False` |
| 重复调用（快速） | `use_cache=True, force_recompute=False` |
| 数据更新后 | `use_cache=True, force_recompute=True` |
| 完全重新计算 | `use_cache=False` |
| 调试模式 | `force_recompute=True` + 检查输出日志 |
