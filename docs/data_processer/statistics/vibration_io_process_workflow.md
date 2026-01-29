# 振动数据处理工作流 (Vibration I/O Process Workflow)

## 📋 概述

`vibration_io_process/workflow.py` 是一个完整的振动数据处理工作流系统，负责从原始振动数据文件中筛选、分析和识别极端振动事件。该工作流采用模块化设计，按照多个步骤依次处理数据，最终输出包含完整样本信息的纯净元数据。

## 🎯 核心功能

工作流系统实现以下核心功能：

1. **数据采集**：自动获取所有振动传感器的数据文件
2. **质量筛选**：基于缺失率过滤低质量数据文件
3. **极端振动识别**：通过RMS统计分析识别极端振动事件
4. **元数据生成**：构建包含完整信息的样本元数据列表
5. **缓存管理**：支持结果缓存，避免重复计算
6. **报告生成**：自动生成详细的处理报告

## 🔄 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    振动数据处理工作流                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Step 0: 获取所有振动文件路径          │
        │  - 遍历数据目录                       │
        │  - 根据传感器ID筛选                   │
        │  - 返回: all_file_paths (1000+ 文件)  │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Step 1: 缺失率筛选                   │
        │  - 并行计算每个文件的数据长度           │
        │  - 计算缺失率 (1 - actual/expected)   │
        │  - 过滤缺失率 > 5% 的文件              │
        │  - 返回: filtered_paths (950+ 文件)   │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Step 2: RMS统计分析与极端振动识别     │
        │  - 并行计算每个文件的RMS值             │
        │  - 计算95%分位值作为阈值               │
        │  - 识别超过阈值的时间窗口              │
        │  - 返回: extreme_indices              │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  元数据构建                           │
        │  - 解析文件路径提取传感器、时间信息     │
        │  - 整合缺失率信息                     │
        │  - 整合极端振动索引                   │
        │  - 生成纯净的元数据列表                │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  输出结果                             │
        │  - metadata: 纯净的元数据列表          │
        │  - 处理参数由 ReportCollector 管理    │
        └──────────────────────────────────────┘
```

## 📊 输出数据结构

### 主要输出：metadata (纯净的元数据列表)

```python
[
    {
        "sensor_id": "ST-VIC-C34-101-01",      # 传感器ID
        "month": "09",                          # 月份
        "day": "01",                            # 日期
        "hour": "12",                           # 小时
        "file_path": "F:\\...\\file.VIC",      # 完整文件路径
        "actual_length": 180000,                # 实际数据长度（采样点数）
        "missing_rate": 0.02,                   # 缺失率（0-1之间）
        "extreme_rms_indices": [5, 12, 23, 35] # 极端振动时间窗口索引列表
    },
    ...
]
```

### 处理参数（存储在 ReportCollector 中）

```python
report_collector.process_params = {
    "missing_rate_threshold": 0.05,           # 缺失率阈值
    "expected_length": 180000,                # 预期数据长度 (50Hz * 3600s)
    "total_files": 1000,                      # 原始文件总数
    "filtered_files": 950,                    # 筛选后文件数
    "rms_threshold_95": 0.1234,               # RMS 95%分位值阈值
    "files_with_extreme_vibration": 45        # 包含极端振动的文件数
}
```

## 💻 使用方法

### 基本使用

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

# 运行完整工作流（使用默认参数）
metadata = run()

# 访问样本信息
for item in metadata:
    print(f"文件: {item['file_path']}")
    print(f"传感器: {item['sensor_id']}")
    print(f"极端振动窗口数: {len(item['extreme_rms_indices'])}")
```

### 自定义参数

```python
# 自定义缺失率阈值和预期长度
metadata = run(
    threshold=0.03,           # 缺失率阈值 3%
    expected_length=180000,   # 预期长度
    use_cache=True,           # 使用缓存
    force_recompute=False     # 不强制重新计算
)
```

### 强制重新计算

```python
# 强制重新计算（忽略缓存）
metadata = run(force_recompute=True)
```

### 禁用缓存

```python
# 不使用缓存系统
metadata = run(use_cache=False)
```

### 访问处理参数

```python
from src.data_processer.statistics.vibration_io_process.workflow import report_collector

# 运行工作流
metadata = run()

# 访问处理参数
total_files = report_collector.get_param('total_files')
rms_threshold = report_collector.get_param('rms_threshold_95')
files_with_extreme = report_collector.get_param('files_with_extreme_vibration')

print(f"总文件数: {total_files}")
print(f"RMS阈值: {rms_threshold:.4f} m/s²")
print(f"包含极端振动的文件: {files_with_extreme}")
```

## 📁 文件结构

```
vibration_io_process/
├── workflow.py                    # 主工作流文件
├── step0_get_vib_data.py         # Step 0: 获取振动文件
├── step1_lackness_filter.py      # Step 1: 缺失率筛选
├── step2_rms_statistics.py       # Step 2: RMS统计与极端振动识别
└── config.py                     # 配置文件（在 config 目录下）
```

## ⚙️ 配置参数

配置文件位置：`src/config/data_processer/statistics/vibration_io_process/config.py`

### 主要配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MISSING_RATE_THRESHOLD` | 0.05 | 缺失率阈值（5%） |
| `EXPECTED_LENGTH` | 180000 | 预期数据长度（50Hz × 3600s） |
| `FS` | 50 | 采样频率（Hz） |
| `TIME_WINDOW` | 60.0 | RMS计算时间窗口（秒） |
| `ALL_VIBRATION_ROOT` | ... | 振动数据根目录 |
| `FILTER_RESULT_PATH` | ... | 筛选结果保存路径 |
| `WORKFLOW_CACHE_PATH` | ... | 工作流缓存路径 |
| `TARGET_VIBRATION_SENSORS` | [...] | 目标传感器ID列表 |

## 🔍 各步骤详解

### Step 0: 获取所有振动文件

**模块**：`step0_get_vib_data.py`

**功能**：
- 遍历振动数据根目录
- 根据目标传感器ID列表筛选文件
- 匹配 `.VIC` 后缀的文件

**输出**：
```python
all_file_paths = [
    "F:\\...\\ST-VIC-C34-101-01_2024_09_01_00.VIC",
    "F:\\...\\ST-VIC-C34-101-01_2024_09_01_01.VIC",
    ...
]
```

### Step 1: 缺失率筛选

**模块**：`step1_lackness_filter.py`

**功能**：
- 使用多进程并行计算每个文件的实际数据长度
- 计算缺失率：`missing_rate = 1 - (actual_length / expected_length)`
- 过滤缺失率超过阈值的文件

**输出**：
```python
filtered_paths = [...]  # 筛选后的文件路径
statistics = {
    'all_lengths': np.array([...]),           # 所有文件的长度
    'all_missing_rates': np.array([...]),     # 所有文件的缺失率
    'filtered_indices': [0, 1, 3, 5, ...]    # 通过筛选的文件索引
}
```

**统计报告示例**：
```
====================================================================
                    样本缺失率筛选报告
====================================================================
1. 筛选参数：
   - 缺失率阈值: 5.0%
   - 预期单文件长度: 180000 (50Hz * 60s * 60m)

2. 处理统计（总计: 1000 个文件）:
   - 平均缺失率: 2.34%
   - 符合条件的文件 (≤5.0%): 950 (95.00%)
   - 不符合条件的文件 (>5.0%): 50 (5.00%)
====================================================================
```

### Step 2: RMS统计分析与极端振动识别

**模块**：`step2_rms_statistics.py`

**功能**：
- 使用多进程并行计算每个文件的RMS值（按时间窗口）
- 计算所有RMS值的95%分位值作为极端振动阈值
- 识别每个文件中超过阈值的时间窗口索引

**输出**：
```python
file_paths = [...]  # 文件路径（原样返回）
rms_statistics = {
    'all_file_rms': [                        # 每个文件的RMS数组列表
        np.array([0.05, 0.08, 0.12, ...]),
        np.array([0.06, 0.07, 0.09, ...]),
        ...
    ],
    'extreme_indices': [                     # 每个文件的极端振动窗口索引
        [5, 12, 23],
        [],
        [8, 15],
        ...
    ],
    'rms_threshold_95': 0.1234               # 95%分位值阈值
}
```

**统计报告示例**：
```
====================================================================
                    RMS极端振动识别报告
====================================================================
1. 基本统计（所有样本）：
   - 总样本数: 57000
   - 平均RMS: 0.0876 m/s²
   - 标准差: 0.0234 m/s²
   - 最小值: 0.0012 m/s²
   - 最大值: 0.5678 m/s²

2. 极端振动阈值（95%分位值）：
   - RMS阈值: 0.1234 m/s²

3. 极端振动识别统计：
   - 小于阈值样本数: 54150 (95.00%)
   - 大于等于阈值样本数: 2850 (5.00%)
   - 包含极端振动的文件数: 45 / 950 (4.74%)
====================================================================
```

## 🎨 设计特点

### 1. 纯净的数据结构

- **metadata** 只包含样本信息，不包含处理参数
- 每个元数据项包含完整的样本信息
- 调用者只需访问 metadata 即可获取所有必要数据

### 2. 处理参数分离

- 处理参数由 `ReportCollector` 统一管理
- 不污染元数据结构
- 需要时可通过 `report_collector.get_param()` 访问

### 3. 模块化设计

- 每个 step 职责单一
- step 之间通过文件路径列表传递
- 便于扩展和维护

### 4. 并行处理

- Step 1 和 Step 2 均使用多进程并行处理
- 显著提升处理速度
- 适合大规模数据处理

### 5. 缓存系统

- 支持结果缓存，避免重复计算
- 自动检查参数匹配
- 参数不匹配时自动重新计算

## 📈 性能优化

### 多进程并行处理

```python
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_func, fp): fp for fp in file_paths}
    for future in tqdm(as_completed(futures), total=len(file_paths)):
        result = future.result()
```

- Step 1 缺失率计算：~1000 文件，约 2-3 分钟
- Step 2 RMS 统计：~950 文件，约 5-8 分钟
- 总处理时间：约 10 分钟（取决于硬件配置）

### 缓存机制

- 首次运行：完整计算（~10分钟）
- 后续运行（参数匹配）：从缓存读取（< 1秒）
- 缓存文件：JSON 格式，易于查看和调试

## 🔧 常见问题

### Q1: 如何查看处理报告？

**A**: 报告自动保存到与缓存文件相同的目录：

```python
# 缓存文件: results/workflow_cache.json
# 报告文件: results/workflow_cache_report.txt
```

### Q2: 如何修改缺失率阈值？

**A**: 通过参数传递：

```python
metadata = run(threshold=0.03)  # 修改为 3%
```

### Q3: 如何只处理特定传感器的数据？

**A**: 修改配置文件中的 `TARGET_VIBRATION_SENSORS`：

```python
TARGET_VIBRATION_SENSORS = [
    'ST-VIC-C34-101-01',
    'ST-VIC-C34-101-02',
]
```

### Q4: 缓存文件过大怎么办？

**A**: 可以禁用缓存或手动删除旧缓存：

```python
# 方法1：禁用缓存
metadata = run(use_cache=False)

# 方法2：强制重新计算（会覆盖旧缓存）
metadata = run(force_recompute=True)
```

### Q5: 如何获取极端振动事件的详细信息？

**A**: 通过 metadata 中的 `extreme_rms_indices` 获取：

```python
for item in metadata:
    if len(item['extreme_rms_indices']) > 0:
        print(f"文件: {item['file_path']}")
        print(f"极端振动窗口: {item['extreme_rms_indices']}")
        print(f"窗口数量: {len(item['extreme_rms_indices'])}")
```

## 📝 示例：完整使用流程

```python
from src.data_processer.statistics.vibration_io_process.workflow import run, report_collector

# 1. 运行工作流
print("开始运行振动数据处理工作流...")
metadata = run(
    threshold=0.05,
    use_cache=True,
    force_recompute=False
)

# 2. 查看基本统计
print(f"\n处理完成！")
print(f"总文件数: {report_collector.get_param('total_files')}")
print(f"筛选后文件数: {report_collector.get_param('filtered_files')}")
print(f"RMS阈值: {report_collector.get_param('rms_threshold_95'):.4f} m/s²")

# 3. 分析极端振动事件
extreme_files = [item for item in metadata if len(item['extreme_rms_indices']) > 0]
print(f"\n包含极端振动的文件: {len(extreme_files)}")

# 4. 查看示例数据
print("\n前3个极端振动样本:")
for i, item in enumerate(extreme_files[:3], 1):
    print(f"\n{i}. 传感器: {item['sensor_id']}")
    print(f"   时间: {item['month']}/{item['day']} {item['hour']}:00")
    print(f"   极端振动窗口: {item['extreme_rms_indices']}")

# 5. 按传感器统计
from collections import defaultdict
sensor_stats = defaultdict(int)
for item in extreme_files:
    sensor_stats[item['sensor_id']] += len(item['extreme_rms_indices'])

print("\n各传感器极端振动事件数:")
for sensor_id, count in sorted(sensor_stats.items()):
    print(f"  {sensor_id}: {count} 次")
```

## 🔗 相关文档

- [RMS 统计分析文档](./rms_statistics.md)
- [数据处理总览](../README.md)
- [配置说明](../../config/README.md)

## 📞 技术支持

如有问题或建议，请联系项目维护者。

---

**文档版本**: 1.0  
**最后更新**: 2024-01-29  
**维护者**: 振动特性研究团队
