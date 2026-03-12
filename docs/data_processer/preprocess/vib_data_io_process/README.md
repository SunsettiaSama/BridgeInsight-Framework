# 振动数据处理模块 (Vibration I/O Process)

## 📋 概述

本模块负责振动加速度数据的获取、质量筛选和统计分析，包含完整的三步工作流。通过缺失率筛选和RMS极端值识别，为后续分析提供高质量的振动数据。

## 📁 模块结构

```
vibration_io_process/
├── workflow.py                      # 完整工作流（主入口）
├── step0_get_vib_data.py            # Step 0: 获取振动数据文件
├── step1_lackness_filter.py         # Step 1: 缺失率筛选
├── step2_rms_statistics.py          # Step 2: RMS统计和极端值识别
├── step3_dominant_freq_statistics.py # Step 3: 主频分布统计
└── README.md                        # 本文档
```

## 🎯 工作流程

### Step 0: 获取振动数据文件

**功能**:
- 遍历振动数据根目录
- 根据目标传感器ID筛选文件
- 匹配 `.VIC` 后缀的文件
- 返回符合条件的文件路径列表

**关键参数**:
- 采样频率: 50 Hz
- 文件后缀: `.VIC`
- 预期数据长度: 180000 点 (50Hz × 3600s)

### Step 1: 缺失率筛选

**功能**:
- 计算每个文件的实际数据长度
- 计算缺失率 = 1 - (实际长度 / 预期长度)
- 筛选出缺失率低于阈值的文件
- 生成详细的统计报告

**关键参数**:
- 缺失率阈值: 5% (默认)
- 预期文件长度: 180000 采样点
- 返回: 筛选通过的文件路径和统计信息

**输出统计**:
```
总样本数: N
平均缺失率: X.XX%
符合条件的文件 (≤5%): M 个
不符合条件的文件 (>5%): K 个
```

### Step 2: RMS统计和极端值识别

**功能**:
- 对每个文件进行滑动窗口RMS计算
- 时间窗口: 60秒 (3000个采样点)
- 动态计算全局RMS的95%分位值作为阈值
- 识别并记录超过阈值的极端振动窗口索引

**RMS计算公式**:
```
RMS = √(mean(x²))
其中 x 是单位时间窗口内的加速度样本
```

**关键参数**:
- 时间窗口: 60秒
- 窗口大小: 3000 采样点
- 极端值阈值: 95% 分位值

**输出统计**:
```
总样本数: N
平均RMS: X.XXXX m/s²
标准差: Y.YYYY m/s²
最小值/最大值: A.AAAA / B.BBBB m/s²

RMS阈值 (95%分位值): Z.ZZZZ m/s²
小于阈值样本: P% (正常)
大于等于阈值样本: Q% (极端)
包含极端振动的文件: M / N (R.RR%)
```

### Step 3: 主频分布统计

**功能**:
- 对每个文件进行滑动窗口主频计算
- 时间窗口: 60秒 (3000个采样点)
- 使用 Welch 方法计算功率谱密度 (PSD)
- 提取指定频率范围内的主频
- 计算所有主频的95%分位值
- 保存统计结果到 JSON 文件

**主频提取步骤**:
```
1. 将振动信号分段（60秒/段）
2. 对每段使用 Welch 方法计算 PSD
3. 在 0-25 Hz 频率范围内找到最大功率对应的频率
4. 收集所有窗口的主频
5. 计算统计特性和95%分位值
```

**关键参数**:
- 时间窗口: 60秒
- 窗口大小: 3000 采样点
- 频率范围: 0-25 Hz
- FFT分辨率: 1024
- 极端值阈值: 95% 分位值

**输出统计**:
```
总样本数: N
频率范围: X.XX ~ Y.YY Hz
平均主频: A.AAAA Hz
标准差: B.BBBB Hz
中位数: C.CCCC Hz

主频95%分位值: D.DDDD Hz
```

**输出文件格式**:
JSON 文件包含以下内容:
```json
{
    "all_dominant_frequencies": [f1, f2, f3, ...],
    "freq_p95": 12.3456,
    "freq_stats": {
        "min": 0.1234,
        "max": 24.5678,
        "mean": 8.9012,
        "std": 3.4567,
        "median": 9.0123,
        "p95": 12.3456,
        "total_samples": 12345
    }
}
```

## 🚀 完整工作流使用

### 使用示例

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

# 默认参数运行（推荐用于首次运行或数据更新）
metadata = run(
    threshold=0.05,           # 缺失率阈值 5%
    expected_length=180000,   # 预期文件长度
    use_cache=True,           # 使用缓存加速
    force_recompute=False     # 不强制重新计算
)

print(f"✓ 筛选完成！共 {len(metadata)} 个高质量文件")

# 查看示例结果
for item in metadata[:3]:
    print(f"\n文件: {item['file_path']}")
    print(f"  传感器: {item['sensor_id']}")
    print(f"  时间: {item['month']}/{item['day']} {item['hour']}:00")
    print(f"  缺失率: {item['missing_rate']*100:.2f}%")
    print(f"  极端RMS窗口数: {len(item['extreme_rms_indices'])}")
```

### 工作流输出元数据字段

每个元数据项包含以下信息:

| 字段 | 类型 | 说明 |
|------|------|------|
| `sensor_id` | str | 传感器ID |
| `month` | str | 月份 (e.g., '09') |
| `day` | str | 日期 (e.g., '01') |
| `hour` | str | 小时 (e.g., '12') |
| `file_path` | str | 完整文件路径 |
| `actual_length` | int | 实际数据长度 |
| `missing_rate` | float | 缺失率 (0-1) |
| `extreme_rms_indices` | list | 极端RMS窗口索引列表 |

## ⚙️ 配置说明

配置文件位置: `src/config/data_processer/statistics/vibration_io_process/config.py`

### 主要配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FS` | 50 | 振动信号采样频率 (Hz) |
| `TIME_WINDOW` | 60.0 | RMS计算时间窗口 (秒) |
| `ALL_VIBRATION_ROOT` | ... | 振动数据根目录路径 |
| `VIBRATION_FILE_SUFFIX` | ".VIC" | 振动数据文件后缀 |
| `MISSING_RATE_THRESHOLD` | 0.05 | 缺失率阈值 (5%) |
| `EXPECTED_LENGTH` | 180000 | 预期文件长度 (50Hz × 3600s) |

### 目标传感器列表

```python
TARGET_VIBRATION_SENSORS = [
    'ST-VIC-C18-101-01',  # 北索塔边跨1/4跨 面内上游
    'ST-VIC-C18-101-02',  # 北索塔边跨1/4跨 面外上游
    'ST-VIC-C18-201-01',  # 北索塔跨中1/4跨 面内上游
    'ST-VIC-C18-201-02',  # 北索塔跨中1/4跨 面外上游
    'ST-VIC-C34-101-01',  # 南索塔边跨1/4跨 面内上游
    'ST-VIC-C34-101-02',  # 南索塔边跨1/4跨 面外上游
    # ... 其他传感器
]
```

## 💾 缓存机制

工作流支持两级缓存:

1. **元数据缓存** (`metadata_cache.json`)
   - 存储筛选后的元数据
   - 加速重复运行

2. **工作流报告** (`metadata_cache_report.txt`)
   - 记录处理过程和统计信息
   - 便于追踪和调试

### 缓存参数匹配

缓存只在以下条件下使用:
- `threshold` 与缓存参数匹配
- `expected_length` 与缓存参数匹配
- `use_cache=True` 且 `force_recompute=False`

## 📊 数据质量指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| 缺失率 | 文件数据完整度 | ≤ 5% |
| RMS阈值 (95%分位) | 极端振动判定标准 | 动态计算 |
| 主频95%分位值 | 极端主频判定标准 | 动态计算 |
| 极端文件比例 | 包含极端振动的文件占比 | 通常 < 30% |
| 通过率 | 通过缺失率筛选的文件占比 | 通常 > 90% |

## 🔍 关键概念

### 缺失率

反映文件数据的完整度:
```
缺失率 = 1 - (实际长度 / 预期长度)
```

- 缺失率 0%: 数据完全
- 缺失率 5%: 允许阈值
- 缺失率 > 5%: 数据不完整，被筛除

### 极端RMS

基于统计分位数识别的异常振动:
- 计算全局RMS的95%分位值
- 超过该值的窗口标记为极端
- 用于后续极端事件分析

## 🔄 与其他模块的关系

```
vibration_io_process (本模块)
    ▼ 输出: vib_metadata
wind_data_io_process
    ▼ 输出: wind_metadata
fig_generation
    ▼ 生成: 风-振动关系图表
```

- **输出给**: `wind_data_io_process` 用于时间戳对齐
- **被使用于**: 后续的特征分析和数据可视化

## 🛠️ 多进程优化

工作流使用 `ProcessPoolExecutor` 并行处理:
- Step 1 缺失率计算: 4 workers
- Step 2 RMS计算: 4 workers
- 显著加速处理速度（尤其是文件数量多时）

## 📝 注意事项

1. **首次运行**: 第一次运行会全量处理所有文件，耗时较长（可能几分钟到十几分钟），后续运行使用缓存会快速返回
2. **参数变更**: 修改 `threshold` 或 `expected_length` 后，需设置 `force_recompute=True` 重新计算
3. **内存占用**: 处理大量文件时内存占用较高，请确保系统有充足内存
4. **数据目录**: 确保 `ALL_VIBRATION_ROOT` 配置正确指向实际数据目录

## 🔗 相关文档

- [风数据处理工作流文档](../wind_data_io_process/README.md)
- [数据解析模块](../../io_unpacker.py)
- [传感器配置](../../../../src/config/sensor_config.py)

---

**更新日期**: 2026-03-11  
**维护者**: 振动特性研究团队
