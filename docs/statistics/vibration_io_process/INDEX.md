# 振动数据处理模块文档导航

欢迎使用 `vibration_io_process` 模块文档！

## 文档结构

### 📖 核心文档

| 文档 | 描述 | 适用场景 |
|------|------|---------|
| [README.md](README.md) | **完整使用指南** - 模块概述、工作流、主入口函数、示例 | 首次使用，全面了解 |
| [API.md](API.md) | **API 参考** - 函数签名、参数说明、数据结构、常量 | 编码时查阅 |
| [QUICKSTART.md](QUICKSTART.md) | **快速开始** - 30秒快速入门、基础示例、常见用途 | 快速上手 |
| [ADVANCED.md](ADVANCED.md) | **深度解析** - Metadata详解、生成过程、查询模式、应用示例 | **重点关注 metadata** |
| [REFACTOR.md](REFACTOR.md) | **架构说明** - 设计决策、性能分析、最佳实践 | 了解设计原理 |

---

## 快速导航

### 🎯 按需求查找

**"我想快速开始"**
→ 阅读 [QUICKSTART.md](QUICKSTART.md) 的前两个示例

**"我需要完整的函数参考"**
→ 查看 [API.md](API.md) 中的 `run()` 函数签名

**"我需要理解 metadata 的构成"**
→ 重点阅读 [ADVANCED.md](ADVANCED.md) 的 "Metadata 数据结构深度解析" 部分

**"我需要处理常见问题"**
→ 查看 [README.md](README.md) 的 "常见问题" 或 [QUICKSTART.md](QUICKSTART.md) 的 "故障排查"

**"我想了解性能特性"**
→ 查看 [REFACTOR.md](REFACTOR.md) 的 "性能特性" 或 [README.md](README.md) 的 "性能特性"

**"我想学习最佳实践"**
→ 阅读 [README.md](README.md) 的 "最佳实践" 或 [REFACTOR.md](REFACTOR.md) 的 "最佳实践建议"

---

## Metadata 关键概念速查

### 字段说明

```python
{
    'sensor_id': str,              # 传感器ID，唯一标识一个物理位置
    'month': int,                  # 月份 (1-12)
    'day': int,                    # 日期 (1-31)
    'hour': int,                   # 小时 (0-23)
    'file_path': str,              # 源数据文件完整路径
    'actual_length': int,          # 实际采样点数（不是字节）
    'missing_rate': float,         # 缺失率 (0.0-1.0)
    'extreme_rms_indices': list    # 极端振动窗口索引 [0, 5, 12, ...]
}
```

### 核心公式

```
缺失率 = 1 - (actual_length / expected_length)
       = 1 - (actual_length / 180000)

极端判定 = RMS值 >= 95%分位数阈值

窗口索引到时间 = index × 60 秒
```

### 数据来源

| 字段 | 来自 | 说明 |
|------|------|------|
| sensor_id, month, day, hour | Step 0 | 从文件路径解析 |
| file_path | Step 1 | 筛选后的文件路径 |
| actual_length, missing_rate | Step 1 | 缺失率计算结果 |
| extreme_rms_indices | Step 2 | RMS统计与极端识别 |

详细说明 → [ADVANCED.md](ADVANCED.md#metadata-数据结构深度解析)

---

## 学习路径

### 初级开发者
1. [QUICKSTART.md](QUICKSTART.md) - 快速入门（10分钟）
2. [README.md](README.md) 的前半部分 - 工作流理解（15分钟）
3. 动手写第一个脚本（15分钟）

### 中级开发者
1. [README.md](README.md) - 完整阅读（30分钟）
2. [API.md](API.md) - 参考查阅（20分钟）
3. [ADVANCED.md](ADVANCED.md) 的查询模式部分（20分钟）

### 高级开发者
1. [ADVANCED.md](ADVANCED.md) - 完整阅读（40分钟）
2. [REFACTOR.md](REFACTOR.md) - 架构理解（30分钟）
3. 代码源文件阅读 - 深度理解（不限）

---

## 常见任务速查表

### 任务 1: 加载和基础处理

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

# 加载元数据
metadata = run(use_cache=True)

# 查看结果数量
print(f"处理了 {len(metadata)} 个文件")
```

**相关文档**: [QUICKSTART.md](QUICKSTART.md#基础使用)

### 任务 2: 理解返回数据结构

```python
# 每条元数据包含这些字段
for item in metadata:
    print(item['sensor_id'])           # 传感器ID
    print(item['actual_length'])       # 采样点数
    print(item['missing_rate'])        # 缺失率
    print(item['extreme_rms_indices']) # 极端窗口
```

**相关文档**: [ADVANCED.md](ADVANCED.md#metadata-数据结构深度解析)

### 任务 3: 找极端振动

```python
# 找出包含极端振动的记录
extreme_records = [m for m in metadata 
                   if len(m['extreme_rms_indices']) > 0]
print(f"包含极端振动的记录: {len(extreme_records)}")
```

**相关文档**: [QUICKSTART.md](QUICKSTART.md#用途-2-极端振动分析)

### 任务 4: 按传感器分析

```python
from collections import defaultdict

# 按传感器分组
by_sensor = defaultdict(list)
for item in metadata:
    by_sensor[item['sensor_id']].append(item)

# 统计每个传感器
for sensor_id, items in by_sensor.items():
    print(f"{sensor_id}: {len(items)} 条记录")
```

**相关文档**: [QUICKSTART.md](QUICKSTART.md#用途-3-按传感器分析)

### 任务 5: 过滤高质量数据

```python
# 缺失率 < 1% 的高质量数据
high_quality = [m for m in metadata if m['missing_rate'] < 0.01]
print(f"高质量数据: {len(high_quality)} / {len(metadata)}")
```

**相关文档**: [README.md](README.md#常见问题) / [QUICKSTART.md](QUICKSTART.md#用途-1-数据质量检查)

---

## 文档统计

| 文档 | 行数 | 大小 | 内容 |
|------|------|------|------|
| README.md | 540+ | 14.6 KB | 完整指南 |
| API.md | 276+ | 8.9 KB | API参考 |
| QUICKSTART.md | 324+ | 8.4 KB | 快速开始 |
| ADVANCED.md | 498+ | 14.4 KB | **深度Metadata解析** |
| REFACTOR.md | 315+ | 9.2 KB | 架构设计 |
| **INDEX.md** | 当前文件 | 导航索引 | |

**总计**: 1900+ 行完整文档

---

## 关键概念速查

### 什么是 Metadata？
Metadata 是对每个1小时振动数据文件的完整描述，包含时间、质量、特征信息。
→ 详见 [ADVANCED.md](ADVANCED.md#metadata-数据结构深度解析)

### 什么是 missing_rate？
缺失率表示文件中的数据相对于预期的 180000 采样点的缺失比例。
→ 详见 [ADVANCED.md](ADVANCED.md#缺失率-float)

### 什么是 extreme_rms_indices？
极端窗口索引列表，记录该文件中哪些60秒窗口的RMS值超过95%分位数阈值。
→ 详见 [ADVANCED.md](ADVANCED.md#extreme_rms_indices-list)

### 工作流的三个步骤是什么？
1. **Step 0**: 文件发现 - 收集所有 .VIC 文件
2. **Step 1**: 缺失率筛选 - 过滤质量差的文件
3. **Step 2**: RMS统计 - 识别极端振动
→ 详见 [README.md](README.md#工作流架构)

### 如何使用缓存？
设置 `use_cache=True`（默认值）。首次运行计算并缓存结果，后续调用直接返回缓存。
→ 详见 [README.md](README.md#缓存机制) 或 [API.md](API.md)

---

## 常见问题快速答案

**Q: Metadata 一定包含什么字段？**
A: 8个字段：sensor_id, month, day, hour, file_path, actual_length, missing_rate, extreme_rms_indices
→ [ADVANCED.md 第64-82行](ADVANCED.md#元数据字段详解)

**Q: actual_length 的单位是什么？**
A: 采样点数（不是字节，不是秒）。对于完整的1小时数据，值为 180000（50Hz × 3600s）
→ [ADVANCED.md 第111-119行](ADVANCED.md#actual_length-int)

**Q: extreme_rms_indices 的 [5, 12] 表示什么？**
A: 第5和第12个60秒窗口的RMS值超过95%阈值。索引5对应时间 300-360秒（5分钟位置）
→ [ADVANCED.md 第143-165行](ADVANCED.md#极端rms_indices-list)

**Q: 缺失率是如何计算的？**
A: missing_rate = 1 - (actual_length / 180000)。例如actual_length=178500时，missing_rate ≈ 0.00833 (0.83%)
→ [ADVANCED.md 第96-115行](ADVANCED.md#缺失率-float)

**Q: 处理需要多长时间？**
A: 1000个文件约200-250秒。启用缓存后的后续调用 <1秒
→ [README.md 的性能特性](README.md#性能特性)

---

## 版本信息

- **模块版本**: vibration_io_process 1.0
- **文档版本**: 1.0
- **最后更新**: 2026-02-26
- **维护状态**: ✅ 主动维护

---

## 相关资源

- **主模块**: `src/data_processer/statistics/workflow.py`
- **配置文件**: `src/config/data_processer/statistics/vibration_io_process/config.py`
- **数据解包**: `src/data_processer/io_unpacker.py`
- **风数据模块**: `docs/statistics/wind_data_io_process/`

---

## 文档使用建议

1. **首次查阅**: 按照上面的 "按需求查找" 选择合适的文档
2. **深度学习**: 按照 "学习路径" 逐个阅读
3. **快速查询**: 使用 "常见任务速查表"
4. **疑难解答**: 查看相应文档的 "常见问题" 部分
5. **架构理解**: 阅读 REFACTOR.md 中的设计决策

---

**💡 提示**: 所有文档都已集成完整的目录和交叉引用，可直接跳转查看。

