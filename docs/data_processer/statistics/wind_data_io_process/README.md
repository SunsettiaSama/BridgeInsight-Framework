# 风数据处理模块 (Wind Data I/O Process)

## 📋 概述

本模块负责风速、风向、风攻角数据的获取、筛选和处理，与振动数据处理模块 `vibration_io_process` 采用相同的架构设计。

## 📁 模块结构

```
wind_data_io_process/
├── __init__.py
├── step0_get_wind_data.py      # Step 0: 获取风数据文件
└── README.md                    # 本文档
```

## 🎯 功能说明

### Step 0: 获取风数据文件

**文件**: `step0_get_wind_data.py`

**功能**:
- 遍历风数据根目录（UAN目录）
- 根据目标风传感器ID列表筛选文件
- 匹配 `.UAN` 后缀的文件
- 返回符合条件的文件路径列表

**使用方法**:

```python
from src.data_processer.statistics.wind_data_io_process.step0_get_wind_data import get_all_wind_files

# 获取所有风数据文件
all_wind_files = get_all_wind_files()
print(f"共获取 {len(all_wind_files)} 个风数据文件")

# 自定义参数
custom_sensors = ['ST-UAN-T01-003-01', 'ST-UAN-T02-003-01']  # 只获取塔顶传感器
tower_wind_files = get_all_wind_files(target_sensor_ids=custom_sensors)
```

## ⚙️ 配置说明

配置文件位置: `src/config/data_processer/statistics/wind_data_io_process/config.py`

### 主要配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FS` | 1 | 风速采样频率（Hz） |
| `TIME_WINDOW` | 60.0 | 统计时间窗口（秒） |
| `ALL_WIND_ROOT` | ... | 风数据根目录路径 |
| `WIND_FILE_SUFFIX` | ".UAN" | 风数据文件后缀 |
| `MISSING_RATE_THRESHOLD` | 0.05 | 缺失率阈值（5%） |
| `EXPECTED_LENGTH` | 3600 | 预期数据长度（1Hz × 3600s） |

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

## 📊 数据格式

### 风数据文件 (.UAN)

每个 `.UAN` 文件包含一小时的风速、风向、风攻角数据：
- **采样频率**: 1 Hz
- **数据长度**: 3600 个采样点（1小时）
- **数据内容**: 
  - 风速 (m/s)
  - 风向 (度, 0-360)
  - 风攻角 (度)

### 文件命名规则

```
传感器ID_年_月_日_小时.UAN
例如: ST-UAN-T01-003-01_2024_09_01_12.UAN
```

## 🔄 数据处理流程

```
┌─────────────────────────────────────┐
│  Step 0: 获取风数据文件              │
│  - 遍历 UAN 目录                     │
│  - 筛选目标传感器                    │
│  - 返回文件路径列表                  │
└─────────────────────────────────────┘
              │
              ▼
    (后续步骤待开发)
```

## 🚀 快速开始

### 测试 Step 0

```python
# 直接运行 step0 文件
python -m src.data_processer.statistics.wind_data_io_process.step0_get_wind_data
```

或在代码中使用：

```python
from src.data_processer.statistics.wind_data_io_process.step0_get_wind_data import get_all_wind_files

# 获取所有风数据文件
wind_files = get_all_wind_files()

# 按传感器分类
from collections import defaultdict
sensor_files = defaultdict(list)
for fp in wind_files:
    for sensor_id in TARGET_WIND_SENSORS:
        if sensor_id in fp:
            sensor_files[sensor_id].append(fp)
            break

# 查看各传感器的文件数量
for sensor_id, files in sensor_files.items():
    print(f"{sensor_id}: {len(files)} 个文件")
```

## 📈 与振动数据模块的对应关系

| 振动数据模块 | 风数据模块 | 说明 |
|-------------|-----------|------|
| `vibration_io_process` | `wind_data_io_process` | 模块名 |
| `.VIC` | `.UAN` | 文件后缀 |
| 50 Hz | 1 Hz | 采样频率 |
| 180000 点/小时 | 3600 点/小时 | 预期长度 |
| `TARGET_VIBRATION_SENSORS` | `TARGET_WIND_SENSORS` | 传感器列表 |

## 🔧 后续开发计划

- [ ] Step 1: 缺失率筛选
- [ ] Step 2: 风参数统计分析
- [ ] 完整工作流 (workflow.py)
- [ ] 风参数特征提取
- [ ] 紊流度计算
- [ ] 极端风事件识别

## 📝 注意事项

1. **数据完整性**: 风数据文件可能存在数据缺失，需要通过 Step 1 进行筛选
2. **传感器位置**: 不同位置的传感器测量的风场特性可能差异较大
3. **数据质量**: 建议先进行数据质量检查，过滤异常值

## 🔗 相关文档

- [振动数据处理工作流文档](../../../docs/data_processer/statistics/vibration_io_process_workflow.md)
- [数据解析模块 (io_unpacker)](../../io_unpacker.py)

---

**创建日期**: 2024-01-29  
**维护者**: 振动特性研究团队
