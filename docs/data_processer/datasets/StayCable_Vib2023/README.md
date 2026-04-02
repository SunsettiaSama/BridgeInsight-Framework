# StayCable_Vib2023 数据集模块

## 📋 模块概述

`src/data_processer/datasets/StayCable_Vib2023/` 是苏通大桥拉索振动2023数据集的专用加载模块。该模块设计用于全年统计分析与振动识别，面向深度学习训练与全量数据识别。

### 核心特性

- **多拉索支持**：同时加载多根拉索（面内+面外传感器对）的振动数据
- **精确时间对齐**：振动数据与风数据按时间戳严格对齐（三路对应）
- **时序排序**：支持按时间顺序排列样本（time_ordered=True），保留时序连续性
- **智能缓存机制**：样本索引缓存（含配置指纹校验），避免重复构建索引
- **灵活划分**：支持随机划分或时序划分的训练/验证集
- **DL识别结果集成**：支持加载和应用深度学习预识别结果

---

## 📂 目录结构

```
src/data_processer/datasets/StayCable_Vib2023/
├── __init__.py                        # 模块导出
└── StayCableVib2023Dataset.py       # 主数据集实现

src/config/data_processer/datasets/StayCableVib2023Dataset/
└── StayCableVib2023Config.py         # 配置类
```

---

## 🏗 架构设计

### 分层架构

```
[用户代码]
    ↓
[StayCableVib2023Dataset]
    ↓
[_SampleRecord (样本索引)]
    ↓
[VICWindowExtractor / parse_single_metadata_to_wind_data]
    ↓
[原始数据文件]
```

### 核心类关系

```
StayCableVib2023Config (配置类)
    ↓
StayCableVib2023Dataset (主数据集)
    ├─持有→ List[_SampleRecord] (样本索引列表)
    └─使用→ VICWindowExtractor (振动数据提取器)
              parse_single_metadata_to_wind_data (风数据解析器)
```

---

## 📖 核心类详解

### 1. StayCableVib2023Dataset

**文件**: `src/data_processer/datasets/StayCable_Vib2023/StayCableVib2023Dataset.py`

#### 设计目标

面向全年统计分析与识别，而非仅服务于 ML 训练。

#### `__getitem__` 返回结构

```python
{
    "inplane":  Tensor (window_size, 1)    # 面内振动加速度
    "outplane": Tensor (window_size, 1)    # 面外振动加速度
    "wind":     Dict | None            # {wind_speed, wind_direction, wind_attack_angle} (ndarray)
    "metadata": Dict                   # 元数据（见下方说明）
}
```

#### metadata 字段

| 字段 | 说明 |
|------|------|
| `cable_pair` | (面内传感器ID, 面外传感器ID) |
| `timestamp` | (month, day, hour) |
| `window_idx` | 当前小时内的窗口编号（0-based） |
| `inplane_sensor_id` | 面内传感器 ID |
| `outplane_sensor_id` | 面外传感器 ID |
| `inplane_file_path` | 面内 VIC 文件路径 |
| `outplane_file_path` | 面外 VIC 文件路径 |
| `missing_rate_in` | 面内数据缺失率 |
| `missing_rate_out` | 面外数据缺失率 |
| `has_wind` | 是否有对应风数据 |
| `dl_label` | DL预测类别ID（0~3，可选） |
| `is_dl_identified` | 是否已应用DL识别结果 |

#### 主要公开接口

| 方法 | 说明 |
|------|------|
| `__len__()` | 返回样本总数 |
| `__getitem__(idx)` | 获取单个样本 |
| `get_train_dataset()` | 获取训练集子集 |
| `get_val_dataset()` | 获取验证集子集 |
| `get_full_dataset()` | 获取全量数据集 |
| `apply_predictions(predictions)` | 应用DL识别结果 |
| `save_predictions(path, model_info)` | 保存识别结果到缓存 |
| `get_metadata_list()` | 获取所有样本的轻量元数据列表 |

---

### 2. _SampleRecord

**文件**: `src/data_processer/datasets/StayCable_Vib2023/StayCableVib2023Dataset.py`

#### 设计说明

样本索引类，内存中保持轻量，原始数据按需加载。

#### 三路元数据对应关系

- `inplane_meta` : cable_pair[0] 传感器 + 当前 (month, day, hour)
- `outplane_meta` : cable_pair[1] 传感器 + 相同 (month, day, hour)
- `wind_meta` : 全局风站 + 相同 (month, day, hour)

三路均通过时间戳精确对齐后才构成一条有效样本；`wind_meta`允许为 None（当 `require_wind_alignment=False` 时）。

---

## 📊 样本索引构建流程

```
Step 1: 振动元数据按 sensor_id 分组，每组再按 (month, day, hour) 建立查找表
Step 2: 风参数元数据按 (month, day, hour) 建立全局查找表
Step 3: 遍历每根拉索对：
          - 取面内/面外传感器共有时间戳
          - 每个时间戳查找对应风数据（可选严格对齐）
          - 每个时间戳按 actual_length // window_size 展开为多个窗口样本
Step 4: 若 time_ordered=True，按 (month, day, hour, window_idx, cable_pair_idx) 排序
```

---

## 🔐 缓存机制

### 指纹校验

缓存文件包含配置指纹，仅当配置完全一致时才复用缓存。

#### 纳入指纹的字段

- `cable_pairs`（排序，避免顺序不同误判）
- `vib_metadata_path` / `wind_metadata_path`
- `window_size` / `require_wind_alignment`
- `time_ordered`（有序/无序缓存不互通）
- 两份元数据的记录数（快速一致性检查，不做全文 hash）

#### 不纳入指纹的字段

- `enable_denoise`：只影响数据内容，不影响样本索引
- `split_ratio` / `split_by_time` / `split_seed`：划分逻辑在索引加载后再计算

### 缓存文件结构

```json
{
  "fingerprint": {...},
  "fingerprint_hash": "...",
  "num_samples": 12345,
  "samples": [...]
}
```

---

## 🚀 使用指南

### 基础使用

```python
from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import StayCableVib2023Config
from src.data_processer.datasets.StayCable_Vib2023 import StayCableVib2023Dataset

# 1. 创建配置
config = StayCableVib2023Config(
    vib_metadata_path="./data/vib_metadata.json",
    wind_metadata_path="./data/wind_metadata.json",
    cable_pairs=[
        ("ST-VIC-C34-101-01", "ST-VIC-C34-101-02"),
        ("ST-VIC-C34-102-01", "ST-VIC-C34-102-02"),
    ],
    window_size=3600,
    time_ordered=True,
    use_cache=True,
)

# 2. 初始化数据集
dataset = StayCableVib2023Dataset(config)

# 3. 获取训练/验证集
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()

# 4. 迭代全量数据集（时序顺序）
full_dataset = dataset.get_full_dataset()
for sample in full_dataset:
    print(sample["metadata"]["timestamp"])
```

### 进阶用法

#### 应用DL识别结果

```python
from src.identifier.deeplearning_methods import DLVibrationIdentifier, FullDatasetRunner

# 1. 创建识别器
identifier = DLVibrationIdentifier.from_checkpoint(
    checkpoint_path="./checkpoints/model.pth",
    model_type="cnn",
    model_config_path="./configs/cnn_config.yaml",
)

# 2. 运行全量识别
runner = FullDatasetRunner(identifier, batch_size=256)
predictions = runner.run(dataset)

# 3. 应用识别结果到数据集
dataset.apply_predictions(predictions)

# 4. 保存识别结果
dataset.save_predictions("./results/predictions.json", model_info="CNN_v1")
```

#### 获取元数据统计

```python
# 获取所有样本的轻量元数据（不触发数据I/O）
metadata_list = dataset.get_metadata_list()

# 统计样本数
print(f"总样本数: {len(metadata_list)}")

# 按拉索分布统计
from collections import defaultdict
cable_count = defaultdict(int)
for meta in metadata_list:
    cable_count[meta["cable_pair"]] += 1
print(cable_count)
```

---

## 📝 配置说明

### StayCableVib2023Config 主要参数

| 参数 | 说明 |
|------|------|
| `vib_metadata_path` | 振动元数据 JSON 文件路径 |
| `wind_metadata_path` | 风参数元数据 JSON 文件路径（可选） |
| `cable_pairs` | 拉索传感器对列表，每对为 (面内传感器ID, 面外传感器ID) |
| `window_size` | 单个时间窗口的样本点数 |
| `enable_denoise` | 是否启用去噪 |
| `require_wind_alignment` | 是否要求严格风数据对齐 |
| `time_ordered` | 是否按时间顺序排列样本 |
| `split_ratio` | 训练集占比 |
| `split_by_time` | 是否按时间划分（True=时序划分，False=随机划分） |
| `split_seed` | 随机划分种子 |
| `use_cache` | 是否启用样本索引缓存 |
| `cache_path` | 缓存文件路径（可选，默认在元数据同目录） |
| `wind_sensor_ids` | 风传感器ID列表（可选，指定优先顺序） |
| `predictions_cache_path` | DL识别结果缓存路径（可选） |

---

## 🔗 相关链接

- [深度学习识别模块文档](../../../../identifier/deeplearning_methods/README.md)
- [数据集模块总览](../README.md)
- [数据预处理文档](../../preprocess/README.md)

---

## 📞 维护信息

- **最后更新**: 2026年4月2日
- **模块维护者**: Data Processing Team
- **依赖库**: torch, numpy, json, hashlib
