# 数据集处理模块文档

## 📋 模块概述

`src/data_processer/datasets/` 是数据处理模块中的核心数据集管理子模块。它提供了一套统一的、基于配置驱动的数据集加载、处理和划分框架，支持多种数据集类型（特别是VIV时序分类数据集）的标准化管理。

### 核心特性

- **配置驱动架构**: 所有数据集参数通过配置类统一管理，无需硬编码
- **自动数据划分**: 支持官方划分和自定义划分，灵活的训练/验证/测试集拆分
- **时序数据专优化**: 针对LSTM等时序网络的特殊处理（序列长度、补全/截断、归一化等）
- **灵活的预处理管道**: 支持数据增强、缓存、多进程加载等功能
- **强类型校验**: 基于Pydantic的全面参数校验，避免运行时错误
- **内存/磁盘缓存**: 支持加速数据重复加载和预处理

---

## 📂 目录结构

```
src/data_processer/datasets/
├── __init__.py
├── data_factory.py                              # 数据集工厂类，统一创建数据集实例
└── VIV2NumClassification/                       # VIV时序分类数据集实现
    ├── BaseDataset.py                           # 通用数据集基类（抽象类）
    ├── VIVTimeseriesClassificationDataset.py    # VIV多分类时序数据集
    └── VIVTimeseriesClassificationDataset_2_num_classes.py  # VIV二分类时序数据集

src/config/data_processer/datasets/
├── data_factory.py                              # 所有数据集的通用配置基类 (BaseDatasetConfig)
└── VIV2NumClassification/
    ├── VIVTimeseriesClassificationDataset.py    # VIV多分类配置
    └── VIVTimeseriesClassificationDataset_2_num_classes.py  # VIV二分类配置
```

---

## 🏗 架构设计

### 分层架构

```
[用户代码]
    ↓
[数据集工厂 (data_factory)]
    ↓
[具体数据集类 (VIVTimeSeriesClassificationDataset)]
    ↓
[基类 (BaseDataset)]
    ↓
[PyTorch Dataset 接口]
```

### 核心类关系

```
BaseDatasetConfig (配置基类)
    ↓
    ├─→ VIVTimeSeriesClassificationDatasetConfig (VIV多分类配置)
    └─→ VIVTimeSeriesClassificationDatasetConfig (VIV二分类配置)

BaseDataset (数据集基类，抽象)
    ↓
    ├─→ VIVTimeSeriesClassificationDataset (VIV多分类实现)
    └─→ VIVTimeSeriesClassificationDataset_2NumClasses (VIV二分类实现)
```

---

## 📖 核心模块详解

### 1. 通用配置基类 - `BaseDatasetConfig`

**文件**: `src/config/data_processer/datasets/data_factory.py`

#### 职责
- 定义所有数据集的通用参数（路径、划分、加载、预处理、增强、缓存等）
- 提供强类型校验（基于Pydantic）
- 定义抽象方法，强制子类实现任务相关逻辑

#### 主要参数分类

| 分类 | 参数 | 说明 |
|------|------|------|
| **基础路径** | `data_dir` | 数据集根目录（必填） |
| | `dataset_type` | 数据集类型：custom/segmentation/classification/regression |
| | `annotation_path` | 标注文件路径（可选） |
| | `has_annotation` | 是否包含标注 |
| **数据划分** | `use_official_split` | 是否使用官方train/val/test划分 |
| | `split_ratio` | 训练集占比（默认0.8） |
| | `test_ratio` | 测试集占比（可选） |
| | `split_seed` | 划分随机种子（保证可复现） |
| **数据加载** | `batch_size` | 批次大小 |
| | `shuffle` | 是否打乱顺序 |
| | `num_workers` | 数据加载进程数 |
| | `pin_memory` | 是否锁页内存（GPU训练时） |
| | `drop_last` | 是否丢弃最后不完整批次 |
| | `max_samples` | 最大加载样本数（调试用） |
| **预处理** | `normalize` | 是否归一化 |
| | `mean`, `std` | 归一化参数 |
| | `resize_size` | 调整图像尺寸 |
| | `keep_aspect_ratio` | 是否保持长宽比 |
| **数据增强** | `train_aug` | 是否启用训练集增强 |
| | `hflip_prob`, `vflip_prob` | 翻转概率 |
| | `rotate_angle` | 旋转角度范围 |
| **缓存** | `cache_in_memory` | 是否缓存到内存 |
| | `cache_dir` | 磁盘缓存路径 |
| **分布式** | `use_dist_sampler` | 是否使用分布式采样器 |

#### 关键方法

```python
# 合并用户字典配置
config.merge_dict({"batch_size": 16, "normalize": True})

# 从YAML文件加载配置
config.load_yaml("config.yaml")
```

---

### 2. 通用数据集基类 - `BaseDataset`

**文件**: `src/data_processer/datasets/VIV2NumClassification/BaseDataset.py`

#### 职责
- 实现PyTorch Dataset接口
- 提供文件路径管理、自动划分、缓存管理等通用功能
- 定义抽象方法，强制子类实现数据解析和预处理

#### 核心功能

| 功能 | 说明 |
|------|------|
| **文件路径管理** | 递归扫描数据目录，支持最大样本限制 |
| **自动数据划分** | 支持官方划分/自定义划分，训练/验证/测试集独立获取 |
| **独立实例获取** | `get_train_dataset()`, `get_val_dataset()`, `get_test_dataset()` 返回独立实例 |
| **缓存管理** | 内存缓存和磁盘缓存支持，加速重复加载 |
| **数据集模式** | 支持full/train/val/test模式，按需差异化配置 |

#### 关键方法

```python
# 初始化数据集
dataset = BaseDataset(config)

# 获取训练/验证/测试集
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()
test_dataset = dataset.get_test_dataset()

# 获取数据加载器
train_loader = train_dataset.get_dataloader()

# 统计信息
num_samples = len(train_dataset)
```

#### 抽象方法（子类必须实现）

```python
@abstractmethod
def _parse_sample(self, file_path: Path) -> dict:
    """
    解析单个样本文件
    :param file_path: 样本文件路径
    :return: {
        'data': ...,        # 样本数据
        'label': ...,       # 标签
        'sample_id': ...    # 样本ID
    }
    """
    pass

@abstractmethod
def __getitem__(self, idx: int) -> tuple:
    """
    获取单个样本，返回格式需适配具体任务
    :param idx: 样本索引
    :return: (data, label) 或其他格式
    """
    pass
```

---

### 3. VIV时序分类配置 - `VIVTimeSeriesClassificationDatasetConfig`

**文件**: `src/config/data_processer/datasets/VIV2NumClassification/VIVTimeseriesClassificationDataset.py`

#### 特化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_first` | bool | True | LSTM输入是否batch_first |
| `output_mode` | str | "time_series" | 输出模式：time_series（时序）/grid_2d（网格） |
| `fix_seq_len` | int | None | 固定序列长度（None=原始长度） |
| `pad_mode` | str | "zero" | 短序列补全模式：zero/repeat/mean |
| `trunc_mode` | str | "tail" | 长序列截断模式：head/tail |
| `normalize` | bool | False | 是否归一化时序数据 |
| `normalize_type` | str | "min-max" | 归一化方式：min-max/z-score |
| `ts_mean`, `ts_std` | list | None | z-score的均值和标准差 |
| `feat_dim` | int | None | 时序数据特征维度 |
| `shuffle` | bool | False | 是否打乱顺序（时序建议False） |

#### 校验逻辑

- **归一化校验**: z-score时mean和std需同时指定或同时为None
- **序列长度校验**: fix_seq_len为None时，pad_mode/trunc_mode配置失效
- **特征维度校验**: 与实际样本特征维度一致

---

### 4. VIV时序分类数据集 - `VIVTimeSeriesClassificationDataset`

**文件**: `src/data_processer/datasets/VIV2NumClassification/VIVTimeseriesClassificationDataset.py`

#### 特性

- **双模式输出**: 支持时序模式（LSTM）和网格模式（CNN）
- **全局归一化**: 基于训练集统计量，避免单样本归一化混乱
- **灵活的序列处理**: 支持变长序列、补全、截断等
- **多标签格式**: 支持标量标签和多标签输出

#### 关键方法

```python
# 初始化
dataset = VIVTimeSeriesClassificationDataset(config)

# 获取训练/验证/测试集
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()

# 创建DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 访问单样本
data, label = train_dataset[0]
# data: (seq_len, feat_dim) 或 (50, 60) 取决于output_mode
# label: 0-9 或其他类别标签
```

#### 数据格式

**输入** (`.mat` 文件)
```python
# 时间序列格式
{
    'data': ndarray with shape (seq_len, feat_dim),  # 时序数据
    'label': int or array with label(s)
}

# 网格格式 (仅当output_mode='grid_2d'时)
```

**输出** (`__getitem__`)
```python
# 时序模式 (output_mode='time_series')
(
    tensor with shape (batch, seq_len, feat_dim),  # batch_first=True时
    tensor with shape (batch,)  # 标签
)

# 网格模式 (output_mode='grid_2d')
(
    tensor with shape (batch, 1, 50, 60),  # 网格化数据
    tensor with shape (batch,)
)
```

---

### 5. VIV二分类数据集 - `VIVTimeSeriesClassificationDataset_2NumClasses`

**文件**: `src/data_processer/datasets/VIV2NumClassification/VIVTimeseriesClassificationDataset_2_num_classes.py`

#### 特化改动

相比多分类版本，二分类版本在初始化时进行**标签转换**：

| 原始标签 | 转换后标签 | 说明 |
|---------|----------|------|
| 0 | 0 | 一般情况（正常） |
| 1 | —— | 过渡样本（剔除） |
| 2 | 1 | 异常情况 |

这样形成二分类任务：
- 类别0：正常振动
- 类别1：异常振动

#### 初始化流程

```python
def __init__(self, config: VIVTimeSeriesClassificationDatasetConfig):
    super().__init__(config)
    # 1. 基类初始化（路径/划分/缓存）
    # 2. 标签过滤：删除所有标签为1的样本
    # 3. 标签映射：标签2转为1
    # 4. 计算全局归一化统计
```

---

### 6. 数据集工厂 - `get_dataset()`

**文件**: `src/data_processer/datasets/data_factory.py`

#### 职责

- 根据配置实例的类型，自动查找对应的数据集类
- 统一的数据集创建接口

#### 工作流程

```python
config_instance = VIVTimeSeriesClassificationDatasetConfig(data_dir="./data")
dataset = get_dataset(config_instance)
# 自动返回 VIVTimeSeriesClassificationDataset 实例
```

#### 核心算法

1. 获取配置实例的类 (e.g., `VIVTimeSeriesClassificationDatasetConfig`)
2. 通过 `CONFIG_CLASS_REGISTRY` 反向查找对应的 `config_type`
3. 通过 `DATASET_CLASS_REGISTRY` 查找该 `config_type` 对应的数据集类
4. 用配置实例初始化并返回数据集

---

## 🚀 使用指南

### 基础使用

#### 1. 定义配置

```python
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)

config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data/viv_timeseries",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    fix_seq_len=1000,
    normalize=True,
    normalize_type="z-score",
    use_official_split=False,
    split_ratio=0.8,
    test_ratio=0.1
)
```

#### 2. 使用工厂创建数据集

```python
from src.data_processer.datasets.data_factory import get_dataset

dataset = get_dataset(config)
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()
test_dataset = dataset.get_test_dataset()
```

#### 3. 创建DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 迭代数据
for data, labels in train_loader:
    print(f"Batch data shape: {data.shape}")
    print(f"Batch labels shape: {labels.shape}")
    break
```

### 进阶用法

#### 配置YAML加载

```python
# config.yaml
data_dir: "./data/viv_timeseries"
batch_size: 16
num_workers: 4
fix_seq_len: 1000
normalize: true
normalize_type: "z-score"
use_official_split: false
split_ratio: 0.8
```

```python
config = VIVTimeSeriesClassificationDatasetConfig(data_dir="./data")
config.load_yaml("config.yaml")

dataset = get_dataset(config)
```

#### 内存缓存加速

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data",
    cache_in_memory=True,  # 启用内存缓存
    max_samples=5000  # 限制样本数，避免OOM
)
dataset = get_dataset(config)
```

#### 自定义时序处理

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data",
    fix_seq_len=2000,           # 固定序列长度为2000
    pad_mode="repeat",           # 短序列重复补全
    trunc_mode="head",           # 长序列截断头部
    normalize=True,
    normalize_type="min-max"
)
dataset = get_dataset(config)
```

#### 二分类数据集

```python
from src.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset_2_num_classes import (
    VIVTimeSeriesClassificationDataset2NumClasses
)

config = VIVTimeSeriesClassificationDatasetConfig(...)
dataset = VIVTimeSeriesClassificationDataset2NumClasses(config)
# 标签1样本自动剔除，标签2自动转为1
```

---

## 📊 配置示例

### 例1：基础LSTM训练配置

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./datasets/viv_lstm",
    dataset_type="classification",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    fix_seq_len=1000,
    normalize=True,
    normalize_type="z-score",
    output_mode="time_series",
    batch_first=True,
    use_official_split=False,
    split_ratio=0.7,
    test_ratio=0.15,
    split_seed=42
)
```

### 例2：CNN网格处理配置

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./datasets/viv_cnn",
    batch_size=16,
    output_mode="grid_2d",  # 输出50×60网格
    fix_seq_len=3000,
    normalize=False,  # CNN通常不需要时序归一化
    train_aug=True,  # 启用数据增强
    hflip_prob=0.5,
    use_official_split=True  # 使用官方划分
)
```

### 例3：调试配置

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./datasets/viv",
    max_samples=100,  # 仅加载100个样本调试
    cache_in_memory=True,
    batch_size=8,
    num_workers=0,  # Windows建议设为0
    shuffle=False
)
```

---

## 🔧 常见问题

### Q1: 如何快速加载小数据集？

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data",
    cache_in_memory=True,  # 启用内存缓存
    num_workers=0  # 关闭多进程加载
)
```

### Q2: 如何处理不同长度的时序数据？

```python
# 方案A: 固定长度 + 补全/截断
config = VIVTimeSeriesClassificationDatasetConfig(
    fix_seq_len=1000,
    pad_mode="zero",  # 或 "repeat", "mean"
    trunc_mode="tail"  # 或 "head"
)

# 方案B: 使用变长序列（需要自定义batch处理）
config = VIVTimeSeriesClassificationDatasetConfig(
    fix_seq_len=None  # 保留原始长度
)
```

### Q3: 如何确保训练/验证划分可复现？

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    split_seed=42,  # 固定随机种子
    use_official_split=False,
    split_ratio=0.8
)
# 每次创建都会使用相同的随机种子，保证可复现
```

### Q4: Windows环境下数据加载缓慢？

```python
# Windows建议关闭多进程加载
config = VIVTimeSeriesClassificationDatasetConfig(
    num_workers=0,  # 使用主线程加载
    pin_memory=False  # Windows GPU支持有限
)
```

### Q5: 如何进行在线数据增强？

```python
config = VIVTimeSeriesClassificationDatasetConfig(
    train_aug=True,
    hflip_prob=0.5,
    vflip_prob=0.3,
    rotate_angle=15
)
# 注: 目前支持图像增强，时序数据增强需自定义
```

---

## 📝 数据集格式规范

### 目录结构

```
data_dir/
├── [可选] train/
│   ├── sample_1.mat
│   ├── sample_2.mat
│   └── ...
├── [可选] val/
├── [可选] test/
└── sample_1.mat          # 当use_official_split=False时，所有样本混在根目录
    sample_2.mat
    ...
```

### `.mat` 文件格式

```python
# 保存格式
import scipy.io as sio
data_dict = {
    'data': timeseries_array,  # shape: (seq_len, feat_dim), dtype: float32/64
    'label': class_label        # int or ndarray
}
sio.savemat('sample.mat', data_dict)

# 加载格式
loaded = sio.loadmat('sample.mat')
timeseries = loaded['data']  # 自动返回ndarray
label = loaded['label']
```

---

## 🔗 相关链接

- [SVM模块文档](../../machine_learning_module/svm/README.md)
- [数据预处理文档](../README.md)
- [统计分析文档](../statistics/vib_data_io_process/README.md)

---

## 📞 维护信息

- **最后更新**: 2026年3月8日
- **模块维护者**: Data Processing Team
- **依赖库**: torch, scipy, numpy, pydantic, pandas
