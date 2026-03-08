# 数据集模块快速入门

## ⚡ 5分钟快速开始

### 最小化示例

```python
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)
from src.data_processer.datasets.data_factory import get_dataset
from torch.utils.data import DataLoader

# 1. 创建配置
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data/viv_timeseries",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    fix_seq_len=1000
)

# 2. 创建数据集
dataset = get_dataset(config)
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()

# 3. 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. 迭代数据
for data, labels in train_loader:
    print(f"Data shape: {data.shape}")      # (32, 1000, feat_dim)
    print(f"Labels shape: {labels.shape}")  # (32,)
    break
```

---

## 📚 常见场景

### 场景1: LSTM时序分类训练

```python
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)
from src.data_processer.datasets.data_factory import get_dataset
from torch.utils.data import DataLoader
import torch.nn as nn

# 配置：适配LSTM
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data/viv",
    batch_size=16,
    shuffle=True,
    num_workers=4,
    fix_seq_len=2000,           # LSTM固定输入长度
    normalize=True,              # LSTM通常需要归一化
    normalize_type="z-score",
    batch_first=True,            # PyTorch LSTM默认batch_first=True
    output_mode="time_series",
    split_ratio=0.8,
    test_ratio=0.1
)

# 创建数据集和加载器
dataset = get_dataset(config)
train_loader = DataLoader(dataset.get_train_dataset(), batch_size=16, shuffle=True)
val_loader = DataLoader(dataset.get_val_dataset(), batch_size=16, shuffle=False)

# 训练循环
model = nn.LSTM(input_size=config.feat_dim, hidden_size=128, batch_first=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for data, labels in train_loader:
        # data shape: (batch_size, seq_len, feat_dim)
        outputs, _ = model(data)
        loss = criterion(outputs[:, -1, :], labels)  # 取最后一步输出
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 场景2: CNN网格分类

```python
# 配置：输出网格格式供CNN处理
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data/viv",
    batch_size=32,
    output_mode="grid_2d",      # 输出50×60网格
    fix_seq_len=3000,
    shuffle=True,
    train_aug=True,             # 启用数据增强
    hflip_prob=0.5
)

dataset = get_dataset(config)
train_loader = DataLoader(dataset.get_train_dataset(), batch_size=32, shuffle=True)

# 此时data shape: (batch_size, 1, 50, 60) 可以直接输入CNN
```

### 场景3: 二分类异常检测

```python
from src.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset_2_num_classes import (
    VIVTimeSeriesClassificationDataset2NumClasses
)

# 使用多分类配置，但创建二分类数据集
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data/viv",
    batch_size=32,
    fix_seq_len=1000
)

# 直接创建二分类数据集（会自动过滤标签1，标签2转为1）
dataset = VIVTimeSeriesClassificationDataset2NumClasses(config)
train_dataset = dataset.get_train_dataset()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 现在只有类别0（正常）和类别1（异常）
for data, labels in train_loader:
    assert labels.max() <= 1  # 最大标签为1
```

### 场景4: 快速调试小数据集

```python
# 配置：快速调试
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data/viv",
    max_samples=100,           # 仅加载100个样本
    cache_in_memory=True,      # 缓存到内存
    num_workers=0,             # 不使用多进程
    batch_size=8,
    shuffle=False,
    fix_seq_len=500
)

dataset = get_dataset(config)
train_dataset = dataset.get_train_dataset()

# 快速迭代
for data, labels in DataLoader(train_dataset, batch_size=8):
    print(data.shape, labels.shape)
    # 这会很快，因为数据已缓存到内存
```

### 场景5: 从YAML配置加载

```yaml
# config.yaml
data_dir: "./data/viv_timeseries"
batch_size: 32
num_workers: 4
shuffle: true
fix_seq_len: 1000
normalize: true
normalize_type: "z-score"
batch_first: true
output_mode: "time_series"
use_official_split: false
split_ratio: 0.8
test_ratio: 0.1
```

```python
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)
from src.data_processer.datasets.data_factory import get_dataset

# 从YAML加载配置
config = VIVTimeSeriesClassificationDatasetConfig(data_dir="./data")
config.load_yaml("config.yaml")

dataset = get_dataset(config)
train_loader = DataLoader(dataset.get_train_dataset(), batch_size=32, shuffle=True)
```

---

## 🎯 常用参数速查表

| 参数 | 推荐值 | 场景 |
|------|--------|------|
| `batch_size` | 16-32 | LSTM通常较小 |
| `num_workers` | 0 (Windows), 4 (Linux) | 并行加载 |
| `fix_seq_len` | 1000-3000 | 时序数据长度 |
| `normalize` | True | LSTM推荐 |
| `normalize_type` | "z-score" | 时序推荐 |
| `shuffle` | False | LSTM时序任务 |
| `shuffle` | True | 分类任务 |
| `output_mode` | "time_series" | LSTM输入 |
| `output_mode` | "grid_2d" | CNN输入 |

---

## 🐛 常见错误排查

### 错误1: `FileNotFoundError: 数据集根目录不存在`

**原因**: `data_dir` 路径不存在

**解决**:
```python
# 检查路径
import os
assert os.path.exists("./data/viv"), "数据集路径不存在"

# 使用绝对路径
config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir=os.path.abspath("./data/viv")
)
```

### 错误2: `RuntimeError: Expected 3D input (batch, seq, feat), got 2D`

**原因**: 数据维度不匹配，通常是 `batch_first` 设置不对

**解决**:
```python
# 确保batch_first与模型保持一致
config = VIVTimeSeriesClassificationDatasetConfig(
    batch_first=True  # PyTorch默认LSTM为True
)

# LSTM需要 (batch, seq_len, feat_dim)
```

### 错误3: `ValueError: split_ratio(0.8) + test_ratio(0.3) 不能超过1.0`

**原因**: 划分比例加起来超过1.0

**解决**:
```python
# 确保总比例 ≤ 1.0
config = VIVTimeSeriesClassificationDatasetConfig(
    split_ratio=0.7,   # 训练集70%
    test_ratio=0.15,   # 测试集15%
    # 验证集自动为 15%
)
```

### 错误4: `MemoryError: Unable to allocate 10.0 GB`

**原因**: 数据集太大，内存缓存溢出

**解决**:
```python
# 关闭内存缓存或限制样本数
config = VIVTimeSeriesClassificationDatasetConfig(
    cache_in_memory=False,  # 关闭缓存
    max_samples=5000        # 或限制样本数
)
```

---

## 📖 进一步阅读

- 详细文档: [数据集模块文档](README.md)
- API参考: 查看源代码中的docstring
- 示例脚本: 见 `examples/` 目录

---

## 💡 性能优化建议

| 优化策略 | 何时使用 | 效果 |
|---------|---------|------|
| 内存缓存 | 小数据集(<10GB) | ⭐⭐⭐⭐⭐ |
| 磁盘缓存 | 中等数据集 | ⭐⭐⭐ |
| 多进程加载 | Linux/Mac + CPU充足 | ⭐⭐⭐⭐ |
| 固定序列长度 | 处理不等长数据 | ⭐⭐⭐ |
| batch_size调优 | GPU显存优化 | ⭐⭐⭐ |

---

## 📞 获取帮助

1. 查看源代码注释和docstring
2. 阅读完整文档 [README.md](README.md)
3. 检查示例代码和测试用例
4. 提交Issue或咨询开发者
