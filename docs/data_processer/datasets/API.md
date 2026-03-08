# 数据集模块API参考

## 模块导入

```python
# 配置类
from src.config.data_processer.datasets.data_factory import BaseDatasetConfig
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)

# 数据集类
from src.data_processer.datasets.data_factory import get_dataset
from src.data_processer.datasets.VIV2NumClassification.BaseDataset import BaseDataset
from src.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDataset
)
from src.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset_2_num_classes import (
    VIVTimeSeriesClassificationDataset2NumClasses
)
```

---

## BaseDatasetConfig 配置基类

### 类定义

```python
class BaseDatasetConfig(BaseConfig, ABC):
    """所有数据集的通用配置基类"""
```

### 初始化参数

#### 必填参数
```python
BaseDatasetConfig(
    data_dir: str,  # 数据集根目录路径
    ...
)
```

#### 数据集类型参数
```python
dataset_type: Literal["custom", "segmentation", "classification", "regression"]
    # 默认: "custom"
    # 说明: 数据集类型定义

annotation_path: Optional[str]
    # 默认: None
    # 说明: 标注文件路径

has_annotation: bool
    # 默认: True
    # 说明: 数据集是否包含标注
```

#### 数据划分参数
```python
use_official_split: bool
    # 默认: False
    # 说明: 是否使用官方train/val/test划分

split_ratio: float (0.0-1.0)
    # 默认: 0.8
    # 说明: 训练集占比

test_ratio: Optional[float] (0.0-1.0)
    # 默认: None
    # 说明: 测试集占比，未指定则剩余为验证集

split_seed: int
    # 默认: 42
    # 说明: 数据划分随机种子

auto_split: bool
    # 默认: False
    # 说明: 是否启用自动划分
```

#### 数据加载参数
```python
batch_size: int (≥1)
    # 默认: 8
    # 说明: 批次大小

shuffle: bool
    # 默认: True
    # 说明: 是否打乱顺序

num_workers: int (≥0)
    # 默认: 4
    # 说明: 数据加载进程数，Windows建议0

pin_memory: bool
    # 默认: True
    # 说明: 是否锁页内存（GPU训练时）

drop_last: bool
    # 默认: False
    # 说明: 是否丢弃最后不完整批次

prefetch_factor: Optional[int] (≥1)
    # 默认: 2
    # 说明: 每个worker预取的批次数

max_samples: Optional[int] (≥1)
    # 默认: None
    # 说明: 最大加载样本数（None=全部）
```

#### 预处理参数
```python
normalize: bool
    # 默认: True
    # 说明: 是否归一化

mean: Union[List[float], Tuple[float, ...]]
    # 默认: [0.485, 0.456, 0.406]  # ImageNet均值
    # 说明: 归一化均值

std: Union[List[float], Tuple[float, ...]]
    # 默认: [0.229, 0.224, 0.225]   # ImageNet标准差
    # 说明: 归一化标准差

resize_size: Optional[Tuple[int, int]]
    # 默认: None
    # 说明: 调整图像尺寸 (H, W)

keep_aspect_ratio: bool
    # 默认: False
    # 说明: 是否保持长宽比
```

#### 数据增强参数
```python
train_aug: bool
    # 默认: True
    # 说明: 是否启用训练集数据增强

hflip_prob: float (0.0-1.0)
    # 默认: 0.5
    # 说明: 随机水平翻转概率

vflip_prob: float (0.0-1.0)
    # 默认: 0.0
    # 说明: 随机垂直翻转概率

rotate_angle: int (0-180)
    # 默认: 0
    # 说明: 随机旋转角度范围（±angle度）
```

#### 缓存参数
```python
cache_in_memory: bool
    # 默认: False
    # 说明: 是否将数据集缓存到内存

cache_dir: Optional[str]
    # 默认: None
    # 说明: 磁盘缓存路径
```

#### 分布式参数
```python
use_dist_sampler: bool
    # 默认: False
    # 说明: 是否使用分布式采样器
```

### 主要方法

#### `merge_dict(user_dict: Dict) -> None`

递归合并用户字典配置。

**参数**:
- `user_dict`: 用户配置字典

**示例**:
```python
config = VIVTimeSeriesClassificationDatasetConfig(data_dir="./data")
config.merge_dict({
    "batch_size": 16,
    "shuffle": False,
    "normalize": True
})
```

#### `load_yaml(yaml_path: str) -> None`

从YAML文件加载配置并合并。

**参数**:
- `yaml_path`: YAML文件路径

**异常**:
- `FileNotFoundError`: 文件不存在
- `ValueError`: 文件格式无效

**示例**:
```python
config = VIVTimeSeriesClassificationDatasetConfig(data_dir="./data")
config.load_yaml("config.yaml")
```

---

## VIVTimeSeriesClassificationDatasetConfig 时序分类配置

### 类定义

```python
class VIVTimeSeriesClassificationDatasetConfig(BaseDatasetConfig):
    """VIV时序分类数据集配置"""
```

### 新增参数

#### LSTM输入格式参数
```python
batch_first: bool
    # 默认: True
    # 说明: LSTM输入是否batch_first格式

output_mode: Literal["time_series", "grid_2d"]
    # 默认: "time_series"
    # 说明: 输出模式（时序或网格）

fix_seq_len: Optional[int] (≥1)
    # 默认: None
    # 说明: 固定序列长度（None=原始长度）

pad_mode: Literal["zero", "repeat", "mean"]
    # 默认: "zero"
    # 说明: 短序列补全模式

trunc_mode: Literal["head", "tail"]
    # 默认: "tail"
    # 说明: 长序列截断模式
```

#### 时序归一化参数
```python
normalize: bool
    # 默认: False
    # 说明: 是否对时序数据归一化

normalize_type: Literal["min-max", "z-score"]
    # 默认: "min-max"
    # 说明: 时序数据归一化方式

ts_mean: Optional[List[float]]
    # 默认: None
    # 说明: z-score归一化的均值

ts_std: Optional[List[float]]
    # 默认: None
    # 说明: z-score归一化的标准差
```

#### 特征维度参数
```python
feat_dim: Optional[int] (≥1)
    # 默认: None
    # 说明: 时序数据特征维度（None=自动）
```

### 覆盖参数

```python
shuffle: bool
    # 默认: False  # 时序数据建议False
    
batch_size: int
    # 默认: 4     # 时序数据通常小批量
```

### 校验方法

#### `validate_normalize_config() -> None`

校验归一化配置关联性。

**检查**:
- z-score模式时mean和std需同时指定或同时为None

#### `validate_seq_config() -> None`

校验序列长度配置关联性。

**检查**:
- 未指定fix_seq_len时，pad_mode/trunc_mode配置失效

#### `validate_feat_dim(v: int) -> int`

校验特征维度合法性。

**检查**:
- 特征维度必须≥1

---

## BaseDataset 数据集基类

### 类定义

```python
class BaseDataset(Dataset, ABC, Generic[DatasetType]):
    """通用数据集基类"""
```

### 初始化

```python
def __init__(self, config: BaseDatasetConfig):
    """
    初始化数据集
    
    参数:
        config: BaseDatasetConfig配置实例
    """
```

### 主要属性

```python
self.config: BaseDatasetConfig
    # 配置实例

self.data_dir: Path
    # 数据集根目录

self.full_file_paths: List[Path]
    # 所有文件路径列表

self.train_paths: List[Path]
    # 训练集文件路径

self.val_paths: List[Path]
    # 验证集文件路径

self.test_paths: List[Path]
    # 测试集文件路径

self._dataset_mode: str
    # 当前模式: "full"/"train"/"val"/"test"
```

### 主要方法

#### `get_train_dataset() -> "BaseDataset"`

获取训练集实例。

**返回**: 独立的训练集数据集实例

**示例**:
```python
dataset = BaseDataset(config)
train_dataset = dataset.get_train_dataset()
```

#### `get_val_dataset() -> "BaseDataset"`

获取验证集实例。

**返回**: 独立的验证集数据集实例

#### `get_test_dataset() -> "BaseDataset"`

获取测试集实例。

**返回**: 独立的测试集数据集实例

#### `get_dataloader(batch_size: int = None, shuffle: bool = None, **kwargs) -> DataLoader`

创建DataLoader。

**参数**:
- `batch_size`: 批次大小（None使用配置值）
- `shuffle`: 是否打乱（None使用配置值）
- `**kwargs`: 其他DataLoader参数

**返回**: PyTorch DataLoader

**示例**:
```python
train_loader = train_dataset.get_dataloader(batch_size=32, shuffle=True)
```

#### `__len__() -> int`

获取数据集样本数。

**返回**: 样本总数

#### `__getitem__(idx: int) -> tuple`

获取单个样本（抽象方法，子类必须实现）。

**参数**:
- `idx`: 样本索引

**返回**: (data, label) 或其他格式

---

## VIVTimeSeriesClassificationDataset 多分类数据集

### 类定义

```python
class VIVTimeSeriesClassificationDataset(BaseDataset):
    """VIV时序分类数据集"""
```

### 初始化

```python
def __init__(self, config: VIVTimeSeriesClassificationDatasetConfig):
    """
    初始化VIV时序分类数据集
    
    参数:
        config: VIVTimeSeriesClassificationDatasetConfig配置实例
    """
```

### 特化属性

```python
self.batch_first: bool
    # LSTM输入是否batch_first

self.fix_seq_len: Optional[int]
    # 固定序列长度

self.normalize: bool
    # 是否归一化

self.output_mode: str
    # 输出模式

self.global_norm_stats: dict
    # 全局归一化统计量
```

### 关键方法

#### `get_train_dataset() -> "VIVTimeSeriesClassificationDataset"`

获取训练集。

**返回**: 训练集实例

#### `get_val_dataset() -> "VIVTimeSeriesClassificationDataset"`

获取验证集。

**返回**: 验证集实例

#### `__getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]`

获取单样本。

**参数**:
- `idx`: 样本索引

**返回**: (data, label)
- `data`: shape (seq_len, feat_dim) 或 (1, 50, 60)
- `label`: int标签

**示例**:
```python
dataset = VIVTimeSeriesClassificationDataset(config)
data, label = dataset[0]
print(data.shape)   # (1000, feat_dim)
print(label)        # 0-9
```

---

## VIVTimeSeriesClassificationDataset2NumClasses 二分类数据集

### 类定义

```python
class VIVTimeSeriesClassificationDataset2NumClasses(BaseDataset):
    """VIV二分类数据集（自动过滤标签1，标签2转为1）"""
```

### 初始化

```python
def __init__(self, config: VIVTimeSeriesClassificationDatasetConfig):
    """
    初始化VIV二分类数据集
    
    特性:
        - 自动剔除所有标签为1的样本
        - 标签2自动转为1
        - 结果只有类别0和1
    
    参数:
        config: VIVTimeSeriesClassificationDatasetConfig配置实例
    """
```

### 特化属性

```python
self.binary_label_map: dict
    # 标签映射：{原标签 -> 新标签}
    # 例: {0: 0, 2: 1}
```

---

## 工厂函数

### `get_dataset(config: BaseConfig) -> object`

根据配置实例创建对应的数据集。

**参数**:
- `config`: 配置实例

**返回**: 对应的数据集实例

**异常**:
- `ValueError`: 配置类型未注册或配置类型未绑定数据集类

**示例**:
```python
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)
from src.data_processer.datasets.data_factory import get_dataset

config = VIVTimeSeriesClassificationDatasetConfig(data_dir="./data")
dataset = get_dataset(config)
# 自动返回 VIVTimeSeriesClassificationDataset 实例
```

---

## 数据格式规范

### 输入格式 (.mat文件)

```python
{
    'data': np.ndarray,     # shape: (seq_len, feat_dim), dtype: float32/64
    'label': int or array   # 0-9 或其他标签值
}
```

### 输出格式 (__getitem__)

#### 时序模式 (output_mode='time_series')

```python
(
    torch.Tensor,           # shape: (seq_len, feat_dim), dtype: float32
    torch.Tensor            # shape: (), dtype: int64（标量）
)

# 示例
# data shape: (1000, 10)
# label: tensor(5)
```

#### 网格模式 (output_mode='grid_2d')

```python
(
    torch.Tensor,           # shape: (1, 50, 60), dtype: float32
    torch.Tensor            # shape: (), dtype: int64（标量）
)

# 示例
# data shape: (1, 50, 60)
# label: tensor(2)
```

### DataLoader输出

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32)
for data, labels in loader:
    # 时序模式
    print(data.shape)     # (32, seq_len, feat_dim)
    print(labels.shape)   # (32,)
    
    # 或网格模式
    print(data.shape)     # (32, 1, 50, 60)
    print(labels.shape)   # (32,)
```

---

## 常见错误

### 错误: `NotImplementedError: _parse_sample not implemented`

**原因**: 子类未实现抽象方法

**解决**: 实现 `_parse_sample()` 方法

### 错误: `KeyError: 配置键「xxx」不存在`

**原因**: 尝试merge不存在的配置键

**解决**: 检查配置键名是否正确

### 错误: `ValueError: split_ratio(0.8) + test_ratio(0.3) 不能超过1.0`

**原因**: 划分比例超过1.0

**解决**: 调整比例使其总和≤1.0

---

## 性能参数调优

| 参数 | 推荐值 | 场景 |
|------|--------|------|
| `num_workers` | 0 | Windows + 小数据集 |
| `num_workers` | 4-8 | Linux + 大数据集 |
| `batch_size` | 4-16 | LSTM（显存受限） |
| `batch_size` | 32-128 | CNN（显存允许） |
| `cache_in_memory` | True | 数据集<10GB |
| `pin_memory` | True | GPU训练 |
| `pin_memory` | False | CPU训练 |

---

## 参考链接

- [完整文档](README.md)
- [快速入门](QUICKSTART.md)
