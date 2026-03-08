# 数据集模块架构设计文档

## 📐 系统设计原则

### 1. 配置驱动架构（Configuration-Driven Architecture）

所有参数通过配置类统一管理，而不是硬编码或函数参数。

**优点**：
- 便于复现实验
- 支持配置文件持久化
- 参数校验集中管理
- 便于版本控制

```
配置文件 (YAML/JSON)
    ↓
配置类 (Pydantic)
    ↓
数据集类
```

### 2. 工厂模式（Factory Pattern）

通过注册表机制自动匹配配置类和数据集类。

**优点**：
- 解耦创建逻辑
- 易于扩展新数据集
- 统一的创建接口

```python
get_dataset(config) 
    → 查找config_type 
    → 查找对应数据集类 
    → 创建并返回实例
```

### 3. 分层设计（Layered Architecture）

```
用户代码层
    ↓
工厂层 (get_dataset)
    ↓
数据集类层 (具体实现)
    ↓
基类层 (BaseDataset)
    ↓
PyTorch Dataset 接口
```

**分层好处**：
- 各层职责清晰
- 代码重用性强
- 便于单元测试

### 4. 配置继承链（Configuration Inheritance Chain）

```
BaseConfig (根配置)
    ↓
BaseDatasetConfig (数据集通用配置)
    ↓
VIVTimeSeriesClassificationDatasetConfig (VIV特化配置)
```

**设计考虑**：
- 每层负责特定领域参数
- 参数校验层级递进
- 支持灵活的参数组合

---

## 🏛 核心模块设计

### BaseDatasetConfig 配置基类设计

#### 参数分类策略

```
基础配置 (路径、类型)
    ↓
数据划分配置 (train/val/test)
    ↓
数据加载配置 (batch_size、worker等)
    ↓
预处理配置 (normalize、resize等)
    ↓
数据增强配置 (flip、rotate等)
    ↓
缓存配置 (内存/磁盘)
    ↓
分布式配置 (多卡训练)
```

**设计理由**：
- 参数从基础到高级递进
- 相关参数聚集在一起
- 便于理解和维护

#### 校验策略

```python
单字段校验
    ↓
多字段关联校验
    ↓
业务逻辑校验
```

**例**：
- 单字段：`split_ratio` 需在0-1之间
- 关联：`split_ratio + test_ratio ≤ 1.0`
- 业务：缓存目录必须存在

---

### BaseDataset 数据集基类设计

#### 职责分离

| 职责 | 实现位置 |
|------|---------|
| 文件管理 | BaseDataset |
| 数据划分 | BaseDataset |
| 缓存管理 | BaseDataset |
| **数据解析** | **子类实现** |
| **预处理流程** | **子类实现** |
| **getitem格式** | **子类实现** |

#### 数据划分机制

```
加载所有文件路径
    ↓
检查auto_split配置
    ├─ True: 自动生成train/val/test划分
    └─ False: 使用全量路径（不划分）
    ↓
getitem时检查_dataset_mode
    ├─ "full": 使用全量路径
    ├─ "train": 使用train_paths
    ├─ "val": 使用val_paths
    └─ "test": 使用test_paths
```

**设计优势**：
- 同一实例可产生不同划分的子实例
- 底层数据共享，减少内存占用
- 训练/验证/测试集配置可独立差异化

#### 缓存策略

```
内存缓存
    ├─ 优点: 最快速度
    └─ 缺点: 显存占用大
    
磁盘缓存
    ├─ 优点: 平衡速度和容量
    └─ 缺点: 磁盘I/O开销
    
无缓存
    ├─ 优点: 节省空间
    └─ 缺点: 重复计算开销
```

---

### VIVTimeSeriesClassificationDataset 时序分类实现设计

#### 双模式输出设计

```
原始时序数据 (shape: seq_len × feat_dim)
    ↓
├─ time_series模式
│  └─ 直接返回
│     输出: (seq_len, feat_dim)
│
└─ grid_2d模式
   └─ Reshape为50×60网格
      输出: (1, 50, 60)
```

**设计考虑**：
- time_series: 适配LSTM、GRU等RNN
- grid_2d: 适配CNN、ResNet等卷积网络

#### 全局归一化设计

```
[问题] 单样本归一化导致
    ├─ 训练集内部分布一致性破坏
    └─ 训练/验证集归一化基础不一致

[解决方案] 全局归一化
    ├─ Step 1: 遍历训练集计算mean/std
    ├─ Step 2: 用训练集统计量作为基准
    ├─ Step 3: 所有样本（含验证/测试）用同一基准
    └─ Result: 分布一致，模型泛化性好
```

#### 序列处理流程

```
不同长度的原始序列
    ↓
检查fix_seq_len配置
    ├─ None: 保留原始长度
    │   └─ 需自定义batch处理（PaddedSequenceBatch）
    │
    └─ Specified: 统一到fix_seq_len
        ├─ 短序列 (len < fix_seq_len)
        │   ├─ pad_mode='zero': 补0
        │   ├─ pad_mode='repeat': 重复最后值
        │   └─ pad_mode='mean': 补特征均值
        │
        └─ 长序列 (len > fix_seq_len)
            ├─ trunc_mode='head': 截断头部
            └─ trunc_mode='tail': 截断尾部
```

---

## 🔄 数据流向

### 数据从文件到模型的流程

```
原始 .mat 文件
    ↓
_parse_sample() 解析
    ├─ 加载 .mat
    ├─ 提取data和label
    └─ 返回原始数据
    ↓
预处理
    ├─ 序列长度处理 (fix_seq_len)
    ├─ 归一化 (normalize)
    └─ 模式转换 (output_mode)
    ↓
缓存 (可选)
    ├─ 内存缓存
    └─ 磁盘缓存
    ↓
DataLoader 批处理
    ├─ 堆叠样本
    ├─ 转为Tensor
    └─ 移到GPU
    ↓
模型输入
```

### 配置从定义到应用的流程

```
YAML 配置文件 (可选)
    ↓
配置类实例化
    ├─ Pydantic 校验
    ├─ 默认值填充
    └─ 类型转换
    ↓
合并用户参数 (可选)
    └─ merge_dict()
    ↓
工厂创建数据集
    └─ get_dataset(config)
    ↓
数据集实例
    ├─ 读取配置参数
    ├─ 加载文件路径
    ├─ 执行数据划分
    └─ 初始化缓存
```

---

## 🔌 扩展机制

### 添加新数据集的步骤

#### Step 1: 创建配置类

```python
# src/config/data_processer/datasets/NewDataset/config.py

from src.config.data_processer.datasets.data_factory import BaseDatasetConfig

class NewDatasetConfig(BaseDatasetConfig):
    # 新增参数
    custom_param1: str = Field(default="value1")
    custom_param2: int = Field(default=100, ge=1)
    
    # 校验器
    @validator("custom_param1")
    def validate_custom(cls, v):
        ...
```

#### Step 2: 创建数据集类

```python
# src/data_processer/datasets/NewDataset/dataset.py

from src.data_processer.datasets.VIV2NumClassification.BaseDataset import BaseDataset
from src.config.data_processer.datasets.NewDataset.config import NewDatasetConfig

class NewDataset(BaseDataset):
    def __init__(self, config: NewDatasetConfig):
        super().__init__(config)
        self.custom_param1 = config.custom_param1
    
    def _parse_sample(self, file_path):
        # 实现数据解析
        ...
    
    def __getitem__(self, idx):
        # 实现getitem
        ...
```

#### Step 3: 注册到注册表

```python
# src/config/registry.py

CONFIG_CLASS_REGISTRY["new_dataset"] = NewDatasetConfig
DATASET_CLASS_REGISTRY["new_dataset"] = NewDataset
```

#### Step 4: 使用新数据集

```python
from src.config.data_processer.datasets.NewDataset.config import NewDatasetConfig
from src.data_processer.datasets.data_factory import get_dataset

config = NewDatasetConfig(data_dir="./data")
dataset = get_dataset(config)
```

---

## 📊 类关系图

```
BaseConfig (pydantic)
    ↑
    │
    └─ BaseDatasetConfig
        │
        ├─ VIVTimeSeriesClassificationDatasetConfig
        │
        └─ (其他数据集配置)

Dataset (PyTorch)
    ↑
    │
    └─ BaseDataset
        │
        ├─ VIVTimeSeriesClassificationDataset
        │
        ├─ VIVTimeSeriesClassificationDataset2NumClasses
        │
        └─ (其他数据集)

工厂函数
    get_dataset(config)
    └─ 通过注册表查找
       └─ 返回对应数据集实例
```

---

## 🎯 设计目标与权衡

| 目标 | 实现方案 | 权衡 |
|------|---------|------|
| **灵活性** | 配置类参数众多 | vs. 过度复杂化 |
| **重用性** | 基类封装通用功能 | vs. 子类实现复杂 |
| **性能** | 支持多层缓存 | vs. 内存占用 |
| **易用性** | 工厂模式隐藏复杂性 | vs. 黑盒难调试 |
| **可维护性** | 分层清晰职责 | vs. 层级过多 |

---

## 🔍 关键设计决策

### 决策1: 为什么使用Pydantic而不是dataclass?

**原因**:
- Pydantic自带类型校验和转换
- 支持复杂的校验器（validator）
- 序列化/反序列化能力强
- 与配置文件生态兼容

### 决策2: 为什么数据集类返回Tensor而不是numpy?

**原因**:
- PyTorch原生接口
- GPU传输更高效
- 与DataLoader无缝集成
- 避免多余的类型转换

### 决策3: 为什么支持multiple output modes?

**原因**:
- LSTM需要(seq_len, feat_dim)格式
- CNN需要(H, W)格式
- 单一格式降低灵活性
- 支持多种网络架构

### 决策4: 为什么use global normalization stats?

**原因**:
- 单样本归一化破坏数据分布
- 训练/验证集缩放基础不同
- 全局基准确保一致性
- 提升模型泛化能力

---

## 📈 性能优化策略

### 1. 缓存分层

```
L1 (最快): 内存缓存
    ├─ 适用于: 小数据集 (<10GB)
    └─ 加速倍数: 10-100×
    
L2 (中速): 磁盘缓存
    ├─ 适用于: 中等数据集
    └─ 加速倍数: 2-5×
    
L3 (最慢): 实时计算
    ├─ 适用于: 大数据集或内存受限
    └─ 加速倍数: 1×
```

### 2. 多进程加载优化

```
单进程 (num_workers=0)
    ├─ 优点: 简单、调试容易、Windows兼容
    └─ 场景: 小数据集、Windows环境
    
多进程 (num_workers=4-8)
    ├─ 优点: 充分利用CPU并行性
    └─ 场景: 大数据集、Linux/Mac
```

### 3. 序列长度优化

```
变长序列
    ├─ 优点: 保留原始信息、无数据丢失
    └─ 缺点: batch处理复杂、GPU利用率低
    
固定长度 + padding
    ├─ 优点: batch处理简单、GPU并行高效
    └─ 缺点: padding可能引入噪声
```

---

## 🧪 测试策略

### 单元测试范围

```
配置类
    ├─ 参数校验
    ├─ merge_dict
    └─ load_yaml
    
数据集基类
    ├─ 文件路径加载
    ├─ 数据划分逻辑
    └─ 缓存机制
    
具体实现
    ├─ _parse_sample
    ├─ __getitem__
    └─ 数据格式验证
    
工厂函数
    ├─ 配置-数据集映射
    └─ 错误处理
```

### 集成测试范围

```
完整流程
    ├─ YAML加载 → 创建数据集 → DataLoader → 模型输入
    ├─ 不同模式下的数据正确性
    └─ 缓存一致性验证
    
性能测试
    ├─ 内存占用
    ├─ 加载速度
    └─ DataLoader吞吐量
```

---

## 📚 参考资源

- [完整文档](README.md)
- [API参考](API.md)
- [快速入门](QUICKSTART.md)
