# 深度学习模块 - 配置参考卡

这是一个快速参考指南，列出所有模型的关键配置参数。

---

## MLP 配置速查

**文件**：`src/config/deep_learning_module/models/SimpleMLPConfig.py`

**类名**：`SimpleMLPConfig`

### 参数说明

```python
SimpleMLPConfig(
    # 必填参数
    input_shape: tuple,                # 输入形状 (不含 batch_size)
    hidden_dims: List[int],            # 隐藏层维度列表，如 [256, 128, 64]
    task_type: str,                    # "classification" 或 "regression"
    num_classes: int = 2,              # 分类任务的类别数
    regression_output_dim: int = 1,    # 回归任务的输出维度
    
    # 可选参数
    activation_type: str = "ReLU",     # 激活函数，如 "GELU", "LeakyReLU"
    dropout: DropoutConfig = None,     # Dropout 配置
)
```

### 使用示例

**分类任务**：
```python
config = SimpleMLPConfig(
    input_shape=(300, 5),
    hidden_dims=[128, 64],
    num_classes=3,
    task_type="classification",
    activation_type="ReLU",
    dropout=DropoutConfig(enable=True, prob=0.2)
)
```

**回归任务**：
```python
config = SimpleMLPConfig(
    input_shape=(100,),
    hidden_dims=[256, 128],
    regression_output_dim=1,
    task_type="regression",
    dropout=DropoutConfig(enable=False)
)
```

### 常用激活函数

- `"ReLU"`：最常用，计算快
- `"GELU"`：更平滑，更接近生物神经元
- `"LeakyReLU"`：避免死神经元
- `"Sigmoid"`：0-1 映射
- `"Tanh"`：-1 到 1 映射

---

## U-Net 配置速查

**文件**：`src/config/deep_learning_module/models/unet.py`

**类名**：`UNetConfig`

### 参数说明

```python
UNetConfig(
    # 必填参数
    in_channels: int,                  # 输入通道数
    num_classes: int,                  # 输出通道/类别数
    encoder: EncoderConfig,            # 编码器配置
    
    # 任务相关
    task_type: str = "segmentation",   # "segmentation"/"classification"/"regression"
    support_timeseries: bool = False,  # True 使用 1D U-Net, False 使用 2D U-Net
    regression_output_dim: int = 1,    # 回归任务的输出维度
    
    # 可选参数
    dropout: DropoutConfig = None,     # Dropout 配置
    input_size: tuple = None,          # 输入大小 (H, W)
    output_size: tuple = None,         # 输出大小 (H, W)
)
```

### 编码器配置

```python
EncoderConfig(
    feature_channels: List[int],       # 各层通道数，如 [64, 128, 256, 512]
    pool_size: int = 2,                # 池化核大小（通常为 2）
)
```

### 使用示例

**2D 分割任务**：
```python
config = UNetConfig(
    in_channels=1,
    num_classes=5,
    encoder=EncoderConfig(
        feature_channels=[64, 128, 256, 512],
        pool_size=2
    ),
    task_type="segmentation",
    support_timeseries=False,
    dropout=DropoutConfig(enable=True, prob=0.1)
)
```

**1D 时序分割**：
```python
config = UNetConfig(
    in_channels=16,
    num_classes=16,
    encoder=EncoderConfig(
        feature_channels=[64, 128, 256],
        pool_size=2
    ),
    task_type="segmentation",
    support_timeseries=True,
    dropout=DropoutConfig(enable=True, prob=0.15)
)
```

**1D 时序分类**：
```python
config = UNetConfig(
    in_channels=5,
    num_classes=3,                     # 分类类别数
    encoder=EncoderConfig(
        feature_channels=[64, 128],
        pool_size=2
    ),
    task_type="classification",
    support_timeseries=True,
    dropout=DropoutConfig(enable=True, prob=0.2)
)
```

### 选择通道数的建议

- 小数据集、快速原型：`[32, 64, 128]`
- 标准任务：`[64, 128, 256]` 或 `[64, 128, 256, 512]`
- 大数据集、复杂任务：`[64, 128, 256, 512, 1024]`

---

## LSTM 配置速查

**文件**：`src/config/deep_learning_module/models/lstm.py`

**类名**：`LSTMConfig`

### 参数说明

```python
LSTMConfig(
    # 必填参数
    input_size: int,                   # 输入特征维度
    hidden_size: int,                  # 隐藏层维度
    num_classes: int,                  # 输出类别/特征维度
    
    # LSTM 参数
    num_layers: int = 1,               # LSTM 层数
    bidirectional: bool = False,       # 是否双向
    dropout: float = 0.0,              # LSTM 层间 dropout（num_layers > 1）
    batch_first: bool = True,          # 输入格式是否为 (batch, seq, features)
    
    # 任务相关
    task_type: str = "classification", # "classification"/"seq2seq"/"regression"
    predict_seq_len: int = 4,          # 回归任务的预测序列长度
    
    # Dropout 配置（各任务独立）
    seq_dropout: float = 0.1,          # seq2seq 任务的 dropout
    classifier_dropout: float = 0.2,   # 分类任务的 dropout
    regression_dropout: float = 0.15,  # 回归任务的 dropout
)
```

### 使用示例

**分类任务**：
```python
config = LSTMConfig(
    input_size=16,
    hidden_size=64,
    num_classes=2,
    num_layers=2,
    bidirectional=True,
    task_type="classification",
    classifier_dropout=0.2
)
```

**Seq2Seq 任务**：
```python
config = LSTMConfig(
    input_size=16,
    hidden_size=128,
    num_classes=16,                    # 每步输出特征维度
    num_layers=2,
    bidirectional=True,
    task_type="seq2seq",
    seq_dropout=0.1
)
```

**回归任务（长→短）**：
```python
config = LSTMConfig(
    input_size=16,
    hidden_size=128,
    num_classes=16,
    num_layers=2,
    bidirectional=True,
    predict_seq_len=4,                 # 预测后 4 步
    task_type="regression",
    regression_dropout=0.15
)
```

### 参数选择建议

| 参数 | 小数据集 | 中等数据集 | 大数据集 |
|------|---------|----------|--------|
| `hidden_size` | 32-64 | 64-128 | 128-256 |
| `num_layers` | 1-2 | 2-3 | 3-4 |
| `bidirectional` | False | True | True |
| `dropout` | 0.1 | 0.1-0.2 | 0.2-0.3 |

---

## CNN 配置速查

**文件**：`src/config/deep_learning_module/models/cnn.py`

**类名**：`CNNConfig`

### 参数说明

```python
CNNConfig(
    # 必填参数
    in_channels: int,                  # 输入通道数
    input_size: tuple,                 # 输入大小 (H, W)
    conv_configs: List[ConvConfig],    # 卷积层配置列表
    num_classes: int,                  # 输出类别数
    
    # 可选参数
    pool_config: PoolConfig = None,    # 池化配置
    fc_configs: List[FCConfig] = None, # 全连接层配置
    dropout: DropoutConfig = None,     # Dropout 配置
    
    # 任务相关
    task_type: str = "classification", # "classification"/"regression"
    regression_output_dim: int = 1,    # 回归任务的输出维度
)
```

### 卷积层配置

```python
ConvConfig(
    out_channels: int,                 # 输出通道数
    kernel_size: int = 3,              # 卷积核大小
    stride: int = 1,                   # 步幅
    padding: int = 1,                  # 填充
)
```

### 池化配置

```python
PoolConfig(
    pool_type: str = "max",            # "max" 或 "avg"
    kernel_size: int = 2,              # 池化核大小
    stride: int = 2,                   # 步幅
)
```

### 全连接层配置

```python
FCConfig(
    out_features: int,                 # 输出特征数
)
```

### 使用示例

**图像分类**：
```python
config = CNNConfig(
    in_channels=3,
    input_size=(224, 224),
    conv_configs=[
        ConvConfig(out_channels=64, kernel_size=3, stride=1, padding=1),
        ConvConfig(out_channels=128, kernel_size=3, stride=1, padding=1),
        ConvConfig(out_channels=256, kernel_size=3, stride=1, padding=1)
    ],
    pool_config=PoolConfig(pool_type="max", kernel_size=2, stride=2),
    fc_configs=[FCConfig(out_features=512)],
    num_classes=10,
    dropout=DropoutConfig(enable=True, prob=0.5),
    task_type="classification"
)
```

---

## RNN 配置速查

**文件**：`src/config/deep_learning_module/models/rnn.py`

**类名**：`RNNConfig`

### 参数说明

```python
RNNConfig(
    # 必填参数
    input_size: int,                   # 输入特征维度
    hidden_size: int,                  # 隐藏层维度
    num_classes: int,                  # 输出维度
    
    # RNN 参数
    num_layers: int = 1,               # RNN 层数
    rnn_type: str = "lstm",            # "rnn"/"lstm"/"gru"
    bidirectional: bool = False,       # 是否双向
    dropout: float = 0.0,              # 层间 dropout
    batch_first: bool = True,          # 输入格式
    
    # 任务相关
    task_type: str = "classification", # "classification"/"seq2seq"/"regression"
    predict_seq_len: int = 4,          # 回归任务的预测序列长度
)
```

### 使用示例

**GRU 分类**：
```python
config = RNNConfig(
    input_size=16,
    hidden_size=64,
    num_classes=3,
    num_layers=2,
    rnn_type="gru",
    bidirectional=True,
    task_type="classification"
)
```

---

## Dropout 配置速查

**文件**：所有配置类通用

**类名**：`DropoutConfig`

### 参数说明

```python
DropoutConfig(
    enable: bool = True,               # 是否启用 dropout
    prob: float = 0.5,                 # Dropout 概率
)
```

### 推荐值

| 场景 | 推荐值 |
|------|--------|
| 小数据集 | 0.3-0.5 |
| 中等数据集 | 0.2-0.3 |
| 大数据集 | 0.1-0.2 |
| 无过拟合风险 | 0.0-0.1 |

---

## 完整配置示例模板

### MLP 完整配置

```python
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig

config = SimpleMLPConfig(
    # 必填
    input_shape=(300, 5),
    hidden_dims=[256, 128, 64],
    task_type="classification",
    num_classes=3,
    
    # 可选
    activation_type="ReLU",
    dropout=DropoutConfig(enable=True, prob=0.2)
)
```

### U-Net 完整配置

```python
from src.config.deep_learning_module.models.unet import UNetConfig, EncoderConfig, DropoutConfig

config = UNetConfig(
    # 必填
    in_channels=1,
    num_classes=5,
    encoder=EncoderConfig(
        feature_channels=[64, 128, 256, 512],
        pool_size=2
    ),
    
    # 可选
    task_type="segmentation",
    support_timeseries=False,
    dropout=DropoutConfig(enable=True, prob=0.1)
)
```

### LSTM 完整配置

```python
from src.config.deep_learning_module.models.lstm import LSTMConfig

config = LSTMConfig(
    # 必填
    input_size=16,
    hidden_size=64,
    num_classes=2,
    
    # 可选
    num_layers=2,
    bidirectional=True,
    dropout=0.1,
    task_type="classification",
    classifier_dropout=0.2
)
```

### CNN 完整配置

```python
from src.config.deep_learning_module.models.cnn import (
    CNNConfig, ConvConfig, PoolConfig, FCConfig, DropoutConfig
)

config = CNNConfig(
    in_channels=3,
    input_size=(224, 224),
    conv_configs=[
        ConvConfig(out_channels=64, kernel_size=3, stride=1, padding=1),
        ConvConfig(out_channels=128, kernel_size=3, stride=1, padding=1),
    ],
    pool_config=PoolConfig(pool_type="max", kernel_size=2, stride=2),
    fc_configs=[FCConfig(out_features=512)],
    num_classes=10,
    dropout=DropoutConfig(enable=True, prob=0.5),
    task_type="classification"
)
```

---

## 快速迁移指南

### 从 PyTorch 模型转换

如果您已有 PyTorch 模型，需要转换为本框架的配置驱动模式：

1. **确定模型类型**（MLP、U-Net、LSTM、CNN、RNN）
2. **提取模型参数**到对应的配置类
3. **验证输入输出维度**
4. **使用工厂创建新模型**

### 示例：转换 PyTorch MLP

```python
# 原始 PyTorch
model = nn.Sequential(
    nn.Linear(1500, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

# 转换为配置驱动
config = SimpleMLPConfig(
    input_shape=(300, 5),              # 1500 = 300 * 5
    hidden_dims=[256, 128, 64],        # 中间隐藏层
    num_classes=3,                     # 最后的输出
    task_type="classification"
)
model = get_model(config)
```

---

## 检查清单

使用本配置参考卡时的检查清单：

- [ ] 已选择合适的模型类型
- [ ] 已准备好输入数据和形状信息
- [ ] 已确定任务类型（分类/分割/回归）
- [ ] 已确定所有必填参数
- [ ] 已调整可选参数（激活函数、Dropout 等）
- [ ] 已验证配置参数的合理性
- [ ] 已通过工厂创建模型
- [ ] 已测试前向传播

---

## 相关文档

- **完整 API 文档**：API.md
- **快速开始指南**：QUICKSTART.md
- **架构和设计**：README.md
- **文档索引**：INDEX.md

