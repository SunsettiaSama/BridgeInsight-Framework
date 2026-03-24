# 深度学习模块 API 文档

## 概述

深度学习模块提供了统一的模型工厂接口和多种预实现的神经网络架构，支持多种任务类型（分类、分割、回归等）。所有模型采用配置驱动的设计模式，通过配置类管理参数，确保代码的可维护性和灵活性。

---

## 核心 API

### 1. 模型工厂：`get_model()`

**文件位置：** `src/deep_learning_module/model_factory.py`

#### 函数签名

```python
def get_model(config: BaseConfig) -> nn.Module
```

#### 功能说明

通过配置类实例自动创建对应的模型实例。工厂遵循以下流程：

1. 根据配置类的类型，从 `MODEL_CONFIG_REGISTRY` 反向查找对应的 `model_type`
2. 从 `MODEL_CLASS_REGISTRY` 获取对应的模型类
3. 用配置类实例初始化模型并返回

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `BaseConfig` | 具体的模型配置类实例（如 `SimpleMLPConfig`、`UNetConfig` 等） |

#### 返回值

| 类型 | 说明 |
|------|------|
| `nn.Module` | PyTorch 模型实例 |

#### 异常

| 异常类型 | 触发条件 |
|---------|---------|
| `ValueError` | 配置类未注册或无对应模型类 |
| `ImportError` | 模型类导入失败 |

#### 使用示例

```python
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig
from src.deep_learning_module.model_factory import get_model
import torch

# 创建模型配置
config = SimpleMLPConfig(
    input_shape=(300, 5),
    hidden_dims=[128, 64],
    num_classes=3,
    task_type="classification"
)

# 通过工厂创建模型
model = get_model(config)

# 前向传播测试
x = torch.randn(16, 300, 5)
output = model(x)
print(f"输出形状：{output.shape}")  # (16, 3)
```

---

## 模型类

### 2. MLP - 多层感知机

**文件位置：** `src/deep_learning_module/models/mlp.py`

#### 模型类签名

```python
class MLP(nn.Module):
    def __init__(self, config: SimpleMLPConfig)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### 核心特性

1. **任务可配置**：支持分类（`classification`）和回归（`regression`）两种任务
2. **轻量化设计**：隐藏层结构简洁，支持 Dropout 正则化
3. **输入兼容**：自动扁平化时序和网格输入
4. **配置驱动**：通过 `SimpleMLPConfig` 管理所有参数

#### 支持的任务类型

| 任务类型 | 输出格式 | 说明 |
|---------|---------|------|
| `classification` | `(batch_size, num_classes)` logits | 多分类任务 |
| `regression` | `(batch_size, regression_output_dim)` | 回归任务 |

#### 输入/输出说明

| 维度 | 格式 | 说明 |
|------|------|------|
| 输入 | `(batch_size, *input_shape)` | 支持任意维度（会自动扁平化） |
| 输出（分类） | `(batch_size, num_classes)` | 分类 logits |
| 输出（回归） | `(batch_size, regression_output_dim)` | 连续预测值 |

#### 配置参数

```python
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig

config = SimpleMLPConfig(
    input_shape=(300, 5),              # 输入形状（扁平化前）
    hidden_dims=[128, 64],             # 隐藏层维度列表
    activation_type="ReLU",            # 激活函数（"ReLU", "GELU", "LeakyReLU" 等）
    num_classes=3,                     # 分类任务：类别数
    regression_output_dim=1,           # 回归任务：输出维度
    task_type="classification",        # 任务类型
    dropout=DropoutConfig(             # Dropout 配置
        enable=True,
        prob=0.2
    )
)
```

#### 使用示例

**分类任务：**

```python
import torch
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig
from src.deep_learning_module.model_factory import get_model

config = SimpleMLPConfig(
    input_shape=(300, 5),
    hidden_dims=[256, 128, 64],
    num_classes=3,
    task_type="classification",
    activation_type="ReLU",
    dropout=DropoutConfig(enable=True, prob=0.2)
)

model = get_model(config)
x = torch.randn(8, 300, 5)
output = model(x)  # shape: (8, 3)
```

**回归任务：**

```python
config = SimpleMLPConfig(
    input_shape=(1, 50, 60),
    hidden_dims=[512, 256],
    regression_output_dim=1,
    task_type="regression",
    activation_type="GELU",
    dropout=DropoutConfig(enable=False)
)

model = get_model(config)
x = torch.randn(4, 1, 50, 60)
output = model(x)  # shape: (4, 1)
```

---

### 3. U-Net - 分割和序列处理

**文件位置：** `src/deep_learning_module/models/unet.py`

#### 模型类签名

```python
class UNet(nn.Module):
    def __new__(cls, config: UNetConfig)  # 工厂方法，返回 UNet2D 或 UNet1D
    
class UNet2D(nn.Module):
    def __init__(self, config: UNetConfig)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    
class UNet1D(nn.Module):
    def __init__(self, config: UNetConfig)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### 核心特性

1. **自适应路由**：根据 `support_timeseries` 自动选择 2D 或 1D U-Net
2. **2D U-Net**：完整的编码器-解码器架构，用于图像数据
3. **1D U-Net**：时序卷积，用于时序数据
4. **多任务支持**：分割、分类、回归
5. **跳跃连接**：编码器特征与解码器特征融合

#### 支持的任务类型

| 任务类型 | 输出格式 | 说明 |
|---------|---------|------|
| `segmentation` | 与输入空间维度相同 | 像素级预测（2D: `(B,C,H,W)`, 1D: `(B,C,T)`) |
| `classification` | `(batch_size, num_classes)` | 全局分类任务 |
| `regression` | `(batch_size, regression_output_dim)` | 回归任务 |

#### 2D U-Net 输入/输出

| 维度 | 格式 | 说明 |
|------|------|------|
| 输入 | `(batch_size, in_channels, height, width)` | 标准图像格式 |
| 输出（分割） | `(batch_size, num_classes, height, width)` | 分割掩码 |
| 输出（分类） | `(batch_size, num_classes)` | 分类 logits |
| 输出（回归） | `(batch_size, regression_output_dim)` | 连续预测值 |

#### 1D U-Net 输入/输出

| 维度 | 格式 | 说明 |
|------|------|------|
| 输入 | `(batch_size, in_channels, time_steps)` 或 `(batch_size, time_steps, in_channels)` | 时序格式，自动转置 |
| 输出（分割） | `(batch_size, num_classes, time_steps)` | 时序预测 |
| 输出（分类） | `(batch_size, num_classes)` | 分类 logits |
| 输出（回归） | `(batch_size, regression_output_dim)` | 连续预测值 |

#### 配置参数

```python
from src.config.deep_learning_module.models.unet import UNetConfig, EncoderConfig, DropoutConfig

config = UNetConfig(
    in_channels=1,                     # 输入通道数
    num_classes=3,                     # 输出类别数/通道数
    encoder=EncoderConfig(
        feature_channels=[64, 128, 256, 512],  # 各层通道数
        pool_size=2                    # 池化核大小
    ),
    dropout=DropoutConfig(
        enable=True,
        prob=0.2
    ),
    task_type="segmentation",          # 任务类型
    support_timeseries=False,          # 是否使用 1D U-Net
    regression_output_dim=1,           # 回归任务输出维度
    input_size=(256, 256),             # 输入大小（可选）
    output_size=(256, 256)             # 输出大小（可选）
)
```

#### 使用示例

**2D 分割任务：**

```python
import torch
from src.config.deep_learning_module.models.unet import UNetConfig, EncoderConfig, DropoutConfig
from src.deep_learning_module.model_factory import get_model

config = UNetConfig(
    in_channels=3,
    num_classes=5,
    encoder=EncoderConfig(
        feature_channels=[64, 128, 256, 512],
        pool_size=2
    ),
    task_type="segmentation",
    support_timeseries=False,
    dropout=DropoutConfig(enable=True, prob=0.1)
)

model = get_model(config)
x = torch.randn(4, 3, 256, 256)
output = model(x)  # shape: (4, 5, 256, 256)
```

**1D 时序分割任务：**

```python
config = UNetConfig(
    in_channels=16,
    num_classes=16,
    encoder=EncoderConfig(
        feature_channels=[64, 128, 256],
        pool_size=2
    ),
    task_type="segmentation",
    support_timeseries=True,           # 使用 1D U-Net
    dropout=DropoutConfig(enable=True, prob=0.15)
)

model = get_model(config)
# 支持两种格式输入
x1 = torch.randn(8, 16, 300)  # (batch, channels, time)
x2 = torch.randn(8, 300, 16)  # (batch, time, channels) - 自动转置
output = model(x1)  # shape: (8, 16, 300)
```

**1D 时序分类任务：**

```python
config = UNetConfig(
    in_channels=5,
    num_classes=3,                     # 分类：3 个类别
    encoder=EncoderConfig(
        feature_channels=[64, 128],
        pool_size=2
    ),
    task_type="classification",
    support_timeseries=True,
    dropout=DropoutConfig(enable=True, prob=0.2)
)

model = get_model(config)
x = torch.randn(16, 300, 5)
output = model(x)  # shape: (16, 3)
```

---

### 4. LSTM - 循环神经网络

**文件位置：** `src/deep_learning_module/models/lstm.py`

#### 模型类签名

```python
class LSTM(nn.Module):
    def __init__(self, config: LSTMConfig)
    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None) -> torch.Tensor | tuple
```

#### 核心特性

1. **三种任务模式**：分类、序列到序列、回归/长序列到短序列
2. **双向 LSTM**：支持双向传播
3. **多层 LSTM**：灵活配置层数
4. **序列化处理**：支持传入隐藏状态用于解码阶段

#### 支持的任务类型

| 任务类型 | 输出格式 | 说明 |
|---------|---------|------|
| `classification` | `(batch_size, num_classes)` | 取最后隐藏状态映射 |
| `seq2seq` | `(batch_size, seq_len, num_classes)` | 每个时间步预测，返回 `(output, (h_n, c_n))` |
| `regression` | `(batch_size, predict_seq_len, num_classes)` | 长序列预测短序列 |

#### 输入/输出说明

| 维度 | 格式 | 说明 |
|------|------|------|
| 输入 | `(batch_size, seq_len, input_size)` | 时序数据（`batch_first=True`） |
| 输出（分类） | `(batch_size, num_classes)` | 分类 logits |
| 输出（seq2seq） | `(batch_size, seq_len, num_classes)` + `(h_n, c_n)` | 序列输出和隐藏状态 |
| 输出（回归） | `(batch_size, predict_seq_len, num_classes)` | 固定长度的预测序列 |

#### 配置参数

```python
from src.config.deep_learning_module.models.lstm import LSTMConfig

config = LSTMConfig(
    input_size=16,                     # 输入特征维度
    hidden_size=64,                    # 隐藏层维度
    num_layers=2,                      # LSTM 层数
    num_classes=3,                     # 输出类别/特征维度
    bidirectional=True,                # 是否双向
    dropout=0.1,                       # LSTM 层间 dropout（num_layers > 1）
    batch_first=True,                  # 输入格式是否为 (batch, seq, features)
    seq_dropout=0.1,                   # Seq2Seq 任务的 dropout
    classifier_dropout=0.2,            # 分类任务的 dropout
    regression_dropout=0.15,           # 回归任务的 dropout
    task_type="classification",        # 任务类型
    predict_seq_len=4                  # 回归任务：预测序列长度（可选）
)
```

#### 使用示例

**分类任务（二分类）：**

```python
import torch
from src.config.deep_learning_module.models.lstm import LSTMConfig
from src.deep_learning_module.model_factory import get_model

config = LSTMConfig(
    input_size=16,
    num_classes=2,
    hidden_size=64,
    num_layers=2,
    task_type="classification",
    bidirectional=True
)

model = get_model(config)
x = torch.randn(8, 100, 16)  # (batch=8, seq_len=100, input_size=16)
output = model(x)  # shape: (8, 2)
```

**Seq2Seq 任务（等长输出）：**

```python
config = LSTMConfig(
    input_size=16,
    num_classes=16,                    # 每个时间步输出 16 维
    hidden_size=128,
    num_layers=2,
    task_type="seq2seq",
    bidirectional=True,
    seq_dropout=0.1
)

model = get_model(config)
x = torch.randn(8, 100, 16)
output, (h_n, c_n) = model(x)  # output shape: (8, 100, 16)
```

**回归任务（长序列预测短序列）：**

```python
config = LSTMConfig(
    input_size=16,
    num_classes=16,
    hidden_size=128,
    num_layers=2,
    predict_seq_len=4,                 # 预测后 4 步
    task_type="regression",
    bidirectional=True,
    regression_dropout=0.15
)

model = get_model(config)
x = torch.randn(8, 100, 16)  # 输入：前 100 步
output = model(x)  # shape: (8, 4, 16) - 预测：后 4 步
```

---

### 5. CNN - 卷积神经网络

**文件位置：** `src/deep_learning_module/models/cnn.py`

#### 模型类签名

```python
class CNN(nn.Module):
    def __init__(self, config: CNNConfig)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### 核心特性

1. **灵活的卷积栈**：支持自定义卷积层配置
2. **多任务支持**：分类、时序分类、回归
3. **可选的全连接层**：支持跳过或自定义 FC 层
4. **Dropout 和池化**：内置正则化选项

#### 支持的任务类型

| 任务类型 | 输出格式 | 说明 |
|---------|---------|------|
| `classification` | `(batch_size, num_classes)` | 图像分类 |
| `time_series_classification` | `(batch_size, num_classes)` | 时序数据分类 |
| `regression` | `(batch_size, regression_output_dim)` | 回归任务 |

#### 配置参数

```python
from src.config.deep_learning_module.models.cnn import (
    CNNConfig, ConvConfig, PoolConfig, FCConfig, DropoutConfig
)

config = CNNConfig(
    in_channels=3,                     # 输入通道数
    input_size=(224, 224),             # 输入大小
    conv_configs=[
        ConvConfig(out_channels=64, kernel_size=3, stride=1, padding=1),
        ConvConfig(out_channels=128, kernel_size=3, stride=1, padding=1),
        ConvConfig(out_channels=256, kernel_size=3, stride=1, padding=1)
    ],
    pool_config=PoolConfig(
        pool_type="max",               # "max" 或 "avg"
        kernel_size=2,
        stride=2
    ),
    fc_configs=[
        FCConfig(out_features=512),
        FCConfig(out_features=256)
    ],
    num_classes=10,                    # 分类类别数
    dropout=DropoutConfig(enable=True, prob=0.5),
    task_type="classification",
    regression_output_dim=1            # 回归任务输出维度
)
```

#### 使用示例

**图像分类：**

```python
import torch
from src.config.deep_learning_module.models.cnn import (
    CNNConfig, ConvConfig, PoolConfig, FCConfig, DropoutConfig
)
from src.deep_learning_module.model_factory import get_model

config = CNNConfig(
    in_channels=3,
    input_size=(224, 224),
    conv_configs=[
        ConvConfig(out_channels=64, kernel_size=3, stride=1, padding=1),
        ConvConfig(out_channels=128, kernel_size=3, stride=1, padding=1)
    ],
    pool_config=PoolConfig(pool_type="max", kernel_size=2, stride=2),
    fc_configs=[FCConfig(out_features=512)],
    num_classes=10,
    dropout=DropoutConfig(enable=True, prob=0.5),
    task_type="classification"
)

model = get_model(config)
x = torch.randn(4, 3, 224, 224)
output = model(x)  # shape: (4, 10)
```

---

### 6. RNN - 循环神经网络

**文件位置：** `src/deep_learning_module/models/rnn.py`

#### 模型类签名

```python
class RNN(nn.Module):
    def __init__(self, config: RNNConfig)
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor | tuple
```

#### 核心特性

1. **多种 RNN 类型**：支持 GRU、LSTM 等变体
2. **双向处理**：支持双向 RNN
3. **多任务支持**：分类、seq2seq、回归
4. **隐藏状态管理**：支持传入和返回隐藏状态

#### 支持的任务类型

| 任务类型 | 输出格式 | 说明 |
|---------|---------|------|
| `classification` | `(batch_size, num_classes)` | 序列分类 |
| `seq2seq` | `(batch_size, seq_len, num_classes)` | 序列到序列 |
| `regression` | `(batch_size, predict_seq_len, num_classes)` | 长序列预测短序列 |

#### 配置参数

```python
from src.config.deep_learning_module.models.rnn import RNNConfig

config = RNNConfig(
    input_size=16,                     # 输入特征维度
    hidden_size=64,                    # 隐藏层维度
    num_layers=2,                      # RNN 层数
    num_classes=3,                     # 输出维度
    rnn_type="gru",                    # "rnn", "gru", "lstm"
    bidirectional=True,                # 是否双向
    dropout=0.1,                       # RNN 层间 dropout
    batch_first=True,                  # 输入是否为 (batch, seq, features)
    task_type="classification",        # 任务类型
    predict_seq_len=4                  # 回归任务：预测序列长度
)
```

---

## 模型注册表

**文件位置：** `src/deep_learning_module/model_registry.py`

#### 功能

维护两个全局注册表，实现配置类与模型类的映射：

| 注册表 | 说明 |
|--------|------|
| `MODEL_CONFIG_REGISTRY` | 映射 `model_type` → 配置类 |
| `MODEL_CLASS_REGISTRY` | 映射 `model_type` → 模型类 |

#### 当前支持的模型类型

```python
MODEL_CONFIG_REGISTRY = {
    "unet": UNetConfig,
    "mlp": SimpleMLPConfig,
    "lstm": LSTMConfig,
    "cnn": CNNConfig,
    "rnn": RNNConfig,
}

MODEL_CLASS_REGISTRY = {
    "unet": UNet,
    "mlp": MLP,
    "lstm": LSTM,
    "cnn": CNN,
    "rnn": RNN,
}
```

#### 扩展模型

要添加新模型，需要：

1. 在 `src/config/deep_learning_module/models/` 创建配置类
2. 在 `src/deep_learning_module/models/` 创建模型类
3. 在 `MODEL_CONFIG_REGISTRY` 和 `MODEL_CLASS_REGISTRY` 中注册映射

---

## 常见使用模式

### 1. 完整的训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig
from src.deep_learning_module.model_factory import get_model

# 1. 创建配置
config = SimpleMLPConfig(
    input_shape=(300, 5),
    hidden_dims=[256, 128, 64],
    num_classes=3,
    task_type="classification",
    dropout=DropoutConfig(enable=True, prob=0.2)
)

# 2. 创建模型
model = get_model(config)

# 3. 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
model.train()
for epoch in range(10):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
```

### 2. 模型评估

```python
model.eval()
with torch.no_grad():
    predictions = []
    for x_batch, _ in test_loader:
        output = model(x_batch)
        predictions.append(output.argmax(dim=1))
    predictions = torch.cat(predictions, dim=0)
```

### 3. 模型保存和加载

```python
# 保存
torch.save(model.state_dict(), "model.pth")

# 加载
model = get_model(config)
model.load_state_dict(torch.load("model.pth"))
```

---

## 错误处理

### 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `ValueError: 未找到与配置类...对应的模型类型` | 配置类未注册 | 检查 `MODEL_CONFIG_REGISTRY` 是否包含该配置类 |
| `ImportError: 模型类导入失败` | 模型文件不存在或导入错误 | 检查模型文件路径和导入语句 |
| `输出形状错误` | 配置参数与预期不符 | 查看配置参数的维度说明，重新检查 `input_shape`、`num_classes` 等 |
| `维度错误（RuntimeError）` | 输入数据格式不符 | 查看模型的输入/输出说明，确认输入维度 |

---

## 性能优化

### 1. 批处理大小

建议根据 GPU 内存调整 `batch_size`，通常 16-256 范围内效果较好。

### 2. Dropout 调整

- 防止过拟合：增大 `dropout.prob`
- 保留更多信息：减小 `dropout.prob`

### 3. 模型深度

- 浅层模型（2-3 层）：训练快，适合小数据集
- 深层模型（4+ 层）：表达能力强，需要更多数据

---

## 参考资源

- PyTorch 官方文档：https://pytorch.org/docs/
- U-Net 论文：https://arxiv.org/abs/1505.04597
- LSTM 论文：https://arxiv.org/abs/1409.1556

