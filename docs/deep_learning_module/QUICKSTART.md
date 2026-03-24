# 深度学习模块快速开始

## 5分钟快速入门

### 第一步：导入必要的库

```python
import torch
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig
from src.deep_learning_module.model_factory import get_model
```

### 第二步：创建模型配置

```python
config = SimpleMLPConfig(
    input_shape=(300, 5),              # 输入形状
    hidden_dims=[128, 64],             # 隐藏层维度
    num_classes=3,                     # 类别数
    task_type="classification",        # 任务类型
    dropout=DropoutConfig(enable=True, prob=0.2)
)
```

### 第三步：通过工厂创建模型

```python
model = get_model(config)
print(f"模型类型: {type(model).__name__}")
```

### 第四步：前向传播测试

```python
x = torch.randn(16, 300, 5)            # (batch_size=16, seq_len=300, feat_dim=5)
with torch.no_grad():
    output = model(x)
print(f"输出形状: {output.shape}")      # (16, 3)
```

---

## 任务类型速查表

### 分类任务

```python
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig
from src.deep_learning_module.model_factory import get_model
import torch

# MLP 分类
config = SimpleMLPConfig(
    input_shape=(100,),
    hidden_dims=[64, 32],
    num_classes=3,
    task_type="classification"
)
model = get_model(config)
x = torch.randn(8, 100)
output = model(x)  # (8, 3)
```

### 分割任务

```python
from src.config.deep_learning_module.models.unet import UNetConfig, EncoderConfig, DropoutConfig
from src.deep_learning_module.model_factory import get_model
import torch

# U-Net 2D 分割
config = UNetConfig(
    in_channels=1,
    num_classes=5,
    encoder=EncoderConfig(
        feature_channels=[64, 128, 256],
        pool_size=2
    ),
    task_type="segmentation",
    support_timeseries=False,
    dropout=DropoutConfig(enable=True, prob=0.1)
)
model = get_model(config)
x = torch.randn(4, 1, 128, 128)
output = model(x)  # (4, 5, 128, 128)
```

### 时序分类

```python
from src.config.deep_learning_module.models.lstm import LSTMConfig
from src.deep_learning_module.model_factory import get_model
import torch

# LSTM 时序分类
config = LSTMConfig(
    input_size=16,
    num_classes=2,
    hidden_size=64,
    num_layers=2,
    task_type="classification",
    bidirectional=True
)
model = get_model(config)
x = torch.randn(8, 100, 16)  # (batch, seq_len, features)
output = model(x)  # (8, 2)
```

### 回归任务

```python
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig
from src.deep_learning_module.model_factory import get_model
import torch

# MLP 回归
config = SimpleMLPConfig(
    input_shape=(50, 5),
    hidden_dims=[128, 64],
    regression_output_dim=1,
    task_type="regression"
)
model = get_model(config)
x = torch.randn(8, 50, 5)
output = model(x)  # (8, 1)
```

---

## 完整训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig, DropoutConfig
from src.deep_learning_module.model_factory import get_model

# ==================== 数据准备 ====================
n_samples = 1000
X = torch.randn(n_samples, 300, 5)
y = torch.randint(0, 3, (n_samples,))

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==================== 模型配置 ====================
config = SimpleMLPConfig(
    input_shape=(300, 5),
    hidden_dims=[256, 128, 64],
    num_classes=3,
    task_type="classification",
    dropout=DropoutConfig(enable=True, prob=0.2)
)

model = get_model(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==================== 训练设置 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# ==================== 训练循环 ====================
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # 前向传播
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# ==================== 评估 ====================
model.eval()
with torch.no_grad():
    test_x = torch.randn(100, 300, 5).to(device)
    test_output = model(test_x)
    predictions = test_output.argmax(dim=1)
    print(f"预测结果形状: {predictions.shape}")  # (100,)
```

---

## 模型对比

### MLP vs U-Net vs LSTM vs CNN

| 特性 | MLP | U-Net | LSTM | CNN |
|------|-----|-------|------|-----|
| **最适应用场景** | 结构化数据 | 图像/分割 | 时序数据 | 图像分类 |
| **输入维度** | 任意（自动扁平） | 2D/1D | 3D (batch, seq, feat) | 4D (batch, C, H, W) |
| **计算速度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **参数数量** | 少 | 多 | 中等 | 多 |
| **分类支持** | ✅ | ✅ | ✅ | ✅ |
| **分割支持** | ❌ | ✅ | ❌ | ❌ |
| **回归支持** | ✅ | ✅ | ✅ | ✅ |
| **学习难度** | 简单 | 中等 | 中等 | 中等 |

---

## 调试技巧

### 1. 检查模型参数数量

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_parameters(model)
print(f"总参数: {total:,}, 可训练: {trainable:,}")
```

### 2. 可视化模型结构

```python
from torchsummary import summary

# 需要先安装: pip install torchsummary
summary(model, input_size=(300, 5), batch_size=16)
```

### 3. 验证输入输出维度

```python
import torch

# 测试输入
x = torch.randn(4, 300, 5)
print(f"输入形状: {x.shape}")

# 前向传播
with torch.no_grad():
    output = model(x)

print(f"输出形状: {output.shape}")
```

### 4. 检查配置

```python
# 打印配置信息
print(config)

# 或逐项查看
print(f"输入形状: {config.input_shape}")
print(f"隐藏层: {config.hidden_dims}")
print(f"任务类型: {config.task_type}")
```

---

## 常见问题

**Q: 如何改变模型的激活函数？**

A: 在配置中指定 `activation_type`：
```python
config = SimpleMLPConfig(
    input_shape=(300, 5),
    hidden_dims=[128, 64],
    activation_type="GELU",  # 或 "ReLU", "LeakyReLU", "Sigmoid" 等
    task_type="classification"
)
```

**Q: 如何防止过拟合？**

A: 调整 Dropout 概率或增加正则化：
```python
config = SimpleMLPConfig(
    ...,
    dropout=DropoutConfig(enable=True, prob=0.5)  # 增加 prob
)
```

**Q: LSTM 和 CNN 的区别是什么？**

A: 
- LSTM：对时序依赖关系建模，适合长期记忆
- CNN：通过卷积提取特征，计算更快

**Q: 如何使用 GPU 训练？**

A:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)
```

---

## 下一步

- 查看 [API.md](./API.md) 了解完整的接口文档
- 阅读 [README.md](./README.md) 了解模块架构
- 查看测试文件 `src/test/deep_learning_module/test_model_factory.py` 获取更多示例

