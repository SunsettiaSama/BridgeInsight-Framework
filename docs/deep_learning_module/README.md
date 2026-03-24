# 深度学习模块

## 模块概述

深度学习模块提供统一的**配置驱动型**神经网络框架，支持多种模型架构和任务类型。核心设计思想是将模型配置和实现解耦，通过工厂模式实现灵活的模型创建。

### 核心设计原则

1. **配置驱动**：所有模型参数通过配置类管理，确保可重现性
2. **工厂模式**：统一的 `get_model()` 接口创建任意模型
3. **任务灵活性**：单个模型支持多种任务类型（分类、分割、回归等）
4. **层级清晰**：配置层、模型层、工厂层分离

---

## 模块结构

```
src/deep_learning_module/
├── model_factory.py          # 模型工厂（统一接口）
├── model_registry.py         # 模型注册表（配置-模型映射）
└── models/
    ├── mlp.py                # MLP（多层感知机）
    ├── unet.py               # U-Net（分割/序列处理）
    ├── lstm.py               # LSTM（循环神经网络）
    ├── cnn.py                # CNN（卷积神经网络）
    ├── rnn.py                # RNN（循环神经网络）
    ├── EfficientVIT.py       # 预训练模型示例
    └── __init__.py

src/config/deep_learning_module/models/
├── SimpleMLPConfig.py        # MLP 配置
├── unet.py                   # U-Net 配置
├── lstm.py                   # LSTM 配置
├── cnn.py                    # CNN 配置
├── rnn.py                    # RNN 配置
└── __init__.py
```

---

## 模型清单

### 1. MLP（多层感知机）

**适用场景**：结构化数据、时序数据扁平化处理

**核心特性**：
- 支持任意数量隐藏层
- 自动输入扁平化
- 支持分类和回归两种任务
- 灵活的激活函数和 Dropout 配置

**配置类**：`SimpleMLPConfig`

**模型文件**：`src/deep_learning_module/models/mlp.py`

### 2. U-Net

**适用场景**：图像分割、时序数据分割、多尺度特征提取

**核心特性**：
- 2D U-Net：标准编码器-解码器架构
- 1D U-Net：时序卷积版本
- 完整跳跃连接
- 支持分割、分类、回归三种任务

**配置类**：`UNetConfig`

**模型文件**：`src/deep_learning_module/models/unet.py`

### 3. LSTM

**适用场景**：时序预测、序列分类、序列到序列任务

**核心特性**：
- 多层 LSTM，支持双向
- 三种任务模式：分类、seq2seq、回归
- 灵活的隐藏状态管理
- 针对不同任务的 Dropout 配置

**配置类**：`LSTMConfig`

**模型文件**：`src/deep_learning_module/models/lstm.py`

### 4. CNN

**适用场景**：图像分类、时序卷积分类

**核心特性**：
- 可配置的卷积栈
- 灵活的池化和全连接层
- 多种任务支持
- 内置正则化选项

**配置类**：`CNNConfig`

**模型文件**：`src/deep_learning_module/models/cnn.py`

### 5. RNN

**适用场景**：序列处理、时序分类、长期依赖建模

**核心特性**：
- 支持 GRU、LSTM、标准 RNN
- 双向处理
- 多任务支持
- 隐藏状态管理

**配置类**：`RNNConfig`

**模型文件**：`src/deep_learning_module/models/rnn.py`

### 6. EfficientVIT（预训练模型）

**适用场景**：迁移学习、高效视觉任务

**特点**：预训练权重、快速推理

**模型文件**：`src/deep_learning_module/models/EfficientVIT.py`

---

## 使用流程

### 典型使用模式

```
配置类 (SimpleMLPConfig)
      ↓
工厂方法 (get_model)
      ↓
模型实例 (MLP)
      ↓
前向传播 (model.forward)
      ↓
输出结果
```

### 详细步骤

1. **导入必要的库**
   ```python
   from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig
   from src.deep_learning_module.model_factory import get_model
   ```

2. **创建配置**
   ```python
   config = SimpleMLPConfig(
       input_shape=(300, 5),
       hidden_dims=[128, 64],
       num_classes=3,
       task_type="classification"
   )
   ```

3. **获取模型**
   ```python
   model = get_model(config)
   ```

4. **前向传播**
   ```python
   import torch
   x = torch.randn(16, 300, 5)
   output = model(x)
   ```

---

## 架构设计说明

### 为什么使用配置类？

1. **可重现性**：配置即代码，易于版本控制
2. **灵活性**：无需修改模型代码，改配置即可调整参数
3. **验证**：Pydantic 配置类自动验证参数有效性
4. **可序列化**：配置可保存为 JSON/YAML

### 模型工厂的优势

```python
# 不需要知道具体模型类型
config = SimpleMLPConfig(...)
model = get_model(config)  # 自动返回 MLP 实例

# 工厂内部逻辑
1. 根据 config 类型查找 model_type
2. 从注册表获取模型类
3. 初始化模型
```

### 注册表机制

```python
MODEL_CONFIG_REGISTRY = {
    "mlp": SimpleMLPConfig,
    "unet": UNetConfig,
    # ...
}

MODEL_CLASS_REGISTRY = {
    "mlp": MLP,
    "unet": UNet,
    # ...
}
```

扩展模型只需：
1. 创建新配置类
2. 创建新模型类
3. 注册映射关系

---

## 任务类型支持

### 分类（Classification）

输出各类别的概率或 logits

```python
config = SimpleMLPConfig(
    ...,
    num_classes=3,
    task_type="classification"
)
# 输出: (batch_size, 3)
```

### 分割（Segmentation）

像素级预测，输出与输入空间维度相同

```python
config = UNetConfig(
    ...,
    num_classes=5,
    task_type="segmentation"
)
# 输出: (batch_size, 5, H, W) 或 (batch_size, 5, T)
```

### 回归（Regression）

连续值预测

```python
config = SimpleMLPConfig(
    ...,
    regression_output_dim=1,
    task_type="regression"
)
# 输出: (batch_size, 1)
```

### 序列到序列（Seq2Seq）

等长序列输出（LSTM/RNN 特定）

```python
config = LSTMConfig(
    ...,
    task_type="seq2seq"
)
# 输出: (batch_size, seq_len, num_classes), (h_n, c_n)
```

---

## 配置管理

### 配置文件示例

虽然当前使用代码创建配置，但配置可以序列化：

```python
import json
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig

config = SimpleMLPConfig(
    input_shape=(300, 5),
    hidden_dims=[128, 64],
    num_classes=3,
    task_type="classification"
)

# 转换为 dict（支持 JSON）
config_dict = config.model_dump()
json_str = json.dumps(config_dict)
```

### 加载配置

```python
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig

config = SimpleMLPConfig(**config_dict)
```

---

## 模型参数统计

### 查看参数数量

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total:,}")
    print(f"可训练: {trainable:,}")

count_parameters(model)
```

### 模型内存占用估算

```python
# 粗略估算（参数数量 × 4 字节/浮点数）
param_count = sum(p.numel() for p in model.parameters())
memory_mb = param_count * 4 / (1024 ** 2)
print(f"模型大小: {memory_mb:.2f} MB")
```

---

## 性能特性

### 计算复杂度对比

| 模型 | 时间复杂度 | 空间复杂度 | 推荐场景 |
|------|-----------|----------|--------|
| MLP | O(n·m) | O(m²) | 小型数据集 |
| CNN | O(k²·n·m) | O(k²·m) | 图像处理 |
| LSTM | O(n·h²) | O(n·h) | 长序列 |
| U-Net | O(n·m·log(n)) | O(n·m) | 大尺寸图像 |
| RNN | O(n·h²) | O(n·h) | 中序列 |

### 推理速度（相对值）

```
MLP:    ⭐⭐⭐⭐⭐ (最快)
CNN:    ⭐⭐⭐⭐
U-Net:  ⭐⭐⭐
RNN:    ⭐⭐⭐
LSTM:   ⭐⭐ (较慢)
```

---

## 错误调试指南

### 常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `ValueError: 未找到与配置类...对应的模型类型` | 配置类未注册 | 检查 `model_registry.py` |
| `ImportError: 模型类导入失败` | 模型文件不存在 | 确保模型文件在正确位置 |
| `RuntimeError: Expected 4D input` | 输入维度错误 | 检查模型对应的输入维度 |
| `CUDA out of memory` | GPU 内存不足 | 减少 batch_size 或模型大小 |

### 调试技巧

1. **验证配置**
   ```python
   print(config)  # 查看所有配置参数
   ```

2. **测试前向传播**
   ```python
   x = torch.randn(1, *config.input_shape)
   output = model(x)
   print(f"输出形状: {output.shape}")
   ```

3. **参数梯度检查**
   ```python
   for name, param in model.named_parameters():
       if param.requires_grad:
           print(f"{name}: {param.shape}")
   ```

---

## 最佳实践

### 1. 配置管理

```python
# ✅ 推荐：集中管理配置
class ExperimentConfig:
    @staticmethod
    def get_mlp_config():
        return SimpleMLPConfig(...)
    
    @staticmethod
    def get_unet_config():
        return UNetConfig(...)

# 使用
config = ExperimentConfig.get_mlp_config()
model = get_model(config)
```

### 2. 模型保存

```python
import torch

# 只保存权重
torch.save(model.state_dict(), "model.pth")

# 加载权重
model = get_model(config)
model.load_state_dict(torch.load("model.pth"))
```

### 3. 数据加载

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### 4. 训练循环

```python
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_output = model(val_x)
```

---

## 文档导航

- **[快速开始](./QUICKSTART.md)**：5分钟入门指南
- **[API 文档](./API.md)**：完整接口文档
- **[本文件](./README.md)**：架构和设计说明

---

## 相关资源

- **PyTorch 文档**：https://pytorch.org/docs/
- **配置类（Pydantic）**：https://docs.pydantic.dev/
- **模型论文**：
  - U-Net: https://arxiv.org/abs/1505.04597
  - LSTM: https://arxiv.org/abs/1409.1556
  - Convolutional Networks: https://arxiv.org/abs/1512.03385

---

## 贡献指南

### 添加新模型

1. 创建配置类：`src/config/deep_learning_module/models/your_model.py`
2. 创建模型类：`src/deep_learning_module/models/your_model.py`
3. 在 `model_registry.py` 中注册映射
4. 添加单元测试：`src/test/deep_learning_module/test_your_model.py`

### 更新文档

文档位于 `docs/deep_learning_module/`：
- `README.md`：架构和设计说明
- `QUICKSTART.md`：快速开始指南
- `API.md`：完整接口文档

---

## 变更日志

### 版本 1.0

- ✅ MLP 模型实现和配置
- ✅ U-Net 2D 和 1D 实现
- ✅ LSTM 多任务支持
- ✅ CNN 卷积网络
- ✅ RNN 循环网络
- ✅ 模型工厂和注册表
- ✅ 完整的 API 文档
- ✅ 快速开始指南

---

## 许可证

该模块遵循项目的许可证。

