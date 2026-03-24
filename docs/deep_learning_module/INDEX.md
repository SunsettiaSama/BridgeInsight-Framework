# 深度学习模块 - 文档索引

## 📚 文档概览

本目录包含深度学习模块的完整文档，帮助您快速上手和深入理解模块设计。

---

## 📖 文档列表

### 1. [README.md](./README.md) - 模块架构和设计说明

**适合人群**：想要理解模块整体设计的开发者

**包含内容**：
- 模块概述和核心设计原则
- 完整的模块结构说明
- 6 种模型的详细介绍
- 使用流程和典型模式
- 任务类型支持说明
- 性能特性对比
- 最佳实践和错误调试

**关键部分**：
- 模块结构树
- 模型清单（MLP、U-Net、LSTM、CNN、RNN、EfficientVIT）
- 架构设计原理
- 任务类型速查表

**阅读时间**：15-20 分钟

---

### 2. [QUICKSTART.md](./QUICKSTART.md) - 快速开始指南

**适合人群**：想快速开始的新手用户

**包含内容**：
- 5 分钟快速入门
- 任务类型速查表
- 完整的训练示例
- 模型对比表
- 调试技巧
- 常见问题解答

**关键代码片段**：
- 完整的 MLP 分类、分割、时序分类、回归示例
- 端到端的训练循环
- 参数统计和维度调试方法

**阅读时间**：10-15 分钟

**特别推荐**：初次使用时必读

---

### 3. [API.md](./API.md) - 完整接口文档

**适合人群**：需要详细接口说明的开发者

**包含内容**：
- 模型工厂 `get_model()` 详解
- 6 种模型的完整 API 说明
  - MLP：多层感知机
  - U-Net：分割和序列处理
  - LSTM：循环神经网络
  - CNN：卷积神经网络
  - RNN：循环神经网络
  - EfficientVIT：预训练模型
- 模型注册表说明
- 完整使用示例
- 常见使用模式
- 错误处理指南
- 性能优化建议
- 参考资源链接

**API 速查**：
| 模型 | 文件 | 配置类 |
|------|------|--------|
| MLP | mlp.py | SimpleMLPConfig |
| U-Net | unet.py | UNetConfig |
| LSTM | lstm.py | LSTMConfig |
| CNN | cnn.py | CNNConfig |
| RNN | rnn.py | RNNConfig |

**阅读时间**：20-30 分钟（作为参考文档）

**特别推荐**：需要详细参数说明时查阅

---

## 🎯 快速导航

### 按用途查找

| 需求 | 推荐文档 | 对应章节 |
|------|---------|--------|
| 第一次使用 | QUICKSTART.md | 5分钟快速入门 |
| 学习模块架构 | README.md | 架构设计说明 |
| 查询 API | API.md | 对应模型章节 |
| 完整训练示例 | QUICKSTART.md | 完整训练示例 |
| 模型对比 | QUICKSTART.md | 模型对比 |
| 错误调试 | README.md | 错误调试指南 |
| 参数优化 | API.md | 性能优化 |

### 按模型查找

#### MLP（多层感知机）
- **快速开始**：QUICKSTART.md → 分类任务
- **完整 API**：API.md → 第2节
- **配置详解**：API.md → MLP 配置参数
- **使用示例**：QUICKSTART.md → 分类任务 / 回归任务

#### U-Net
- **快速开始**：QUICKSTART.md → 分割任务
- **完整 API**：API.md → 第3节
- **2D/1D 对比**：API.md → U-Net 特性
- **使用示例**：API.md → 使用示例

#### LSTM
- **快速开始**：QUICKSTART.md → 时序分类
- **完整 API**：API.md → 第4节
- **任务模式**：API.md → LSTM 任务类型
- **使用示例**：API.md → 使用示例

#### CNN
- **快速开始**：QUICKSTART.md → 模型对比
- **完整 API**：API.md → 第5节
- **配置详解**：API.md → CNN 配置参数
- **使用示例**：API.md → 使用示例

#### RNN
- **快速开始**：QUICKSTART.md → 模型对比
- **完整 API**：API.md → 第6节
- **任务支持**：API.md → RNN 支持的任务类型
- **使用示例**：API.md → 使用示例

### 按任务查找

#### 分类任务
1. 检查 QUICKSTART.md → 任务类型速查表 → 分类任务
2. 选择合适模型（MLP/CNN/LSTM）
3. 参考 API.md 查看完整配置说明
4. 参考 QUICKSTART.md → 完整训练示例

#### 分割任务
1. 检查 README.md → 任务类型支持 → 分割
2. 使用 U-Net 模型
3. 参考 API.md → U-Net 分割示例
4. 参考配置参数说明

#### 回归任务
1. 检查 QUICKSTART.md → 回归任务
2. 选择 MLP、U-Net 或 LSTM
3. 设置 `task_type="regression"`
4. 配置 `regression_output_dim`

#### 时序任务
1. 检查 QUICKSTART.md → 时序分类
2. 使用 LSTM 或 U-Net (1D)
3. 设置 `support_timeseries=True` (U-Net 用)
4. 参考对应模型的使用示例

---

## 📋 使用流程指南

### 第一次使用

```
阅读 QUICKSTART.md (5-10分钟)
        ↓
运行快速入门代码 (2-3分钟)
        ↓
选择适合的模型 (检查模型对比表)
        ↓
参考相应章节修改配置
        ↓
开始训练
```

### 遇到问题

```
遇到错误或异常
        ↓
检查 README.md → 错误调试指南
        ↓
查看对应模型的 API.md
        ↓
参考 QUICKSTART.md → 调试技巧
        ↓
查看源代码注释
```

### 性能优化

```
训练太慢？
        ↓
查看 README.md → 性能特性
        ↓
选择更快的模型或减少层数
        ↓
参考 API.md → 性能优化

模型不收敛？
        ↓
检查 QUICKSTART.md → 常见问题
        ↓
调整 dropout、学习率、batch_size
        ↓
查看完整训练示例
```

---

## 🔍 API 快速查询

### 模型工厂

```python
from src.deep_learning_module.model_factory import get_model

# 最通用的接口
model = get_model(config)
```

**相关文档**：API.md → 核心 API → 1. 模型工厂

### 配置创建

```python
# MLP
from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig
config = SimpleMLPConfig(...)

# U-Net
from src.config.deep_learning_module.models.unet import UNetConfig
config = UNetConfig(...)

# LSTM
from src.config.deep_learning_module.models.lstm import LSTMConfig
config = LSTMConfig(...)
```

**相关文档**：API.md → 对应模型部分

### 前向传播

```python
import torch
x = torch.randn(batch_size, *input_shape)
output = model(x)
```

**相关文档**：QUICKSTART.md → 第四步

---

## 📊 信息速查表

### 模型选择

| 任务类型 | 推荐模型 | 优点 | 缺点 |
|---------|---------|------|------|
| 结构化分类 | MLP | 简单快速 | 不能提取复杂特征 |
| 图像分类 | CNN | 特征提取强 | 参数多 |
| 图像分割 | U-Net | 精准分割 | 计算量大 |
| 时序分类 | LSTM | 捕捉长期依赖 | 速度慢 |
| 时序分割 | U-Net 1D | 精准分割 | 计算量大 |
| 长序列预测 | LSTM | 记忆能力强 | 训练复杂 |

### 输入输出维度速查

| 模型 | 输入 | 输出（分类） | 输出（分割） | 输出（回归） |
|------|------|----------|----------|----------|
| MLP | (B, *) | (B, C) | - | (B, D) |
| U-Net 2D | (B, C, H, W) | (B, C) | (B, C, H, W) | (B, D) |
| U-Net 1D | (B, C, T) | (B, C) | (B, C, T) | (B, D) |
| LSTM | (B, T, F) | (B, C) | - | (B, T', F) |
| CNN | (B, C, H, W) | (B, C) | - | (B, D) |
| RNN | (B, T, F) | (B, C) | - | (B, T', F) |

其中：B=batch_size, C=通道/类别数, T=时间步, F=特征维度, D=输出维度, H/W=高/宽

---

## 💡 学习路径建议

### 初学者（第一天）

1. 阅读 QUICKSTART.md 的前两部分（5分钟快速入门 + 任务类型速查表）
2. 运行快速入门代码，验证环境
3. 尝试修改参数，观察输出变化
4. 阅读模型对比部分，选择合适的模型

### 中级用户（第一周）

1. 阅读 README.md，理解模块架构
2. 阅读 API.md 的相关模型部分
3. 研究完整训练示例
4. 在自己的数据集上尝试训练

### 高级用户（持续学习）

1. 研究源代码实现
2. 修改或扩展模型
3. 阅读相关论文（U-Net、LSTM 等）
4. 贡献新模型或改进

---

## 🔗 相关资源链接

### 本项目资源

- **模型源代码**：`src/deep_learning_module/models/`
- **配置源代码**：`src/config/deep_learning_module/models/`
- **测试代码**：`src/test/deep_learning_module/`
- **运行脚本**：`src/train_eval/deep_learning_module/`

### 外部资源

- **PyTorch 官方文档**：https://pytorch.org/docs/
- **Pydantic 配置验证**：https://docs.pydantic.dev/
- **论文阅读**：
  - U-Net: https://arxiv.org/abs/1505.04597
  - LSTM: https://arxiv.org/abs/1409.1556
  - ResNet: https://arxiv.org/abs/1512.03385

---

## ❓ 常见问题速查

**Q: 应该从哪里开始？**
→ 从 QUICKSTART.md 开始，按照 5 分钟快速入门操作

**Q: 如何选择合适的模型？**
→ 查看 QUICKSTART.md 的模型对比表或 README.md 的模型清单

**Q: 如何查询完整的 API？**
→ 查看 API.md 中对应模型的章节

**Q: 遇到错误如何调试？**
→ 参考 README.md 的错误调试指南或 QUICKSTART.md 的调试技巧

**Q: 如何优化训练性能？**
→ 参考 API.md 的性能优化部分或 README.md 的性能特性

**Q: 需要完整的训练示例吗？**
→ 查看 QUICKSTART.md 的完整训练示例部分

---

## 📝 文档维护

本文档由多个 Markdown 文件组成：

- **README.md**：架构和总体设计（约 2000 行）
- **QUICKSTART.md**：快速开始和示例（约 400 行）
- **API.md**：完整接口文档（约 1500 行）
- **INDEX.md**（本文件）：导航和索引

所有文档保持同步，定期更新。

---

**祝您使用愉快！如有问题或建议，欢迎反馈。**

