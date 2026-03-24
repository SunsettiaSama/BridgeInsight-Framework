# 训练框架模块

## 模块概述

训练框架模块提供了统一的深度学习模型训练工具，支持多种模型架构和任务类型的端到端训练。核心设计基于**抽象基类 + 工厂模式 + 配置驱动**，确保高度的可维护性和可扩展性。

### 核心特性

1. **配置驱动**：所有训练参数通过 `BaseConfig` 配置类管理
2. **灵活的模型支持**：兼容任何 PyTorch `nn.Module` 模型
3. **多任务支持**：分类、回归、序列任务自动适配
4. **现代特性集**：混合精度、梯度累积、分布式训练、断点续训
5. **完整的监控**：TensorBoard、日志、指标统计
6. **生产就绪**：异常处理、资源清理、模型保存机制

---

## 模块结构

```
src/train_eval/deep_learning_module/
├── base.py              # 基础训练器（抽象基类）
├── sft.py               # SFT训练器（实现具体逻辑）
└── run.py               # 训练工作流入口

src/config/trainer/
└── base_config.py       # 训练配置基类（Pydantic）
```

---

## 核心类

### 1. BaseTrainer（基础训练器）

**位置**：`src/train_eval/deep_learning_module/base.py`

抽象基类，定义统一的训练接口和通用功能：

- **通用方法**：
  - `_init_device()` - 设备初始化
  - `_init_logger()` - 日志系统
  - `_get_loss()` - 损失函数工厂
  - `_get_optimizer()` - 优化器工厂
  - `save_checkpoint()` - 断点保存
  - `load_checkpoint()` - 断点加载
  - `update_best_metric()` - 最优指标更新

- **抽象方法**（子类必须实现）：
  - `_init_dataloaders()` - 数据加载器初始化
  - `_init_model()` - 模型初始化
  - `_init_optimizer_scheduler()` - 优化器和调度器
  - `train_step()` - 单步训练逻辑
  - `val_step()` - 单步验证逻辑
  - `train()` - 完整训练流程

### 2. SFTTrainer（SFT训练器）

**位置**：`src/train_eval/deep_learning_module/sft.py`

继承 `BaseTrainer`，实现专用于深度学习模型的训练逻辑：

- **特有方法**：
  - `_dataset_to_dataloader()` - Dataset 自动转换
  - `_freeze_model_layers()` - 层冻结
  - `_load_pretrained_weights()` - 预训练权重加载
  - `_compute_metrics()` - 指标计算
  - `eval()` - 独立评估

- **支持特性**：
  - 分层学习率（特征提取层 + 预测头）
  - 梯度累积和梯度裁剪
  - 混合精度训练
  - 分布式训练（DDP/DataParallel）
  - 自动指标监控

### 3. TrainStepState（训练步状态）

**位置**：`src/train_eval/deep_learning_module/base.py` 第 61-194 行

统一管理训练步的各类状态信息：

- **基础标识**：epoch、batch_idx、global_step
- **性能指标**：loss、metrics、batch_loss
- **优化器状态**：learning_rate、gradient_norm
- **资源效率**：耗时、内存使用
- **特殊参数**：梯度累积、分布式标记

---

## 使用流程

### 标准训练流程

```python
from src.config.trainer.base_config import BaseConfig
from src.train_eval.deep_learning_module import create_trainer, train_model

# 1. 创建配置
config = BaseConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    optimizer="Adam",
    device="cuda:0",
    # ... 其他配置
)

# 2. 创建模型（任何 nn.Module）
model = YourModel(...)

# 3. 准备数据加载器
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

# 4. 创建训练器
trainer = create_trainer(config)

# 5. 执行训练
train_model(trainer, model, train_loader, val_loader)
```

### 主入口方式

```python
from src.train_eval.deep_learning_module import main

main(
    config=config,
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader
)
```

---

## 配置参数

### 基本配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `epochs` | int | 训练轮数 |
| `batch_size` | int | 批次大小 |
| `learning_rate` | float | 初始学习率 |
| `device` | str \| list | 训练设备 |
| `output_dir` | str | 输出目录 |

### 优化器配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `optimizer` | str | 优化器类型（Adam/AdamW/SGD/RMSprop） |
| `weight_decay` | float | 权重衰减 |
| `optimizer_params` | dict | 额外参数 |

### 学习率调度配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `scheduler` | str | 调度器类型（StepLR/CosineAnnealingLR/OneCycleLR） |
| `scheduler_params` | dict | 调度器参数 |

### 训练特性配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `use_mixed_precision` | bool | 是否使用混合精度 |
| `gradient_accumulation_steps` | int | 梯度累积步数 |
| `gradient_clip_norm` | float | 梯度裁剪范数 |
| `use_distributed` | bool | 是否使用分布式训练 |

### 模型保存配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `save_best_model` | bool | 是否保存最优模型 |
| `save_freq` | int | 模型保存频率（轮数） |
| `best_model_metric` | str | 最优模型评估指标 |

### 监控配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `use_tensorboard` | bool | 是否使用 TensorBoard |
| `tensorboard_log_dir` | str | TensorBoard 日志目录 |
| `save_log_file` | bool | 是否保存日志文件 |
| `log_freq` | int | 日志打印频率 |

---

## 支持的特性

### 多任务类型

- ✅ 单标签分类（Binary Classification）
- ✅ 多标签分类（Multi-label Classification）
- ✅ 多分类（Multi-class Classification）
- ✅ 回归（Regression）
- ✅ 时序分类（Time Series Classification）
- ✅ 时序回归（Time Series Regression）

### 损失函数

- ✅ CrossEntropyLoss
- ✅ MSELoss / L1Loss
- ✅ BCELoss / BCEWithLogitsLoss
- ✅ FocalLoss（自定义）

### 优化器

- ✅ Adam
- ✅ AdamW（推荐）
- ✅ SGD（带动量）
- ✅ RMSprop

### 学习率调度器

- ✅ StepLR
- ✅ CosineAnnealingLR
- ✅ ReduceLROnPlateau
- ✅ ExponentialLR
- ✅ OneCycleLR

### 高级特性

- ✅ 混合精度训练（Automatic Mixed Precision）
- ✅ 梯度累积（Gradient Accumulation）
- ✅ 梯度裁剪（Gradient Clipping）
- ✅ 分层学习率（Layerwise Learning Rate）
- ✅ 分布式训练（DistributedDataParallel）
- ✅ 多卡训练（DataParallel）
- ✅ 断点续训（Checkpoint Resume）
- ✅ 预训练权重加载（Pretrained Weights）
- ✅ 层冻结（Layer Freezing）

---

## 输出结构

训练完成后的输出目录结构：

```
output_dir/
├── checkpoints/
│   ├── latest_checkpoint.pth           # 最新断点
│   ├── latest_checkpoint.json          # 最新断点信息
│   ├── best_checkpoint.pth             # 最优模型
│   ├── best_checkpoint.json            # 最优模型信息
│   ├── epoch_5_checkpoint.pth          # 中间断点（如果启用）
│   └── epoch_5_checkpoint.json
├── train_log.txt                       # 训练日志
└── tensorboard/
    └── events.out.tfevents.*           # TensorBoard 事件文件
```

---

## 最佳实践

### 1. 配置管理

推荐创建专门的配置类：

```python
class MyTrainerConfig(BaseConfig):
    # 自定义配置字段
    model_name: str = "my_model"
    custom_param: float = 0.1
```

### 2. 模型初始化

确保模型支持所需的任务输出：

```python
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ... 模型层定义 ...
        self.head = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # ... 前向传播 ...
        return self.head(x)
```

### 3. 数据加载器

支持 DataLoader 和 Dataset 两种输入方式：

```python
# 方式1：直接使用 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32)

# 方式2：传入 Dataset，自动转换
train_dataset = MyDataset(...)
# 训练器会自动转换为 DataLoader
```

### 4. 学习率调整

利用分层学习率微调预训练模型：

```python
config = BaseConfig(
    learning_rate=0.001,              # 特征提取层学习率
    head_lr_scale=10.0,               # 预测头学习率倍数
    fix_feature_extractor=True,       # 冻结特征提取层
)
```

---

## 错误处理

框架内置了全面的错误处理：

- 异常自动捕获并记录
- 异常发生时自动保存断点
- 资源自动清理（finally 块）
- 详细的错误日志输出

---

## 相关文档

- **[快速开始](./QUICKSTART.md)**：5分钟入门指南
- **[API 文档](./API.md)**：完整接口参考

---

## 性能指标

### 推荐配置

| 场景 | batch_size | learning_rate | optimizer | scheduler |
|------|-----------|---------------|-----------|-----------|
| 小数据集 | 16-32 | 1e-3 | Adam | StepLR |
| 中等数据集 | 32-64 | 1e-4 | AdamW | CosineAnnealingLR |
| 大数据集 | 64-256 | 1e-4 | AdamW | OneCycleLR |
| 微调模型 | 16-32 | 1e-5 | AdamW | ReduceLROnPlateau |

### 资源占用（概估）

- **内存**：约 batch_size × input_size × 4 字节（模型权重另计）
- **GPU 显存**：建议预留 1-2 GB 余量
- **速度**：单步训练 10-500ms（取决于模型和硬件）

---

## 故障排查

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 显存溢出 | batch_size 过大 | 减小 batch_size 或启用梯度累积 |
| 训练缓慢 | 数据加载瓶颈 | 增加 num_workers 或启用 pin_memory |
| 模型不收敛 | 学习率过大 | 减小 learning_rate 或使用调度器 |
| 断点加载失败 | 配置不匹配 | 确保当前配置与保存时一致 |

---

## 下一步

- 查看 **[QUICKSTART.md](./QUICKSTART.md)** 快速开始
- 查看 **[API.md](./API.md)** 了解完整接口
- 查阅源代码获取更多细节

