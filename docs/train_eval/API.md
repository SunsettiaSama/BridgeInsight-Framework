# 训练框架 API 文档

## 核心入口函数

### 1. create_trainer()

**文件位置**：`src/train_eval/deep_learning_module/run.py`

创建并初始化训练器实例。

#### 函数签名

```python
def create_trainer(config: BaseConfig) -> SFTTrainer
```

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `BaseConfig` | 训练配置对象 |

#### 返回值

| 类型 | 说明 |
|------|------|
| `SFTTrainer` | 初始化完成的训练器实例 |

#### 异常

| 异常类型 | 触发条件 |
|---------|---------|
| `TypeError` | config 不是 BaseConfig 实例 |
| `ValueError` | 配置参数不合法 |

#### 使用示例

```python
from src.config.trainer.base_config import BaseConfig
from src.train_eval.deep_learning_module import create_trainer

config = BaseConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    optimizer="Adam",
    device="cuda:0"
)

trainer = create_trainer(config)
```

---

### 2. train_model()

**文件位置**：`src/train_eval/deep_learning_module/run.py`

执行完整的模型训练流程。

#### 函数签名

```python
def train_model(
    trainer: SFTTrainer,
    model: nn.Module,
    train_dataloader: Union[DataLoader, Dataset],
    val_dataloader: Optional[Union[DataLoader, Dataset]] = None,
    epochs: Optional[int] = None
) -> None
```

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `trainer` | `SFTTrainer` | SFT 训练器实例 |
| `model` | `nn.Module` | PyTorch 模型 |
| `train_dataloader` | `DataLoader \| Dataset` | 训练数据加载器或数据集 |
| `val_dataloader` | `Optional[DataLoader \| Dataset]` | 验证数据加载器或数据集（可选） |
| `epochs` | `Optional[int]` | 覆盖配置中的轮数（可选） |

#### 使用示例

```python
from torch.utils.data import DataLoader
from src.train_eval.deep_learning_module import train_model

train_model(
    trainer=trainer,
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=10  # 可选，覆盖配置中的 epochs
)
```

---

### 3. main()

**文件位置**：`src/train_eval/deep_learning_module/run.py`

主训练入口函数，整合了 `create_trainer()` 和 `train_model()`。

#### 函数签名

```python
def main(
    config: BaseConfig,
    model: nn.Module,
    train_dataloader: Union[DataLoader, Dataset],
    val_dataloader: Optional[Union[DataLoader, Dataset]] = None,
) -> None
```

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `BaseConfig` | 训练配置对象 |
| `model` | `nn.Module` | PyTorch 模型 |
| `train_dataloader` | `DataLoader \| Dataset` | 训练数据加载器或数据集 |
| `val_dataloader` | `Optional[DataLoader \| Dataset]` | 验证数据加载器或数据集（可选） |

#### 使用示例

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

## 主要类

### SFTTrainer

**文件位置**：`src/train_eval/deep_learning_module/sft.py`

专用于深度学习模型的训练器。继承自 `BaseTrainer`。

#### 初始化

```python
trainer = SFTTrainer(
    config=config,
    model=None,  # 可选，也可在 train() 时传入
    train_dataloader=None,  # 可选
    val_dataloader=None  # 可选
)
```

#### 核心方法

##### train()

执行完整训练流程。

```python
trainer.train(
    model: Optional[nn.Module] = None,
    train_dataloader: Optional[Union[DataLoader, Dataset]] = None,
    val_dataloader: Optional[Union[DataLoader, Dataset]] = None
) -> None
```

**参数**：
- `model`：待训练模型（优先级高于初始化时的 model）
- `train_dataloader`：训练数据（支持 DataLoader 和 Dataset）
- `val_dataloader`：验证数据（支持 DataLoader 和 Dataset）

**功能**：
1. 校验和初始化核心组件
2. 模型初始化（预训练权重加载、层冻结）
3. 优化器和调度器初始化
4. 完整的 epoch 循环
5. 模型保存和监控
6. 资源清理

**示例**：

```python
trainer.train(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader
)
```

---

##### train_step()

单步训练逻辑（内部调用）。

```python
train_step(batch_data: Any, model: nn.Module) -> Dict[str, float]
```

**参数**：
- `batch_data`：批次数据，格式为 `(inputs, targets)`
- `model`：当前训练的模型

**返回值**：训练指标字典，包含 loss 和各类任务指标

**内部功能**：
- 前向传播（支持混合精度）
- 损失计算和反向传播
- 梯度裁剪和累积
- 优化器更新
- 指标计算

---

##### val_step()

单步验证逻辑（内部调用）。

```python
val_step(batch_data: Any, model: nn.Module) -> Dict[str, float]
```

**参数**：
- `batch_data`：批次数据，格式为 `(inputs, targets)`
- `model`：当前验证的模型

**返回值**：验证指标字典，包含 loss 和各类任务指标

---

##### eval()

独立评估流程。

```python
eval(
    model: Optional[nn.Module] = None,
    val_dataloader: Optional[Union[DataLoader, Dataset]] = None
) -> Dict[str, float]
```

**参数**：
- `model`：待评估模型（优先级高于实例属性）
- `val_dataloader`：验证数据

**返回值**：平均指标字典

**示例**：

```python
metrics = trainer.eval(model=model, val_dataloader=test_loader)
print(f"测试准确率：{metrics['accuracy']:.4f}")
```

---

##### save_checkpoint()

保存模型断点。

```python
save_checkpoint(is_best: bool = False, epoch: Optional[int] = None) -> None
```

**参数**：
- `is_best`：是否为最优模型
- `epoch`：当前轮次（默认使用 self.epoch）

**保存文件**：
- `latest_checkpoint.pth`：最新断点
- `best_checkpoint.pth`：最优模型
- `epoch_N_checkpoint.pth`：中间断点（可选）
- 对应的 `.json` 文件：模型结构和训练状态信息

---

##### load_checkpoint()

加载模型断点。

```python
load_checkpoint(checkpoint_path: str) -> None
```

**参数**：
- `checkpoint_path`：断点文件路径

**恢复内容**：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 混合精度缩放器状态
- 训练步数和最优指标

---

#### 配置参数（BaseConfig）

**文件位置**：`src/config/trainer/base_config.py`

##### 基本训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs` | int | 10 | 训练轮数 |
| `batch_size` | int | 32 | 批次大小 |
| `learning_rate` | float | 1e-3 | 初始学习率 |
| `device` | str \| list | "cpu" | 训练设备 |
| `output_dir` | str | "./output" | 输出目录 |

##### 优化器配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optimizer` | str | "Adam" | 优化器类型 |
| `weight_decay` | float | 1e-5 | 权重衰减 |
| `optimizer_params` | dict | {} | 额外参数 |

##### 学习率调度

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scheduler` | str \| None | None | 调度器类型 |
| `scheduler_params` | dict | {} | 调度器参数 |

##### 训练特性

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_mixed_precision` | bool | False | 混合精度训练 |
| `mixed_precision_type` | str | "float16" | 精度类型 |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |
| `gradient_clip_norm` | float | 0.0 | 梯度裁剪范数 |
| `use_distributed` | bool | False | 分布式训练 |

##### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pretrained_weight_path` | str \| None | None | 预训练权重路径 |
| `load_pretrained_head` | bool | False | 是否加载预测头 |
| `fix_feature_extractor` | bool | False | 冻结特征提取层 |
| `freeze_layer_prefixes` | list | [] | 要冻结的层前缀列表 |
| `head_param_prefixes` | list | ["head.", "classifier.", ...] | 预测头参数前缀 |
| `head_lr_scale` | float | 1.0 | 预测头学习率倍数 |

##### 模型保存

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_best_model` | bool | True | 保存最优模型 |
| `save_freq` | int | 1 | 保存频率（轮数） |
| `best_model_metric` | str | "loss" | 最优指标 |

##### 监控配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_tensorboard` | bool | False | 使用 TensorBoard |
| `tensorboard_log_dir` | str | "./tensorboard" | TensorBoard 日志目录 |
| `save_log_file` | bool | True | 保存日志文件 |
| `log_freq` | int | 10 | 日志频率 |

##### 任务配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sft_task_type` | str | "classification" | 任务类型 |
| `loss_type` | str | "CrossEntropyLoss" | 损失函数 |
| `is_multi_label` | bool | False | 多标签分类 |

---

## 损失函数

### 支持的损失函数

| 损失函数 | 用途 | 配置值 |
|---------|------|--------|
| CrossEntropyLoss | 多分类 | "CrossEntropyLoss" |
| BCELoss | 二分类（概率输入） | "BCELoss" |
| BCEWithLogitsLoss | 二分类（logits输入） | "BCEWithLogitsLoss" |
| MSELoss | 回归 | "MSELoss" |
| L1Loss | 回归（L1范数） | "L1Loss" |
| FocalLoss | 不平衡分类 | "FocalLoss" |

### FocalLoss 参数

```python
config = BaseConfig(
    loss_type="FocalLoss",
    focal_gamma=2.0,       # Focal Loss 的 gamma 参数
    num_classes=3,         # 分类任务的类别数
)
```

---

## 优化器和调度器

### 支持的优化器

| 优化器 | 配置值 | 推荐场景 |
|--------|--------|---------|
| Adam | "Adam" | 通用 |
| AdamW | "AdamW" | 推荐（带权重衰减） |
| SGD | "SGD" | 大批次训练 |
| RMSprop | "RMSprop" | RNN 任务 |

### 支持的调度器

| 调度器 | 配置值 | 用途 |
|--------|--------|------|
| StepLR | "StepLR" | 固定周期衰减 |
| CosineAnnealingLR | "CosineAnnealingLR" | 余弦退火 |
| ReduceLROnPlateau | "ReduceLROnPlateau" | 监控指标自动衰减 |
| ExponentialLR | "ExponentialLR" | 指数衰减 |
| OneCycleLR | "OneCycleLR" | 单周期学习率 |

### 调度器配置示例

```python
# StepLR
config.scheduler = "StepLR"
config.scheduler_params = {
    "step_size": 10,      # 每 10 轮衰减一次
    "gamma": 0.1          # 衰减因子
}

# CosineAnnealingLR
config.scheduler = "CosineAnnealingLR"
config.scheduler_params = {
    "T_max": 10,          # 最大迭代数
    "eta_min": 1e-5       # 最小学习率
}

# OneCycleLR
config.scheduler = "OneCycleLR"
config.scheduler_params = {
    "max_lr": 0.1,        # 最大学习率
    "div_factor": 25.0    # 初始学习率 = max_lr / div_factor
}
```

---

## 指标计算

### 分类任务指标

| 指标 | 说明 | 配置 |
|------|------|------|
| accuracy | 准确率 | "accuracy" |
| f1 | F1 分数 | "f1" |
| precision | 精准率 | "precision" |
| recall | 召回率 | "recall" |
| top_k_accuracy | Top-K 准确率 | "top_k_accuracy" |
| hamming_loss | 汉明损失（多标签） | "hamming_loss" |
| jaccard_score | 杰卡德相似度 | "jaccard_score" |

### 回归任务指标

| 指标 | 说明 | 配置 |
|------|------|------|
| mse | 均方误差 | "mse" |
| mae | 平均绝对误差 | "mae" |
| rmse | 均方根误差 | "rmse" |
| r2_score | R² 分数 | "r2_score" |

### 配置指标

```python
from src.config.trainer.base_config import BaseConfig

class MyConfig(BaseConfig):
    def get_train_evaluation_metrics(self) -> list:
        return ["loss", "accuracy", "f1", "precision", "recall"]

config = MyConfig(...)
```

---

## 高级用法

### 自定义模型初始化

```python
# 子类化 SFTTrainer
from src.train_eval.deep_learning_module.sft import SFTTrainer

class MyTrainer(SFTTrainer):
    def _init_model(self, model):
        # 自定义模型初始化逻辑
        model = super()._init_model(model)
        # 添加自定义处理
        return model
```

### 自定义指标计算

```python
class MyTrainer(SFTTrainer):
    def _compute_metrics(self, preds, targets, num_classes, is_multi_label):
        metrics = super()._compute_metrics(preds, targets, num_classes, is_multi_label)
        # 添加自定义指标
        metrics['custom_metric'] = 0.5
        return metrics
```

### 自定义训练步骤

```python
class MyTrainer(SFTTrainer):
    def train_step(self, batch_data, model):
        # 自定义训练逻辑
        metrics = super().train_step(batch_data, model)
        # 添加自定义处理
        return metrics
```

---

## 错误处理和调试

### 常见异常

| 异常 | 原因 | 解决方案 |
|------|------|---------|
| `RuntimeError: CUDA out of memory` | 显存不足 | 减小 batch_size 或启用梯度累积 |
| `ValueError: expected shape mismatch` | 输入输出维度不匹配 | 检查模型的前向传播 |
| `FileNotFoundError` | 断点文件不存在 | 检查文件路径 |
| `TypeError: unsupported operand type` | 数据类型错误 | 确保 inputs 和 targets 类型正确 |

### 启用调试模式

```python
config = BaseConfig(
    debug_mode=True,
    debug_max_steps=10,      # 最多 10 步后停止
    debug_val_freq=5,        # 每 5 个 batch 验证一次
)
```

---

## 性能参考

### 典型配置

| 场景 | batch_size | learning_rate | scheduler | 预计收敛时间 |
|------|-----------|---------------|-----------|-----------|
| MNIST（小数据） | 32 | 1e-3 | StepLR | 2-5 min |
| CIFAR-10（中数据） | 128 | 1e-4 | CosineAnnealingLR | 30-60 min |
| ImageNet（大数据） | 256 | 1e-4 | OneCycleLR | 数小时 |
| 微调预训练模型 | 16-32 | 1e-5 | ReduceLROnPlateau | 10-30 min |

---

## 相关文档

- **[快速开始](./QUICKSTART.md)**：5分钟入门指南
- **[架构说明](./README.md)**：模块设计和使用流程

