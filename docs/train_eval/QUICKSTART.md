# 训练框架快速开始

## 5分钟快速入门

### 第一步：导入必要的库

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.trainer.base_config import BaseConfig
from src.train_eval.deep_learning_module import create_trainer, train_model
```

### 第二步：准备数据

```python
# 创建虚拟数据集（实际使用自己的数据）
train_X = torch.randn(1000, 10)
train_y = torch.randint(0, 3, (1000,))
train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_X = torch.randn(200, 10)
val_y = torch.randint(0, 3, (200,))
val_dataset = TensorDataset(val_X, val_y)
val_loader = DataLoader(val_dataset, batch_size=32)
```

### 第三步：创建模型

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel()
```

### 第四步：创建训练配置

```python
config = BaseConfig(
    epochs=5,
    batch_size=32,
    learning_rate=0.001,
    optimizer="Adam",
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="./output",
    loss_type="CrossEntropyLoss",
    sft_task_type="classification",
    best_model_metric="accuracy",
    save_best_model=True,
    use_tensorboard=True,
)
```

### 第五步：创建训练器并训练

```python
trainer = create_trainer(config)
train_model(trainer, model, train_loader, val_loader)
```

**完成！** 模型已开始训练，输出保存在 `./output` 目录。

---

## 完整训练脚本示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.trainer.base_config import BaseConfig
from src.train_eval.deep_learning_module import create_trainer, train_model

# ==================== 数据准备 ====================
print("准备数据...")
train_X = torch.randn(1000, 10)
train_y = torch.randint(0, 3, (1000,))
train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_X = torch.randn(200, 10)
val_y = torch.randint(0, 3, (200,))
val_dataset = TensorDataset(val_X, val_y)
val_loader = DataLoader(val_dataset, batch_size=32)

# ==================== 模型定义 ====================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel()
print(f"模型创建完成，参数数量：{sum(p.numel() for p in model.parameters()):,}")

# ==================== 训练配置 ====================
config = BaseConfig(
    epochs=5,
    batch_size=32,
    learning_rate=0.001,
    optimizer="Adam",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    output_dir="./output",
    loss_type="CrossEntropyLoss",
    sft_task_type="classification",
    best_model_metric="accuracy",
    save_best_model=True,
    save_freq=1,
    use_tensorboard=True,
    tensorboard_log_dir="./tensorboard",
)

# ==================== 训练 ====================
print("开始训练...")
trainer = create_trainer(config)
train_model(trainer, model, train_loader, val_loader)

print("\n训练完成！")
print(f"最优模型指标：{trainer.best_metric:.4f} （第 {trainer.best_epoch+1} 轮）")
print(f"模型已保存至：{config.output_dir}")
```

运行结果示例：

```
准备数据...
模型创建完成，参数数量：10,275
开始训练...
============================================================
初始化训练流程
============================================================
损失函数初始化完成：CrossEntropyLoss
...
============================================================
开始SFT模型训练流程
============================================================

========== 第 1/5 轮训练 ==========
[训练进度] 10/32 (31.2%)
...
[Epoch 1 汇总]
训练平均损失：1.0234 | 验证平均损失：0.9856
训练accuracy：0.3456 | 验证accuracy：0.4123
==================================================

...

训练完成！
最优模型指标：0.5234 （第 3 轮）
模型已保存至：./output
```

---

## 常见任务示例

### 分类任务

```python
config = BaseConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    optimizer="Adam",
    loss_type="CrossEntropyLoss",
    sft_task_type="classification",    # 分类任务
    best_model_metric="accuracy",      # 以准确率评估
)
```

### 回归任务

```python
config = BaseConfig(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    optimizer="Adam",
    loss_type="MSELoss",               # 回归损失
    sft_task_type="regression",        # 回归任务
    best_model_metric="loss",          # 以损失评估
)
```

### 时序分类

```python
config = BaseConfig(
    epochs=10,
    batch_size=16,
    learning_rate=0.001,
    optimizer="Adam",
    loss_type="CrossEntropyLoss",
    sft_task_type="timeseries_classification",  # 时序分类
    best_model_metric="accuracy",
)
```

---

## 使用预训练模型微调

```python
# 1. 加载预训练权重
pretrained_model_path = "./pretrained_model.pth"

# 2. 创建配置（启用预训练权重加载）
config = BaseConfig(
    epochs=5,
    batch_size=32,
    learning_rate=0.0001,              # 微调使用更小学习率
    optimizer="AdamW",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    output_dir="./finetune_output",
    loss_type="CrossEntropyLoss",
    
    # 预训练相关配置
    pretrained_weight_path=pretrained_model_path,
    load_pretrained_head=False,        # 不加载原预测头
    fix_feature_extractor=True,        # 冻结特征提取层
    head_lr_scale=10.0,                # 预测头学习率提高 10 倍
)

# 3. 训练
trainer = create_trainer(config)
train_model(trainer, model, train_loader, val_loader)
```

---

## 启用高级特性

### 混合精度训练

```python
config = BaseConfig(
    # ... 其他配置 ...
    use_mixed_precision=True,          # 启用混合精度
    mixed_precision_type="float16",    # 精度类型
)
```

### 梯度累积

```python
config = BaseConfig(
    # ... 其他配置 ...
    gradient_accumulation_steps=4,     # 每 4 步更新一次参数
    batch_size=8,                      # 配合小 batch_size 使用
)
```

### 梯度裁剪

```python
config = BaseConfig(
    # ... 其他配置 ...
    gradient_clip_norm=1.0,            # 梯度范数上限
)
```

### 学习率调度

```python
from src.config.trainer.base_config import BaseConfig

config = BaseConfig(
    # ... 其他配置 ...
    scheduler="CosineAnnealingLR",     # 使用余弦退火调度器
    scheduler_params={
        "T_max": 10,                   # 最大迭代数
        "eta_min": 0.00001
    }
)
```

### 分布式训练

```python
config = BaseConfig(
    # ... 其他配置 ...
    device=["cuda:0", "cuda:1"],       # 多 GPU
    use_distributed=True,              # 启用分布式
)
```

---

## 模型评估

使用训练器的评估方法：

```python
# 加载最优模型
checkpoint = torch.load("./output/checkpoints/best_checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# 独立评估
trainer = create_trainer(config)
metrics = trainer.eval(model=model, val_dataloader=test_loader)

print(f"测试集结果：{metrics}")
```

---

## TensorBoard 可视化

启动 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir=./tensorboard
```

然后在浏览器打开 `http://localhost:6006` 查看：

- 训练/验证损失曲线
- 各类指标曲线
- 学习率变化
- 其他自定义标量

---

## 调试和日志

### 启用调试模式

```python
config = BaseConfig(
    # ... 其他配置 ...
    debug_mode=True,                   # 启用调试模式
    debug_max_steps=10,                # 最多执行 10 步
    debug_val_freq=5,                  # 每 5 个 batch 验证一次
    log_freq=2,                        # 每 2 个 batch 打印日志
)
```

### 查看日志

训练日志自动保存在：`output_dir/train_log.txt`

```bash
# 查看实时日志
tail -f ./output/train_log.txt
```

---

## 常见错误

### 1. 显存溢出

```
RuntimeError: CUDA out of memory
```

**解决方案**：
```python
# 减小 batch_size
config.batch_size = 16

# 或启用梯度累积
config.gradient_accumulation_steps = 4
```

### 2. 数据加载缓慢

**解决方案**：
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,      # 增加工作进程
    pin_memory=True,    # 固定内存
)
```

### 3. 模型不收敛

**解决方案**：
```python
config.learning_rate = 0.0001  # 减小学习率
config.scheduler = "CosineAnnealingLR"  # 使用调度器
```

---

## 下一步

- 查看 [API.md](./API.md) 了解详细的接口说明
- 查看 [README.md](./README.md) 了解架构和设计
- 查阅源代码获取更多自定义选项

