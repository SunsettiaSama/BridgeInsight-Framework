# 拉索振动特性研究项目文档

欢迎来到拉索振动特性研究项目的文档中心！

## 📋 项目概述

本项目专注于桥梁拉索振动特性的研究，提供了完整的数据处理、深度学习识别、可视化分析等功能模块。

---

## 📂 文档目录结构

```
docs/
├── README.md                          # 本文档 - 项目文档总索引
├── annotation_system/                 # 标注系统文档
│   └── README.md
├── data_processer/                    # 数据处理模块文档
│   ├── README.md
│   ├── INDEX.md
│   ├── datasets/                      # 数据集子模块
│   │   ├── README.md
│   │   ├── INDEX.md
│   │   ├── API.md
│   │   ├── ARCHITECTURE.md
│   │   ├── QUICKSTART.md
│   │   └── StayCable_Vib2023/       # StayCable_Vib2023 数据集文档（新增）
│   │       └── README.md
│   └── preprocess/                    # 数据预处理子模块
│       ├── README.md
│       ├── INDEX.md
│       ├── metadata_parser.md
│       ├── vib_data_io_process/
│       ├── wind_data_io_process/
│       └── workflow/
├── deep_learning_module/              # 深度学习模块文档
│   ├── README.md
│   ├── INDEX.md
│   ├── API.md
│   ├── QUICKSTART.md
│   └── CONFIG_REFERENCE.md
├── identifier/                        # 识别器模块文档（新增）
│   └── deeplearning_methods/         # 深度学习识别方法文档
│       └── README.md
├── machine_learning_module/           # 机器学习模块文档
│   ├── README.md
│   └── QUICKSTART.md
├── train_eval/                        # 训练评估模块文档
│   ├── README.md
│   ├── API.md
│   └── QUICKSTART.md
└── visualize_tools/                   # 可视化工具文档
    ├── README.md
    ├── annotation/
    ├── annotation_system/
    └── utils/
```

---

## 🔗 快速链接

### 🌟 新增文档

| 模块 | 文档 | 说明 |
|------|------|------|
| **StayCable_Vib2023 数据集** | [data_processer/datasets/StayCable_Vib2023/README.md](./data_processer/datasets/StayCable_Vib2023/README.md) | 苏通大桥拉索振动2023数据集专用加载模块 |
| **深度学习识别方法** | [identifier/deeplearning_methods/README.md](./identifier/deeplearning_methods/README.md) | 基于深度学习的拉索振动分类识别模块 |

---

### 📚 核心模块文档

| 模块 | 文档 | 说明 |
|------|------|------|
| **数据处理模块** | [data_processer/README.md](./data_processer/README.md) | 数据加载、预处理、数据集管理 |
| **数据集模块** | [data_processer/datasets/README.md](./data_processer/datasets/README.md) | 数据集加载与管理框架 |
| **深度学习模块** | [deep_learning_module/README.md](./deep_learning_module/README.md) | MLP/CNN/RNN/LSTM 模型框架 |
| **识别器模块** | [identifier/deeplearning_methods/README.md](./identifier/deeplearning_methods/README.md) | 深度学习振动识别 |
| **训练评估模块** | [train_eval/README.md](./train_eval/README.md) | 模型训练与评估 |
| **可视化工具** | [visualize_tools/README.md](./visualize_tools/README.md) | 数据可视化与标注工具 |
| **标注系统** | [annotation_system/README.md](./annotation_system/README.md) | 数据标注系统 |

---

## 🚀 快速开始

### 典型工作流

1. **数据准备** → [数据预处理文档](./data_processer/preprocess/README.md)
2. **数据集加载** → [StayCable_Vib2023 数据集文档](./data_processer/datasets/StayCable_Vib2023/README.md)
3. **模型训练** → [深度学习模块文档](./deep_learning_module/README.md)
4. **振动识别** → [深度学习识别方法文档](./identifier/deeplearning_methods/README.md)
5. **结果可视化** → [可视化工具文档](./visualize_tools/README.md)

### 代码示例

```python
# 1. 加载 StayCable_Vib2023 数据集
from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import StayCableVib2023Config
from src.data_processer.datasets.StayCable_Vib2023 import StayCableVib2023Dataset

config = StayCableVib2023Config(...)
dataset = StayCableVib2023Dataset(config)

# 2. 使用深度学习识别
from src.identifier.deeplearning_methods import DLVibrationIdentifier, FullDatasetRunner

identifier = DLVibrationIdentifier.from_checkpoint(...)
runner = FullDatasetRunner(identifier)
predictions = runner.run(dataset)

# 3. 应用识别结果
dataset.apply_predictions(predictions)
```

---

## 📖 文档导航

### 按功能分类

| 类别 | 相关文档 |
|------|----------|
| **数据处理** | [data_processer/README.md](./data_processer/README.md), [data_processer/preprocess/README.md](./data_processer/preprocess/README.md) |
| **数据集** | [data_processer/datasets/README.md](./data_processer/datasets/README.md), [data_processer/datasets/StayCable_Vib2023/README.md](./data_processer/datasets/StayCable_Vib2023/README.md) |
| **深度学习** | [deep_learning_module/README.md](./deep_learning_module/README.md), [identifier/deeplearning_methods/README.md](./identifier/deeplearning_methods/README.md) |
| **机器学习** | [machine_learning_module/README.md](./machine_learning_module/README.md) |
| **训练评估** | [train_eval/README.md](./train_eval/README.md) |
| **可视化** | [visualize_tools/README.md](./visualize_tools/README.md) |

---

## 📞 维护信息

- **项目**: 拉索振动特性研究
- **最后更新**: 2026年4月2日
- **文档版本**: 1.0
