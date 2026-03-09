

# 斜拉索振动特性研究项目

## 项目概述

本项目专注于**斜拉索结构的振动特性研究**，通过收集和分析大规模振动与风速数据，识别极端振动现象（如涡激振动VIV），并利用机器学习算法对振动工况进行分类与预测。项目涵盖从原始数据处理、统计分析、可视化展示到机器学习建模的完整工作流。

### 核心研究方向

- **振动特性识别**: 检测和分析涡激振动（VIV）等极端振动现象
- **风-振耦合分析**: 研究风荷载与斜拉索振动之间的相互影响关系
- **数据驱动建模**: 使用传统机器学习算法对振动工况进行分类
- **大规模数据处理**: 支持数千小时的多传感器时序数据处理

---

## 项目结构

```
├── src/                              # 核心源代码
│   ├── data_processer/              # 数据处理模块
│   │   ├── io_unpacker.py           # 低级I/O和数据解包
│   │   ├── algorithms.py            # 算法库（VIV检测、涡流强度计算等）
│   │   ├── persistence_utils.py     # 数据持久化工具
│   │   ├── database_manager.py      # 数据库管理和缓存
│   │   ├── pipeline_orchestrator.py # 处理管道编排
│   │   ├── preprocess/              # 预处理模块（风/振数据I/O处理）
│   │   └── statistics/              # 统计分析模块
│   │       ├── vibration_io_process/# 振动数据处理工作流
│   │       └── wind_data_io_process/# 风数据处理工作流
│   ├── machine_learning_module/     # 机器学习模块
│   │   ├── svm/                     # SVM分类器
│   │   ├── naive_bayes/             # 朴素贝叶斯分类器
│   │   ├── cellular_automata/       # 元胞自动机
│   │   └── cellular_automata_with_bayes/ # CA+贝叶斯混合模型
│   ├── visualize_tools/             # 可视化工具
│   │   └── utils.py                 # PlotLib 绘图库
│   └── config/                      # 配置管理
│
├── docs/                            # 文档目录（详见下方导航）
│   ├── data_processer/              # 数据处理模块文档
│   ├── statistics/                  # 统计分析文档
│   ├── machine_learning_module/     # 机器学习文档
│   ├── visualize_tools/             # 可视化工具文档
│   └── development.md               # 开发日志
│
├── config/                          # 项目配置文件
├── results/                         # 结果输出目录
└── requirements.txt                 # Python依赖
```

---

## 📚 完整文档导航

### 1. 数据处理模块 (`data_processer`)

- **[完整文档](docs/data_processer/README.md)** - 模块架构、核心类和使用示例

#### 预处理子模块

- **振动数据I/O处理**
  - [README](docs/data_processer/preprocess/vib_data_io_process/README.md)
  - [工作流详解](docs/data_processer/preprocess/vib_data_io_process/vibration_io_process_workflow.md)
  - [RMS统计](docs/data_processer/preprocess/vib_data_io_process/rms_statistics.md)

- **风数据I/O处理**
  - [README](docs/data_processer/preprocess/wind_data_io_process/README.md)

- **数据集模块**
  - [README](docs/data_processer/datasets/README.md)
  - [快速开始](docs/data_processer/datasets/QUICKSTART.md)
  - [API文档](docs/data_processer/datasets/API.md)
  - [架构设计](docs/data_processer/datasets/ARCHITECTURE.md)
  - [索引与查询](docs/data_processer/datasets/INDEX.md)

### 2. 统计分析模块 (`statistics`)

#### 振动数据处理工作流

- **[README](docs/statistics/vibration_io_process/README.md)** - 三步骤处理流程、元数据格式
- [快速开始](docs/statistics/vibration_io_process/QUICKSTART.md)
- [API参考](docs/statistics/vibration_io_process/API.md)
- [高级主题](docs/statistics/vibration_io_process/ADVANCED.md)
- [重构说明](docs/statistics/vibration_io_process/REFACTOR.md)
- [模块索引](docs/statistics/vibration_io_process/INDEX.md)

#### 统计工作流整合

- **[README](docs/statistics/workflow/README.md)** - 振动与风数据整合工作流
- [快速开始](docs/statistics/workflow/QUICKSTART.md)
- [API参考](docs/statistics/workflow/API.md)
- [重构说明](docs/statistics/workflow/REFACTOR.md)

#### 其他统计工具

- [统计模块索引](docs/statistics/INDEX.md)

### 3. 机器学习模块 (`machine_learning_module`)

- **[完整文档](docs/machine_learning_module/README.md)** - 四种分类算法工作流

#### 支持的算法

- **SVM (支持向量机)** - rbf/linear/poly 核支持
- **朴素贝叶斯** - GaussianNB 实现
- **元胞自动机 (CA)** - Rule 30 进化 + 模板匹配
- **CA+朴素贝叶斯混合模型** - 特征提取 + 分类

#### 主要功能

- [快速开始](docs/machine_learning_module/QUICKSTART.md)
- 统一的工作流接口与便捷函数
- 配置驱动的训练与推理
- PyTorch DataLoader 无缝对接

### 4. 可视化工具模块 (`visualize_tools`)

- **[README](docs/visualize_tools/README.md)** - 模块概览与学习建议

#### PlotLib 绘图库

- **[完整参考](docs/visualize_tools/utils/README.md)** - 包含 fig/ax 机制详解
- [快速入门](docs/visualize_tools/utils/QUICKSTART.md)
- [API参考](docs/visualize_tools/utils/API.md)
- [设计决策](docs/visualize_tools/utils/REFACTOR.md)
- [模块索引](docs/visualize_tools/utils/INDEX.md)

### 5. 开发信息

- **[开发日志](docs/development.md)** - 项目修改历史与技术要点

---

## 🚀 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 poetry (如果项目使用)
poetry install
```

### 典型工作流示例

#### 步骤 1: 振动数据处理

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

# 获取处理后的振动数据元数据（包含极端振动标记）
metadata = run(use_cache=True)
print(f"处理了 {len(metadata)} 个数据文件")
```

#### 步骤 2: 风数据处理与整合

```python
from src.data_processer.statistics.workflow import run_wind_workflow

# 处理风数据，仅保留极端振动对应的时段
wind_metadata = run_wind_workflow(extreme_only=True)
```

#### 步骤 3: 特征提取与模型训练

```python
from src.machine_learning_module.svm.run import run_svm_workflow

# 准备数据加载器（详见数据集模块文档）
results = run_svm_workflow(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    infer_dataloader=infer_loader,
    infer_has_label=True
)
```

#### 步骤 4: 结果可视化

```python
from src.visualize_tools.utils import PlotLib

lib = PlotLib()
lib.plot(y=results['eval']['predictions'], title='Model Predictions')
lib.show()
```

---

## 📊 核心概念

### 振动数据处理流程

```
[Step 0] 获取所有振动文件
    ↓
[Step 1] 缺失率筛选（≤5%）
    ↓
[Step 2] RMS统计与极端识别（95%分位值）
    ↓
[元数据构建] 完整的时间、传感器、质量、极端标记信息
```

**关键参数**:
- 采样频率: **50 Hz**（振动数据）/ **1 Hz**（风数据）
- RMS时间窗口: **60秒**
- 极端阈值: **95%分位值**

### 机器学习分类

支持四种分类算法，统一的训练/推理接口：

| 算法 | 特点 | 适用场景 |
|------|------|---------|
| SVM | 经典判别模型 | 中等规模特征空间 |
| 朴素贝叶斯 | 轻量级概率模型 | 实时推理 |
| CA | 元胞自动机 | 模式匹配、模板学习 |
| CA+Bayes | 混合模型 | 特征提取 + 高阶分类 |

---

## 🔧 配置与自定义

各模块均支持灵活的配置：

- **数据源**: 修改 `src/config/` 下的配置文件
- **处理参数**: 缺失率阈值、时间窗口、采样率等
- **保存路径**: 模型、缓存、结果输出位置
- **缓存策略**: 启用/禁用缓存，强制重新计算

详见各模块的 **README** 或 **QUICKSTART** 文档。

---

## 📖 学习路径建议

### 初学者

1. 读本文档了解整体结构
2. 查看 [数据处理模块文档](docs/data_processer/README.md)
3. 运行 [数据处理快速开始](docs/statistics/vibration_io_process/QUICKSTART.md)
4. 查看 [机器学习模块文档](docs/machine_learning_module/README.md)

### 进阶用户

1. 深入阅读各模块 README（包含架构与设计理念）
2. 查阅 API 文档进行自定义开发
3. 阅读 REFACTOR/ADVANCED 文档理解高级特性
4. 查看 [开发日志](docs/development.md) 了解最新修改

### 贡献者

1. 阅读 [完整项目架构文档](docs/data_processer/README.md)
2. 查看 [开发日志](docs/development.md) 了解现有工作
3. 参考各模块的 REFACTOR 文档理解设计决策

---

## 🛠️ 开发工具

- **语言**: Python 3.7+
- **主要库**: numpy, scipy, pandas, scikit-learn, torch, matplotlib
- **配置管理**: YAML 配置文件
- **数据格式**: Parquet（支持列表数据）
- **并行处理**: multiprocessing, concurrent.futures

---

## 📝 项目特性

✅ **大规模数据处理**
- 支持数千小时多传感器时序数据
- 智能缓存与并行处理

✅ **完整的分析工作流**
- 从原始数据到机器学习模型的端到端流程
- 极端振动自动识别

✅ **灵活的机器学习框架**
- 多种算法选择
- 统一的工作流接口
- 易于扩展新模型

✅ **丰富的可视化工具**
- PlotLib 统一绘图库
- 支持多子图、交互式展示

✅ **完善的文档**
- 详细的模块级文档
- 快速开始指南与API参考
- 开发日志与技术要点

---

## 📞 项目信息

- **更新日期**: 2026年3月9日
- **Python版本**: 3.7+
- **依赖管理**: 见 `requirements.txt`

---

## 相关资源

- 📂 [完整文档目录](docs/)
- 🔍 [开发日志](docs/development.md) - 查看最新修改与技术决策
- 📊 [数据处理架构](docs/data_processer/README.md) - 理解核心数据流
- 🤖 [机器学习模块](docs/machine_learning_module/README.md) - 四种分类算法
- 📈 [可视化工具](docs/visualize_tools/README.md) - 绘图与展示

