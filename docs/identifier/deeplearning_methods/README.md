# 深度学习振动识别模块

## 📋 模块概述

`src/identifier/deeplearning_methods/` 是基于深度学习的拉索振动分类识别模块。该模块提供了与数据集完全解耦的识别器，支持 MLP / CNN / RNN / LSTM 四种模型架构，并提供全量数据集识别的高效运行器。

### 核心特性

- **解耦设计**：识别器与数据集完全解耦，仅接收信号张量，返回类别预测
- **多模型支持**：支持 MLP、CNN、RNN、LSTM 四种模型架构
- **自动输入适配**：内部按 model_type 自动变换输入形状
- **多进程加速**：全量识别支持 DataLoader 多进程加载
- **结果持久化**：识别结果可保存为 JSON 缓存，支持按文件索引格式
- **Checkpoint 加载**：从训练好的 checkpoint 便捷构建识别器

---

## 📂 目录结构

```
src/identifier/deeplearning_methods/
├── __init__.py              # 模块导出
├── dl_identifier.py         # 深度学习识别器（DLVibrationIdentifier）
├── full_dataset_runner.py   # 全量数据集识别运行器（FullDatasetRunner）
└── run.py                   # 运行脚本（预留）
```

---

## 🏗 架构设计

### 分层架构

```
[用户代码]
    ↓
[FullDatasetRunner (全量识别)]
    ↓
[DLVibrationIdentifier (单批次识别)]
    ↓
[PyTorch 模型]
```

### 核心类关系

```
DLVibrationIdentifier
    ├─持有→ nn.Module (PyTorch模型)
    └─提供→ predict_batch() 批量预测接口

FullDatasetRunner
    ├─持有→ DLVibrationIdentifier
    ├─使用→ _InplaneWindowDataset (轻量推理数据集)
    └─提供→ run() 全量识别接口
```

---

## 📖 核心类详解

### 1. DLVibrationIdentifier

**文件**: `src/identifier/deeplearning_methods/dl_identifier.py`

#### 设计说明

深度学习振动分类识别器。与数据集完全解耦：仅接收信号张量，返回类别预测。

#### 输入约定

`predict_batch` 接收形状 `(B, window_size, 1)` 的 float32 张量，内部按 model_type 自动变换为各模型期望的输入格式。

#### 标签说明

| 类别ID | 标签名 | 说明 |
|--------|--------|------|
| 0 | Normal | 正常振动 |
| 1 | VIV | 涡激振动（Vortex-Induced Vibration） |
| 2 | RWIV | 随机风致振动（Random Wind-Induced Vibration） |
| 3 | Transition | 过渡状态 |

#### 主要公开接口

| 方法 | 说明 |
|------|------|
| `__init__(model, model_type, num_classes, device)` | 初始化识别器 |
| `from_checkpoint(checkpoint_path, model_type, model_config_path, ...)` | 从checkpoint构建识别器（工厂方法） |
| `predict_batch(x)` | 批量预测，返回类别ID数组 |

#### 输入形状变换

| 模型类型 | 输入变换 |
|----------|----------|
| CNN | `(B, window_size, 1)` → `(B, 1, window_size)` |
| RNN/LSTM | 保持 `(B, window_size, 1)` |
| MLP | `(B, window_size, 1)` → `(B, window_size)` |

---

### 2. FullDatasetRunner

**文件**: `src/identifier/deeplearning_methods/full_dataset_runner.py`

#### 设计说明

对 StayCableVib2023Dataset 的全量样本执行 DL 识别。

#### 设计原则

- 与数据集解耦：通过 `dataset._samples` / `dataset.config` 访问必要信息，不在 FullDatasetRunner 内部存储数据集引用
- 多进程加速：DataLoader 的 `num_workers` 并行加载 VIC 文件，`batch_size` 控制单次 GPU 推理量
- 结果格式：`run()` 返回 `{sample_idx: predicted_label}`；`to_file_indexed()` 将其转换为按文件组织的窗口预测列表
- 去噪透传：`run()` 从 `dataset.config.enable_denoise` / `dataset.config.denoise_freq_threshold` 读取去噪配置，透传给 `_InplaneWindowDataset`

#### 主要公开接口

| 方法 | 说明 |
|------|------|
| `__init__(identifier, batch_size, num_workers)` | 初始化运行器 |
| `run(dataset)` | 对数据集所有样本执行推理，返回 `{sample_idx: predicted_label}` |
| `to_file_indexed(predictions, dataset)` | 转换为按文件组织的窗口预测列表 |
| `save_predictions(path, predictions, dataset, model_info)` | 保存识别结果到 JSON |
| `load_predictions(path)` | 从 JSON 加载识别结果 |

---

### 3. _InplaneWindowDataset

**文件**: `src/identifier/deeplearning_methods/full_dataset_runner.py`

#### 设计说明

仅加载面内振动窗口，用于 DL 推理阶段。与 StayCableVib2023Dataset 解耦：只持有 `_SampleRecord` 列表和基本参数，不依赖完整数据集实例，从而支持 DataLoader `num_workers > 0`。

构造参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `records` | `List[_SampleRecord]` | 经过预验证的样本记录列表 |
| `window_size` | `int` | 单窗口采样点数 |
| `enable_denoise` | `bool` | 是否启用分层去噪 |
| `original_indices` | `Optional[List[int]]` | 原始样本索引，用于过滤后映射回 dataset._samples |
| `freq_threshold` | `Optional[float]` | 分层去噪频率阈值（Hz），透传给 `VICWindowExtractor` |

---

### 4. VICWindowExtractor（去噪核心）

**文件**: `src/data_processer/preprocess/get_data_vib.py`

#### 分层去噪策略

`_apply_denoise()` 按以下优先级决定是否对窗口去噪：

```
1. STRATIFIED_DENOISE_ENABLED == False
   → 不分层，全部去噪

2. 元数据中存在 extreme_freq_indices（预计算极端主频索引）
   → 命中 → 跳过去噪（保留极端窗口原始特征）
   → 未命中 → 应用去噪

3. extreme_freq_indices 缺失 且 freq_threshold 已设置（硬编码阈值）
   → 实时 FFT 计算窗口主频
   → 主频 > freq_threshold → 跳过去噪
   → 主频 ≤ freq_threshold → 应用去噪

4. 其余情况（无预计算索引、无硬编码阈值）
   → 全部应用去噪
```

#### 频率阈值来源

| 来源 | 配置方式 | 优先级 |
|------|----------|--------|
| YAML 硬编码 | `denoise_freq_threshold: 2.5`（正数，单位 Hz） | **高** |
| 统计文件自动读取 | `denoise_freq_threshold: null`（默认） | 低 |

统计文件路径由 `vib_metadata2data_config.py` 中的 `DOMINANT_FREQ_RESULT_PATH` 指定，阈值取所有样本主频的 **95% 分位数**（`DOMINANT_FREQ_PERCENTILE = 0.95`）。

---

## 🚀 使用指南

### 基础使用：从 Checkpoint 构建识别器

```python
from src.identifier.deeplearning_methods import DLVibrationIdentifier

# 从 checkpoint 构建识别器
identifier = DLVibrationIdentifier.from_checkpoint(
    checkpoint_path="./checkpoints/cnn_model.pth",
    model_type="cnn",
    model_config_path="./configs/cnn_config.yaml",
    num_classes=4,
    device="cuda",  # 或 "cpu"，None 自动选择
)
```

### 单批次预测

```python
import torch

# 构造输入：形状 (B, window_size, 1)
x = torch.randn(32, 3600, 1)  # batch_size=32, window_size=3600

# 批量预测
predictions = identifier.predict_batch(x)
# predictions: np.ndarray, shape (32,), dtype int32
print(predictions)  # [0, 1, 0, 2, ...]
```

### 全量数据集识别

```python
from src.identifier.deeplearning_methods import FullDatasetRunner
from src.data_processer.datasets.StayCable_Vib2023 import StayCableVib2023Dataset

# 1. 加载数据集
dataset = StayCableVib2023Dataset(config)

# 2. 创建识别器和运行器
identifier = DLVibrationIdentifier.from_checkpoint(...)
runner = FullDatasetRunner(
    identifier,
    batch_size=256,
    num_workers=4,
)

# 3. 运行全量识别
predictions = runner.run(dataset)
# predictions: {sample_idx: predicted_label}
print(f"识别完成，共 {len(predictions)} 条结果")
```

### 使用硬编码频率阈值的分层去噪

在 YAML 配置中直接指定阈值（单位 Hz），跳过统计文件读取：

```yaml
# config/train/datasets/total_staycable_vib.yaml
enable_denoise: true
denoise_freq_threshold: 2.5   # 主频 > 2.5 Hz 的窗口不去噪
```

等效 Python 写法：

```python
from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import StayCableVib2023Config

config = StayCableVib2023Config(
    ...,
    enable_denoise=True,
    denoise_freq_threshold=2.5,  # Hz；None 则自动从统计文件读取 95% 分位数
)
dataset = StayCableVib2023Dataset(config)
```

### 结果保存与加载

```python
# 保存识别结果
FullDatasetRunner.save_predictions(
    path="./results/predictions.json",
    predictions=predictions,
    dataset=dataset,
    model_info="CNN_v1_20260402",
)

# 加载识别结果
loaded_predictions = FullDatasetRunner.load_predictions("./results/predictions.json")

# 应用到数据集
dataset.apply_predictions(loaded_predictions)
```

### 转换为按文件索引的格式

```python
# 转换为按文件组织的窗口预测列表
file_indexed = FullDatasetRunner.to_file_indexed(predictions, dataset)

# file_indexed 结构:
# {
#   "ST-VIC-C34-101-01_9_1_0": [0, 0, 1, 0, ...],
#   ...
# }
# 列表下标 = window_idx，值 = 预测类别（-1 表示该窗口无预测结果）
```

### 完整工作流示例

```python
from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import StayCableVib2023Config
from src.data_processer.datasets.StayCable_Vib2023 import StayCableVib2023Dataset
from src.identifier.deeplearning_methods import DLVibrationIdentifier, FullDatasetRunner

# Step 1: 配置并加载数据集
config = StayCableVib2023Config(
    vib_metadata_path="./data/vib_metadata.json",
    wind_metadata_path="./data/wind_metadata.json",
    cable_pairs=[("ST-VIC-C34-101-01", "ST-VIC-C34-101-02")],
    window_size=3600,
    time_ordered=True,
    use_cache=True,
)
dataset = StayCableVib2023Dataset(config)

# Step 2: 创建识别器
identifier = DLVibrationIdentifier.from_checkpoint(
    checkpoint_path="./checkpoints/lstm_model.pth",
    model_type="lstm",
    model_config_path="./configs/lstm_config.yaml",
    device="cuda",
)

# Step 3: 运行全量识别
runner = FullDatasetRunner(identifier, batch_size=256, num_workers=4)
predictions = runner.run(dataset)

# Step 4: 保存结果
runner.save_predictions(
    "./results/full_predictions.json",
    predictions,
    dataset,
    model_info="LSTM_v1",
)

# Step 5: 应用结果到数据集
dataset.apply_predictions(predictions)

# Step 6: 访问带标签的样本
sample = dataset[0]
print(f"预测标签: {sample['metadata']['dl_label']}")
```

---

## 📊 识别结果 JSON 格式

```json
{
  "metadata": {
    "created_at": "2026-04-02 14:30:00",
    "num_samples": 12345,
    "num_classes": 4,
    "model_info": "CNN_v1"
  },
  "predictions": {
    "0": 1,
    "1": 0,
    "2": 2,
    ...
  },
  "by_file": {
    "ST-VIC-C34-101-01_9_1_0": [0, 0, 1, 0, ...],
    ...
  }
}
```

- `predictions`：平铺格式，键为样本索引字符串，值为预测类别
- `by_file`：按（传感器, 月, 日, 时）分组的窗口预测列表，列表下标即 window_idx

---

## 🔗 相关链接

- [StayCable_Vib2023 数据集文档](../../data_processer/datasets/StayCable_Vib2023/README.md)
- [深度学习模型模块文档](../../deep_learning_module/README.md)
- [识别器模块总览](../README.md)

---

## 📞 维护信息

- **最后更新**: 2026年4月2日
- **模块维护者**: Deep Learning Team
- **依赖库**: torch, numpy, yaml, tqdm
