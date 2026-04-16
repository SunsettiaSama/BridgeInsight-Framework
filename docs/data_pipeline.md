# 全量数据处理流水线（Data Pipeline）

> 入口脚本：`src/data_pipeline.py`  
> 默认配置：`config/data_pipeline.yaml`

---

## 目录

1. [概览](#1-概览)
2. [快速开始](#2-快速开始)
3. [数据流总览](#3-数据流总览)
4. [步骤详解](#4-步骤详解)
   - [步骤1 数据预处理](#步骤1-数据预处理)
   - [步骤2 全量识别](#步骤2-全量识别)
   - [步骤3 特征计算与归档](#步骤3-特征计算与归档)
5. [去噪策略总览](#5-去噪策略总览)
6. [步骤开关与依赖关系](#6-步骤开关与依赖关系)
7. [输出文件结构](#7-输出文件结构)
8. [内存管理](#8-内存管理)
9. [文档索引](#9-文档索引)
10. [配置文件索引](#10-配置文件索引)

---

## 1. 概览

本流水线将原始拉索振动与风数据，经过三个串联步骤，最终产出按振动类别归档的多维特征数据集：

| 步骤 | 名称 | 核心模块 | 输入 | 输出 |
|------|------|----------|------|------|
| 1 | 数据预处理 | `data_processer.preprocess` | 原始 `.VIC` / `.UAN` 文件 | 振动元数据 JSON + 风元数据 JSON |
| 2 | 全量识别 | `identifier.deeplearning_methods` | 元数据 JSON + VIC 文件 | 预测结果 JSON |
| 3 | 特征计算与归档 | `identifier.feature_analysis` | 预测结果 JSON + 风元数据 JSON | 按类别归档的特征 JSON |

---

## 2. 快速开始

```bash
# 全量执行三个步骤（使用默认配置）
python src/data_pipeline.py

# 指定配置文件
python src/data_pipeline.py --config config/data_pipeline.yaml

# 跳过步骤1和2，仅跑特征计算（需已有识别结果）
# 在 config/data_pipeline.yaml 中：
#   steps.preprocess:     false
#   steps.identification: false
#   steps.feature_analysis: true
python src/data_pipeline.py
```

---

## 3. 数据流总览

```
原始数据文件
├── VIC 振动文件   （*.VIC 二进制）
└── UAN 风速文件   （*.UAN 文本）
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  步骤1  数据预处理                                    │
│  src/data_processer/preprocess/                      │
│  ┌────────────────┐    ┌───────────────────────┐    │
│  │ 振动 workflow  │    │ 风 workflow            │    │
│  │ - 缺失率筛选   │    │ - 时间戳对齐           │    │
│  │ - RMS 统计     │    │ - 风速/方向/攻角提取   │    │
│  │ - 主频统计     │    └───────────────────────┘    │
│  └────────────────┘                                  │
└─────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
vibration_metadata JSON          wind_metadata JSON
(97,390 条窗口元数据)            (20,386 条时间对齐记录)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  步骤2  全量识别                                      │
│  src/identifier/deeplearning_methods/                │
│  ┌───────────────────┐   ┌──────────────────────┐  │
│  │ StayCableVib2023  │   │ DLVibrationIdentifier│  │
│  │ Dataset           │──▶│ (ResCNN, CUDA)       │  │
│  │ 1,453,059 样本    │   │ batch_size=256       │  │
│  └───────────────────┘   └──────────────────────┘  │
│         ↑                                           │
│  VICWindowExtractor（读 VIC 文件 + 分层去噪，见§5）  │
└─────────────────────────────────────────────────────┘
         │
         ▼
res_cnn_full_dataset_{timestamp}.json
{ predictions, sample_metadata, by_file }
         │                    │
         │              wind_metadata JSON
         ▼                    ▼
┌─────────────────────────────────────────────────────┐
│  步骤3  特征计算与归档                                │
│  src/identifier/feature_analysis/                    │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ PSD 主导模态 │  │ 时域统计 │  │ 面内外耦合   │  │
│  │ 谱熵/带宽    │  │ RMS/峭度 │  │ 互相关/相干  │  │
│  └──────────────┘  └──────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │ 风统计量     │  │ 折减风速（可选）              │ │
│  └──────────────┘  └──────────────────────────────┘ │
│  多进程并行（n_workers 可配置）                       │
└─────────────────────────────────────────────────────┘
         │
         ▼
results/enriched_stats/
├── class_0_随机振动/
├── class_1_涡激共振/
├── class_2_风雨振/
└── class_3_其他振动/
```

---

## 4. 步骤详解

### 步骤1 数据预处理

**入口函数**：`run_preprocess()` in `src/data_pipeline.py`

**子步骤**

| 子步骤 | 模块 | 功能 |
|--------|------|------|
| 1/2 振动预处理 | `src/data_processer/preprocess/vibration_io_process/workflow.py` | 缺失率筛选 → 逐窗口 RMS 统计 → 主频分析（Welch PSD）→ 95% 分位数阈值计算 |
| 2/2 风预处理 | `src/data_processer/preprocess/wind_data_io_process/workflow.py` | 与振动元数据时间戳对齐，提取风速 / 风向 / 风攻角 |

**配置**：`config/data_processer/preprocess.yaml`

**缓存机制**：两个 workflow 均支持 `use_cache=True`，结果写磁盘后可复用，避免重复计算。

**产出字段（振动元数据 JSON 单条记录）**

```json
{
  "sensor_id": "ST-VIC-C34-101-01",
  "file_path": "...",
  "missing_rate": 0.005,
  "window_count": 15,
  "dominant_freq_per_window": [0.312, 0.318, ...],
  "rms_per_window": [0.042, 0.038, ...],
  "freq_p95": 1.25,
  "extreme_rms_indices": [...],
  "extreme_freq_indices": [...]
}
```

→ 详细文档：[`docs/data_processer/preprocess/`](data_processer/preprocess/README.md)

---

### 步骤2 全量识别

**入口函数**：`run_identification()` in `src/data_pipeline.py`

**执行链**

```
StayCableVib2023Dataset（加载元数据索引）
        ↓
FullDatasetRunner._validate_records()   ← 预验证文件完整性（tqdm 进度条）
        ↓
_InplaneWindowDataset（按窗口懒加载）
        ↓
DataLoader（num_workers=4, pin_memory=True）
        ↓
DLVibrationIdentifier.predict_batch()   ← ResCNN, CUDA
        ↓
FullDatasetRunner.save_predictions()
```

**核心组件**

| 组件 | 文件 | 职责 |
|------|------|------|
| `DLVibrationIdentifier` | `src/identifier/deeplearning_methods/dl_identifier.py` | 从 checkpoint 加载模型，执行批量推理 |
| `FullDatasetRunner` | `src/identifier/deeplearning_methods/full_dataset_runner.py` | 编排 DataLoader 推理、预验证、结果保存 |
| `VICWindowExtractor` | `src/data_processer/preprocess/get_data_vib.py` | 读取 VIC 窗口信号，执行分层去噪（详见 [§5 去噪策略总览](#5-去噪策略总览)） |
| `StayCableVib2023Dataset` | `src/data_processer/datasets/StayCable_Vib2023/` | 加载元数据索引、提供样本迭代 |

**配置文件**

- 数据集配置：`config/identifier/dl_identifier/total_staycable_vib.yaml`
- 模型配置：`config/train/models/res_cnn.yaml`
- checkpoint：`results/training_result/deep_learning_module/res_cnn/checkpoints/ResCNN_20260402_111429/best_checkpoint.pth`

**产出 JSON 结构**

```json
{
  "metadata": {
    "created_at": "2026-04-15 14:44:47",
    "num_samples": 1453059,
    "num_classes": 4,
    "model_info": "ResCNN (...)",
    "dataset_fingerprint_hash": "62127b2573ae92e9"
  },
  "predictions": { "0": 0, "1": 1, ... },
  "sample_metadata": {
    "0": {
      "inplane_sensor_id": "ST-VIC-C34-101-01",
      "timestamp": [9, 1, 0],
      "window_idx": 0,
      "missing_rate_in": 0.005,
      "has_wind": true
    }
  },
  "by_file": {
    "ST-VIC-C34-101-01_9_1_0": [0, 0, 1, 0, ...]
  }
}
```

→ 详细文档：[`docs/identifier/deeplearning_methods/README.md`](identifier/deeplearning_methods/README.md)

---

### 步骤3 特征计算与归档

**入口函数**：`run_feature_analysis()` in `src/data_pipeline.py`

**计算内容（可按开关独立控制）**

| 开关字段 | 计算内容 | 子模块 |
|----------|----------|--------|
| `enable_psd_modes` | Welch PSD 前 N 阶主导模态（频率 + 功率） | `_modal.py` |
| `enable_spectral_features` | 谱熵 / 谱带宽 / 主频能量占比 | `_modal.py` |
| `enable_time_stats` | RMS / 峭度 / 偏度 / 波峰因子 / 过零率 | `_signal.py` |
| `enable_cross_coupling` | 面内外互相关 / 轨迹椭圆率 / 相干性 / 相位差 | `_coupling.py` |
| `enable_wind_stats` | 风速 / 风向 / 紊流度统计量 | `_wind.py` |
| `enable_reduced_velocity` | 折减风速 Vr = U / (f₁·D)（需配置拉索外径） | `_wind.py` |

**并行策略**：多进程 `Pool`（`n_workers` 可在配置文件中设定），每个进程独立处理单个样本。

**配置文件**：`config/identifier/feature_analysis/default.yaml`

**产出目录结构**

```
results/enriched_stats/
├── class_0/
│   ├── all.json                    # 全传感器合并
│   └── ST-VIC-C34-101-01.json     # 按传感器分割（split_by_sensor=true 时）
├── class_1/
├── class_2/
└── class_3/
```

→ 详细文档：[`docs/identifier/process_full_data/README.md`](identifier/process_full_data/README.md)

---

## 5. 去噪策略总览

全链路中**预处理（步骤1）不做信号去噪**，仅产出供后续使用的主频统计元数据；
其余三个使用 VIC 信号的环节均启用分层去噪。

### 各环节去噪状态

| 环节 | 去噪开关 | 实际行为 | 配置来源 |
|------|----------|----------|----------|
| **步骤1** 预处理 | 无 | 不处理信号，仅计算元数据（`dominant_freq_per_window`、`freq_p95`） | — |
| **模型训练** | `enable_denoise: true` | 分层去噪（有预处理主频时直接比对，否则实时 FFT fallback） | `config/train/datasets/annotation_dataset.yaml` |
| **步骤2** 推理 | `enable_denoise: true` | 分层去噪（元数据携带 `dominant_freq_per_window`，优先用预处理值） | `config/identifier/dl_identifier/total_staycable_vib.yaml` |
| **步骤3** 特征计算 | `enable_denoise: true` | 分层去噪（通过 `VICWindowExtractor`，实时 FFT fallback 计算主频） | `config/identifier/feature_analysis/default.yaml` |

### 步骤2 分层去噪策略（推理阶段）

代码入口：`VICWindowExtractor._apply_denoise()`（`src/data_processer/preprocess/get_data_vib.py`）

全局开关 `STRATIFIED_DENOISE_ENABLED = True`（`src/config/data_processer/preprocess/vib_metadata2data_config.py`），分层逻辑如下：

```
每个 VIC 窗口
    ↓
从元数据 dominant_freq_per_window[window_index] 取预处理主频
    ↓（若缺失则实时 FFT 计算作为 fallback）
与阈值 freq_p95 比对
    ├── dominant_freq > freq_p95  →  跳过去噪，保留原始信号（极端高频窗口）
    └── dominant_freq ≤ freq_p95  →  小波去噪（低频 / 常规窗口）
```

**阈值来源**（优先级由高到低）：

1. YAML 中 `denoise_freq_threshold` 指定的固定值（当前为 `null`，不启用）
2. 从 `dominant_freq_statistics_result.json` 读取全量数据主频的第 95 百分位数
3. 以上均不可用 → 退化为全窗口去噪，并打印 `logger.warning`

**小波去噪参数**（来自 `vib_metadata2data_config.py`）：

| 参数 | 值 |
|------|----|
| 小波基 | `db4` |
| 分解层数 | 自适应（`None` → 由信号长度决定） |
| 阈值类型 | `soft`（软阈值） |
| 阈值方法 | `sqtwolog`（通用阈值） |
| 分层阈值 | `True`（各分解层独立计算阈值） |

### 训练—推理信号分布说明

训练、推理、特征计算三个阶段均启用分层去噪，信号分布一致。
步骤2（推理）因元数据中含 `dominant_freq_per_window`，可直接复用预处理时计算的主频，精度最高；
训练与步骤3 在无预存主频时自动使用实时 FFT 计算主频进行分层判断，行为等价。

---

## 6. 步骤开关与依赖关系

在 `config/data_pipeline.yaml` 的 `steps` 节点中控制各步骤是否执行：

```yaml
steps:
  preprocess:       true   # 步骤1
  identification:   true   # 步骤2
  feature_analysis: true   # 步骤3
```

**依赖矩阵**

| 步骤 | 依赖 | 跳过时的替代方案 |
|------|------|-----------------|
| 步骤1 | 无 | — |
| 步骤2 | 步骤1产出的元数据文件（已存在磁盘即可） | 可单独跳过步骤1，元数据文件已有时直接跑步骤2 |
| 步骤3 | 步骤2产出的预测 JSON + 步骤1产出的风元数据 | 可在 `feature_analysis.result_path` 和 `feature_analysis.wind_metadata_path` 中手动指定文件路径 |

**步骤间数据传递方式**

- 步骤1 → 步骤3：通过函数返回值传递 `wind_metadata_path`（字符串路径）
- 步骤2 → 步骤3：通过函数返回值传递 `result_path`（字符串路径）
- 若步骤被跳过，对应路径为 `None`，步骤3 将自动检测磁盘上已有文件或读取 `preprocess.yaml` 中的路径

---

## 7. 输出文件结构

```
results/
├── preprocessed_full_data/
│   ├── vibration_metadata/
│   │   └── files_after_lackness_filter.json    ← 步骤1 产出（振动元数据）
│   └── wind_metadata/
│       └── files_after_timestamp_align.json    ← 步骤1 产出（风元数据）
│
├── full_vib_metadata/
│   └── staycable_vib2023_index_cache.json      ← 步骤2 数据集索引缓存
│
├── identification_result/
│   └── res_cnn_full_dataset_{timestamp}.json   ← 步骤2 产出（预测结果）
│
└── enriched_stats/
    ├── class_0/                                ← 步骤3 产出（按类别特征）
    ├── class_1/
    ├── class_2/
    └── class_3/
```

---

## 8. 内存管理

各步骤在完成后显式释放大对象，防止步骤间内存堆积：

| 步骤 | 释放对象 | 方式 |
|------|----------|------|
| 步骤1结束 | `vib_metadata`（97k 条）、`wind_metadata`（20k 条） | `del` + `gc.collect()` |
| 步骤2推理循环内 | 每批 `signals` / `preds` tensor | 每 batch `del` |
| 步骤2推理结束 | DataLoader worker 进程、`pred_ds`、`valid_records` | `del loader` 触发 worker 退出 |
| 步骤2函数返回前 | `predictions`（145万条 dict）、`dataset`、`identifier`（GPU 模型） | `del` + `model.cpu()` + `torch.cuda.empty_cache()` |

---

## 9. 文档索引

| 模块 / 主题 | 文档路径 |
|-------------|---------|
| 数据预处理总览 | [`docs/data_processer/README.md`](data_processer/README.md) |
| 数据预处理索引 | [`docs/data_processer/INDEX.md`](data_processer/INDEX.md) |
| 振动预处理 workflow | [`docs/data_processer/preprocess/workflow/README.md`](data_processer/preprocess/workflow/README.md) |
| 振动预处理 API | [`docs/data_processer/preprocess/workflow/API.md`](data_processer/preprocess/workflow/API.md) |
| 振动 IO 处理流程 | [`docs/data_processer/preprocess/vib_data_io_process/vibration_io_process_workflow.md`](data_processer/preprocess/vib_data_io_process/vibration_io_process_workflow.md) |
| RMS 统计 | [`docs/data_processer/preprocess/vib_data_io_process/rms_statistics.md`](data_processer/preprocess/vib_data_io_process/rms_statistics.md) |
| 风数据预处理 | [`docs/data_processer/preprocess/wind_data_io_process/README.md`](data_processer/preprocess/wind_data_io_process/README.md) |
| 元数据字段说明 | [`docs/data_processer/preprocess/metadata_parser.md`](data_processer/preprocess/metadata_parser.md) |
| StayCable_Vib2023 数据集 | [`docs/data_processer/datasets/StayCable_Vib2023/README.md`](data_processer/datasets/StayCable_Vib2023/README.md) |
| 数据集 API | [`docs/data_processer/datasets/API.md`](data_processer/datasets/API.md) |
| 数据集架构设计 | [`docs/data_processer/datasets/ARCHITECTURE.md`](data_processer/datasets/ARCHITECTURE.md) |
| 深度学习识别模块 | [`docs/identifier/deeplearning_methods/README.md`](identifier/deeplearning_methods/README.md) |
| 特征计算模块 | [`docs/identifier/process_full_data/README.md`](identifier/process_full_data/README.md) |
| 深度学习模型文档 | [`docs/deep_learning_module/README.md`](deep_learning_module/README.md) |
| 训练评估文档 | [`docs/train_eval/README.md`](train_eval/README.md) |
| 统计拟合模块 | [`docs/statistics/README.md`](statistics/README.md) |

---

## 10. 配置文件索引

| 配置文件 | 作用 | 关联步骤 |
|----------|------|----------|
| `config/data_pipeline.yaml` | 主流水线总开关与各步骤参数 | 全部 |
| `config/data_processer/preprocess.yaml` | 预处理路径、缺失率阈值、缓存策略 | 步骤1 |
| `config/identifier/dl_identifier/total_staycable_vib.yaml` | 数据集配置（元数据路径、去噪策略、窗口参数） | 步骤2 |
| `config/train/models/res_cnn.yaml` | ResCNN 模型结构参数 | 步骤2 |
| `config/identifier/feature_analysis/default.yaml` | 特征计算开关、PSD 参数、并行进程数 | 步骤3 |
| `config/cables/cable_params.yaml` | 拉索外径（折减风速计算所需） | 步骤3（可选） |
