# 数据预处理模块 (Preprocess) 完整文档

## 模块概述

`src/data_processer/preprocess/` 是本项目的**核心数据预处理模块**，负责从原始数据文件中提取、解析、清理和组织振动及风数据，为后续的统计分析、可视化和机器学习任务提供统一的数据接口。

### 核心职责

1. **元数据管理**: 从原始数据文件中提取和解析元数据
2. **数据解析**: 将二进制 VIC/VWD 文件解析为结构化数据
3. **数据清理**: 处理缺失值、异常值和错误数据
4. **数据对齐**: 同步振动和风数据的时间窗口
5. **数据导出**: 生成PyTorch兼容的训练数据集

---

## 模块架构

```
src/data_processer/preprocess/
│
├── workflow.py                           # 集成工作流（主入口）
│
├── metadata_parser.py                    # 元数据和数据集解析器 [新增]
│   ├── MetadataParser                   # 基础元数据解析器
│   └── AnnotatedDatasetParser            # 标注数据集解析器
│
├── vib_metadata2data.py                  # 振动数据解析器
│   ├── parse_single_metadata_to_vibration_data()
│   └── [振动数据处理逻辑]
│
├── wind_metadata2data.py                 # 风数据解析器
│   ├── parse_single_metadata_to_wind_data()
│   └── [风数据处理逻辑]
│
├── vibration_io_process/                 # 振动数据IO处理子模块
│   ├── workflow.py                       # 振动数据工作流
│   ├── step1_metadata_extract.py         # 第1步：元数据提取
│   ├── step2_vib_trim_segmentation.py    # 第2步：数据裁剪和切分
│   └── step3_dominant_freq_statistics.py # 第3步：频率统计
│
└── wind_data_io_process/                 # 风数据IO处理子模块
    ├── workflow.py                       # 风数据工作流
    ├── step1_metadata_extract.py         # 第1步：元数据提取
    ├── step2_filter_pairing.py           # 第2步：过滤和配对
    └── step3_out_of_range.py             # 第3步：超限检测
```

---

## 核心组件详解

### 1. `metadata_parser.py` - 元数据和数据集解析器 [新增功能]

**用途**: 统一管理振动和风数据元数据，支持队列式批量解析和标注数据集生成

#### 核心类

##### `MetadataParser` - 基础元数据解析器

```python
from src.data_processer.preprocess.metadata_parser import MetadataParser

# 创建解析器（单进程模式，PyTorch DataLoader兼容）
parser = MetadataParser(
    vibration_metadata=vib_meta_list,
    wind_metadata=wind_meta_list,
    num_workers=None,                  # None=单进程, 0=自动, >0=指定进程数
    validate_metadata=False             # 是否检查文件有效性
)

# 解析数据（统一接口）
data = parser.parse_data(
    mode='vibration',                  # 'vibration'/'wind'/'both'
    batch_size=32,                     # 批处理大小
    enable_extreme_window=False,
    enable_denoise=False,
    show_progress=True
)
```

**主要特性**:
- ✅ 元数据自动验证和过滤（可关闭）
- ✅ 真正的批处理逻辑（控制内存占用）
- ✅ 多进程/单进程灵活切换（兼容PyTorch）
- ✅ 统一返回格式（`List[Dict]`）
- ✅ 数据自动对齐（'both'模式）

**日志控制**:
```python
from src.data_processer.preprocess.metadata_parser import LogConfig

# 禁用日志（不污染控制台）
LogConfig.disable_logging()

# 启用日志并输出到控制台
LogConfig.enable_logging(enable=True, to_console=True)

# 检查日志是否启用
if LogConfig.is_enabled():
    print("日志已启用")
```

##### `AnnotatedDatasetParser` - 标注数据集解析器

```python
from src.data_processer.preprocess.metadata_parser import AnnotatedDatasetParser

# 创建标注数据集解析器
parser = AnnotatedDatasetParser(
    vibration_metadata=vib_meta_list,
    annotation_result_path="annotations.json",
    num_workers=None,
    validate_metadata=False
)

# 构建分类任务数据集（基于标注）
dataset = parser.build_classification_dataset(
    data_base='annotation',            # 'annotation' 或 'data'
    mode='vibration',                  # 'vibration'/'wind'/'both'
    show_progress=True
)
# 返回: List[Dict] 包含 'data'、'label'、'metadata'、'statistics'

# 构建预测任务数据集
dataset = parser.build_prediction_dataset(
    prediction_step=100,               # 用100个样本预测接下来4步
    data_base='annotation',
    mode='vibration'
)
# 返回: List[Dict] 包含 'input'、'label'、'metadata'、'statistics'

# 保存为PyTorch格式
parser.save_dataset_as_torch(
    dataset=dataset,
    output_path="dataset.pt",
    format='pt'                        # 'pt' 或 'npz'
)
```

**数据基准选项**:
- `'annotation'`: 以标注结果为基准，仅包含已标注的样本
- `'data'`: 以原始数据为基准，未标注的样本标记为-1

---

### 2. `workflow.py` - 集成工作流

**用途**: 提供统一的主入口函数，集成所有预处理步骤

```python
from src.data_processer.preprocess.workflow import get_data_pairs

# 主工作流（获取对齐的振动-风数据对）
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',  # 可选
    enable_extreme_window=True,         # 使用极端窗口
    window_duration_minutes=None,       # 与上一个参数互斥
    use_vib_cache=True,
    use_wind_cache=True
)

# 处理数据对
for pair in data_pairs:
    metadata = pair['vib_metadata']
    config = pair['segment_config']
    windows = pair['segmented_windows']
    
    for vib_seg, (wind_speed, wind_dir, wind_angle) in windows:
        # 使用数据进行分析
        pass
```

---

### 3. `vib_metadata2data.py` - 振动数据解析器

**用途**: 将振动元数据转换为实际的振动数据

```python
from src.data_processer.preprocess.vib_metadata2data import (
    parse_single_metadata_to_vibration_data
)

# 解析单条振动元数据
vib_data = parse_single_metadata_to_vibration_data(
    metadata=vib_meta,
    enable_extreme_window=False,
    enable_denoise=False
)
```

**元数据格式**:
```python
{
    'file_path': str,              # VIC文件路径
    'sensor_id': str,              # 传感器ID
    'month': int,                  # 月份
    'day': int,                    # 日期
    'hour': int,                   # 小时
    'actual_length': int,          # 实际数据长度
    'missing_rate': float,         # 缺失率
    'extreme_rms_indices': list    # 极端窗口索引
}
```

---

### 4. `wind_metadata2data.py` - 风数据解析器

**用途**: 将风元数据转换为实际的风数据（风速、风向、风角）

```python
from src.data_processer.preprocess.wind_metadata2data import (
    parse_single_metadata_to_wind_data
)

# 解析单条风元数据
wind_data = parse_single_metadata_to_wind_data(
    metadata=wind_meta,
    enable_denoise=False
)
```

---

### 5. `vibration_io_process/` - 振动数据IO处理子模块

**3步工作流**:

| 步骤 | 功能 | 输入 | 输出 |
|------|------|------|------|
| Step 1 | 元数据提取 | VIC文件 | 振动元数据 |
| Step 2 | 数据裁剪和切分 | 振动元数据 | 裁剪后的数据和统计 |
| Step 3 | 频率统计 | 裁剪数据 | 主频率、极端RMS等 |

---

### 6. `wind_data_io_process/` - 风数据IO处理子模块

**3步工作流**:

| 步骤 | 功能 | 输入 | 输出 |
|------|------|------|------|
| Step 1 | 元数据提取 | VWD文件 | 风元数据 |
| Step 2 | 过滤和配对 | 风元数据 | 过滤后的配对数据 |
| Step 3 | 超限检测 | 风数据 | 超限标记 |

---

## 使用流程

### 场景1: 使用工作流进行数据配对

```python
from src.data_processer.preprocess.workflow import get_data_pairs

# Step 1: 获取对齐的数据对
data_pairs = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    enable_extreme_window=True
)

# Step 2: 处理数据
for pair in data_pairs:
    for vib_seg, (wind_speed, wind_dir, wind_angle) in pair['segmented_windows']:
        # 进行分析
        pass
```

### 场景2: 直接使用元数据解析器

```python
from src.data_processer.preprocess.metadata_parser import MetadataParser
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.preprocess.wind_data_io_process.workflow import run as run_wind_workflow

# Step 1: 运行工作流获取元数据
vib_metadata = run_vib_workflow(use_cache=True)
wind_metadata = run_wind_workflow(use_cache=True)

# Step 2: 创建元数据解析器
parser = MetadataParser(
    vibration_metadata=vib_metadata,
    wind_metadata=wind_metadata,
    num_workers=None              # 单进程模式
)

# Step 3: 解析数据
vib_data = parser.parse_data(mode='vibration', batch_size=32)
both_data = parser.parse_data(mode='both', batch_size=32)
```

### 场景3: 构建标注数据集

```python
from src.data_processer.preprocess.metadata_parser import AnnotatedDatasetParser

# Step 1: 创建标注数据集解析器
parser = AnnotatedDatasetParser(
    vibration_metadata=vib_metadata,
    annotation_result_path="results/dataset_annotation/annotation_results.json",
    num_workers=None
)

# Step 2: 构建数据集
dataset = parser.build_classification_dataset(
    data_base='annotation',    # 基于已标注数据
    mode='vibration'
)

# Step 3: 保存为PyTorch格式
parser.save_dataset_as_torch(
    dataset=dataset,
    output_path="datasets/vib_classification.pt",
    format='pt'
)
```

---

## 关键参数说明

### num_workers - 多进程控制

| 值 | 模式 | 用途 | 兼容性 |
|---|------|------|--------|
| `None` | 单进程 | PyTorch DataLoader | ✅ 完全兼容 |
| `0` | 多进程(自动) | 大数据量处理 | ⚠️ 需小心 |
| `>0` | 多进程(指定) | 大数据量处理 | ⚠️ 需小心 |

### batch_size - 批处理大小

- **多进程模式**: 每批提交给进程池的任务数
- **单进程模式**: 每批处理的任务数（用于进度显示）
- 建议值: 32-128

### validate_metadata - 元数据验证

- `False` (默认): 跳过验证，适合已预处理的数据
- `True`: 检查文件存在性，过滤无效元数据

---

## 返回数据格式

### parse_data() 返回格式

#### mode='vibration' 或 mode='wind'

```python
[
  {
    'vibration'/'wind': 单个数据结果(数组),
    'statistics': {
      'index': 原始列表中的索引,
      'status': 'success' 或 'failed',
      'metadata_id': 文件路径,
      'error': 错误信息
    }
  },
  ...
]
```

#### mode='both' - 对齐数据

```python
[
  {
    'vibration': 振动数据或None,
    'wind': 风数据或None,
    'statistics': {
      'vib_status': 'success'/'failed'/'missing',
      'wind_status': 'success'/'failed'/'missing',
      'vib_metadata_id': 文件路径,
      'wind_metadata_id': 文件路径,
      'vib_error': 错误或None,
      'wind_error': 错误或None
    }
  },
  ...
]
```

### build_classification_dataset() 返回格式

```python
[
  {
    'data': {
      'vibration': 数组,
      'wind': 数组或None,
    },
    'label': 0/1/2/3,  # -1表示未标注
    'metadata': {
      'sensor_id': str,
      'file_path': str,
      'annotation_raw': 原始标注字符串
    },
    'statistics': {
      'vib_status': 'success'/'failed'/'missing',
      'wind_status': 'success'/'failed'/'missing'
    }
  },
  ...
]
```

### save_dataset_as_torch() 输出

#### format='pt' (PyTorch)
```
dataset.pt  # 包含 'data'、'labels'、'metadata'
```

#### format='npz' (NumPy)
```
dataset.npz              # 压缩的标签数组
dataset_metadata.json    # 元数据和数据引用
data_*.npy              # 各样本数据文件
```

---

## 性能特性

### 内存管理

| 场景 | 策略 | 效果 |
|------|------|------|
| PyTorch DataLoader | 单进程模式 | 无多进程冲突 |
| 大数据量 | 多进程+批处理 | 内存受控，效率高 |
| 元数据验证 | 可关闭 | 跳过文件检查，加快初始化 |

### 批处理优势

- **多进程模式**: 每批分别提交，避免任务队列过长
- **单进程模式**: 逐个处理，适合内存有限的环境
- 内存占用: O(batch_size) 而不是 O(total_size)

---

## 日志系统

### 日志开关

```python
from src.data_processer.preprocess.metadata_parser import LogConfig

# 完全禁用日志
LogConfig.disable_logging()

# 启用日志到文件
LogConfig.enable_logging(enable=True, to_console=False)

# 启用日志到控制台
LogConfig.enable_logging(enable=True, to_console=True)

# 检查状态
if LogConfig.is_enabled():
    print("日志已启用")
```

### 日志级别

| 级别 | 用途 | 示例 |
|------|------|------|
| INFO | 关键信息 | "开始解析数据..." |
| WARNING | 警告信息 | "无振动元数据可解析" |
| DEBUG | 调试信息 | "过滤掉无效元数据..." |
| ERROR | 错误信息 | 解析失败时 |

---

## 错误处理

### 异常类型

| 异常 | 原因 | 解决方案 |
|------|------|--------|
| ValueError | mode参数非法 | 使用'vibration'/'wind'/'both' |
| FileNotFoundError | 文件不存在 | 启用validate_metadata检查 |
| KeyError | 元数据字段缺失 | 检查元数据结构 |
| Exception (worker) | 解析失败 | 查看result['error'] |

### 容错机制

- Worker函数自动捕获异常
- 单个任务失败不影响其他任务
- 错误信息保存在result中供检查

---

## 采样频率和窗口配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 振动采样频率 | 50 Hz | 标准振动传感器采样率 |
| 风采样频率 | 1 Hz | 标准气象数据采样率 |
| 标准窗口时长 | 60 秒 | 振动和风数据都采用60秒窗口 |
| 振动窗口大小 | 3000点 | 50 Hz × 60 秒 |
| 风窗口大小 | 60点 | 1 Hz × 60 秒 |
| 频率比 | 50:1 | 振动:风 |

---

## 最佳实践

### ✅ 推荐用法

```python
# 1. PyTorch训练时
parser = MetadataParser(
    vibration_metadata=vib_meta,
    num_workers=None,           # 单进程
    validate_metadata=False     # 跳过验证
)

# 2. 大数据量处理时
parser = MetadataParser(
    vibration_metadata=vib_meta,
    num_workers=4,              # 多进程
    validate_metadata=False,
    batch_size=64               # 大批次
)

# 3. 标注数据集构建
parser = AnnotatedDatasetParser(
    vibration_metadata=vib_meta,
    annotation_result_path="annotations.json",
    num_workers=None
)
dataset = parser.build_classification_dataset(
    data_base='annotation',
    mode='vibration'
)
```

### ❌ 避免

```python
# 1. 不要在PyTorch DataLoader中使用多进程
parser = MetadataParser(num_workers=4)  # 会导致冲突

# 2. 不要频繁启用元数据验证（已预处理）
parser = MetadataParser(validate_metadata=True)  # 减速

# 3. 不要设置过小的batch_size
batch_size=1  # 效率低，进度条噪音多
```

---

## 常见问题

### Q: 什么时候应该使用多进程？

**A**: 当处理10万+条元数据且不使用PyTorch DataLoader时。
```python
parser = MetadataParser(num_workers=4)  # 使用多进程
```

### Q: 为什么PyTorch DataLoader中要用单进程？

**A**: 多进程会导致DataLoader中的worker产生冲突。
```python
parser = MetadataParser(num_workers=None)  # 必须单进程
```

### Q: 如何控制日志输出？

**A**: 使用LogConfig类。
```python
LogConfig.disable_logging()  # 完全禁用
LogConfig.enable_logging(to_console=False)  # 输出到日志文件
```

### Q: 元数据验证会减速吗？

**A**: 是的，如果数据已预处理，建议禁用。
```python
parser = MetadataParser(validate_metadata=False)  # 默认值
```

### Q: 如何保证振动和风数据对齐？

**A**: 使用 `mode='both'` 会自动对齐，通过索引配对。

---

## 相关文档

- [振动IO流程详解](./vib_data_io_process/README.md)
- [风IO流程详解](./wind_data_io_process/README.md)
- [工作流快速开始](./workflow/QUICKSTART.md)
- [工作流API文档](./workflow/API.md)

---

## 版本信息

- **最后更新**: 2026年3月12日
- **模块版本**: 2.0
- **核心改进**:
  - ✅ 统一的parse_data()接口
  - ✅ 元数据自动验证和过滤
  - ✅ 真正的批处理逻辑
  - ✅ 多进程/单进程灵活切换
  - ✅ 标注数据集构建
  - ✅ PyTorch格式导出
  - ✅ 灵活的日志控制

---

## 反馈与改进

如有问题或改进建议，请：

1. 查看相关子模块文档
2. 检查日志输出获取更多调试信息
3. 使用validate_metadata=True排查数据问题
