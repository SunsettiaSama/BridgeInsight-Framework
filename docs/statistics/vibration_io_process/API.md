# API 参考

## 公开接口

### `run()`

主入口函数，执行完整的振动数据处理工作流。

**签名:**
```python
def run(threshold: float = MISSING_RATE_THRESHOLD,
        expected_length: int = EXPECTED_LENGTH,
        save_path: str = FILTER_RESULT_PATH,
        cache_path: str = WORKFLOW_CACHE_PATH,
        use_cache: bool = True,
        force_recompute: bool = False) -> List[Dict]
```

**参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `threshold` | float | 0.05 | 缺失率阈值 (0-1) |
| `expected_length` | int | 180000 | 预期单文件长度 |
| `save_path` | str | 配置值 | 元数据保存路径 |
| `cache_path` | str | 配置值 | 缓存文件路径 |
| `use_cache` | bool | True | 是否使用缓存 |
| `force_recompute` | bool | False | 是否强制重新计算 |

**返回:**
```python
List[Dict] - 元数据列表，每项为：
[
    {
        'sensor_id': str,                # 传感器 ID
        'month': int,                    # 月份
        'day': int,                      # 日期
        'hour': int,                     # 小时
        'file_path': str,                # 文件路径
        'actual_length': int,            # 实际长度（采样点数）
        'missing_rate': float,           # 缺失率 (0-1)
        'extreme_rms_indices': list      # 极端振动窗口索引
    },
    ...
]
```

**异常:**
- 无异常抛出。返回空列表表示处理失败。

**示例:**
```python
# 基础用法
metadata = run()

# 自定义参数
metadata = run(threshold=0.01, force_recompute=True)

# 查看结果
for item in metadata:
    print(f"{item['sensor_id']}: {len(item['extreme_rms_indices'])} 个极端窗口")
```

---

## 内部函数（私有）

以下函数为模块内部使用，不建议外部直接调用。

### `step0_get_vib_data.get_all_vibration_files()`

获取所有振动数据文件路径。

**签名:**
```python
def get_all_vibration_files(root_dir: str = ALL_VIBRATION_ROOT,
                           target_sensor_ids: list = TARGET_VIBRATION_SENSORS,
                           suffix: str = VIBRATION_FILE_SUFFIX) -> List[str]
```

**参数:**
- `root_dir` (str): 数据根目录
- `target_sensor_ids` (list): 目标传感器ID列表
- `suffix` (str): 文件后缀 (默认 '.VIC')

**返回:**
- `List[str]` - 文件路径列表

---

### `step1_lackness_filter.run_lackness_filter()`

执行缺失率筛选。

**签名:**
```python
def run_lackness_filter(all_file_paths: List[str],
                       threshold: float = MISSING_RATE_THRESHOLD,
                       expected_length: int = EXPECTED_LENGTH,
                       logger: Optional[Logger] = None) -> Tuple[List[str], Dict]
```

**参数:**
- `all_file_paths` (List[str]): 所有文件路径
- `threshold` (float): 缺失率阈值
- `expected_length` (int): 预期单文件长度
- `logger` (Optional): 日志记录器

**返回:**
```python
Tuple[List[str], Dict] - (筛选后的路径列表, 统计信息字典)

统计信息包含：
{
    'all_lengths': np.ndarray,           # 所有文件的长度数组
    'all_missing_rates': np.ndarray,     # 所有文件的缺失率数组
    'filtered_indices': list             # 筛选通过的索引列表
}
```

---

### `step2_rms_statistics.run_rms_statistics()`

执行RMS统计分析和极端振动识别。

**签名:**
```python
def run_rms_statistics(file_paths: List[str],
                      fs: float = FS,
                      time_window: float = TIME_WINDOW,
                      logger: Optional[Logger] = None) -> Tuple[List[str], Dict]
```

**参数:**
- `file_paths` (List[str]): 文件路径列表
- `fs` (float): 采样频率 (默认 50 Hz)
- `time_window` (float): 时间窗口 (默认 60秒)
- `logger` (Optional): 日志记录器

**返回:**
```python
Tuple[List[str], Dict] - (文件路径列表, 统计信息字典)

统计信息包含：
{
    'all_file_rms': list,                # 每个文件的RMS数组列表
    'extreme_indices': list,             # 每个文件的极端窗口索引
    'rms_threshold_95': float            # 95%分位值阈值 (m/s²)
}
```

---

## 数据结构

### vib_metadata 结构

元数据字典结构（每条记录代表一个1小时的数据文件）：

```python
{
    'sensor_id': str,                    # 传感器ID (e.g., 'ST-VIC-C18-101-01')
    'month': int,                        # 月份 (1-12)
    'day': int,                          # 日期 (1-31)
    'hour': int,                         # 小时 (0-23)
    'file_path': str,                    # 源文件完整路径
    'actual_length': int,                # 实际数据长度（采样点数）
    'missing_rate': float,               # 缺失率 (0.0-1.0)
    'extreme_rms_indices': list          # 极端RMS窗口索引 [0, 5, 12, ...]
}
```

**字段说明:**

| 字段 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `sensor_id` | str | - | 唯一标识传感器 |
| `month` | int | 1-12 | 数据采集月份 |
| `day` | int | 1-31 | 数据采集日期 |
| `hour` | int | 0-23 | 数据采集小时 |
| `file_path` | str | - | 用于数据加载 |
| `actual_length` | int | 0-180000 | 单位：采样点 |
| `missing_rate` | float | 0.0-1.0 | 0 = 完整, 1 = 全失 |
| `extreme_rms_indices` | list | [int, ...] | 窗口索引 |

### extreme_rms_indices 详解

```python
# 示例
'extreme_rms_indices': [5, 12, 23]

# 说明：
# - 索引从 0 开始
# - 每个索引对应一个 60 秒的时间窗口
# - 索引 5 对应时间 300-360 秒 (5 × 60)
# - 索引 12 对应时间 720-780 秒 (12 × 60)
# - 索引 23 对应时间 1380-1440 秒 (23 × 60)
```

---

## 常量

### 缺失率相关

| 常量 | 值 | 说明 |
|------|-----|------|
| `MISSING_RATE_THRESHOLD` | 0.05 | 缺失率筛选阈值 (5%) |
| `EXPECTED_LENGTH` | 180000 | 预期单文件长度 (50Hz × 3600s) |

### 采样频率相关

| 常量 | 值 | 说明 |
|------|-----|------|
| `FS` | 50 | 振动采样频率 (Hz) |
| `TIME_WINDOW` | 60.0 | RMS计算窗口 (秒) |
| `WINDOW_SIZE` | 3000 | 窗口采样点数 (50 × 60) |

### 文件配置

| 常量 | 说明 |
|------|------|
| `ALL_VIBRATION_ROOT` | 振动数据根目录 |
| `VIBRATION_FILE_SUFFIX` | 文件后缀 ('.VIC') |
| `TARGET_VIBRATION_SENSORS` | 目标传感器ID列表 |

### 路径配置

| 常量 | 说明 |
|------|------|
| `FILTER_RESULT_PATH` | 元数据保存路径 |
| `WORKFLOW_CACHE_PATH` | 缓存文件路径 |

---

## 错误处理

函数不抛出异常，而是通过返回值表示结果：

### 成功返回
```python
# 正常结果
metadata = run()
# 返回非空列表
assert len(metadata) > 0
```

### 失败返回
```python
# 处理失败
metadata = run()
if not metadata:
    # 原因可能：
    # 1. 无文件匹配
    # 2. 数据读取失败
    # 3. 配置错误
    pass
```

---

## 使用模式

### 模式 1: 简单调用

```python
from src.data_processer.statistics.vibration_io_process.workflow import run

metadata = run()
print(f"处理了 {len(metadata)} 个文件")
```

### 模式 2: 自定义参数

```python
metadata = run(
    threshold=0.01,           # 严格筛选
    force_recompute=True      # 强制更新
)
```

### 模式 3: 缓存管理

```python
# 使用缓存（快速）
metadata = run(use_cache=True)

# 禁用缓存（完整计算）
metadata = run(use_cache=False)
```

### 模式 4: 结果处理

```python
metadata = run()

# 按传感器分组
from collections import defaultdict
by_sensor = defaultdict(list)
for item in metadata:
    by_sensor[item['sensor_id']].append(item)

# 找极端振动
extreme_items = [m for m in metadata if len(m['extreme_rms_indices']) > 0]

# 统计分析
print(f"总记录数: {len(metadata)}")
print(f"极端记录数: {len(extreme_items)}")
print(f"平均缺失率: {np.mean([m['missing_rate'] for m in metadata]):.2%}")
```

---

## 性能参考

### 处理时间（参考值）

| 文件数 | Step 0 | Step 1 | Step 2 | 总计 |
|--------|--------|--------|--------|------|
| 100 | <1s | 10s | 15s | ~25s |
| 1000 | <1s | 100s | 150s | ~250s |
| 10000 | <1s | 1000s | 1500s | ~2500s |

注：实际时间取决于硬件性能和I/O速度

### 内存占用（参考值）

| 文件数 | 元数据 | 缓存 | 处理 | 总计 |
|--------|--------|------|------|------|
| 100 | 20KB | 50KB | 200MB | ~200MB |
| 1000 | 200KB | 500KB | 500MB | ~500MB |
| 10000 | 2MB | 5MB | 2GB | ~2GB |

注：处理内存随并行度增加而变化

---

## 版本信息

- **最后更新**: 2026-02-26
- **模块版本**: 1.0
- **API 兼容性**: 稳定
