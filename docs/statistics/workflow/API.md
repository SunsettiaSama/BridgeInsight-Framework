# API 参考

## 公开接口

### get_data_pairs()

主入口函数，用于批量提取和处理振动与风数据对。

**签名:**
```python
get_data_pairs(
    wind_sensor_id: str,
    vib_sensor_id: str | None = None,
    use_multiprocess: bool = False,
    enable_extreme_window: bool = False,
    window_duration_minutes: float | None = None,
    use_vib_cache: bool = True,
    use_wind_cache: bool = True
) -> List[Dict]
```

**必需参数:**
- `wind_sensor_id` (str): 风传感器 ID

**可选参数:**
- `vib_sensor_id` (str, optional): 振动传感器 ID，用于筛选特定拉索
- `use_multiprocess` (bool): 是否启用多进程，默认 False
- `enable_extreme_window` (bool): 是否进行极端窗口筛选，默认 False
- `window_duration_minutes` (float, optional): 时间窗口长度（分钟）
- `use_vib_cache` (bool): 是否使用振动数据缓存，默认 True
- `use_wind_cache` (bool): 是否使用风数据缓存，默认 True

**返回:**
```
List[Dict] - 数据对列表，每个数据对包含：
  {
      'vib_metadata': Dict,       # 振动元数据
      'segment_config': Dict,      # 切分配置
      'segmented_windows': List    # 数据窗口对
  }
```

**异常:**
- 无异常抛出，返回空列表表示处理失败

**示例:**
```python
# 极端窗口切分
result = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    enable_extreme_window=True
)

# 常规窗口切分
result = get_data_pairs(
    wind_sensor_id='ST-UAN-G04-001-01',
    vib_sensor_id='ST-VIC-C18-101-01',
    window_duration_minutes=2.0
)
```

---

## 内部接口（私有）

以下是模块内部使用的私有函数，不推荐外部直接调用。

### _get_processed_metadata()

获取处理后的元数据（内部函数）

### _segment_windows_by_duration()

按时间长度切分数据（内部函数）

### _segment_extreme_wind_windows()

按极端索引切分数据（内部函数）

### _load_vibration_and_wind_data()

加载原始数据（内部函数）

---

## 数据结构

### vib_metadata 结构

```python
{
    'sensor_id': str,              # 传感器 ID
    'month': int,                  # 月份
    'day': int,                    # 日期
    'hour': int,                   # 小时
    'file_path': str,              # 文件路径
    'actual_length': int,          # 实际数据长度
    'missing_rate': float,         # 缺失率
    'extreme_rms_indices': List    # 极端 RMS 索引
}
```

### segment_config 结构

```python
{
    'vib_sensor_id': str | None,           # 振动传感器 ID
    'wind_sensor_id': str,                 # 风传感器 ID
    'enable_extreme_window': bool,         # 是否极端窗口
    'window_duration_minutes': float | None,  # 窗口时长
    'vib_fs': int,                         # 振动采样频率 (50 Hz)
    'wind_fs': int                         # 风速采样频率 (1 Hz)
}
```

### segmented_windows 结构

```python
[
    (
        vib_segment: np.ndarray,   # 振动数据段
        (
            wind_speed: np.ndarray,
            wind_direction: np.ndarray,
            wind_angle: np.ndarray
        )
    ),
    ...
]
```

---

## 常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `_VIB_FS` | 50 | 振动采样频率 (Hz) |
| `_WIND_FS_CONFIG` | 1 | 风速采样频率 (Hz) |
| `_VIB_TIME_WINDOW` | 60.0 | 振动时间窗口 (s) |
| `_WIND_TIME_WINDOW` | 60.0 | 风速时间窗口 (s) |
| `_VIB_WINDOW_SIZE` | 3000 | 振动窗口大小 (采样点) |
| `_WIND_WINDOW_SIZE` | 60 | 风速窗口大小 (采样点) |

---

## 错误处理

函数通过返回空列表来表示处理失败，不抛出异常。具体原因可通过打印的信息日志查看。

### 常见返回空列表的情况

1. 无可用的元数据
2. 指定的传感器无匹配数据
3. 数据文件读取失败
4. 切分参数无效

---

## 性能特性

- **内存占用**: O(n × m)，其中 n 为元数据数量，m 为每个数据对的平均窗口数
- **时间复杂度**: O(n × m × p)，其中 p 为窗口数据的平均长度
- **缓存支持**: 支持两级缓存（振动数据和风数据）

---

## 变更日志

### 2026-02-25
- 重构主入口函数 `get_data_pairs()`，使其无需预加载元数据
- 所有辅助函数转为私有（`_` 前缀）
- 集成 `vibration_io_process` 和 `wind_data_io_process` 工作流
- 添加完整的使用文档
