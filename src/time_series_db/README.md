# TimeSeriesFileIndex - 时序文件索引管理库

## 概述

`TimeSeriesFileIndex` 是一个用于高效管理和查询时序文件的工具库。它将文件按时间（精确到小时）组织，支持快速按时间、数据类型、传感器ID等维度进行查询。该库特别适用于处理传感器数据、气象数据等时间序列数据。

## 核心设计

### 数据基础单元

数据以**时间戳（精确到小时）** 作为基础单元，每个时间戳对应一个数据点，包含以下信息：

| 组件 | 描述 | 示例 |
|------|------|------|
| **时间戳** | 日期时间（精确到小时） | `2023-01-01 05:00` |
| **月份** | 月份（1-12） | `1` |
| **日期** | 日期（1-31） | `1` |
| **小时** | 小时（0-23） | `5` |
| **数据类型** | 文件数据的类型 | `wind_speed`, `acceleration` |
| **文件分组** | 按数据类型分组的文件路径列表 | `{ "wind_speed": ["file1.UAN", "file2.UAN"] }` |

### 索引结构

索引表使用Pandas DataFrame存储，结构如下：

| 列名 | 类型 | 说明 |
|------|------|------|
| `month` | int8 | 月份（1-12） |
| `day` | int8 | 日期（1-31） |
| `hour` | int8 | 小时（0-23） |
| `data_types` | object | 该时间点可用的数据类型列表 |
| `file_groups` | object | 按数据类型分组的文件路径列表 |

索引表的行索引是时间戳（`timestamp`），精确到小时，确保时间序列的正确顺序。

## 主要接口

### 初始化

```python
# 创建新的索引表
db = TimeSeriesFileIndex()

# 从配置文件初始化
db = TimeSeriesFileIndex("mapping_config.yaml")
```

### 添加文件

```python
# 添加单个文件
db.add_file("path/to/file.UAN")

# 批量添加文件（支持多线程）
db.add_files(["file1.UAN", "file2.UAN"], max_workers=4)
```

### 查询文件

```python
# 按时间戳和数据类型查询
files = db.get_files_by_time_and_type(datetime(2023, 1, 1, 5), "wind_speed")

# 按月、日、时和数据类型查询
files = db.get_files_by_hour_and_type(1, 1, 5, "wind_speed")

# 按传感器ID查询
files = db.get_files_by_sensor_id("ST-UAN-T01-003-01")

# 搜索多个传感器ID
results = db.search_sensor_ids(["ST-UAN-T01-003-01", "ST-ACC-T01-002"])
```

### 数据分析

```python
# 获取数据覆盖统计
coverage = db.get_data_coverage()

# 获取可用数据类型
types = db.get_available_data_types()
```

### 保存与加载

```python
# 保存到目录
db.save_to_parquet("data/index")

# 从目录加载
db = TimeSeriesFileIndex.load_from_parquet("data/index")
```

## 文件命名与结构

### 文件命名规范

文件名遵循以下结构：`<sensor_id>_<time>.<extension>`

- `<sensor_id>`: 传感器ID，如`ST-UAN-T01-003-01`
- `<time>`: 时间，6位数字格式（HHMMSS），如`050000`（05:00:00）
- `<extension>`: 文件扩展名，如`.UAN`

### 文件夹结构

文件夹结构遵循以下约定：

```
data/
└── [month]/  # 月份文件夹（如01）
    └── [day]/  # 日期文件夹（如01）
        └── ST-UAN-T01-003-01_050000.UAN
```

- `[month]`: 月份（1-12），如`01`表示1月
- `[day]`: 日期（1-31），如`01`表示1日

## 映射配置

`FileTypeMappingConfig` 管理文件类型映射规则，支持多种映射策略：

1. **扩展名映射**：`.uan` → `wind_speed`
2. **传感器ID映射**：`ST-UAN` → `wind_speed`
3. **传感器类型映射**：`UAN` → `wind_speed`
4. **文件名模式映射**：`wind.*` → `wind_speed`

默认配置包括常见传感器数据类型的映射，可通过YAML/JSON文件自定义。

## 工作流程示例

```python
# 加载现有索引
db = TimeSeriesFileIndex.load_from_parquet("data/index")

# 添加新文件
new_files = ["data/01/01/ST-UAN-T01-003-01_050000.UAN"]
db.add_files(new_files)

# 保存更新后的索引
db.save_to_parquet("data/updated_index")

# 查询特定传感器的数据
files = db.get_files_by_sensor_id("ST-UAN-T01-003-01")
```

## 优势

1. **高效查询**：基于时间戳的索引，支持快速时间范围查询
2. **灵活映射**：多种映射策略，可适应不同数据源
3. **数据可视化**：提供清晰的摘要信息和数据覆盖统计
4. **可扩展性**：支持自定义映射配置，适应不同传感器类型
5. **高效存储**：使用Parquet格式存储，节省空间并提高读取速度

## 使用建议

1. **组织文件**：按照`[month]/[day]/`结构组织文件，便于自动解析
2. **命名规范**：遵循`<sensor_id>_<time>.<extension>`命名规则
3. **配置文件**：使用YAML配置文件管理映射规则，便于维护
4. **批量处理**：使用`add_files`进行批量添加，利用多线程提高效率
5. **定期保存**：定期调用`save_to_parquet`保存索引，避免数据丢失

## 总结

`TimeSeriesFileIndex` 为时序文件管理提供了一个高效、灵活的解决方案，特别适合处理传感器数据和时间序列数据。通过精确到小时的时间索引和多种映射策略，它能够轻松应对各种数据组织需求，同时提供丰富的查询和分析功能。