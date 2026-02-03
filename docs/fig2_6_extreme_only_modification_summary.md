# fig2_6_wind_turbulence.py 修改总结

## 修改目标
将风数据紊流度风玫瑰图的数据来源限定为**仅极端振动对应的风荷载数据**，排除所有非极端振动时段的风数据。

---

## 核心修改内容

### 1. 数据来源控制（Line 467）
```python
extreme_only=True  # 仅使用极端振动对应的风数据
```
- 在 `run_wind_workflow()` 中启用极端模式
- 工作流返回的 `wind_metadata` 包含 `extreme_time_ranges` 信息

### 2. 新增时间窗口截取函数（Line 211-247）
```python
def extract_extreme_time_windows(wind_velocity, wind_direction, extreme_time_ranges, fs=WIND_FS):
```
**功能**：
- 接收完整的风数据和极端时间窗口列表 `[(start_sec, end_sec), ...]`
- 按时间窗口截取风速和风向数据
- 支持1Hz采样频率（索引 = 秒数 × 采样率）
- 返回拼接后的极端窗口数据数组

**关键逻辑**：
```python
for start_sec, end_sec in extreme_time_ranges:
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    # 确保索引有效
    start_idx = max(0, start_idx)
    end_idx = min(len(wind_velocity), end_idx)
    # 截取并拼接
    extracted_velocities.extend(wind_velocity[start_idx:end_idx])
```

### 3. 修改单文件数据加载函数（Line 250-287）
```python
def process_single_wind_file(file_path, extreme_time_ranges=None, valid_threshold=WIND_VALID_THRESHOLD):
```
**新增参数**：
- `extreme_time_ranges`：极端时间窗口列表（默认None保持向后兼容）

**处理流程**：
1. 解析完整风数据文件
2. 调用 `extract_extreme_time_windows()` 截取极端时间窗口
3. 过滤无效风速（低于阈值的数据）
4. 返回有效的极端窗口数据

### 4. 修改传感器数据加载函数（Line 290-349）
```python
def load_wind_data_by_sensor(wind_metadata, sensor_id, use_multiprocess=True, max_workers=None):
```
**关键修改**：
- 从 `wind_metadata` 提取每个文件的 `extreme_time_ranges`
- 将 `(file_path, extreme_time_ranges)` 配对传递给 `process_single_wind_file()`
- 支持多进程并行处理

**核心代码**：
```python
file_info_list = []
for item in sensor_files:
    file_path = item.get('file_path')
    extreme_time_ranges = item.get('extreme_time_ranges', [])
    file_info_list.append((file_path, extreme_time_ranges))

# 多进程调用
futures = {executor.submit(process_single_wind_file, fp, etr): (fp, etr) 
          for fp, etr in file_info_list}
```

### 5. 增强传感器处理函数输出（Line 357-388）
```python
def process_sensor_data(sensor_id, wind_metadata, interval_nums=36, use_multiprocess=True):
```
**新增统计输出**：
- 极端数据文件数
- 极端时间窗口数
- 有效样本数（1Hz采样）
- 明确标注"极端窗口数据"

**输出示例**：
```
极端数据文件数: 45
极端时间窗口数: 128
✓ 极端窗口数据加载完成
  有效样本数: 7680 (1Hz采样)
  风速范围: 2.35 ~ 18.42 m/s
  平均风速: 8.73 m/s
```

### 6. 主函数修改（Line 441-522）
**标题更新**：
```python
print(" "*20 + "图2.6: 风数据紊流度分析（极端振动时段）")
```

**新增统计输出**（Line 471-473）：
```python
total_extreme_windows = sum(len(item.get('extreme_time_ranges', [])) for item in wind_metadata)
print(f"✓ 极端时间窗口总数: {total_extreme_windows}")
```

**完成信息增强**（Line 515-517）：
```python
print(f"✓ 数据来源：仅极端振动对应的风荷载数据（1Hz采样）")
print(f"✓ 极端时间窗口总数: {total_extreme_windows}")
```

### 7. 文档更新（Line 1-20）
- 更新文件顶部docstring，说明仅使用极端振动数据
- 添加完整的数据处理流程说明
- 标注1Hz采样频率和0-100%颜色映射范围

---

## 技术要点

### 时间窗口格式
- `extreme_time_ranges`: `[(start_sec, end_sec), ...]`
- 时间以文件开始时刻为0秒的相对时间
- 单位：秒（浮点数）

### 采样频率处理
- 风数据原生采样率：**1Hz**（从配置 `WIND_FS = 1` 导入）
- 索引计算：`index = time_in_seconds × 1Hz`
- 无需降采样或插值处理

### 数据一致性保证
- ✅ 时间戳精确匹配
- ✅ 无数据重复
- ✅ 无数据遗漏
- ✅ 风速、风向完整对应

### 保留原有功能
- ✅ 紊流度计算（样本标准差 `ddof=1`）
- ✅ 异常值截断（`max_ti=MAX_TURBULENCE_INTENSITY`）
- ✅ 颜色映射统一（`plt.Normalize(0, 100)`）
- ✅ 桥轴线标注
- ✅ 百分比归一化显示
- ✅ 多进程并行加载
- ✅ PlotLib统一管理

---

## 代码验证

### Linter检查
```bash
✓ 无语法错误
✓ 无类型错误
✓ 无导入错误
```

### 关键函数测试建议

**测试1：时间窗口截取**
```python
# 模拟数据
velocity = np.arange(0, 3600)  # 3600秒 = 1小时
direction = np.random.uniform(0, 360, 3600)
extreme_ranges = [(60, 120), (1800, 1860)]  # 两个60秒窗口

# 调用截取函数
vel_extracted, dir_extracted = extract_extreme_time_windows(velocity, direction, extreme_ranges)

# 验证
assert len(vel_extracted) == 120  # 60 + 60
assert vel_extracted[0] == 60  # 第一个窗口起点
assert vel_extracted[60] == 1800  # 第二个窗口起点
```

**测试2：文件加载**
```python
# 准备测试元数据
test_metadata = [{
    'sensor_id': 'ST-UAN-T01-003-01',
    'file_path': '/path/to/wind_file.txt',
    'extreme_time_ranges': [(0, 60), (120, 180)]
}]

# 加载数据
velocities, directions = load_wind_data_by_sensor(test_metadata, 'ST-UAN-T01-003-01')

# 验证数据来源仅为极端窗口
print(f"加载样本数: {len(velocities)}")  # 应为 60 + 60 = 120 左右（过滤后）
```

---

## 运行方式

```bash
# 在项目根目录执行
cd "f:\Research\Vibration Characteristics In Cable Vibration"
python -m src.figs.figs_for_thesis.fig2_6_wind_turbulence
```

**预期输出**：
```
================================================================================
              图2.6: 风数据紊流度分析（极端振动时段）
================================================================================

[Step 1] 运行振动数据工作流...
✓ 获取振动数据元数据: XXX 条

[Step 2] 运行风数据工作流（筛选极端振动时段）...
✓ 获取极端振动对应的风数据元数据: XXX 条
✓ 极端时间窗口总数: XXX

各传感器文件数统计：
  ...

[Step 3] 为每个传感器处理数据并绘图...
处理传感器: ST-UAN-T01-003-01
极端数据文件数: XXX
极端时间窗口数: XXX
✓ 极端窗口数据加载完成
  有效样本数: XXX (1Hz采样)
  ...
```

---

## 注意事项

1. **缓存机制**：首次运行会生成缓存，后续运行会使用缓存加速
2. **强制重算**：如需重新计算，设置 `force_recompute=True`
3. **无效数据处理**：若某传感器无极端振动数据，会输出警告并跳过
4. **内存优化**：使用 `plt.close('all')` 释放内存

---

## 相关文件

- **主程序**：`src/figs/figs_for_thesis/fig2_6_wind_turbulence.py`
- **工作流**：`src/data_processer/statistics/wind_data_io_process/workflow.py`
- **极端筛选**：`src/data_processer/statistics/wind_data_io_process/step2_extreme_filter.py`
- **配置文件**：`src/config/sensor_config.py`
- **开发日志**：`docs/development.md`

---

## 修改日期
2026-02-03
