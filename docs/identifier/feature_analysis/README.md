# process_full_data — 全量识别结果后处理模块

## 概述

该模块承接深度学习识别阶段的输出（enriched JSON），对全量样本并行计算振动特征与风场统计量，并按振动类别（可选再按传感器）分割保存，为后续统计分析和作图提供结构化、可直接查询的特征文件。

**输入** → enriched 识别结果 JSON + 原始风元数据 JSON  
**输出** → 按类别×传感器分组的特征 JSON 文件集

---

## 目录结构

```
src/identifier/process_full_data/
├── run.py          主流水线（CLI 入口 + Python API）
├── _modal.py       频域特征：PSD 主导模态 / 谱熵 / 谱带宽 / 模态能量占比
├── _signal.py      时域特征：RMS / 峭度 / 偏度 / 波峰因子 / 过零率
├── _coupling.py    耦合特征：面内外互相关 / 轨迹椭圆率 / 相干性 / 相位差
├── _wind.py        风场特征：多传感器统计量 / 紊流度 / 折减风速
└── _splitter.py    结果分割与保存

src/config/identifier/process_full_data/
└── config.py       ProcessFullDataConfig（Pydantic）+ load_config()

config/identifier/process_full_data/
└── default.yaml    默认配置（所有开关的预设值）
```

---

## 特征目录

所有特征均可通过配置开关独立开启/关闭。

### 频域特征（`enable_psd_modes` / `enable_spectral_features`）

| 字段 | 说明 |
|------|------|
| `psd_inplane.frequencies` | 面内 PSD 前 N 阶主导频率（Hz），按频率升序 |
| `psd_inplane.powers` | 对应的功率谱密度值 |
| `psd_outplane.*` | 面外方向，同上 |
| `spectral_inplane.spectral_entropy` | 谱熵 H = −Σ p·ln(p)，越低越窄带（涡激），越高越宽带（随机） |
| `spectral_inplane.spectral_centroid_hz` | 谱质心（加权平均频率） |
| `spectral_inplane.spectral_bandwidth_hz` | 谱带宽（质心的加权标准差） |
| `spectral_inplane.top_modes_energy_ratio` | 前 N 峰值能量之和 / 总能量 |
| `spectral_inplane.dominant_mode_energy_ratio` | 第一主频能量 / 总能量 |
| `spectral_outplane.*` | 面外方向，同上 |

### 时域统计（`enable_time_stats`）

| 字段 | 说明 |
|------|------|
| `time_stats_inplane.rms` | 均方根值，振幅度量 |
| `time_stats_inplane.kurtosis` | 峭度（Fisher 定义），正态分布≈0；高值表示冲击成分 |
| `time_stats_inplane.skewness` | 偏度，分布不对称程度 |
| `time_stats_inplane.crest_factor` | 波峰因子 = peak / RMS |
| `time_stats_inplane.zero_crossing_rate` | 过零率，与主频近似正相关 |
| `time_stats_outplane.*` | 面外方向，同上 |

### 面内外耦合（`enable_cross_coupling`）

| 字段 | 说明 |
|------|------|
| `cross_coupling.cross_correlation` | Pearson 互相关系数，风雨振的耦合特征 |
| `cross_coupling.ellipticity` | 轨迹椭圆率（PCA），0→线性，趋近1→圆形轨迹 |
| `cross_coupling.dominant_coherence` | 在面内主频处的互谱相干值 [0,1] |
| `cross_coupling.phase_difference_deg` | 面内主频处的面内外相位差（°）；±90°对应椭圆轨迹 |

### 风场统计（`enable_wind_stats`）

每个样本返回列表，每项对应一个风传感器（同一时间戳可能有 k 个）。

| 字段 | 说明 |
|------|------|
| `wind_stats[i].sensor_id` | 风传感器 ID |
| `wind_stats[i].mean_wind_speed` | 平均风速 Ū（m/s） |
| `wind_stats[i].std_wind_speed` | 风速标准差 σ_u |
| `wind_stats[i].mean_wind_direction` | 平均风向（°） |
| `wind_stats[i].std_wind_direction` | 风向标准差 |
| `wind_stats[i].mean_wind_attack_angle` | 平均风攻角（°） |
| `wind_stats[i].std_wind_attack_angle` | 风攻角标准差 |
| `wind_stats[i].turbulence_intensity` | 紊流度 Iu = σ_u / Ū |

### 折减风速（`enable_reduced_velocity`，需提供 `cable_diameter_map`）

每个样本按 `inplane_sensor_id` 在 `cable_diameter_map` 中查找对应拉索外径后计算，不同拉索外径各异时可精确对应。

| 字段 | 说明 |
|------|------|
| `reduced_velocity[i].sensor_id` | 对应风传感器 |
| `reduced_velocity[i].reduced_velocity` | Vr = Ū / (f₁ × D)；VIV 通常在 Vr ≈ 5~12 |

---

## 配置参考

配置类位于 `src/config/identifier/process_full_data/config.py`，通过 YAML 文件加载。

```yaml
# config/identifier/process_full_data/default.yaml

fs: 50.0              # 振动采样频率（Hz），须与识别阶段一致
window_size: 3000     # 单窗口采样点数

psd_nperseg: 2048                 # Welch 方法 nperseg
psd_n_modes: 10                   # 提取的主导模态阶数
psd_min_peak_distance_hz: 0.1     # 峰值最小间距（Hz）

# cable_diameter_map:          # key=inplane_sensor_id，value=外径(m)
#   ST-VIC-C34-101-01: 0.109  # enable_reduced_velocity=true 时必填
#   ST-VIC-C34-102-01: 0.109
#   ST-VIC-C34-103-01: 0.127

enable_psd_modes: true
enable_spectral_features: true
enable_time_stats: true
enable_cross_coupling: true
enable_wind_stats: true
enable_reduced_velocity: false    # 开启时须同时填写 cable_diameter_map

n_workers: 8
split_by_sensor: true
```

关闭某个开关后，对应的所有字段在输出 JSON 中**不存在**（而非为 null），节省存储。

---

## 输出结构

```
results/enriched_stats/
├── class_0_normal/
│   ├── ST-VIC-C34-101-01.json    ← 该传感器的全部 Normal 样本
│   └── ST-VIC-C34-102-01.json
├── class_1_viv/
│   ├── ST-VIC-C34-101-01.json
│   └── ...
├── class_2_rwiv/
└── class_3_transition/
```

若 `split_by_sensor: false`，则每个类别输出单文件 `class_{id}_{label}.json`。

每个文件结构：

```json
{
  "metadata": {
    "class_id": 0,
    "class_label": "normal",
    "class_label_cn": "正常振动",
    "num_samples": 12345,
    "source_result": "res_cnn_full_dataset_20260402_enriched.json",
    "created_at": "2026-04-06 10:00:00"
  },
  "samples": [
    {
      "sample_idx": 42,
      "predicted_class": 0,
      "cable_pair": ["ST-VIC-C34-101-01", "ST-VIC-C34-101-02"],
      "timestamp": [7, 15, 10],
      "window_idx": 3,
      "inplane_sensor_id": "ST-VIC-C34-101-01",
      "outplane_sensor_id": "ST-VIC-C34-101-02",
      "psd_inplane":  {"frequencies": [0.85, 1.70, ...], "powers": [0.023, 0.011, ...]},
      "psd_outplane": {"frequencies": [0.85, 1.70, ...], "powers": [0.019, 0.008, ...]},
      "spectral_inplane": {
        "spectral_entropy": 4.21,
        "spectral_centroid_hz": 1.34,
        "spectral_bandwidth_hz": 0.87,
        "top_modes_energy_ratio": 0.63,
        "dominant_mode_energy_ratio": 0.41
      },
      "time_stats_inplane": {
        "rms": 0.042, "kurtosis": 0.12, "skewness": -0.03,
        "crest_factor": 3.8, "zero_crossing_rate": 0.057
      },
      "cross_coupling": {
        "cross_correlation": 0.31,
        "ellipticity": 0.18,
        "dominant_coherence": 0.74,
        "phase_difference_deg": -32.5
      },
      "wind_stats": [
        {
          "sensor_id": "ST-WIND-001",
          "mean_wind_speed": 8.3, "std_wind_speed": 1.2,
          "mean_wind_direction": 225.0, "std_wind_direction": 12.4,
          "mean_wind_attack_angle": -2.1, "std_wind_attack_angle": 3.5,
          "turbulence_intensity": 0.145
        }
      ]
    }
  ]
}
```

---

## 使用方式

### CLI

```bash
python -m src.identifier.process_full_data.run \
    --result  results/identification_result/res_cnn_full_dataset_20260402_enriched.json \
    --wind    results/metadata/wind_metadata_filtered.json \
    --output  results/enriched_stats \
    --config  config/identifier/process_full_data/default.yaml
```

### Python API

```python
from src.identifier.process_full_data import run
from src.config.identifier.process_full_data.config import ProcessFullDataConfig

cfg = ProcessFullDataConfig(
    enable_reduced_velocity=True,
    cable_diameter_map={
        "ST-VIC-C34-101-01": 0.109,
        "ST-VIC-C34-102-01": 0.109,
        "ST-VIC-C34-103-01": 0.127,
    },
    n_workers=12,
)

run(
    result_path="results/identification_result/res_cnn_full_dataset_20260402_enriched.json",
    wind_metadata_path="results/metadata/wind_metadata_filtered.json",
    output_dir="results/enriched_stats",
    cfg=cfg,
)
```

### 读取结果

```python
import json

with open("results/enriched_stats/class_1_viv/ST-VIC-C34-101-01.json") as f:
    data = json.load(f)

samples = data["samples"]
# 取所有 VIV 样本的谱熵
entropies = [s["spectral_inplane"]["spectral_entropy"] for s in samples]
```

---

## 类别定义

| class_id | label | 中文 |
|----------|-------|------|
| 0 | normal | 正常振动 |
| 1 | viv | 涡激振动 |
| 2 | rwiv | 随机风致振动 |
| 3 | transition | 过渡状态 |

---

## 依赖

- `scipy`：Welch PSD、互谱、峰值检测
- `numpy`：所有数值计算
- `pydantic`：配置校验
- `tqdm`：并行进度显示
- 项目内：`src.data_processer.io_unpacker.UNPACK`、`src.data_processer.preprocess.get_data_wind`
