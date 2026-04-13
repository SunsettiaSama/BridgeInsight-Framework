# statistics — 振动模态统计分析模块

## 概述

该模块对 `enriched_stats` 中的正常振动样本进行频域模态统计分析，主要包含两项任务：

1. **边缘分布拟合**：提取每个样本前 N 阶主导模态的频率和能量占比，为每个变量独立拟合最优边缘概率密度分布（AIC 选优）
2. **相关性分析**：计算所有模态变量两两之间的 Pearson / Spearman / Kendall 相关系数矩阵

**输入** → `results/enriched_stats/class_0_normal/*.json`（PSD 前 N 阶模态数据）  
**输出** → `results/statistics/normal_vib_mode_analysis.json`（拟合参数 + 相关矩阵）

---

## 目录结构

```
src/statistics/
├── __init__.py         公共 API 导出
├── fitting.py          边缘分布拟合：FitResult / fit_distribution / fit_curve / fit
├── copula.py           Copula 拟合：Gaussian / t / Gumbel / Clayton / Frank
├── multivariate.py     多元分析：correlation_analysis / fit_multivariate
└── run.py              主流水线（CLI 入口 + Python API）

src/config/statistics/
├── __init__.py
└── config.py           StatisticsConfig（Pydantic）+ load_config()

config/statistics/
└── default.yaml        默认配置（所有参数的预设值）

results/statistics/
└── normal_vib_mode_analysis.json   分析结果（自动生成）
```

---

## 变量体系

每次运行提取 `4 × N_MODES` 个变量（默认 N_MODES=8，共 32 个）：

| 变量名 | 方向 | 阶序 | 含义 |
|--------|------|------|------|
| `freq_in_{k}` | 面内 | 1–N | 第 k 主导模态频率（Hz），按功率降序排名 |
| `energy_in_{k}` | 面内 | 1–N | 第 k 主导模态能量占比（/ 前N阶峰值总能量） |
| `freq_out_{k}` | 面外 | 1–N | 同上 |
| `energy_out_{k}` | 面外 | 1–N | 同上 |

> **能量占比定义**：`energy_k = power_k / Σ(power_1..N)`，即该阶峰值功率占前 N 阶存储峰值功率之和的比例，第 1 阶（最主导）占比最高，累积到第 N 阶等于 1。

---

## 边缘分布拟合

### 候选分布

| 变量类型 | 候选分布 | 约束 |
|----------|----------|------|
| 主频（Hz）| `gamma`, `lognorm` | 正值，`floc=0` |
| 能量占比 | `gamma`, `lognorm` | 正值，`floc=0` |
| 任意变量（GMM）| `gmm_2`, `gmm_3`, … | 高斯混合，无约束 |

所有单峰候选均由 `scipy.stats` 提供，经 MLE 拟合；GMM 由 `sklearn.mixture.GaussianMixture` 拟合（`n_init=8`，保证全局收敛）。所有形式统一按 **AIC 最小**选优。

### 多峰分布（GMM）

当 `enable_gmm: true` 时，每个变量在单峰候选之外额外尝试 2 … `gmm_max_components` 阶高斯混合模型，AIC 同台竞选。能量占比等出现双峰的变量会自动选择 GMM。

GMM 自由度计算：`k = (n_components − 1) + n_components + n_components`  
（权重 + 均值 + 方差，不含 loc/scale 冗余项）

### 输出字段（`marginals`）

单峰分布示例：

```json
"freq_in_1": {
  "n_valid": 98765,
  "best": {
    "form": "lognorm",
    "params": {"s": 0.31, "loc": 0.0, "scale": 2.87},
    "ks_statistic": 0.0043,
    "ks_pvalue": 0.712,
    "aic": -345678.9,
    "bic": -345660.2,
    "n_valid": 98765
  },
  "candidates": [
    {"form": "lognorm", "aic": -345678.9, ...},
    {"form": "gamma",   "aic": -344100.3, ...},
    {"form": "gmm_2",   "aic": -346200.0, ...}
  ]
}
```

GMM 双峰分布示例：

```json
"energy_in_1": {
  "n_valid": 98765,
  "best": {
    "form": "gmm_2",
    "params": {
      "n_components": 2,
      "weights":   [0.58, 0.42],
      "means":     [0.031, 0.085],
      "variances": [0.00012, 0.00031]
    },
    "aic": -412300.5,
    "bic": -412265.1,
    "n_valid": 98765
  },
  "candidates": [...]
}
```

> `ks_statistic` / `ks_pvalue` 仅对单峰 scipy 分布有效；GMM 暂不计算 KS 统计量（字段缺失）。

---

## 相关性分析

### 计算方法

| 指标 | 特点 | 适用场景 |
|------|------|----------|
| **Pearson ρ** | 线性相关，对正态假设敏感 | 近线性关系 |
| **Spearman ρ** | 秩相关，对单调关系鲁棒 | 非线性单调关系 |
| **Kendall τ** | 秩相关，更鲁棒但计算慢 | 小样本或噪声较多时 |

### 抽样策略

当完整样本数（所有 32 列均非 NaN）超过 `corr_max_n`（默认 8000）时，随机抽样该数量的行进行相关性计算，以避免 Kendall τ 逐对计算时间过长。

### 输出字段（`correlation`）

```json
"correlation": {
  "n_samples": 8000,
  "variable_names": ["freq_in_1", "energy_in_1", ...],
  "pearson":  [[1.0, ...], ...],
  "spearman": [[1.0, ...], ...],
  "kendall":  [[1.0, ...], ...]
}
```

---

## 配置参考

配置类位于 `src/config/statistics/config.py`，通过 YAML 文件加载。

```yaml
# config/statistics/default.yaml

class_label: "class_0_normal"   # enriched_stats 子目录名

n_modes: 8                      # 前 N 阶主导模态（≤10）
min_valid_samples: 30           # 单变量最少有效样本数

candidate_dists_freq:           # 主频候选分布
  - gamma
  - lognorm

candidate_dists_energy:         # 能量占比候选分布
  - gamma
  - lognorm

enable_gmm: true                # 是否将 GMM 纳入 AIC 竞选
gmm_max_components: 2           # 最大分量数（2 = 仅尝试双峰）

corr_max_n: 8000                # 相关性分析最大样本数
corr_rng_seed: 42               # 抽样随机种子

output_subdir: "statistics"
output_filename: "normal_vib_mode_analysis.json"
```

---

## 输出结构

```
results/statistics/normal_vib_mode_analysis.json
├── metadata
│   ├── class              振动类别标签
│   ├── n_modes            分析阶数
│   ├── n_samples_total    JSON 中的总样本数
│   ├── n_samples_valid    成功提取的有效样本数
│   ├── variable_names     所有变量名列表（32个）
│   ├── config             运行时使用的配置快照
│   └── created_at         生成时间
├── marginals              每个变量的边缘分布拟合结果
└── correlation            32×32 三种相关系数矩阵
```

---

## 使用方式

### CLI

```bash
python -m src.statistics.run
```

使用自定义配置：

```bash
python -c "
from src.config.statistics.config import load_config
from src.statistics.run import run
cfg = load_config('config/statistics/default.yaml')
cfg.n_modes = 6
run(cfg)
"
```

### Python API

```python
from src.statistics import run
from src.config.statistics.config import StatisticsConfig

# 使用默认配置
run()

# 使用自定义配置
cfg = StatisticsConfig(
    n_modes=6,
    corr_max_n=5000,
    candidate_dists_freq=["lognorm", "gamma"],
)
run(cfg=cfg)
```

### 读取结果

```python
import json, numpy as np

with open("results/statistics/normal_vib_mode_analysis.json") as f:
    result = json.load(f)

# 读取第1阶面内主频的最优拟合分布
m = result["marginals"]["freq_in_1"]["best"]
print(m["form"], m["params"])   # e.g. lognorm {'s': 0.31, 'loc': 0.0, 'scale': 2.87}

# 读取 Spearman 相关矩阵
spearman = np.array(result["correlation"]["spearman"])
var_names = result["correlation"]["variable_names"]
```

---

## 依赖

- `scipy`：连续分布拟合（`scipy.stats`）
- `sklearn`：高斯混合模型（`sklearn.mixture.GaussianMixture`）
- `numpy`：数值计算
- `pydantic`：配置校验（通过 `src.config.base_config.BaseConfig`）
- `yaml`：配置文件解析
- 项目内：`src.statistics.fitting`、`src.statistics.multivariate`
