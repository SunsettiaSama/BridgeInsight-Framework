
# 斜拉索振动特性研究项目

## 项目概述

本项目面向**桥梁斜拉索振动特性的全流程研究**，覆盖从原始多传感器时序数据的采集与预处理、基于深度学习的振动工况识别、多维振动特征提取与统计建模，到论文级图像生成的完整工作链路。

### 核心研究方向

- **振动工况识别**：基于 ResCNN 深度学习模型，对随机振动、涡激振动（VIV）、风雨振（RWIV）等四类工况进行全量自动分类
- **特征统计分析**：提取主频、谱熵、RMS、耦合特征等多维振动统计量，支持分布拟合与 Copula 多变量联合建模
- **风-振耦合分析**：融合多传感器风场数据，计算折减风速与紊流度，研究风荷载与拉索振动的统计耦合规律
- **大规模并行处理**：多进程并行特征计算，支持数十万个振动窗口的全量处理

---

## 项目结构

```
├── src/                                   # 核心源代码
│   ├── data_pipeline.py                   # 主流水线入口（三步全量处理）
│   ├── data_processer/                    # 数据处理基础模块
│   │   ├── io_unpacker.py                 # 低级 I/O 与数据解包
│   │   ├── preprocess/                    # 振动 / 风数据预处理工作流
│   │   ├── datasets/                      # 数据集加载与管理（StayCableVib2023）
│   │   └── signals/                       # 信号处理（小波去噪等）
│   ├── deep_learning_module/              # 深度学习模型（MLP / CNN / ResCNN / LSTM）
│   ├── train_eval/                        # 模型训练与评估框架
│   ├── identifier/                        # 识别器模块
│   │   ├── deeplearning_methods/          # 深度学习全量识别（DLVibrationIdentifier）
│   │   └── process_full_data/             # 识别结果后处理（特征计算 + 按类归档）
│   ├── statistics/                        # 统计分析模块（新增）
│   │   ├── fitting.py                     # 单变量分布 / 曲线拟合
│   │   ├── copula.py                      # 五种 Copula 实现（Gaussian/t/Gumbel/Clayton/Frank）
│   │   └── multivariate.py               # 多变量联合建模全流水线
│   ├── figure_paintings/                  # 论文图像生成
│   │   └── figs_for_thesis/
│   │       ├── config.py                  # 全局字体 / 配色 / 尺寸配置
│   │       ├── Chapter2/                  # 第二章图像
│   │       └── Chapter3/                  # 第三章图像（振动特性分析）
│   ├── machine_learning_module/           # 传统机器学习模块（SVM / 朴素贝叶斯 / CA）
│   ├── visualize_tools/                   # 可视化工具（PlotLib）
│   └── config/                            # 配置类与加载器
│
├── config/                                # YAML 配置文件
│   ├── data_pipeline.yaml                 # 主流水线配置
│   ├── data_processer/                    # 预处理配置
│   ├── identifier/                        # 识别器配置
│   └── train/                             # 训练配置
│
├── docs/                                  # 文档目录
│   ├── README.md                          # 文档总索引
│   ├── identifier/
│   │   ├── process_full_data/README.md    # 识别结果后处理模块详细文档
│   │   └── deeplearning_methods/README.md
│   ├── deep_learning_module/
│   ├── train_eval/
│   ├── data_processer/
│   └── visualize_tools/
│
├── results/                               # 结果输出目录
│   ├── identification_result/             # 全量识别结果 JSON
│   ├── enriched_stats/                    # 按类别归档的特征 JSON
│   │   ├── class_0_normal/
│   │   ├── class_1_viv/
│   │   ├── class_2_rwiv/
│   │   └── class_3_transition/
│   ├── training_result/                   # 模型 checkpoint
│   └── figures/                           # 输出图像
│
└── requirements.txt
```

---

## 完整数据分析流水线

主流水线由 `src/data_pipeline.py` 统一编排，三个步骤均可通过 `config/data_pipeline.yaml` 独立开关：

```
┌─────────────────────────────────────────────────────────────┐
│  步骤 1  preprocess                                          │
│  振动 + 风数据预处理 → 元数据 JSON                           │
│  缺失率筛选 / RMS统计 / 极端振动标记                         │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤 2  identification                                      │
│  ResCNN 全量振动识别 → 预测结果 JSON                         │
│  四类振动工况：随机振动 / VIV / RWIV / 过渡状态              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤 3  feature_analysis                                    │
│  多进程特征计算 → enriched_stats（按类别×传感器归档）         │
│  PSD主频 / 谱熵 / RMS / 耦合特征 / 风场统计 / 折减风速       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤 4  统计建模（src/statistics/）                         │
│  单变量分布拟合 / 相关性分析 / Copula 多变量联合建模          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤 5  论文图像（src/figure_paintings/）                   │
│  各章节图像独立生成，支持 PlotLib 交互预览与保存             │
└─────────────────────────────────────────────────────────────┘
```

### 运行主流水线

```bash
# 使用默认配置
python src/data_pipeline.py

# 指定配置文件
python src/data_pipeline.py --config config/data_pipeline.yaml
```

```yaml
# config/data_pipeline.yaml 步骤开关示例
steps:
  preprocess:       false   # 已有元数据时可跳过
  identification:   true    # ResCNN 全量推理
  feature_analysis: true    # 特征计算与归档
```

---

## 模块说明

### 1. 数据处理 (`data_processer`)

负责从原始振动/风数据文件到结构化元数据的全流程预处理。

- 振动数据：缺失率筛选 → RMS统计 → 极端振动标记
- 风数据：多传感器时序对齐 → 统计量聚合
- 数据集：`StayCableVib2023Dataset` — 支持窗口化采样、去噪、指纹校验

文档：[docs/data_processer/README.md](docs/data_processer/README.md)

---

### 2. 深度学习识别 (`deep_learning_module` + `identifier`)

基于 ResCNN 的振动工况四分类识别框架。

| 类别 | label | 说明 |
|------|-------|------|
| 0 | normal | 随机振动 |
| 1 | viv | 涡激振动 |
| 2 | rwiv | 风雨振 |
| 3 | transition | 过渡状态 |

- **训练**：`src/train_eval/` — 支持早停、学习率调度、断点续训
- **推理**：`src/identifier/deeplearning_methods/` — `DLVibrationIdentifier` 全量批量推理
- **后处理**：`src/identifier/process_full_data/` — 多进程特征计算，按类别×传感器归档

文档：[docs/identifier/process_full_data/README.md](docs/identifier/process_full_data/README.md)

---

### 3. 特征后处理 (`identifier/process_full_data`)

对识别结果 JSON 并行计算振动特征，按类别归档保存（`results/enriched_stats/`）。

输出字段（可通过配置开关控制）：

| 特征组 | 字段示例 |
|--------|---------|
| **频域** | `psd_inplane.frequencies`、`spectral_inplane.dominant_mode_energy_ratio` |
| **时域** | `time_stats_inplane.rms`、`kurtosis`、`crest_factor` |
| **耦合** | `cross_coupling.ellipticity`、`dominant_coherence` |
| **风场** | `wind_stats[i].mean_wind_speed`、`turbulence_intensity` |
| **折减风速** | `reduced_velocity[i].reduced_velocity` |

输出 metadata 现包含完整溯源路径：

```json
{
  "metadata": {
    "source_result": "res_cnn_full_dataset_20260402_enriched.json",
    "source_result_path": "/full/path/to/...json",
    "wind_metadata_path": "/full/path/to/wind_metadata.json",
    "created_at": "2026-04-07 10:00:00"
  }
}
```

---

### 4. 统计分析 (`statistics`) — 新增

提供从单变量分布拟合到多变量 Copula 联合建模的完整统计工具链。

#### `fitting.py` — 单变量拟合

```python
from src.statistics import fit

# 分布拟合：对 VIV 样本 RMS 拟合 Weibull 分布
result = fit(rms_values, form="weibull_min")
print(result)   # 参数 + KS检验 + AIC/BIC

# 曲线拟合：RMS 随平均风速的幂函数关系
result = fit(rms_values, form="power", x=wind_speed_values)
print(result.params)     # {'a': ..., 'b': ...}
print(result.r_squared)

# 自定义函数
result = fit(y, form="custom", x=x,
             func=lambda x, a, b, c: a * np.exp(-b * x) + c,
             param_names=["a", "b", "c"])
```

内置分布：所有 `scipy.stats` 连续分布（`norm`、`lognorm`、`weibull_min`、`gamma`、`beta` 等）

内置曲线形式：`linear` / `power` / `exponential` / `logarithmic` / `quadratic` / `cubic` / `sine`

#### `copula.py` + `multivariate.py` — 多变量 Copula 建模

```python
from src.statistics import fit_multivariate, compare_copulas, sample_from_multivariate

# 构造数据矩阵：前 k 阶主频 + 主频能量
# data.shape = (n_samples, 2k)
result = fit_multivariate(
    data,
    variable_names=["freq_1", "freq_2", "energy_1", "energy_2"],
    marginal_distributions=["lognorm", "lognorm", "beta", "beta"],
    copula_type="gaussian",
)
print(result.summary())

# 模型选择：比较所有适合当前维度的 Copula，按 AIC 升序
compare_copulas(result.u_matrix)

# 从拟合的联合分布采样
x_new, u_new = sample_from_multivariate(result, n_samples=5000)
```

支持的 Copula 类型：

| 类型 | 适用维度 | 参数估计 | 尾部特征 |
|------|---------|---------|---------|
| `gaussian` | d ≥ 2 | Spearman → van der Waerden | 对称、无尾依赖 |
| `t` | d ≥ 2 | Kendall + ν 极大似然 | 对称重尾 |
| `gumbel` | d = 2 | Kendall + MLE | 上尾依赖 |
| `clayton` | d = 2 | Kendall + MLE | 下尾依赖 |
| `frank` | d = 2 | Kendall + MLE | 对称，允许负相关 |

---

### 5. 论文图像 (`figure_paintings`)

按章节组织，每张图独立脚本，统一使用 `PlotLib` 交互预览。

**第三章图像（振动特性分析）**：

| 脚本 | 内容 |
|------|------|
| `fig3_2_all_data_display.py` | 全年四类振动占比饼图 |
| `fig3_3_normal_vib_timeseries.py` | 随机振动时序展示 |
| `fig3_4_normal_vib_mean.py` | 随机振动 RMS 分布（主体 / 尾部 / 散点） |
| `fig3_5_normal_vib_trajectory.py` | 振动轨迹云图（面内 vs 面外散点） |
| `fig3_6_normal_vib_modal.py` | 主频分布 / 主频-能量散点 / 主频能量分布 |
| `fig3_7_normal_vib_modal.py` | 主频深度分析 |

---

### 6. 可视化工具 (`visualize_tools`)

`PlotLib` 统一绘图库，支持多图管理、交互式预览与批量保存。

```python
from src.visualize_tools.utils import PlotLib

lib = PlotLib()
lib.figs.append(fig1)
lib.figs.append(fig2)
lib.show()   # 交互式窗口，支持键盘切换图像
```

文档：[docs/visualize_tools/README.md](docs/visualize_tools/README.md)

---

### 7. 传统机器学习 (`machine_learning_module`)

早期分类方法，现已由深度学习识别器接替主要推理任务，保留作对比基线。

- SVM（rbf / linear / poly 核）
- 朴素贝叶斯（GaussianNB）
- 元胞自动机（CA，Rule 30 进化 + 模板匹配）
- CA + 朴素贝叶斯混合模型

文档：[docs/machine_learning_module/README.md](docs/machine_learning_module/README.md)

---

## 文档导航

| 文档 | 路径 | 说明 |
|------|------|------|
| 文档总索引 | [docs/README.md](docs/README.md) | 所有子模块文档入口 |
| 识别结果后处理 | [docs/identifier/process_full_data/README.md](docs/identifier/process_full_data/README.md) | 特征字段说明、配置参考、输出结构 |
| 深度学习识别器 | [docs/identifier/deeplearning_methods/README.md](docs/identifier/deeplearning_methods/README.md) | 推理接口与配置 |
| 深度学习模块 | [docs/deep_learning_module/README.md](docs/deep_learning_module/README.md) | 模型架构与训练配置 |
| 训练评估 | [docs/train_eval/README.md](docs/train_eval/README.md) | 训练流程与评估指标 |
| 数据处理 | [docs/data_processer/README.md](docs/data_processer/README.md) | 预处理与数据集 |
| 可视化工具 | [docs/visualize_tools/README.md](docs/visualize_tools/README.md) | PlotLib 参考 |

---

## 关键参数

| 参数 | 值 |
|------|----|
| 振动采样频率 | 50 Hz |
| 单窗口长度 | 3000 点（60 秒） |
| PSD 计算方法 | Welch，nperseg=2048 |
| 提取主导模态阶数 | 10 |
| 振动分类数 | 4 类 |
| 识别模型 | ResCNN（val_acc=99.20%） |

---

## 开发工具

- **语言**：Python 3.10+
- **主要库**：numpy、scipy、torch、matplotlib、pydantic、tqdm
- **配置管理**：YAML + Pydantic 配置类
- **并行处理**：`multiprocessing.Pool`
- **数据格式**：JSON（元数据与特征）、`.mat` / 二进制（原始振动）

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 2026-04-07 | 新增 `src/statistics/`：单变量拟合（`fitting.py`）、五种 Copula（`copula.py`）、多变量联合建模流水线（`multivariate.py`） |
| 2026-04-07 | `identifier/process_full_data` 输出 metadata 补充 `source_result_path` 与 `wind_metadata_path` 完整溯源路径 |
| 2026-04-07 | 新增第三章论文图像：`fig3_6_normal_vib_modal.py`（主频分布、主频-能量散点、主频能量分布） |
| 2026-04-06 | `identifier/process_full_data` 完成：多进程特征计算、按类别×传感器归档 |
| 2026-04-02 | ResCNN 全量识别完成（val_acc=99.20%），识别结果归档 |
| 2026-03-09 | 主流水线 `data_pipeline.py` 上线，统一编排三步处理流程 |
