# 机器学习分类模块文档

## 📋 模块概述

`src/machine_learning_module/` 提供多种传统机器学习分类算法的工作流，支持训练、推理及完整流程编排。所有模型与结果默认保存至 `results/classification_results/machine_learning/` 目录下各子模块对应路径。

### 核心特性

- **统一工作流接口**：SVM、朴素贝叶斯、元胞自动机(CA)、CA+朴素贝叶斯均提供 `Workflow` 类与 `run_xxx_workflow()` 便捷函数
- **保存路径开关**：路径为 `None` 时不保存，有值则保存至指定路径
- **配置驱动**：通过 `TRAIN_CONFIG`、`EVAL_CONFIG` 控制模型路径、结果路径等
- **DataLoader 兼容**：输入为 PyTorch DataLoader，与数据集模块无缝对接

---

## 📂 目录结构

```
src/machine_learning_module/
├── svm/                              # SVM 支持向量机
│   ├── train.py
│   ├── eval.py
│   └── run.py
├── naive_bayes/                      # 朴素贝叶斯
│   ├── train.py
│   ├── eval.py
│   └── run.py
├── cellunar_automata/                # 纯元胞自动机（模板匹配）
│   ├── train.py
│   ├── eval.py
│   └── run.py
└── cellunar_automata_with_bayes/      # 元胞自动机 + 朴素贝叶斯
    ├── train.py
    ├── eval.py
    └── run.py

results/classification_results/machine_learning/
├── svm/                              # SVM 模型与结果
├── naive_bayes/                      # 朴素贝叶斯模型与结果
├── ca/                               # 纯 CA 模板与结果
└── ca_bayes/                         # CA+NB 模型与结果
```

---

## 🚀 快速开始

### 1. SVM 工作流

```python
from torch.utils.data import DataLoader
from machine_learning_module.svm.run import run_svm_workflow

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
infer_dataloader = DataLoader(infer_dataset, batch_size=32, shuffle=False)

results = run_svm_workflow(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    infer_dataloader=infer_dataloader,
    infer_has_label=True
)

print(results['train']['train_metrics']['accuracy'])
print(results['eval']['metrics']['accuracy'])
```

### 2. 朴素贝叶斯工作流

```python
from machine_learning_module.naive_bayes.run import run_naive_bayes_workflow

results = run_naive_bayes_workflow(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    infer_dataloader=infer_dataloader,
    infer_has_label=True
)
```

### 3. 元胞自动机工作流

```python
from machine_learning_module.cellunar_automata.run import run_ca_workflow

results = run_ca_workflow(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    infer_dataloader=infer_dataloader,
    infer_has_label=True
)
```

### 4. 元胞自动机 + 朴素贝叶斯工作流

```python
from machine_learning_module.cellunar_automata_with_bayes.run import run_ca_nb_workflow

results = run_ca_nb_workflow(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    infer_dataloader=infer_dataloader,
    infer_has_label=True
)
```

---

## ⚙️ 保存路径配置

### 默认保存位置

| 模块 | 模型/模板路径 | 训练结果 | 推理结果 |
|------|---------------|----------|----------|
| SVM | `results/.../svm/svm_model.pkl` | `svm_train_result.json` | `svm_infer_result.json` |
| Naive Bayes | `results/.../naive_bayes/nb_model.pkl` | `nb_train_result.json` | `nb_infer_result.json` |
| CA | `results/.../ca/ca_class_templates.pkl` | `ca_e2e_train_result.json` | `ca_e2e_infer_result.json` |
| CA+Bayes | `results/.../ca_bayes/ca_params.pkl` + `ca_nb_model.pkl` | `ca_nb_train_result.json` | `ca_nb_infer_result.json` |

### 不保存结果

将对应路径设为 `None` 即可禁用保存：

```python
# SVM / Naive Bayes
from config.machine_learning_module.svm.workflow_config import SVMWorkflowConfig

config = SVMWorkflowConfig()
config.TRAIN_CONFIG = {'model_save_path': None, 'result_save_path': None}
config.EVAL_CONFIG = {'result_path': None}

from machine_learning_module.svm.run import SVMWorkflow
workflow = SVMWorkflow(config)
results = workflow.run(...)
```

```python
# CA / CA+Bayes（使用 dict 配置）
config = {
    'TRAIN_CONFIG': {'template_save_path': None, 'result_save_path': None},
    'EVAL_CONFIG': {'result_path': None}
}
results = run_ca_workflow(..., config=config)
```

### 自定义保存路径

```python
config = SVMWorkflowConfig()
config.TRAIN_CONFIG = {
    'model_save_path': './my_models/svm.pkl',
    'result_save_path': './my_results/svm_train.json'
}
config.EVAL_CONFIG = {
    'model_load_path': './my_models/svm.pkl',
    'result_path': './my_results/svm_infer.json'
}
```

---

## 📖 各模块说明

### SVM

- **算法**：sklearn SVC，支持 rbf/linear/poly 核
- **输入**：DataLoader 返回 `(data, label)`，`data` 展平为一维特征
- **输出**：训练/推理指标、预测标签与置信度

### 朴素贝叶斯

- **算法**：sklearn GaussianNB
- **输入**：同 SVM
- **输出**：训练/推理指标、预测标签与置信度

### 元胞自动机 (CA)

- **算法**：一维 CA Rule 30 演化 + 类别模板匹配
- **输入**：同 SVM
- **输出**：训练生成类别模板，推理通过最近邻匹配分类

### 元胞自动机 + 朴素贝叶斯 (CA+Bayes)

- **算法**：CA 提取特征 → 标准化 → GaussianNB 分类
- **输入**：同 SVM
- **输出**：训练/推理指标、预测标签与置信度

---

## 🧪 测试

测试脚本位于 `src/test/machine_learning_module/`：

```bash
# 从项目根目录运行
python src/test/machine_learning_module/svm_workflow_test.py
python src/test/machine_learning_module/naive_bayes_workflow_test.py
python src/test/machine_learning_module/ca_workflow_test.py
python src/test/machine_learning_module/ca_nb_workflow_test.py
```

---

## 🔗 相关链接

- [数据集模块文档](../data_processer/datasets/README.md)
- [数据预处理文档](../data_processer/README.md)

---

## 📞 维护信息

- **最后更新**: 2026年3月8日
- **依赖库**: torch, sklearn, numpy, scipy
