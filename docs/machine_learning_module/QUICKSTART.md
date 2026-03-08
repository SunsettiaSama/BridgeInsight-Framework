# 机器学习模块快速入门

## ⚡ 5分钟快速开始

### 最小化示例（SVM）

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import Dataset, DataLoader
import torch
from machine_learning_module.svm.run import run_svm_workflow

class SimpleDataset(Dataset):
    def __init__(self, size=500, num_classes=3, feature_dim=20):
        self.size, self.num_classes, self.feature_dim = size, num_classes, feature_dim
    def __len__(self): return self.size
    def __getitem__(self, idx):
        return torch.randn(self.feature_dim), torch.randint(0, self.num_classes, (1,)).item()

train_loader = DataLoader(SimpleDataset(500), batch_size=32, shuffle=True)
val_loader = DataLoader(SimpleDataset(100), batch_size=32, shuffle=False)
infer_loader = DataLoader(SimpleDataset(150), batch_size=32, shuffle=False)

results = run_svm_workflow(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    infer_dataloader=infer_loader,
    infer_has_label=True
)

print("训练准确率:", results['train']['train_metrics']['accuracy'])
print("推理准确率:", results['eval']['metrics']['accuracy'])
```

---

## 📚 常见场景

### 场景1: 仅训练不保存

```python
from config.machine_learning_module.svm.workflow_config import SVMWorkflowConfig
from machine_learning_module.svm.run import SVMWorkflow

config = SVMWorkflowConfig()
config.TRAIN_CONFIG = {'model_save_path': None, 'result_save_path': None}
config.ENABLE_EVAL = False

workflow = SVMWorkflow(config)
results = workflow.run(train_dataloader=train_loader, val_dataloader=val_loader)
```

### 场景2: 仅推理（加载已有模型）

```python
config = SVMWorkflowConfig()
config.ENABLE_TRAIN = False

workflow = SVMWorkflow(config)
results = workflow.run(infer_dataloader=infer_loader, infer_has_label=True)
```

### 场景3: 自定义保存路径

```python
config = SVMWorkflowConfig()
config.TRAIN_CONFIG = {
    'model_save_path': 'output/svm_model.pkl',
    'result_save_path': 'output/svm_train.json'
}
config.EVAL_CONFIG = {
    'model_load_path': 'output/svm_model.pkl',
    'result_path': 'output/svm_infer.json'
}

workflow = SVMWorkflow(config)
results = workflow.run(...)
```

### 场景4: 与 VIV 数据集结合

```python
from data_processer.datasets.data_factory import get_dataset
from config.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import (
    VIVTimeSeriesClassificationDatasetConfig
)
from torch.utils.data import DataLoader
from machine_learning_module.naive_bayes.run import run_naive_bayes_workflow

config = VIVTimeSeriesClassificationDatasetConfig(
    data_dir="./data/viv",
    batch_size=32,
    fix_seq_len=1000
)
dataset = get_dataset(config)
train_dataset = dataset.get_train_dataset()
val_dataset = dataset.get_val_dataset()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

results = run_naive_bayes_workflow(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    infer_dataloader=val_loader,
    infer_has_label=True
)
```

---

## 🎯 模块速查表

| 模块 | 导入 | 便捷函数 |
|------|------|----------|
| SVM | `from machine_learning_module.svm.run import run_svm_workflow` | `run_svm_workflow(...)` |
| Naive Bayes | `from machine_learning_module.naive_bayes.run import run_naive_bayes_workflow` | `run_naive_bayes_workflow(...)` |
| CA | `from machine_learning_module.cellunar_automata.run import run_ca_workflow` | `run_ca_workflow(...)` |
| CA+Bayes | `from machine_learning_module.cellunar_automata_with_bayes.run import run_ca_nb_workflow` | `run_ca_nb_workflow(...)` |

---

## 📖 进一步阅读

- 详细文档: [机器学习模块文档](README.md)
- 数据集文档: [数据集模块文档](../data_processer/datasets/README.md)
