"""
SVM 分类模块
包含数据集构建、算法实现和训练流程
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from machine_learning_module.svm.run import SVMWorkflow, run_svm_workflow
from machine_learning_module.svm.train import train_svm
from machine_learning_module.svm.eval import infer_svm

__all__ = [
    "SVMWorkflow",
    "run_svm_workflow",
    "train_svm",
    "infer_svm",
]
