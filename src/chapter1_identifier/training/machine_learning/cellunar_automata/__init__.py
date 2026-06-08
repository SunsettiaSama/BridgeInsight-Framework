"""
元胞自动机分类模块
包含训练、推理和完整工作流
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from machine_learning_module.cellunar_automata.train import train_ca_classifier
from machine_learning_module.cellunar_automata.eval import infer_ca_classifier
from machine_learning_module.cellunar_automata.run import CAWorkflow, run_ca_workflow

__all__ = [
    "train_ca_classifier",
    "infer_ca_classifier",
    "CAWorkflow",
    "run_ca_workflow",
]
