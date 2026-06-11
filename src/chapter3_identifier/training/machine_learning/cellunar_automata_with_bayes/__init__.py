"""
元胞自动机+朴素贝叶斯分类模块
包含训练、推理和完整工作流
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from machine_learning_module.cellunar_automata_with_bayes.train import train_ca_nb
from machine_learning_module.cellunar_automata_with_bayes.eval import infer_ca_nb
from machine_learning_module.cellunar_automata_with_bayes.run import CANBWorkflow, run_ca_nb_workflow

__all__ = [
    "train_ca_nb",
    "infer_ca_nb",
    "CANBWorkflow",
    "run_ca_nb_workflow",
]
