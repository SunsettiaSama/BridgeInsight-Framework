"""
SVM 分类模块
包含数据集构建、算法实现和训练流程
"""

from .algorithm import SVMClassifier
from .config import (
    ANNOTATION_RESULTS_PATH,
    LABEL_TO_NAME,
    SELECTED_SENSORS,
)

__all__ = [
    "SVMClassifier",
    "ANNOTATION_RESULTS_PATH",
    "LABEL_TO_NAME",
    "SELECTED_SENSORS",
]
