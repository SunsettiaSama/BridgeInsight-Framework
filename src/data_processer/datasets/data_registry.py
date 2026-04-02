"""
数据集注册表：专用于数据处理模块的Config和Dataset类映射管理
- 独立于深度学习模块的registry
- 直接管理数据集Config和数据集类的注册关系
"""
from typing import Type, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入BaseConfig基类
from src.config.base_config import BaseConfig

# --------------------------
# 数据集Config类注册表：管理数据集配置
# --------------------------

# 1. 测试数据集 - 二分类Config
from src.config.data_processer.datasets.TestDatasets.BinaryClassificationDatasetConfig import BinaryClassificationDatasetConfig

# 2. 测试数据集 - 回归Config
from src.config.data_processer.datasets.TestDatasets.RegressionDatasetConfig import RegressionDatasetConfig

# 3. 标注数据集Config
from src.config.data_processer.datasets.AnnotationDataset.AnnotationDatasetConfig import AnnotationDatasetConfig

# 4. VIV数据集Config
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeSeriesClassificationDatasetConfig import VIVTimeSeriesClassificationDatasetConfig
from src.config.data_processer.datasets.VIV2NumClassification.VIVTimeSeriesClassificationDatasetConfig_2_num_classes import VIVTimeSeriesClassificationDatasetConfig2NumClasses

# 5. StayCable_Vib2023 数据集Config
from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import StayCableVib2023Config

DATASET_CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    # 格式："config_type" → 对应的Config类
    "binary_classification": BinaryClassificationDatasetConfig,
    "regression": RegressionDatasetConfig,
    "annotation": AnnotationDatasetConfig,
    "viv_timeseries_classification": VIVTimeSeriesClassificationDatasetConfig,
    "viv_timeseries_classification_2_num_classes": VIVTimeSeriesClassificationDatasetConfig2NumClasses,
    "staycable_vib2023": StayCableVib2023Config,
}

# --------------------------
# 数据集类注册表：管理数据集实现
# --------------------------

# 1. 测试数据集 - 二分类Dataset
from src.data_processer.datasets.TestDatasets.BinaryClassificationDataset import BinaryClassificationDataset

# 2. 测试数据集 - 回归Dataset
from src.data_processer.datasets.TestDatasets.RegressionDataset import RegressionDataset

# 3. 标注数据集
from src.data_processer.datasets.AnnotationDataset.AnnotationDataset import AnnotationDataset

# 4. VIV数据集
from src.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset import VIVTimeSeriesClassificationDataset
from src.data_processer.datasets.VIV2NumClassification.VIVTimeseriesClassificationDataset_2_num_classes import VIVTimeSeriesClassificationDataset as VIVTimeSeriesClassificationDataset_2Classes

# 5. StayCable_Vib2023 数据集
from src.data_processer.datasets.StayCable_Vib2023.StayCableVib2023Dataset import StayCableVib2023Dataset

DATASET_CLASS_REGISTRY: Dict[str, Type] = {
    # 格式："config_type" → 对应的Dataset类（与DATASET_CONFIG_REGISTRY的key一一对应）
    "binary_classification": BinaryClassificationDataset,
    "regression": RegressionDataset,
    "annotation": AnnotationDataset,
    "viv_timeseries_classification": VIVTimeSeriesClassificationDataset,
    "viv_timeseries_classification_2_num_classes": VIVTimeSeriesClassificationDataset_2Classes,
    "staycable_vib2023": StayCableVib2023Dataset,
}


