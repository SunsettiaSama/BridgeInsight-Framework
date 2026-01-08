

# configs/registry.py
"""
注册表：统一管理Config类、模型类、数据集类的映射关系
- 新增模型/数据集/训练器时，只需在对应子注册表中添加「config_type: 类」的映射
- 拆分Config类注册表为子模块，解决耦合问题，保留原有聚合注册表保证兼容性
"""
from typing import Type, Dict

# 导入BaseConfig基类（确保所有Config类都继承它）
from .base_config import BaseConfig

# --------------------------
# 子注册表：按类型拆分Config类，解耦管理（核心拆分逻辑）
# --------------------------
# 1. 模型Config类注册表：专门管理模型相关Config（独立模块，不与其他类型耦合）
from .models.SimpleCNN import SimpleCNNConfig
from .models.SimpleMLPConfig import SimpleMLPConfig
from .models.LSTMConfig import LSTMConfig

MODEL_CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    # 格式："config标识" → 对应的模型Config类
    "simple_cnn": SimpleCNNConfig,
    "mlp": SimpleMLPConfig, 
    "lstm": LSTMConfig,
    # 预留拓展位：新增模型Config时，仅在此处添加即可，不影响其他注册表
    # "efficient_vit": EfficientViTConfig,
    # "resnet50": ResNet50Config,
    # "llama_sft": LLaMASFTConfig,
}

# 2. 数据集Config类注册表：专门管理数据集相关Config（独立模块）
from .datasets.VIVTimeseriesClassificationDataset import VIVTimeSeriesClassificationDatasetConfig
from .datasets.VIVTimeseriesClassificationDataset_2_num_classes import VIVTimeSeriesClassificationDatasetConfig as VIVTimeSeriesClassificationDatasetConfig_2_num_classes

DATASET_CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    # 格式："config标识" → 对应的数据集Config类
    "viv_timeseries_classification": VIVTimeSeriesClassificationDatasetConfig,
    "viv_timeseries_classification_2_num_classes": VIVTimeSeriesClassificationDatasetConfig_2_num_classes,
    # 预留拓展位：新增数据集Config时，仅在此处添加即可
    # "viv_img_dataset": VIVImgDatasetConfig,
    # "industrial_vibration": IndustrialVibrationDatasetConfig,
}

from .trainer.sft import SFTTrainerConfig

# 3. 训练器Config类注册表：预留训练器Config管理（后续扩展SFT/RLHF等训练器时使用）
# 先创建空注册表，后续添加TrainerConfig类时直接补充，无需改动其他部分
TRAINER_CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    # 预留拓展位：新增训练器Config时，仅在此处添加即可
    "sft": SFTTrainerConfig,
    # "rlhf_trainer": RLHFSpecificConfig,
    # "lora_trainer": LoRASpecificConfig,
}

# --------------------------
# 原有聚合Config类注册表：维持变量名不变，保证兼容性（无需修改原有调用代码）
# 自动合并所有子Config注册表，既解耦又兼容历史逻辑
# --------------------------
CONFIG_CLASS_REGISTRY: Dict[str, Type[BaseConfig]] = {}
# 合并模型Config注册表
CONFIG_CLASS_REGISTRY.update(MODEL_CONFIG_REGISTRY)
# 合并数据集Config注册表
CONFIG_CLASS_REGISTRY.update(DATASET_CONFIG_REGISTRY)
# 合并训练器Config注册表（后续添加训练器Config后自动生效）
CONFIG_CLASS_REGISTRY.update(TRAINER_CONFIG_REGISTRY)

# --------------------------
# 2. 模型类注册表：config_type → 对应的模型类（用于初始化模型）
# --------------------------
# 导入现有模型类（你的SimpleCNN）
from ..models.SimpleCNN import SimpleCNN
from ..models.SimpleMLP import SimpleMLP
from ..models.LSTM import LSTMModel

MODEL_CLASS_REGISTRY: Dict[str, Type] = {
    # 格式："config标识" → 对应的模型类（和MODEL_CONFIG_REGISTRY的key一一对应）
    "simple_cnn": SimpleCNN,
    "mlp": SimpleMLP, 
    "lstm": LSTMModel
    # 预留拓展位：新增模型时，仅在此处添加即可
    # "efficient_vit": EfficientViT,
    # "resnet50": ResNet50,
    # "llama_sft": LLaMASFTModel,
}

# --------------------------
# 3. 数据集类注册表（预留，后续拓展数据集用）
# --------------------------
from ..datasets.VIVTimeseriesClassificationDataset import VIVTimeSeriesClassificationDataset
from ..datasets.VIVTimeseriesClassificationDataset_2_num_classes import VIVTimeSeriesClassificationDataset as VIVTimeSeriesClassificationDataset_2Classes

DATASET_CLASS_REGISTRY: Dict[str, Type] = {
    # 格式："config标识" → 对应的数据集类（和DATASET_CONFIG_REGISTRY的key一一对应）
    "viv_timeseries_classification": VIVTimeSeriesClassificationDataset,
    "viv_timeseries_classification_2_num_classes": VIVTimeSeriesClassificationDataset_2Classes,

    # 预留拓展位：新增数据集时，仅在此处添加即可
    # "viv_img_dataset": VIVImgDataset,
    # "industrial_vibration": IndustrialVibrationDataset,
}

from ..trainer.sft import SFTTrainer

# --------------------------
# 4. 训练器类注册表：预留训练器管理（后续扩展SFT/RLHF等训练器时使用）
# --------------------------
TRAINER_CLASS_REGISTRY: Dict[str, Type] = {
    # 预留拓展位：新增训练器时，仅在此处添加即可，与TRAINER_CONFIG_REGISTRY的key一一对应
    "sft": SFTTrainer,
    # "rlhf_trainer": RLHFTrainer,
    # "lora_trainer": LoRATrainer,
}

