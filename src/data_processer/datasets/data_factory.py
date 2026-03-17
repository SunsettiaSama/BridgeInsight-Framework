import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Type
from src.config.base_config import BaseConfig
from .data_registry import DATASET_CONFIG_REGISTRY, DATASET_CLASS_REGISTRY


def get_dataset(config: BaseConfig) -> object:
    """
    通过Config实例创建对应的Dataset实例
    
    流程：
    1. 通过Config实例的类，自动匹配注册表中的config_type
    2. 用config_type从DATASET_CLASS_REGISTRY获取数据集类
    3. 用Config实例初始化数据集
    
    参数：
        config: BaseConfig实例
    
    返回：
        对应的Dataset实例
    
    抛出：
        ValueError: 如果Config类型未注册或无对应Dataset类
    """

    def _get_config_type_by_class(config_class: Type[BaseConfig]) -> str:
        """根据Config类反向找config_type"""
        for typ, cls in DATASET_CONFIG_REGISTRY.items():
            if cls == config_class:
                return typ
        return ""

    # 步骤1：通过Config实例的类，反向查找对应的config_type
    config_class: Type[BaseConfig] = type(config)
    config_type = _get_config_type_by_class(config_class)
    if not config_type:
        raise ValueError(
            f"未找到与配置类「{config_class.__name__}」对应的注册类型。"
            f"已注册的类型：{list(DATASET_CONFIG_REGISTRY.keys())}"
        )

    # 步骤2：从数据集注册表中获取对应的数据集类
    if config_type not in DATASET_CLASS_REGISTRY:
        raise ValueError(
            f"配置类型「{config_type}」未绑定任何数据集类！"
            f"已注册类型：{list(DATASET_CLASS_REGISTRY.keys())}"
        )
    dataset_class = DATASET_CLASS_REGISTRY[config_type]

    # 步骤3：用Config实例初始化数据集（数据集__init__需接收Config实例）
    return dataset_class(config=config)
