
# NN/trainers/trainer_manager.py
from typing import Type
from ..configs.base_config import BaseConfig
from ..configs.registry import CONFIG_CLASS_REGISTRY, TRAINER_CLASS_REGISTRY  # 依赖统一注册表
# 假设你有训练器基类，可根据实际路径调整导入
from .base_trainer import BaseTrainer  

def get_trainer(config: BaseConfig) -> BaseTrainer:
    """
    完全基于Config实例创建训练器：
    1. 通过Config实例的类，自动找到对应的config_type；
    2. 用config_type从注册表中获取训练器类；
    3. 用Config实例初始化训练器（训练器__init__需接收Config实例）。
    """
    @staticmethod
    def _get_config_type_by_class(config_class: Type[BaseConfig]) -> str:
        """根据Config类，反向查找注册表中的config_type"""
        for typ, cls in CONFIG_CLASS_REGISTRY.items():
            if cls == config_class:
                return typ
        return ""

    # 步骤1：通过Config实例的类，匹配注册表中的config_type
    config_class: Type[BaseConfig] = type(config)
    config_type = _get_config_type_by_class(config_class)
    if not config_type:
        raise ValueError(f"未找到与配置类「{config_class.__name__}」对应的注册类型")

    # 步骤2：从注册表中获取对应的训练器类
    if config_type not in TRAINER_CLASS_REGISTRY:
        raise ValueError(f"配置类型「{config_type}」未绑定任何训练器类，请先在注册表中注册")
    trainer_class = TRAINER_CLASS_REGISTRY[config_type]

    # 步骤3：用Config实例初始化训练器（训练器__init__需接收Config实例）
    return trainer_class(config=config)

