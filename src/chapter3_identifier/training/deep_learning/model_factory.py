"""
模型工厂：通过Config实例创建对应的模型实例
- 仿照数据集工厂的设计模式
- 支持通过Config类型自动匹配模型类型
"""
import sys
from pathlib import Path
from typing import Type

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base_config import BaseConfig
from .model_registry import MODEL_CONFIG_REGISTRY, MODEL_CLASS_REGISTRY


def get_model(config: BaseConfig):
    """
    通过Config实例创建对应的模型实例
    
    流程：
    1. 通过Config实例的类，自动匹配注册表中的model_type
    2. 用model_type从MODEL_CLASS_REGISTRY获取模型类
    3. 用Config实例初始化模型
    
    参数：
        config: BaseConfig实例（通常是某个具体的模型Config，如SimpleMLPConfig）
    
    返回：
        对应的模型实例
    
    抛出：
        ValueError: 如果Config类型未注册或无对应模型类
    
    示例：
        ```python
        from src.config.deep_learning_module.models.SimpleMLPConfig import SimpleMLPConfig
        from src.deep_learning_module.model_factory import get_model
        
        # 创建模型Config
        config = SimpleMLPConfig(
            input_shape=(300, 5),
            hidden_dims=[128, 64],
            num_classes=3,
            task_type="classification"
        )
        
        # 通过工厂创建模型
        model = get_model(config)
        ```
    """
    
    def _get_model_type_by_class(config_class: Type[BaseConfig]) -> str:
        """根据Config类反向找model_type"""
        for model_type, cls in MODEL_CONFIG_REGISTRY.items():
            if cls == config_class:
                return model_type
        return ""
    
    # 步骤1：通过Config实例的类，反向查找对应的model_type
    config_class: Type[BaseConfig] = type(config)
    model_type = _get_model_type_by_class(config_class)
    if not model_type:
        raise ValueError(
            f"未找到与配置类「{config_class.__name__}」对应的模型类型。"
            f"已注册的类型：{list(MODEL_CONFIG_REGISTRY.keys())}"
        )
    
    # 步骤2：从模型注册表中获取对应的模型类
    if model_type not in MODEL_CLASS_REGISTRY:
        raise ValueError(
            f"模型类型「{model_type}」未绑定任何模型类！"
            f"已注册类型：{list(MODEL_CLASS_REGISTRY.keys())}"
        )
    model_class = MODEL_CLASS_REGISTRY[model_type]
    
    if model_class is None:
        raise ImportError(
            f"模型类「{model_type}」导入失败，请检查模型文件是否存在"
        )
    
    # 步骤3：用Config实例初始化模型（模型__init__需接收Config实例）
    return model_class(config=config)
