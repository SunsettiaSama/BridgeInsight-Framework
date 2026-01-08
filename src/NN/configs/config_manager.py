# configs/config_manager.py
import yaml
from typing import Union, Dict, TypeVar, Any
from .base_config import BaseConfig
# 导入注册表
from .registry import CONFIG_CLASS_REGISTRY
from .parser import parse_external_config
from pathlib import Path


T = TypeVar("T", bound=BaseConfig)

@staticmethod
def get_config(config_type: str, config_input: Union[str, Dict[str, Any], Path]) -> T:
    """
    对外统一接口：输入config类型+配置，返回校验后的Config实例
    :param config_type: 注册表中的标识（如"simple_cnn"）；
    :param config_input: yaml路径/字典/JSON路径；
    :return: 校验后的Config实例
    """
    # 1. 校验config_type是否在注册表中
    if config_type not in CONFIG_CLASS_REGISTRY:
        raise ValueError(
            f"未知的配置类型：{config_type}！"
            f"支持的类型：{list(CONFIG_CLASS_REGISTRY.keys())}"
        )
    
    # 2. 解析外部输入为原始字典（调用parser工具类）
    raw_dict = parse_external_config(config_input)
    
    # 3. 用基类的from_dict解析并校验（核心：交给基类处理）
    config_class = CONFIG_CLASS_REGISTRY[config_type]
    config_instance = config_class.from_dict(raw_dict)
    
    return config_instance
