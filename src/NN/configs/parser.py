# configs/parser.py
"""
多格式配置解析工具：专门处理「外部输入→原始字典」
- 支持：yaml路径、字典、JSON路径/字符串
- 后续拓展：可加toml、ini等格式，只改这个文件
"""
from typing import Union, Dict, Any
from pathlib import Path
import yaml
from .base_config import BaseConfig
from warnings import warn

def parse_external_config(config_input: Union[str, Dict[str, Any], Path]) -> Dict[str, Any]:
    """
    统一解析外部配置输入为原始字典：
    :param config_input: 支持3种输入：
        1. 字符串/Path → yaml/json文件路径（自动识别后缀）；
        2. 字典 → 直接返回；
    :return: 原始参数字典（未校验，交给BaseConfig.from_dict校验）
    """
    # 1. 处理路径输入（yaml/json文件）
    if isinstance(config_input, (str, Path)):
        config_path = Path(config_input)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_path}")
        
        # 按后缀解析不同格式
        suffix = config_path.suffix.lower()
        if suffix in [".yaml", ".yml"]:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_dict = yaml.safe_load(f)
        elif suffix == ".json":
            from json import load
            with open(config_path, "r", encoding="utf-8") as f:
                raw_dict = load(f)
        else:
            raise ValueError(f"不支持的文件格式：{suffix}，仅支持yaml/yml/json")
        
        if not isinstance(raw_dict, dict):
            raise ValueError(f"配置文件{config_path}解析后不是字典！")
        return raw_dict
    
    # 2. 处理字典输入（直接返回）
    elif isinstance(config_input, dict):
        return config_input
    
    # 3. 不支持的输入类型
    else:
        raise ValueError(
            f"配置输入错误：不支持的配置输入类型：{type(config_input)}！"
            "仅支持：yaml/json文件路径（字符串/Path）、字典"
        )

    