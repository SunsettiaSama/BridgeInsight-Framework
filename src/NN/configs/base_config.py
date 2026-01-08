# configs/base_config.py
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Type, TypeVar, Union
import json
from pathlib import Path

# 定义泛型，方便后续子类复用
T = TypeVar("T", bound="BaseConfig")

class BaseConfig(BaseModel):
    """所有Config类的基类：自带解析/校验/格式转换能力"""
    class Config:
        arbitrary_types_allowed = True  # 允许自定义类型（如路径、torch.device）
        validate_assignment = True     # 赋值时自动触发校验（比如config.dropout=1.5会直接报错）
        frozen = False                 # 允许后续修改配置（如需不可改，设为True）

    # --------------------------
    # 基类自带的核心解析能力（所有子类直接用）
    # --------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Config实例 → 字典（方便模型初始化时取值）"""
        return self.dict(exclude_unset=True)  # 只返回显式设置的参数，排除默认值

    def to_json(self, file_path: str = None) -> str:
        """Config实例 → JSON字符串（可选保存到文件）"""
        json_str = self.json(indent=4, ensure_ascii=False)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any] = None) -> T:
        """字典 → Config实例（核心解析逻辑，自动触发校验）"""
        try:
            return cls(**config_dict)
        except ValidationError as e:
            # 自定义报错信息，方便定位问题
            raise ValueError(f"解析字典到{cls.__name__}失败！错误：{e}") from e

    @classmethod
    def from_json(cls: Type[T], json_input: Union[str, Path]) -> T:
        """JSON字符串/JSON文件路径 → Config实例（拓展能力）"""
        if isinstance(json_input, (str, Path)) and Path(json_input).exists():
            # 是文件路径 → 读文件
            with open(json_input, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            # 是JSON字符串 → 直接解析
            config_dict = json.loads(json_input)
        return cls.from_dict(config_dict)
    

