from typing import Optional, Dict, List, Tuple, Union, Literal, Any
from pydantic import Field, root_validator, validator
from pathlib import Path
import yaml
from abc import ABC, abstractmethod

# 继承pydantic的BaseConfig（而非dataclass，保证配置体系统一）
from ..base_config import BaseConfig


class BaseDatasetConfig(BaseConfig, ABC):
    """
    所有数据集的通用配置基类（pydantic封装，自带类型/范围校验+通用功能）
    核心设计：
    1. 统一继承BaseConfig，与模型配置类保持一致的方法体系（merge_dict/load_yaml等）；
    2. 补充数据集全流程通用参数（路径/划分/预处理/增强/加载优化/缓存等）；
    3. 强参数校验，避免运行时错误；
    4. 定义抽象方法，强制子类实现任务相关逻辑；
    5. 兼容分布式训练/调试场景的参数配置。
    """
    # --------------------------
    # 1. 基础路径与类型配置（核心必选）
    # --------------------------
    # 数据集根目录（必填，无默认值，强制用户指定）
    data_dir: str = Field(
        ...,  # 无默认值，必须显式配置
        description="数据集根目录（绝对路径/相对路径），必填"
    )
    # 数据集类型（关联任务类型，限制可选值）
    dataset_type: Literal["custom", "segmentation", "classification", "regression"] = Field(
        default="custom",
        description="数据集类型：custom(自定义)/segmentation(分割)/classification(分类)/regression(回归)"
    )
    # 标注文件路径（可选，部分数据集标注独立存储）
    annotation_path: Optional[str] = Field(
        default=None,
        description="标注文件路径（如json/csv/txt），无则从dataset_root自动查找"
    )
    # 是否存在标注（无标注场景：如无监督学习）
    has_annotation: bool = Field(
        default=True,
        description="数据集是否包含标注（无标注场景设为False）"
    )

    # --------------------------
    # 2. 数据划分配置（兼容官方划分/自定义划分）
    # --------------------------
    # 是否使用官方划分（优先于split_ratio）
    auto_split : bool = Field(
        default=False,
        description="数据集的自动划分 是否启用"
    )

    use_official_split: bool = Field(
        default=False,
        description="启用官方train/val/test划分比例（优先于split_ratio），若是，则需要在目录中包含train、val、test文件夹"
    )

    # 自定义划分比例（训练集占比，仅use_official_split=False时生效）
    split_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="训练集占总数据的比例（0~1），仅use_official_split=False时生效"
    )
    # 测试集比例（可选，未指定则剩余数据为验证集）
    test_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="测试集占总数据的比例（0~1），未指定则split_ratio外的为验证集"
    )
    # 自定义划分的随机种子（保证划分可复现）
    split_seed: int = Field(
        default=42,
        description="数据划分的随机种子（保证不同运行划分结果一致）"
    )

    # --------------------------
    # 3. 数据加载配置（性能优化+兼容性）
    # --------------------------
    # 批次大小（训练/验证可通过子类覆盖）
    batch_size: int = Field(
        default=8,
        ge=1,
        description="数据加载批次大小，≥1"
    )
    # 训练集是否打乱（验证/测试集建议False）
    shuffle: bool = Field(
        default=True,
        description="训练集是否打乱顺序（验证/测试集建议设为False）"
    )
    # 数据加载线程数（0=主线程加载，≥0）
    num_workers: int = Field(
        default=4,
        ge=0,
        description="数据加载子进程数（0=主线程加载，Windows建议设为0）"
    )
    # 是否锁页内存（加速GPU数据传输）
    pin_memory: bool = Field(
        default=True,
        description="是否使用锁页内存（GPU训练建议开启，CPU训练可关闭）"
    )
    # 是否丢弃最后不完整批次（避免维度不匹配）
    drop_last: bool = Field(
        default=False,
        description="是否丢弃最后一个不完整的批次（分布式训练建议开启）"
    )
    # 数据预取因子（num_workers>0时生效，优化加载速度）
    prefetch_factor: Optional[int] = Field(
        default=2,
        ge=1,
        description="每个worker预取的批次数量（num_workers>0时生效）"
    )
    # 最大加载样本数（调试用，限制加载数据量）
    max_samples: Optional[int] = Field(
        default=None,
        ge=1,
        description="最大加载样本数（调试时限制数据量，None=加载全部）"
    )

    # --------------------------
    # 4. 通用预处理配置（全任务适配）
    # --------------------------
    # 是否归一化（需配合mean/std）
    normalize: bool = Field(
        default=True,
        description="是否对数据进行归一化（需配置mean/std）"
    )
    # 归一化均值（RGB默认ImageNet均值，灰度图可单值）
    mean: Union[List[float], Tuple[float, ...]] = Field(
        default=[0.485, 0.456, 0.406],
        description="归一化均值（RGB图默认ImageNet均值，灰度图传[0.5]）"
    )
    # 归一化标准差（RGB默认ImageNet标准差）
    std: Union[List[float], Tuple[float, ...]] = Field(
        default=[0.229, 0.224, 0.225],
        description="归一化标准差（RGB图默认ImageNet标准差，灰度图传[0.5]）"
    )
    # 调整尺寸（可选，(H,W)，None=不调整）
    resize_size: Optional[Tuple[int, int]] = Field(
        default=None,
        description="统一调整图像尺寸 (H,W)，None=保留原始尺寸"
    )
    # 是否保持长宽比（resize时生效）
    keep_aspect_ratio: bool = Field(
        default=False,
        description="resize时是否保持图像长宽比（避免拉伸）"
    )

    # --------------------------
    # 5. 数据增强配置（训练/验证区分）
    # --------------------------
    # 是否启用训练集数据增强
    train_aug: bool = Field(
        default=True,
        description="是否启用训练集数据增强（验证/测试集禁用）"
    )
    # 随机水平翻转概率
    hflip_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="随机水平翻转的概率（train_aug=True时生效）"
    )
    # 随机垂直翻转概率
    vflip_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="随机垂直翻转的概率（train_aug=True时生效）"
    )
    # 随机旋转角度范围（±angle）
    rotate_angle: int = Field(
        default=0,
        ge=0,
        le=180,
        description="随机旋转的角度范围（±angle度，train_aug=True时生效）"
    )

    # --------------------------
    # 6. 缓存配置（加速重复加载）
    # --------------------------
    # 是否缓存数据到内存（小数据集建议开启）
    cache_in_memory: bool = Field(
        default=False,
        description="是否将数据集缓存到内存（小数据集加速，大数据集禁用）"
    )
    # 磁盘缓存路径（可选，缓存预处理后的数据）
    cache_dir: Optional[str] = Field(
        default=None,
        description="磁盘缓存路径（缓存预处理后的数据，避免重复计算）"
    )

    # --------------------------
    # 7. 分布式训练配置
    # --------------------------
    # 是否使用分布式采样器
    use_dist_sampler: bool = Field(
        default=False,
        description="是否使用分布式采样器（多卡训练时开启）"
    )

    # --------------------------
    # 校验器：保证参数合法性
    # --------------------------
    @validator("mean", "std")
    def validate_mean_std_length(cls, v, values):
        """校验mean/std长度匹配（归一化时必需）"""
        if values.get("normalize") and len(v) == 0:
            raise ValueError("开启normalize时，mean/std不能为空")
        return v

    @root_validator(skip_on_failure=True)
    def validate_split_ratio(cls, values):
        """校验数据划分比例合法性"""
        use_official = values.get("use_official_split")
        if use_official:
            return values  # 官方划分无需校验
        
        split_ratio = values.get("split_ratio")
        test_ratio = values.get("test_ratio")
        
        # 测试集比例+训练集比例≤1
        if test_ratio is not None:
            total = split_ratio + test_ratio
            if total > 1.0:
                raise ValueError(f"split_ratio({split_ratio}) + test_ratio({test_ratio}) 不能超过1.0")
        return values

    @root_validator(skip_on_failure=True)
    def validate_cache_config(cls, values):
        """校验缓存配置合理性"""
        cache_in_memory = values.get("cache_in_memory")
        cache_dir = values.get("cache_dir")
        max_samples = values.get("max_samples")
        
        # 大数据量时禁止内存缓存
        if cache_in_memory and max_samples is None:
            import warnings
            warnings.warn("未限制max_samples时开启cache_in_memory，可能导致内存溢出")
        
        # 磁盘缓存需指定路径
        if cache_dir is not None and not Path(cache_dir).parent.exists():
            raise ValueError(f"缓存目录父路径不存在：{Path(cache_dir).parent}")
        return values

    # --------------------------
    # 通用方法：与模型配置类对齐
    # --------------------------
    def merge_dict(self, user_dict: Dict) -> None:
        """递归合并用户字典配置（仅覆盖指定键，保留默认值）"""
        def _recursive_merge(default_obj: Any, user_dict: Dict):
            for k, v in user_dict.items():
                if hasattr(default_obj, k):
                    attr = getattr(default_obj, k)
                    if isinstance(attr, BaseConfig) and isinstance(v, dict):
                        _recursive_merge(attr, v)
                    else:
                        setattr(default_obj, k, v)
                else:
                    raise KeyError(f"配置键「{k}」不存在于BaseDatasetConfig中")
        _recursive_merge(self, user_dict)

    def load_yaml(self, yaml_path: str) -> None:
        """加载YAML配置文件并合并"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML配置文件不存在：{yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f)
        
        if not isinstance(yaml_dict, dict):
            raise ValueError(f"YAML文件「{yaml_path}」解析后不是字典")
        
        self.merge_dict(yaml_dict)

    # --------------------------
    # 抽象方法：强制子类实现任务相关逻辑
    # --------------------------
    # @abstractmethod
    # def get_task_metrics(self) -> List[str]:
    #     """
    #     获取当前数据集对应的任务评估指标（子类必须实现）
    #     示例：分类任务返回["accuracy", "f1"], 分割任务返回["miou", "dice"]
    #     """
    #     pass

    # @abstractmethod
    # def validate_dataset(self) -> bool:
    #     """
    #     校验数据集完整性（子类必须实现）
    #     检查dataset_root/annotation_path是否存在，样本是否有效等
    #     """
    #     pass