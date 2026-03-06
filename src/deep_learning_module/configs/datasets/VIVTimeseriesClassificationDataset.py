from typing import Optional, List, Literal
from pydantic import Field, validator, root_validator
from pathlib import Path
import numpy as np

# 继承通用数据集配置基类
from .base_dataset_config import BaseDatasetConfig


class VIVTimeSeriesClassificationDatasetConfig(BaseDatasetConfig):
    """
    LSTM时序分类数据集专属配置类（适配TimeSeriesClassificationDataset）
    核心特性：
    1. 继承通用数据集配置（路径/加载/划分等），复用基础能力；
    2. 补充LSTM输入适配的专属参数（batch_first/序列长度/归一化等）；
    3. 针对时序数据的校验（序列长度/特征维度/归一化方式）；
    4. 实现抽象方法，适配时序分类任务的指标和数据集校验。
    """
    # --------------------------
    # 1. LSTM输入格式专属配置
    # --------------------------
    # 是否开启batch_first（LSTM层需对应配置，默认True）
    batch_first: bool = Field(
        default=True,
        description="LSTM输入是否开启batch_first，需与模型的batch_first保持一致"
    )

    # 输出的形状：网格还是时序
    output_mode: str = Field(default="time_series", description="输出模式：time_series（时序）/ grid_2d（50×60网格）")

    # 固定序列长度（None=使用原始序列长度）
    fix_seq_len: Optional[int] = Field(
        default=None,
        ge=1,
        description="固定LSTM输入序列长度，None=保留原始长度；指定时需≥1"
    )
    # 序列补全模式（fix_seq_len生效时，短序列补全策略）
    pad_mode: Literal["zero", "repeat", "mean"] = Field(
        default="zero",
        description="短序列补全模式：zero(补0)/repeat(重复最后值)/mean(补特征均值)"
    )
    # 序列截断模式（fix_seq_len生效时，长序列截断策略）
    trunc_mode: Literal["head", "tail"] = Field(
        default="tail",
        description="长序列截断模式：head(截头部)/tail(截尾部)"
    )

    # --------------------------
    # 2. 时序数据归一化配置（特化）
    # --------------------------
    # 是否对时序数据归一化（覆盖基类的normalize，默认False）
    normalize: bool = Field(
        default=False,
        description="是否对时序数据进行归一化（提升LSTM训练稳定性）"
    )
    # 归一化方式（时序数据专用）
    normalize_type: Literal["min-max", "z-score"] = Field(
        default="min-max",
        description="时序数据归一化方式：min-max(0~1)/z-score(均值0方差1)"
    )
    # 手动指定归一化均值（z-score时生效，None则自动计算）
    ts_mean: Optional[List[float]] = Field(
        default=None,
        description="时序数据归一化均值（z-score模式，None则按特征维度自动计算）"
    )
    # 手动指定归一化标准差（z-score时生效，None则自动计算）
    ts_std: Optional[List[float]] = Field(
        default=None,
        description="时序数据归一化标准差（z-score模式，None则按特征维度自动计算）"
    )

    # --------------------------
    # 3. 时序数据特征维度配置
    # --------------------------
    # 强制特征维度（None=自动适配，指定时校验数据特征维度）
    feat_dim: Optional[int] = Field(
        default=None,
        ge=1,
        description="强制时序数据的特征维度，None=自动适配；指定时需≥1"
    )

    # --------------------------
    # 4. 数据加载适配（LSTM特化）
    # --------------------------
    # 覆盖基类：LSTM建议关闭shuffle（时序数据顺序敏感，按需开启）
    shuffle: bool = Field(
        default=False,
        description="LSTM时序数据是否打乱（时序任务建议关闭，分类任务可开启）"
    )
    # 覆盖基类：时序数据通常小批量，默认batch_size=4
    batch_size: int = Field(
        default=4,
        ge=1,
        description="LSTM批次大小（时序数据显存占用高，建议小批量）"
    )

    # --------------------------
    # 校验器：保证LSTM配置合法性
    # --------------------------
    @root_validator(skip_on_failure=True)
    def validate_normalize_config(cls, values):
        """校验归一化配置的关联性"""
        normalize = values.get("normalize")
        normalize_type = values.get("normalize_type")
        ts_mean = values.get("ts_mean")
        ts_std = values.get("ts_std")

        # 开启归一化时，校验z-score的mean/std
        if normalize and normalize_type == "z-score":
            if (ts_mean is not None and ts_std is None) or (ts_mean is None and ts_std is not None):
                raise ValueError("z-score归一化时，ts_mean和ts_std需同时指定或同时为None")
        return values

    @root_validator(skip_on_failure=True)
    def validate_seq_config(cls, values):
        """校验序列长度配置的关联性"""
        fix_seq_len = values.get("fix_seq_len")
        pad_mode = values.get("pad_mode")
        trunc_mode = values.get("trunc_mode")

        # 仅当指定fix_seq_len时，校验补全/截断模式（避免无效配置）
        if fix_seq_len is None and (pad_mode != "zero" or trunc_mode != "tail"):
            import warnings
            warnings.warn("未指定fix_seq_len，pad_mode/trunc_mode配置将失效")
        return values

    @validator("feat_dim")
    def validate_feat_dim(cls, v):
        """校验特征维度合法性"""
        if v is not None and v < 1:
            raise ValueError(f"feat_dim必须≥1，当前值：{v}")
        return v

    # --------------------------
    # 抽象方法实现（适配时序分类任务）
    # --------------------------
    def get_task_metrics(self) -> List[str]:
        """
        时序分类任务核心评估指标
        Returns:
            指标列表：accuracy(准确率)/f1(宏平均F1)/precision(精确率)/recall(召回率)/auc(曲线下面积)
        """
        return ["accuracy", "f1", "precision", "recall", "auc"]

    def validate_dataset(self) -> bool:
        """
        校验LSTM时序数据集完整性（强制实现）
        检查：数据目录存在、mat文件有效、特征维度匹配等
        """
        # 1. 基础路径校验
        data_dir = Path(self.dataset_root)
        if not data_dir.exists():
            raise FileNotFoundError(f"LSTM数据集根目录不存在：{data_dir}")

        # 2. 检查mat文件数量
        mat_files = list(data_dir.glob("*.mat"))
        if len(mat_files) == 0:
            raise ValueError(f"数据集目录「{data_dir}」下未找到任何.mat文件")

        # 3. 抽样校验mat文件格式（避免全量校验耗时）
        sample_mat = mat_files[0]
        try:
            from scipy.io import loadmat
            mat_data = loadmat(sample_mat)
            # 校验是否包含ndarray类型的键值对（时序数据）
            ndarray_items = [(k, v) for k, v in mat_data.items() if isinstance(v, np.ndarray)]
            if len(ndarray_items) == 0:
                raise ValueError(f"样本文件「{sample_mat}」中未找到时序数据（ndarray类型）")
            
            # 4. 校验特征维度（若指定）
            _, ts_data = ndarray_items[0]
            ts_data = ts_data.reshape(-1, 1) if ts_data.ndim == 1 else ts_data.squeeze()
            actual_feat_dim = ts_data.shape[1] if ts_data.ndim == 2 else 1
            
            if self.feat_dim is not None and actual_feat_dim != self.feat_dim:
                raise ValueError(
                    f"特征维度不匹配！配置feat_dim={self.feat_dim}，样本实际={actual_feat_dim}"
                )

        except Exception as e:
            raise RuntimeError(f"样本文件「{sample_mat}」解析失败：{str(e)}")

        # 5. 校验通过
        print(f"LSTM时序数据集校验通过：\n- 数据目录：{data_dir}\n- 有效mat文件数：{len(mat_files)}\n- 样本特征维度：{actual_feat_dim}")
        return True
