


"""
# 训练器通用配置基类（BaseTrainerConfig）参数汇总表
| 配置分类                     | 参数名                     | 默认值                                      | 参数类型                                                                 | 校验规则                                                                 | 参数描述                                                                 |
|------------------------------|----------------------------|---------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 训练器基础标识               | type                       | "trainer_base"                              | str                                                                      | 需与TRAINER_CONFIG_REGISTRY注册表key一一对应，不可随意修改               | 训练器配置类型标识，用于配置与训练器的自动关联，子类需覆盖为专属标识     |
| 核心训练参数                 | epochs                     | 100                                         | int                                                                      | 必须≥1                                                                   | 训练总轮数，数据集被完整遍历的次数，调试模式下会被debug_max_steps覆盖     |
| 核心训练参数                 | batch_size                 | 32                                          | int                                                                      | 必须≥1                                                                   | 每次迭代输入模型的样本数量，优先级高于数据集配置的batch_size，需适配显存 |
| 核心训练参数                 | learning_rate              | 1e-4（0.0001）                              | float                                                                    | 必须>0                                                                   | 优化器初始学习率，控制参数更新步长，可通过调度器动态调整                 |
| 核心训练参数                 | weight_decay               | 1e-5（0.00001）                             | float                                                                    | 必须≥0                                                                   | 权重衰减系数（L2正则化），用于惩罚过大模型参数，防止过拟合               |
| 优化器&学习率调度器配置       | optimizer                  | "AdamW"                                     | Literal["Adam", "AdamW", "SGD", "RMSprop"]                               | 仅支持指定4种优化器类型                                                   | 模型参数优化器类型，AdamW为当前主流选择，收敛稳定适配多数场景             |
| 优化器&学习率调度器配置       | scheduler                  | "StepLR"                                    | Optional[Literal["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "ExponentialLR"]] | 仅支持指定4种调度器类型或None                                             | 学习率调度器类型，None表示不使用调度器，保持初始学习率不变               |
| 优化器&学习率调度器配置       | scheduler_params           | {"step_size": 30, "gamma": 0.8}             | dict                                                                     | 需与选定的scheduler类型严格匹配，包含对应必填参数                         | 学习率调度器配套参数，如StepLR需指定step_size和gamma                     |
| 优化器&学习率调度器配置       | optimizer_params           | {}（空字典）                                 | Optional[Dict[str, Any]]                                                 | 需符合对应优化器的参数规范                                               | 优化器额外补充参数，如Adam的betas、SGD的momentum等                       |
| 设备配置                     | device                     | 自动适配（有CUDA为"cuda"，无则为"cpu"）       | Union[str, List[str]]                                                    | 字符串需为"cpu"/"cuda"/"cuda:X"，列表需为多卡cuda格式                     | 训练计算设备，支持单卡、多卡、CPU三种模式，多卡需配合分布式训练           |
| 设备配置                     | use_distributed            | False                                       | bool                                                                     | 无（布尔类型）                                                           | 是否启用分布式训练模式，多卡场景可提升训练速度，需配置多卡device         |
| 设备配置                     | dist_port                  | "29500"                                     | str                                                                      | 仅当use_distributed=True时生效，需确保端口未被占用                       | 分布式训练设备间通信端口，用于数据同步与参数传递                         |
| 输出&模型保存配置            | output_dir                 | "./outputs/trainer_output"                  | str                                                                      | 目录不存在时自动创建                                                     | 训练结果根目录，用于保存模型权重、日志、配置文件、TensorBoard日志等       |
| 输出&模型保存配置            | save_freq                  | 0                                           | int                                                                      | 必须≥0                                                                   | 模型检查点保存频率（轮数），0表示仅保存最优模型和最后一轮模型             |
| 输出&模型保存配置            | save_best_model            | True                                        | bool                                                                     | 无（布尔类型）                                                           | 是否根据验证集指标保存最优模型，判断依据为best_model_metric               |
| 输出&模型保存配置            | best_model_metric          | "accuracy"                                  | str                                                                      | 需与训练任务的评估指标一致（分类用大值优，回归用小值优）                 | 判断最优模型的评估指标，子类需根据任务类型覆盖默认值                     |
| 输出&模型保存配置            | resume_from_checkpoint     | None                                        | Optional[str]                                                            | 若指定则需为有效.pth/.pt格式文件路径                                     | 断点续训的模型检查点路径，None表示从头开始训练                           |
| 日志&可视化配置              | use_tensorboard            | True                                        | bool                                                                     | 无（布尔类型）                                                           | 是否启用TensorBoard可视化，用于监控训练/验证的损失与评估指标变化         |
| 日志&可视化配置              | tensorboard_log_dir        | None                                        | Optional[str]                                                            | None时自动在output_dir下创建"tensorboard_logs"目录                       | TensorBoard日志保存目录，需确保目录可写入                                 |
| 日志&可视化配置              | log_freq                   | 10                                          | int                                                                      | 必须≥1                                                                   | 训练日志打印频率（步数），数值过小会导致日志冗余，过大不利于实时监控     |
| 日志&可视化配置              | save_log_file              | True                                        | bool                                                                     | 无（布尔类型）                                                           | 是否将训练日志保存为txt文件，日志文件存储在output_dir下                   |
| 梯度优化配置                 | gradient_clip_norm         | 1.0                                         | Optional[float]                                                          | 必须≥0，None表示不进行梯度裁剪                                           | 梯度裁剪的L2范数阈值，用于限制梯度最大范数，防止梯度爆炸                 |
| 梯度优化配置                 | gradient_accumulation_steps | 1                                          | int                                                                      | 必须≥1                                                                   | 梯度累积步数，有效批次大小=batch_size×该值，显存不足时可增大该值         |
| 训练模式配置                 | use_mixed_precision        | False                                       | bool                                                                     | 无（布尔类型），仅CUDA设备启用有效                                       | 是否启用混合精度训练，平衡训练速度与数值稳定性，可节省约50%显存         |
| 训练模式配置                 | mixed_precision_type       | "fp16"                                      | Literal["fp16", "bf16"]                                                  | 仅当use_mixed_precision=True时生效                                       | 混合精度训练的精度类型，fp16兼容性好，bf16数值稳定性更高（需新一代GPU）  |
| 调试配置                     | debug_mode                 | False                                       | bool                                                                     | 无（布尔类型）                                                           | 是否启用调试模式，用于快速验证训练流程完整性，忽略epochs参数             |
| 调试配置                     | debug_max_steps            | 100                                         | int                                                                      | 必须≥1，仅debug_mode=True时生效                                          | 调试模式下的最大训练步数，训练达到该步数后终止流程                       |
| 调试配置                     | debug_val_freq             | 10                                          | int                                                                      | 必须≥1，仅debug_mode=True时生效                                          | 调试模式下的验证频率（步数），用于快速验证评估流程是否正常               |

"""



"""

# 传统深度学习SFT训练器（DLSFTTrainerConfig）参数汇总表
| 配置分类                                   | 参数名                  | 默认值                | 参数类型                                                                 | 校验规则                                                                 | 参数描述                                                                 |
|--------------------------------------------|-------------------------|-----------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 深度学习SFT基础专属配置                     | sft_task_type           | "classification"      | Literal["classification", "regression", "timeseries_classification", "timeseries_regression"] | 仅支持指定4种任务类型                                                     | 指定SFT任务类型，关联损失函数与评估指标自动匹配，适配分类/回归/时序任务场景 |
| 深度学习SFT基础专属配置                     | fix_feature_extractor   | False                 | bool                                                                     | 无（布尔类型）                                                           | 是否冻结模型特征提取层（如CNN Backbone、LSTM编码层），仅微调预测头，节省显存 |
| 深度学习模型微调专属配置（预训练/冻结）     | pretrained_weight_path  | None                  | Optional[str]                                                            | 若指定则需为有效本地路径，文件格式仅支持.pth/.pt                          | 预训练模型权重路径，用于SFT初始化，None表示从头训练，微调场景建议指定       |
| 深度学习模型微调专属配置（预训练/冻结）     | freeze_layer_prefixes   | []（空列表）          | Optional[List[str]]                                                      | 列表内元素需为非空字符串，支持前缀匹配                                   | 需精确冻结的模型层名称前缀列表，用于精细化控制冻结范围（如["backbone.conv1"]） |
| 深度学习模型微调专属配置（预训练/冻结）     | head_lr_scale           | 10.0                  | float                                                                    | 必须>0                                                                   | 预测头学习率倍增系数，预测头学习率=基础学习率×该系数，建议设置5~20倍       |
| 深度学习模型微调专属配置（预训练/冻结）     | load_pretrained_head    | False                 | bool                                                                     | 无（布尔类型）                                                           | 是否加载预训练权重中的预测头，新任务与预训练任务目标不一致时需设为False     |
| 深度学习SFT损失函数专属配置                 | loss_type               | "CrossEntropyLoss"    | Literal["CrossEntropyLoss", "MSELoss", "L1Loss", "BCELoss", "FocalLoss"] | 仅支持指定5种损失函数，需与任务类型匹配                                   | 训练损失函数类型，分类用CrossEntropyLoss/FocalLoss，回归用MSELoss/L1Loss   |
| 深度学习SFT损失函数专属配置                 | focal_gamma             | 2.0                   | float                                                                    | 必须≥0，仅当loss_type="FocalLoss"时生效                                   | Focal Loss聚焦参数，控制难样本权重分配，数值为0时退化为普通CrossEntropyLoss |
| 覆盖基类的适配性参数（传统深度学习模型专用） | batch_size              | 16                    | int                                                                      | 必须≥1                                                                   | 训练批次大小，需根据模型大小与显存容量调整，传统深度学习模型可适当增大     |
| 覆盖基类的适配性参数（传统深度学习模型专用） | learning_rate           | 1e-3（0.001）         | float                                                                    | 必须>0                                                                   | 模型基础学习率，传统深度学习模型学习率通常高于LLM微调，可按需调整           |
| 覆盖基类的适配性参数（传统深度学习模型专用） | use_mixed_precision     | False                 | bool                                                                     | 无（布尔类型）                                                           | 是否启用混合精度训练（FP16/BF16），节省显存并加速训练，需CUDA设备支持     |
| 覆盖基类的适配性参数（传统深度学习模型专用） | best_model_metric       | "accuracy"            | str                                                                      | 需与任务类型对应的评估指标一致（分类用大值优，回归用小值优）               | 判断最优模型的评估指标，分类常用"accuracy"/"f1"，回归常用"mse"/"mae"       |


"""

"""
传统深度学习SFT训练器配置类（DLSFTTrainerConfig）参数说明文档
====================================================================
一、配置类概述
该配置类是适配CNN/LSTM/MLP等传统深度学习模型的监督微调（SFT）专属配置，继承自通用训练器配置基类（BaseTrainerConfig），
专注于传统深度学习任务（分类/回归/时序分析），提供精细化的微调参数配置与合法性校验，无需依赖大型LLM相关组件。

二、参数详细说明
--------------------------------------------------------------------
模块1：深度学习SFT基础专属配置
--------------------------------------------------------------------
1.  sft_task_type
    - 类型：Literal（枚举字符串）
    - 可选值："classification" / "regression" / "timeseries_classification" / "timeseries_regression"
    - 默认值："classification"
    - 含义：指定SFT任务类型，对应不同的深度学习任务场景
      - classification：图像分类、特征分类等通用分类任务
      - regression：数值回归、预测等回归任务
      - timeseries_classification：时间序列分类（如振动信号故障分类）
      - timeseries_regression：时间序列回归（如时序数据趋势预测）
    - 注意：任务类型将关联损失函数选择与评估指标自动匹配

2.  fix_feature_extractor
    - 类型：bool（布尔值）
    - 默认值：False
    - 含义：是否启用特征提取层固定模式
      - True：冻结模型的特征提取部分（如CNN的Backbone、LSTM的编码层），仅微调后续的分类头/回归头
      - False：所有模型层均参与微调训练
    - 注意：开启后可节省显存，适合小样本微调场景，建议配合预训练权重使用

--------------------------------------------------------------------
模块2：深度学习模型微调专属配置（预训练/冻结）
--------------------------------------------------------------------
1.  pretrained_weight_path
    - 类型：Optional[str]（可选字符串）
    - 默认值：None
    - 含义：预训练模型权重文件路径，用于SFT微调的模型初始化
      - 支持格式：本地.pth/.pt格式权重文件
      - None：表示不使用预训练权重，模型从头开始训练
    - 注意：SFT微调场景下建议指定有效预训练权重，提升训练效果与收敛速度

2.  freeze_layer_prefixes
    - 类型：Optional[List[str]]（可选字符串列表）
    - 默认值：[]（空列表）
    - 含义：需精确冻结的模型层名称前缀列表，用于精细化控制冻结范围
      - 示例：["backbone.conv1", "encoder.layer.0", "lstm.layer1"]
      - 空列表：表示不额外冻结指定层
    - 注意：前缀匹配模式，只要模型层名称以列表中的字符串开头，该层将被冻结

3.  head_lr_scale
    - 类型：float（浮点型）
    - 默认值：10.0
    - 约束：必须大于0
    - 含义：预测头（分类头/回归头）学习率倍增系数，计算公式为：
      预测头学习率 = 基础学习率（learning_rate） × 该系数
    - 注意：通常预测头需要更高的学习率以快速适配新任务，建议设置为5~20倍

4.  load_pretrained_head
    - 类型：bool（布尔值）
    - 默认值：False
    - 含义：是否加载预训练权重中的预测头部分
      - True：加载预训练权重中的分类头/回归头参数
      - False：仅加载特征提取层参数，预测头将随机初始化
    - 注意：当新任务与预训练任务的类别数/回归目标不一致时，必须设为False

--------------------------------------------------------------------
模块3：深度学习SFT损失函数专属配置
--------------------------------------------------------------------
1.  loss_type
    - 类型：Literal（枚举字符串）
    - 可选值："CrossEntropyLoss" / "MSELoss" / "L1Loss" / "BCELoss" / "FocalLoss"
    - 默认值："CrossEntropyLoss"
    - 含义：指定训练使用的损失函数，需与任务类型匹配
      - CrossEntropyLoss：多分类任务默认损失
      - MSELoss：回归任务默认损失（均方误差）
      - L1Loss：回归任务损失（平均绝对误差）
      - BCELoss：二分类任务损失（二元交叉熵）
      - FocalLoss：难样本聚焦分类损失，适合类别不平衡场景
    - 注意：配置时需与sft_task_type对应，否则会触发警告提示

3.  focal_gamma
    - 类型：float（浮点型）
    - 默认值：2.0
    - 约束：必须≥0
    - 含义：Focal Loss的聚焦参数，用于控制难样本的权重分配
      - 数值越大：难样本的权重越高，易样本的权重越低
      - 数值为0：Focal Loss退化为普通CrossEntropyLoss
    - 注意：仅当loss_type为"FocalLoss"时生效，其他损失函数下该参数无效

--------------------------------------------------------------------
模块4：覆盖基类的适配性参数（传统深度学习模型专用）
--------------------------------------------------------------------
1.  batch_size
    - 类型：int（整数）
    - 默认值：16
    - 约束：必须≥1
    - 含义：训练批次大小，即每次迭代输入模型的样本数量
    - 注意：需根据模型大小与显存容量调整，传统深度学习模型可适当增大，显存不足时可减小

2.  learning_rate
    - 类型：float（浮点型）
    - 默认值：1e-3（0.001）
    - 约束：必须>0
    - 含义：模型基础学习率，是优化器的初始学习率
    - 注意：传统深度学习模型的基础学习率通常高于LLM微调，可根据训练收敛情况调整

3.  use_mixed_precision
    - 类型：bool（布尔值）
    - 默认值：False
    - 含义：是否启用混合精度训练
      - True：使用FP16/BF16精度训练，节省显存并加速训练
      - False：使用FP32精度训练，稳定性更高
    - 注意：传统深度学习模型显存压力较小，默认关闭；仅当模型较大时建议开启，需CUDA设备支持

4.  best_model_metric
    - 类型：str（字符串）
    - 默认值："accuracy"
    - 含义：判断最优模型的评估指标，用于模型保存
      - 分类任务：常用"accuracy"、"f1"（数值越大越优）
      - 回归任务：常用"mse"、"mae"（数值越小越优）
    - 注意：需与任务类型对应的评估指标一致，否则无法正确筛选最优模型

三、使用注意事项
2.  预训练权重：微调场景下建议优先使用与任务相关的预训练权重，提升模型性能
3.  冻结策略：fix_feature_extractor与freeze_layer_prefixes不可同时使用（互斥关系）
4.  评估指标：配置类会根据sft_task_type自动返回对应评估指标，无需手动指定
5.  合法性校验：所有参数均有内置校验逻辑，非法参数会抛出明确的错误提示
"""


from typing import Optional, List, Literal, Dict, Any
from pydantic import Field, validator, root_validator
from pathlib import Path
import torch

# 继承通用训练器配置基类
from .base_config import BaseTrainerConfig

# 顶部新增logger导入（仅需添加这行）
import logging

# 获取logger实例（若项目有全局logger，可直接使用项目logger）
logger = logging.getLogger(__name__)

class SFTTrainerConfig(BaseTrainerConfig):
    """
    常规深度学习模型SFT（监督微调）专属配置类（适配CNN/LSTM/MLP等模型，非LLM）
    核心特性：
    1. 继承通用训练器配置，复用基础能力（优化器/设备/输出等）；
    2. 补充传统深度学习SFT专属参数（模型冻结/预训练权重/任务适配等）；
    3. 针对CNN/LSTM等模型的关联性校验（预训练权重/损失函数/任务匹配性）；
    4. 实现抽象方法，适配传统深度学习任务的评估指标和配置校验。
    """

    # 深度学习SFT任务类型（适配传统深度学习任务）
    sft_task_type: Literal["classification", "regression", "timeseries_classification", "timeseries_regression"] = Field(
        default="classification",
        description="深度学习SFT任务类型：classification(图像/特征分类)/regression(回归预测)/timeseries_classification(时序分类)/timeseries_regression(时序回归)"
    )
    # 是否启用特征固定模式（仅微调分类头/回归头，冻结特征提取层）
    fix_feature_extractor: bool = Field(
        default=False,
        description="是否固定特征提取层（如CNN的backbone/LSTM的编码层），仅微调预测头"
    )

    # --------------------------
    # 2. 深度学习模型微调专属配置（预训练/冻结）
    # --------------------------
    # 预训练模型权重路径（本地.pth/.pt格式，SFT微调初始化）
    pretrained_weight_path: Optional[str] = Field(
        default=None,
        description="预训练模型权重路径（本地.pth/.pt文件），None表示从头训练，SFT微调建议指定"
    )
    # 需冻结的模型层名称前缀（如"backbone."/"encoder."，精确控制冻结范围）
    freeze_layer_prefixes: Optional[List[str]] = Field(
        default_factory=list,
        description="需冻结的模型层名称前缀列表（如['backbone.conv1', 'encoder.layer']）"
    )
    # 预测头学习率倍增系数（预测头学习率=基础lr×此系数，强化微调效果）
    head_lr_scale: float = Field(
        default=10.0,
        gt=0.0,
        description="预测头（分类/回归头）学习率倍增系数，需>0，默认10倍于基础学习率"
    )
    # 是否加载预训练权重中的预测头（False=仅加载特征提取层，适配新任务）
    load_pretrained_head: bool = Field(
        default=False,
        description="是否加载预训练权重中的预测头，新任务类别/回归目标不同时建议设为False"
    )

    # --------------------------
    # 3. 深度学习SFT损失函数专属配置
    # --------------------------
    # 损失函数类型（适配传统深度学习任务）
    loss_type: Literal["CrossEntropyLoss", "MSELoss", "L1Loss", "BCELoss", "FocalLoss"] = Field(
        default="CrossEntropyLoss",
        description="损失函数类型：分类用CrossEntropyLoss/FocalLoss，回归用MSELoss/L1Loss，二分类用BCELoss"
    )

    # Focal Loss聚焦参数（仅loss_type=FocalLoss时生效）
    focal_gamma: float = Field(
        default=2.0,
        ge=0.0,
        description="Focal Loss的gamma参数，控制难样本权重，仅loss_type=FocalLoss时生效"
    )

    # --------------------------
    # 4. 覆盖基类参数（适配传统深度学习模型）
    # --------------------------
    # 覆盖基类：传统深度学习模型批次大小（根据模型大小调整，默认16）
    batch_size: int = Field(
        default=16,
        ge=1,
        description="深度学习SFT训练批次大小，≥1，根据模型显存占用调整"
    )
    # 覆盖基类：传统深度学习模型学习率（默认1e-3，高于LLM）
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="深度学习模型基础学习率，需>0，默认1e-3（高于LLM微调）"
    )
    # 覆盖基类：传统深度学习模型默认不开启混合精度（按需开启）
    use_mixed_precision: bool = Field(
        default=False,
        description="是否使用混合精度训练（传统深度学习模型显存压力较小，默认关闭）"
    )
    # 覆盖基类：最优模型指标（分类用accuracy，回归用mse）
    best_model_metric: str = Field(
        default="accuracy",
        description="最优模型判断指标：分类用accuracy/f1，回归用mse/mae（越小越优）"
    )

    exclude_bias_from_weight_decay: bool = Field(
        default=True,
        description="启用以排除模型的偏置项的权重衰减,可以提升收敛速度和泛化能力"
    )

    head_param_prefixes: str = Field(
        default="head.",
        description="推理头的参数名称前缀"
    )

    # --------------------------
    # 校验器：保证深度学习SFT配置合法性
    # --------------------------
    @root_validator(skip_on_failure=True)
    def validate_task_and_params(cls, values):
        """校验任务类型与关联参数的匹配性"""
        sft_task_type = values.get("sft_task_type")
        loss_type = values.get("loss_type")

        # 分类类任务校验
        if sft_task_type in ["classification", "timeseries_classification"]:
            if loss_type in ["MSELoss", "L1Loss"]:
                import warnings
                warnings.warn(f"sft_task_type='{sft_task_type}'时，建议使用分类专用损失（CrossEntropyLoss/FocalLoss），当前为{loss_type}")
        # 回归类任务校验
        elif sft_task_type in ["regression", "timeseries_regression"]:
            if loss_type in ["CrossEntropyLoss", "FocalLoss", "BCELoss"]:
                import warnings
                warnings.warn(f"sft_task_type='{sft_task_type}'时，建议使用回归专用损失（MSELoss/L1Loss），当前为{loss_type}")
        return values

    @root_validator(skip_on_failure=True)
    def validate_pretrained_config(cls, values):
        """校验预训练权重配置的合理性"""
        pretrained_weight_path = values.get("pretrained_weight_path")
        load_pretrained_head = values.get("load_pretrained_head")
        fix_feature_extractor = values.get("fix_feature_extractor")

        # 预训练权重路径校验
        if pretrained_weight_path is not None:
            weight_path = Path(pretrained_weight_path)
            if not weight_path.exists():
                raise FileNotFoundError(f"预训练模型权重文件不存在：{pretrained_weight_path}")
            if weight_path.suffix not in [".pth", ".pt"]:
                raise ValueError(f"预训练权重仅支持.pth/.pt格式，当前为{weight_path.suffix}")
        
        # 固定特征提取层时，建议指定预训练权重
        if fix_feature_extractor and pretrained_weight_path is None:
            import warnings
            warnings.warn("开启fix_feature_extractor时，建议指定pretrained_weight_path，否则特征提取层无有效初始化")
        
        # 加载预测头与固定特征提取层的兼容性
        if fix_feature_extractor and load_pretrained_head:
            raise ValueError("fix_feature_extractor=True时，不可同时加载预训练预测头（load_pretrained_head需设为False）")
        return values

    @validator("freeze_layer_prefixes")
    def validate_freeze_layer_prefixes(cls, v):
        """校验冻结层前缀列表合法性"""
        if v is not None:
            for prefix in v:
                if not isinstance(prefix, str) or len(prefix) == 0:
                    raise ValueError(f"冻结层前缀必须为非空字符串，当前存在无效值：{prefix}")
        return v

    @validator("head_lr_scale")
    def validate_head_lr_scale(cls, v):
        """校验预测头学习率倍增系数合法性"""
        if v <= 0:
            raise ValueError(f"预测头学习率倍增系数必须>0，当前值：{v}")
        return v

    @validator("focal_gamma")
    def validate_focal_gamma(cls, v, values):
        """校验Focal Loss参数合法性"""
        if values.get("loss_type") == "FocalLoss" and v < 0:
            raise ValueError("Focal Loss的gamma参数必须≥0")
        return v

    # --------------------------
    # 抽象方法实现（适配传统深度学习SFT任务）
    # --------------------------
    def get_train_evaluation_metrics(self) -> List[str]:
        """
        传统深度学习SFT任务核心评估指标（按任务类型返回对应指标）
        Returns:
            指标列表：适配分类/回归/时序任务，保证评估全面性
        """
        sft_task_type = self.sft_task_type
        if sft_task_type in ["classification", "timeseries_classification"]:
            # 分类类任务指标
            return ["accuracy", "f1", "precision", "recall", "top_k_accuracy"]
        elif sft_task_type in ["regression", "timeseries_regression"]:
            # 回归类任务指标（越小越优）
            return ["mse", "mae", "rmse", "r2_score"]
        else:
            # 默认通用指标
            return ["loss", "accuracy"]

    def validate_trainer_config(self) -> bool:
        """
        校验传统深度学习SFT训练器配置的完整性与合理性（强制实现）
        检查：预训练权重有效性、设备兼容性、参数关联性等
        """
        # 1. 预训练权重文件校验
        pretrained_weight_path = self.pretrained_weight_path
        if pretrained_weight_path is not None:
            weight_path = Path(pretrained_weight_path)
            # 尝试加载权重文件（轻量校验，不加载到模型）
            try:
                torch.load(weight_path, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"预训练权重文件解析失败：{pretrained_weight_path}，错误信息：{str(e)}")

        # 2. 设备与模型兼容性校验
        device = self.device
        if isinstance(device, list) and len(device) > 1 and not self.use_distributed:
            raise RuntimeError("配置多卡设备时，需开启use_distributed=True（分布式训练）")

        # 3. 损失函数与任务类型二次校验
        sft_task_type = self.sft_task_type
        loss_type = self.loss_type
        if sft_task_type in ["classification", "timeseries_classification"] and loss_type == "FocalLoss" :
            raise ValueError("Focal Loss仅支持多分类/二分类任务")

        # 4. 学习率配置校验
        base_lr = self.learning_rate
        head_lr = base_lr * self.head_lr_scale
        if head_lr <= 0:
            raise ValueError(f"预测头学习率（{head_lr}）无效，由基础lr（{base_lr}）×倍增系数（{self.head_lr_scale}）计算得出")

        # 5. 校验通过（print替换为logger.info，格式不变）
        logger.info(f"传统深度学习SFT训练器配置校验通过：")
        logger.info(f"- 任务类型：{self.sft_task_type}")
        logger.info(f"- 预训练权重：{pretrained_weight_path if pretrained_weight_path else '无（从头训练）'}")
        logger.info(f"- 训练设备：{self.device}，批次大小：{self.batch_size}")
        logger.info(f"- 基础学习率：{base_lr}，预测头学习率：{head_lr}")
        logger.info(f"- 评估指标：{self.get_train_evaluation_metrics()}")
        return True
    









