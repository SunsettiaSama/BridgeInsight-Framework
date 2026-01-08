




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
训练器通用配置基类（BaseTrainerConfig）参数说明文档
====================================================================
一、配置类概述
该类是所有训练器的抽象通用配置基类，基于Pydantic封装并继承项目统一BaseConfig，同时实现ABC抽象类接口，
为各类训练器（SFT/常规训练/分布式训练等）提供统一的参数体系与功能支撑，具备强类型校验、通用方法复用、
任务逻辑强制约束等特性，确保训练流程的规范性与可扩展性。

核心设计价值：
1.  统一配置标准：与数据集配置类保持一致的方法体系（merge_dict/load_yaml/save_config），降低使用成本；
2.  全流程参数覆盖：涵盖训练核心、优化器、设备、输出、可视化、梯度优化等全环节通用参数；
3.  提前规避风险：内置参数合法性校验，避免运行时出现设备不兼容、调度器参数缺失等错误；
4.  强制任务适配：定义抽象方法，要求子类实现任务专属的评估指标与配置校验逻辑；
5.  兼容多场景：支持单卡/多卡分布式、混合精度、调试验证等多种训练场景。

二、参数详细说明
--------------------------------------------------------------------
模块1：训练器基础标识（核心必选/关联注册表）
--------------------------------------------------------------------
1.  type
    - 类型：str（字符串）
    - 默认值："trainer_base"
    - 含义：训练器配置类型标识，与项目中TRAINER_CONFIG_REGISTRY注册表的key一一对应，用于配置与训练器的自动关联
    - 注意：该字段为配置注册标识，不可随意修改，子类需覆盖为专属标识（如"sft_trainer"），否则无法正常注册和调用

--------------------------------------------------------------------
模块2：核心训练参数（训练流程基础配置）
--------------------------------------------------------------------
1.  epochs
    - 类型：int（整数）
    - 默认值：100
    - 约束：必须≥1
    - 含义：训练总轮数，即整个训练过程中数据集被完整遍历的次数
    - 注意：
      - 轮数过少可能导致模型欠拟合，轮数过多可能导致过拟合，需结合验证集指标调整；
      - 调试模式下该参数会被忽略，以debug_max_steps为准。

2.  batch_size
    - 类型：int（整数）
    - 默认值：32
    - 约束：必须≥1
    - 含义：每次训练迭代输入模型的样本数量，是控制训练效率与显存占用的核心参数
    - 注意：
      - 若与数据集配置中的batch_size冲突，优先使用训练器的该配置；
      - 显存不足时可减小该值，或配合gradient_accumulation_steps增大有效批次。

3.  learning_rate
    - 类型：float（浮点型）
    - 默认值：1e-4（0.0001）
    - 约束：必须>0
    - 含义：优化器的初始学习率，控制模型参数更新的步长
    - 注意：
      - 学习率过大会导致训练震荡不收敛，过小会导致训练速度过慢；
      - 可通过学习率调度器（scheduler）动态调整该值。

4.  weight_decay
    - 类型：float（浮点型）
    - 默认值：1e-5（0.00001）
    - 约束：必须≥0
    - 含义：权重衰减系数，本质是L2正则化项，用于惩罚过大的模型参数，防止过拟合
    - 注意：
      - 数值越大，正则化约束越强，可根据模型过拟合程度调整；
      - 对偏差项（bias）通常不施加权重衰减，优化器会自动处理。

--------------------------------------------------------------------
模块3：优化器&学习率调度器配置（训练优化核心）
--------------------------------------------------------------------
1.  optimizer
    - 类型：Literal（枚举字符串）
    - 可选值："Adam" / "AdamW" / "SGD" / "RMSprop"
    - 默认值："AdamW"
    - 含义：模型参数优化器类型，用于更新模型权重以降低损失函数值
    - 适用场景：
      - AdamW：带权重衰减的Adam，目前主流选择，收敛稳定，适合大多数场景；
      - Adam：无权重衰减的经典优化器，易出现过拟合；
      - SGD：随机梯度下降，收敛速度慢但泛化能力强，适合大规模数据集；
      - RMSprop：适合处理非平稳目标，常用于时序数据训练。

2.  scheduler
    - 类型：Optional[Literal]（可选枚举字符串）
    - 可选值："StepLR" / "CosineAnnealingLR" / "ReduceLROnPlateau" / "ExponentialLR" / None
    - 默认值："StepLR"
    - 含义：学习率调度器类型，用于在训练过程中动态调整学习率，提升模型收敛效果
    - 注意：None表示不使用学习率调度，保持初始learning_rate不变。

3.  scheduler_params
    - 类型：dict（字典）
    - 默认值：{"step_size": 30, "gamma": 0.8}
    - 含义：学习率调度器的配套参数，需与选定的scheduler类型严格匹配
    - 常用调度器必填参数参考：
      - StepLR：step_size（多少轮更新一次学习率）、gamma（学习率衰减系数）；
      - CosineAnnealingLR：T_max（学习率周期长度）；
      - ReduceLROnPlateau：mode（指标模式，max/min）、patience（多少轮指标不提升则衰减）；
      - ExponentialLR：gamma（指数衰减系数）。

4.  optimizer_params
    - 类型：Optional[Dict[str, Any]]（可选字典）
    - 默认值：{}（空字典）
    - 含义：优化器的额外补充参数，用于配置优化器的非默认属性
    - 示例：
      - Adam/AdamW：{"betas": (0.9, 0.999), "eps": 1e-08}；
      - SGD：{"momentum": 0.9, "nesterov": True}。

--------------------------------------------------------------------
模块4：设备配置（兼容单卡/多卡/CPU）
--------------------------------------------------------------------
1.  device
    - 类型：Union[str, List[str]]（字符串或字符串列表）
    - 默认值：自动适配（有CUDA则为"cuda"，无则为"cpu"）
    - 含义：指定训练使用的计算设备，支持单卡、多卡、CPU三种模式
    - 配置示例：
      - CPU："cpu"；
      - 单卡："cuda" 或 "cuda:0"；
      - 多卡：["cuda:0", "cuda:1", "cuda:2"]。
    - 注意：多卡配置需配合use_distributed=True使用，否则无法实现分布式训练。

2.  use_distributed
    - 类型：bool（布尔值）
    - 默认值：False
    - 含义：是否启用分布式训练模式，用于多卡场景提升训练速度
    - 注意：
      - 启用前需确保有多个可用CUDA设备，且device配置为多卡列表；
      - 分布式训练会自动拆分数据集，无需手动调整batch_size。

3.  dist_port
    - 类型：str（字符串）
    - 默认值："29500"
    - 含义：分布式训练中各设备间通信的端口号，用于设备间的数据同步与参数传递
    - 注意：仅当use_distributed=True时生效，需确保该端口未被其他进程占用。

--------------------------------------------------------------------
模块5：输出&模型保存配置（结果持久化）
--------------------------------------------------------------------
1.  output_dir
    - 类型：str（字符串）
    - 默认值："./outputs/trainer_output"
    - 含义：训练结果的根目录，用于保存模型权重、日志文件、配置文件、TensorBoard日志等
    - 注意：目录不存在时会自动创建，建议按任务类型划分子目录，避免不同训练任务的结果混淆。

2.  save_freq
    - 类型：int（整数）
    - 默认值：0
    - 约束：必须≥0
    - 含义：模型检查点（checkpoint）的保存频率，以训练轮数为单位
    - 注意：
      - 0表示仅保存最优模型（按best_model_metric判断）和最后一轮模型；
      - 大于0表示每save_freq轮保存一次中间模型，适合长时间训练的断点续训。

3.  save_best_model
    - 类型：bool（布尔值）
    - 默认值：True
    - 含义：是否根据验证集指标保存性能最优的模型
    - 注意：最优模型的判断依据是best_model_metric，分类任务通常选"accuracy"（越大越优），回归任务通常选"mse"（越小越优）。

4.  best_model_metric
    - 类型：str（字符串）
    - 默认值："accuracy"
    - 含义：判断最优模型的评估指标名称，需与训练任务的评估指标保持一致
    - 注意：子类需根据任务类型覆盖该默认值，如回归任务改为"mse"。

5.  resume_from_checkpoint
    - 类型：Optional[str]（可选字符串）
    - 默认值：None
    - 含义：断点续训的模型检查点路径，指向已保存的.pth/.pt格式模型权重文件
    - 注意：
      - None表示从头开始训练，不加载任何预训练检查点；
      - 若指定路径，需确保文件存在且格式合法，否则会触发文件不存在异常。

--------------------------------------------------------------------
模块6：日志&可视化配置（训练监控）
--------------------------------------------------------------------
1.  use_tensorboard
    - 类型：bool（布尔值）
    - 默认值：True
    - 含义：是否启用TensorBoard可视化工具，用于监控训练/验证过程中的损失、评估指标等变化
    - 注意：启用后会自动在output_dir下创建tensorboard_logs目录（若未指定tensorboard_log_dir）。

2.  tensorboard_log_dir
    - 类型：Optional[str]（可选字符串）
    - 默认值：None
    - 含义：TensorBoard日志文件的保存目录
    - 注意：None表示自动在output_dir下创建"tensorboard_logs"目录，手动指定时需确保目录可写入。

3.  log_freq
    - 类型：int（整数）
    - 默认值：10
    - 约束：必须≥1
    - 含义：训练日志的打印频率，以训练步数（step）为单位，即每训练多少个批次打印一次损失、学习率等信息
    - 注意：数值过小会导致日志输出过于频繁，影响训练效率；数值过大不利于实时监控训练状态。

4.  save_log_file
    - 类型：bool（布尔值）
    - 默认值：True
    - 含义：是否将训练日志（终端打印的内容）保存为txt格式文件
    - 注意：日志文件会自动保存到output_dir下，文件名为"train_log.txt"，便于后续复盘训练过程。

--------------------------------------------------------------------
模块7：梯度优化配置（防止梯度爆炸/消失）
--------------------------------------------------------------------
1.  gradient_clip_norm
    - 类型：Optional[float]（可选浮点型）
    - 默认值：1.0
    - 约束：必须≥0
    - 含义：梯度裁剪的L2范数阈值，用于限制梯度的最大范数，防止梯度爆炸
    - 注意：
      - None表示不进行梯度裁剪；
      - 当训练过程中出现损失NaN/Inf时，可适当减小该值加强裁剪效果。

2.  gradient_accumulation_steps
    - 类型：int（整数）
    - 默认值：1
    - 约束：必须≥1
    - 含义：梯度累积步数，即累积多少个批次的梯度后再进行一次参数更新
    - 注意：
      - 显存不足时，可增大该值实现“小批次输入，大批次更新”的效果，有效批次大小=batch_size×gradient_accumulation_steps；
      - 累积步数过大可能导致梯度更新延迟，影响模型收敛速度。

--------------------------------------------------------------------
模块8：训练模式配置（精度/效率优化）
--------------------------------------------------------------------
1.  use_mixed_precision
    - 类型：bool（布尔值）
    - 默认值：False
    - 含义：是否启用混合精度训练模式，在训练过程中同时使用FP16/BF16和FP32精度，平衡训练速度与数值稳定性
    - 注意：
      - 仅支持CUDA设备，CPU设备启用无效；
      - 启用后可节省约50%显存，提升训练速度，适合大型模型训练。

2.  mixed_precision_type
    - 类型：Literal（枚举字符串）
    - 可选值："fp16" / "bf16"
    - 默认值："fp16"
    - 含义：混合精度训练的精度类型，仅当use_mixed_precision=True时生效
    - 注意：
      - fp16：兼容性更好，绝大多数CUDA设备支持；
      - bf16：数值稳定性更高，但仅支持新一代NVIDIA/Ampere架构GPU（如RTX 30系列及以上）。

--------------------------------------------------------------------
模块9：调试配置（快速验证流程）
--------------------------------------------------------------------
1.  debug_mode
    - 类型：bool（布尔值）
    - 默认值：False
    - 含义：是否启用调试模式，用于快速验证训练流程的完整性，无需完整训练
    - 注意：调试模式下会忽略epochs参数，仅训练少量步数后停止。

2.  debug_max_steps
    - 类型：int（整数）
    - 默认值：100
    - 约束：必须≥1
    - 含义：调试模式下的最大训练步数，即仅训练该步数后终止训练流程
    - 注意：仅当debug_mode=True时生效，用于快速验证数据加载、模型前向/反向传播、日志保存等流程是否正常。

3.  debug_val_freq
    - 类型：int（整数）
    - 默认值：10
    - 约束：必须≥1
    - 含义：调试模式下的验证频率，以训练步数为单位，即每训练多少步进行一次验证集评估
    - 注意：仅当debug_mode=True时生效，用于快速验证评估流程是否正常。

三、内置校验器说明
--------------------------------------------------------------------
1.  validate_device：校验设备配置合法性，确保设备格式为cpu/cuda/cuda:X或多卡cuda列表，避免无效设备调用；
2.  validate_scheduler_config：校验调度器与调度器参数的匹配性，确保选定的scheduler有对应的必填参数，避免调度器初始化失败；
3.  validate_output_dir：校验输出目录合理性，自动创建不存在的目录，补全TensorBoard日志目录，校验断点续训文件是否存在；
4.  validate_distributed_config：校验分布式训练配置合理性，确保分布式训练时使用多卡设备且有可用CUDA，避免分布式训练异常。

四、通用方法说明
--------------------------------------------------------------------
1.  merge_dict(user_dict: Dict) -> None：递归合并用户字典配置，仅覆盖指定键，保留未指定的默认值，支持嵌套配置合并；
2.  load_yaml(yaml_path: str) -> None：加载YAML配置文件并递归合并到当前配置，实现配置的文件化管理；
3.  save_config(save_path: Optional[str] = None) -> None：将当前配置保存为YAML文件，默认保存到output_dir下，便于复现训练过程。

五、抽象方法说明（子类必须实现）
--------------------------------------------------------------------
1.  get_train_evaluation_metrics() -> List[str]：获取当前训练任务的评估指标列表，需根据任务类型返回对应指标（如分类任务返回["accuracy", "f1"]）；
2.  validate_trainer_config() -> bool：校验训练器配置的完整性与合理性，检查任务专属参数的关联性，返回True表示校验通过，否则抛出异常。

六、使用注意事项
--------------------------------------------------------------------
1.  子类继承：自定义训练器配置时，需继承该基类，并覆盖type字段为专属标识，实现两个抽象方法；
2.  参数优先级：训练器配置的batch_size优先级高于数据集配置，若需统一批次大小，建议仅在一处配置；
3.  分布式训练：启用use_distributed=True时，需确保device为多卡列表，且有可用CUDA设备，避免端口冲突；
4.  断点续训：指定resume_from_checkpoint时，需确保文件格式为.pth/.pt，且包含完整的模型参数、优化器状态等信息；
5.  调试优先：新任务训练前，建议先开启debug_mode=True验证流程完整性，再进行完整训练，提高调试效率；
6.  配置保存：训练前建议调用save_config方法保存配置，便于后续复现实验结果和追溯参数设置。
"""

from typing import Optional, Dict, List, Tuple, Union, Literal, Any
from pydantic import Field, root_validator, validator
from pathlib import Path
import yaml
from abc import ABC, abstractmethod
import torch

# 继承项目统一的BaseConfig（与数据集配置保持一致）
from ..base_config import BaseConfig

class BaseTrainerConfig(BaseConfig, ABC):
    """
    所有训练器的通用配置基类（pydantic封装，自带类型/范围校验+通用功能）
    核心设计：
    1. 统一继承BaseConfig，与数据集配置类保持一致的方法体系（merge_dict/load_yaml等）；
    2. 补充训练器全流程通用参数（标识/核心训练/优化器/设备/输出/可视化等）；
    3. 强参数校验，避免运行时错误；
    4. 定义抽象方法，强制子类实现任务相关逻辑；
    5. 兼容分布式训练/混合精度/调试场景的参数配置。
    """
    # --------------------------
    # 1. 训练器基础标识（核心必选/关联注册表）
    # --------------------------
    # 配置类型（与注册表中trainer_config标识对应，不可随意修改）
    # type: str = Field(
    #     default="trainer_base",
    #     description="训练器配置类型（与TRAINER_CONFIG_REGISTRY中的key一一对应）"
    # )

    # --------------------------
    # 2. 核心训练参数（训练流程基础配置）
    # --------------------------
    # 训练总轮数（≥1，避免无效训练）
    epochs: int = Field(
        default=100,
        ge=1,
        description="训练总轮数，必须≥1"
    )
    # 批次大小（≥1，与数据集配置batch_size可联动，优先训练器配置）
    batch_size: int = Field(
        default=32,
        ge=1,
        description="训练批次大小，≥1，若与数据集配置冲突，优先使用此配置"
    )
    # 初始学习率（>0，避免无效学习率）
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="优化器初始学习率，必须>0"
    )
    # 权重衰减系数（≥0，防止过拟合）
    weight_decay: float = Field(
        default=1e-5,
        ge=0.0,
        description="权重衰减系数（L2正则），≥0"
    )

    # --------------------------
    # 3. 优化器&学习率调度器配置（训练优化核心）
    # --------------------------
    # 优化器类型（限制常用可选值，避免无效配置）
    optimizer: Literal["Adam", "AdamW", "SGD", "RMSprop"] = Field(
        default="AdamW",
        description="优化器类型：Adam/AdamW/SGD/RMSprop"
    )
    # 学习率调度器类型（可选，None表示不使用调度器）
    scheduler: Optional[Literal["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "ExponentialLR"]] = Field(
        default="StepLR",
        description="学习率调度器类型，None表示不使用学习率调度"
    )
    # 调度器参数（字典格式，需与调度器类型匹配）
    scheduler_params: dict = Field(
        default_factory=lambda: {"step_size": 30, "gamma": 0.8},
        description="学习率调度器参数，需与选定的scheduler类型匹配"
    )
    # 优化器参数（额外补充，如Adam的betas参数）
    optimizer_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="优化器额外参数（如Adam的betas=(0.9,0.999)），默认空字典"
    )

    # --------------------------
    # 4. 设备配置（兼容单卡/多卡/CPU）
    # --------------------------
    # 训练设备（自动适配，支持指定多卡）
    device: Union[str, List[str]] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="训练设备：cpu/cuda/cuda:0/['cuda:0','cuda:1']（多卡）"
    )
    # 是否使用分布式训练（多卡场景）
    use_distributed: bool = Field(
        default=False,
        description="是否使用分布式训练（多卡时建议开启）"
    )
    # 分布式训练端口（仅use_distributed=True时生效）
    dist_port: str = Field(
        default="29500",
        description="分布式训练通信端口，仅use_distributed=True时生效"
    )

    # --------------------------
    # 5. 输出&模型保存配置（结果持久化）
    # --------------------------
    # 输出根目录（模型/日志/配置文件保存路径）
    output_dir: str = Field(
        default="./outputs/trainer_output",
        description="训练结果输出根目录（保存模型、日志、配置文件等）"
    )
    # 模型保存频率（每多少轮保存一次 checkpoint，0表示仅保存最优和最后一轮）
    save_freq: int = Field(
        default=0,
        ge=0,
        description="模型保存频率（轮数），0表示仅保存最优模型和最后一轮模型"
    )
    # 是否保存最优模型（按验证集指标判断）
    save_best_model: bool = Field(
        default=True,
        description="是否根据验证集指标保存最优模型"
    )
    # 最优模型判断指标（如"accuracy"/"loss"/"f1"）
    best_model_metric: str = Field(
        default="accuracy",
        description="判断最优模型的指标（需与训练任务的评估指标一致）"
    )
    # 断点续训路径（None表示从头训练）
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="断点续训路径（指向保存的checkpoint文件），None表示从头训练"
    )

    # --------------------------
    # 6. 日志&可视化配置（训练监控）
    # --------------------------
    # 是否启用TensorBoard可视化
    use_tensorboard: bool = Field(
        default=True,
        description="是否启用TensorBoard可视化训练过程"
    )
    # TensorBoard日志保存目录（默认在output_dir下）
    tensorboard_log_dir: Optional[str] = Field(
        default=None,
        description="TensorBoard日志保存目录，None表示自动在output_dir下创建"
    )
    # 训练日志打印频率（每多少个step打印一次日志）
    log_freq: int = Field(
        default=10,
        ge=1,
        description="训练日志打印频率（步数），≥1"
    )
    # 是否保存训练日志文件（txt格式）
    save_log_file: bool = Field(
        default=True,
        description="是否将训练日志保存为txt文件（存储在output_dir下）"
    )

    # --------------------------
    # 7. 梯度优化配置（防止梯度爆炸/消失）
    # --------------------------
    # 梯度裁剪阈值（None表示不裁剪）
    gradient_clip_norm: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        description="梯度裁剪的L2范数阈值，None表示不进行梯度裁剪"
    )
    # 是否启用梯度累积（小显存场景增大有效批次）
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="梯度累积步数，≥1，显存不足时可增大此值"
    )

    # --------------------------
    # 8. 训练模式配置（精度/效率优化）
    # --------------------------
    # 是否使用混合精度训练（加速训练/节省显存）
    use_mixed_precision: bool = Field(
        default=False,
        description="是否使用混合精度训练（需CUDA支持，加速训练并节省显存）"
    )
    # 混合精度训练精度类型（fp16/bf16）
    mixed_precision_type: Literal["fp16", "bf16"] = Field(
        default="fp16",
        description="混合精度训练的精度类型，仅use_mixed_precision=True时生效"
    )

    # --------------------------
    # 9. 调试配置（快速验证流程）
    # --------------------------
    # 是否启用调试模式（仅训练少量步数/轮数验证流程）
    debug_mode: bool = Field(
        default=False,
        description="是否启用调试模式（仅训练少量数据验证流程完整性）"
    )
    # 调试模式下的最大训练步数（仅debug_mode=True时生效）
    debug_max_steps: int = Field(
        default=100,
        ge=1,
        description="调试模式下的最大训练步数，仅debug_mode=True时生效"
    )
    # 调试模式下的验证频率（仅debug_mode=True时生效）
    debug_val_freq: int = Field(
        default=10,
        ge=1,
        description="调试模式下的验证频率（步数），仅debug_mode=True时生效"
    )

    # --------------------------
    # 校验器：保证参数合法性（与数据集配置对齐）
    # --------------------------
    @validator("device")
    def validate_device(cls, v):
        """校验设备配置合法性"""
        if isinstance(v, str):
            if v not in ["cpu"] and not v.startswith("cuda"):
                raise ValueError(f"无效设备配置：{v}，支持cpu/cuda/cuda:0格式")
        elif isinstance(v, list):
            for dev in v:
                if not dev.startswith("cuda"):
                    raise ValueError(f"多卡设备仅支持cuda格式，无效设备：{dev}")
        return v

    @root_validator(skip_on_failure=True)
    def validate_scheduler_config(cls, values):
        """校验调度器与调度器参数的匹配性"""
        scheduler = values.get("scheduler")
        scheduler_params = values.get("scheduler_params")
        
        if scheduler is None:
            # 不使用调度器时，scheduler_params可为空
            return values
        
        # 常用调度器必填参数校验
        required_params = {
            "StepLR": ["step_size", "gamma"],
            "CosineAnnealingLR": ["T_max"],
            "ReduceLROnPlateau": ["mode", "patience"],
            "ExponentialLR": ["gamma"]
        }
        
        if scheduler in required_params:
            for param in required_params[scheduler]:
                if param not in scheduler_params:
                    raise ValueError(f"scheduler={scheduler}时，scheduler_params必须包含参数：{param}")
        
        return values

    @root_validator(skip_on_failure=True)
    def validate_output_dir(cls, values):
        """校验输出目录合理性"""
        output_dir = Path(values.get("output_dir"))
        # 若断点续训，校验checkpoint路径是否存在
        resume_ckpt = values.get("resume_from_checkpoint")
        if resume_ckpt is not None and not Path(resume_ckpt).exists():
            raise FileNotFoundError(f"断点续训文件不存在：{resume_ckpt}")
        
        # 自动创建输出目录（若不存在）
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 自动补全TensorBoard日志目录
        tb_log_dir = values.get("tensorboard_log_dir")
        if values.get("use_tensorboard") and tb_log_dir is None:
            tb_log_dir = str(output_dir / "tensorboard_logs")
            values["tensorboard_log_dir"] = tb_log_dir
            Path(tb_log_dir).mkdir(parents=True, exist_ok=True)
        
        return values

    @root_validator(skip_on_failure=True)
    def validate_distributed_config(cls, values):
        """校验分布式训练配置合理性"""
        use_dist = values.get("use_distributed")
        device = values.get("device")
        
        if use_dist:
            # 分布式训练必须使用多卡
            if isinstance(device, str) or (isinstance(device, list) and len(device) < 2):
                raise ValueError("使用分布式训练时，device必须配置为多卡（如['cuda:0','cuda:1']）")
            # 分布式训练需启用CUDA
            if not torch.cuda.is_available():
                raise RuntimeError("无可用CUDA设备，无法启用分布式训练")
        
        return values

    # --------------------------
    # 通用方法：与数据集配置类完全对齐（保证体系统一）
    # --------------------------
    def merge_dict(self, user_dict: Dict) -> None:
        """递归合并用户字典配置（仅覆盖指定键，保留默认值，支持嵌套配置）"""
        def _recursive_merge(default_obj: Any, user_dict: Dict):
            for k, v in user_dict.items():
                if hasattr(default_obj, k):
                    attr = getattr(default_obj, k)
                    if isinstance(attr, BaseConfig) and isinstance(v, dict):
                        _recursive_merge(attr, v)
                    else:
                        setattr(default_obj, k, v)
                else:
                    raise KeyError(f"配置键「{k}」不存在于BaseTrainerConfig中，无法合并")
        _recursive_merge(self, user_dict)

    def load_yaml(self, yaml_path: str) -> None:
        """加载YAML配置文件并递归合并（覆盖默认配置）"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"训练器YAML配置文件不存在：{yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f)
        
        if not isinstance(yaml_dict, dict):
            raise ValueError(f"YAML文件「{yaml_path}」解析后不是字典格式，无法加载")
        
        self.merge_dict(yaml_dict)

    def save_config(self, save_path: Optional[str] = None) -> None:
        """保存当前配置到YAML文件（补充数据集配置的保存功能，保持一致）"""
        if save_path is None:
            save_path = str(Path(self.output_dir) / "trainer_config.yaml")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化字典（排除pydantic内部属性）
        config_dict = self.dict(exclude_unset=True, exclude_none=False)
        
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"训练器配置已保存至：{save_path}")

    # --------------------------
    # 抽象方法：强制子类实现任务相关逻辑（与数据集配置对齐）
    # --------------------------
    @abstractmethod
    def get_train_evaluation_metrics(self) -> List[str]:
        """
        获取当前训练任务的评估指标（子类必须实现）
        示例：SFT分类任务返回["accuracy", "f1", "precision"]；生成任务返回["bleu", "rouge"]
        """
        pass

    @abstractmethod
    def validate_trainer_config(self) -> bool:
        """
        校验训练器配置的完整性与合理性（子类必须实现）
        检查：模型配置与训练器配置的兼容性、评估指标与任务匹配性等
        返回：校验通过返回True，否则抛出异常
        """
        pass