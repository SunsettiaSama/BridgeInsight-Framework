from typing import Optional, List, Literal, Dict, Any
from pydantic import Field, validator, root_validator

from pathlib import Path
import torch

# 继承通用训练器配置基类
from .base_config import BaseTrainerConfig


class SFTTrainerConfig(BaseTrainerConfig):
    """
    SFT（监督微调）训练器专属配置类（适配SFTTrainer）
    核心特性：
    1. 继承通用训练器配置（标识/核心训练/优化器/设备等），复用基础能力；
    2. 补充SFT任务专属参数（任务类型/模型冻结/提示词/生成配置等）；
    3. 针对SFT任务的关联性校验（冻结策略/损失函数/生成参数匹配性）；
    4. 实现抽象方法，适配SFT任务的评估指标和配置校验。
    """
    # --------------------------
    # 1. SFT任务基础专属配置
    # --------------------------
    # 配置类型（关联训练器配置注册表，固定为"sft_trainer"）
    type: str = Field(
        default="sft_trainer",
        description="SFT训练器配置类型，固定为'sft_trainer'（与TRAINER_CONFIG_REGISTRY对应）"
    )
    # SFT任务类型（区分分类/生成/问答等任务）
    sft_task_type: Literal["classification", "generation", "qa", "sequence_labeling"] = Field(
        default="classification",
        description="SFT任务类型：classification(分类)/generation(文本生成)/qa(问答)/sequence_labeling(序列标注)"
    )
    # 是否使用提示词模板（生成/问答任务必备）
    use_prompt_template: bool = Field(
        default=False,
        description="是否使用提示词模板格式化输入（生成/问答任务建议开启，分类任务可关闭）"
    )
    # 提示词模板路径（use_prompt_template=True时生效）
    prompt_template_path: Optional[str] = Field(
        default=None,
        description="提示词模板文件路径（json/yaml格式），仅use_prompt_template=True时生效"
    )

    # --------------------------
    # 2. SFT模型微调专属配置（冻结/分层学习率）
    # --------------------------
    # 是否冻结模型底层参数（仅微调顶层，节省显存）
    freeze_backbone: bool = Field(
        default=False,
        description="是否冻结模型backbone/底层参数，仅微调顶层分类头/生成层"
    )
    # 冻结层名称列表（精确指定冻结层，与freeze_backbone互斥）
    freeze_layer_names: Optional[List[str]] = Field(
        default=None,
        description="精确冻结的层名称列表（如['encoder.layer.0', 'encoder.layer.1']），与freeze_backbone互斥"
    )
    # 分层学习率配置（不同层使用不同学习率，key为层名称前缀，value为学习率）
    layerwise_lr: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="分层学习率配置，如{'encoder.layer.': 1e-5, 'classifier.': 1e-4}"
    )
    # 预训练模型路径（SFT必备，加载预训练权重初始化）
    pretrained_model_path: str = Field(
        ...,  # 无默认值，必须显式配置
        description="预训练模型保存路径（本地目录或huggingface模型名），SFT训练必填"
    )
    # 是否加载预训练模型的优化器状态（断点续训时生效）
    load_optimizer_state: bool = Field(
        default=False,
        description="是否加载预训练模型对应的优化器状态，仅断点续训时建议开启"
    )

    # --------------------------
    # 3. SFT损失函数专属配置
    # --------------------------
    # 损失函数类型（适配SFT不同任务）
    loss_type: Literal["CrossEntropyLoss", "CrossEntropyLossWithIgnore", "MSELoss"] = Field(
        default="CrossEntropyLoss",
        description="SFT损失函数类型：分类任务用CrossEntropyLoss，生成任务用CrossEntropyLossWithIgnore"
    )
    # 忽略的标签值（CrossEntropyLossWithIgnore时生效，通常为-100）
    ignore_index: int = Field(
        default=-100,
        description="损失计算时忽略的标签值，仅CrossEntropyLossWithIgnore生效"
    )
    # 分类任务类别数（classification任务必填）
    num_classes: Optional[int] = Field(
        default=None,
        ge=2,
        description="SFT分类任务的类别数，仅sft_task_type='classification'时必填，需≥2"
    )

    # --------------------------
    # 4. 生成类SFT任务专属配置（generation/qa任务）
    # --------------------------
    # 最大生成长度（生成任务生效）
    max_generate_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="生成任务的最大序列长度，仅sft_task_type='generation'/'qa'时生效"
    )
    # 生成温度系数（控制生成随机性，0~1之间）
    generate_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="生成温度系数，越小生成越确定，越大随机性越强（仅生成任务生效）"
    )
    # 是否使用束搜索（生成任务生效）
    use_beam_search: bool = Field(
        default=False,
        description="生成任务是否使用束搜索，替代随机采样"
    )
    # 束搜索宽度（use_beam_search=True时生效）
    beam_size: int = Field(
        default=4,
        ge=1,
        description="束搜索宽度，仅use_beam_search=True时生效，需≥1"
    )

    # --------------------------
    # 5. 覆盖基类参数（SFT任务适配）
    # --------------------------
    # 覆盖基类：SFT模型通常较大，默认批次大小更小
    batch_size: int = Field(
        default=8,
        ge=1,
        description="SFT训练批次大小（模型显存占用高，建议小批量，≥1）"
    )
    # 覆盖基类：SFT建议开启混合精度训练，加速且节省显存
    use_mixed_precision: bool = Field(
        default=True,
        description="SFT训练建议开启混合精度，加速训练并节省显存（需CUDA支持）"
    )
    # 覆盖基类：SFT最优模型指标根据任务调整，默认accuracy
    best_model_metric: str = Field(
        default="accuracy",
        description="SFT最优模型判断指标：分类用accuracy/f1，生成用bleu/rouge"
    )

    # --------------------------
    # 校验器：保证SFT配置合法性（与数据集子类对齐）
    # --------------------------
    @root_validator(skip_on_failure=True)
    def validate_freeze_config(cls, values):
        """校验模型冻结配置的互斥性"""
        freeze_backbone = values.get("freeze_backbone")
        freeze_layer_names = values.get("freeze_layer_names")

        if freeze_backbone and freeze_layer_names is not None:
            raise ValueError("freeze_backbone与freeze_layer_names不可同时配置，二者互斥")
        return values

    @root_validator(skip_on_failure=True)
    def validate_task_config(cls, values):
        """校验SFT任务类型与关联参数的匹配性"""
        sft_task_type = values.get("sft_task_type")
        num_classes = values.get("num_classes")
        max_generate_length = values.get("max_generate_length")
        use_prompt_template = values.get("use_prompt_template")
        prompt_template_path = values.get("prompt_template_path")

        # 分类任务校验
        if sft_task_type == "classification":
            if num_classes is None or num_classes < 2:
                raise ValueError("sft_task_type='classification'时，num_classes必须指定且≥2")
        # 生成/问答任务校验
        elif sft_task_type in ["generation", "qa"]:
            if max_generate_length is None or max_generate_length < 1:
                raise ValueError(f"sft_task_type='{sft_task_type}'时，max_generate_length必须指定且≥1")
            if use_prompt_template and prompt_template_path is None:
                raise ValueError("use_prompt_template=True时，必须指定prompt_template_path")

        # 提示词模板路径校验
        if prompt_template_path is not None and not Path(prompt_template_path).exists():
            raise FileNotFoundError(f"提示词模板文件不存在：{prompt_template_path}")
        return values

    @root_validator(skip_on_failure=True)
    def validate_loss_config(cls, values):
        """校验损失函数与任务类型的匹配性"""
        sft_task_type = values.get("sft_task_type")
        loss_type = values.get("loss_type")

        if sft_task_type in ["generation", "qa"] and loss_type != "CrossEntropyLossWithIgnore":
            import warnings
            warnings.warn(f"sft_task_type='{sft_task_type}'时，建议使用'CrossEntropyLossWithIgnore'损失函数")
        return values

    @validator("pretrained_model_path")
    def validate_pretrained_model(cls, v):
        """校验预训练模型路径合法性"""
        # 支持本地路径和huggingface远程模型名
        if not Path(v).exists() and not v.count("/"):
            # 简单判断是否为huggingface模型名（如"bert-base-uncased"）
            raise ValueError(f"预训练模型路径无效：本地路径不存在，且非标准huggingface模型名：{v}")
        return v

    @validator("layerwise_lr")
    def validate_layerwise_lr(cls, v):
        """校验分层学习率配置合法性"""
        if v is not None:
            for lr in v.values():
                if lr <= 0:
                    raise ValueError(f"分层学习率必须>0，当前存在无效值：{lr}")
        return v

    # --------------------------
    # 抽象方法实现（适配SFT任务）
    # --------------------------
    def get_train_evaluation_metrics(self) -> List[str]:
        """
        SFT任务核心评估指标（根据任务类型返回对应指标）
        Returns:
            指标列表：按任务类型适配，保证评估全面性
        """
        sft_task_type = self.sft_task_type
        if sft_task_type == "classification":
            # 分类任务指标
            return ["accuracy", "f1", "precision", "recall", "auc"]
        elif sft_task_type in ["generation", "qa"]:
            # 生成/问答任务指标
            return ["bleu-1", "bleu-4", "rouge-1", "rouge-2", "rouge-l"]
        elif sft_task_type == "sequence_labeling":
            # 序列标注任务指标
            return ["f1", "precision", "recall", "entity_accuracy"]
        else:
            # 默认通用指标
            return ["accuracy", "loss"]

    def validate_trainer_config(self) -> bool:
        """
        校验SFT训练器配置的完整性与合理性（强制实现）
        检查：预训练模型有效性、设备与模型兼容性、参数关联性等
        """
        # 1. 预训练模型路径校验（已通过字段校验，此处补充模型类型校验）
        pretrained_path = self.pretrained_model_path
        try:
            # 尝试加载模型配置，验证模型有效性（轻量校验，不加载完整权重）
            from transformers import AutoConfig
            AutoConfig.from_pretrained(pretrained_path)
        except Exception as e:
            raise RuntimeError(f"预训练模型配置加载失败，模型路径无效：{pretrained_path}，错误信息：{str(e)}")

        # 2. 设备与混合精度兼容性校验
        device = self.device
        use_mixed_precision = self.use_mixed_precision
        mixed_precision_type = self.mixed_precision_type

        if use_mixed_precision:
            if isinstance(device, str) and device == "cpu":
                raise RuntimeError("CPU设备不支持混合精度训练，需关闭use_mixed_precision")
            if mixed_precision_type == "bf16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("当前CUDA设备不支持bf16精度，建议切换为fp16或关闭混合精度")

        # 3. 分层学习率与冻结配置兼容性校验
        layerwise_lr = self.layerwise_lr
        freeze_backbone = self.freeze_backbone
        if layerwise_lr and len(layerwise_lr) > 0 and freeze_backbone:
            raise ValueError("开启分层学习率时，不可同时冻结backbone，二者冲突")

        # 4. 校验通过
        print(f"SFT训练器配置校验通过：")
        print(f"- 任务类型：{self.sft_task_type}")
        print(f"- 预训练模型：{self.pretrained_model_path}")
        print(f"- 训练设备：{self.device}")
        print(f"- 批次大小：{self.batch_size}，训练轮数：{self.epochs}")
        print(f"- 评估指标：{self.get_train_evaluation_metrics()}")
        return True