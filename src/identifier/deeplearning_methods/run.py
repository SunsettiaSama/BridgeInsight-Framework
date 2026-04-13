
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import logging
import yaml
from datetime import datetime

from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (
    StayCableVib2023Config,
)
from src.data_processer.datasets.data_factory import get_dataset
from src.identifier.deeplearning_methods.dl_identifier import DLVibrationIdentifier
from src.identifier.deeplearning_methods.full_dataset_runner import FullDatasetRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """
    全量数据集识别工作流（仅识别，预处理须提前通过 example/run_preprocess.py 完成）：
    1. 加载 StayCable_Vib2023 数据集（依赖预处理产出的元数据文件）
    2. 加载最优 ResCNN 模型
    3. 对全量数据集进行识别
    4. 保存识别结果
    """
    # -------------------------------------------------------------------------
    # 1. 配置路径
    # -------------------------------------------------------------------------
    project_root = Path(__file__).parent.parent.parent.parent

    dataset_config_path = project_root / "config" / "identifier" / "dl_identifier" / "total_staycable_vib.yaml"
    checkpoint_path     = project_root / "results" / "training_result" / "deep_learning_module" / "res_cnn" / "checkpoints" / "ResCNN_20260402_111429" / "best_checkpoint.pth"
    model_config_path   = project_root / "config" / "train" / "models" / "res_cnn.yaml"

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "results" / "identification_result" / f"res_cnn_full_dataset_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. 加载数据集（使用数据工厂）
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 1/4: 加载 StayCable_Vib2023 数据集")
    logger.info("=" * 80)

    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config_dict = yaml.safe_load(f)

    dataset_config = StayCableVib2023Config(**dataset_config_dict)
    dataset = get_dataset(dataset_config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")

    # -------------------------------------------------------------------------
    # 3. 加载最优 ResCNN 模型
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 2/4: 加载最优 ResCNN 模型")
    logger.info("=" * 80)

    identifier = DLVibrationIdentifier.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_type="res_cnn",
        model_config_path=str(model_config_path),
        num_classes=4,
    )
    logger.info(f"模型加载完成：checkpoint={checkpoint_path}")

    # -------------------------------------------------------------------------
    # 4. 执行全量数据集识别
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 3/4: 执行全量数据集识别")
    logger.info("=" * 80)

    runner = FullDatasetRunner(
        identifier=identifier,
        batch_size=256,
        num_workers=4,
    )

    predictions = runner.run(dataset)
    logger.info(f"识别完成，共 {len(predictions)} 条预测结果")

    # -------------------------------------------------------------------------
    # 5. 保存识别结果
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 4/4: 保存识别结果")
    logger.info("=" * 80)

    model_info = (
        f"ResCNN (epoch 36, val_acc=0.9920, params=66306) | "
        f"checkpoint=ResCNN_20260402_111429"
    )

    FullDatasetRunner.save_predictions(
        path=str(output_path),
        predictions=predictions,
        dataset=dataset,
        model_info=model_info,
    )

    logger.info("=" * 80)
    logger.info("全量数据集识别工作流执行完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

