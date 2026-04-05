
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import logging
import yaml
from datetime import datetime

from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (
    StayCableVib2023Config,
)
from src.config.data_processer.preprocess.preprocess_config import load_preprocess_config
from src.data_processer.datasets.data_factory import get_dataset
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib_workflow
from src.data_processer.preprocess.wind_data_io_process.workflow import run as run_wind_workflow
from src.identifier.deeplearning_methods.dl_identifier import DLVibrationIdentifier
from src.identifier.deeplearning_methods.full_dataset_runner import FullDatasetRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """
    全量数据集识别工作流：
    1. 预处理振动元数据（缺失率筛选 + RMS + 主频统计）
    2. 预处理风元数据（时间戳对齐）
    3. 加载 StayCable_Vib2023 数据集
    4. 加载最优 ResCNN 模型
    5. 对全量数据集进行识别
    6. 保存识别结果
    """
    print("This is a test")
    # -------------------------------------------------------------------------
    # 1. 配置路径
    # -------------------------------------------------------------------------

    project_root = Path(__file__).parent.parent.parent.parent

    # 从 preprocess.yaml 读取元数据路径（单一数据源，避免路径不一致）
    _vib_cfg, _wind_cfg = load_preprocess_config()
    vib_metadata_path       = _vib_cfg.filter_result_path
    vib_metadata_cache_path = _vib_cfg.workflow_cache_path
    wind_metadata_path      = _wind_cfg.filter_result_path
    wind_metadata_cache_path = _wind_cfg.workflow_cache_path

    # 数据集路径
    dataset_config_path = project_root / "config" / "train" / "datasets" / "total_staycable_vib.yaml"

    # 模型相关路径
    checkpoint_path = project_root / "results" / "training_result" / "deep_learning_module" / "res_cnn" / "checkpoints" / "ResCNN_20260402_111429" / "best_checkpoint.pth"
    model_config_path = project_root / "config" / "train" / "models" / "res_cnn.yaml"

    # 结果保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "results" / "identification_result" / f"res_cnn_full_dataset_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. 振动元数据预处理
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 1/6: 振动数据预处理（缺失率筛选 / RMS / 主频统计）")
    logger.info("=" * 80)

    vib_metadata = run_vib_workflow(
        save_path=vib_metadata_path,
        cache_path=vib_metadata_cache_path,
        use_cache=True,
        force_recompute=_vib_cfg.force_recompute,
    )
    logger.info(f"振动元数据预处理完成，共 {len(vib_metadata)} 条记录 → {vib_metadata_path}")

    # -------------------------------------------------------------------------
    # 3. 风元数据预处理
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 2/6: 风数据预处理（时间戳对齐）")
    logger.info("=" * 80)

    wind_metadata = run_wind_workflow(
        vib_metadata=vib_metadata,
        save_path=wind_metadata_path,
        cache_path=wind_metadata_cache_path,
        use_cache=True,
        force_recompute=_wind_cfg.force_recompute,
        extreme_only=False,
    )
    logger.info(f"风元数据预处理完成，共 {len(wind_metadata)} 条记录 → {wind_metadata_path}")

    # -------------------------------------------------------------------------
    # 4. 加载数据集（使用数据工厂）
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 3/6: 加载 StayCable_Vib2023 数据集")
    logger.info("=" * 80)

    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config_dict = yaml.safe_load(f)

    dataset_config = StayCableVib2023Config(**dataset_config_dict)
    dataset = get_dataset(dataset_config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")

    # -------------------------------------------------------------------------
    # 5. 加载最优 ResCNN 模型
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 4/6: 加载最优 ResCNN 模型")
    logger.info("=" * 80)

    identifier = DLVibrationIdentifier.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_type="res_cnn",
        model_config_path=str(model_config_path),
        num_classes=4,
    )
    logger.info(f"模型加载完成：checkpoint={checkpoint_path}")

    # -------------------------------------------------------------------------
    # 6. 执行全量数据集识别
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 5/6: 执行全量数据集识别")
    logger.info("=" * 80)

    runner = FullDatasetRunner(
        identifier=identifier,
        batch_size=256,
        num_workers=4,
    )

    predictions = runner.run(dataset)
    logger.info(f"识别完成，共 {len(predictions)} 条预测结果")

    # -------------------------------------------------------------------------
    # 7. 保存识别结果
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("步骤 6/6: 保存识别结果")
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

