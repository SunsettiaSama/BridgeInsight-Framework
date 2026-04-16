"""
振动特征识别主流水线
====================

步骤
----
1. 数据预处理   (preprocess)       → 振动 / 风元数据 JSON
2. 全量识别     (identification)   → 预测结果 JSON
3. 特征计算     (feature_analysis) → 按类别归档的特征 JSON

用法
----
    # 使用默认配置（src/pipeline.yaml）
    python src/pipeline.py

    # 指定配置文件
    python src/pipeline.py --config path/to/my_pipeline.yaml

步骤开关
--------
在 pipeline.yaml 的 steps 节点下将对应步骤设为 false 可跳过：

    steps:
      preprocess:       true
      identification:   false   # 跳过识别，直接用已有结果
      feature_analysis: true

依赖说明
--------
- 步骤2 依赖步骤1产出的元数据文件（由数据集配置中的路径读取）
- 步骤3 依赖步骤2产出的识别结果 JSON；
  如果跳过步骤2，请在 pipeline.yaml 的 feature_analysis.result_path 中手动指定
"""

import gc
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE_YAML = Path(__file__).parent.parent / "config" / "data_pipeline.yaml"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _banner(step: str, title: str) -> None:
    logger.info("=" * 72)
    logger.info(f"{step}  {title}")
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# 步骤1：数据预处理
# ---------------------------------------------------------------------------

def run_preprocess(step_cfg: dict) -> str:
    """
    执行振动 + 风数据预处理 workflow。

    Parameters
    ----------
    step_cfg : preprocess 节点配置（当前未使用额外字段，参数均由 preprocess.yaml 提供）

    Returns
    -------
    str
        风元数据 JSON 文件路径（供步骤3使用）
    """
    from src.config.data_processer.preprocess.preprocess_config import load_preprocess_config
    from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vib
    from src.data_processer.preprocess.wind_data_io_process.workflow import run as run_wind

    vib_cfg, wind_cfg = load_preprocess_config()

    _banner("步骤1 [1/2]", "振动数据预处理（缺失率筛选 / RMS / 主频统计）")
    vib_metadata = run_vib(
        save_path=vib_cfg.filter_result_path,
        cache_path=vib_cfg.workflow_cache_path,
        use_cache=True,
        force_recompute=vib_cfg.force_recompute,
    )
    logger.info(f"完成：{len(vib_metadata)} 条振动记录 → {vib_cfg.filter_result_path}")

    _banner("步骤1 [2/2]", "风数据预处理（时间戳对齐）")
    wind_metadata = run_wind(
        vib_metadata=vib_metadata,
        save_path=wind_cfg.filter_result_path,
        cache_path=wind_cfg.workflow_cache_path,
        use_cache=True,
        force_recompute=wind_cfg.force_recompute,
        extreme_only=False,
    )
    logger.info(f"完成：{len(wind_metadata)} 条风数据记录 → {wind_cfg.filter_result_path}")

    wind_path = str(wind_cfg.filter_result_path)

    # 显式释放大对象，避免元数据列表占用内存进入步骤2
    del vib_metadata, wind_metadata
    gc.collect()

    return wind_path


# ---------------------------------------------------------------------------
# 步骤2：DL 全量识别
# ---------------------------------------------------------------------------

def run_identification(step_cfg: dict) -> str:
    """
    加载 ResCNN checkpoint，对全量数据集执行推理，保存预测结果 JSON。

    Parameters
    ----------
    step_cfg : identification 节点配置

    Returns
    -------
    str
        识别结果 JSON 文件路径（供步骤3使用）
    """
    import yaml as _yaml
    from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (
        StayCableVib2023Config,
    )
    from src.data_processer.datasets.data_factory import get_dataset
    from src.identifier.deeplearning_methods.dl_identifier import DLVibrationIdentifier
    from src.identifier.deeplearning_methods.full_dataset_runner import FullDatasetRunner

    dataset_config_path = project_root / step_cfg["dataset_config"]
    checkpoint_path     = project_root / step_cfg["checkpoint"]
    model_config_path   = project_root / step_cfg["model_config"]

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "results" / "identification_result" / f"res_cnn_full_dataset_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _banner("步骤2 [1/3]", "加载 StayCable_Vib2023 数据集")
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config = StayCableVib2023Config(**_yaml.safe_load(f))
    dataset = get_dataset(dataset_config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")
    # Dataset 内部已释放 _vib_meta_all / _wind_meta_all，此处再做一次 GC 确保回收
    gc.collect()

    _banner("步骤2 [2/3]", "加载 ResCNN 模型")
    identifier = DLVibrationIdentifier.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_type=step_cfg.get("model_type", "res_cnn"),
        model_config_path=str(model_config_path),
        num_classes=step_cfg.get("num_classes", 4),
    )
    logger.info(f"模型加载完成：{checkpoint_path.name}")

    _banner("步骤2 [3/3]", "执行全量数据集识别")
    runner = FullDatasetRunner(
        identifier=identifier,
        batch_size=step_cfg.get("batch_size", 256),
        num_workers=step_cfg.get("num_workers", 4),
    )
    predictions = runner.run(dataset)
    logger.info(f"识别完成，共 {len(predictions)} 条预测结果")

    FullDatasetRunner.save_predictions(
        path=str(output_path),
        predictions=predictions,
        dataset=dataset,
        model_info=step_cfg.get("model_info", ""),
    )
    logger.info(f"结果已保存：{output_path}")

    # 保存完毕后释放大对象：预测字典、数据集索引、DataLoader runner、GPU 模型
    del predictions, runner
    del dataset
    identifier.model.cpu()
    del identifier

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU 显存已释放")

    return str(output_path)


# ---------------------------------------------------------------------------
# 步骤3：特征计算与归档
# ---------------------------------------------------------------------------

def run_feature_analysis(
    step_cfg: dict,
    result_path: str = None,
    wind_metadata_path: str = None,
) -> None:
    """
    对识别结果做多维特征计算，按振动类别归档输出。

    Parameters
    ----------
    step_cfg           : feature_analysis 节点配置
    result_path        : 步骤2传入的识别结果 JSON 路径（可覆盖 yaml 中的配置）
    wind_metadata_path : 步骤1传入的风元数据 JSON 路径（可覆盖 yaml 中的配置）
    """
    from src.identifier.feature_analysis.run import run as fa_run
    from src.config.data_processer.preprocess.preprocess_config import load_preprocess_config

    # 识别结果路径：步骤2传入 > yaml 指定 > 自动找最新文件
    _result = result_path or step_cfg.get("result_path")
    if not _result:
        result_dir = project_root / "results" / "identification_result"
        candidates = sorted(result_dir.glob("res_cnn_full_dataset_*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"未找到识别结果文件，请先执行步骤2或在 pipeline.yaml 中指定 "
                f"feature_analysis.result_path。\n搜索路径：{result_dir}"
            )
        _result = str(candidates[-1])
        logger.info(f"自动选取识别结果：{Path(_result).name}")

    # 风元数据路径：步骤1传入 > yaml 指定 > 从 preprocess.yaml 读取
    _wind = wind_metadata_path or step_cfg.get("wind_metadata_path")
    if not _wind:
        _, wind_cfg = load_preprocess_config()
        _wind = str(wind_cfg.filter_result_path)
        logger.info(f"自动读取风元数据：{_wind}")

    output_dir  = str(project_root / step_cfg.get("output_dir", "results/enriched_stats"))
    config_yaml = str(project_root / step_cfg["config"]) if step_cfg.get("config") else None

    _banner("步骤3", "特征计算与归档")
    fa_run(
        result_path=_result,
        wind_metadata_path=_wind,
        output_dir=output_dir,
        config_yaml=config_yaml,
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main(pipeline_yaml: Path = None) -> None:
    yaml_path = pipeline_yaml or _DEFAULT_PIPELINE_YAML
    logger.info(f"加载主流水线配置：{yaml_path}")
    raw = _load_yaml(yaml_path)

    steps   = raw.get("steps", {})
    do_pre  = steps.get("preprocess",       True)
    do_id   = steps.get("identification",   True)
    do_fa   = steps.get("feature_analysis", True)

    logger.info(
        f"流水线步骤：预处理={'ON' if do_pre else 'OFF'}  "
        f"识别={'ON' if do_id else 'OFF'}  "
        f"特征计算={'ON' if do_fa else 'OFF'}"
    )

    # 步骤间依赖通过返回值传递
    wind_metadata_path: str = None
    result_path:        str = None

    if do_pre:
        wind_metadata_path = run_preprocess(raw.get("preprocess", {}))

    if do_id:
        result_path = run_identification(raw.get("identification", {}))

    if do_fa:
        run_feature_analysis(
            raw.get("feature_analysis", {}),
            result_path=result_path,
            wind_metadata_path=wind_metadata_path,
        )

    _banner("完成", "主流水线全部步骤执行完毕")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="振动特征识别主流水线")
    parser.add_argument(
        "--config", default=None,
        help=f"pipeline YAML 路径（默认：src/pipeline.yaml）",
    )
    args = parser.parse_args()
    main(Path(args.config) if args.config else None)
