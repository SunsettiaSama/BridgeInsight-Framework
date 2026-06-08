"""
识别结果元数据补全脚本

功能：
- 加载现有识别结果 JSON（旧格式，缺少 sample_metadata）
- 加载数据集并读取样本信息
- 将识别结果与样本元数据逐一对应
- 生成完整结果 JSON（含 sample_metadata）
- 支持追溯每个窗口的完整信息（传感器、时间、缺失率等）

优势：
- 无需重新运行耗时的识别过程
- 直接利用现有识别结果和缓存数据集
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import logging
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict

import yaml

from src.config.data_processer.datasets.StayCableVib2023Dataset.StayCableVib2023Config import (
    StayCableVib2023Config,
)
from src.data_processer.datasets.data_factory import get_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_old_result(result_path: str) -> Dict[str, Any]:
    """
    加载旧格式的识别结果 JSON（不含 sample_metadata）。
    
    Parameters
    ----------
    result_path : str
        识别结果 JSON 文件路径
    
    Returns
    -------
    Dict[str, Any]
        包含 metadata、predictions、by_file 的字典
    """
    logger.info(f"加载现有识别结果：{result_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    logger.info(f"  - 样本总数: {result.get('metadata', {}).get('num_samples', 'N/A')}")
    logger.info(f"  - 生成时间: {result.get('metadata', {}).get('created_at', 'N/A')}")
    
    return result


def build_sample_metadata_mapping(dataset) -> Dict[int, Dict[str, Any]]:
    """
    从数据集构建 sample_idx → 元数据 的映射。
    
    Parameters
    ----------
    dataset : StayCableVib2023Dataset
        加载的数据集
    
    Returns
    -------
    Dict[int, Dict[str, Any]]
        格式: {idx: {cable_pair, timestamp, window_idx, ...}}
    """
    logger.info(f"构建样本元数据映射（共 {len(dataset._samples)} 个样本）...")
    
    mapping = {}
    for idx, rec in enumerate(dataset._samples):
        mapping[idx] = {
            "cable_pair":          list(rec.cable_pair),
            "timestamp":           list(rec.timestamp_key),  # (month, day, hour)
            "window_idx":          rec.window_idx,
            "inplane_sensor_id":   rec.inplane_meta.get("sensor_id"),
            "outplane_sensor_id":  rec.outplane_meta.get("sensor_id"),
            "inplane_file_path":   rec.inplane_meta.get("file_path"),
            "outplane_file_path":  rec.outplane_meta.get("file_path"),
            "missing_rate_in":     rec.inplane_meta.get("missing_rate"),
            "missing_rate_out":    rec.outplane_meta.get("missing_rate"),
            "has_wind":            rec.wind_meta is not None,
        }
    
    logger.info("元数据映射构建完成")
    return mapping


def enrich_result_with_metadata(
    old_result: Dict[str, Any],
    sample_metadata_mapping: Dict[int, Dict[str, Any]],
    dataset_fingerprint_hash: str,
) -> Dict[str, Any]:
    """
    将识别结果与样本元数据合并。
    
    Parameters
    ----------
    old_result : Dict[str, Any]
        旧格式识别结果（不含 sample_metadata）
    sample_metadata_mapping : Dict[int, Dict[str, Any]]
        样本元数据映射
    dataset_fingerprint_hash : str
        数据集指纹哈希（用于验证一致性）
    
    Returns
    -------
    Dict[str, Any]
        扩充后的识别结果（含 sample_metadata）
    """
    logger.info("补全样本元数据...")
    
    # 转换 predictions 键为整数
    predictions = {int(k): int(v) for k, v in old_result["predictions"].items()}
    
    # 构建 sample_metadata，使用字符串键（JSON 兼容）
    sample_metadata = {}
    missing_count = 0
    for idx in predictions.keys():
        if idx in sample_metadata_mapping:
            sample_metadata[str(idx)] = sample_metadata_mapping[idx]
        else:
            logger.warning(f"样本 {idx} 缺少元数据映射")
            missing_count += 1
    
    if missing_count > 0:
        logger.warning(f"共 {missing_count} 个样本缺少元数据映射")
    
    # 构建新结果
    enriched_result = {
        "metadata": old_result.get("metadata", {}).copy(),
        "predictions": {str(k): int(v) for k, v in predictions.items()},
        "sample_metadata": sample_metadata,
        "by_file": old_result.get("by_file", {}),
    }
    
    # 更新生成时间和备注
    enriched_result["metadata"]["enriched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enriched_result["metadata"]["enrichment_note"] = "Sample metadata补全于识别后，无需重新识别"
    enriched_result["metadata"]["dataset_fingerprint_hash"] = dataset_fingerprint_hash
    
    logger.info(f"元数据补全完成，共补全 {len(sample_metadata)} 个样本")
    
    return enriched_result


def save_enriched_result(
    enriched_result: Dict[str, Any],
    output_path: str,
) -> None:
    """
    保存扩充后的识别结果。
    
    Parameters
    ----------
    enriched_result : Dict[str, Any]
        扩充后的识别结果
    output_path : str
        保存路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存扩充后的结果：{output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"保存完成")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent.parent.parent
    
    # ---- 1. 定位识别结果文件 ----
    result_dir = project_root / "results" / "identification_result"
    if not result_dir.exists():
        logger.error(f"识别结果目录不存在：{result_dir}")
        return
    
    result_files = sorted(result_dir.glob("res_cnn_full_dataset_*.json"))
    if not result_files:
        logger.error("未找到识别结果文件")
        return
    
    # 选择最新的结果文件
    result_path = result_files[-1]
    logger.info(f"选择最新的识别结果文件：{result_path.name}")
    
    # ---- 2. 加载旧格式识别结果 ----
    old_result = load_old_result(str(result_path))
    
    # ---- 3. 加载数据集 ----
    logger.info("=" * 80)
    logger.info("步骤 1/3: 加载数据集")
    logger.info("=" * 80)
    
    dataset_config_path = project_root / "config" / "train" / "datasets" / "total_staycable_vib.yaml"
    with open(dataset_config_path, "r", encoding="utf-8") as f:
        dataset_config_dict = yaml.safe_load(f)
    
    dataset_config = StayCableVib2023Config(**dataset_config_dict)
    dataset = get_dataset(dataset_config)
    logger.info(f"数据集加载完成，共 {len(dataset)} 个样本")
    
    # ---- 4. 构建元数据映射 ----
    logger.info("=" * 80)
    logger.info("步骤 2/3: 构建样本元数据映射")
    logger.info("=" * 80)
    
    sample_metadata_mapping = build_sample_metadata_mapping(dataset)
    
    # 计算数据集指纹
    dataset_fingerprint = dataset._compute_fingerprint()
    dataset_fingerprint_hash = dataset._fingerprint_hash(dataset_fingerprint)
    logger.info(f"数据集指纹: {dataset_fingerprint_hash}")
    
    # ---- 5. 补全识别结果 ----
    logger.info("=" * 80)
    logger.info("步骤 3/3: 补全识别结果")
    logger.info("=" * 80)
    
    enriched_result = enrich_result_with_metadata(
        old_result, 
        sample_metadata_mapping,
        dataset_fingerprint_hash,
    )
    
    # ---- 6. 保存补全后的结果 ----
    # 生成输出文件名（在原文件名基础上添加 _enriched 后缀）
    output_filename = result_path.stem + "_enriched.json"
    output_path = result_dir / output_filename
    
    save_enriched_result(enriched_result, str(output_path))
    
    # ---- 7. 验证和统计 ----
    logger.info("=" * 80)
    logger.info("验证和统计")
    logger.info("=" * 80)
    
    predictions_count = len(enriched_result["predictions"])
    metadata_count = len(enriched_result["sample_metadata"])
    
    logger.info(f"预测结果数: {predictions_count}")
    logger.info(f"元数据数: {metadata_count}")
    
    if predictions_count == metadata_count:
        logger.info("✓ 预测结果与元数据完全对应")
    else:
        logger.warning(f"⚠ 预测结果与元数据数量不匹配（差异: {abs(predictions_count - metadata_count)}）")
    
    # 统计各月份
    monthly_counts = defaultdict(int)
    for idx_str, meta in enriched_result["sample_metadata"].items():
        month = meta["timestamp"][0]
        monthly_counts[month] += 1
    
    logger.info("\n月份分布：")
    for month in sorted(monthly_counts.keys()):
        logger.info(f"  {month:2d} 月: {monthly_counts[month]:8d} 个样本")
    
    logger.info("\n" + "=" * 80)
    logger.info("补全完成！")
    logger.info(f"输出文件：{output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
