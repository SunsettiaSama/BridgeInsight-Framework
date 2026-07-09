from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.chapter4_characteristics._bootstrap import ensure_paths
from src.chapter4_characteristics.analysis.index_builder import build_others_index
from src.chapter4_characteristics.analysis.reference_builder import post_enrich_artifacts
from src.chapter4_characteristics.feature_analysis._compactor import compact_enriched_dir
from src.chapter4_characteristics.settings import (
    get_enriched_dir,
    get_predictions_enriched_path,
    load_config,
)
from src.config.identifier.feature_analysis.config import load_config as load_feature_config

ensure_paths()

from src.chapter4_characteristics.feature_analysis.run import run as feature_analysis_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXCLUDED_C34_ANOMALY_SENSOR_IDS = {
    "ST-VIC-C34-201-01",
    "ST-VIC-C34-201-02",
    "ST-VIC-C34-202-01",
    "ST-VIC-C34-202-02",
    "ST-VIC-C34-301-01",
    "ST-VIC-C34-301-02",
}


def _resolve_compact_flags(
    cfg: dict,
    compact_batches: bool | None,
    compact_force: bool | None,
) -> tuple[bool, bool]:
    do_compact = (
        bool(cfg.get("compact_enriched_batches", True))
        if compact_batches is None
        else compact_batches
    )
    force_compact = (
        bool(cfg.get("compact_enriched_force", False))
        if compact_force is None
        else compact_force
    )
    return do_compact, force_compact


def run_enrichment(
    limit: int | None = None,
    config_path: str | None = None,
    result_path: str | None = None,
    output_dir: str | None = None,
    exclude_c34_anomalies: bool = False,
    batch_size: int | None = None,
    skip_artifacts: bool = False,
    compact_batches: bool | None = None,
    compact_only: bool = False,
    compact_force: bool | None = None,
) -> dict:
    cfg = load_config(config_path)
    if limit is None and int(cfg.get("dev_limit_samples", 0)) > 0:
        limit = int(cfg["dev_limit_samples"])

    out_dir = Path(output_dir) if output_dir else get_enriched_dir(cfg)
    cfg["enriched_stats_dir"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    do_compact, force_compact = _resolve_compact_flags(cfg, compact_batches, compact_force)
    excluded = EXCLUDED_C34_ANOMALY_SENSOR_IDS if exclude_c34_anomalies else set()

    if compact_only:
        logger.info(f"开始 enriched batch 整理 → {out_dir}")
        compact_summary = compact_enriched_dir(
            out_dir,
            force=force_compact,
            excluded_sensor_ids=excluded if excluded else None,
        )
        if not skip_artifacts:
            post_enrich_artifacts(cfg)
            build_others_index(cfg)
        logger.info("整理完成")
        return {
            "output_dir": str(out_dir),
            "compact_only": True,
            "compact_summary": compact_summary,
            "exclude_c34_anomalies": exclude_c34_anomalies,
        }

    enriched_src = Path(result_path) if result_path else get_predictions_enriched_path(cfg)
    if not enriched_src.exists():
        raise FileNotFoundError(f"识别结果不存在，请先 infer：{enriched_src}")

    logger.info(f"开始特征归档 → {out_dir}")
    feature_config = cfg.get("feature_analysis_config")
    feature_cfg = load_feature_config(str(feature_config) if feature_config else None)
    if batch_size is not None and int(batch_size) > 0:
        feature_cfg.batch_size = int(batch_size)
    feature_analysis_run(
        result_path=str(enriched_src),
        wind_metadata_path=str(cfg["wind_metadata_path"]),
        output_dir=str(out_dir),
        cfg=feature_cfg,
        limit=limit,
        excluded_sensor_ids=excluded if excluded else None,
    )

    compact_summary = None
    if do_compact:
        logger.info(f"开始 enriched batch 整理 → {out_dir}")
        compact_summary = compact_enriched_dir(
            out_dir,
            force=force_compact,
            excluded_sensor_ids=excluded if excluded else None,
        )

    if not skip_artifacts:
        post_enrich_artifacts(cfg)
        build_others_index(cfg)

    logger.info("归档完成")
    return {
        "output_dir": str(out_dir),
        "limit": limit,
        "result_path": str(enriched_src),
        "exclude_c34_anomalies": exclude_c34_anomalies,
        "batch_size": int(feature_cfg.batch_size),
        "compact_summary": compact_summary,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chapter4 特征归档")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--result", type=str, default=None, help="识别结果 JSON，默认读取 chapter4 inference/predictions_enriched.json")
    parser.add_argument("--output-dir", type=str, default=None, help="enriched 输出目录，默认读取配置")
    parser.add_argument("--exclude-c34-anomalies", action="store_true", help="剔除 C34-201/202/301")
    parser.add_argument("--batch-size", type=int, default=None, help="特征计算小 batch 大小")
    parser.add_argument("--skip-artifacts", action="store_true", help="跳过 reference/others 后处理产物")
    parser.add_argument("--no-compact", action="store_true", help="跳过 enriched batch 整理")
    parser.add_argument("--compact-only", action="store_true", help="仅整理已有 batch，不重新计算特征")
    parser.add_argument("--compact-force", action="store_true", help="强制重建 canonical JSON")
    args = parser.parse_args(argv)
    run_enrichment(
        limit=args.limit,
        config_path=args.config,
        result_path=args.result,
        output_dir=args.output_dir,
        exclude_c34_anomalies=args.exclude_c34_anomalies,
        batch_size=args.batch_size,
        skip_artifacts=args.skip_artifacts,
        compact_batches=False if args.no_compact else None,
        compact_only=args.compact_only,
        compact_force=True if args.compact_force else None,
    )


if __name__ == "__main__":
    main()
