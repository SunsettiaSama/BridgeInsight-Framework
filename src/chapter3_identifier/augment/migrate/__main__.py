from __future__ import annotations

import argparse
import logging

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.migrate.build_pair_key_dataset import migrate_round_dataset
from src.chapter3_identifier.augment.settings import (
    get_round_dir,
    get_round_inference_path,
    get_round_merged_training_pair_key_path,
    get_round_pair_key_migration_report_path,
    load_config,
)

ensure_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="从 merged_training.json 派生 pair_key schema v2 数据集")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--merged-training", type=str, default=None)
    parser.add_argument("--inference", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--enable-prediction-fill", action="store_true")
    parser.add_argument("--disable-gold-fill", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    round_idx = int(args.round)
    merged_path = args.merged_training or str(get_round_dir(cfg, round_idx) / "merged_training.json")
    output_path = args.output or str(get_round_merged_training_pair_key_path(cfg, round_idx))
    report_path = args.report or str(get_round_pair_key_migration_report_path(cfg, round_idx))
    inference_path = args.inference
    if inference_path is None and round_idx > 1:
        inference_path = str(get_round_inference_path(cfg, round_idx - 1))

    report = migrate_round_dataset(
        merged_training_path=merged_path,
        output_path=output_path,
        report_path=report_path,
        inference_path=inference_path,
        num_classes=int(cfg.get("num_classes", 4)),
        enable_prediction_fill=bool(args.enable_prediction_fill),
        enable_gold_fill=not bool(args.disable_gold_fill),
    )
    logger.info(
        "pair_key 派生完成：round=%s pair_total=%s same_file_skipped=%s outplane_anchor_skipped=%s report=%s",
        round_idx,
        report["build_stats"]["pair_total"],
        report["build_stats"]["same_file_pair"],
        report["build_stats"]["outplane_anchor"],
        report_path,
    )


if __name__ == "__main__":
    main()
