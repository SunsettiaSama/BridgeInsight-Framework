from __future__ import annotations

import argparse
import logging

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.infer.run import run_inference
from src.chapter3_identifier.augment.loop.job_state import read_job_state, write_job_state
from src.chapter3_identifier.augment.settings import load_config
from src.chapter3_identifier.augment.train.run import run_training

ensure_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_loop(max_rounds: int = 10, config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    job_path = cfg["job_state_path"]

    for round_idx in range(1, max_rounds + 1):
        logger.info(f"=== Round {round_idx}: 训练 ===")
        write_job_state(job_path, {"status": "running", "phase": "train", "round": round_idx, "error": None})
        run_training(round_idx=round_idx, gold_only=None, config_path=config_path)
        write_job_state(job_path, {"status": "done", "phase": "train", "round": round_idx, "error": None})

        logger.info(f"=== Round {round_idx}: 全量识别 ===")
        write_job_state(job_path, {"status": "running", "phase": "infer", "round": round_idx, "error": None})
        run_inference(round_idx=round_idx, config_path=config_path)
        write_job_state(job_path, {"status": "idle", "phase": "infer", "round": round_idx, "error": None})

        logger.info(
            f"Round {round_idx} 完成。请启动 WebUI 进行人工标注，"
            f"标注完成后再次运行 loop 或手动 train/infer。"
        )
        break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Augment 迭代编排")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    run_loop(max_rounds=args.max_rounds, config_path=args.config)


if __name__ == "__main__":
    main()
