from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _prepend_project_root() -> None:
    root = Path(__file__).resolve().parents[3]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main(argv: list[str] | None = None) -> None:
    _prepend_project_root()
    from src.chapter3_identifier.augment_eval_compare._bootstrap import ensure_paths

    ensure_paths()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="augment_eval_compare",
        description="Augment 后续：解耦 vs 联合模型在固定 pair 标注集上的性能对比",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("run", "推理 + 指标 + 报告"),
        ("report", "基于已有 predictions 重算 compare_report.json"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--round", type=int, default=None)
        cmd.add_argument("--config", type=str, default=None)
        cmd.add_argument(
            "--eval-split",
            type=str,
            choices=("val", "all_pairs"),
            default=None,
            help="覆盖配置中的 eval_split",
        )

    args = parser.parse_args(argv)

    from src.chapter3_identifier.augment_eval_compare.dataset.eval_pairs import build_eval_dataset
    from src.chapter3_identifier.augment_eval_compare.infer.run_compare import run_compare
    from src.chapter3_identifier.augment_eval_compare.report.build_report import build_compare_report
    from src.chapter3_identifier.augment_eval_compare.settings import (
        get_pair_key_dataset_path,
        load_compare_config,
    )

    cfg = load_compare_config(args.config)
    round_idx = int(args.round if args.round is not None else cfg.get("round_idx", 9))
    if args.eval_split is not None:
        cfg["eval_split"] = args.eval_split

    pair_key_path = get_pair_key_dataset_path(cfg, round_idx)
    if not pair_key_path.exists():
        raise FileNotFoundError(f"pair 标注集不存在：{pair_key_path}")

    eval_split = cfg.get("eval_split", "val")
    dataset = build_eval_dataset(
        pair_key_path,
        cfg["_augment_cfg"],
        eval_split=eval_split,
    )
    logging.info(
        "评估集：round=%s split=%s pairs=%s path=%s",
        round_idx,
        eval_split,
        len(dataset.entries),
        pair_key_path,
    )

    if args.command == "run":
        run_compare(cfg, round_idx, dataset)
        report_path = build_compare_report(cfg, round_idx)
        print(f"对比完成：{report_path}")
    elif args.command == "report":
        report_path = build_compare_report(cfg, round_idx)
        print(f"报告已生成：{report_path}")


if __name__ == "__main__":
    main()
