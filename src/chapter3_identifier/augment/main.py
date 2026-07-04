from __future__ import annotations

import argparse
import json
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
    from src.chapter3_identifier.augment._bootstrap import ensure_paths

    ensure_paths()
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="augment",
        description="Augment：DualStream 训练 / 全量识别 / 标注 WebUI / 迭代循环",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="DualStream 训练")
    p_train.add_argument("--round", type=int, default=1)
    p_train.add_argument("--gold-only", action="store_true", help="强制仅金标")
    p_train.add_argument("--with-manual", action="store_true", help="强制金标+人工")
    p_train.add_argument("--config", type=str, default=None)
    p_train.add_argument("--profile", type=str, default=None)

    p_infer = sub.add_parser("infer", help="2024-09 全量识别")
    p_infer.add_argument("--round", type=int, default=1)
    p_infer.add_argument("--config", type=str, default=None)

    p_webui = sub.add_parser("webui", help="启动标注 WebUI")
    p_webui.add_argument("--port", type=int, default=None)
    p_webui.add_argument("--config", type=str, default=None)

    p_loop = sub.add_parser("loop", help="迭代编排（train + infer）")
    p_loop.add_argument("--max-rounds", type=int, default=10)
    p_loop.add_argument("--config", type=str, default=None)

    p_finalize = sub.add_parser("finalize", help="终盘整理：生成论文用 final rounds")
    p_finalize.add_argument("--from-round", type=int, default=1)
    p_finalize.add_argument("--to-round", type=int, default=None)
    p_finalize.add_argument("--canonical-round", type=int, default=None)
    p_finalize.add_argument("--overwrite-final", action="store_true")
    p_finalize.add_argument("--dry-run", action="store_true")
    p_finalize.add_argument("--config", type=str, default=None)

    p_workflow = sub.add_parser("workflow", help="全局 workflow 配置")
    workflow_sub = p_workflow.add_subparsers(dest="workflow_command", required=True)
    p_workflow_migrate = workflow_sub.add_parser("migrate", help="从 round 基准与 default.yaml 生成 workflow_config.json")
    p_workflow_migrate.add_argument("--baseline-round", type=int, default=8)
    p_workflow_migrate.add_argument("--config", type=str, default=None)
    p_workflow_diff = workflow_sub.add_parser("diff", help="比较两个 round 的 resolved workflow")
    p_workflow_diff.add_argument("--round-a", type=int, required=True)
    p_workflow_diff.add_argument("--round-b", type=int, required=True)
    p_workflow_diff.add_argument("--config", type=str, default=None)
    p_workflow_validate = workflow_sub.add_parser("validate", help="验证 round workflow 快照可复现")
    p_workflow_validate.add_argument("--round", type=int, required=True)
    p_workflow_validate.add_argument("--config", type=str, default=None)

    p_check = sub.add_parser("check-metadata", help="检查 202409 metadata")
    p_check.add_argument("--config", type=str, default=None)

    p_env = sub.add_parser("check-env", help="检查 python_executable 与 torch")
    p_env.add_argument("--config", type=str, default=None)

    sub.add_parser("smoke", help="模拟数据冒烟（标注注入 round 数据集）")

    args = parser.parse_args(argv)

    if args.command == "train":
        from src.chapter3_identifier.augment.train.run import run_training

        if args.gold_only and args.with_manual:
            parser.error("--gold-only 与 --with-manual 不能同时使用")
        gold_only = True if args.gold_only else (False if args.with_manual else None)
        result = run_training(round_idx=args.round, gold_only=gold_only, config_path=args.config, profile_path=args.profile)
        logging.getLogger(__name__).info("训练完成：%s", result)
    elif args.command == "infer":
        from src.chapter3_identifier.augment.infer.run import run_inference

        run_inference(round_idx=args.round, config_path=args.config)
    elif args.command == "webui":
        from src.chapter3_identifier.augment.webui.app import main as webui_main

        webui_argv = []
        if args.port is not None:
            webui_argv.extend(["--port", str(args.port)])
        if args.config:
            webui_argv.extend(["--config", args.config])
        webui_main(webui_argv)
    elif args.command == "loop":
        from src.chapter3_identifier.augment.loop.orchestrator import run_loop

        run_loop(max_rounds=args.max_rounds, config_path=args.config)
    elif args.command == "finalize":
        from src.chapter3_identifier.augment.finalize.run import run_finalize

        result = run_finalize(
            config_path=args.config,
            from_round=args.from_round,
            to_round=args.to_round,
            canonical_round=args.canonical_round,
            overwrite_final=args.overwrite_final,
            dry_run=args.dry_run,
        )
        logging.getLogger(__name__).info("终盘整理完成：%s", result.get("manifest_path", result))
    elif args.command == "workflow":
        from src.chapter3_identifier.augment.settings import load_config
        from src.chapter3_identifier.augment.workflow_config import (
            bootstrap_workflow_config,
            diff_round_snapshots,
            validate_round_reproducibility,
        )

        cfg = load_config(args.config)
        if args.workflow_command == "migrate":
            saved = bootstrap_workflow_config(cfg, baseline_round=args.baseline_round, write=True)
            print(json.dumps({"path": saved.get("_path"), "workflow_config_version": saved.get("workflow_config_version")}, ensure_ascii=False, indent=2))
        elif args.workflow_command == "diff":
            result = diff_round_snapshots(cfg, args.round_a, args.round_b)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.workflow_command == "validate":
            result = validate_round_reproducibility(cfg, args.round)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["checks"]["reproducible"]:
                raise SystemExit(1)
    elif args.command == "check-metadata":
        from src.chapter3_identifier.augment.settings import check_inference_metadata, load_config

        cfg = load_config(args.config)
        path = check_inference_metadata(cfg)
        print(f"metadata OK: {path}")
    elif args.command == "check-env":
        from src.chapter3_identifier.augment.settings import load_config, resolve_python_executable

        cfg = load_config(args.config)
        py = resolve_python_executable(cfg)
        print(f"python_executable: {py}")
        print(f"当前解释器:       {sys.executable}")
        import subprocess

        proc = subprocess.run(
            [py, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(proc.stdout.strip())
        else:
            print(proc.stderr.strip() or proc.stdout.strip())
            raise SystemExit(proc.returncode)
    elif args.command == "smoke":
        from src.chapter3_identifier.augment.smoke_test import run_smoke_tests

        run_smoke_tests()


if __name__ == "__main__":
    main()
