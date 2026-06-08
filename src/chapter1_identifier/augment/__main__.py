from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    from src.chapter1_identifier.augment._bootstrap import ensure_paths

    ensure_paths()
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(prog="python -m src.chapter1_identifier.augment")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="DualStream 训练")
    p_train.add_argument("--round", type=int, default=1)
    p_train.add_argument("--gold-only", action="store_true", help="强制仅金标")
    p_train.add_argument("--with-manual", action="store_true", help="强制金标+人工")
    p_train.add_argument("--config", type=str, default=None)

    p_infer = sub.add_parser("infer", help="2024-09 全量识别")
    p_infer.add_argument("--round", type=int, default=1)
    p_infer.add_argument("--config", type=str, default=None)

    p_webui = sub.add_parser("webui", help="启动标注 WebUI")
    p_webui.add_argument("--port", type=int, default=None)
    p_webui.add_argument("--config", type=str, default=None)

    p_loop = sub.add_parser("loop", help="迭代编排")
    p_loop.add_argument("--max-rounds", type=int, default=10)
    p_loop.add_argument("--config", type=str, default=None)

    p_check = sub.add_parser("check-metadata", help="检查 202409 metadata")
    p_check.add_argument("--config", type=str, default=None)

    args = parser.parse_args(argv)

    if args.command == "train":
        from src.chapter1_identifier.augment.train.run import run_training
        if args.gold_only and args.with_manual:
            parser.error("--gold-only 与 --with-manual 不能同时使用")
        gold_only = True if args.gold_only else (False if args.with_manual else None)
        run_training(round_idx=args.round, gold_only=gold_only, config_path=args.config)
    elif args.command == "infer":
        from src.chapter1_identifier.augment.infer.run import run_inference
        run_inference(round_idx=args.round, config_path=args.config)
    elif args.command == "webui":
        from src.chapter1_identifier.augment.webui.app import main as webui_main
        webui_argv = []
        if args.port is not None:
            webui_argv.extend(["--port", str(args.port)])
        if args.config:
            webui_argv.extend(["--config", args.config])
        webui_main(webui_argv)
    elif args.command == "loop":
        from src.chapter1_identifier.augment.loop.orchestrator import run_loop
        run_loop(max_rounds=args.max_rounds, config_path=args.config)
    elif args.command == "check-metadata":
        from src.chapter1_identifier.augment.settings import check_inference_metadata, load_config
        cfg = load_config(args.config)
        path = check_inference_metadata(cfg)
        print(f"metadata OK: {path}")


if __name__ == "__main__":
    main()
