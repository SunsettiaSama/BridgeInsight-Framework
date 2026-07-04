from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _prepend_project_root() -> None:
    root = Path(__file__).resolve().parents[3]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _config_path(args) -> str | None:
    return getattr(args, "config", None)


def main(argv: list[str] | None = None) -> None:
    _prepend_project_root()
    from src.chapter3_identifier.early_warning._bootstrap import ensure_paths

    ensure_paths()
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="early_warning",
        description="Chapter3 预警识别系统",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("build-cache", "构建预测特征缓存"),
        ("check-data", "检查数据契约和可构造样本数"),
        ("train", "训练多任务风险预测模型"),
        ("infer", "生成 forecast.json 预警结果"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--round", type=int, default=1)
        cmd.add_argument("--config", type=str, default=None)

    p_webui = sub.add_parser("webui", help="启动预警识别 WebUI")
    p_webui.add_argument("--port", type=int, default=None)
    p_webui.add_argument("--config", type=str, default=None)

    args = parser.parse_args(argv)
    config_path = _config_path(args)

    if args.command == "build-cache":
        from src.chapter3_identifier.regression_forecast.features.cache import build_feature_cache

        path = build_feature_cache(round_idx=args.round, config_path=config_path)
        print(f"缓存完成：{path}")
    elif args.command == "check-data":
        from src.chapter3_identifier.regression_forecast.features.cache import check_data

        payload = check_data(round_idx=args.round, config_path=config_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "train":
        from src.chapter3_identifier.regression_forecast.train.run import run_training

        run_training(round_idx=args.round, config_path=config_path)
        print("训练完成")
    elif args.command == "infer":
        from src.chapter3_identifier.regression_forecast.infer.run import run_inference

        path = run_inference(round_idx=args.round, config_path=config_path)
        print(f"预测完成：{path}")
    elif args.command == "webui":
        from src.chapter3_identifier.early_warning.webui.app import main as webui_main

        webui_argv = []
        if args.port is not None:
            webui_argv.extend(["--port", str(args.port)])
        if config_path:
            webui_argv.extend(["--config", config_path])
        webui_main(webui_argv)


if __name__ == "__main__":
    main()
