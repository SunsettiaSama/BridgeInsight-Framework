from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _prepend_project_root() -> None:
    root = Path(__file__).resolve().parents[3]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main(argv: list[str] | None = None) -> None:
    _prepend_project_root()
    from src.chapter3_identifier.regression_forecast._bootstrap import ensure_paths

    ensure_paths()
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="regression_forecast",
        description="Chapter3 长期振动类型风险预测与指标回归",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cache = sub.add_parser("build-cache", help="构建预测特征缓存")
    p_cache.add_argument("--round", type=int, default=1)
    p_cache.add_argument("--config", type=str, default=None)

    p_check = sub.add_parser("check-data", help="检查数据契约和可构造样本数")
    p_check.add_argument("--round", type=int, default=1)
    p_check.add_argument("--config", type=str, default=None)

    p_train = sub.add_parser("train", help="训练多任务风险预测模型")
    p_train.add_argument("--round", type=int, default=1)
    p_train.add_argument("--config", type=str, default=None)

    p_infer = sub.add_parser("infer", help="生成 forecast.json")
    p_infer.add_argument("--round", type=int, default=1)
    p_infer.add_argument("--config", type=str, default=None)

    p_webui = sub.add_parser("webui", help="启动回归预测 WebUI")
    p_webui.add_argument("--port", type=int, default=None)
    p_webui.add_argument("--config", type=str, default=None)

    p_smoke = sub.add_parser("smoke", help="运行小样本 smoke 流程")
    p_smoke.add_argument("--config", type=str, default=None)

    args = parser.parse_args(argv)

    if args.command == "build-cache":
        from src.chapter3_identifier.regression_forecast.features.cache import build_feature_cache

        path = build_feature_cache(round_idx=args.round, config_path=args.config)
        print(f"缓存完成：{path}")
    elif args.command == "check-data":
        from src.chapter3_identifier.regression_forecast.features.cache import check_data

        payload = check_data(round_idx=args.round, config_path=args.config)
        print(payload)
    elif args.command == "train":
        from src.chapter3_identifier.regression_forecast.train.run import run_training

        run_training(round_idx=args.round, config_path=args.config)
        print("训练完成")
    elif args.command == "infer":
        from src.chapter3_identifier.regression_forecast.infer.run import run_inference

        path = run_inference(round_idx=args.round, config_path=args.config)
        print(f"预测完成：{path}")
    elif args.command == "webui":
        from src.chapter3_identifier.regression_forecast.webui.app import main as webui_main

        webui_argv = []
        if args.port is not None:
            webui_argv.extend(["--port", str(args.port)])
        if args.config:
            webui_argv.extend(["--config", args.config])
        webui_main(webui_argv)
    elif args.command == "smoke":
        from src.chapter3_identifier.regression_forecast.smoke import run_smoke

        run_smoke(config_path=args.config)


if __name__ == "__main__":
    main()

