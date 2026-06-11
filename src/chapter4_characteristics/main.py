from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _prepend_project_root() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main(argv: list[str] | None = None) -> None:
    _prepend_project_root()
    from src.chapter4_characteristics._bootstrap import ensure_paths

    ensure_paths()
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="chapter4_characteristics",
        description="Chapter4：2023 全量识别 / 特征归档 / 特性分析 WebUI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_webui = sub.add_parser("webui", help="启动特性分析 WebUI")
    p_webui.add_argument("--port", type=int, default=None)
    p_webui.add_argument("--config", type=str, default=None)

    p_infer = sub.add_parser("infer", help="2023 全量识别（augment 模型）")
    p_infer.add_argument("--round", type=int, default=1)
    p_infer.add_argument("--limit", type=int, default=None)
    p_infer.add_argument("--config", type=str, default=None)

    p_enrich = sub.add_parser("enrich", help="特征归档")
    p_enrich.add_argument("--round", type=int, default=1)
    p_enrich.add_argument("--limit", type=int, default=None)
    p_enrich.add_argument("--config", type=str, default=None)

    p_copula = sub.add_parser("copula", help="Copula 拟合")
    p_copula.add_argument("--round", type=int, default=1)
    p_copula.add_argument("--class-id", dest="class_id", type=int, default=0)
    p_copula.add_argument("--config", type=str, default=None)

    p_check = sub.add_parser("check-env", help="检查 python/torch 环境")
    p_check.add_argument("--config", type=str, default=None)

    p_preflight = sub.add_parser("check-preflight", help="预检数据与 checkpoint")
    p_preflight.add_argument("--round", type=int, default=1)
    p_preflight.add_argument("--config", type=str, default=None)

    args = parser.parse_args(argv)

    if args.command == "webui":
        from src.chapter4_characteristics.webui.app import main as webui_main

        webui_argv = []
        if args.port is not None:
            webui_argv.extend(["--port", str(args.port)])
        if args.config:
            webui_argv.extend(["--config", args.config])
        webui_main(webui_argv)
    elif args.command == "infer":
        from src.chapter4_characteristics.infer.run import run_inference

        run_inference(round_idx=args.round, limit=args.limit, config_path=args.config)
    elif args.command == "enrich":
        from src.chapter4_characteristics.enrich.run import run_enrichment

        run_enrichment(round_idx=args.round, limit=args.limit, config_path=args.config)
    elif args.command == "copula":
        from src.chapter4_characteristics.analysis.copula_service import run_copula_job

        run_copula_job(round_idx=args.round, class_id=args.class_id, config_path=args.config)
    elif args.command == "check-env":
        from src.chapter4_characteristics.settings import load_config, resolve_python_executable

        cfg = load_config(args.config)
        py = resolve_python_executable(cfg)
        print(f"python_executable: {py}")
        print(f"当前解释器:       {sys.executable}")
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
    elif args.command == "check-preflight":
        from src.chapter4_characteristics.infer.preflight import run_preflight

        run_preflight(round_idx=args.round, config_path=args.config)


if __name__ == "__main__":
    main()
