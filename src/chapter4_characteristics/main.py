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
    argv = list(argv) if argv is not None else list(sys.argv[1:])
    if not argv:
        argv = ["webui"]
    elif argv[0].startswith("-"):
        argv = ["webui", *argv]

    parser = argparse.ArgumentParser(
        prog="chapter4_characteristics",
        description="Chapter4：2023 全量识别 / 特征归档 / 特性分析 WebUI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_webui = sub.add_parser("webui", help="启动特性分析 WebUI")
    p_webui.add_argument("--port", type=int, default=None)
    p_webui.add_argument("--config", type=str, default=None)

    p_infer = sub.add_parser("infer", help="2023 全量识别（augment 模型）")
    p_infer.add_argument("--limit", type=int, default=None)
    p_infer.add_argument("--config", type=str, default=None)

    p_enrich = sub.add_parser("enrich", help="特征归档")
    p_enrich.add_argument("--limit", type=int, default=None)
    p_enrich.add_argument("--config", type=str, default=None)
    p_enrich.add_argument("--result", type=str, default=None)
    p_enrich.add_argument("--output-dir", type=str, default=None)
    p_enrich.add_argument("--exclude-c34-anomalies", action="store_true")
    p_enrich.add_argument("--batch-size", type=int, default=None)
    p_enrich.add_argument("--skip-artifacts", action="store_true")
    p_enrich.add_argument("--no-compact", action="store_true", help="跳过 enriched batch 整理")
    p_enrich.add_argument("--compact-only", action="store_true", help="仅整理已有 batch，不重新计算特征")
    p_enrich.add_argument("--compact-force", action="store_true", help="强制重建 canonical JSON")

    p_copula = sub.add_parser("copula", help="Copula：extract / marginals / joint / run")
    p_copula.add_argument(
        "stage",
        nargs="?",
        default="run",
        choices=["extract", "marginals", "joint", "run"],
        help="流水线阶段（默认 run=extract→marginals→joint）",
    )
    p_copula.add_argument(
        "--class-id",
        dest="class_id",
        type=str,
        default="all",
        help="类别 id（0-3）或 all",
    )
    p_copula.add_argument("--config", type=str, default=None)
    p_copula.add_argument("--refresh", action="store_true", help="强制重算模态缓存")
    p_copula.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="模态提取最大样本数（仅 extract/run 有效）",
    )

    p_td = sub.add_parser(
        "td_copula",
        help="窗间时序 Copula：sequence / fit / sample / run / intra(口子)",
    )
    p_td.add_argument(
        "stage",
        nargs="?",
        default="run",
        choices=["sequence", "fit", "sample", "run", "intra"],
        help="默认 run=sequence→fit→sample；intra 为雨流口子（未实现）",
    )
    p_td.add_argument("--class-id", dest="class_id", type=str, default="all")
    p_td.add_argument("--config", type=str, default=None)
    p_td.add_argument("--refresh", action="store_true", help="强制重算 td 序列缓存")
    p_td.add_argument("--max-samples", type=int, default=None)
    p_td.add_argument("--n-paths", type=int, default=None, help="MC 轨迹条数")
    p_td.add_argument("--n-steps", type=int, default=None, help="每条轨迹窗步数")

    p_check = sub.add_parser("check-env", help="检查 python/torch 环境")
    p_check.add_argument("--config", type=str, default=None)

    p_preflight = sub.add_parser("check-preflight", help="预检数据与 checkpoint")
    p_preflight.add_argument("--config", type=str, default=None)

    p_demo_setup = sub.add_parser("demo-setup", help="生成可直接展示的 demo 数据集（含 infer/enrich 产物）")
    p_demo_setup.add_argument("--force", action="store_true")

    p_demo_webui = sub.add_parser("demo-webui", help="生成 demo 数据并直接启动 demo WebUI")
    p_demo_webui.add_argument("--force", action="store_true")
    p_demo_webui.add_argument("--port", type=int, default=None)

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

        run_inference(limit=args.limit, config_path=args.config)
    elif args.command == "enrich":
        from src.chapter4_characteristics.enrich.run import run_enrichment

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
    elif args.command == "copula":
        from src.chapter4_characteristics.analysis.copula_service import run_copula_job

        result = run_copula_job(
            class_id=args.class_id,
            config_path=args.config,
            stage=args.stage,
            refresh=args.refresh,
            max_samples=args.max_samples,
        )
        print(result)
    elif args.command == "td_copula":
        from src.chapter4_characteristics.statistics.td_copula import run_td_copula_job

        result = run_td_copula_job(
            class_id=args.class_id,
            config_path=args.config,
            stage=args.stage,
            refresh=args.refresh,
            max_samples=args.max_samples,
            n_paths=args.n_paths,
            n_steps=args.n_steps,
        )
        print(result)
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

        run_preflight(config_path=args.config)
    elif args.command == "demo-setup":
        from src.chapter4_characteristics.demo_fixtures import ensure_demo_fixtures

        output_dir = ensure_demo_fixtures(force=args.force)
        print(f"demo 输出目录：{output_dir}")
        print("demo 配置：src/chapter4_characteristics/config/demo.yaml")
    elif args.command == "demo-webui":
        from src.chapter4_characteristics.demo_fixtures import ensure_demo_fixtures
        from src.chapter4_characteristics.webui.app import main as webui_main

        ensure_demo_fixtures(force=args.force)
        demo_config = str((Path(__file__).resolve().parent / "config" / "demo.yaml").resolve())
        webui_argv = ["--config", demo_config]
        if args.port is not None:
            webui_argv.extend(["--port", str(args.port)])
        webui_main(webui_argv)


if __name__ == "__main__":
    main()
