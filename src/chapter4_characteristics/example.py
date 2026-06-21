r"""
Chapter4 启用示例 — 复制下方命令，或在项目根目录直接运行本脚本。

所有命令均需在仓库根目录执行（F:\\Research\\Vibration Characteristics In Cable Vibration）。

────────────────────────────────────────────────────────────
一、冒烟验证（无需真实 2023 数据，推荐首次启用）
────────────────────────────────────────────────────────────

  # 1) 生成 mock 数据 / checkpoint / 配置
  python src/chapter4_characteristics/example.py smoke-setup

  # 2) 一键跑通：环境检查 → 预检 → 识别 → 特征归档
  python src/chapter4_characteristics/example.py smoke-run

  # 3) 启动 WebUI（默认 http://127.0.0.1:8766）
  python src/chapter4_characteristics/example.py smoke-webui

  # 或等价于逐步手动执行：
  python -m src.chapter4_characteristics check-env       --config src/chapter4_characteristics/config/smoke.yaml
  python -m src.chapter4_characteristics check-preflight --config src/chapter4_characteristics/config/smoke.yaml
  python -m src.chapter4_characteristics infer           --config src/chapter4_characteristics/config/smoke.yaml
  python -m src.chapter4_characteristics enrich          --config src/chapter4_characteristics/config/smoke.yaml
  python -m src.chapter4_characteristics webui           --config src/chapter4_characteristics/config/smoke.yaml

────────────────────────────────────────────────────────────
二、生产流程（2023 全量数据 + augment 训练产物）
────────────────────────────────────────────────────────────

  # 0) 编辑 src/chapter4_characteristics/config/default.yaml
  #    确认 inference_dataset_config / identifier_checkpoint_path / wind_metadata_path 等路径

  python -m src.chapter4_characteristics check-env
  python -m src.chapter4_characteristics check-preflight
  python -m src.chapter4_characteristics infer                # 全量识别
  python -m src.chapter4_characteristics enrich               # 特征归档
  python -m src.chapter4_characteristics copula --class-id 2   # 可选：Copula
  python -m src.chapter4_characteristics webui                      # 特性分析 WebUI

  # 指定端口（Windows 上若 8766 不可用，可换端口）
  python -m src.chapter4_characteristics webui --port 8767 --config src/chapter4_characteristics/config/default.yaml

────────────────────────────────────────────────────────────
三、前置依赖
────────────────────────────────────────────────────────────

  - Python 环境需安装 torch / fastapi / uvicorn 等（与 chapter3 augment 相同）
  - infer 依赖 identifier_checkpoint_path 指向的 chapter3 最终识别器
  - enrich 依赖 infer 产出的 predictions_enriched.json
  - WebUI 建议在 infer + enrich 完成后启动，以便浏览各类特性图与 Others 样本

────────────────────────────────────────────────────────────
四、Windows PowerShell 无法 conda activate 时
────────────────────────────────────────────────────────────

  现象：conda-hook.ps1 报「禁止运行脚本」→ activate 失败 → 落到 base 的 python，
        torch DLL 加载失败。

  方案 A（推荐，无需改策略）—— 直接用环境解释器或 conda run：

    D:\anaconda\envs\vib_experimental\python.exe -m src.chapter4_characteristics check-env
    conda run -n vib_experimental python -m src.chapter4_characteristics webui --config src/chapter4_characteristics/config/smoke.yaml

  方案 B —— 改用 CMD（activate.bat 不受 ps1 策略限制）：

    cmd /k "D:\anaconda\Scripts\activate.bat vib_experimental"

  方案 C —— 放宽当前用户脚本策略（需管理员/组策略未锁定时）：

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    conda init powershell    # 重启终端后 conda activate vib_experimental 可用
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepend_root() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _run_module(*args: str) -> None:
    cmd = [sys.executable, "-m", "src.chapter4_characteristics", *args]
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(_project_root()))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


SMOKE_CONFIG = "src/chapter4_characteristics/config/smoke.yaml"
DEFAULT_CONFIG = "src/chapter4_characteristics/config/default.yaml"


def cmd_guide(_: argparse.Namespace) -> None:
    print(__doc__)


def cmd_smoke_setup(_: argparse.Namespace) -> None:
    _prepend_root()
    from src.chapter4_characteristics.smoke_fixtures import ensure_smoke_fixtures

    data_dir = ensure_smoke_fixtures(force=False)
    print(f"[example] 冒烟夹具就绪：{data_dir}")
    print(f"[example] 配置文件：{SMOKE_CONFIG}")


def cmd_smoke_run(args: argparse.Namespace) -> None:
    cmd_smoke_setup(args)
    _run_module("check-env", "--config", SMOKE_CONFIG)
    _run_module("check-preflight", "--config", SMOKE_CONFIG)
    infer_args = ["infer", "--config", SMOKE_CONFIG]
    if args.limit is not None:
        infer_args.extend(["--limit", str(args.limit)])
    _run_module(*infer_args)
    _run_module("enrich", "--config", SMOKE_CONFIG)
    print("[example] 冒烟流水线完成，可执行：python src/chapter4_characteristics/example.py smoke-webui")


def cmd_smoke_webui(args: argparse.Namespace) -> None:
    webui_args = ["webui", "--config", SMOKE_CONFIG]
    if args.port is not None:
        webui_args.extend(["--port", str(args.port)])
    _run_module(*webui_args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="Chapter4_example",
        description="Chapter4 启用示例：冒烟验证与命令参考",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "子命令 smoke-setup / smoke-run / smoke-webui 用于本地快速验证；"
            "guide 打印完整说明。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_guide = sub.add_parser("guide", help="打印启用说明（同模块 docstring）")
    p_guide.set_defaults(func=cmd_guide)

    p_setup = sub.add_parser("smoke-setup", help="生成/刷新冒烟夹具")
    p_setup.set_defaults(func=cmd_smoke_setup)

    p_run = sub.add_parser("smoke-run", help="冒烟：setup → check → infer → enrich")
    p_run.add_argument("--limit", type=int, default=None, help="infer 样本上限（默认用 smoke.yaml 的 dev_limit_samples）")
    p_run.set_defaults(func=cmd_smoke_run)

    p_webui = sub.add_parser("smoke-webui", help="以 smoke 配置启动 WebUI")
    p_webui.add_argument("--port", type=int, default=None)
    p_webui.set_defaults(func=cmd_smoke_webui)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
