from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.chapter3_identifier.early_warning._bootstrap import ensure_paths
from src.chapter3_identifier.early_warning.settings import load_config, resolve_python_executable
from src.chapter3_identifier.early_warning.webui.deps import build_deps
from src.chapter3_identifier.early_warning.webui.routes.config import build_config_router
from src.chapter3_identifier.early_warning.webui.routes.figures import build_figures_router
from src.chapter3_identifier.early_warning.webui.routes.jobs import build_jobs_router
from src.chapter3_identifier.early_warning.webui.routes.pages import build_pages_router
from src.chapter3_identifier.early_warning.webui.routes.warnings import build_warnings_router

ensure_paths()

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    deps = build_deps(cfg, config_path=config_path)
    app = FastAPI(title="Early Warning WebUI")
    app.include_router(build_pages_router())
    app.include_router(build_config_router(deps))
    app.include_router(build_warnings_router(deps))
    app.include_router(build_figures_router(deps))
    app.include_router(build_jobs_router(deps))
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Early Warning WebUI")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    host = str(cfg.get("webui_host", "127.0.0.1"))
    port = int(args.port or cfg.get("webui_port", 8776))
    job_python = resolve_python_executable(cfg)
    print(f"预警识别 WebUI: http://{host}:{port}")
    print(f"后台任务使用 Python: {job_python}")
    uvicorn.run(create_app(args.config), host=host, port=port)


if __name__ == "__main__":
    main()
