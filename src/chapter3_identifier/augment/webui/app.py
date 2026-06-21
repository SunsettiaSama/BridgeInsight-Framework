from __future__ import annotations

import argparse
import atexit
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.settings import load_config, resolve_python_executable
from src.chapter3_identifier.augment.webui.deps import build_deps
from src.chapter3_identifier.augment.webui.routes.annotations import build_annotations_router
from src.chapter3_identifier.augment.webui.routes.config import build_config_router
from src.chapter3_identifier.augment.webui.routes.figures import build_figures_router
from src.chapter3_identifier.augment.webui.routes.jobs import build_jobs_router
from src.chapter3_identifier.augment.webui.routes.pages import build_pages_router
from src.chapter3_identifier.augment.webui.routes.queue import build_queue_router
from src.chapter3_identifier.augment.webui.routes.training_profiles import build_training_profiles_router
from src.data_infra.bootstrap import ensure_docker_mysql, ensure_mysql_metadata_cache

ensure_paths()

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _start_mysql_bootstrap_background(cfg: dict) -> None:
    def _run() -> None:
        try:
            print("Checking MySQL Docker service...")
            ensure_docker_mysql(max_wait_s=int(cfg.get("webui_mysql_start_timeout_s", 120)))
            print("MySQL is ready")
            print("Checking MySQL metadata cache...")
            ensure_mysql_metadata_cache(cfg)
            print("MySQL metadata is ready")
        except Exception as exc:
            print(f"MySQL bootstrap failed: {exc}")

    threading.Thread(target=_run, name="augment-mysql-bootstrap", daemon=True).start()


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    if bool(cfg.get("webui_auto_start_mysql", True)) and not bool(cfg.get("wind_force_dict_mode", False)):
        if bool(cfg.get("webui_mysql_bootstrap_background", True)):
            print("MySQL bootstrap started in background")
            _start_mysql_bootstrap_background(cfg)
        else:
            print("Checking MySQL Docker service...")
            ensure_docker_mysql(max_wait_s=int(cfg.get("webui_mysql_start_timeout_s", 120)))
            print("MySQL is ready")
            print("Checking MySQL metadata cache...")
            ensure_mysql_metadata_cache(cfg)
            print("MySQL metadata is ready")
    deps = build_deps(cfg, config_path=config_path)
    deps.figures.start()
    if bool(cfg.get("webui_warmup_on_start", True)):
        print("WebUI queue/figure warmup started in background")
        threading.Thread(
            target=deps.warmup_figure_buffer,
            name="augment-webui-warmup",
            daemon=True,
        ).start()
    atexit.register(deps.figures.shutdown)
    app = FastAPI(title="Augment Annotation WebUI")
    app.include_router(build_pages_router())
    app.include_router(build_config_router(deps))
    app.include_router(build_queue_router(deps))
    app.include_router(build_annotations_router(deps))
    app.include_router(build_figures_router(deps))
    app.include_router(build_jobs_router(deps))
    app.include_router(build_training_profiles_router(deps))

    @app.websocket("/ws/ws")
    async def ide_live_stub(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    import sys

    parser = argparse.ArgumentParser(description="Augment WebUI")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    port = args.port or int(cfg["webui_port"])
    host = str(cfg.get("webui_host", "127.0.0.1"))
    job_python = resolve_python_executable(cfg)
    app = create_app(args.config)
    print(f"Augment 标注 WebUI: http://{host}:{port}")
    print(f"后台 train/infer 使用 Python: {job_python}")
    if job_python != sys.executable:
        print(f"（WebUI 当前解释器: {sys.executable}）")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
