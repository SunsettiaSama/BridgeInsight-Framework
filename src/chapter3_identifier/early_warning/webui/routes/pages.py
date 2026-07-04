from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def build_pages_router() -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def index_page():
        return FileResponse(str(STATIC_DIR / "index.html"))

    return router
