from __future__ import annotations

from fastapi import APIRouter

from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.webui.deps import AppDeps


def build_config_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/config")
    def app_config():
        label_names = get_label_names(deps.cfg)
        return {
            "num_classes": int(deps.cfg.get("num_classes", len(label_names))),
            "label_names": label_names,
        }

    return router
