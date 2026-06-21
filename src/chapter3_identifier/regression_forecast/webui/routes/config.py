from __future__ import annotations

from fastapi import APIRouter

from src.chapter3_identifier.regression_forecast.webui.deps import AppDeps


def build_config_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/config")
    def app_config(round_idx: int | None = None):
        active_round = int(round_idx or deps.cfg.get("webui_init_round", 1))
        return {
            "module": "regression_forecast",
            "webui_init_round": active_round,
            "class_names": deps.cfg.get("class_names", []),
            "horizons_hours": deps.cfg.get("horizons_hours", []),
            "metric_names": deps.cfg.get("target_metric_names", []),
            "layout_profile": deps.cfg.get("webui_layout_profile", "wide_fill_v2"),
            "data_status": deps.data_status(active_round),
        }

    return router

