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
            "title": "预警识别系统",
            "webui_init_round": active_round,
            "class_names": deps.cfg.get("class_names", []),
            "monitor_classes": list(deps.policy.monitor_classes),
            "horizons_hours": deps.cfg.get("horizons_hours", []),
            "display_horizons": deps.cfg.get("display_horizons", deps.cfg.get("horizons_hours", [])),
            "metric_names": deps.cfg.get("target_metric_names", []),
            "warning_levels": deps.policy.levels_payload(),
            "layout_profile": deps.cfg.get("webui_layout_profile", "wide_fill_v2"),
            "feature_cache_mode": str(deps.cfg.get("feature_cache_mode", "real")),
            "wind_features_display_order": deps.cfg.get("wind_features_display_order", []),
            "vibration_features_display_order": deps.cfg.get("vibration_features_display_order", []),
            "girder_features_display_order": deps.cfg.get("girder_features_display_order", []),
            "data_status": deps.data_status(active_round),
        }

    return router
