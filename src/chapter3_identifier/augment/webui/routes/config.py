from __future__ import annotations

from fastapi import APIRouter

from src.chapter3_identifier.augment.features.wind_index import should_degrade_to_dict_mode
from src.chapter3_identifier.augment.figures.layout import DEFAULT_LAYOUT_PROFILE, layout_protocol
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.webui.deps import AppDeps


def build_config_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/config")
    def app_config():
        label_names = get_label_names(deps.cfg)
        dict_mode = bool(should_degrade_to_dict_mode(deps.cfg))
        wind_runtime_mode = "dict" if dict_mode else "sql"
        layout_profile = str(deps.cfg.get("webui_layout_profile", DEFAULT_LAYOUT_PROFILE))
        return {
            "num_classes": int(deps.cfg.get("num_classes", len(label_names))),
            "label_names": label_names,
            "webui_init_round": int(deps.cfg.get("webui_init_round", 1)),
            "webui_layout_profile": layout_profile,
            "figure_layout_protocol": layout_protocol(layout_profile),
            "wind_runtime_mode": wind_runtime_mode,
            "wind_runtime_mode_label": "字典启动" if dict_mode else "SQL 启动",
            "wind_dataset_tag": str(deps.cfg.get("wind_dataset_tag", "202409")),
            "inference_dataset_config": str(deps.cfg.get("inference_dataset_config", "")),
        }

    return router
