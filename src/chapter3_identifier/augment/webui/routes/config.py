from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from src.chapter3_identifier.augment.features.wind_index import should_degrade_to_dict_mode
from src.chapter3_identifier.augment.filters import normalize_inference_filter_config
from src.chapter3_identifier.augment.workflow_config import (
    apply_workflow_to_runtime_cfg,
    ensure_workflow_config,
    resolve_round_workflow,
    update_workflow_defaults,
)
from src.chapter3_identifier.augment.figures.layout import DEFAULT_LAYOUT_PROFILE, layout_protocol
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.webui.deps import AppDeps


def _sensor_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.replace("\n", ",").split(",")
    elif isinstance(value, list):
        raw = value
    else:
        raise HTTPException(status_code=422, detail="infer_exclude_sensor_ids 必须是字符串或字符串列表")
    return sorted({str(item).strip() for item in raw if str(item).strip()})


def _training_filter_config(cfg: dict) -> dict:
    return {
        "enable_sensor_exclusion": bool(cfg.get("enable_sensor_exclusion", False)),
        "exclude_sensor_ids": _sensor_ids(cfg.get("exclude_sensor_ids", [])),
    }


def _effective_sensor_ids(filters: dict) -> list[str]:
    if not bool(filters.get("enable_sensor_exclusion", False)):
        return []
    return _sensor_ids(filters.get("exclude_sensor_ids", []))


def _sensor_filter_consistency(training_filters: dict, inference_filters: dict) -> dict:
    training_ids = _effective_sensor_ids(training_filters)
    inference_ids = _effective_sensor_ids(inference_filters)
    return {
        "matched": training_ids == inference_ids,
        "training_effective_sensor_ids": training_ids,
        "inference_effective_sensor_ids": inference_ids,
    }


def _workflow_runtime_cfg(deps: AppDeps, round_idx: int | None = None) -> dict:
    ensure_workflow_config(deps.cfg)
    effective_round = int(round_idx if round_idx is not None else deps.cfg.get("webui_init_round", 1))
    resolved = resolve_round_workflow(deps.cfg, effective_round)
    return apply_workflow_to_runtime_cfg(deps.cfg, resolved)


def build_config_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/config")
    def app_config():
        runtime_cfg = _workflow_runtime_cfg(deps)
        label_names = get_label_names(deps.cfg)
        dict_mode = bool(should_degrade_to_dict_mode(deps.cfg))
        wind_runtime_mode = "dict" if dict_mode else "sql"
        layout_profile = str(deps.cfg.get("webui_layout_profile", DEFAULT_LAYOUT_PROFILE))
        infer_filters = normalize_inference_filter_config(runtime_cfg)
        training_filters = _training_filter_config(runtime_cfg)
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
            "inference_filters": infer_filters,
            "training_filters": training_filters,
            "sensor_filter_consistency": _sensor_filter_consistency(training_filters, infer_filters),
        }

    @router.get("/api/config/inference_filters")
    def get_inference_filters():
        return normalize_inference_filter_config(_workflow_runtime_cfg(deps))

    @router.put("/api/config/inference_filters")
    def put_inference_filters(payload: dict = Body(...)):
        state = deps.jobs.poll()
        if state.get("status") == "running":
            raise HTTPException(status_code=409, detail="后台任务运行中，不能修改推理过滤设置")
        updates = {
            "infer_enable_sensor_exclusion": bool(payload.get("enable_sensor_exclusion", False)),
            "infer_exclude_sensor_ids": _sensor_ids(payload.get("exclude_sensor_ids", [])),
            "infer_exclude_gold_annotations": bool(payload.get("exclude_gold_annotations", False)),
            "infer_exclude_manual_annotations": bool(payload.get("exclude_manual_annotations", False)),
        }
        update_workflow_defaults(deps.cfg, "inference", updates)
        runtime_cfg = _workflow_runtime_cfg(deps)
        deps.cfg.update(
            {
                key: runtime_cfg[key]
                for key in updates
                if key in runtime_cfg
            }
        )
        return normalize_inference_filter_config(runtime_cfg)

    return router
