from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.chapter3_identifier.early_warning.warning.service import list_warnings, record_to_warning_detail
from src.chapter3_identifier.early_warning.webui.deps import AppDeps


def build_warnings_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/warnings")
    def list_warning_records(
        round_idx: int = 1,
        horizon_hours: int | None = None,
        warning_level: str | None = None,
        risk_class: str | None = None,
        min_risk: float = 0.0,
        cable_pair: str | None = None,
        limit: int = 200,
    ):
        records = deps.forecasts.records(round_idx)
        items = list_warnings(
            records,
            deps.policy,
            [str(x) for x in deps.cfg.get("class_names", [])],
            horizon_hours=horizon_hours,
            warning_level=warning_level,
            risk_class=risk_class,
            min_risk=min_risk,
            cable_pair=cable_pair,
            limit=limit,
            wind_order=[str(x) for x in deps.cfg.get("wind_features_display_order", [])],
            vibration_order=[str(x) for x in deps.cfg.get("vibration_features_display_order", [])],
            girder_order=[str(x) for x in deps.cfg.get("girder_features_display_order", [])],
        )
        return {"round_idx": round_idx, "total": len(items), "items": items}

    @router.get("/api/warnings/{sample_idx}")
    def get_warning(sample_idx: int, round_idx: int = 1):
        record = deps.forecasts.find_record(round_idx, sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="warning not found")
        return record_to_warning_detail(
            record,
            deps.policy,
            [str(x) for x in deps.cfg.get("class_names", [])],
            wind_order=[str(x) for x in deps.cfg.get("wind_features_display_order", [])],
            vibration_order=[str(x) for x in deps.cfg.get("vibration_features_display_order", [])],
            girder_order=[str(x) for x in deps.cfg.get("girder_features_display_order", [])],
        )

    return router
