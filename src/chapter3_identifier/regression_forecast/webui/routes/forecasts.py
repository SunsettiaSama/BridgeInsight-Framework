from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.chapter3_identifier.regression_forecast.webui.deps import AppDeps


def _record_best_horizon(record: dict, horizon_filter: int | None = None) -> dict:
    horizons = record.get("horizons", [])
    selected = [h for h in horizons if horizon_filter is None or int(h.get("horizon_hours", -1)) == int(horizon_filter)]
    if not selected:
        selected = horizons
    if not selected:
        return {"risk_score": 0.0, "risk_class": "Normal", "horizon_hours": None}
    return max(selected, key=lambda row: float(row.get("risk_score", 0.0)))


def build_forecasts_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/forecasts")
    def list_forecasts(
        round_idx: int = 1,
        horizon_hours: int | None = None,
        risk_class: str | None = None,
        min_risk: float = 0.0,
        cable_pair: str | None = None,
        limit: int = 200,
    ):
        records = []
        for record in deps.forecasts.records(round_idx):
            best = _record_best_horizon(record, horizon_hours)
            if risk_class and str(best.get("risk_class")) != risk_class:
                continue
            if float(best.get("risk_score", 0.0)) < float(min_risk):
                continue
            cable_text = "|".join(str(x) for x in record.get("cable_pair", []))
            if cable_pair and cable_pair not in cable_text:
                continue
            records.append(
                {
                    "sample_idx": record.get("sample_idx"),
                    "timestamp": record.get("timestamp"),
                    "cable_pair": record.get("cable_pair", []),
                    "risk_class": best.get("risk_class"),
                    "risk_score": best.get("risk_score"),
                    "horizon_hours": best.get("horizon_hours"),
                    "dominant_class": best.get("dominant_class"),
                }
            )
        records.sort(key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
        max_items = max(1, int(limit))
        return {"round_idx": round_idx, "total": len(records), "items": records[:max_items]}

    @router.get("/api/forecasts/{sample_idx}")
    def get_forecast(sample_idx: int, round_idx: int = 1):
        record = deps.forecasts.find_record(round_idx, sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="forecast not found")
        return record

    return router

