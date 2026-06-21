from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.chapter3_identifier.regression_forecast.webui.deps import AppDeps


def build_samples_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/samples")
    def list_samples(round_idx: int = 1, page: int = 0, page_size: int | None = None):
        size = int(page_size or deps.cfg.get("queue_page_size", 50))
        records = deps.forecasts.records(round_idx)
        start = max(0, int(page)) * size
        end = start + size
        return {
            "round_idx": round_idx,
            "page": int(page),
            "page_size": size,
            "total": len(records),
            "items": records[start:end],
        }

    @router.get("/api/samples/{sample_idx}")
    def get_sample(sample_idx: int, round_idx: int = 1):
        record = deps.forecasts.find_record(round_idx, sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        return record

    return router

