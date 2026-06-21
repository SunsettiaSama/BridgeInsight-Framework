from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from src.chapter3_identifier.regression_forecast.figures.types import FIGURE_NAMES
from src.chapter3_identifier.regression_forecast.webui.deps import AppDeps


def build_figures_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/figures/{sample_idx}/{figure_name}")
    def get_figure(sample_idx: int, figure_name: str, round_idx: int = 1, metric_name: str = "rms"):
        if figure_name not in FIGURE_NAMES:
            raise HTTPException(status_code=404, detail="unknown figure")
        record = deps.forecasts.find_record(round_idx, sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        png = deps.figures.render_png(record, figure_name=figure_name, metric_name=metric_name)
        return Response(content=png, media_type="image/png")

    return router

