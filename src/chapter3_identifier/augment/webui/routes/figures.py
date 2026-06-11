from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from src.chapter3_identifier.augment.figures import FigureNotReadyError, SAMPLE_FIGURE_NAMES
from src.chapter3_identifier.augment.webui.deps import AppDeps


class PreloadRequest(BaseModel):
    sample_indices: list[int]
    direction: str = "inplane"
    both_directions: bool = True
    layout_profile: str = "wide_fill_v1"
    round_idx: int = 1
    priority: bool = False


def build_figures_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/figures/{sample_idx}/{figure_name}")
    def get_figure(
        sample_idx: int,
        figure_name: str,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v1",
        direction: str = "inplane",
    ):
        if figure_name not in SAMPLE_FIGURE_NAMES:
            raise HTTPException(status_code=404, detail=f"figure {figure_name} not found")
        record = deps.find_record(sample_idx, round_idx=round_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        try:
            png = deps.figures.get_sample_png(
                record,
                figure_name,
                layout_profile=layout_profile,
                prediction_direction=direction if direction in ("inplane", "outplane") else "inplane",
            )
        except FigureNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=png, media_type="image/png")

    @router.get("/api/figures/{sample_idx}/context/timeseries")
    def get_context_timeseries(
        sample_idx: int,
        direction: str = "inplane",
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v1",
    ):
        record = deps.find_record(sample_idx, round_idx=round_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        ctx = deps.context_params(
            direction if direction in ("inplane", "outplane") else "inplane",
            layout_profile=layout_profile,
        )
        try:
            png = deps.figures.get_context_png(record, "timeseries", ctx)
        except FigureNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=png, media_type="image/png")

    @router.get("/api/figures/{sample_idx}/context/spectrogram")
    def get_context_spectrogram(
        sample_idx: int,
        direction: str = "inplane",
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v1",
    ):
        record = deps.find_record(sample_idx, round_idx=round_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        ctx = deps.context_params(
            direction if direction in ("inplane", "outplane") else "inplane",
            layout_profile=layout_profile,
        )
        try:
            png = deps.figures.get_context_png(record, "spectrogram", ctx)
        except FigureNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=png, media_type="image/png")

    @router.post("/api/preload")
    def preload_figures(req: PreloadRequest):
        priority = set(req.sample_indices) if req.priority else None
        lookup = lambda sample_idx: deps.find_record(sample_idx, round_idx=req.round_idx)
        if req.both_directions:
            in_ctx = deps.context_params("inplane", layout_profile=req.layout_profile)
            out_ctx = deps.context_params("outplane", layout_profile=req.layout_profile)
            queued_in = deps.figures.schedule_by_indices(
                lookup,
                req.sample_indices,
                in_ctx,
                priority_samples=priority,
            )
            queued_out = deps.figures.schedule_by_indices(
                lookup,
                req.sample_indices,
                out_ctx,
                priority_samples=priority,
            )
            return {"ok": True, "queued": queued_in + queued_out}

        direction = req.direction if req.direction in ("inplane", "outplane") else "inplane"
        ctx = deps.context_params(direction, layout_profile=req.layout_profile)
        queued = deps.figures.schedule_by_indices(
            lookup,
            req.sample_indices,
            ctx,
            priority_samples=priority,
        )
        return {"ok": True, "queued": queued}

    return router
