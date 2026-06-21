from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from src.chapter3_identifier.augment.figures import FigureNotReadyError, SAMPLE_FIGURE_NAMES
from src.chapter3_identifier.augment.webui.deps import AppDeps


class PreloadRequest(BaseModel):
    sample_indices: list[int]
    direction: str = "inplane"
    both_directions: bool = True
    layout_profile: str = "wide_fill_v3"
    round_idx: int = 1
    priority: bool = False


def build_figures_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    def _get_record(sample_idx: int, round_idx: int) -> dict:
        record = deps.find_record(sample_idx, round_idx=round_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        return record

    @router.get("/api/figures/{sample_idx}/_bundle_status")
    def get_bundle_status(
        sample_idx: int,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
        wait_ms: int = 0,
    ):
        record = _get_record(sample_idx, round_idx)
        if wait_ms > 0:
            deps.figures.wait_bundle_ready(
                sample_idx,
                "inplane",
                record=record,
                round_idx=round_idx,
                layout_profile=layout_profile,
                wait_ms=wait_ms,
            )
        inplane_ready = deps.figures.bundle_ready(
            sample_idx,
            "inplane",
            record=record,
            round_idx=round_idx,
            layout_profile=layout_profile,
        )
        outplane_ready = deps.figures.bundle_ready(
            sample_idx,
            "outplane",
            record=record,
            round_idx=round_idx,
            layout_profile=layout_profile,
        )
        return {
            "sample_idx": sample_idx,
            "round_idx": round_idx,
            "layout_profile": layout_profile,
            "wind_stats": deps.figures.get_wind_stats(record, round_idx=round_idx),
            "sample_ready": deps.figures.sample_ready(
                sample_idx,
                round_idx=round_idx,
                layout_profile=layout_profile,
            ),
            "inplane_context_ready": deps.figures.context_ready(
                sample_idx,
                "inplane",
                round_idx=round_idx,
                layout_profile=layout_profile,
            ),
            "outplane_context_ready": deps.figures.context_ready(
                sample_idx,
                "outplane",
                round_idx=round_idx,
                layout_profile=layout_profile,
            ),
            "inplane_bundle_ready": inplane_ready,
            "outplane_bundle_ready": outplane_ready,
        }

    @router.get("/api/figures/{sample_idx}/context/timeseries")
    def get_context_timeseries(
        sample_idx: int,
        direction: str = "inplane",
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
        wait_ms: int = 0,
    ):
        record = _get_record(sample_idx, round_idx)
        ctx = deps.context_params(
            direction if direction in ("inplane", "outplane") else "inplane",
            round_idx=round_idx,
            layout_profile=layout_profile,
        )
        try:
            png = deps.figures.wait_context_png(record, "timeseries", ctx, wait_ms=wait_ms)
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
        layout_profile: str = "wide_fill_v3",
        wait_ms: int = 0,
    ):
        record = _get_record(sample_idx, round_idx)
        ctx = deps.context_params(
            direction if direction in ("inplane", "outplane") else "inplane",
            round_idx=round_idx,
            layout_profile=layout_profile,
        )
        try:
            png = deps.figures.wait_context_png(record, "spectrogram", ctx, wait_ms=wait_ms)
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
            in_ctx = deps.context_params(
                "inplane",
                round_idx=req.round_idx,
                layout_profile=req.layout_profile,
            )
            out_ctx = deps.context_params(
                "outplane",
                round_idx=req.round_idx,
                layout_profile=req.layout_profile,
            )
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
        ctx = deps.context_params(
            direction,
            round_idx=req.round_idx,
            layout_profile=req.layout_profile,
        )
        queued = deps.figures.schedule_by_indices(
            lookup,
            req.sample_indices,
            ctx,
            priority_samples=priority,
        )
        return {"ok": True, "queued": queued}

    @router.get("/api/figures/{sample_idx}/wind_stats")
    def get_wind_stats(sample_idx: int, round_idx: int = 1):
        record = _get_record(sample_idx, round_idx)
        return deps.figures.get_wind_stats(record, round_idx=round_idx)

    @router.get("/api/figures/{sample_idx}/{figure_name}")
    def get_figure(
        sample_idx: int,
        figure_name: str,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
        direction: str = "inplane",
        wait_ms: int = 0,
    ):
        if figure_name not in SAMPLE_FIGURE_NAMES:
            raise HTTPException(status_code=404, detail=f"figure {figure_name} not found")
        record = _get_record(sample_idx, round_idx)
        try:
            png = deps.figures.wait_sample_png(
                record,
                figure_name,
                layout_profile=layout_profile,
                prediction_direction=direction if direction in ("inplane", "outplane") else "inplane",
                wait_ms=wait_ms,
                round_idx=round_idx,
            )
        except FigureNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=png, media_type="image/png")

    return router
