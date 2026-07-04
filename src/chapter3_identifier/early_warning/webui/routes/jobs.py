from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.chapter3_identifier.early_warning.webui.deps import AppDeps
from src.chapter3_identifier.regression_forecast.webui.monitors.inference import build_infer_monitor_payload
from src.chapter3_identifier.regression_forecast.webui.monitors.training import build_training_monitor_payload


def _ensure_idle(deps: AppDeps) -> None:
    state = deps.jobs.poll()
    if state.get("status") == "running":
        raise HTTPException(status_code=409, detail="已有任务在运行")


def _config_path(deps: AppDeps) -> str | None:
    return deps.config_path or deps.cfg.get("_config_path")


def build_jobs_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.post("/api/jobs/build-cache")
    def start_build_cache(round_idx: int = 1):
        _ensure_idle(deps)
        return deps.jobs.start_build_cache(round_idx=round_idx, config_path=_config_path(deps))

    @router.post("/api/jobs/train")
    def start_train(round_idx: int = 1):
        _ensure_idle(deps)
        return deps.jobs.start_train(round_idx=round_idx, config_path=_config_path(deps))

    @router.post("/api/jobs/infer")
    def start_infer(round_idx: int = 1):
        _ensure_idle(deps)
        deps.forecasts.invalidate()
        return deps.jobs.start_infer(round_idx=round_idx, config_path=_config_path(deps))

    @router.post("/api/jobs/reset")
    def reset_job():
        return deps.jobs.reset_job()

    @router.get("/api/jobs/status")
    def job_status():
        state = deps.jobs.poll()
        if state.get("status") == "done" and state.get("phase") == "infer":
            deps.forecasts.invalidate()
        return state

    @router.get("/api/jobs/log")
    def job_log(round_idx: int = 1, phase: str = "train", tail: int = 8000):
        return {
            "round_idx": round_idx,
            "phase": phase,
            "text": deps.jobs.read_log_tail(round_idx, phase=phase, max_chars=tail),
        }

    @router.get("/api/jobs/monitor")
    def training_monitor(round_idx: int = 1):
        return build_training_monitor_payload(
            deps.cfg,
            round_idx,
            deps.jobs.poll(),
            deps.jobs.read_log_tail(round_idx, "train"),
        )

    @router.get("/api/jobs/infer/monitor")
    def infer_monitor(round_idx: int = 1):
        payload = build_infer_monitor_payload(
            deps.cfg,
            round_idx,
            deps.jobs.poll(),
            deps.jobs.read_log_tail(round_idx, "infer"),
        )
        if payload.get("forecast_ready"):
            deps.forecasts.invalidate()
        return payload

    return router
