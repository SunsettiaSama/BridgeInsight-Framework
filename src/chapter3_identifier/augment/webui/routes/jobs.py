from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from src.chapter3_identifier.augment.train.profile import load_training_profile
from src.chapter3_identifier.augment.webui.deps import AppDeps
from src.chapter3_identifier.augment.webui.monitors.inference import build_infer_monitor_payload
from src.chapter3_identifier.augment.webui.monitors.training import (
    build_monitor_payload,
    latest_confusion_path,
    load_metrics_history,
)


def build_jobs_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.post("/api/jobs/train")
    def start_train(round_idx: int = 1):
        try:
            profile_payload = load_training_profile(deps.cfg, round_idx)
            meta = profile_payload["metadata"]
            profile_path = meta["path"] if meta["source"] == "saved" else None
            return deps.jobs.start_train(
                round_idx=round_idx,
                config_path=deps.config_path,
                profile_path=profile_path,
                profile_summary=meta["summary"],
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/jobs/infer")
    def start_infer(round_idx: int = 1):
        try:
            return deps.jobs.start_infer(round_idx=round_idx, config_path=deps.config_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/jobs/reset")
    def reset_job():
        return deps.jobs.reset_job()

    @router.get("/api/jobs/status")
    def job_status():
        return deps.jobs.poll()

    @router.get("/api/jobs/monitor")
    def training_monitor(round_idx: int = 1):
        state = deps.jobs.poll()
        log_tail = deps.jobs.read_log_tail(round_idx, phase="train", max_chars=8000)
        profile_payload = load_training_profile(deps.cfg, round_idx)
        epochs_total = int(profile_payload["profile"]["epochs"])
        return build_monitor_payload(
            deps.cfg,
            round_idx,
            epochs_total,
            state,
            log_tail,
        )

    @router.get("/api/jobs/infer/monitor")
    def infer_monitor(round_idx: int = 1):
        state = deps.jobs.poll()
        log_tail = deps.jobs.read_log_tail(round_idx, phase="infer", max_chars=8000)
        typical_topk = int(deps.cfg.get("infer_monitor_typical_topk", 2))
        payload = build_infer_monitor_payload(
            deps.cfg,
            round_idx,
            state,
            log_tail,
            deps.inference_path(round_idx),
            typical_topk=typical_topk,
            inference_cache=deps.inference_cache,
        )
        if payload.get("inference_ready"):
            typical_records = []
            for class_samples in payload.get("typical_samples", {}).values():
                for brief in class_samples:
                    record = deps.find_record(int(brief["sample_idx"]), round_idx=round_idx)
                    if record is not None:
                        typical_records.append(record)
            if typical_records:
                ctx = deps.context_params(round_idx=round_idx)
                deps.figures.schedule_preload(typical_records, ctx, replace=False)
        return payload

    @router.get("/api/jobs/log")
    def job_log(round_idx: int = 1, phase: str = "train", tail: int = 8000):
        text = deps.jobs.read_log_tail(round_idx, phase=phase, max_chars=tail)
        return {"round_idx": round_idx, "phase": phase, "text": text}

    @router.get("/api/training/{round_idx}/confusion")
    def training_confusion(round_idx: int, epoch: int | None = None, kind: str = "merged"):
        path = latest_confusion_path(deps.cfg, round_idx, epoch, kind=kind)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="confusion matrix not found")
        return Response(content=path.read_bytes(), media_type="image/png")

    @router.get("/api/metrics")
    def metrics(round_idx: int = 1):
        history = load_metrics_history(deps.cfg, round_idx)
        return {"history": history, "round_idx": round_idx}

    return router
