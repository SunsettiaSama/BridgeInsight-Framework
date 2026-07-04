from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Response

from src.chapter3_identifier.augment.finalize.run import build_finalize_summary
from src.chapter3_identifier.augment.train.profile import load_training_profile
from src.chapter3_identifier.augment.workflow_config import (
    ensure_workflow_config,
    resolve_round_workflow,
    save_round_workflow_snapshot,
)
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
            ensure_workflow_config(deps.cfg)
            resolved = resolve_round_workflow(deps.cfg, round_idx)
            snapshot_path = save_round_workflow_snapshot(deps.cfg, round_idx, resolved)
            profile_payload = load_training_profile(deps.cfg, round_idx)
            meta = profile_payload["metadata"]
            profile_path = meta["path"] if meta["source"] == "explicit_profile_path" else None
            result = deps.jobs.start_train(
                round_idx=round_idx,
                config_path=deps.config_path,
                profile_path=profile_path,
                profile_summary=meta["summary"],
                workflow_snapshot_path=str(snapshot_path),
            )
            result["workflow_resolved_path"] = str(snapshot_path)
            result["workflow_config_path"] = resolved["metadata"]["workflow_config_path"]
            return result
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/api/jobs/infer")
    def start_infer(round_idx: int = 1):
        try:
            ensure_workflow_config(deps.cfg)
            resolved = resolve_round_workflow(deps.cfg, round_idx)
            snapshot_path = save_round_workflow_snapshot(deps.cfg, round_idx, resolved)
            result = deps.jobs.start_infer(round_idx=round_idx, config_path=deps.config_path)
            result["workflow_resolved_path"] = str(snapshot_path)
            result["workflow_config_path"] = resolved["metadata"]["workflow_config_path"]
            return result
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/jobs/finalize/summary")
    def finalize_summary(
        from_round: int = 1,
        to_round: int | None = None,
        canonical_round: int | None = None,
        overwrite_final: bool = False,
    ):
        try:
            return build_finalize_summary(
                config_path=deps.config_path,
                from_round=from_round,
                to_round=to_round,
                canonical_round=canonical_round,
                overwrite_final=overwrite_final,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/api/jobs/finalize")
    def start_finalize(payload: dict = Body(default={})):
        try:
            return deps.jobs.start_finalize(
                config_path=deps.config_path,
                from_round=int(payload.get("from_round", 1)),
                to_round=int(payload["to_round"]) if payload.get("to_round") is not None else None,
                canonical_round=int(payload["canonical_round"])
                if payload.get("canonical_round") is not None
                else None,
                overwrite_final=bool(payload.get("overwrite_final", False)),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/api/jobs/finalize/monitor")
    def finalize_monitor():
        state = deps.jobs.poll()
        log_tail = deps.jobs.read_log_tail(0, phase="finalize", max_chars=12000)
        summary = None
        if state.get("status") != "running":
            try:
                summary = build_finalize_summary(
                    config_path=deps.config_path,
                    from_round=int(state.get("finalize_from_round", 1)),
                    to_round=state.get("finalize_to_round"),
                    canonical_round=state.get("finalize_canonical_round"),
                    overwrite_final=bool(state.get("finalize_overwrite_final", False)),
                )
            except ValueError:
                summary = None
        return {
            "job": state,
            "log_tail": log_tail,
            "summary": summary,
        }

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
