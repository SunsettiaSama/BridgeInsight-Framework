from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from src.chapter3_identifier.augment.workflow_config import (
    bootstrap_workflow_config,
    diff_round_snapshots,
    ensure_workflow_config,
    load_workflow_config,
    resolve_round_workflow,
    save_round_workflow_snapshot,
    save_workflow_config,
    update_workflow_defaults,
    validate_round_reproducibility,
    workflow_schema,
)
from src.chapter3_identifier.augment.webui.deps import AppDeps


def build_workflow_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/workflow/schema")
    def get_schema():
        return workflow_schema()

    @router.get("/api/workflow/config")
    def get_workflow_config():
        workflow = ensure_workflow_config(deps.cfg)
        resolved_example = resolve_round_workflow(deps.cfg, int(deps.cfg.get("webui_init_round", 1)))
        return {
            "workflow_config_version": int(workflow.get("workflow_config_version", 1)),
            "workflow_defaults": workflow.get("workflow_defaults", {}),
            "round_overrides": workflow.get("round_overrides", {}),
            "metadata": {
                "source": workflow.get("_source", "saved"),
                "path": workflow.get("_path"),
            },
            "resolved_example_round": resolved_example["round_idx"],
            "resolved_example_override_sources": resolved_example["metadata"].get("override_sources", []),
        }

    @router.put("/api/workflow/config")
    def put_workflow_config(payload: dict = Body(...)):
        state = deps.jobs.poll()
        if state.get("status") == "running":
            raise HTTPException(status_code=409, detail="后台任务运行中，不能修改全局 workflow 配置")
        current = load_workflow_config(deps.cfg)
        merged = {
            "workflow_config_version": int(payload.get("workflow_config_version", current.get("workflow_config_version", 1))),
            "workflow_defaults": payload.get("workflow_defaults", current.get("workflow_defaults", {})),
            "round_overrides": payload.get("round_overrides", current.get("round_overrides", {})),
        }
        saved = save_workflow_config(deps.cfg, merged)
        return {
            "workflow_config_version": saved["workflow_config_version"],
            "workflow_defaults": saved["workflow_defaults"],
            "round_overrides": saved.get("round_overrides", {}),
            "metadata": {
                "source": saved.get("_source", "saved"),
                "path": saved.get("_path"),
            },
        }

    @router.put("/api/workflow/config/section/{section}")
    def put_workflow_section(section: str, payload: dict = Body(...)):
        state = deps.jobs.poll()
        if state.get("status") == "running":
            raise HTTPException(status_code=409, detail="后台任务运行中，不能修改全局 workflow 配置")
        saved = update_workflow_defaults(deps.cfg, section, payload)
        return {
            "section": section,
            "workflow_defaults": saved["workflow_defaults"],
            "metadata": {
                "source": saved.get("_source", "saved"),
                "path": saved.get("_path"),
            },
        }

    @router.post("/api/workflow/migrate")
    def migrate_workflow(baseline_round: int = 8):
        state = deps.jobs.poll()
        if state.get("status") == "running":
            raise HTTPException(status_code=409, detail="后台任务运行中，不能迁移 workflow 配置")
        saved = bootstrap_workflow_config(deps.cfg, baseline_round=baseline_round, write=True)
        return {
            "workflow_config_version": saved["workflow_config_version"],
            "workflow_defaults": saved["workflow_defaults"],
            "round_overrides": saved.get("round_overrides", {}),
            "metadata": {
                "source": saved.get("_source", "bootstrap"),
                "path": saved.get("_path"),
                "baseline_round": int(baseline_round),
            },
        }

    @router.get("/api/workflow/resolved")
    def get_resolved(round_idx: int = 1, save_snapshot: bool = False):
        resolved = resolve_round_workflow(deps.cfg, round_idx)
        if save_snapshot:
            save_round_workflow_snapshot(deps.cfg, round_idx, resolved)
        return resolved

    @router.get("/api/workflow/diff")
    def get_diff(round_a: int, round_b: int):
        return diff_round_snapshots(deps.cfg, round_a, round_b)

    @router.get("/api/workflow/validate")
    def validate_round(round_idx: int):
        return validate_round_reproducibility(deps.cfg, round_idx)

    return router
