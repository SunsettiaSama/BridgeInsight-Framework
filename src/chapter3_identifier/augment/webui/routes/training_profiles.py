from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from src.chapter3_identifier.augment.train.profile import (
    load_training_profile,
    reset_training_profile,
    save_training_profile,
    training_profile_options,
)
from src.chapter3_identifier.augment.webui.deps import AppDeps


def build_training_profiles_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/training/profile")
    def get_profile(round_idx: int = 1):
        payload = load_training_profile(deps.cfg, round_idx)
        payload["options"] = training_profile_options()
        return payload

    @router.put("/api/training/profile")
    def put_profile(round_idx: int = 1, payload: dict = Body(...)):
        state = deps.jobs.poll()
        if state.get("status") == "running" and state.get("phase") == "train" and int(state.get("round", 0)) == int(round_idx):
            raise HTTPException(status_code=409, detail="当前轮次训练正在运行，不能修改训练配置")
        saved = save_training_profile(deps.cfg, round_idx, payload)
        saved["options"] = training_profile_options()
        return saved

    @router.post("/api/training/profile/reset")
    def reset_profile(round_idx: int = 1):
        state = deps.jobs.poll()
        if state.get("status") == "running" and state.get("phase") == "train" and int(state.get("round", 0)) == int(round_idx):
            raise HTTPException(status_code=409, detail="当前轮次训练正在运行，不能重置训练配置")
        payload = reset_training_profile(deps.cfg, round_idx)
        payload["options"] = training_profile_options()
        return payload

    @router.get("/api/training/profile/options")
    def profile_options():
        return training_profile_options()

    return router
