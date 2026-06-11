from __future__ import annotations

from typing import Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.chapter3_identifier.augment.queue.loader import filter_queue
from src.chapter3_identifier.augment.webui.deps import AppDeps


def build_queue_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    def _parse_pred_classes(pred_classes: str | None) -> Tuple[int, ...]:
        if pred_classes is None or pred_classes.strip() == "":
            return (1, 2, 3)
        out = []
        for token in pred_classes.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.lstrip("-").isdigit():
                continue
            value = int(token)
            if 0 <= value <= 3:
                out.append(value)
        uniq = sorted(set(out))
        return tuple(uniq) if uniq else (1, 2, 3)

    @router.get("/api/queue/abnormal")
    def abnormal_queue(
        background_tasks: BackgroundTasks,
        round_idx: int = 1,
        pred_classes: str | None = None,
    ):
        selected_classes = _parse_pred_classes(pred_classes)
        items = deps.abnormal_queue_cache.get_items(
            deps.inference_path(round_idx),
            round_idx,
            predicted_classes=selected_classes,
        )
        preload_n = int(deps.cfg.get("webui_preload_count", 20))
        background_tasks.add_task(deps.schedule_queue_preload, round_idx, items)
        return {
            "round_idx": round_idx,
            "pred_classes": list(selected_classes),
            "total": len(items),
            "preload_count": min(preload_n, len(items)),
            "items": items,
        }

    @router.get("/api/samples")
    def list_samples(
        page: int = 0,
        sensor_id: str | None = None,
        only_unannotated: bool = False,
        only_abnormal: bool = True,
        round_idx: int = 1,
    ):
        records = deps.inference_cache.get_records(deps.inference_path(round_idx))
        return filter_queue(
            records,
            sensor_id=sensor_id,
            only_unannotated=only_unannotated,
            only_abnormal=only_abnormal,
            page=page,
            page_size=int(deps.cfg["queue_page_size"]),
        )

    @router.get("/api/samples/{sample_idx}")
    def get_sample(sample_idx: int, round_idx: int = 1):
        record = deps.find_record(sample_idx, round_idx=round_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        row = dict(record)
        row.update(deps.annotation_context(row, round_idx))
        return row

    @router.get("/api/samples/{sample_idx}/gold_references")
    def gold_references_stub(sample_idx: int, round_idx: int = 1):
        if deps.find_record(sample_idx, round_idx=round_idx) is None:
            raise HTTPException(status_code=404, detail="sample not found")
        return {"sample_idx": sample_idx, "references": []}

    return router
