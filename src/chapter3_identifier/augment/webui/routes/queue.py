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
        focus_sample_idx: int | None = None,
        jump_refresh: bool = False,
    ):
        selected_classes = _parse_pred_classes(pred_classes)
        max_items = max(1, int(deps.cfg.get("webui_queue_max_items", 500)))
        fetched = deps.abnormal_queue_cache.get_items(
            deps.inference_path(round_idx),
            round_idx,
            predicted_classes=selected_classes,
            max_items=max_items + 1,
        )
        truncated = len(fetched) > max_items
        items = fetched[:max_items] if truncated else fetched
        preload_back = int(deps.cfg.get("webui_preload_back", 5))
        preload_forward = int(deps.cfg.get("webui_preload_forward", 20))
        preload_n = max(1, preload_back + preload_forward + 1)
        background_tasks.add_task(
            deps.schedule_queue_preload,
            round_idx,
            items,
            focus_sample_idx,
            bool(jump_refresh),
        )
        return {
            "round_idx": round_idx,
            "pred_classes": list(selected_classes),
            "total": len(items),
            "returned": len(items),
            "truncated": truncated,
            "max_items": max_items,
            "preload_count": min(preload_n, len(items)),
            "items": items,
        }

    @router.post("/api/queue/preload-focus")
    def preload_focus(
        round_idx: int = 1,
        sample_idx: int = 0,
        pred_classes: str | None = None,
        jump_refresh: bool = False,
        jump_distance: int = 0,
    ):
        selected_classes = _parse_pred_classes(pred_classes)
        max_items = max(1, int(deps.cfg.get("webui_queue_max_items", 500)))
        items = deps.abnormal_queue_cache.get_items(
            deps.inference_path(round_idx),
            round_idx,
            predicted_classes=selected_classes,
            max_items=max_items,
        )
        deps.schedule_queue_preload(
            round_idx,
            items,
            center_sample_idx=sample_idx,
            jump_reset=bool(jump_refresh),
        )
        return {
            "ok": True,
            "total": len(items),
            "focus_sample_idx": sample_idx,
            "jump_refresh": bool(jump_refresh),
            "jump_distance": int(jump_distance),
        }

    @router.get("/api/samples")
    def list_samples(
        page: int = 0,
        sensor_id: str | None = None,
        only_unannotated: bool = False,
        only_abnormal: bool = True,
        round_idx: int = 1,
        pred_classes: str | None = None,
    ):
        page_size = int(deps.cfg["queue_page_size"])
        if only_abnormal:
            selected_classes = _parse_pred_classes(pred_classes)
            records = deps.abnormal_queue_cache.get_items(
                deps.inference_path(round_idx),
                round_idx,
                predicted_classes=selected_classes,
            )
            if sensor_id:
                records = [
                    r
                    for r in records
                    if r.get("inplane_sensor_id") == sensor_id or r.get("outplane_sensor_id") == sensor_id
                ]
            if only_unannotated:
                records = [r for r in records if not r.get("already_annotated")]
            total = len(records)
            start = max(0, int(page)) * page_size
            end = start + page_size
            return {
                "total": total,
                "page": int(page),
                "page_size": page_size,
                "items": records[start:end],
            }

        all_records = deps.inference_cache.get_records(deps.inference_path(round_idx))
        return filter_queue(
            all_records,
            sensor_id=sensor_id,
            only_unannotated=only_unannotated,
            only_abnormal=False,
            page=page,
            page_size=page_size,
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
