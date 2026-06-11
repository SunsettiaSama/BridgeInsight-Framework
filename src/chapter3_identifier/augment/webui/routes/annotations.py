from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.annotation.store import annotation_store_for_round
from src.chapter3_identifier.augment.labels import get_label_names, label_name
from src.chapter3_identifier.augment.webui.deps import AppDeps


class AnnotateRequest(BaseModel):
    sample_idx: int
    inplane_annotation: int
    outplane_annotation: int
    round_idx: int = 1


def build_annotations_router(deps: AppDeps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/annotations/state")
    def annotation_state(round_idx: int = 1):
        return deps.annotation_state_payload(round_idx)

    @router.post("/api/annotate")
    def annotate(req: AnnotateRequest):
        record = deps.find_record(req.sample_idx, round_idx=req.round_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")

        label_names = get_label_names(deps.cfg)
        num_classes = int(deps.cfg.get("num_classes", len(label_names)))
        if req.inplane_annotation < 0 or req.inplane_annotation >= num_classes:
            raise HTTPException(
                status_code=400,
                detail=f"inplane_annotation={req.inplane_annotation} 超出类别范围 0..{num_classes - 1}",
            )
        if req.outplane_annotation < 0 or req.outplane_annotation >= num_classes:
            raise HTTPException(
                status_code=400,
                detail=f"outplane_annotation={req.outplane_annotation} 超出类别范围 0..{num_classes - 1}",
            )

        round_store = annotation_store_for_round(deps.cfg, req.round_idx)
        in_fp = record.get("inplane_file_path")
        wi = int(record.get("window_index", 0))
        gold_index = deps.annotation_lookup.gold_index(deps.gold_store.load_gold())
        key = annotation_key(in_fp, wi) if in_fp else None
        gold_entry = gold_index.get(key) if key else None
        is_gold = gold_entry is not None
        row = round_store.upsert_manual(
            file_path=in_fp,
            window_index=wi,
            inplane_annotation=req.inplane_annotation,
            outplane_annotation=req.outplane_annotation,
            outplane_file_path=record.get("outplane_file_path"),
            is_gold=is_gold,
            sample_id=(gold_entry or {}).get("sample_id"),
            round_idx=req.round_idx,
        )
        manual_count = len(round_store.load_manual_edits())
        merge_scheduled = deps.schedule_merge_training(req.round_idx)
        deps.annotation_lookup.invalidate_manual()
        deps.abnormal_queue_cache.invalidate()
        return {
            "ok": True,
            "entry": row,
            "round_idx": req.round_idx,
            "inplane_annotation": int(req.inplane_annotation),
            "inplane_annotation_name": label_name(int(req.inplane_annotation), deps.cfg),
            "outplane_annotation": int(req.outplane_annotation),
            "outplane_annotation_name": label_name(int(req.outplane_annotation), deps.cfg),
            "manual_edits_path": str(round_store.manual_edits_path),
            "manual_count": manual_count,
            "merge_scheduled": merge_scheduled,
            "saved_at": row.get("updated_at"),
        }

    return router
