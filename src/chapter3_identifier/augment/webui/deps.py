from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
from src.chapter3_identifier.augment.annotation.store import (
    AnnotationStore,
    annotation_store_for_round,
    load_cumulative_manual_edits,
)
from src.chapter3_identifier.augment.features.wind_index import should_degrade_to_dict_mode
from src.chapter3_identifier.augment.figures import ContextParams, FigureService
from src.chapter3_identifier.augment.labels import label_name
from src.chapter3_identifier.augment.queue.abnormal_cache import AbnormalQueueCache
from src.chapter3_identifier.augment.queue.enrich import AnnotationLookupCache, enrich_record
from src.chapter3_identifier.augment.queue.inference_cache import InferenceSnapshotCache
from src.chapter3_identifier.augment.settings import (
    get_round_inference_snapshot_path,
    resolve_python_executable,
)
from src.chapter3_identifier.augment.webui.jobs.manager import JobManager


@dataclass
class AppDeps:
    cfg: dict
    config_path: Optional[str]
    figures: FigureService
    inference_cache: InferenceSnapshotCache
    annotation_lookup: AnnotationLookupCache
    gold_store: AnnotationStore
    abnormal_queue_cache: AbnormalQueueCache
    jobs: JobManager
    merge_lock: threading.Lock = field(default_factory=threading.Lock)
    merge_running: bool = False
    merge_last_started_at: float = 0.0
    _gold_index_cache: Optional[dict] = field(default=None, init=False, repr=False)

    def inference_path(self, round_idx: int) -> str:
        return str(get_round_inference_snapshot_path(self.cfg, round_idx))

    def find_record(self, sample_idx: int, round_idx: int = 1) -> dict | None:
        return self.inference_cache.get_record(self.inference_path(round_idx), sample_idx)

    def gold_index(self) -> dict:
        if self._gold_index_cache is None:
            self._gold_index_cache = self.annotation_lookup.gold_index(self.gold_store.load_gold())
        return self._gold_index_cache

    def context_params(
        self,
        direction: str = "inplane",
        *,
        round_idx: int = 1,
        layout_profile: str = "wide_fill_v3",
    ) -> ContextParams:
        window_size = int(self.cfg.get("window_size", 3000))
        fs = float(self.cfg.get("fs", 50.0))
        window_seconds = window_size / fs if fs > 0 else 60.0
        target_seconds = float(self.cfg.get("context_total_seconds", 1500.0))
        total_windows = max(1, math.ceil(target_seconds / window_seconds))
        if total_windows % 2 == 0:
            total_windows += 1
        context_side_windows = (total_windows - 1) // 2
        return ContextParams(
            direction=direction,
            round_idx=int(round_idx),
            layout_profile=layout_profile,
            windows_before=context_side_windows,
            windows_after=context_side_windows,
            spectrogram_segment_s=float(self.cfg["context_spectrogram_segment_s"]),
            figure_cache_size=int(self.cfg["context_figure_cache_size"]),
        )

    def annotation_context(self, record: dict, round_idx: int) -> dict:
        gold_index = self.gold_index()
        manual_index = self.annotation_lookup.manual_index(round_idx)
        changed_keys = self.annotation_lookup.changed_manual_keys(round_idx)
        blind_gold_keys = self.annotation_lookup.blind_validation_keys() & set(gold_index.keys())
        return enrich_record(
            record,
            gold_index,
            manual_index,
            self.cfg,
            changed_keys=changed_keys,
            blind_gold_keys=blind_gold_keys,
        )

    def schedule_queue_preload(
        self,
        round_idx: int,
        items: list,
        center_sample_idx: int | None = None,
        jump_reset: bool = False,
    ) -> None:
        if jump_reset:
            self.figures.on_user_jump_reset()
        preload_back = int(self.cfg.get("webui_preload_back", 5))
        preload_forward = int(self.cfg.get("webui_preload_forward", 20))
        preload_total = max(1, preload_back + preload_forward + 1)
        center_pos = 0
        if center_sample_idx is not None:
            for i, row in enumerate(items):
                if int(row.get("sample_idx", -1)) == int(center_sample_idx):
                    center_pos = i
                    break
        start = max(0, center_pos - preload_back)
        end = min(len(items), center_pos + preload_forward + 1)
        window_rows = items[start:end]
        if not window_rows:
            window_rows = items[:preload_total]

        preload_records = []
        for row in window_rows[:preload_total]:
            record = self.find_record(int(row["sample_idx"]), round_idx=round_idx)
            if record is not None:
                preload_records.append(record)
        if not preload_records:
            return
        local_center = 0
        if center_sample_idx is not None:
            local_center = max(0, min(len(window_rows) - 1, center_pos - start))
        pri_back = max(1, preload_back)
        pri_start = max(0, local_center - pri_back)
        pri_end = min(len(window_rows), local_center + pri_back + 1)
        priority_rows = window_rows[pri_start:pri_end]
        if center_sample_idx is not None:
            center_row = next(
                (row for row in window_rows if int(row.get("sample_idx", -1)) == int(center_sample_idx)),
                None,
            )
            if center_row is not None and center_row not in priority_rows:
                priority_rows = [center_row] + priority_rows
        if not priority_rows:
            priority_rows = window_rows[:1]
        priority_samples = {int(r["sample_idx"]) for r in priority_rows if "sample_idx" in r}
        layout_profile = str(self.cfg.get("webui_layout_profile", "wide_fill_v3"))
        inplane_ctx = self.context_params("inplane", round_idx=round_idx, layout_profile=layout_profile)
        outplane_ctx = self.context_params("outplane", round_idx=round_idx, layout_profile=layout_profile)
        self.figures.schedule_preload(
            preload_records,
            inplane_ctx,
            replace=False,
            priority_samples=priority_samples,
        )
        self.figures.schedule_preload(
            preload_records,
            outplane_ctx,
            replace=False,
            priority_samples=priority_samples,
        )

    def warmup_figure_buffer(self) -> None:
        round_idx = int(self.cfg.get("webui_init_round", 1))
        inference_path = self.inference_path(round_idx)
        # Prime non-zero class queues to make class-switching instant in annotation UI.
        self.abnormal_queue_cache.get_items(inference_path, round_idx, predicted_classes=(1,))
        self.abnormal_queue_cache.get_items(inference_path, round_idx, predicted_classes=(2,))
        self.abnormal_queue_cache.get_items(inference_path, round_idx, predicted_classes=(3,))
        items = self.abnormal_queue_cache.get_items(inference_path, round_idx, predicted_classes=(1, 2, 3))
        self.schedule_queue_preload(round_idx, items)

    def schedule_merge_training(self, round_idx: int) -> bool:
        min_interval_s = float(self.cfg.get("webui_merge_min_interval_s", 15.0))
        now = time.time()
        if now - float(self.merge_last_started_at) < min_interval_s:
            return False
        if self.merge_running:
            with self.merge_lock:
                if self.merge_running:
                    return False

        with self.merge_lock:
            if self.merge_running:
                return False
            self.merge_running = True
            self.merge_last_started_at = now

        def _run() -> None:
            try:
                round_store = annotation_store_for_round(self.cfg, round_idx)
                prior_manual = load_cumulative_manual_edits(self.cfg, round_idx - 1)
                round_store.merge(
                    gold_only=False,
                    prior_manual=prior_manual,
                    shuffle_seed=int(self.cfg.get("merge_shuffle_seed", 42)) + round_idx,
                )
                self.annotation_lookup.invalidate_manual()
                self.abnormal_queue_cache.invalidate()
            finally:
                with self.merge_lock:
                    self.merge_running = False

        threading.Thread(target=_run, name="aug-merge", daemon=True).start()
        return True

    def annotation_state_payload(self, round_idx: int) -> dict:
        round_store = annotation_store_for_round(self.cfg, round_idx)
        edits = round_store.load_manual_edits()
        cumulative = load_cumulative_manual_edits(self.cfg, round_idx)
        by_key = self.inference_cache.get_key_index(self.inference_path(round_idx))
        entries = []
        for entry in edits:
            fp = entry.get("file_path")
            wi = int(entry.get("window_index", 0))
            inplane_ann = int(entry.get("inplane_annotation", entry.get("annotation", 0)))
            outplane_ann = int(entry.get("outplane_annotation", entry.get("annotation", 0)))
            key = annotation_key(fp, wi) if fp else None
            entries.append(
                {
                    "sample_idx": by_key.get(key) if key else None,
                    "file_path": fp,
                    "window_index": wi,
                    "inplane_annotation": inplane_ann,
                    "inplane_annotation_name": label_name(inplane_ann, self.cfg),
                    "outplane_annotation": outplane_ann,
                    "outplane_annotation_name": label_name(outplane_ann, self.cfg),
                    "updated_at": entry.get("updated_at"),
                    "is_gold": bool(entry.get("is_gold")),
                }
            )
        entries.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
        return {
            "round_idx": round_idx,
            "manual_count": len(edits),
            "cumulative_count": len(cumulative),
            "manual_edits_path": str(round_store.manual_edits_path),
            "merged_training_path": str(round_store.merged_output_path),
            "entries": entries,
        }


def build_deps(cfg: dict, config_path: Optional[str] = None) -> AppDeps:
    should_degrade_to_dict_mode(cfg)
    inference_cache = InferenceSnapshotCache()
    figures = FigureService(
        max_workers=int(cfg.get("figure_bundle_workers", 2)),
        image_workers=int(cfg.get("figure_image_workers", 6)),
        cache_size=int(cfg.get("figure_cache_size", 256)),
    )
    gold_store = AnnotationStore(
        gold_path=cfg["gold_annotation_path"],
        manual_edits_path=cfg.get("manual_edits_path", cfg["gold_annotation_path"]),
        merged_output_path=cfg.get("merged_training_path", cfg["gold_annotation_path"]),
    )
    annotation_lookup = AnnotationLookupCache(cfg)
    abnormal_queue_cache = AbnormalQueueCache(inference_cache, annotation_lookup, gold_store, cfg)
    jobs = JobManager(cfg["job_state_path"], python_executable=resolve_python_executable(cfg))
    return AppDeps(
        cfg=cfg,
        config_path=config_path,
        figures=figures,
        inference_cache=inference_cache,
        annotation_lookup=annotation_lookup,
        gold_store=gold_store,
        abnormal_queue_cache=abnormal_queue_cache,
        jobs=jobs,
    )
