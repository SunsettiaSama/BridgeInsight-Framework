from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chapter1_identifier.augment._bootstrap import ensure_paths
from src.chapter1_identifier.augment.annotation.gold_reference import GoldReferenceFinder
from src.chapter1_identifier.augment.annotation.store import AnnotationStore
from src.chapter1_identifier.augment.settings import load_config, resolve_python_executable
from src.chapter1_identifier.augment.webui.figure_service import FigureRenderService, SAMPLE_FIGURE_NAMES
from src.chapter1_identifier.augment.webui.job_manager import JobManager
from src.chapter1_identifier.augment.webui.queue import filter_queue, load_inference_records
from src.chapter1_identifier.augment.webui.training_monitor import (
    build_monitor_payload,
    latest_confusion_path,
    load_metrics_history,
)

ensure_paths()

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AnnotateRequest(BaseModel):
    sample_idx: int
    annotation: int
    round_idx: int = 1


class PreloadRequest(BaseModel):
    sample_indices: list[int]
    direction: str = "inplane"


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    app = FastAPI(title="Augment Annotation WebUI")
    store = AnnotationStore(
        gold_path=cfg["gold_annotation_path"],
        manual_edits_path=cfg["manual_edits_path"],
        merged_output_path=cfg["merged_training_path"],
    )
    jobs = JobManager(cfg["job_state_path"], python_executable=resolve_python_executable(cfg))
    records = load_inference_records(cfg["inference_results_path"])
    figure_service = FigureRenderService(max_workers=2, cache_size=96)
    gold_finder: GoldReferenceFinder | None = None

    def _get_gold_finder() -> GoldReferenceFinder:
        nonlocal gold_finder
        if gold_finder is None:
            gold_finder = GoldReferenceFinder(
                store.load_gold(),
                window_size=int(cfg["window_size"]),
                fs=float(cfg["fs"]),
                nfft=int(cfg["nfft"]),
                freq_max_hz=float(cfg["freq_max_hz"]),
            )
        return gold_finder

    def _find_record(sample_idx: int) -> dict | None:
        nonlocal records
        records = load_inference_records(cfg["inference_results_path"])
        for r in records:
            if int(r["sample_idx"]) == sample_idx:
                return r
        return None

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/samples")
    def list_samples(
        page: int = 0,
        sensor_id: str | None = None,
        only_unannotated: bool = False,
        only_abnormal: bool = True,
    ):
        nonlocal records
        records = load_inference_records(cfg["inference_results_path"])
        return filter_queue(
            records,
            sensor_id=sensor_id,
            only_unannotated=only_unannotated,
            only_abnormal=only_abnormal,
            page=page,
            page_size=int(cfg["queue_page_size"]),
        )

    @app.get("/api/samples/{sample_idx}")
    def get_sample(sample_idx: int):
        nonlocal records
        records = load_inference_records(cfg["inference_results_path"])
        for r in records:
            if int(r["sample_idx"]) == sample_idx:
                r = dict(r)
                in_fp = r.get("inplane_file_path", "")
                wi = int(r.get("window_index", 0))
                r["is_gold"] = store.is_gold_member(in_fp, wi) if in_fp else False
                return r
        raise HTTPException(status_code=404, detail="sample not found")

    @app.get("/api/samples/{sample_idx}/gold_references")
    def gold_references(sample_idx: int):
        record = _find_record(sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        in_fp = record.get("inplane_file_path")
        if not in_fp:
            raise HTTPException(status_code=404, detail="file path missing")
        refs = _get_gold_finder().find_topk(
            in_fp,
            int(record.get("window_index", 0)),
            topk=int(cfg["gold_reference_topk"]),
        )
        return {"sample_idx": sample_idx, "references": refs}

    @app.get("/api/figures/{sample_idx}/{figure_name}")
    def get_figure(sample_idx: int, figure_name: str):
        if figure_name not in SAMPLE_FIGURE_NAMES:
            raise HTTPException(status_code=404, detail=f"figure {figure_name} not found")
        record = _find_record(sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        png = figure_service.get_sample_figure(record, figure_name)
        return Response(content=png, media_type="image/png")

    @app.get("/api/figures/{sample_idx}/context/timeseries")
    def get_context_timeseries(sample_idx: int, direction: str = "inplane"):
        record = _find_record(sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        png = figure_service.get_context_figure(
            record,
            direction,
            "timeseries",
            int(cfg["context_windows_before"]),
            int(cfg["context_windows_after"]),
            float(cfg["context_spectrogram_segment_s"]),
            int(cfg["context_figure_cache_size"]),
        )
        return Response(content=png, media_type="image/png")

    @app.get("/api/figures/{sample_idx}/context/spectrogram")
    def get_context_spectrogram(sample_idx: int, direction: str = "inplane"):
        record = _find_record(sample_idx)
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")
        png = figure_service.get_context_figure(
            record,
            direction,
            "spectrogram",
            int(cfg["context_windows_before"]),
            int(cfg["context_windows_after"]),
            float(cfg["context_spectrogram_segment_s"]),
            int(cfg["context_figure_cache_size"]),
        )
        return Response(content=png, media_type="image/png")

    @app.post("/api/preload")
    def preload_figures(req: PreloadRequest):
        preload_records = []
        for sample_idx in req.sample_indices:
            record = _find_record(int(sample_idx))
            if record is not None:
                preload_records.append(record)
        queued = figure_service.preload_records(
            preload_records,
            req.direction,
            int(cfg["context_windows_before"]),
            int(cfg["context_windows_after"]),
            float(cfg["context_spectrogram_segment_s"]),
            int(cfg["context_figure_cache_size"]),
        )
        return {"ok": True, "queued": queued}

    @app.post("/api/annotate")
    def annotate(req: AnnotateRequest):
        nonlocal records
        records = load_inference_records(cfg["inference_results_path"])
        record = None
        for r in records:
            if int(r["sample_idx"]) == req.sample_idx:
                record = r
                break
        if record is None:
            raise HTTPException(status_code=404, detail="sample not found")

        in_fp = record.get("inplane_file_path")
        wi = int(record.get("window_index", 0))
        is_gold = store.is_gold_member(in_fp, wi) if in_fp else False
        row = store.upsert_manual(
            file_path=in_fp,
            window_index=wi,
            annotation=req.annotation,
            outplane_file_path=record.get("outplane_file_path"),
            is_gold=is_gold,
            round_idx=req.round_idx,
        )
        return {"ok": True, "entry": row}

    @app.post("/api/jobs/train")
    def start_train(round_idx: int = 1):
        try:
            return jobs.start_train(round_idx=round_idx, config_path=config_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/infer")
    def start_infer(round_idx: int = 1):
        try:
            return jobs.start_infer(round_idx=round_idx, config_path=config_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/jobs/status")
    def job_status():
        return jobs.poll()

    @app.get("/api/jobs/monitor")
    def training_monitor(round_idx: int = 1):
        from src.chapter1_identifier.augment._bootstrap import resolve_path

        state = jobs.poll()
        log_tail = jobs.read_log_tail(round_idx, phase="train", max_chars=8000)
        payload = build_monitor_payload(
            cfg["training_output_dir"],
            round_idx,
            int(cfg["epochs"]),
            state,
            log_tail,
        )
        return payload

    @app.get("/api/jobs/log")
    def job_log(round_idx: int = 1, phase: str = "train", tail: int = 8000):
        text = jobs.read_log_tail(round_idx, phase=phase, max_chars=tail)
        return {"round_idx": round_idx, "phase": phase, "text": text}

    @app.get("/api/training/{round_idx}/confusion")
    def training_confusion(round_idx: int, epoch: int | None = None):
        path = latest_confusion_path(cfg["training_output_dir"], round_idx, epoch)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="confusion matrix not found")
        return Response(content=path.read_bytes(), media_type="image/png")

    @app.get("/api/metrics")
    def metrics(round_idx: int = 1):
        history = load_metrics_history(cfg["training_output_dir"], round_idx)
        return {"history": history, "round_idx": round_idx}

    @app.websocket("/ws/ws")
    async def ide_live_stub(websocket: WebSocket):
        """Cursor/IDE 内置浏览器会探测此路径做 live reload，接受连接避免 403 日志。"""
        await websocket.accept()
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    import sys

    parser = argparse.ArgumentParser(description="Augment WebUI")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    port = args.port or int(cfg["webui_port"])
    job_python = resolve_python_executable(cfg)
    app = create_app(args.config)
    print(f"Augment 标注 WebUI: http://localhost:{port}")
    print(f"后台 train/infer 使用 Python: {job_python}")
    if job_python != sys.executable:
        print(f"（WebUI 当前解释器: {sys.executable}）")
    uvicorn.run(app, host="localhost", port=port)


if __name__ == "__main__":
    main()
