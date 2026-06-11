from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.chapter4_characteristics._bootstrap import ensure_paths
from src.chapter4_characteristics.analysis.copula_service import run_copula_analysis
from src.chapter4_characteristics.analysis.data_loader import get_data_status, load_class_samples
from src.chapter4_characteristics.analysis.others_inspector import (
    FIGURE_NAMES,
    build_sample_detail,
    build_timeline,
    export_report,
    find_neighbors,
    list_others_samples,
    render_figure,
)
from src.chapter4_characteristics.analysis.plot_registry import registry_for_class
from src.chapter4_characteristics.analysis.plots import copula_viz, common as plot_common
from src.chapter4_characteristics.analysis.plots.copula_viz import load_copula_result
from src.chapter4_characteristics.infer.preflight import run_preflight
from src.chapter4_characteristics.settings import (
    effective_round,
    get_exports_dir,
    write_active_round,
    load_config,
    resolve_python_executable,
)
from src.chapter4_characteristics.webui.infer_monitor import build_infer_monitor_payload
from src.chapter4_characteristics.webui.job_manager import JobManager

ensure_paths()

STATIC_DIR = Path(__file__).resolve().parent / "static"


class SavePlotRequest(BaseModel):
    class_id: int
    plot_id: str
    round_idx: int = 1


class ActiveRoundRequest(BaseModel):
    round_idx: int


class PreloadRequest(BaseModel):
    indices: list[int]
    round_idx: int = 1


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    app = FastAPI(title="Chapter4 振动特性分析 WebUI")
    jobs = JobManager(cfg["job_state_path"], python_executable=resolve_python_executable(cfg))

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/config")
    def app_config():
        return {
            "num_classes": int(cfg.get("num_classes", 4)),
            "label_names": cfg.get("label_names", []),
            "active_round": effective_round(cfg),
        }

    @app.get("/api/status")
    def status(round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        return get_data_status(cfg, rid)

    @app.get("/api/preflight")
    def preflight(round_idx: int = 1):
        return run_preflight(round_idx=round_idx, config_path=config_path)

    @app.post("/api/active-round")
    def set_active_round(body: ActiveRoundRequest):
        st = get_data_status(cfg, body.round_idx)
        if not st.get("enrich_ready"):
            raise HTTPException(404, detail=f"round {body.round_idx} 尚未完成 enrich")
        write_active_round(cfg, body.round_idx)
        return {"active_round": body.round_idx}

    @app.post("/api/jobs/infer")
    def start_infer(round_idx: int = 1, limit: int | None = None):
        try:
            return jobs.start_infer(round_idx=round_idx, config_path=config_path, limit=limit)
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    @app.post("/api/jobs/enrich")
    def start_enrich(round_idx: int = 1, limit: int | None = None):
        try:
            return jobs.start_enrich(round_idx=round_idx, config_path=config_path, limit=limit)
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    @app.post("/api/jobs/copula")
    def start_copula(round_idx: int = 1, class_id: int = 0):
        try:
            return jobs.start_copula(round_idx=round_idx, class_id=class_id, config_path=config_path)
        except RuntimeError as exc:
            raise HTTPException(409, detail=str(exc)) from exc

    @app.post("/api/jobs/reset")
    def reset_job():
        return jobs.reset_job()

    @app.get("/api/jobs/status")
    def job_status():
        return jobs.poll()

    @app.get("/api/jobs/log")
    def job_log(round_idx: int = 1, phase: str = "infer", tail: int = 8000):
        return {"log": jobs.read_log_tail(round_idx, phase=phase, max_chars=tail)}

    @app.get("/api/jobs/infer/monitor")
    def infer_monitor(round_idx: int = 1):
        state = jobs.poll()
        log_tail = jobs.read_log_tail(round_idx, phase="infer", max_chars=8000)
        payload = build_infer_monitor_payload(cfg, round_idx, state, log_tail)
        return payload

    @app.get("/api/plots/registry")
    def plots_registry(class_id: int):
        return registry_for_class(class_id)

    @app.get("/api/plots/{class_id}/{plot_id}.png")
    def plot_png(class_id: int, plot_id: str, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        samples = load_class_samples(class_id, cfg, rid)
        if not samples and plot_id != "class_distribution":
            raise HTTPException(404, detail="无 enriched 数据")
        extra = None
        if plot_id == "class_distribution":
            extra = {"class_counts": get_data_status(cfg, rid)["class_counts"]}
        png = plot_common.render_plot(plot_id, samples, extra=extra)
        return Response(content=png, media_type="image/png")

    @app.post("/api/plots/save")
    def save_plot(body: SavePlotRequest):
        rid = effective_round(cfg, body.round_idx)
        samples = load_class_samples(body.class_id, cfg, rid)
        extra = None
        if body.plot_id == "class_distribution":
            extra = {"class_counts": get_data_status(cfg, rid)["class_counts"]}
        png = plot_common.render_plot(body.plot_id, samples, extra=extra)
        out_dir = get_exports_dir(cfg) / f"class_{body.class_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{body.plot_id}_{ts}.png"
        path.write_bytes(png)
        return {"path": str(path)}

    @app.get("/api/copula/{class_id}/status")
    def copula_status(class_id: int, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        result = load_copula_result(cfg, rid, class_id)
        if result is None:
            return {"fitted": False}
        return {"fitted": True, **{k: result[k] for k in ("n_samples", "n_vars", "best_copula_type", "comparison") if k in result}}

    @app.get("/api/copula/{class_id}/marginals.png")
    def copula_marginals(class_id: int, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        from src.chapter4_characteristics.analysis.copula_service import load_mode_matrix
        samples = load_class_samples(class_id, cfg, rid)
        matrix, names = load_mode_matrix(samples, int(cfg.get("copula_n_modes", 8)), int(cfg.get("copula_max_samples", 5000)))
        png = copula_viz.plot_marginal_grid(matrix, names)
        return Response(content=png, media_type="image/png")

    @app.get("/api/copula/{class_id}/comparison.png")
    def copula_comparison(class_id: int, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        result = load_copula_result(cfg, rid, class_id)
        if result is None:
            result = run_copula_analysis(class_id, cfg, rid)
        png = copula_viz.plot_comparison_bar(result.get("comparison", []))
        return Response(content=png, media_type="image/png")

    @app.get("/api/copula/{class_id}/result.json")
    def copula_result_json(class_id: int, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        result = load_copula_result(cfg, rid, class_id)
        if result is None:
            raise HTTPException(404, detail="尚未拟合 Copula")
        return result

    @app.get("/api/others/samples")
    def others_samples(page: int = 0, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        return list_others_samples(cfg, rid, page=page)

    @app.get("/api/others/samples/{sample_idx}")
    def others_detail(sample_idx: int, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        return build_sample_detail(sample_idx, cfg, rid)

    @app.get("/api/others/samples/{sample_idx}/neighbors")
    def others_neighbors(sample_idx: int, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        return find_neighbors(sample_idx, cfg, rid, int(cfg.get("others_neighbor_topk", 3)))

    @app.get("/api/others/samples/{sample_idx}/timeline")
    def others_timeline(sample_idx: int, radius: int = 10, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        return build_timeline(sample_idx, cfg, rid, radius=radius)

    @app.get("/api/others/samples/{sample_idx}/figures/{name}.png")
    def others_figure(sample_idx: int, name: str, round_idx: int | None = None):
        if name not in FIGURE_NAMES:
            raise HTTPException(404, detail=f"未知 figure: {name}")
        rid = effective_round(cfg, round_idx)
        png = render_figure(sample_idx, name, cfg, rid)
        if png is None:
            raise HTTPException(404, detail="无法生成图像（可能缺风数据）")
        return Response(content=png, media_type="image/png")

    @app.post("/api/others/samples/{sample_idx}/export")
    def others_export(sample_idx: int, round_idx: int | None = None):
        rid = effective_round(cfg, round_idx)
        zip_path = export_report(sample_idx, cfg, rid)
        return FileResponse(zip_path, filename=Path(zip_path).name, media_type="application/zip")

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    port = args.port or int(cfg.get("webui_port", 8766))
    host = cfg.get("webui_host", "127.0.0.1")
    app = create_app(args.config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
