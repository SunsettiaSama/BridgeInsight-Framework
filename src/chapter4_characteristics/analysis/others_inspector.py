from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.chapter4_characteristics.analysis.data_loader import (
    get_nested,
    inference_by_idx,
    load_class_samples,
    load_others_index,
)
from src.chapter4_characteristics.settings import (
    get_chapter4_root,
    get_reference_psd_path,
    get_reference_stats_path,
    load_config,
)
from src.chapter3_identifier.augment.figures.render import (
    context as context_figures,
    sample as figures,
)
from src.data_processer.preprocess.get_data_wind import parse_single_metadata_to_wind_data
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

FIGURE_NAMES = [
    "timeseries",
    "spectrum",
    "trajectory",
    "context_timeseries",
    "context_spectrogram",
    "wind_timeseries",
    "vib_wind_joint",
    "psd_class_overlay",
    "inplane_outplane_overlay",
]


def _load_reference_stats(cfg: dict) -> dict:
    path = get_reference_stats_path(cfg)
    if not path.exists():
        return {"classes": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_reference_psd(cfg: dict) -> dict:
    path = get_reference_psd_path(cfg)
    if not path.exists():
        return {"classes": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_enriched_sample(sample_idx: int, cfg: dict) -> Optional[dict]:
    for s in load_class_samples(3, cfg):
        if int(s.get("sample_idx", -1)) == sample_idx:
            return s
    return None


def compute_deviations(features: dict, ref_stats: dict, keys: List[str]) -> List[dict]:
    out = []
    for k in keys:
        val = get_nested(features, k)
        if val is None:
            continue
        val = float(val)
        best_class = None
        best_z = 0.0
        for cid in ("0", "1", "2"):
            ref = (ref_stats.get("classes") or {}).get(cid, {}).get(k)
            if not ref or ref.get("std", 0) <= 1e-12:
                continue
            z = (val - ref["mean"]) / ref["std"]
            if abs(z) > abs(best_z):
                best_z = z
                best_class = int(cid)
        out.append({
            "key": k,
            "value": val,
            "ref_class": best_class,
            "z_score": float(best_z),
            "highlight": abs(best_z) > 2.0,
        })
    out.sort(key=lambda x: -abs(x["z_score"]))
    return out


def generate_clues(detail: dict) -> List[str]:
    clues: List[str] = []
    if detail.get("inplane_prediction") != detail.get("outplane_prediction"):
        clues.append("面内/面外预测不一致，可能为混合态或传感器对不齐")
    proba = detail.get("proba") or []
    if len(proba) >= 4 and float(detail.get("uncertainty", 0)) > 0.5:
        ordered = sorted(enumerate(proba), key=lambda x: -x[1])
        clues.append(f"高不确定性，次候选类别为 {ordered[1][0]} (p={ordered[1][1]:.2f})")
    entropy = get_nested(detail.get("features") or {}, "spectral_inplane.spectral_entropy")
    dom_ratio = get_nested(detail.get("features") or {}, "spectral_inplane.dominant_mode_energy_ratio")
    ellip = get_nested(detail.get("features") or {}, "cross_coupling.ellipticity")
    if entropy is not None and dom_ratio is not None and float(entropy) > 2.5 and float(dom_ratio) < 0.3:
        clues.append("高谱熵 + 低主频能量占比，形态接近随机振动")
    if dom_ratio is not None and float(dom_ratio) > 0.6 and ellip is not None and float(ellip) < 0.3:
        clues.append("窄带高能量 + 低椭圆率，形态接近 VIV")
    if ellip is not None and float(ellip) > 0.6:
        clues.append("高椭圆率，形态接近 RWIV 风雨振")
    return clues


def _feature_vector(sample: dict, keys: List[str]) -> Optional[np.ndarray]:
    vals = []
    for k in keys:
        v = get_nested(sample, k)
        if v is None:
            return None
        vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def find_neighbors(sample_idx: int, cfg: dict, topk: int = 3) -> List[dict]:
    target = _find_enriched_sample(sample_idx, cfg)
    if target is None:
        return []
    keys = list(cfg.get("reference_feature_keys", []))
    tv = _feature_vector(target, keys)
    if tv is None:
        return []

    candidates: List[tuple[float, dict]] = []
    for class_id in (0, 1, 2):
        for s in load_class_samples(class_id, cfg):
            fv = _feature_vector(s, keys)
            if fv is None:
                continue
            dist = float(np.linalg.norm((fv - tv) / (np.abs(tv) + 1e-6)))
            candidates.append((dist, {
                "sample_idx": s.get("sample_idx"),
                "class_id": class_id,
                "distance": dist,
                "rms_in": get_nested(s, "time_stats_inplane.rms"),
                "dominant_freq_in": (s.get("psd_inplane") or {}).get("frequencies", [None])[0],
            }))
    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates[:topk]]


def build_timeline(sample_idx: int, cfg: dict, radius: int = 10) -> List[dict]:
    infer = inference_by_idx(cfg)
    rec = infer.get(sample_idx)
    if rec is None:
        return []
    fp = rec.get("inplane_file_path")
    wi = int(rec.get("window_index", 0))
    rows = []
    for idx, r in infer.items():
        if r.get("inplane_file_path") != fp:
            continue
        w = int(r.get("window_index", 0))
        if abs(w - wi) <= radius:
            rows.append({
                "sample_idx": idx,
                "window_index": w,
                "prediction": r.get("prediction"),
                "uncertainty": r.get("uncertainty"),
                "is_current": idx == sample_idx,
            })
    rows.sort(key=lambda x: x["window_index"])
    return rows


def list_others_samples(
    cfg: dict,
    page: int = 0,
    page_size: Optional[int] = None,
) -> dict:
    page_size = page_size or int(cfg.get("others_queue_page_size", 30))
    index = load_others_index(cfg)
    samples = index.get("samples", [])
    start = page * page_size
    end = start + page_size
    return {
        "total": len(samples),
        "page": page,
        "page_size": page_size,
        "samples": samples[start:end],
    }


def build_sample_detail(sample_idx: int, cfg: dict) -> dict:
    features = _find_enriched_sample(sample_idx, cfg)
    if features is None:
        raise FileNotFoundError(f"Others 样本不存在：{sample_idx}")
    infer = inference_by_idx(cfg).get(sample_idx, {})
    ref_stats = _load_reference_stats(cfg)
    keys = list(cfg.get("reference_feature_keys", []))
    proba = infer.get("proba") or [0, 0, 0, 0]
    ordered = sorted(enumerate(proba), key=lambda x: -x[1])
    detail = {
        "sample_idx": sample_idx,
        "prediction": infer.get("prediction", 3),
        "proba": proba,
        "uncertainty": infer.get("uncertainty"),
        "inplane_prediction": infer.get("inplane_prediction"),
        "outplane_prediction": infer.get("outplane_prediction"),
        "metadata": {
            "timestamp": features.get("timestamp"),
            "inplane_sensor_id": features.get("inplane_sensor_id"),
            "outplane_sensor_id": features.get("outplane_sensor_id"),
            "window_index": features.get("window_idx"),
            "inplane_file_path": features.get("inplane_file_path"),
            "outplane_file_path": features.get("outplane_file_path"),
        },
        "features": features,
        "deviations": compute_deviations(features, ref_stats, keys),
        "second_candidate": {"class_id": int(ordered[1][0]), "proba": float(ordered[1][1])} if len(ordered) > 1 else None,
    }
    detail["clues"] = generate_clues({**detail, **infer})
    return detail


def _record_from_detail(detail: dict) -> dict:
    m = detail["metadata"]
    return {
        "inplane_file_path": m.get("inplane_file_path"),
        "outplane_file_path": m.get("outplane_file_path"),
        "window_index": m.get("window_index"),
        "inplane_sensor_id": m.get("inplane_sensor_id"),
        "outplane_sensor_id": m.get("outplane_sensor_id"),
        "proba": detail.get("proba"),
        "prediction": detail.get("prediction"),
    }


def render_figure(sample_idx: int, name: str, cfg: dict) -> Optional[bytes]:
    detail = build_sample_detail(sample_idx, cfg)
    rec = _record_from_detail(detail)
    m = detail["metadata"]
    before = int(cfg.get("context_windows_before", 5))
    after = int(cfg.get("context_windows_after", 5))

    if name == "timeseries":
        figs = figures.render_sample_figures(rec)
        return figs.get("in_timeseries") or figs.get("out_timeseries")
    if name == "spectrum":
        figs = figures.render_sample_figures(rec)
        return figs.get("in_spectrum")
    if name == "trajectory":
        figs = figures.render_sample_figures(rec)
        return figs.get("trajectory")
    if name == "context_timeseries":
        pngs = context_figures.render_context_figures(
            m["inplane_file_path"],
            int(m["window_index"]),
            direction="inplane",
            sensor_id=str(m.get("inplane_sensor_id", "in")),
            before=before,
            after=after,
        )
        return pngs[0]
    if name == "context_spectrogram":
        pngs = context_figures.render_context_figures(
            m["inplane_file_path"],
            int(m["window_index"]),
            direction="inplane",
            sensor_id=str(m.get("inplane_sensor_id", "in")),
            before=before,
            after=after,
        )
        return pngs[1]
    if name == "inplane_outplane_overlay":
        return _plot_overlay(m, before, after)
    if name == "wind_timeseries":
        return _plot_wind_timeseries(detail["features"], cfg)
    if name == "vib_wind_joint":
        return _plot_vib_wind_joint(m, before, after, detail["features"], cfg)
    if name == "psd_class_overlay":
        return _plot_psd_overlay(detail["features"], cfg)
    return None


def _plot_overlay(m: dict, before: int, after: int) -> bytes:
    from src.chapter3_identifier.augment.features.context_window import extract_context_window

    ctx_in = extract_context_window(m["inplane_file_path"], int(m["window_index"]), before=before, after=after)
    ctx_out = extract_context_window(m["outplane_file_path"], int(m["window_index"]), before=before, after=after)
    t = np.arange(len(ctx_in.signal)) / ctx_in.fs + ctx_in.t0_offset_s
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, ctx_in.signal, label="面内", alpha=0.8, lw=0.8)
    ax.plot(t, ctx_out.signal, label="面外", alpha=0.8, lw=0.8)
    ax.axvspan(ctx_in.current_start_s + ctx_in.t0_offset_s, ctx_in.current_end_s + ctx_in.t0_offset_s, color="yellow", alpha=0.2)
    ax.legend()
    ax.set_title("面内/面外长时程叠加")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plot_wind_timeseries(features: dict, cfg: dict) -> Optional[bytes]:
    ts = features.get("timestamp") or []
    if len(ts) < 3:
        return None
    from src.chapter3_identifier.identifier.feature_analysis._wind import (
        build_wind_lookup,
        load_wind_metadata,
    )

    wind_list = load_wind_metadata(str(cfg["wind_metadata_path"]))
    lookup = build_wind_lookup(wind_list)
    key = (int(ts[0]), int(ts[1]), int(ts[2]))
    records = lookup.get(key)
    if not records:
        return None
    fig, axes = plt.subplots(len(records), 1, figsize=(10, 2.5 * len(records)), squeeze=False)
    for i, meta in enumerate(records):
        parsed = parse_single_metadata_to_wind_data(meta, enable_denoise=False)
        data = parsed.get("data")
        if data is None:
            continue
        speed = data[:, 0] if data.ndim > 1 else data
        axes[i, 0].plot(speed, lw=0.8)
        axes[i, 0].set_title(meta.get("sensor_id", f"wind_{i}"))
        axes[i, 0].grid(True, alpha=0.3)
    fig.suptitle("风传感器时程")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plot_vib_wind_joint(m: dict, before: int, after: int, features: dict, cfg: dict) -> Optional[bytes]:
    from src.chapter3_identifier.augment.features.context_window import extract_context_window

    ctx = extract_context_window(m["inplane_file_path"], int(m["window_index"]), before=before, after=after)
    ws = cfg.get("window_size", 3000)
    rms = []
    for i in range(0, len(ctx.signal) - ws + 1, ws):
        seg = ctx.signal[i : i + ws]
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
    wind_png = _plot_wind_timeseries(features, cfg)
    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(rms, "b-", label="振动 RMS")
    ax1.set_ylabel("RMS")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("振动 RMS 滑动曲线")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plot_psd_overlay(features: dict, cfg: dict) -> Optional[bytes]:
    ref = _load_reference_psd(cfg)
    psd = features.get("psd_inplane") or {}
    freqs = psd.get("frequencies") or []
    powers = psd.get("powers") or []
    if not freqs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs, powers, "k-", lw=1.5, label="当前样本")
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    labels = ["Normal", "VIV", "RWIV"]
    for i, (cid, label) in enumerate(zip(("0", "1", "2"), labels)):
        curve = (ref.get("classes") or {}).get(cid, {}).get("inplane")
        if curve and curve.get("frequencies"):
            ax.plot(curve["frequencies"], curve["powers"], "--", color=colors[i], alpha=0.8, label=f"{label} 均值")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD 类间对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def export_report(sample_idx: int, cfg: dict) -> str:
    detail = build_sample_detail(sample_idx, cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = get_chapter4_root(cfg) / "others_reports" / f"{sample_idx}_{ts}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    neighbors = find_neighbors(sample_idx, cfg, int(cfg.get("others_neighbor_topk", 3)))
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump({"deviations": detail["deviations"], "neighbors": neighbors}, f, ensure_ascii=False, indent=2)

    lines = [
        f"# Others 样本调查报告",
        f"- sample_idx: {sample_idx}",
        f"- uncertainty: {detail.get('uncertainty')}",
        f"- clues: " + "; ".join(detail.get("clues") or []),
    ]
    for d in detail.get("deviations", [])[:5]:
        lines.append(f"- 偏离 {d['key']}: z={d['z_score']:.2f}")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    for name in FIGURE_NAMES:
        png = render_figure(sample_idx, name, cfg)
        if png:
            (fig_dir / f"{name}.png").write_bytes(png)

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(out_dir.parent))
    return str(zip_path)
