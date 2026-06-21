from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


_DEMO_DIR = Path(__file__).resolve().parent
_DATA_DIR = _DEMO_DIR / "data"
_OUTPUT_DIR = _DEMO_DIR / "output"
_READY_MARKER = _DATA_DIR / ".ready"


def _write_mock_vic(path: Path, sensor_id: str, phase_shift: float, n_samples: int = 9000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = 50.0
    t = np.arange(n_samples, dtype=np.float32) / fs
    seed = sum(ord(c) for c in sensor_id) + int(phase_shift * 100)
    rng = np.random.default_rng(seed)
    freq_hz = 0.9 + (seed % 9) * 0.15
    base = 0.08 * np.sin(2 * np.pi * freq_hz * t + phase_shift)
    mod = 0.02 * np.sin(2 * np.pi * 0.1 * t)
    noise = 0.01 * rng.standard_normal(n_samples)
    signal = (base + mod + noise).astype(np.float32)
    header = sensor_id.encode("utf-8") + b"_"
    path.write_bytes(header + signal.tobytes())


def _write_mock_wind(path: Path, n_points: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_points):
        speed = 4.0 + 0.03 * i
        direction = 160.0 + 0.2 * i
        angle = 40.0 + 0.07 * i
        lines.append(f"{speed:.3f},{direction:.3f},{angle:.3f}")
    path.write_text(",".join(lines), encoding="utf-8")


def _class_feature_profile(class_id: int) -> dict:
    if class_id == 0:
        return {"rms_in": 0.10, "rms_out": 0.11, "kurt_in": 3.1, "kurt_out": 3.0, "ellip": 0.20, "wind": 5.2, "entropy": 1.8}
    if class_id == 1:
        return {"rms_in": 0.22, "rms_out": 0.16, "kurt_in": 4.6, "kurt_out": 4.2, "ellip": 0.28, "wind": 7.0, "entropy": 1.4}
    if class_id == 2:
        return {"rms_in": 0.28, "rms_out": 0.31, "kurt_in": 3.9, "kurt_out": 4.1, "ellip": 0.72, "wind": 9.0, "entropy": 2.0}
    return {"rms_in": 0.16, "rms_out": 0.19, "kurt_in": 5.2, "kurt_out": 4.8, "ellip": 0.55, "wind": 8.1, "entropy": 2.8}


def _make_psd(base_freq: float, scale: float = 1.0) -> tuple[list[float], list[float]]:
    freqs = [round(base_freq + i * 0.35, 3) for i in range(10)]
    raw = np.array([1.0 / (1.0 + i) for i in range(10)], dtype=np.float64)
    raw = raw * scale
    return freqs, raw.tolist()


def _proba_for_class(class_id: int, sample_offset: int) -> list[float]:
    p = [0.04, 0.04, 0.04, 0.04]
    p[class_id] = 0.80 - 0.02 * (sample_offset % 3)
    second = (class_id + 1) % 4
    p[second] = 0.10 + 0.01 * (sample_offset % 2)
    remain = 1.0 - p[class_id] - p[second]
    slots = [i for i in range(4) if i not in (class_id, second)]
    p[slots[0]] = remain * 0.55
    p[slots[1]] = remain * 0.45
    return [float(round(v, 6)) for v in p]


def _sample_row(
    sample_idx: int,
    class_id: int,
    inplane_sensor_id: str,
    outplane_sensor_id: str,
    inplane_file_path: str,
    outplane_file_path: str,
    timestamp: tuple[int, int, int],
    window_index: int,
) -> tuple[dict, dict]:
    profile = _class_feature_profile(class_id)
    offset = sample_idx % 6
    base_freq = 0.9 + 0.4 * class_id + 0.03 * offset
    in_freqs, in_powers = _make_psd(base_freq, scale=1.0 + 0.1 * class_id)
    out_freqs, out_powers = _make_psd(base_freq * 1.03, scale=0.9 + 0.08 * class_id)
    proba = _proba_for_class(class_id, offset)
    uncertainty = 1.0 - max(proba)

    infer = {
        "sample_idx": sample_idx,
        "prediction": class_id,
        "proba": proba,
        "uncertainty": float(round(uncertainty, 6)),
        "inplane_prediction": class_id if class_id != 3 else (1 if offset % 2 == 0 else 2),
        "outplane_prediction": class_id if class_id != 3 else (2 if offset % 2 == 0 else 1),
        "inplane_file_path": inplane_file_path,
        "outplane_file_path": outplane_file_path,
        "window_index": window_index,
        "inplane_sensor_id": inplane_sensor_id,
        "outplane_sensor_id": outplane_sensor_id,
        "timestamp": list(timestamp),
    }

    enriched = {
        "sample_idx": sample_idx,
        "window_idx": window_index,
        "timestamp": list(timestamp),
        "inplane_sensor_id": inplane_sensor_id,
        "outplane_sensor_id": outplane_sensor_id,
        "inplane_file_path": inplane_file_path,
        "outplane_file_path": outplane_file_path,
        "time_stats_inplane": {
            "rms": float(round(profile["rms_in"] + 0.006 * offset, 6)),
            "kurtosis": float(round(profile["kurt_in"] + 0.12 * (offset % 3), 6)),
        },
        "time_stats_outplane": {
            "rms": float(round(profile["rms_out"] + 0.005 * offset, 6)),
            "kurtosis": float(round(profile["kurt_out"] + 0.11 * (offset % 4), 6)),
        },
        "spectral_inplane": {
            "spectral_entropy": float(round(profile["entropy"] + 0.05 * (offset % 2), 6)),
            "dominant_mode_energy_ratio": float(round(0.62 - 0.10 * class_id + 0.02 * (offset % 2), 6)),
        },
        "cross_coupling": {
            "ellipticity": float(round(profile["ellip"] + 0.03 * (offset % 3), 6)),
        },
        "wind_stats": [
            {
                "mean_wind_speed": float(round(profile["wind"] + 0.2 * (offset % 4), 6)),
                "mean_wind_direction": float(round(170.0 + 0.5 * offset, 6)),
            }
        ],
        "psd_inplane": {"frequencies": in_freqs, "powers": in_powers},
        "psd_outplane": {"frequencies": out_freqs, "powers": out_powers},
    }
    return infer, enriched


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_demo_fixtures(force: bool = False) -> Path:
    from src.chapter4_characteristics._bootstrap import ensure_paths
    from src.chapter4_characteristics.analysis.index_builder import build_others_index
    from src.chapter4_characteristics.analysis.reference_builder import post_enrich_artifacts
    from src.chapter4_characteristics.settings import load_config

    ensure_paths()
    checkpoint_path = _DATA_DIR / "identifier" / "best_checkpoint.pth"
    inference_path = _OUTPUT_DIR / "inference" / "inference.json"
    enriched_path = _OUTPUT_DIR / "enriched" / "class_0_normal"
    if _READY_MARKER.exists() and checkpoint_path.exists() and inference_path.exists() and enriched_path.exists() and not force:
        return _OUTPUT_DIR

    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    if _DATA_DIR.exists():
        shutil.rmtree(_DATA_DIR)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    vic_dir = _DATA_DIR / "vic"
    wind_dir = _DATA_DIR / "wind"

    cable_pairs = [("DEMO-IN-01", "DEMO-OUT-01"), ("DEMO-IN-02", "DEMO-OUT-02")]
    timestamps = [(1, 1, 0), (1, 1, 1), (1, 1, 2)]

    vib_records = []
    wind_records = []
    for month, day, hour in timestamps:
        wind_path = wind_dir / f"WIND-DEMO_{month:02d}{day:02d}{hour:02d}.txt"
        _write_mock_wind(wind_path)
        wind_records.append(
            {
                "sensor_id": "WIND-DEMO",
                "month": month,
                "day": day,
                "hour": hour,
                "file_path": str(wind_path.resolve()),
            }
        )
        for pair_idx, (in_id, out_id) in enumerate(cable_pairs):
            in_path = vic_dir / f"{in_id}_{month:02d}{day:02d}{hour:02d}.VIC"
            out_path = vic_dir / f"{out_id}_{month:02d}{day:02d}{hour:02d}.VIC"
            _write_mock_vic(in_path, in_id, phase_shift=0.2 * pair_idx)
            _write_mock_vic(out_path, out_id, phase_shift=0.4 + 0.2 * pair_idx)
            for sensor_id, fp in ((in_id, in_path), (out_id, out_path)):
                vib_records.append(
                    {
                        "sensor_id": sensor_id,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "file_path": str(fp.resolve()),
                        "missing_rate": 0.0,
                    }
                )

    _write_json(_DATA_DIR / "vib_metadata.json", vib_records)
    _write_json(_DATA_DIR / "wind_metadata.json", wind_records)

    dataset_yaml = _DEMO_DIR / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"vib_metadata_path: {(_DATA_DIR / 'vib_metadata.json').as_posix()}",
                f"wind_metadata_path: {(_DATA_DIR / 'wind_metadata.json').as_posix()}",
                "cable_pairs:",
                "  - [DEMO-IN-01, DEMO-OUT-01]",
                "  - [DEMO-IN-02, DEMO-OUT-02]",
                "wind_sensor_ids:",
                "  - WIND-DEMO",
                "window_size: 3000",
                "require_wind_alignment: false",
                "use_cache: false",
                "time_ordered: true",
                "missing_rate_threshold: 0.05",
            ]
        ),
        encoding="utf-8",
    )

    feature_yaml = _DEMO_DIR / "feature_analysis.yaml"
    feature_yaml.write_text(
        "\n".join(
            [
                "fs: 50.0",
                "window_size: 3000",
                "enable_denoise: false",
                "psd_nperseg: 2048",
                "psd_n_modes: 6",
                "psd_min_peak_distance_hz: 0.1",
                "enable_psd_modes: true",
                "enable_spectral_features: true",
                "enable_time_stats: true",
                "enable_cross_coupling: true",
                "enable_wind_stats: true",
                "enable_reduced_velocity: false",
                "n_workers: 1",
                "split_by_sensor: false",
            ]
        ),
        encoding="utf-8",
    )

    inference_records: list[dict] = []
    by_class: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    sample_metadata = {}

    sample_idx = 0
    for class_id in (0, 1, 2, 3):
        for local_i in range(6):
            pair = cable_pairs[local_i % len(cable_pairs)]
            ts = timestamps[local_i % len(timestamps)]
            in_fp = vic_dir / f"{pair[0]}_{ts[0]:02d}{ts[1]:02d}{ts[2]:02d}.VIC"
            out_fp = vic_dir / f"{pair[1]}_{ts[0]:02d}{ts[1]:02d}{ts[2]:02d}.VIC"
            infer, enriched = _sample_row(
                sample_idx=sample_idx,
                class_id=class_id,
                inplane_sensor_id=pair[0],
                outplane_sensor_id=pair[1],
                inplane_file_path=str(in_fp.resolve()),
                outplane_file_path=str(out_fp.resolve()),
                timestamp=ts,
                window_index=local_i,
            )
            inference_records.append(infer)
            by_class[class_id].append(enriched)
            sample_metadata[str(sample_idx)] = {
                "cable_pair": [pair[0], pair[1]],
                "timestamp": [ts[0], ts[1], ts[2]],
                "window_idx": local_i,
                "inplane_sensor_id": pair[0],
                "outplane_sensor_id": pair[1],
                "inplane_file_path": str(in_fp.resolve()),
                "outplane_file_path": str(out_fp.resolve()),
                "missing_rate_in": 0.0,
                "missing_rate_out": 0.0,
                "has_wind": True,
            }
            sample_idx += 1

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"demo checkpoint placeholder")

    inference_dir = _OUTPUT_DIR / "inference"
    _write_json(
        inference_dir / "inference.json",
        {
            "generated_at": "demo-fixtures",
            "checkpoint": str(checkpoint_path.resolve()),
            "record_count": len(inference_records),
            "records": inference_records,
        },
    )
    _write_json(
        inference_dir / "predictions_enriched.json",
        {
            "metadata": {
                "checkpoint": str(checkpoint_path.resolve()),
                "dataset_config": str((_DEMO_DIR / "dataset.yaml").as_posix()),
                "enriched_at": "demo-fixtures",
                "num_samples": len(inference_records),
                "source": "chapter4_demo_fixtures",
            },
            "predictions": {str(r["sample_idx"]): int(r["prediction"]) for r in inference_records},
            "sample_metadata": sample_metadata,
            "by_file": {},
        },
    )
    _write_json(
        inference_dir / "manifest.json",
        {
            "generated_at": "demo-fixtures",
            "checkpoint": str(checkpoint_path.resolve()),
            "inference": "inference.json",
            "predictions_enriched": "predictions_enriched.json",
            "record_count": len(inference_records),
            "limit": None,
        },
    )

    enriched_dir = _OUTPUT_DIR / "enriched"
    class_dirs = {
        0: "class_0_normal",
        1: "class_1_viv",
        2: "class_2_rwiv",
        3: "class_3_transition",
    }
    for class_id, samples in by_class.items():
        class_dir = enriched_dir / class_dirs[class_id]
        class_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[str, list[dict]] = {}
        for sample in samples:
            sid = str(sample.get("inplane_sensor_id", "DEMO-SENSOR"))
            grouped.setdefault(sid, []).append(sample)
        for sensor_id, sensor_samples in grouped.items():
            _write_json(
                class_dir / f"{sensor_id}.json",
                {"sensor_id": sensor_id, "samples": sensor_samples},
            )

    demo_cfg = load_config(str((_DEMO_DIR.parent / "config" / "demo.yaml").resolve()))
    post_enrich_artifacts(demo_cfg)
    build_others_index(demo_cfg)
    _write_json(
        _OUTPUT_DIR / "job_state.json",
        {"status": "idle", "phase": None, "pid": None, "log_path": None, "error": None},
    )
    _READY_MARKER.write_text("ok", encoding="utf-8")
    return _OUTPUT_DIR
