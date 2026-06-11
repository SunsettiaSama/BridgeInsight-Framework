from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

_FIXTURES_DIR = Path(__file__).resolve().parent / "data"
_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_READY_MARKER = _FIXTURES_DIR / ".ready"


def _write_mock_vic(path: Path, sensor_id: str, n_samples: int = 9000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = 50.0
    t = np.arange(n_samples, dtype=np.float32) / fs
    seed = sum(ord(c) for c in sensor_id)
    rng = np.random.default_rng(seed)
    freq_hz = 1.2 + (seed % 7) * 0.1
    signal = (0.12 * np.sin(2 * np.pi * freq_hz * t) + 0.01 * rng.standard_normal(n_samples)).astype(
        np.float32
    )
    header = sensor_id.encode("utf-8") + b"_"
    path.write_bytes(header + signal.tobytes())


def ensure_smoke_fixtures(force: bool = False) -> Path:
    from src.chapter3_identifier.augment._bootstrap import ensure_paths, resolve_path

    ensure_paths()
    if _READY_MARKER.exists() and not force:
        return _FIXTURES_DIR

    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vic_dir = _FIXTURES_DIR / "vic"
    in_path = vic_dir / "MOCK-IN-01_010101.VIC"
    out_path = vic_dir / "MOCK-OUT-01_010101.VIC"
    _write_mock_vic(in_path, "MOCK-IN-01")
    _write_mock_vic(out_path, "MOCK-OUT-01")

    in_fp = str(in_path.resolve())
    out_fp = str(out_path.resolve())

    gold = [
        {
            "sample_id": "gold_0",
            "annotation": 1,
            "file_path": in_fp,
            "window_index": 0,
            "data_type": "vic",
            "is_gold": True,
        }
    ]
    (_FIXTURES_DIR / "gold_annotations.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    split = {"train_keys": [[in_fp, 0]], "val_keys": []}
    (_FIXTURES_DIR / "split_indices.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    inference_records = [
        {
            "sample_idx": 1001,
            "prediction": 2,
            "proba": [0.05, 0.1, 0.8, 0.05],
            "uncertainty": 0.2,
            "inplane_prediction": 2,
            "outplane_prediction": 2,
            "inplane_file_path": in_fp,
            "outplane_file_path": out_fp,
            "window_index": 1,
            "inplane_sensor_id": "MOCK-IN-01",
            "outplane_sensor_id": "MOCK-OUT-01",
            "timestamp": [1, 1, 1],
            "already_annotated": False,
            "is_gold": False,
        }
    ]
    round_dir = _OUTPUT_DIR / "rounds" / "round_01"
    round_dir.mkdir(parents=True, exist_ok=True)
    inference_payload = {
        "round_idx": 1,
        "generated_at": "smoke",
        "records": inference_records,
    }
    (round_dir / "inference.json").write_text(
        json.dumps(inference_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg = {
        "gold_annotation_path": str((_FIXTURES_DIR / "gold_annotations.json").resolve()),
        "split_indices_path": str((_FIXTURES_DIR / "split_indices.json").resolve()),
        "rounds_output_dir": str(_OUTPUT_DIR / "rounds"),
        "job_state_path": str((_OUTPUT_DIR / "job_state.json").resolve()),
        "window_size": 3000,
        "fs": 50.0,
        "nfft": 2048,
        "freq_max_hz": 25.0,
        "queue_page_size": 50,
        "context_total_seconds": 180,
        "context_spectrogram_segment_s": 2.0,
        "context_figure_cache_size": 30,
        "webui_port": 8765,
        "num_classes": 4,
        "label_names": ["Normal", "VIV", "RWIV", "Others"],
        "best_params": str((_FIXTURES_DIR / "best_params.json").resolve()),
        "epochs": 1,
        "train_val_ratio": 0.8,
        "random_seed": 42,
        "batch_size": 2,
        "class_weights": [1.0, 1.0, 1.0, 1.0],
        "focal_gamma": 2.0,
    }
    (_FIXTURES_DIR / "best_params.json").write_text(
        json.dumps({"best_params": {"batch_size": 2, "learning_rate": 1e-4}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    smoke_cfg_path = _FIXTURES_DIR / "smoke.yaml"
    import yaml

    smoke_cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    _READY_MARKER.write_text("ok", encoding="utf-8")
    return _FIXTURES_DIR


def smoke_config_path() -> Path:
    ensure_smoke_fixtures()
    return _FIXTURES_DIR / "smoke.yaml"
