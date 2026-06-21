from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

_FIXTURES_DIR = Path(__file__).resolve().parent / "data"
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


def _write_mock_wind(path: Path, n_points: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_points):
        speed = 5.0 + 0.01 * i
        direction = 180.0 + 0.1 * i
        angle = 60.0 + 0.05 * i
        lines.append(f"{speed:.2f},{direction:.2f},{angle:.2f}")
    path.write_text(",".join(lines), encoding="utf-8")


def _write_checkpoint(path: Path) -> None:
    import torch

    from src.chapter3_identifier.augment.features.spectrum import psd_bin_count
    from src.chapter3_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN
    from src.chapter3_identifier.augment.settings import DualStreamResCNNConfig

    fs, nfft, freq_max_hz = 50.0, 2048, 25.0
    psd_bins = psd_bin_count(fs, nfft, freq_max_hz)
    cfg_dict = {
        "num_classes": 4,
        "time_branch": {"input_size": 3000},
        "spec_branch": {"input_size": psd_bins},
    }
    model = DualStreamResCNN(DualStreamResCNNConfig.from_dict(cfg_dict))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg_dict,
            "epoch": 0,
            "smoke": True,
        },
        path,
    )


def ensure_smoke_fixtures(force: bool = False) -> Path:
    from src.chapter4_characteristics._bootstrap import ensure_paths

    ensure_paths()
    checkpoint_path = _FIXTURES_DIR / "identifier" / "best_checkpoint.pth"
    if _READY_MARKER.exists() and checkpoint_path.exists() and not force:
        return _FIXTURES_DIR

    output_dir = Path(__file__).resolve().parent / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if _FIXTURES_DIR.exists():
        shutil.rmtree(_FIXTURES_DIR)

    vic_dir = _FIXTURES_DIR / "vic"
    wind_dir = _FIXTURES_DIR / "wind"
    timestamps = [(1, 1, 0), (1, 1, 1), (1, 1, 2)]
    pairs = [("MOCK-IN-01", "MOCK-OUT-01")]

    vib_records = []
    wind_records = []
    for month, day, hour in timestamps:
        wind_path = wind_dir / f"WIND-01_{month:02d}{day:02d}{hour:02d}.txt"
        _write_mock_wind(wind_path)
        wind_records.append(
            {
                "sensor_id": "WIND-01",
                "month": month,
                "day": day,
                "hour": hour,
                "file_path": str(wind_path.resolve()),
            }
        )
        for in_id, out_id in pairs:
            in_path = vic_dir / f"{in_id}_{month:02d}{day:02d}{hour:02d}.VIC"
            out_path = vic_dir / f"{out_id}_{month:02d}{day:02d}{hour:02d}.VIC"
            _write_mock_vic(in_path, in_id)
            _write_mock_vic(out_path, out_id)
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

    (_FIXTURES_DIR / "vib_metadata.json").write_text(
        json.dumps(vib_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (_FIXTURES_DIR / "wind_metadata.json").write_text(
        json.dumps(wind_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_yaml = Path(__file__).resolve().parent / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"vib_metadata_path: {(_FIXTURES_DIR / 'vib_metadata.json').as_posix()}",
                f"wind_metadata_path: {(_FIXTURES_DIR / 'wind_metadata.json').as_posix()}",
                "cable_pairs:",
                "  - [MOCK-IN-01, MOCK-OUT-01]",
                "wind_sensor_ids:",
                "  - WIND-01",
                "window_size: 3000",
                "require_wind_alignment: false",
                "use_cache: false",
                "time_ordered: true",
                "missing_rate_threshold: 0.05",
            ]
        ),
        encoding="utf-8",
    )

    _write_checkpoint(checkpoint_path)

    feature_yaml = Path(__file__).resolve().parent / "feature_analysis.yaml"
    feature_yaml.write_text(
        "\n".join(
            [
                "fs: 50.0",
                "window_size: 3000",
                "enable_denoise: false",
                "psd_nperseg: 2048",
                "psd_n_modes: 4",
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

    _READY_MARKER.write_text("ok", encoding="utf-8")
    return _FIXTURES_DIR
