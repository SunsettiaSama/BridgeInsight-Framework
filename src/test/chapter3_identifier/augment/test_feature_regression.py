from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SMOKE_FIXTURES = _PROJECT_ROOT / "src" / "chapter3_identifier" / "augment" / "smoke_fixtures" / "data"


def _write_mock_vic(path: Path, sensor_id: str, n_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = 50.0
    t = np.arange(n_samples, dtype=np.float32) / fs
    signal = (0.1 * np.sin(2 * np.pi * 1.5 * t)).astype(np.float32)
    header = sensor_id.encode("utf-8") + b"_"
    path.write_bytes(header + signal.tobytes())


def run_feature_regression_tests() -> None:
    from src.chapter3_identifier.augment._bootstrap import ensure_paths
    from src.chapter3_identifier.augment.features.context_window import (
        _files_are_time_adjacent,
        _parse_vic_start_datetime,
        extract_context_window,
    )
    from src.chapter3_identifier.augment.features.wind_index import WindMetadataIndex
    from src.chapter3_identifier.augment.figures.render.wind import (
        build_wind_render_payload,
        plot_wind_speed_rose,
        plot_wind_turbulence_rose,
    )
    from src.chapter3_identifier.augment.figures.types import SAMPLE_FIGURE_NAMES
    from src.chapter3_identifier.augment.smoke_fixtures.bootstrap import ensure_smoke_fixtures, smoke_config_path
    from src.chapter3_identifier.augment.settings import load_config
    from src.data_processer.preprocess.get_data_vib import VICWindowExtractor

    ensure_paths()
    ensure_smoke_fixtures(force=True)
    cfg = load_config(str(smoke_config_path()))
    passed = 0

    assert "wind_speed_rose" in SAMPLE_FIGURE_NAMES
    assert "wind_turbulence_rose" in SAMPLE_FIGURE_NAMES
    passed += 1
    print("[ok] 风特征图类型已注册")

    fixtures = _SMOKE_FIXTURES
    vic_dir = fixtures / "vic"
    hour_a = vic_dir / "MOCK-IN-01" / "01" / "01" / "MOCK-IN-01_010000.VIC"
    hour_b = vic_dir / "MOCK-IN-01" / "01" / "01" / "MOCK-IN-01_020000.VIC"
    hour_gap = vic_dir / "MOCK-IN-01" / "01" / "01" / "MOCK-IN-01_050000.VIC"
    hour_samples = 50 * 3600
    _write_mock_vic(hour_a, "MOCK-IN-01", hour_samples)
    _write_mock_vic(hour_b, "MOCK-IN-01", hour_samples)
    _write_mock_vic(hour_gap, "MOCK-IN-01", hour_samples)

    assert _parse_vic_start_datetime(str(hour_a)) < _parse_vic_start_datetime(str(hour_b))
    assert _files_are_time_adjacent(str(hour_a), hour_samples, str(hour_b), 50.0)
    assert not _files_are_time_adjacent(str(hour_b), hour_samples, str(hour_gap), 50.0)
    passed += 1
    print("[ok] VIC 文件时间邻接判断")

    ctx = extract_context_window(
        file_path=str(hour_b),
        window_index=1,
        window_size=3000,
        fs=50.0,
        before=3,
        after=3,
        extractor=VICWindowExtractor(enable_denoise=False),
        allow_cross_file=True,
    )
    assert ctx.current_end_s > ctx.current_start_s
    assert ctx.current_end_s - ctx.current_start_s == 60.0
    passed += 1
    print("[ok] 长窗口高亮区间按真实中心窗长度计算")

    ctx_gap = extract_context_window(
        file_path=str(hour_gap),
        window_index=1,
        window_size=3000,
        fs=50.0,
        before=6,
        after=0,
        extractor=VICWindowExtractor(enable_denoise=False),
        allow_cross_file=True,
    )
    assert ctx_gap.discontinuity_note is not None
    passed += 1
    print("[ok] 非相邻跨文件拼接会记录不连续提示")

    wind_meta = [
        {
            "sensor_id": "ST-UAN-G04-001-01",
            "month": 1,
            "day": 1,
            "hour": 1,
            "file_path": str((fixtures / "wind" / "MOCK-WIND-01_010100.txt").resolve()),
        }
    ]
    wind_meta_path = fixtures / "wind_metadata.json"
    wind_meta_path.write_text(json.dumps(wind_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    wind_file = Path(wind_meta[0]["file_path"])
    wind_file.parent.mkdir(parents=True, exist_ok=True)
    wind_speed = np.linspace(3.0, 8.0, 120, dtype=np.float32)
    wind_direction = np.linspace(0.0, 350.0, 120, dtype=np.float32)
    wind_attack = np.zeros(120, dtype=np.float32)
    triples = ",".join(
        f"{s:.3f},{d:.3f},{a:.3f}"
        for s, d, a in zip(wind_speed, wind_direction, wind_attack)
    )
    wind_file.write_text(triples, encoding="utf-8")

    dataset_yaml = fixtures / "dataset_with_wind.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"vib_metadata_path: {(fixtures / 'vib_metadata.json').resolve().as_posix()}",
                f"wind_metadata_path: {wind_meta_path.resolve().as_posix()}",
                "wind_sensor_ids:",
                "  - ST-UAN-G04-001-01",
            ]
        ),
        encoding="utf-8",
    )
    cfg_with_wind = dict(cfg)
    cfg_with_wind["inference_dataset_config"] = str(dataset_yaml.resolve())

    index = WindMetadataIndex(str(wind_meta_path), sensor_ids=["ST-UAN-G04-001-01"])
    assert index.lookup(1, 1, 1, "ST-UAN-G04-001-01") is not None
    passed += 1
    print("[ok] 风元数据索引按时间戳查找")

    record = {
        "sample_idx": 1001,
        "window_index": 1,
        "timestamp": [1, 1, 1],
        "inplane_sensor_id": "ST-VIC-C18-102-01",
    }
    payload = build_wind_render_payload(record, cfg_with_wind)
    assert payload.avg_wind_speed > 0
    assert payload.avg_turbulence >= 0
    speed_png = plot_wind_speed_rose(payload)
    ti_png = plot_wind_turbulence_rose(payload)
    assert len(speed_png) > 100
    assert len(ti_png) > 100
    passed += 1
    print("[ok] 风玫瑰图渲染与平均值标注")

    print(f"\nAugment feature regression: {passed}/{passed} passed")


if __name__ == "__main__":
    run_feature_regression_tests()
