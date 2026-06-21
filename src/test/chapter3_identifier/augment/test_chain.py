from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _png_ok(content: bytes) -> bool:
    return len(content) > 200 and content[:8] == b"\x89PNG\r\n\x1a\n"


def run_chain_tests(port: int = 8766) -> None:
    from src.chapter3_identifier.augment._bootstrap import ensure_paths
    from src.chapter3_identifier.augment.features.wind_index import (
        get_wind_index,
        resolve_wind_meta,
        resolve_wind_sensor_id,
    )
    from src.chapter3_identifier.augment.figures.render.wind import build_wind_render_payload
    from src.chapter3_identifier.augment.figures.types import SAMPLE_FIGURE_NAMES
    from src.chapter3_identifier.augment.settings import get_round_inference_snapshot_path, load_config
    from fastapi import FastAPI
    from src.chapter3_identifier.augment.webui.deps import build_deps
    from src.chapter3_identifier.augment.webui.routes.figures import build_figures_router
    from src.chapter3_identifier.augment.webui.routes.queue import build_queue_router
    from starlette.testclient import TestClient

    ensure_paths()
    cfg = load_config()
    base = f"http://{cfg.get('webui_host', '127.0.0.1')}:{port}"
    round_idx = 2
    inference_path = get_round_inference_snapshot_path(cfg, round_idx)
    payload = json.loads(inference_path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    record = next(r for r in records if int(r.get("prediction", 0)) in (1, 2, 3))
    sample_idx = int(record["sample_idx"])

    passed = 0
    print(f"[info] test sample={sample_idx} timestamp={record.get('timestamp')} window={record.get('window_index')}")

    assert "wind_speed_rose" in SAMPLE_FIGURE_NAMES
    assert "wind_turbulence_rose" in SAMPLE_FIGURE_NAMES
    passed += 1
    print("[ok] 风图类型已注册")

    wind_index = get_wind_index(cfg)
    assert wind_index is not None, "wind_metadata_path 未加载"
    wind_meta = resolve_wind_meta(record, cfg)
    assert wind_meta is not None, f"样本 {sample_idx} 无风元数据对齐"
    assert Path(str(wind_meta["file_path"])).exists(), "风数据文件不存在"
    expected_sensor = resolve_wind_sensor_id(record)
    assert wind_meta.get("sensor_id") == expected_sensor, (
        f"风传感器应为跨中映射 {expected_sensor}，实际 {wind_meta.get('sensor_id')}"
    )
    passed += 1
    print(f"[ok] 风元数据对齐 -> {wind_meta.get('sensor_id')}")

    wind_payload = build_wind_render_payload(record, cfg)
    assert wind_payload.avg_wind_speed > 0
    passed += 1
    print(f"[ok] 风载荷构建 avg_speed={wind_payload.avg_wind_speed:.2f} avg_ti={wind_payload.avg_turbulence:.2f}")

    deps = build_deps(cfg)
    deps.figures.start()
    app = FastAPI()
    app.include_router(build_figures_router(deps))
    app.include_router(build_queue_router(deps))
    with TestClient(app) as client:
        preload = client.post(
            "/api/preload",
            json={
                "sample_indices": [sample_idx],
                "round_idx": round_idx,
                "priority": True,
                "both_directions": True,
                "layout_profile": "wide_fill_v1",
            },
        )
        assert preload.status_code == 200, preload.text

        q = client.get(f"/api/queue/abnormal?round_idx={round_idx}&pred_classes=1,2,3&page=0&page_size=5")
        assert q.status_code == 200, q.text
        items = q.json().get("items", [])
        assert items, "异常队列为空"
        passed += 1
        print(f"[ok] 队列 API items={len(items)}")

        status = client.get(
            f"/api/figures/{sample_idx}/_bundle_status?round_idx={round_idx}&layout_profile=wide_fill_v1&wait_ms=20000"
        )
        assert status.status_code == 200, status.text
        bundle = status.json()
        wind_stats = bundle.get("wind_stats") or {}
        assert wind_stats.get("ready"), f"wind_stats 未就绪: {wind_stats}"
        assert wind_stats.get("avg_wind_speed", 0) > 0
        assert wind_stats.get("avg_turbulence", -1) >= 0
        passed += 1
        print(
            f"[ok] wind_stats avg_speed={wind_stats['avg_wind_speed']:.2f} "
            f"avg_ti={wind_stats['avg_turbulence']:.2f}%"
        )

        wait_ms = 20000
        for fig in ("wind_speed_rose", "wind_turbulence_rose", "in_timeseries", "context/timeseries"):
            if fig.startswith("context/"):
                url = f"/api/figures/{sample_idx}/{fig}?round_idx={round_idx}&direction=inplane&wait_ms={wait_ms}"
            else:
                url = f"/api/figures/{sample_idx}/{fig}?round_idx={round_idx}&layout_profile=wide_fill_v1&wait_ms={wait_ms}"
            resp = client.get(url)
            assert resp.status_code == 200, f"{fig} -> {resp.status_code} {resp.text[:200]}"
            assert _png_ok(resp.content), f"{fig} 非 PNG"
            passed += 1
            print(f"[ok] ASGI {fig} bytes={len(resp.content)}")

        bundle = client.get(
            f"/api/figures/{sample_idx}/_bundle_status?round_idx={round_idx}&wait_ms=5000"
        )
        assert bundle.status_code == 200
        body = bundle.json()
        assert body.get("sample_ready") is True
        passed += 1
        print(f"[ok] bundle_status sample_ready={body.get('sample_ready')}")

    import httpx

    try:
        with httpx.Client(base_url=base, timeout=120.0) as live:
            health = live.get("/")
            if health.status_code != 200:
                print(f"[skip] 实端口 {port} 未启动 (status={health.status_code})")
            else:
                for fig in ("wind_speed_rose", "wind_turbulence_rose"):
                    url = f"/api/figures/{sample_idx}/{fig}?round_idx={round_idx}&layout_profile=wide_fill_v1&wait_ms=8000"
                    t0 = time.perf_counter()
                    resp = live.get(url)
                    dt = time.perf_counter() - t0
                    assert resp.status_code == 200, f"live {fig} {resp.status_code}"
                    assert _png_ok(resp.content)
                    passed += 1
                    print(f"[ok] live:{port} {fig} bytes={len(resp.content)} dt={dt:.2f}s")
    except httpx.ConnectError:
        print(f"[skip] 实端口 {port} 未启动，仅完成 ASGI 链路测试")

    print(f"\nChain test passed: {passed} checks")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    run_chain_tests(port=port)
