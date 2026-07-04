from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.chapter3_identifier.early_warning.figures.service import FigureService
from src.chapter3_identifier.early_warning.warning.policy import WarningPolicy
from src.chapter3_identifier.regression_forecast.settings import get_round_forecast_path, read_json
from src.chapter3_identifier.regression_forecast.webui.jobs.manager import JobManager
from src.chapter3_identifier.early_warning.settings import resolve_python_executable


class ForecastCache:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self._payload_by_path: dict[str, tuple[float, dict[str, Any]]] = {}

    def _mtime(self, path: Path) -> float:
        if not path.exists():
            return -1.0
        return float(path.stat().st_mtime)

    def payload(self, round_idx: int) -> dict[str, Any]:
        path = get_round_forecast_path(self.cfg, round_idx)
        key = str(path)
        mtime = self._mtime(path)
        cached = self._payload_by_path.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        if path.exists():
            payload = read_json(path)
        else:
            payload = {"records": [], "record_count": 0}
        self._payload_by_path[key] = (mtime, payload)
        return payload

    def invalidate(self) -> None:
        self._payload_by_path.clear()

    def records(self, round_idx: int) -> list[dict[str, Any]]:
        return list(self.payload(round_idx).get("records", []))

    def find_record(self, round_idx: int, sample_idx: int) -> dict[str, Any] | None:
        for record in self.records(round_idx):
            if int(record.get("sample_idx", -1)) == int(sample_idx):
                return record
        return None


@dataclass
class AppDeps:
    cfg: dict[str, Any]
    config_path: Optional[str]
    forecasts: ForecastCache
    figures: FigureService
    jobs: JobManager
    policy: WarningPolicy

    def forecast_path(self, round_idx: int) -> str:
        return str(get_round_forecast_path(self.cfg, round_idx))

    def data_status(self, round_idx: int) -> dict[str, Any]:
        path = Path(self.forecast_path(round_idx))
        exists = path.exists()
        return {
            "forecast_exists": exists,
            "forecast_path": str(path),
            "record_count": len(self.forecasts.records(round_idx)) if exists else 0,
            "feature_cache_mode": str(self.cfg.get("feature_cache_mode", "real")),
        }


def build_deps(cfg: dict[str, Any], config_path: Optional[str] = None) -> AppDeps:
    jobs = JobManager(
        cfg["job_state_path"],
        python_executable=resolve_python_executable(cfg),
        module_name="src.chapter3_identifier.early_warning",
    )
    return AppDeps(
        cfg=cfg,
        config_path=config_path,
        forecasts=ForecastCache(cfg),
        figures=FigureService(),
        jobs=jobs,
        policy=WarningPolicy.from_config(cfg),
    )
