from __future__ import annotations

from typing import Any

from src.chapter3_identifier.regression_forecast.settings import get_round_forecast_path, read_json


def build_infer_monitor_payload(cfg: dict[str, Any], round_idx: int, job_state: dict, log_tail: str) -> dict[str, Any]:
    path = get_round_forecast_path(cfg, round_idx)
    payload = read_json(path) if path.exists() else {"records": []}
    records = payload.get("records", [])
    class_names = [str(x) for x in cfg.get("class_names", [])]
    distribution = {name: 0 for name in class_names}
    high_risk = []
    for record in records:
        best_score = 0.0
        best_class = class_names[0] if class_names else "Normal"
        for horizon in record.get("horizons", []):
            score = float(horizon.get("risk_score", 0.0))
            if score >= best_score:
                best_score = score
                best_class = str(horizon.get("risk_class", best_class))
        distribution[best_class] = distribution.get(best_class, 0) + 1
        if best_score > 0.5:
            high_risk.append({"sample_idx": record.get("sample_idx"), "risk_class": best_class, "risk_score": best_score})
    high_risk.sort(key=lambda row: float(row["risk_score"]), reverse=True)
    return {
        "round_idx": int(round_idx),
        "job": job_state,
        "forecast_ready": path.exists(),
        "record_count": len(records),
        "risk_distribution": distribution,
        "high_risk_samples": high_risk[:20],
        "log_tail": log_tail,
    }

