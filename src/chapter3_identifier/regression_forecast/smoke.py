from __future__ import annotations

from src.chapter3_identifier.regression_forecast.features.cache import build_feature_cache, check_data
from src.chapter3_identifier.regression_forecast.infer.run import run_inference
from src.chapter3_identifier.regression_forecast.train.run import run_training
from src.chapter3_identifier.regression_forecast.webui.app import create_app


def run_smoke(config_path: str | None = None) -> dict:
    round_idx = 1
    data = check_data(round_idx=round_idx, config_path=config_path)
    cache_path = build_feature_cache(round_idx=round_idx, config_path=config_path)
    train_payload = run_training(round_idx=round_idx, config_path=config_path)
    forecast_path = run_inference(round_idx=round_idx, config_path=config_path)
    app = create_app(config_path)
    result = {
        "data": data,
        "cache_path": cache_path,
        "best_val_loss": train_payload.get("best_val_loss"),
        "forecast_path": forecast_path,
        "routes": len(app.routes),
    }
    print(result)
    return result

