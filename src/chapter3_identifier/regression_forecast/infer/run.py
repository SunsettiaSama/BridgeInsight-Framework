from __future__ import annotations

from datetime import datetime

import torch

from src.chapter3_identifier.regression_forecast.datasets.forecast_dataset import load_forecast_dataset
from src.chapter3_identifier.regression_forecast.infer.runner import ForecastRunner
from src.chapter3_identifier.regression_forecast.models.multitask import MultiTaskForecastModel
from src.chapter3_identifier.regression_forecast.settings import (
    get_round_checkpoint_path,
    get_round_forecast_path,
    load_config,
    write_json,
)


def _make_model(schema: dict, cfg: dict) -> MultiTaskForecastModel:
    model_cfg = cfg.get("model", {})
    return MultiTaskForecastModel(
        input_dim=len(schema.get("input_feature_names", [])),
        horizon_count=len(schema.get("horizons_hours", [])),
        class_count=len(schema.get("class_names", [])),
        metric_count=len(schema.get("metric_names", [])),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def run_inference(round_idx: int = 1, config_path: str | None = None) -> str:
    cfg = load_config(config_path)
    ckpt_path = get_round_checkpoint_path(cfg, round_idx)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}，请先训练")
    dataset, splits, schema = load_forecast_dataset(round_idx=round_idx, config_path=config_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    schema = checkpoint.get("schema", schema)
    model = _make_model(schema, cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = ForecastRunner(model, schema, device)
    records = runner.run(dataset, batch_size=int(cfg.get("training", {}).get("batch_size", 32)))
    payload = {
        "round_idx": int(round_idx),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": schema.get("schema_version", "regression_forecast_v1"),
        "horizons_hours": schema.get("horizons_hours", []),
        "class_names": schema.get("class_names", []),
        "metric_names": schema.get("metric_names", []),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "record_count": len(records),
        "records": records,
    }
    forecast_path = get_round_forecast_path(cfg, round_idx)
    write_json(forecast_path, payload)
    return str(forecast_path)


if __name__ == "__main__":
    run_inference()

