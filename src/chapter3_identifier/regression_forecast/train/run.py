from __future__ import annotations

import random
from typing import Any

import torch

from src.chapter3_identifier.regression_forecast.datasets.forecast_dataset import load_forecast_dataset, subset
from src.chapter3_identifier.regression_forecast.models.multitask import MultiTaskForecastModel
from src.chapter3_identifier.regression_forecast.settings import (
    ensure_round_dir,
    get_round_checkpoint_path,
    get_round_live_metrics_path,
    get_round_metrics_path,
    load_config,
    write_json,
)
from src.chapter3_identifier.regression_forecast.train.trainer import ForecastTrainer, make_loader


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _make_model(schema: dict[str, Any], cfg: dict[str, Any]) -> MultiTaskForecastModel:
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


def run_training(round_idx: int = 1, config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    ensure_round_dir(cfg, round_idx)
    train_cfg = cfg.get("training", {})
    _set_seed(int(train_cfg.get("random_seed", 42)))

    dataset, splits, schema = load_forecast_dataset(round_idx=round_idx, config_path=config_path)
    train_indices = splits.get("train", [])
    val_indices = splits.get("val", [])
    if not train_indices:
        raise ValueError("训练 split 为空，请检查样本数、horizon 和 split_policy")
    if not val_indices:
        val_indices = train_indices

    batch_size = int(train_cfg.get("batch_size", 32))
    train_loader = make_loader(subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = make_loader(subset(dataset, val_indices), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(schema, cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    trainer = ForecastTrainer(model, cfg, device)

    epochs = int(train_cfg.get("epochs", 3))
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        train_metrics = trainer.run_epoch(train_loader, optimizer=optimizer)
        val_metrics = trainer.run_epoch(val_loader, optimizer=None)
        row = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "split_sizes": {name: len(values) for name, values in splits.items()},
        }
        history.append(row)
        write_json(get_round_live_metrics_path(cfg, round_idx), row)
        val_loss = float(val_metrics.get("loss", 0.0))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "schema": schema,
        "config": cfg,
        "round_idx": int(round_idx),
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, get_round_checkpoint_path(cfg, round_idx))
    payload = {
        "round_idx": int(round_idx),
        "history": history,
        "best_val_loss": best_val_loss,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "checkpoint": str(get_round_checkpoint_path(cfg, round_idx)),
    }
    write_json(get_round_metrics_path(cfg, round_idx), payload)
    return payload


if __name__ == "__main__":
    run_training()

