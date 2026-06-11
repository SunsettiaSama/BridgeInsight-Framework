from __future__ import annotations

from types import SimpleNamespace

import yaml

from src.chapter3_identifier.augment._bootstrap import ensure_paths, resolve_path
from src.chapter3_identifier.augment.settings import check_inference_metadata, load_config, load_yaml
from src.data_processer.datasets.StayCable_Vib2023.StayCableVib2023Dataset import StayCableVib2023Dataset

ensure_paths()


_DATASET_DEFAULTS = {
    "wind_metadata_path": None,
    "wind_sensor_ids": None,
    "require_wind_alignment": False,
    "enable_denoise": False,
    "denoise_freq_threshold": None,
    "missing_rate_threshold": 0.05,
    "time_ordered": True,
    "split_ratio": -1,
    "split_by_time": False,
    "split_seed": 42,
    "use_cache": True,
    "cache_path": None,
    "predictions_cache_path": None,
}


def load_staycable_dataset(config_path: str):
    cfg_path = resolve_path(config_path)
    raw = {**_DATASET_DEFAULTS, **load_yaml(cfg_path)}

    raw["vib_metadata_path"] = str(resolve_path(raw["vib_metadata_path"]))

    if raw.get("wind_metadata_path"):
        raw["wind_metadata_path"] = str(resolve_path(raw["wind_metadata_path"]))
    if raw.get("cache_path"):
        raw["cache_path"] = str(resolve_path(raw["cache_path"]))
    if raw.get("predictions_cache_path"):
        raw["predictions_cache_path"] = str(resolve_path(raw["predictions_cache_path"]))

    config = SimpleNamespace(**raw)
    return StayCableVib2023Dataset(config=config)


def ensure_inference_ready(augment_cfg: dict) -> None:
    check_inference_metadata(augment_cfg)
