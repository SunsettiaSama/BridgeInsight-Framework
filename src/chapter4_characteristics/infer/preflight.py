from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml

from src.chapter4_characteristics._bootstrap import ensure_paths, resolve_path
from src.chapter4_characteristics.settings import get_augment_checkpoint, load_config

ensure_paths()
logger = logging.getLogger(__name__)


def run_preflight(round_idx: int = 1, config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    issues: list[str] = []
    ok_items: list[str] = []

    ds_path = resolve_path(cfg["inference_dataset_config"])
    if not ds_path.exists():
        issues.append(f"数据集配置不存在：{ds_path}")
    else:
        ok_items.append(f"dataset config: {ds_path.name}")
        with open(ds_path, "r", encoding="utf-8") as f:
            ds_raw = yaml.safe_load(f) or {}
        vib_meta = resolve_path(ds_raw["vib_metadata_path"])
        if not vib_meta.exists():
            issues.append(f"振动元数据不存在：{vib_meta}")
        else:
            ok_items.append(f"vib metadata: {vib_meta.name}")

    ckpt = get_augment_checkpoint(cfg, round_idx)
    if not ckpt.exists():
        issues.append(f"checkpoint 不存在：{ckpt}")
    else:
        ok_items.append(f"checkpoint: {ckpt}")

    wind_path = resolve_path(cfg["wind_metadata_path"])
    if not wind_path.exists():
        issues.append(f"风元数据不存在（enrich 需要）：{wind_path}")
    else:
        ok_items.append(f"wind metadata: {wind_path.name}")

    cuda_ok = torch.cuda.is_available()
    ok_items.append(f"cuda: {cuda_ok}")

    result = {
        "round_idx": round_idx,
        "ok": len(issues) == 0,
        "issues": issues,
        "checks": ok_items,
    }
    for item in ok_items:
        logger.info(f"  OK  {item}")
    for item in issues:
        logger.error(f"  FAIL  {item}")
    if issues:
        raise RuntimeError("预检未通过：" + "; ".join(issues))
    return result
