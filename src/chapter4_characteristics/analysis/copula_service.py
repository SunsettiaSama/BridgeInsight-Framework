from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.chapter4_characteristics.settings import get_copula_dir, load_config
from src.chapter4_characteristics.statistics.mode_extract import (
    DEFAULT_N_MODES,
    build_var_names,
    load_modes,
    matrix_from_arrays,
    modes_cache_path,
)
from src.chapter4_characteristics.statistics.pipeline import (
    fit_class_joint,
    fit_class_marginals,
    run_extract,
    run_full_pipeline,
    run_joint,
    run_marginals,
)

logger = logging.getLogger(__name__)


def load_mode_matrix_from_cache(
    cfg: dict,
    class_id: int,
) -> tuple[np.ndarray, List[str]]:
    n_modes = int(cfg.get("copula_n_modes", DEFAULT_N_MODES))
    nfft = int(cfg.get("copula_nfft", 128))
    path = modes_cache_path(cfg, class_id, n_modes=n_modes, nfft=nfft)
    freq_in, energy_in, freq_out, energy_out, _ = load_modes(path)
    matrix = matrix_from_arrays(freq_in, energy_in, freq_out, energy_out)
    return matrix, build_var_names(n_modes)


def load_mode_matrix(samples: List[dict], n_modes: int, max_samples: int) -> tuple[np.ndarray, List[str]]:
    """兼容旧 WebUI：优先读缓存；无缓存时抛错提示先 extract。"""
    raise RuntimeError(
        "已改为从原始 VIC 重算模态缓存。"
        "请先运行：python -m src.chapter4_characteristics copula extract --class-id all"
    )


def load_copula_result(cfg: dict, class_id: int) -> Optional[dict]:
    path = get_copula_dir(cfg) / f"class_{class_id}_copula.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_copula_analysis(class_id: int, cfg: dict) -> dict:
    """单类完整流水线：extract → marginals → joint。"""
    config_path = cfg.get("_config_path")
    run_extract(class_id=class_id, config_path=config_path, refresh=False)
    fit_class_marginals(class_id, config_path=config_path)
    return fit_class_joint(class_id, config_path=config_path)


def run_copula_job(
    class_id: int | str = "all",
    config_path: str | None = None,
    stage: str = "run",
    refresh: bool = False,
    max_samples: int | None = None,
) -> dict:
    """
    CLI / WebUI 入口。

    stage:
      extract | marginals | joint | run
    """
    logger.info(f"Copula stage={stage} class_id={class_id} refresh={refresh}")
    if stage == "extract":
        paths = run_extract(
            class_id=class_id,
            config_path=config_path,
            refresh=refresh,
            max_samples=max_samples,
        )
        return {"stage": stage, "paths": [str(p) for p in paths]}
    if stage == "marginals":
        results = run_marginals(class_id=class_id, config_path=config_path)
        return {
            "stage": stage,
            "results": [{"class_id": r["class_id"], "n_samples": r["n_samples"]} for r in results],
        }
    if stage == "joint":
        results = run_joint(class_id=class_id, config_path=config_path)
        return {
            "stage": stage,
            "results": [
                {
                    "class_id": r["class_id"],
                    "best_copula_type": r["best_copula_type"],
                    "n_samples": r["n_samples"],
                }
                for r in results
            ],
        }
    if stage == "run":
        return run_full_pipeline(
            class_id=class_id,
            config_path=config_path,
            refresh=refresh,
            max_samples=max_samples,
        )
    raise ValueError(f"未知 stage: {stage}")
