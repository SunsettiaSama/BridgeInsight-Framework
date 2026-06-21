from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.chapter4_characteristics.analysis.data_loader import load_class_samples
from src.chapter4_characteristics.settings import CLASS_DIRS, get_copula_dir, load_config
from src.chapter4_characteristics.statistics.multivariate import compare_copulas, fit_multivariate

logger = logging.getLogger(__name__)


@dataclass
class CopulaAnalysisConfig:
    n_modes: int = 8
    max_samples: int = 5000
    copula_types: List[str] = field(default_factory=lambda: ["gaussian", "t", "gumbel", "clayton", "frank"])
    marginal_distributions: List[str] = field(default_factory=lambda: ["gamma"])


def _build_var_names(n_modes: int) -> List[str]:
    names: List[str] = []
    for k in range(1, n_modes + 1):
        names.append(f"freq_in_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"energy_in_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"freq_out_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"energy_out_{k}")
    return names


def _extract_row(sample: dict, n_modes: int) -> Optional[np.ndarray]:
    psd_in = sample.get("psd_inplane") or {}
    psd_out = sample.get("psd_outplane") or {}
    fi = psd_in.get("frequencies") or []
    pi = psd_in.get("powers") or []
    fo = psd_out.get("frequencies") or []
    po = psd_out.get("powers") or []
    if not fi or not fo:
        return None

    def top_n(freqs, powers, n):
        f = np.asarray(freqs, dtype=np.float64)
        p = np.asarray(powers, dtype=np.float64)
        order = np.argsort(p)[::-1]
        f_out = np.full(n, np.nan)
        e_out = np.full(n, np.nan)
        total = p[order].sum()
        if total <= 0:
            return f_out, e_out
        m = min(n, len(order))
        f_out[:m] = f[order[:m]]
        e_out[:m] = p[order[:m]] / total
        return f_out, e_out

    f_in, e_in = top_n(fi, pi, n_modes)
    f_out, e_out = top_n(fo, po, n_modes)
    return np.concatenate([f_in, e_in, f_out, e_out])


def load_mode_matrix(samples: List[dict], n_modes: int, max_samples: int) -> tuple[np.ndarray, List[str]]:
    rows = []
    for s in samples:
        row = _extract_row(s, n_modes)
        if row is not None and np.all(np.isfinite(row[row != 0])):
            rows.append(row)
    if not rows:
        raise ValueError("无有效 PSD 模态样本")
    mat = np.asarray(rows, dtype=np.float64)
    if mat.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(mat.shape[0], size=max_samples, replace=False)
        mat = mat[idx]
    names = _build_var_names(n_modes)
    return mat, names


def run_copula_analysis(class_id: int, cfg: dict) -> dict:
    ccfg = CopulaAnalysisConfig(
        n_modes=int(cfg.get("copula_n_modes", 8)),
        max_samples=int(cfg.get("copula_max_samples", 5000)),
    )
    samples = load_class_samples(class_id, cfg)
    matrix, var_names = load_mode_matrix(samples, ccfg.n_modes, ccfg.max_samples)

    u_matrix = _pit_matrix(matrix, var_names, ccfg)
    comparison = compare_copulas(u_matrix, copula_types=ccfg.copula_types)
    best_type = comparison[0].copula_type if comparison else "gaussian"
    mv = fit_multivariate(
        matrix,
        variable_names=var_names,
        marginal_distributions=ccfg.marginal_distributions,
        copula_type=best_type,
    )

    payload = {
        "class_id": class_id,
        "class_label": CLASS_DIRS[class_id],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(matrix.shape[0]),
        "n_vars": len(var_names),
        "variable_names": var_names,
        "best_copula_type": best_type,
        "comparison": [
            {
                "copula_type": r.copula_type,
                "aic": r.aic,
                "bic": r.bic,
                "log_likelihood": r.log_likelihood,
            }
            for r in comparison
        ],
        "copula_params": mv.copula.params,
        "marginals_summary": {
            name: {"form": r.form, "aic": r.aic, "params": r.params}
            for name, r in mv.marginals.items()
        },
    }
    out_dir = get_copula_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"class_{class_id}_copula.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _pit_matrix(matrix: np.ndarray, var_names: List[str], ccfg: CopulaAnalysisConfig) -> np.ndarray:
    from src.chapter4_characteristics.statistics.multivariate import pit_transform
    from src.chapter4_characteristics.statistics.fitting import fit_distribution

    marginals = {}
    for j, name in enumerate(var_names):
        col = matrix[:, j]
        valid = col[np.isfinite(col) & (col > 0)]
        marginals[name] = fit_distribution(valid, distribution="gamma", floc=0)
    return pit_transform(matrix, marginals, var_names)


def run_copula_job(class_id: int = 0, config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    logger.info(f"Copula 拟合 class={class_id}")
    result = run_copula_analysis(class_id, cfg)
    logger.info(f"Copula 拟合完成：best={result['best_copula_type']}")
    return result
