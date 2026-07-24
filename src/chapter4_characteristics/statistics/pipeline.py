from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from src.chapter4_characteristics.settings import (
    CLASS_DIRS,
    CLASS_LABELS,
    DEFAULT_LABEL_NAMES,
    get_copula_dir,
    load_config,
)
from src.chapter4_characteristics.statistics.config import load_config as load_stats_config
from src.chapter4_characteristics.statistics.copula import fit_copula
from src.chapter4_characteristics.statistics.fitting import FitResult, fit_distribution, fit_gmm
from src.chapter4_characteristics.statistics.mode_extract import (
    DEFAULT_N_MODES,
    build_var_names,
    extract_all_classes,
    load_modes,
    matrix_from_arrays,
    modes_cache_path,
)
from src.chapter4_characteristics.statistics.multivariate import compare_copulas

_DIST_FIT_KWARGS: dict = {
    "gamma": {"floc": 0},
    "lognorm": {"floc": 0},
    "expon": {"floc": 0},
    "norm": {},
}

ELLIPTICAL_COPULAS = ["gaussian", "t"]


def _parse_class_ids(class_id: str | int) -> list[int]:
    if isinstance(class_id, int):
        return [class_id]
    text = str(class_id).strip().lower()
    if text == "all":
        return [0, 1, 2, 3]
    return [int(text)]


def _fitresult_to_dict(r: FitResult, n_valid: int) -> dict:
    return {
        "form": r.form,
        "params": r.params,
        "ks_statistic": r.ks_statistic,
        "ks_pvalue": r.ks_pvalue,
        "aic": r.aic,
        "bic": r.bic,
        "n_valid": n_valid,
    }


def _fit_best(
    col: np.ndarray,
    candidates: list[str],
    min_valid: int,
    enable_gmm: bool,
    gmm_max_components: int,
) -> Optional[dict]:
    valid = col[np.isfinite(col) & (col > 0)]
    n_valid = int(len(valid))
    if n_valid < min_valid or float(np.std(valid)) < 1e-12:
        return None

    results: list[FitResult] = []
    for dist_name in candidates:
        kw = _DIST_FIT_KWARGS.get(dist_name, {})
        results.append(fit_distribution(valid, distribution=dist_name, **kw))

    if enable_gmm:
        for n_comp in range(2, gmm_max_components + 1):
            if n_valid >= n_comp * 5:
                results.append(fit_gmm(valid, n_components=n_comp))

    best = min(results, key=lambda r: r.aic if r.aic is not None else float("inf"))
    return {
        "n_valid": n_valid,
        "best": _fitresult_to_dict(best, n_valid),
        "candidates": [_fitresult_to_dict(r, n_valid) for r in results],
    }


def _load_class_matrix(cfg: dict, class_id: int) -> tuple[np.ndarray, list[str], dict]:
    n_modes = int(cfg.get("copula_n_modes", DEFAULT_N_MODES))
    nfft = int(cfg.get("copula_nfft", 128))
    path = modes_cache_path(cfg, class_id, n_modes=n_modes, nfft=nfft)
    freq_in, energy_in, freq_out, energy_out, mode_cfg = load_modes(path)
    matrix = matrix_from_arrays(freq_in, energy_in, freq_out, energy_out)
    return matrix, build_var_names(n_modes), mode_cfg


def marginals_path(cfg: dict, class_id: int) -> Path:
    return get_copula_dir(cfg) / f"class_{class_id}_marginals.json"


def copula_result_path(cfg: dict, class_id: int) -> Path:
    return get_copula_dir(cfg) / f"class_{class_id}_copula.json"


def fit_class_marginals(
    class_id: int,
    config_path: str | None = None,
    stats_yaml: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    scfg = load_stats_config(stats_yaml)
    n_modes = int(cfg.get("copula_n_modes", DEFAULT_N_MODES))
    scfg.n_modes = n_modes

    matrix, var_names, mode_cfg = _load_class_matrix(cfg, class_id)
    print("=" * 80)
    print(
        f"边缘分布拟合 class={class_id} ({CLASS_LABELS.get(class_id, '?')}) "
        f"n={matrix.shape[0]} d={matrix.shape[1]}"
    )
    print("=" * 80)

    freq_set = set(range(0, n_modes)) | set(range(2 * n_modes, 3 * n_modes))
    energy_set = set(range(n_modes, 2 * n_modes)) | set(range(3 * n_modes, 4 * n_modes))

    marginals: dict = {}
    for j, name in enumerate(var_names):
        if j in freq_set:
            candidates = scfg.candidate_dists_freq
        elif j in energy_set:
            candidates = scfg.candidate_dists_energy
        else:
            candidates = scfg.candidate_dists_freq

        result = _fit_best(
            matrix[:, j],
            candidates,
            scfg.min_valid_samples,
            enable_gmm=scfg.enable_gmm,
            gmm_max_components=scfg.gmm_max_components,
        )
        if result is None:
            print(f"  [{j+1:2d}/{len(var_names)}] {name:<20} 跳过")
        else:
            best = result["best"]
            ks_p = best.get("ks_pvalue")
            ks_str = f"{ks_p:.4f}" if ks_p is not None else "N/A"
            print(
                f"  [{j+1:2d}/{len(var_names)}] {name:<20} → {best['form']:<12} "
                f"AIC={best['aic']:10.2f}  KS-p={ks_str:<8}  n={result['n_valid']}"
            )
        marginals[name] = result

    payload = {
        "class_id": class_id,
        "class_label": CLASS_DIRS[class_id],
        "class_name": DEFAULT_LABEL_NAMES[class_id] if class_id < len(DEFAULT_LABEL_NAMES) else CLASS_LABELS[class_id],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(matrix.shape[0]),
        "n_modes": n_modes,
        "n_vars": len(var_names),
        "variable_names": var_names,
        "mode_config": mode_cfg,
        "stats_config": scfg.to_dict(),
        "marginals": marginals,
    }
    out_path = marginals_path(cfg, class_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  已保存：{out_path}")
    return payload


def _gmm_cdf(x: np.ndarray, params: dict) -> np.ndarray:
    weights = np.asarray(params["weights"], dtype=np.float64)
    means = np.asarray(params["means"], dtype=np.float64)
    variances = np.asarray(params["variances"], dtype=np.float64)
    cdf = np.zeros_like(x, dtype=np.float64)
    for w, m, v in zip(weights, means, variances):
        cdf += w * stats.norm.cdf(x, loc=m, scale=np.sqrt(max(v, 1e-30)))
    return cdf


def _parametric_pit_column(col: np.ndarray, best: dict) -> np.ndarray:
    form = best["form"]
    params = best["params"]
    valid_mask = np.isfinite(col)
    u = np.full(col.shape[0], np.nan, dtype=np.float64)
    x = col[valid_mask]
    if form.startswith("gmm_"):
        u[valid_mask] = _gmm_cdf(x, params)
    else:
        dist_obj = getattr(stats, form)
        u[valid_mask] = dist_obj.cdf(x, *list(params.values()))
    return np.clip(u, 1e-10, 1.0 - 1e-10)


def _empirical_pit(matrix: np.ndarray) -> np.ndarray:
    u = np.empty_like(matrix, dtype=np.float64)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        ranks = stats.rankdata(col, method="average")
        u[:, j] = ranks / (matrix.shape[0] + 1.0)
    return np.clip(u, 1e-10, 1.0 - 1e-10)


def _load_marginals_json(cfg: dict, class_id: int) -> dict:
    path = marginals_path(cfg, class_id)
    if not path.exists():
        raise FileNotFoundError(f"边缘拟合结果不存在：{path}；请先运行 copula marginals")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fit_class_joint(
    class_id: int,
    config_path: str | None = None,
    pit_mode: str = "empirical",
) -> dict:
    """椭圆 Copula 联合拟合（gaussian / t）。"""
    cfg = load_config(config_path)
    matrix, var_names, mode_cfg = _load_class_matrix(cfg, class_id)
    marg_payload = _load_marginals_json(cfg, class_id)

    complete = np.all(np.isfinite(matrix) & (matrix > 0), axis=1)
    mat = matrix[complete]
    if mat.shape[0] < 30:
        raise ValueError(f"完整样本不足：{mat.shape[0]}")

    max_n = int(cfg.get("copula_joint_max_samples", cfg.get("copula_max_samples", 5000)))
    if mat.shape[0] > max_n:
        rng = np.random.default_rng(int(cfg.get("copula_rng_seed", 42)))
        idx = rng.choice(mat.shape[0], size=max_n, replace=False)
        mat = mat[idx]
        print(f"  联合拟合抽样 {max_n} / {int(complete.sum())}")

    print("=" * 80)
    print(
        f"椭圆 Copula 联合拟合 class={class_id} ({CLASS_LABELS.get(class_id, '?')}) "
        f"n={mat.shape[0]} d={mat.shape[1]}"
    )
    print("=" * 80)

    if pit_mode == "parametric":
        u_matrix = np.empty_like(mat, dtype=np.float64)
        for j, name in enumerate(var_names):
            entry = marg_payload["marginals"].get(name)
            if entry is None or entry.get("best") is None:
                raise ValueError(f"缺少边缘拟合：{name}")
            u_matrix[:, j] = _parametric_pit_column(mat[:, j], entry["best"])
    else:
        u_matrix = _empirical_pit(mat)

    print("  比较椭圆 Copula：gaussian / t ...")
    comparison = compare_copulas(u_matrix, copula_types=ELLIPTICAL_COPULAS)
    best = comparison[0]
    print(f"  最优：{best.copula_type}  AIC={best.aic:.4f}")

    best_full = fit_copula(u_matrix, copula_type=best.copula_type)

    pearson = np.corrcoef(mat.T)
    rho_raw, _ = stats.spearmanr(mat)
    if mat.shape[1] == 2:
        rho = float(rho_raw)
        spearman = np.array([[1.0, rho], [rho, 1.0]])
    else:
        spearman = np.asarray(rho_raw, dtype=float)
        np.fill_diagonal(spearman, 1.0)

    n_modes_local = int(mode_cfg.get("n_modes", DEFAULT_N_MODES))
    payload = {
        "class_id": class_id,
        "class_label": CLASS_DIRS[class_id],
        "class_name": DEFAULT_LABEL_NAMES[class_id] if class_id < len(DEFAULT_LABEL_NAMES) else CLASS_LABELS[class_id],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(mat.shape[0]),
        "n_vars": len(var_names),
        "n_modes": n_modes_local,
        "variable_names": var_names,
        "mode_config": mode_cfg,
        "pit_mode": pit_mode,
        "best_copula_type": best_full.copula_type,
        "comparison": [
            {
                "copula_type": r.copula_type,
                "aic": r.aic,
                "bic": r.bic,
                "log_likelihood": r.log_likelihood,
                "n_samples": r.n_samples,
                "n_vars": r.n_vars,
            }
            for r in comparison
        ],
        "copula_params": best_full.params,
        "correlation": {
            "pearson": pearson.tolist(),
            "spearman": spearman.tolist(),
        },
        "marginals_summary": {
            name: {
                "form": (entry or {}).get("best", {}).get("form"),
                "aic": (entry or {}).get("best", {}).get("aic"),
                "params": (entry or {}).get("best", {}).get("params"),
            }
            for name, entry in marg_payload.get("marginals", {}).items()
        },
        "u_matrix_preview": {
            "n_preview": min(200, u_matrix.shape[0]),
            "pairs": [
                [0, 1],
                [0, n_modes_local],
                [n_modes_local, 2 * n_modes_local],
            ],
        },
    }

    # 存一小段 PIT 供探索图（完整矩阵太大）
    preview_n = min(2000, u_matrix.shape[0])
    payload["u_matrix_sample"] = u_matrix[:preview_n].tolist()

    out_path = copula_result_path(cfg, class_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  已保存：{out_path}")
    return payload


def run_extract(
    class_id: str | int = "all",
    config_path: str | None = None,
    refresh: bool = False,
    max_samples: int | None = None,
) -> list[Path]:
    ids = _parse_class_ids(class_id)
    return extract_all_classes(
        config_path=config_path,
        refresh=refresh,
        class_ids=ids,
        max_samples=max_samples,
    )


def run_marginals(class_id: str | int = "all", config_path: str | None = None) -> list[dict]:
    results = []
    for cid in _parse_class_ids(class_id):
        results.append(fit_class_marginals(cid, config_path=config_path))
    return results


def run_joint(class_id: str | int = "all", config_path: str | None = None) -> list[dict]:
    results = []
    for cid in _parse_class_ids(class_id):
        results.append(fit_class_joint(cid, config_path=config_path))
    return results


def run_full_pipeline(
    class_id: str | int = "all",
    config_path: str | None = None,
    refresh: bool = False,
    max_samples: int | None = None,
) -> dict:
    paths = run_extract(
        class_id=class_id,
        config_path=config_path,
        refresh=refresh,
        max_samples=max_samples,
    )
    marginals = run_marginals(class_id=class_id, config_path=config_path)
    joints = run_joint(class_id=class_id, config_path=config_path)
    return {
        "extract_paths": [str(p) for p in paths],
        "marginals": [{"class_id": m["class_id"], "n_samples": m["n_samples"]} for m in marginals],
        "joints": [
            {"class_id": j["class_id"], "best_copula_type": j["best_copula_type"], "n_samples": j["n_samples"]}
            for j in joints
        ],
    }
