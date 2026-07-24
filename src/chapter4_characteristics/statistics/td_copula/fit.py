"""核心特征边缘 + 窗间一阶 Markov 椭圆 Copula（拟合 [U_t; U_{t+1}]）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

from src.chapter4_characteristics.settings import (
    CLASS_DIRS,
    CLASS_LABELS,
    DEFAULT_LABEL_NAMES,
    load_config,
)
from src.chapter4_characteristics.statistics.config import load_config as load_stats_config
from src.chapter4_characteristics.statistics.copula import fit_copula
from src.chapter4_characteristics.statistics.pipeline import _empirical_pit, _fit_best
from src.chapter4_characteristics.statistics.td_copula.sequence import (
    build_lag1_pairs,
    get_td_dir,
    load_td_seq,
    td_seq_path,
)

ELLIPTICAL = ["gaussian", "t"]


def td_fit_path(cfg: dict, class_id: int) -> Path:
    return get_td_dir(cfg) / f"class_{class_id}_td_copula.json"


def _freq_energy_sets(var_names: list[str]) -> tuple[set[int], set[int]]:
    freq_idx: set[int] = set()
    energy_idx: set[int] = set()
    for j, name in enumerate(var_names):
        if name.startswith("freq_"):
            freq_idx.add(j)
        else:
            energy_idx.add(j)
    return freq_idx, energy_idx


def _lag1_diag_spearman(xt: np.ndarray, xt1: np.ndarray) -> list[float]:
    d = xt.shape[1]
    out: list[float] = []
    for j in range(d):
        rho, _ = stats.spearmanr(xt[:, j], xt1[:, j])
        out.append(float(rho) if np.isfinite(rho) else 0.0)
    return out


def fit_class_td(
    class_id: int,
    config_path: str | None = None,
    stats_yaml: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    scfg = load_stats_config(stats_yaml)
    seq = load_td_seq(td_seq_path(cfg, class_id))
    features = seq["features"]
    var_names = list(seq["var_names"])
    energy_top_k = int(seq["config"].get("energy_top_k", (len(var_names) - 2) // 2))

    xt, xt1 = build_lag1_pairs(
        features, seq["in_paths"], seq["out_paths"], seq["window_idx"]
    )
    stacked_raw = np.hstack([xt, xt1])
    complete = np.all(np.isfinite(stacked_raw) & (stacked_raw > 0), axis=1)
    xt_f = xt[complete]
    xt1_f = xt1[complete]
    stacked = np.hstack([xt_f, xt1_f])

    print("=" * 80)
    print(
        f"[td] 边缘+窗间 Copula class={class_id} ({CLASS_LABELS.get(class_id, '?')}) "
        f"n_windows={features.shape[0]} n_pairs={stacked.shape[0]} d={features.shape[1]}"
    )
    print("=" * 80)

    if stacked.shape[0] < 30:
        raise ValueError(f"td_copula：完整相邻窗对不足：{stacked.shape[0]}")

    freq_set, energy_set = _freq_energy_sets(var_names)
    marginals: dict = {}
    for j, name in enumerate(var_names):
        candidates = (
            scfg.candidate_dists_freq if j in freq_set else scfg.candidate_dists_energy
        )
        result = _fit_best(
            features[:, j],
            candidates,
            scfg.min_valid_samples,
            enable_gmm=scfg.enable_gmm,
            gmm_max_components=scfg.gmm_max_components,
        )
        if result is None:
            print(f"  [{j+1:2d}/{len(var_names)}] {name:<16} 跳过")
        else:
            best = result["best"]
            print(
                f"  [{j+1:2d}/{len(var_names)}] {name:<16} → {best['form']:<12} "
                f"AIC={best['aic']:10.2f}"
            )
        marginals[name] = result

    td = cfg.get("td_copula") or {}
    max_n = int(td.get("joint_max_pairs", cfg.get("td_joint_max_pairs", 5000)))
    if stacked.shape[0] > max_n:
        rng = np.random.default_rng(int(td.get("rng_seed", cfg.get("copula_rng_seed", 42))))
        idx = rng.choice(stacked.shape[0], size=max_n, replace=False)
        stacked = stacked[idx]
        xt_f = xt_f[idx]
        xt1_f = xt1_f[idx]
        print(f"  [td] 联合拟合抽样 {max_n} pairs")

    u_stacked = _empirical_pit(stacked)
    pair_names = [f"{n}_t" for n in var_names] + [f"{n}_t1" for n in var_names]

    print("  [td] 比较椭圆 Copula：gaussian / t ...")
    comparison: list[dict] = []
    best_result = None
    best_aic = float("inf")
    for ctype in ELLIPTICAL:
        result = fit_copula(u_stacked, copula_type=ctype)
        entry = {
            "copula_type": result.copula_type,
            "log_likelihood": result.log_likelihood,
            "aic": result.aic,
            "bic": result.bic,
            "n_samples": result.n_samples,
            "n_vars": result.n_vars,
            "params": result.params,
        }
        comparison.append(entry)
        print(f"    {ctype}: AIC={result.aic:.2f} BIC={result.bic:.2f}")
        if result.aic < best_aic:
            best_aic = result.aic
            best_result = entry

    if best_result is None:
        raise RuntimeError("td_copula：椭圆 Copula 拟合失败")

    rho_raw, _ = stats.spearmanr(stacked)
    spearman = np.asarray(rho_raw, dtype=np.float64)
    if spearman.ndim == 0:
        spearman = np.array([[1.0, float(spearman)], [float(spearman), 1.0]])

    lag1_diag = _lag1_diag_spearman(xt_f, xt1_f)

    payload = {
        "class_id": class_id,
        "class_label": CLASS_DIRS[class_id],
        "class_name": (
            DEFAULT_LABEL_NAMES[class_id]
            if class_id < len(DEFAULT_LABEL_NAMES)
            else CLASS_LABELS[class_id]
        ),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "td_copula_lag1_markov",
        "energy_top_k": energy_top_k,
        "n_windows": int(features.shape[0]),
        "n_pairs": int(stacked.shape[0]),
        "n_vars": len(var_names),
        "variable_names": var_names,
        "pair_variable_names": pair_names,
        "sequence_config": seq["config"],
        "stats_config": scfg.to_dict(),
        "marginals": marginals,
        "best_copula_type": best_result["copula_type"],
        "best_copula": best_result,
        "comparison": comparison,
        "lag1_spearman_diag": lag1_diag,
        "correlation": {
            "spearman": spearman.tolist(),
        },
        "pit_mode": "empirical",
        "note": (
            "椭圆 Copula 拟合于 [U_t; U_{t+1}]；"
            "MC 用条件抽样生成窗间轨迹；"
            "雨流需窗内过程层（stage=intra），本结果不含。"
        ),
    }

    out_path = td_fit_path(cfg, class_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  [td] 已保存：{out_path}")
    return payload
