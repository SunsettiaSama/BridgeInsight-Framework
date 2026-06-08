from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.statistics.fitting import fit_distribution, fit_gmm, FitResult
from src.statistics.multivariate import correlation_analysis, CorrelationResult
from src.config.statistics.config import StatisticsConfig, load_config


# ==================== 固定内部常量 ====================
# floc=0 强制将位置参数固定在 0，保证支持域严格为 x>0，与频率/能量占比的物理意义一致。
# gamma(floc=0)：mode=(a-1)*scale，a>1 时峰值在 x>0，是主频和能量占比的首选分布。
# lognorm(floc=0)：次优，s 较大时峰值偏低，但 AIC 选优后一般不会误选。
# norm / expon 已从默认候选中移除，此处仅作为扩展备用。

_DIST_FIT_KWARGS: dict = {
    "gamma":   {"floc": 0},
    "lognorm": {"floc": 0},
    "expon":   {"floc": 0},
    "norm":    {},
}


# ==================== 变量名 ====================

def _build_var_names(n_modes: int) -> list[str]:
    names: list[str] = []
    for k in range(1, n_modes + 1):
        names.append(f"freq_in_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"energy_in_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"freq_out_{k}")
    for k in range(1, n_modes + 1):
        names.append(f"energy_out_{k}")
    return names


# ==================== 数据加载 ====================

def _extract_top_freq_energy(
    freqs: list, powers: list, n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    f = np.array(freqs, dtype=np.float64)
    p = np.array(powers, dtype=np.float64)

    order    = np.argsort(p)[::-1]
    f_sorted = f[order]
    p_sorted = p[order]

    f_out = np.full(n_modes, np.nan)
    e_out = np.full(n_modes, np.nan)

    n     = min(len(f_sorted), n_modes)
    total = p_sorted.sum()
    if total <= 0:
        return f_out, e_out

    f_out[:n] = f_sorted[:n]
    e_out[:n] = p_sorted[:n] / total
    return f_out, e_out


def _extract_row(sample: dict, n_modes: int) -> Optional[np.ndarray]:
    psd_in  = sample.get("psd_inplane")  or {}
    psd_out = sample.get("psd_outplane") or {}

    freqs_in   = psd_in.get("frequencies")  or []
    powers_in  = psd_in.get("powers")       or []
    freqs_out  = psd_out.get("frequencies") or []
    powers_out = psd_out.get("powers")      or []

    if not freqs_in or not freqs_out:
        return None

    f_in,  e_in  = _extract_top_freq_energy(freqs_in,  powers_in,  n_modes)
    f_out, e_out = _extract_top_freq_energy(freqs_out, powers_out, n_modes)

    return np.concatenate([f_in, e_in, f_out, e_out])


def load_mode_data(cfg: StatisticsConfig) -> tuple[np.ndarray, int]:
    stats_dir = (
        project_root / "results" / "enriched_stats" / cfg.class_label
    )
    json_files = sorted(stats_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"无 JSON 文件：{stats_dir}")

    rows: list[np.ndarray] = []
    n_total = 0

    for jf in json_files:
        print(f"  读取 {jf.name} ...")
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            n_total += 1
            row = _extract_row(sample, cfg.n_modes)
            if row is not None:
                rows.append(row)

    if not rows:
        raise ValueError("未提取到任何有效样本")

    return np.array(rows, dtype=np.float64), n_total


# ==================== 边缘分布拟合 ====================

def _fitresult_to_dict(r: FitResult, n_valid: int) -> dict:
    return {
        "form":         r.form,
        "params":       r.params,
        "ks_statistic": r.ks_statistic,
        "ks_pvalue":    r.ks_pvalue,
        "aic":          r.aic,
        "bic":          r.bic,
        "n_valid":      n_valid,
    }


def _fit_best(
    col: np.ndarray,
    candidates: list[str],
    min_valid: int,
    enable_gmm: bool = False,
    gmm_max_components: int = 2,
) -> Optional[dict]:
    valid   = col[np.isfinite(col) & (col > 0)]
    n_valid = int(len(valid))

    if n_valid < min_valid or float(np.std(valid)) < 1e-12:
        return None

    results: list[FitResult] = []
    for dist_name in candidates:
        kw = _DIST_FIT_KWARGS.get(dist_name, {})
        r  = fit_distribution(valid, distribution=dist_name, **kw)
        results.append(r)

    if enable_gmm:
        for n_comp in range(2, gmm_max_components + 1):
            if n_valid >= n_comp * 5:
                results.append(fit_gmm(valid, n_components=n_comp))

    best = min(results, key=lambda r: r.aic)

    return {
        "n_valid":    n_valid,
        "best":       _fitresult_to_dict(best, n_valid),
        "candidates": [_fitresult_to_dict(r, n_valid) for r in results],
    }


def fit_all_marginals(
    matrix: np.ndarray,
    var_names: list[str],
    cfg: StatisticsConfig,
) -> dict:
    n_modes = cfg.n_modes
    n_vars  = len(var_names)

    freq_set   = set(range(0,          n_modes)) | set(range(2 * n_modes, 3 * n_modes))
    energy_set = set(range(n_modes,    2 * n_modes)) | set(range(3 * n_modes, 4 * n_modes))

    marginals: dict = {}

    for j in range(n_vars):
        name = var_names[j]
        col  = matrix[:, j]

        if j in freq_set:
            candidates = cfg.candidate_dists_freq
        elif j in energy_set:
            candidates = cfg.candidate_dists_energy
        else:
            candidates = cfg.candidate_dists_freq

        result = _fit_best(
            col, candidates, cfg.min_valid_samples,
            enable_gmm=cfg.enable_gmm,
            gmm_max_components=cfg.gmm_max_components,
        )

        if result is None:
            print(f"  [{j+1:2d}/{n_vars}] {name:<20} 跳过（有效样本不足）")
        else:
            best_form = result["best"]["form"]
            aic_val   = result["best"]["aic"]
            ks_p      = result["best"].get("ks_pvalue")
            ks_str    = f"{ks_p:.4f}" if ks_p is not None else "N/A"
            print(f"  [{j+1:2d}/{n_vars}] {name:<20} → {best_form:<12} "
                  f"AIC={aic_val:10.2f}  KS-p={ks_str:<8}  n={result['n_valid']}")

        marginals[name] = result

    return marginals


# ==================== 相关性分析 ====================

def _corrresult_to_dict(r: CorrelationResult, n_samples: int) -> dict:
    return {
        "n_samples":      n_samples,
        "variable_names": r.variable_names,
        "pearson":        r.pearson.tolist(),
        "spearman":       r.spearman.tolist(),
        "kendall":        r.kendall.tolist(),
    }


def run_correlation(
    matrix: np.ndarray,
    var_names: list[str],
    cfg: StatisticsConfig,
) -> dict:
    complete_mask = np.all(np.isfinite(matrix), axis=1)
    mat_complete  = matrix[complete_mask]
    n_complete    = int(mat_complete.shape[0])
    print(f"  完整行（无 NaN）数量：{n_complete} / {matrix.shape[0]}")

    if n_complete < cfg.min_valid_samples:
        raise ValueError(
            f"完整样本数不足（{n_complete} < {cfg.min_valid_samples}）"
        )

    if n_complete > cfg.corr_max_n:
        rng      = np.random.default_rng(cfg.corr_rng_seed)
        idx      = rng.choice(n_complete, size=cfg.corr_max_n, replace=False)
        mat_corr = mat_complete[idx]
        n_used   = cfg.corr_max_n
        print(f"  相关性分析随机抽样 {n_used} 行（避免 Kendall τ 计算过慢）")
    else:
        mat_corr = mat_complete
        n_used   = n_complete

    corr = correlation_analysis(mat_corr, var_names)
    return _corrresult_to_dict(corr, n_used)


# ==================== 保存结果 ====================

def save_results(
    marginals: dict,
    correlation: dict,
    var_names: list[str],
    n_loaded: int,
    n_extracted: int,
    cfg: StatisticsConfig,
) -> Path:
    output_dir  = project_root / "results" / cfg.output_subdir
    output_file = output_dir / cfg.output_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "class":           cfg.class_label,
            "n_modes":         cfg.n_modes,
            "n_samples_total": n_loaded,
            "n_samples_valid": n_extracted,
            "variable_names":  var_names,
            "config":          cfg.to_dict(),
            "created_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "marginals":   marginals,
        "correlation": correlation,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output_file


# ==================== 主流水线 ====================

def run(cfg: Optional[StatisticsConfig] = None, yaml_path: str = None) -> None:
    if cfg is None:
        cfg = load_config(yaml_path)

    print("=" * 80)
    print(f"正常振动 前{cfg.n_modes}阶主频/能量 边缘分布拟合 + 相关性分析")
    print("=" * 80)
    print(f"  类别：{cfg.class_label}")
    print(f"  模态阶数：{cfg.n_modes}")
    print(f"  频率候选分布：{cfg.candidate_dists_freq}")
    print(f"  能量候选分布：{cfg.candidate_dists_energy}")

    var_names = _build_var_names(cfg.n_modes)
    print(f"\n变量数：{len(var_names)}  ({var_names[0]} … {var_names[-1]})")

    print(f"\n[步骤 1/3] 加载 PSD 模态数据...")
    matrix, n_loaded = load_mode_data(cfg)
    n_extracted = matrix.shape[0]
    print(f"  ✓ 总样本 {n_loaded}，有效提取 {n_extracted}，矩阵 {matrix.shape}")

    print(f"\n[步骤 2/3] 边缘分布拟合（AIC 选优）...")
    marginals = fit_all_marginals(matrix, var_names, cfg)
    n_fitted  = sum(1 for v in marginals.values() if v is not None)
    print(f"  ✓ 成功拟合：{n_fitted} / {len(var_names)} 个变量")

    print(f"\n[步骤 3/3] 相关性分析（Pearson / Spearman / Kendall）...")
    correlation = run_correlation(matrix, var_names, cfg)
    print(f"  ✓ 相关性矩阵 {len(var_names)}×{len(var_names)} 计算完成")

    print(f"\n[保存] 写出结果文件...")
    out_path = save_results(
        marginals, correlation, var_names, n_loaded, n_extracted, cfg
    )
    print(f"  ✓ 已保存：{out_path}")

    print("\n" + "=" * 80)
    print("运行完成")
    print("=" * 80)


if __name__ == "__main__":
    run()
