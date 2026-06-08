from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .fitting import FitResult, fit_distribution
from .copula import CopulaResult, fit_copula, sample_from_copula


# ─── Result containers ────────────────────────────────────────────────────────

@dataclass
class CorrelationResult:
    variable_names: List[str]
    pearson:  np.ndarray = field(repr=False)
    spearman: np.ndarray = field(repr=False)
    kendall:  np.ndarray = field(repr=False)

    def summary(self, max_show: int = 6) -> str:
        d = len(self.variable_names)
        show = min(d, max_show)
        lines = [f"CorrelationResult (d={d})"]
        header = "  " + "  ".join(f"{v[:8]:>8}" for v in self.variable_names[:show])
        lines.append(header)
        for i in range(show):
            row = "  ".join(f"{self.spearman[i, j]:+8.3f}" for j in range(show))
            lines.append(f"  {self.variable_names[i][:8]:>8}  {row}")
        if d > show:
            lines.append(f"  ... ({d - show} 个变量已省略，仅展示 Spearman ρ)")
        return "\n".join(lines)


@dataclass
class MultivariateFitResult:
    variable_names: List[str]
    marginals:   Dict[str, FitResult]
    correlation: CorrelationResult
    copula:      CopulaResult
    u_matrix:    np.ndarray = field(repr=False)

    def summary(self) -> str:
        n = self.u_matrix.shape[0]
        d = len(self.variable_names)
        lines = ["=" * 68, "MultivariateFitResult", "=" * 68,
                 f"  变量数: {d}  |  样本数: {n}", "",
                 "── 边缘分布 ──"]
        for name, r in self.marginals.items():
            ps = ", ".join(f"{k}={v:.4g}" for k, v in r.params.items())
            ks_str = (f"  KS-p={r.ks_pvalue:.4f}  AIC={r.aic:.2f}  BIC={r.bic:.2f}"
                      if r.ks_pvalue is not None else "")
            lines.append(f"  {name:<28} [{r.form}]  {ps}")
            if ks_str:
                lines.append(f"  {'':<28} {ks_str}")
        lines += ["", "── 相关性（Spearman ρ） ──",
                  self.correlation.summary(), "",
                  "── Copula ──",
                  str(self.copula),
                  "=" * 68]
        return "\n".join(lines)


# ─── Correlation analysis ─────────────────────────────────────────────────────

def correlation_analysis(
    data: np.ndarray,
    variable_names: List[str],
) -> CorrelationResult:
    n, d = data.shape
    if len(variable_names) != d:
        raise ValueError(
            f"variable_names 长度 ({len(variable_names)}) 与数据列数 ({d}) 不符"
        )

    pearson = np.corrcoef(data.T)

    rho_raw, _ = stats.spearmanr(data)
    if d == 2:
        rho = float(rho_raw)
        spearman = np.array([[1.0, rho], [rho, 1.0]])
    else:
        spearman = np.asarray(rho_raw, dtype=float)
        np.fill_diagonal(spearman, 1.0)

    kendall = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = stats.kendalltau(data[:, i], data[:, j])
            kendall[i, j] = kendall[j, i] = float(tau)

    return CorrelationResult(
        variable_names=list(variable_names),
        pearson=pearson,
        spearman=spearman,
        kendall=kendall,
    )


# ─── Probability Integral Transform ──────────────────────────────────────────

def pit_transform(
    data: np.ndarray,
    marginals: Dict[str, FitResult],
    variable_names: List[str],
) -> np.ndarray:
    n, d = data.shape
    u = np.empty((n, d), dtype=float)
    for j, name in enumerate(variable_names):
        r = marginals[name]
        dist_obj = getattr(stats, r.form)
        u[:, j] = dist_obj.cdf(data[:, j], *list(r.params.values()))
    return np.clip(u, 1e-10, 1.0 - 1e-10)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def fit_multivariate(
    data: np.ndarray,
    variable_names: Sequence[str],
    marginal_distributions: Sequence[str],
    copula_type: str = "gaussian",
    marginal_fit_kwargs: Optional[Dict[str, Dict]] = None,
) -> MultivariateFitResult:
    """
    四步流水线：

      1. 对每个变量独立拟合边缘分布（marginal_distributions 可传单个字符串广播）
      2. PIT 变换 → 均匀边缘矩阵 U ∈ [0,1]^{n×d}
      3. 对原始数据计算 Pearson / Spearman / Kendall 相关矩阵
      4. 对 U 拟合指定 copula（gaussian / t / gumbel / clayton / frank）

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_vars)
    variable_names : sequence of str
    marginal_distributions : sequence of str
        scipy.stats 分布名称列表，长度须为 1（广播）或等于变量数
    copula_type : str
        目标 copula 类型
    marginal_fit_kwargs : dict, optional
        {variable_name: {fit kwargs}} 透传给 fit_distribution()

    Returns
    -------
    MultivariateFitResult
    """
    data = np.asarray(data, dtype=float)
    variable_names = list(variable_names)
    marginal_distributions = list(marginal_distributions)
    n, d = data.shape

    if len(variable_names) != d:
        raise ValueError(
            f"variable_names 长度 ({len(variable_names)}) 须等于数据列数 ({d})"
        )
    if len(marginal_distributions) == 1:
        marginal_distributions = marginal_distributions * d
    if len(marginal_distributions) != d:
        raise ValueError(
            f"marginal_distributions 长度须为 1（广播）或 {d}，"
            f"当前为 {len(marginal_distributions)}"
        )

    marginal_fit_kwargs = marginal_fit_kwargs or {}

    # ── Step 1: 边缘分布拟合 ──────────────────────────────────────────────────
    print(f"[1/4] 拟合边缘分布（{d} 个变量）...")
    marginals: Dict[str, FitResult] = {}
    for j, (name, dist) in enumerate(zip(variable_names, marginal_distributions)):
        col = data[:, j]
        col = col[np.isfinite(col)]
        kw = marginal_fit_kwargs.get(name, {})
        r = fit_distribution(col, distribution=dist, **kw)
        marginals[name] = r
        print(f"  [{j+1:2d}/{d}] {name:<28} [{dist:<12}]  "
              f"AIC={r.aic:9.2f}  KS-p={r.ks_pvalue:.4f}")

    # ── Step 2: PIT 变换 ──────────────────────────────────────────────────────
    print("[2/4] 概率积分变换（PIT）→ 均匀边缘...")
    u_matrix = pit_transform(data, marginals, variable_names)

    # ── Step 3: 相关性分析 ────────────────────────────────────────────────────
    print("[3/4] 相关性分析（Pearson / Spearman / Kendall）...")
    correlation = correlation_analysis(data, variable_names)
    show = min(d, 5)
    print("  Spearman ρ（前 {}×{} 块）：".format(show, show))
    for i in range(show):
        row = "  ".join(f"{correlation.spearman[i, j]:+.3f}" for j in range(show))
        print(f"    {variable_names[i][:16]:<16}  {row}")
    if d > show:
        print(f"    ... ({d - show} 个变量已省略)")

    # ── Step 4: Copula 拟合 ───────────────────────────────────────────────────
    print(f"[4/4] 拟合 {copula_type} Copula（d={d}）...")
    copula_result = fit_copula(u_matrix, copula_type=copula_type)
    print(f"  ✓  log-likelihood = {copula_result.log_likelihood:.4f}"
          f"  AIC = {copula_result.aic:.4f}"
          f"  BIC = {copula_result.bic:.4f}")

    return MultivariateFitResult(
        variable_names=variable_names,
        marginals=marginals,
        correlation=correlation,
        copula=copula_result,
        u_matrix=u_matrix,
    )


# ─── Sampling ─────────────────────────────────────────────────────────────────

def sample_from_multivariate(
    result: MultivariateFitResult,
    n_samples: int,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从已拟合的联合分布中采样。

    Returns
    -------
    x_samples : ndarray, shape (n_samples, n_vars)
        原始尺度的联合样本
    u_samples : ndarray, shape (n_samples, n_vars)
        Copula 空间（均匀边缘）的样本
    """
    u_samples = sample_from_copula(result.copula, n_samples, rng)
    n, d = u_samples.shape
    x = np.empty((n, d), dtype=float)
    for j, name in enumerate(result.variable_names):
        r = result.marginals[name]
        dist_obj = getattr(stats, r.form)
        x[:, j] = dist_obj.ppf(u_samples[:, j], *list(r.params.values()))
    return x, u_samples


# ─── Goodness-of-fit summary ──────────────────────────────────────────────────

def compare_copulas(
    u: np.ndarray,
    copula_types: Optional[List[str]] = None,
) -> List[CopulaResult]:
    """
    对同一数据并行拟合多种 copula，按 AIC 升序排列返回结果列表。

    Parameters
    ----------
    u : ndarray, shape (n_samples, n_vars)
        均匀边缘数据（已经过 PIT 变换）
    copula_types : list of str, optional
        待比较的 copula 类型；None 则自动选择适合当前维度的所有类型

    Returns
    -------
    List[CopulaResult]  按 AIC 升序排列
    """
    from .copula import SUPPORTED_COPULAS, _ARCHIMEDEAN

    u = np.asarray(u, dtype=float)
    d = u.shape[1]

    if copula_types is None:
        copula_types = list(SUPPORTED_COPULAS)
        if d > 2:
            copula_types = [c for c in copula_types if c not in _ARCHIMEDEAN]

    results: List[CopulaResult] = []
    for ctype in copula_types:
        if ctype in _ARCHIMEDEAN and d != 2:
            print(f"  跳过 {ctype}（仅支持 d=2，当前 d={d}）")
            continue
        r = fit_copula(u, copula_type=ctype)
        results.append(r)
        print(f"  {ctype:<10}  log-lik={r.log_likelihood:10.4f}"
              f"  AIC={r.aic:10.4f}  BIC={r.bic:10.4f}")

    results.sort(key=lambda r: r.aic)
    return results
