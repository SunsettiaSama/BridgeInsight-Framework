"""窗间 Markov Copula 的 MC 轨迹抽样。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from src.chapter4_characteristics.settings import CLASS_LABELS, load_config
from src.chapter4_characteristics.statistics.copula import _clip, _nearest_pd, _stabilize_corr
from src.chapter4_characteristics.statistics.td_copula.fit import td_fit_path
from src.chapter4_characteristics.statistics.td_copula.sequence import get_td_dir


def td_sample_path(cfg: dict, class_id: int) -> Path:
    return get_td_dir(cfg) / f"class_{class_id}_td_paths.npz"


def _gmm_ppf(u: np.ndarray, params: dict) -> np.ndarray:
    weights = np.asarray(params["weights"], dtype=np.float64)
    means = np.asarray(params["means"], dtype=np.float64)
    variances = np.asarray(params["variances"], dtype=np.float64)
    scales = np.sqrt(np.maximum(variances, 1e-30))

    def _cdf_scalar(x: float) -> float:
        return float(np.sum(weights * stats.norm.cdf(x, loc=means, scale=scales)))

    out = np.empty_like(u, dtype=np.float64)
    x_lo = float(np.min(means - 8.0 * scales))
    x_hi = float(np.max(means + 8.0 * scales))
    for i, ui in enumerate(np.asarray(u, dtype=np.float64).ravel()):
        ui = float(np.clip(ui, 1e-10, 1.0 - 1e-10))
        out.ravel()[i] = optimize.brentq(
            lambda x: _cdf_scalar(x) - ui, x_lo, x_hi, xtol=1e-8, maxiter=200
        )
    return out.reshape(u.shape)


def _marginal_ppf(u: np.ndarray, best: dict) -> np.ndarray:
    form = best["form"]
    params = best["params"]
    u = np.clip(np.asarray(u, dtype=np.float64), 1e-10, 1.0 - 1e-10)
    if form.startswith("gmm_"):
        return _gmm_ppf(u, params)
    dist_obj = getattr(stats, form)
    return dist_obj.ppf(u, *list(params.values()))


def _inverse_margins(u: np.ndarray, var_names: list[str], marginals: dict) -> np.ndarray:
    x = np.empty_like(u, dtype=np.float64)
    for j, name in enumerate(var_names):
        entry = marginals.get(name)
        if entry is None or entry.get("best") is None:
            raise ValueError(f"缺少边缘拟合：{name}")
        x[:, j] = _marginal_ppf(u[:, j], entry["best"])
    return x


def _sample_gaussian_block(R: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    R = _stabilize_corr(np.asarray(R, dtype=np.float64), shrink=1e-6)
    L = np.linalg.cholesky(R)
    z = rng.standard_normal((n, R.shape[0])) @ L.T
    return stats.norm.cdf(z)


def _cond_gaussian(
    R: np.ndarray,
    u_cond: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """给定 U_t，从 Gaussian Copula 条件抽样 U_{t+1}。"""
    u_cond = _clip(np.asarray(u_cond, dtype=np.float64))
    n, d = u_cond.shape
    R = _stabilize_corr(np.asarray(R, dtype=np.float64), shrink=1e-6)
    r11 = R[:d, :d]
    r12 = R[:d, d:]
    r21 = R[d:, :d]
    r22 = R[d:, d:]
    z1 = stats.norm.ppf(u_cond)
    mean = (r21 @ np.linalg.solve(r11, z1.T)).T
    cov = r22 - r21 @ np.linalg.solve(r11, r12)
    cov = 0.5 * (cov + cov.T)
    if np.min(np.linalg.eigvalsh(cov)) <= 0:
        cov = _nearest_pd(cov)
    L = np.linalg.cholesky(cov)
    z2 = mean + rng.standard_normal((n, d)) @ L.T
    return stats.norm.cdf(z2)


def _cond_t(
    R: np.ndarray,
    nu: float,
    u_cond: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """多元 t Copula 条件抽样（latent MV-t 条件公式）。"""
    u_cond = _clip(np.asarray(u_cond, dtype=np.float64))
    n, d = u_cond.shape
    R = _stabilize_corr(np.asarray(R, dtype=np.float64), shrink=1e-6)
    r11 = R[:d, :d]
    r12 = R[:d, d:]
    r21 = R[d:, :d]
    r22 = R[d:, d:]
    z1 = stats.t.ppf(u_cond, df=nu)
    # Mahalanobis of z1 under r11
    quad = np.einsum("ni,ij,nj->n", z1, np.linalg.inv(r11), z1)
    mean = (r21 @ np.linalg.solve(r11, z1.T)).T
    base_cov = r22 - r21 @ np.linalg.solve(r11, r12)
    base_cov = 0.5 * (base_cov + base_cov.T)
    if np.min(np.linalg.eigvalsh(base_cov)) <= 0:
        base_cov = _nearest_pd(base_cov)
    nu2 = nu + d
    scale = (nu + quad) / nu2
    # sample from t_{nu2}(mean, scale * base_cov)
    L = np.linalg.cholesky(base_cov)
    g = rng.standard_normal((n, d)) @ L.T
    chi = rng.chisquare(nu2, size=n) / nu2
    z2 = mean + np.sqrt(scale)[:, None] * g / np.sqrt(chi)[:, None]
    return stats.t.cdf(z2, df=nu)


def sample_paths(
    payload: dict,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    返回 shape (n_paths, n_steps, d) 的特征轨迹（物理尺度）。

    初始 U_0 ~ Copula(R11)；其后 U_{t+1}|U_t 按椭圆条件抽样。
    """
    rng = np.random.default_rng(rng)
    var_names = list(payload["variable_names"])
    d = len(var_names)
    best = payload["best_copula"]
    ctype = best["copula_type"]
    R = np.asarray(best["params"]["correlation_matrix"], dtype=np.float64)
    if R.shape[0] != 2 * d:
        raise ValueError(f"相关矩阵维数不匹配：{R.shape} vs 2d={2*d}")

    r11 = R[:d, :d]
    u = np.empty((n_paths, n_steps, d), dtype=np.float64)
    if ctype == "gaussian":
        u[:, 0, :] = _sample_gaussian_block(r11, n_paths, rng)
        for t in range(n_steps - 1):
            u[:, t + 1, :] = _cond_gaussian(R, u[:, t, :], rng)
    elif ctype == "t":
        nu = float(best["params"]["nu"])
        # 初始：用边际相关块的 t Copula
        from src.chapter4_characteristics.statistics.copula import CopulaResult, sample_t_copula

        init = CopulaResult(
            "t",
            {"correlation_matrix": r11.tolist(), "nu": nu},
            0.0,
            0.0,
            0.0,
            n_paths,
            d,
        )
        u[:, 0, :] = sample_t_copula(init, n_paths, rng=rng)
        for t in range(n_steps - 1):
            u[:, t + 1, :] = _cond_t(R, nu, u[:, t, :], rng)
    else:
        raise ValueError(f"不支持的时序 Copula 类型：{ctype}")

    # 逐步逆边缘
    x = np.empty_like(u)
    for t in range(n_steps):
        x[:, t, :] = _inverse_margins(u[:, t, :], var_names, payload["marginals"])
    return x


def run_sample_class(
    class_id: int,
    config_path: str | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
) -> Path:
    cfg = load_config(config_path)
    td = cfg.get("td_copula") or {}
    n_paths = int(n_paths if n_paths is not None else td.get("n_paths", 100))
    n_steps = int(n_steps if n_steps is not None else td.get("n_steps", 60))
    seed = int(td.get("rng_seed", cfg.get("copula_rng_seed", 42)))

    fit_path = td_fit_path(cfg, class_id)
    if not fit_path.exists():
        raise FileNotFoundError(f"td 拟合结果不存在：{fit_path}；请先运行 td_copula fit")
    with open(fit_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    print("=" * 80)
    print(
        f"[td] MC 轨迹 class={class_id} ({CLASS_LABELS.get(class_id, '?')}) "
        f"n_paths={n_paths} n_steps={n_steps}"
    )
    print("=" * 80)

    paths = sample_paths(payload, n_paths=n_paths, n_steps=n_steps, rng=np.random.default_rng(seed))
    out = td_sample_path(cfg, class_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        paths=paths,
        variable_names_json=np.asarray(
            json.dumps(payload["variable_names"], ensure_ascii=False)
        ),
        class_id=np.asarray(class_id),
        n_paths=np.asarray(n_paths),
        n_steps=np.asarray(n_steps),
        best_copula_type=np.asarray(payload["best_copula_type"]),
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    print(f"  [td] 已保存轨迹：{out}  shape={paths.shape}")
    return out
