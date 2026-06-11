from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

SUPPORTED_COPULAS = frozenset({"gaussian", "t", "gumbel", "clayton", "frank"})
_ARCHIMEDEAN    = frozenset({"gumbel", "clayton", "frank"})
_CLIP_EPS: float = 1e-10


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class CopulaResult:
    copula_type: str
    params: Dict
    log_likelihood: float
    aic: float
    bic: float
    n_samples: int
    n_vars: int

    def __str__(self) -> str:
        lines = [
            f"CopulaResult(type='{self.copula_type}', n={self.n_samples}, d={self.n_vars})",
            f"  log-likelihood = {self.log_likelihood:.4f}",
            f"  AIC = {self.aic:.4f},  BIC = {self.bic:.4f}",
        ]
        for k, v in self.params.items():
            if isinstance(v, list):
                dim = len(v)
                lines.append(f"  {k}: <{dim}×{len(v[0]) if isinstance(v[0], list) else 1} matrix>")
            else:
                lines.append(f"  {k} = {v:.6g}")
        return "\n".join(lines)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _clip(u: np.ndarray) -> np.ndarray:
    return np.clip(u, _CLIP_EPS, 1.0 - _CLIP_EPS)


def _is_pd(M: np.ndarray) -> bool:
    return bool(np.all(np.linalg.eigvalsh(M) > 0))


def _nearest_pd(M: np.ndarray) -> np.ndarray:
    B = (M + M.T) / 2.0
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    A = (B + H) / 2.0
    A = (A + A.T) / 2.0
    eps = np.spacing(np.linalg.norm(M))
    I = np.eye(M.shape[0])
    k = 1
    while not _is_pd(A):
        mineig = float(np.min(np.real(np.linalg.eigvals(A))))
        A += I * (-mineig * k ** 2 + eps)
        k += 1
    return A


def _spearman_matrix(u: np.ndarray) -> np.ndarray:
    d = u.shape[1]
    rho_raw, _ = stats.spearmanr(u)
    if d == 2:
        rho = float(rho_raw)
        return np.array([[1.0, rho], [rho, 1.0]])
    mat = np.asarray(rho_raw, dtype=float)
    np.fill_diagonal(mat, 1.0)
    return mat


def _kendall_matrix(u: np.ndarray) -> np.ndarray:
    d = u.shape[1]
    K = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = stats.kendalltau(u[:, i], u[:, j])
            K[i, j] = K[j, i] = float(tau)
    return K


def _aic_bic(ll: float, k_params: int, n: int):
    return 2.0 * k_params - 2.0 * ll, k_params * float(np.log(n)) - 2.0 * ll


# ─── Gaussian Copula ──────────────────────────────────────────────────────────

def _spearman_to_gauss_corr(rho_s: np.ndarray) -> np.ndarray:
    # van der Waerden conversion: ρ_gaussian = 2 sin(π ρ_s / 6)
    return 2.0 * np.sin(np.pi * rho_s / 6.0)


def _gauss_logpdf(u: np.ndarray, R: np.ndarray) -> np.ndarray:
    z = stats.norm.ppf(_clip(u))
    d = R.shape[0]
    R_inv = np.linalg.inv(R)
    _, logdet = np.linalg.slogdet(R)
    quad = np.einsum('ni,ij,nj->n', z, R_inv - np.eye(d), z)
    return -0.5 * logdet - 0.5 * quad


def fit_gaussian_copula(u: np.ndarray) -> CopulaResult:
    u = _clip(np.asarray(u, dtype=float))
    n, d = u.shape
    R = _spearman_to_gauss_corr(_spearman_matrix(u))
    np.fill_diagonal(R, 1.0)
    if not _is_pd(R):
        R = _nearest_pd(R)
    ll = float(np.sum(_gauss_logpdf(u, R)))
    k = d * (d - 1) // 2
    aic, bic = _aic_bic(ll, k, n)
    return CopulaResult("gaussian", {"correlation_matrix": R.tolist()}, ll, aic, bic, n, d)


def sample_gaussian_copula(result: CopulaResult, n_samples: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    R = np.asarray(result.params["correlation_matrix"])
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n_samples, R.shape[0])) @ L.T
    return stats.norm.cdf(Z)


# ─── Student-t Copula ─────────────────────────────────────────────────────────

def _kendall_to_t_corr(tau: np.ndarray) -> np.ndarray:
    # exact relationship for t-copula: ρ = sin(π τ / 2)
    return np.sin(np.pi / 2.0 * tau)


def _t_logpdf(u: np.ndarray, R: np.ndarray, nu: float) -> np.ndarray:
    z = stats.t.ppf(_clip(u), df=nu)
    d = R.shape[0]
    R_inv = np.linalg.inv(R)
    _, logdet = np.linalg.slogdet(R)
    quad_R = np.einsum('ni,ij,nj->n', z, R_inv, z)
    log_c = (
        gammaln((nu + d) / 2.0)
        - d * gammaln((nu + 1.0) / 2.0)
        + (d - 1) * gammaln(nu / 2.0)
        - 0.5 * logdet
        - ((nu + d) / 2.0) * np.log(1.0 + quad_R / nu)
        + ((nu + 1.0) / 2.0) * np.sum(np.log(1.0 + z ** 2 / nu), axis=1)
    )
    return log_c


def _fit_nu(u: np.ndarray, R: np.ndarray) -> float:
    result = optimize.minimize_scalar(
        lambda log_nu: -float(np.sum(_t_logpdf(u, R, float(np.exp(log_nu))))),
        bounds=(np.log(2.1), np.log(120.0)),
        method='bounded',
    )
    return float(np.exp(result.x))


def fit_t_copula(u: np.ndarray) -> CopulaResult:
    u = _clip(np.asarray(u, dtype=float))
    n, d = u.shape
    R = _kendall_to_t_corr(_kendall_matrix(u))
    np.fill_diagonal(R, 1.0)
    if not _is_pd(R):
        R = _nearest_pd(R)
    nu = _fit_nu(u, R)
    ll = float(np.sum(_t_logpdf(u, R, nu)))
    k = d * (d - 1) // 2 + 1
    aic, bic = _aic_bic(ll, k, n)
    return CopulaResult("t", {"correlation_matrix": R.tolist(), "nu": nu}, ll, aic, bic, n, d)


def sample_t_copula(result: CopulaResult, n_samples: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    R = np.asarray(result.params["correlation_matrix"])
    nu = float(result.params["nu"])
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n_samples, R.shape[0])) @ L.T
    chi2 = rng.chisquare(nu, n_samples) / nu
    return stats.t.cdf(Z / np.sqrt(chi2[:, None]), df=nu)


# ─── Archimedean Copulas (Bivariate) ─────────────────────────────────────────

def _assert_bivariate(u: np.ndarray, name: str) -> None:
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(
            f"{name} copula 仅支持二元数据（d=2），"
            f"当前 shape={u.shape}"
        )


# ── Gumbel ────────────────────────────────────────────────────────────────────

def _gumbel_logpdf(u: np.ndarray, theta: float) -> np.ndarray:
    u = _clip(u)
    a = -np.log(u[:, 0])
    b = -np.log(u[:, 1])
    s = a ** theta + b ** theta
    t = s ** (1.0 / theta)
    bracket = np.clip(t + theta - 1.0, _CLIP_EPS, None)
    return (
        -t
        + a + b
        + (theta - 1.0) * (np.log(a) + np.log(b))
        + (1.0 / theta - 2.0) * np.log(s)
        + np.log(bracket)
    )


def _gumbel_cond_cdf(u2: float, u1: float, theta: float) -> float:
    # P(U2 <= u2 | U1 = u1) = dC/du1
    a = -np.log(np.clip(u1, _CLIP_EPS, 1.0 - _CLIP_EPS))
    b = -np.log(np.clip(u2, _CLIP_EPS, 1.0 - _CLIP_EPS))
    s = a ** theta + b ** theta
    t = s ** (1.0 / theta)
    return float(np.exp(-t + a) * (a ** (theta - 1.0)) * (s ** (1.0 / theta - 1.0)))


def fit_gumbel_copula(u: np.ndarray) -> CopulaResult:
    u = _clip(np.asarray(u, dtype=float))
    _assert_bivariate(u, "Gumbel")
    n = u.shape[0]
    tau, _ = stats.kendalltau(u[:, 0], u[:, 1])
    theta0 = max(1.001, 1.0 / (1.0 - float(np.clip(tau, 0.01, 0.99))))
    res = optimize.minimize(
        lambda t: -float(np.sum(_gumbel_logpdf(u, float(t[0])))),
        x0=[theta0], bounds=[(1.001, None)], method='L-BFGS-B',
    )
    theta = float(res.x[0])
    ll = float(np.sum(_gumbel_logpdf(u, theta)))
    aic, bic = _aic_bic(ll, 1, n)
    return CopulaResult("gumbel", {"theta": theta}, ll, aic, bic, n, 2)


def sample_gumbel_copula(result: CopulaResult, n_samples: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    theta = float(result.params["theta"])
    u1 = rng.uniform(_CLIP_EPS, 1.0 - _CLIP_EPS, n_samples)
    q  = rng.uniform(_CLIP_EPS, 1.0 - _CLIP_EPS, n_samples)
    u2 = np.empty(n_samples)
    for i in range(n_samples):
        f = lambda v: _gumbel_cond_cdf(v, u1[i], theta) - q[i]
        u2[i] = optimize.brentq(f, _CLIP_EPS, 1.0 - _CLIP_EPS, xtol=1e-8, maxiter=200)
    return np.column_stack([u1, u2])


# ── Clayton ───────────────────────────────────────────────────────────────────

def _clayton_logpdf(u: np.ndarray, theta: float) -> np.ndarray:
    u = _clip(u)
    x, y = u[:, 0], u[:, 1]
    inner = np.clip(x ** (-theta) + y ** (-theta) - 1.0, _CLIP_EPS, None)
    return (
        np.log(theta + 1.0)
        - (theta + 1.0) * (np.log(x) + np.log(y))
        - (2.0 + 1.0 / theta) * np.log(inner)
    )


def fit_clayton_copula(u: np.ndarray) -> CopulaResult:
    u = _clip(np.asarray(u, dtype=float))
    _assert_bivariate(u, "Clayton")
    n = u.shape[0]
    tau, _ = stats.kendalltau(u[:, 0], u[:, 1])
    tau = float(np.clip(tau, 0.01, 0.99))
    theta0 = max(0.001, 2.0 * tau / (1.0 - tau))
    res = optimize.minimize(
        lambda t: -float(np.sum(_clayton_logpdf(u, float(t[0])))),
        x0=[theta0], bounds=[(1e-4, None)], method='L-BFGS-B',
    )
    theta = float(res.x[0])
    ll = float(np.sum(_clayton_logpdf(u, theta)))
    aic, bic = _aic_bic(ll, 1, n)
    return CopulaResult("clayton", {"theta": theta}, ll, aic, bic, n, 2)


def sample_clayton_copula(result: CopulaResult, n_samples: int, rng=None) -> np.ndarray:
    # Closed-form conditional inversion: McNeil et al. (2005)
    rng = np.random.default_rng(rng)
    theta = float(result.params["theta"])
    u1 = rng.uniform(_CLIP_EPS, 1.0 - _CLIP_EPS, n_samples)
    q  = rng.uniform(_CLIP_EPS, 1.0 - _CLIP_EPS, n_samples)
    u2 = np.clip(
        (1.0 + u1 ** (-theta) * (q ** (-theta / (theta + 1.0)) - 1.0)) ** (-1.0 / theta),
        _CLIP_EPS, 1.0 - _CLIP_EPS,
    )
    return np.column_stack([u1, u2])


# ── Frank ─────────────────────────────────────────────────────────────────────

def _frank_logpdf(u: np.ndarray, theta: float) -> np.ndarray:
    if abs(theta) < 1e-8:
        return np.zeros(len(u))
    u = _clip(u)
    x, y = u[:, 0], u[:, 1]
    em    = np.expm1(-theta)           # e^{-θ} - 1
    ex    = np.expm1(-theta * x)       # e^{-θx} - 1
    ey    = np.expm1(-theta * y)       # e^{-θy} - 1
    denom = em + ex * ey
    log_d = np.log(np.maximum(np.abs(denom), _CLIP_EPS))
    return np.log(abs(theta)) + np.log(abs(em)) - theta * (x + y) - 2.0 * log_d


def fit_frank_copula(u: np.ndarray) -> CopulaResult:
    u = _clip(np.asarray(u, dtype=float))
    _assert_bivariate(u, "Frank")
    n = u.shape[0]
    tau, _ = stats.kendalltau(u[:, 0], u[:, 1])
    tau = float(np.clip(tau, -0.99, 0.99))
    theta0 = 4.0 * tau / (1.0 - abs(tau)) if abs(tau) > 0.01 else 0.5
    res = optimize.minimize(
        lambda t: -float(np.sum(_frank_logpdf(u, float(t[0])))),
        x0=[theta0], bounds=[(-50.0, 50.0)], method='L-BFGS-B',
    )
    theta = float(res.x[0])
    ll = float(np.sum(_frank_logpdf(u, theta)))
    aic, bic = _aic_bic(ll, 1, n)
    return CopulaResult("frank", {"theta": theta}, ll, aic, bic, n, 2)


def sample_frank_copula(result: CopulaResult, n_samples: int, rng=None) -> np.ndarray:
    # Closed-form conditional inversion
    rng = np.random.default_rng(rng)
    theta = float(result.params["theta"])
    u1 = rng.uniform(_CLIP_EPS, 1.0 - _CLIP_EPS, n_samples)
    q  = rng.uniform(_CLIP_EPS, 1.0 - _CLIP_EPS, n_samples)
    D = np.expm1(-theta)              # e^{-θ} - 1
    E = np.exp(-theta * u1)           # e^{-θ u1}
    # u2 = -1/θ · ln(1 + q·D / (E·(1-q) + q))
    denom_q = E * (1.0 - q) + q
    inner = np.clip(1.0 + q * D / denom_q, _CLIP_EPS, None)
    u2 = np.clip(-1.0 / theta * np.log(inner), _CLIP_EPS, 1.0 - _CLIP_EPS)
    return np.column_stack([u1, u2])


# ─── Dispatchers ──────────────────────────────────────────────────────────────

_FITTERS = {
    "gaussian": fit_gaussian_copula,
    "t":        fit_t_copula,
    "gumbel":   fit_gumbel_copula,
    "clayton":  fit_clayton_copula,
    "frank":    fit_frank_copula,
}

_SAMPLERS = {
    "gaussian": sample_gaussian_copula,
    "t":        sample_t_copula,
    "gumbel":   sample_gumbel_copula,
    "clayton":  sample_clayton_copula,
    "frank":    sample_frank_copula,
}


def fit_copula(u: np.ndarray, copula_type: str = "gaussian") -> CopulaResult:
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError("fit_copula 要求输入形状为 (n_samples, n_vars)")
    if copula_type not in _FITTERS:
        raise ValueError(
            f"未知 copula 类型: '{copula_type}'。"
            f"支持的类型: {sorted(_FITTERS)}"
        )
    return _FITTERS[copula_type](u)


def sample_from_copula(
    result: CopulaResult,
    n_samples: int,
    rng=None,
) -> np.ndarray:
    if result.copula_type not in _SAMPLERS:
        raise ValueError(f"不支持从 '{result.copula_type}' copula 中采样")
    return _SAMPLERS[result.copula_type](result, n_samples, rng)
