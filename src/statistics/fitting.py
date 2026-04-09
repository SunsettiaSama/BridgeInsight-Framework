from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    form: str
    params: Dict[str, float]

    # 分布拟合优度
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None

    # 曲线拟合优度
    r_squared: Optional[float] = None
    residuals: Optional[np.ndarray] = field(default=None, repr=False)
    param_std_errors: Optional[Dict[str, float]] = None

    def __str__(self) -> str:
        lines = [f"FitResult(form='{self.form}')"]
        lines.append("  params:")
        for k, v in self.params.items():
            lines.append(f"    {k} = {v:.6g}")
        if self.ks_statistic is not None:
            lines.append(f"  KS statistic = {self.ks_statistic:.4f},  p-value = {self.ks_pvalue:.4f}")
        if self.aic is not None:
            lines.append(f"  AIC = {self.aic:.4f},  BIC = {self.bic:.4f}")
        if self.r_squared is not None:
            lines.append(f"  R² = {self.r_squared:.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 内置曲线形式
# ---------------------------------------------------------------------------

_CURVE_FUNCS: Dict[str, Callable] = {
    "linear":      lambda x, a, b:          a * x + b,
    "power":       lambda x, a, b:          a * np.power(np.abs(x), b),
    "exponential": lambda x, a, b:          a * np.exp(b * x),
    "logarithmic": lambda x, a, b:          a * np.log(np.abs(x) + 1e-12) + b,
    "quadratic":   lambda x, a, b, c:       a * x**2 + b * x + c,
    "cubic":       lambda x, a, b, c, d:    a * x**3 + b * x**2 + c * x + d,
    "sine":        lambda x, a, b, c, d:    a * np.sin(b * x + c) + d,
}

_CURVE_PARAM_NAMES: Dict[str, List[str]] = {
    "linear":      ["a", "b"],
    "power":       ["a", "b"],
    "exponential": ["a", "b"],
    "logarithmic": ["a", "b"],
    "quadratic":   ["a", "b", "c"],
    "cubic":       ["a", "b", "c", "d"],
    "sine":        ["a", "b", "c", "d"],
}


# ---------------------------------------------------------------------------
# 分布拟合
# ---------------------------------------------------------------------------

def fit_distribution(
    data: Sequence[float],
    distribution: str = "norm",
    floc: Optional[float] = None,
    fscale: Optional[float] = None,
) -> FitResult:
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("fit_distribution 仅接受一维数据")
    if len(data) < 2:
        raise ValueError("数据点数不足，至少需要 2 个")

    dist_obj = getattr(stats, distribution, None)
    if dist_obj is None or not isinstance(dist_obj, stats.rv_continuous):
        raise ValueError(
            f"未知连续分布: '{distribution}'。"
            f"请使用 scipy.stats 中的分布名称（如 norm / lognorm / weibull_min / gamma / expon）"
        )

    fit_kwargs: Dict = {}
    if floc is not None:
        fit_kwargs["floc"] = floc
    if fscale is not None:
        fit_kwargs["fscale"] = fscale

    fitted_params = dist_obj.fit(data, **fit_kwargs)

    shape_names: List[str] = (
        [s.strip() for s in dist_obj.shapes.split(",")] if dist_obj.shapes else []
    )
    param_names = shape_names + ["loc", "scale"]
    params = dict(zip(param_names, fitted_params))

    ks_stat, ks_p = stats.kstest(data, distribution, args=fitted_params)

    log_l = float(np.sum(dist_obj.logpdf(data, *fitted_params)))
    k = len(fitted_params)
    n = len(data)
    aic = 2 * k - 2 * log_l
    bic = k * np.log(n) - 2 * log_l

    return FitResult(
        form=distribution,
        params=params,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        aic=aic,
        bic=bic,
    )


# ---------------------------------------------------------------------------
# 曲线拟合
# ---------------------------------------------------------------------------

def fit_curve(
    x: Sequence[float],
    y: Sequence[float],
    form: str = "linear",
    func: Optional[Callable] = None,
    param_names: Optional[Sequence[str]] = None,
    p0: Optional[Sequence[float]] = None,
    **curve_fit_kwargs,
) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x 与 y 须为相同长度的一维数组")

    if func is not None:
        fn = func
        n_params = fn.__code__.co_argcount - 1
        pnames: List[str] = list(param_names) if param_names else [f"p{i}" for i in range(n_params)]
    elif form in _CURVE_FUNCS:
        fn = _CURVE_FUNCS[form]
        pnames = _CURVE_PARAM_NAMES[form]
    else:
        raise ValueError(
            f"未知拟合形式: '{form}'。"
            f"内置形式: {list(_CURVE_FUNCS)}。"
            f"也可通过 func= 传入自定义可调用对象。"
        )

    popt, pcov = optimize.curve_fit(fn, x, y, p0=p0, **curve_fit_kwargs)

    params = dict(zip(pnames, popt.tolist()))

    perr = np.sqrt(np.diag(pcov))
    param_std_errors = dict(zip(pnames, perr.tolist()))

    y_pred = fn(x, *popt)
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return FitResult(
        form=form,
        params=params,
        r_squared=r_squared,
        residuals=residuals,
        param_std_errors=param_std_errors,
    )


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def fit(
    data: Sequence[float],
    form: str,
    x: Optional[Sequence[float]] = None,
    func: Optional[Callable] = None,
    param_names: Optional[Sequence[str]] = None,
    p0: Optional[Sequence[float]] = None,
    **kwargs,
) -> FitResult:
    """
    统一拟合入口，根据是否提供 x 自动分派：

    - 仅提供 data         → 概率分布拟合，form 为 scipy.stats 分布名称
    - 同时提供 x 和 data  → 曲线拟合（data 视为 y），form 为内置曲线名称或通过 func 自定义

    Parameters
    ----------
    data : array-like
        待拟合数据（分布拟合时为样本集合；曲线拟合时为因变量 y）
    form : str
        拟合形式。分布拟合时：scipy.stats 分布名（如 "norm"、"lognorm"、"weibull_min"）；
        曲线拟合时：内置形式（"linear"/"power"/"exponential"/"logarithmic"/"quadratic"/"cubic"/"sine"）
    x : array-like, optional
        自变量序列；提供时触发曲线拟合
    func : callable, optional
        自定义曲线函数 f(x, *params)；提供时优先于 form 中的内置函数
    param_names : sequence of str, optional
        自定义函数的参数名称列表，与 func 参数位置对应
    p0 : sequence of float, optional
        曲线拟合初始猜测值
    **kwargs
        透传给 scipy.optimize.curve_fit（曲线拟合）或 scipy.stats.dist.fit（分布拟合）

    Returns
    -------
    FitResult
        包含拟合参数及拟合质量指标的结果对象

    Examples
    --------
    # 分布拟合：对所有 VIV 样本的 RMS 拟合正态分布
    >>> result = fit(rms_values, form="norm")
    >>> result.params  # {'loc': ..., 'scale': ...}

    # 分布拟合：Weibull 分布
    >>> result = fit(wind_speeds, form="weibull_min")
    >>> result.params  # {'c': ..., 'loc': ..., 'scale': ...}

    # 曲线拟合：幂函数拟合 RMS vs 风速
    >>> result = fit(rms_values, form="power", x=wind_speeds)
    >>> result.params  # {'a': ..., 'b': ...}

    # 曲线拟合：自定义函数
    >>> import numpy as np
    >>> result = fit(y, form="custom", x=x,
    ...              func=lambda x, a, b, c: a * np.exp(-b * x) + c,
    ...              param_names=["a", "b", "c"])
    """
    if x is not None:
        return fit_curve(
            x=x,
            y=data,
            form=form,
            func=func,
            param_names=param_names,
            p0=p0,
            **kwargs,
        )
    return fit_distribution(data=data, distribution=form, **kwargs)
