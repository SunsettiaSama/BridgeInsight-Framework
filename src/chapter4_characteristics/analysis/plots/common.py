from __future__ import annotations

import io
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.chapter4_characteristics.analysis.data_loader import get_nested

SCATTER_MAX = 50_000


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _downsample(x: np.ndarray, y: np.ndarray, max_n: int = SCATTER_MAX):
    n = len(x)
    if n <= max_n:
        return x, y
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=max_n, replace=False)
    return x[idx], y[idx]


def _plane_rms(samples: List[dict]) -> tuple[np.ndarray, np.ndarray]:
    rin, rout = [], []
    for s in samples:
        a = get_nested(s, "time_stats_inplane.rms")
        b = get_nested(s, "time_stats_outplane.rms")
        if a is not None and b is not None:
            rin.append(float(a))
            rout.append(float(b))
    return np.asarray(rin), np.asarray(rout)


def plot_rms_histogram(samples: List[dict], title: str = "RMS 分布") -> bytes:
    rin, rout = _plane_rms(samples)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(rin):
        ax.hist(rin, bins=80, alpha=0.6, label="面内", density=True)
    if len(rout):
        ax.hist(rout, bins=80, alpha=0.6, label="面外", density=True)
    ax.set_xlabel("RMS")
    ax.set_ylabel("密度")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_rms_scatter(samples: List[dict], title: str = "面内外 RMS") -> bytes:
    rin, rout = _plane_rms(samples)
    x, y = _downsample(rout, rin)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=4, alpha=0.25)
    ax.set_xlabel("面外 RMS")
    ax.set_ylabel("面内 RMS")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_kurtosis_histogram(samples: List[dict], title: str = "峭度分布") -> bytes:
    kin, kout = [], []
    for s in samples:
        a = get_nested(s, "time_stats_inplane.kurtosis")
        b = get_nested(s, "time_stats_outplane.kurtosis")
        if a is not None:
            kin.append(float(a))
        if b is not None:
            kout.append(float(b))
    fig, ax = plt.subplots(figsize=(8, 4))
    if kin:
        ax.hist(kin, bins=60, alpha=0.6, label="面内", density=True)
    if kout:
        ax.hist(kout, bins=60, alpha=0.6, label="面外", density=True)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_freq_histogram(samples: List[dict], title: str = "主频分布") -> bytes:
    vals = []
    for s in samples:
        freqs = (s.get("psd_inplane") or {}).get("frequencies") or []
        if freqs:
            vals.append(float(freqs[0]))
    fig, ax = plt.subplots(figsize=(8, 4))
    if vals:
        ax.hist(vals, bins=60, alpha=0.75, density=True)
    ax.set_xlabel("主频 (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_freq_energy_scatter(samples: List[dict], title: str = "主频-能量占比") -> bytes:
    fx, ey = [], []
    for s in samples:
        psd = s.get("psd_inplane") or {}
        freqs = psd.get("frequencies") or []
        powers = psd.get("powers") or []
        if freqs and powers:
            total = float(np.sum(powers))
            if total > 0:
                fx.append(float(freqs[0]))
                ey.append(float(powers[0]) / total)
    x, y = _downsample(np.asarray(fx), np.asarray(ey))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=5, alpha=0.3)
    ax.set_xlabel("主频 (Hz)")
    ax.set_ylabel("主频能量占比")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_energy_histogram(samples: List[dict], title: str = "主频能量占比") -> bytes:
    vals = []
    for s in samples:
        psd = s.get("psd_inplane") or {}
        powers = psd.get("powers") or []
        if powers:
            total = float(np.sum(powers))
            if total > 0:
                vals.append(float(powers[0]) / total)
    fig, ax = plt.subplots(figsize=(8, 4))
    if vals:
        ax.hist(vals, bins=60, alpha=0.75, density=True)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_energy_cumsum(samples: List[dict], title: str = "前10阶累积能量") -> bytes:
    curves = []
    for s in samples:
        psd = s.get("psd_inplane") or {}
        powers = psd.get("powers") or []
        if len(powers) >= 2:
            p = np.asarray(powers[:10], dtype=np.float64)
            total = p.sum()
            if total > 0:
                curves.append(np.cumsum(p / total))
    fig, ax = plt.subplots(figsize=(7, 4))
    if curves:
        arr = np.stack([np.pad(c, (0, 10 - len(c)), constant_values=np.nan) for c in curves])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = np.arange(1, 11)
        ax.plot(x, mean, "b-", lw=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("模态阶次")
    ax.set_ylabel("累积能量占比")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_trajectory_cloud(samples: List[dict], title: str = "轨迹云图") -> bytes:
    xs, ys = [], []
    for s in samples:
        # 用窗口 RMS 近似轨迹点（无原始信号时）
        a = get_nested(s, "time_stats_inplane.rms")
        b = get_nested(s, "time_stats_outplane.rms")
        if a is not None and b is not None:
            xs.append(float(a))
            ys.append(float(b))
    x, y = _downsample(np.asarray(xs), np.asarray(ys))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=3, alpha=0.2)
    ax.set_xlabel("面内 RMS")
    ax.set_ylabel("面外 RMS")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_ellipticity_hist(samples: List[dict], title: str = "椭圆率分布") -> bytes:
    vals = []
    for s in samples:
        v = get_nested(s, "cross_coupling.ellipticity")
        if v is not None:
            vals.append(float(v))
    fig, ax = plt.subplots(figsize=(8, 4))
    if vals:
        ax.hist(vals, bins=50, alpha=0.75, density=True)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_wind_rms_scatter(samples: List[dict], title: str = "风速-RMS") -> bytes:
    wx, ry = [], []
    for s in samples:
        ws = (s.get("wind_stats") or [{}])[0]
        u = ws.get("mean_wind_speed")
        r = get_nested(s, "time_stats_inplane.rms")
        if u is not None and r is not None:
            wx.append(float(u))
            ry.append(float(r))
    x, y = _downsample(np.asarray(wx), np.asarray(ry))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=5, alpha=0.3)
    ax.set_xlabel("平均风速 (m/s)")
    ax.set_ylabel("面内 RMS")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)


def plot_class_distribution(counts: dict, title: str = "类别分布") -> bytes:
    labels = ["Normal", "VIV", "RWIV", "Others"]
    vals = [counts.get(i, 0) for i in range(4)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylabel("样本数")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom")
    return fig_to_png_bytes(fig)


PLOT_FUNCS = {
    "rms_histogram": plot_rms_histogram,
    "rms_scatter": plot_rms_scatter,
    "kurtosis_histogram": plot_kurtosis_histogram,
    "freq_histogram": plot_freq_histogram,
    "freq_energy_scatter": plot_freq_energy_scatter,
    "energy_histogram": plot_energy_histogram,
    "energy_cumsum": plot_energy_cumsum,
    "trajectory_cloud": plot_trajectory_cloud,
    "ellipticity_hist": plot_ellipticity_hist,
    "wind_rms_scatter": plot_wind_rms_scatter,
}


def render_plot(plot_id: str, samples: List[dict], extra: Optional[dict] = None) -> bytes:
    if plot_id == "class_distribution" and extra and "class_counts" in extra:
        return plot_class_distribution(extra["class_counts"])
    fn = PLOT_FUNCS.get(plot_id)
    if fn is None:
        raise ValueError(f"未知 plot_id={plot_id}")
    return fn(samples)
