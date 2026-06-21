from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.chapter4_characteristics.analysis.copula_service import load_mode_matrix
from src.chapter4_characteristics.analysis.data_loader import load_class_samples
from src.chapter4_characteristics.settings import get_copula_dir


def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_comparison_bar(comparison: List[dict], title: str = "Copula AIC 比较") -> bytes:
    types = [c["copula_type"] for c in comparison]
    aics = [c["aic"] for c in comparison]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(types, aics, color="#4C72B0")
    ax.set_ylabel("AIC")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _fig_bytes(fig)


def plot_marginal_grid(matrix: np.ndarray, var_names: List[str], max_cols: int = 4) -> bytes:
    n_vars = min(8, matrix.shape[1])
    n_cols = min(max_cols, n_vars)
    n_rows = int(np.ceil(n_vars / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axes = np.atleast_2d(axes)
    for j in range(n_vars):
        r, c = divmod(j, n_cols)
        ax = axes[r, c]
        col = matrix[:, j]
        valid = col[np.isfinite(col) & (col > 0)]
        if len(valid):
            ax.hist(valid, bins=30, density=True, alpha=0.6, color="#4C72B0")
            if len(valid) > 10:
                shape, loc, scale = stats.gamma.fit(valid, floc=0)
                xs = np.linspace(valid.min(), valid.max(), 100)
                ax.plot(xs, stats.gamma.pdf(xs, shape, loc, scale), "r-", lw=1.5)
        ax.set_title(var_names[j][:12], fontsize=8)
    for j in range(n_vars, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")
    fig.suptitle("边缘分布拟合", fontsize=11)
    fig.tight_layout()
    return _fig_bytes(fig)


def plot_contour(u_matrix: np.ndarray, var_x: int = 0, var_y: int = 1, title: str = "Copula 等高线 (PIT)") -> bytes:
    x = u_matrix[:, var_x]
    y = u_matrix[:, var_y]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=4, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"u_{var_x}")
    ax.set_ylabel(f"u_{var_y}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _fig_bytes(fig)


def load_copula_result(cfg: dict, class_id: int) -> Optional[dict]:
    path = get_copula_dir(cfg) / f"class_{class_id}_copula.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
