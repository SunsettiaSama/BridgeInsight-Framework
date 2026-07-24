"""
图4-x：Copula 边缘概率密度探索图（每类一张密集子图）。

- 96 维：面内/面外 × 主频/能量 × 24 阶（相依在联合图中体现）
- 面内蓝、面外淡红；标注用 f_k / E_k（面外加 ⊥）

依赖：
  python -m src.chapter4_characteristics copula extract --class-id all
  python -m src.chapter4_characteristics copula marginals --class-id all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.settings import DEFAULT_LABEL_NAMES, load_config
from src.chapter4_characteristics.statistics.mode_extract import (
    load_modes,
    matrix_from_arrays,
    modes_cache_path,
)
from src.chapter4_characteristics.statistics.pipeline import marginals_path
from src.figure_paintings.figs_for_thesis.Chapter4._copula_fig_style import (
    INPLANE_COLOR,
    OUTPLANE_COLOR,
    paper_label,
    parse_var_meta,
    plane_color,
    short_form,
)
from matplotlib.font_manager import FontProperties

from src.figure_paintings.figs_for_thesis.config import REC_FONT_SIZE
from src.visualize_tools.web_dashboard import push as web_push

# 长方形论文图字号；子图标题略小一档以免 96 格挤叠
_TITLE_FS = REC_FONT_SIZE
_SUB_FS = REC_FONT_SIZE - 8  # 16
_LEGEND_FS = REC_FONT_SIZE - 6  # 18
_CN_TITLE = FontProperties(family="SimSun", size=_TITLE_FS)
_CN_LEGEND = FontProperties(family="SimSun", size=_LEGEND_FS)


def _gmm_pdf(x: np.ndarray, params: dict) -> np.ndarray:
    weights = np.asarray(params["weights"], dtype=np.float64)
    means = np.asarray(params["means"], dtype=np.float64)
    variances = np.asarray(params["variances"], dtype=np.float64)
    pdf = np.zeros_like(x, dtype=np.float64)
    for w, m, v in zip(weights, means, variances):
        pdf += w * stats.norm.pdf(x, loc=m, scale=np.sqrt(max(v, 1e-30)))
    return pdf


def _pdf_curve(xs: np.ndarray, best: dict) -> np.ndarray:
    form = best["form"]
    params = best["params"]
    if form.startswith("gmm_"):
        return _gmm_pdf(xs, params)
    dist_obj = getattr(stats, form)
    return dist_obj.pdf(xs, *list(params.values()))


def _load_payload(cfg: dict, class_id: int) -> tuple[np.ndarray, list[str], dict]:
    n_modes = int(cfg.get("copula_n_modes", 24))
    nfft = int(cfg.get("copula_nfft", 128))
    path = modes_cache_path(cfg, class_id, n_modes=n_modes, nfft=nfft)
    freq_in, energy_in, freq_out, energy_out, _ = load_modes(path)
    matrix = matrix_from_arrays(freq_in, energy_in, freq_out, energy_out)

    marg_path = marginals_path(cfg, class_id)
    if not marg_path.exists():
        raise FileNotFoundError(f"缺少边缘结果：{marg_path}")
    with open(marg_path, "r", encoding="utf-8") as f:
        marg = json.load(f)
    return matrix, marg["variable_names"], marg


def plot_class_marginal_grid(
    matrix: np.ndarray,
    var_names: list[str],
    marg: dict,
    class_id: int,
    fig_tag: str = "图4-x",
) -> plt.Figure:
    n_vars = matrix.shape[1]
    n_cols = 8
    n_rows = int(np.ceil(n_vars / n_cols))
    # 子图略放大，容纳 REC_FONT_SIZE 量级标题
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows))
    axes = np.atleast_2d(axes)
    title = marg.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
    freq_limit_hz = float((marg.get("mode_config") or {}).get("freq_limit_hz", 25.0))

    for j in range(n_vars):
        r, c = divmod(j, n_cols)
        ax = axes[r, c]
        col = matrix[:, j]
        valid = col[np.isfinite(col) & (col > 0)]
        if len(valid) == 0:
            ax.axis("off")
            continue
        color = plane_color(var_names[j])
        kind, _, _ = parse_var_meta(var_names[j])
        if kind == "freq":
            x_min, x_max = 0.0, freq_limit_hz
            hist_kwargs = {"range": (x_min, x_max)}
        else:
            x_min, x_max = float(np.min(valid)), float(np.max(valid))
            hist_kwargs = {}
        ax.hist(
            valid,
            bins=28,
            density=True,
            alpha=0.62,
            color=color,
            edgecolor="none",
            **hist_kwargs,
        )
        entry = marg.get("marginals", {}).get(var_names[j])
        if entry and entry.get("best"):
            xs = np.linspace(x_min, x_max, 120)
            ys = _pdf_curve(xs, entry["best"])
            ax.plot(xs, ys, color="#222222", lw=1.15)
            form = short_form(entry["best"]["form"])
        else:
            form = "?"
        ax.set_title(
            f"{paper_label(var_names[j])}  {form}",
            fontsize=_SUB_FS,
        )
        ax.tick_params(labelsize=_SUB_FS - 6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_vars, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    forms = [
        (marg.get("marginals", {}).get(n) or {}).get("best", {}).get("form")
        for n in var_names
    ]
    n_gmm = sum(1 for f in forms if isinstance(f, str) and f.startswith("gmm_"))
    fig.suptitle(
        f"{fig_tag} Copula 边缘密度 | {title} | n={matrix.shape[0]}, d={n_vars}"
        f" | GMM {n_gmm}/{n_vars}（蓝=面内，淡红=面外；$f$ 主频，$E$ 能量，$\\perp$ 面外）",
        fontsize=_TITLE_FS,
        fontproperties=_CN_TITLE,
    )
    legend = fig.legend(
        handles=[
            Patch(facecolor=INPLANE_COLOR, edgecolor="none", label="面内"),
            Patch(facecolor=OUTPLANE_COLOR, edgecolor="none", label="面外"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.995),
        fontsize=_LEGEND_FS,
        framealpha=0.92,
        prop=_CN_LEGEND,
        ncol=2,
    )
    for t in legend.get_texts():
        t.set_fontproperties(_CN_LEGEND)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Copula 边缘密度探索图")
    parser.add_argument("--class-id", type=str, default="all")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.class_id.strip().lower() == "all":
        class_ids = [0, 1, 2, 3]
    else:
        class_ids = [int(args.class_id)]

    for class_id in class_ids:
        print(f"绘制 class={class_id} ...")
        matrix, var_names, marg = _load_payload(cfg, class_id)
        fig = plot_class_marginal_grid(matrix, var_names, marg, class_id, fig_tag="图4-x")
        name = marg.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
        web_push(
            fig,
            page="fig4_x Copula边缘密度",
            slot=class_id,
            title=f"{name} 边缘 PDF",
            page_cols=2,
        )
        plt.close(fig)
    print("OK 已推送：fig4_x Copula边缘密度")


if __name__ == "__main__":
    main()
