"""
图4-x：椭圆 Copula 联合探索图（拆分为独立图，成组推送）。

每类一组（同页多 slot）：
  0  Spearman 相关热图（96 维面内+面外相依）
  1  椭圆 Copula AIC 对比
  2–4  三组 PIT hexbin 密度图

标注：f_k / E_k（面外加 ⊥）；面内蓝、面外淡红仅用于边缘图，热图用块边界区分。

依赖：
  python -m src.chapter4_characteristics copula joint --class-id all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.settings import DEFAULT_LABEL_NAMES, load_config
from src.chapter4_characteristics.statistics.pipeline import copula_result_path
from src.figure_paintings.figs_for_thesis.Chapter4._copula_fig_style import (
    paper_label,
    paper_pair_label,
)
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    SQUARE_FIG_SIZE,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import push as web_push


def _load_joint(cfg: dict, class_id: int) -> dict:
    path = copula_result_path(cfg, class_id)
    if not path.exists():
        raise FileNotFoundError(f"缺少联合结果：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_cn(ax) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CN_FONT)
    ax.tick_params(labelsize=FONT_SIZE - 8)


def plot_spearman_heatmap(payload: dict, class_id: int, fig_tag: str = "图4-x") -> plt.Figure:
    spearman = np.asarray(payload["correlation"]["spearman"], dtype=np.float64)
    var_names = payload.get("variable_names", [])
    n_modes = int(payload.get("n_modes", 24))
    title = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
    d = spearman.shape[0]

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    im = ax.imshow(spearman, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # 刻度：每块取若干代表阶次，用论文符号；避开相邻分块边界的 24/1 重叠。
    tick_idx: list[int] = []
    for block in range(4):
        base = block * n_modes
        local_ticks = (0, 7, 15, 23) if block == 3 else (0, 7, 15)
        for local in local_ticks:
            idx = base + local
            if idx < d:
                tick_idx.append(idx)
    tick_idx = sorted(set(tick_idx))
    tick_labels = [paper_label(var_names[i]) for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels, fontsize=FONT_SIZE - 12, rotation=90)
    ax.set_yticklabels(tick_labels, fontsize=FONT_SIZE - 12)

    # 块边界：f^in | E^in | f^⊥ | E^⊥
    for b in (n_modes, 2 * n_modes, 3 * n_modes):
        if b < d:
            ax.axhline(b - 0.5, color="k", lw=0.8, alpha=0.5)
            ax.axvline(b - 0.5, color="k", lw=0.8, alpha=0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 10)
    ax.set_title(
        f"{fig_tag} Spearman ρ（面内+面外相依，d={d}）| {title}",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 4,
        pad=12,
    )
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.18, top=0.90)
    return fig


def plot_aic_bars(payload: dict, class_id: int, fig_tag: str = "图4-x") -> plt.Figure:
    comparison = payload.get("comparison", [])
    title = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
    accent = get_full_color_map(style="discrete").colors[class_id % 10]
    best = payload.get("best_copula_type", "?")

    types = [c["copula_type"] for c in comparison]
    aics = [c["aic"] for c in comparison]
    colors = [accent if t == best else "#9aa0a6" for t in types]

    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)
    ax.barh(types, aics, color=colors, height=0.55)
    ax.set_xlabel("AIC（越小越好）", fontproperties=CN_FONT, fontsize=FONT_SIZE - 4)
    ax.set_title(
        f"{fig_tag} 椭圆 Copula AIC | {title} | best={best}",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 4,
        pad=12,
    )
    _apply_cn(ax)
    ax.grid(True, axis="x", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.88)
    return fig


def plot_pit_hexbin(
    payload: dict,
    class_id: int,
    i: int,
    j: int,
    pair_label: str,
    seed: int,
    fig_tag: str = "图4-x",
) -> plt.Figure:
    u_sample = np.asarray(payload.get("u_matrix_sample", []), dtype=np.float64)
    var_names = payload.get("variable_names", [])
    title = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]

    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    if u_sample.size == 0 or i >= u_sample.shape[1] or j >= u_sample.shape[1]:
        ax.text(0.5, 0.5, "无 PIT 样本", ha="center", va="center", fontproperties=CN_FONT)
        ax.set_axis_off()
        return fig

    rng = np.random.default_rng(seed)
    ux = np.clip(u_sample[:, i] + rng.normal(0, 0.004, size=u_sample.shape[0]), 0, 1)
    uy = np.clip(u_sample[:, j] + rng.normal(0, 0.004, size=u_sample.shape[0]), 0, 1)
    hb = ax.hexbin(ux, uy, gridsize=32, cmap="viridis", mincnt=1, extent=(0, 1, 0, 1))
    ax.plot([0, 1], [0, 1], color="white", lw=1.0, alpha=0.55, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xlab = paper_label(var_names[i]) if i < len(var_names) else f"$u_{i}$"
    ylab = paper_label(var_names[j]) if j < len(var_names) else f"$u_{j}$"
    ax.set_xlabel(xlab, fontsize=FONT_SIZE - 4)
    ax.set_ylabel(ylab, fontsize=FONT_SIZE - 4)
    ax.set_title(
        f"{fig_tag} PIT 密度 | {pair_label} | {title}",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 4,
        pad=12,
    )
    ax.set_aspect("equal")
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 10)
    ax.tick_params(labelsize=FONT_SIZE - 8)
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.14, top=0.90)
    return fig


def build_class_figures(
    payload: dict,
    class_id: int,
    fig_tag: str = "图4-x",
) -> list[tuple[plt.Figure, str]]:
    n_modes = int(payload.get("n_modes", 24))
    var_names = payload.get("variable_names", [])
    pairs = [
        (0, 1),
        (0, n_modes),
        (n_modes, 2 * n_modes),
    ]
    figs: list[tuple[plt.Figure, str]] = [
        (plot_spearman_heatmap(payload, class_id, fig_tag=fig_tag), "Spearman相关热图"),
        (plot_aic_bars(payload, class_id, fig_tag=fig_tag), "椭圆Copula AIC"),
    ]
    for k, (i, j) in enumerate(pairs):
        lab = paper_pair_label(var_names[i], var_names[j])
        figs.append(
            (
                plot_pit_hexbin(
                    payload, class_id, i, j, lab, seed=42 + k, fig_tag=fig_tag
                ),
                f"PIT {lab}",
            )
        )
    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description="Copula 联合探索图（分图推送）")
    parser.add_argument("--class-id", type=str, default="all")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.class_id.strip().lower() == "all":
        class_ids = [0, 1, 2, 3]
    else:
        class_ids = [int(args.class_id)]

    for class_id in class_ids:
        payload = _load_joint(cfg, class_id)
        name = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
        page = f"fig4_x Copula联合|{name}"
        print(f"绘制 class={class_id} ({name}) → 页 {page} ...")
        figures = build_class_figures(payload, class_id)
        for slot, (fig, title) in enumerate(figures):
            web_push(
                fig,
                page=page,
                slot=slot,
                title=title,
                page_cols=2,
            )
            plt.close(fig)
        print(f"  已推送 {len(figures)} 张")
    print("OK 已分图推送：fig4_x Copula联合|*")


if __name__ == "__main__":
    main()
