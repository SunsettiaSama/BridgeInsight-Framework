"""
图4-x：窗间时序 Copula 探索图（不占正式图号）。

每类一组：
  0  lag-1 分量 Spearman（X_t,j vs X_{t+1},j）
  1  椭圆 Copula AIC（拟合 [U_t; U_{t+1}]）
  2  MC 轨迹示意（一条 path 的若干核心特征）

依赖：
  python -m src.chapter4_characteristics td_copula run --class-id all
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
from src.chapter4_characteristics.statistics.td_copula.fit import td_fit_path
from src.chapter4_characteristics.statistics.td_copula.sample import td_sample_path
from src.figure_paintings.figs_for_thesis.Chapter4._copula_fig_style import paper_label
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import push as web_push


def _load_fit(cfg: dict, class_id: int) -> dict:
    path = td_fit_path(cfg, class_id)
    if not path.exists():
        raise FileNotFoundError(f"缺少 td 拟合结果：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_lag1_spearman(payload: dict, class_id: int) -> plt.Figure:
    names = payload["variable_names"]
    diag = np.asarray(payload["lag1_spearman_diag"], dtype=np.float64)
    title = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
    palette = list(get_full_color_map(style="discrete").colors)
    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)
    x = np.arange(len(names))
    ax.bar(x, diag, color=palette[0], edgecolor="none")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([paper_label(n) for n in names], fontsize=FONT_SIZE - 10, rotation=45)
    ax.set_ylabel(r"Spearman $\rho(X_t, X_{t+1})$", fontsize=FONT_SIZE - 6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        f"图4-x lag-1 分量相关 | {title}",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 2,
        pad=12,
    )
    ax.tick_params(labelsize=FONT_SIZE - 8)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28, top=0.88)
    return fig


def plot_aic(payload: dict, class_id: int) -> plt.Figure:
    title = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
    comparison = payload.get("comparison", [])
    types = [c["copula_type"] for c in comparison]
    aics = [c["aic"] for c in comparison]
    best = payload.get("best_copula_type")
    palette = list(get_full_color_map(style="discrete").colors)
    bar_colors = [palette[9] if t == best else palette[0] for t in types]
    fig, ax = plt.subplots(figsize=REC_FIG_SIZE)
    ax.bar(types, aics, color=bar_colors, edgecolor="none")
    ax.set_ylabel("AIC", fontsize=FONT_SIZE - 6)
    ax.set_title(
        f"图4-x 窗间椭圆 Copula AIC | {title}（优={best}）",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 2,
        pad=12,
    )
    ax.tick_params(labelsize=FONT_SIZE - 8)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.88)
    return fig


def plot_mc_path(cfg: dict, payload: dict, class_id: int) -> plt.Figure:
    title = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
    path = td_sample_path(cfg, class_id)
    if not path.exists():
        raise FileNotFoundError(f"缺少 MC 轨迹：{path}；请先 td_copula sample")
    data = np.load(path, allow_pickle=False)
    paths = np.asarray(data["paths"], dtype=np.float64)
    names = payload["variable_names"]
    # 取第 0 条轨迹，画 f1_in / E1_in / E1_out
    show = []
    for key in ("freq_in_1", "energy_in_1", "energy_out_1"):
        if key in names:
            show.append(names.index(key))
    if not show:
        show = list(range(min(3, len(names))))

    fig, axes = plt.subplots(len(show), 1, figsize=REC_FIG_SIZE, sharex=True)
    if len(show) == 1:
        axes = [axes]
    palette = list(get_full_color_map(style="discrete").colors)
    line_colors = [palette[0], palette[9], palette[7]]
    t = np.arange(paths.shape[1])
    for ax, j, col in zip(axes, show, line_colors):
        ax.plot(t, paths[0, :, j], color=col, lw=1.5)
        ax.set_ylabel(paper_label(names[j]), fontsize=FONT_SIZE - 8)
        ax.tick_params(labelsize=FONT_SIZE - 10)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("window step", fontsize=FONT_SIZE - 8)
    axes[0].set_title(
        f"图4-x MC 轨迹示意（path 0）| {title}",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 2,
        pad=10,
    )
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.12, top=0.90, hspace=0.15)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="窗间时序 Copula 探索图")
    parser.add_argument("--class-id", type=str, default="all")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.class_id.strip().lower() == "all":
        class_ids = [0, 1, 2, 3]
    else:
        class_ids = [int(args.class_id)]

    for class_id in class_ids:
        payload = _load_fit(cfg, class_id)
        name = payload.get("class_name") or DEFAULT_LABEL_NAMES[class_id]
        page = f"fig4_x tdCopula|{name}"
        print(f"绘制 class={class_id} ({name}) → 页 {page} ...")
        figures = [
            (plot_lag1_spearman(payload, class_id), "lag1 Spearman"),
            (plot_aic(payload, class_id), "窗间Copula AIC"),
            (plot_mc_path(cfg, payload, class_id), "MC轨迹示意"),
        ]
        for slot, (fig, title) in enumerate(figures):
            web_push(fig, page=page, slot=slot, title=title, page_cols=2)
            plt.close(fig)
        print(f"  已推送 {len(figures)} 张")
    print("OK 已分图推送：fig4_x tdCopula|*")


if __name__ == "__main__":
    main()
