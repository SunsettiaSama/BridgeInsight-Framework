"""
图4-37：RWIV 椭圆 Copula Spearman 相关热图。

依赖：
  python -m src.chapter4_characteristics copula joint --class-id 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.settings import load_config
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_x_copula_joint_explore import (
    _load_joint,
    plot_spearman_heatmap,
)
from src.visualize_tools.web_dashboard import push as web_push

CLASS_ID = 2
FIG_TAG = "图4-37"
WEB_PAGE = "fig4_37 RWIV联合分布"


def main() -> None:
    parser = argparse.ArgumentParser(description=FIG_TAG)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"{FIG_TAG}：绘制 RWIV Spearman 相关热图...")
    payload = _load_joint(cfg, CLASS_ID)
    fig = plot_spearman_heatmap(payload, CLASS_ID, fig_tag=FIG_TAG)
    web_push(
        fig,
        page=WEB_PAGE,
        slot=0,
        title="Spearman相关热图",
        page_cols=1,
    )
    plt.close(fig)
    print(f"OK 已推送：{WEB_PAGE}")


if __name__ == "__main__":
    main()
