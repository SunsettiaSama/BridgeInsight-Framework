"""
图4-27：VIV Copula 边缘概率密度（面内+面外，前 24 阶主频/能量）。

依赖：
  python -m src.chapter4_characteristics copula extract --class-id 1
  python -m src.chapter4_characteristics copula marginals --class-id 1
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
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_x_copula_marginals_grid import (
    _load_payload,
    plot_class_marginal_grid,
)
from src.visualize_tools.web_dashboard import push as web_push

CLASS_ID = 1
FIG_TAG = "图4-27"
WEB_PAGE = "fig4_27 VIV边缘密度"


def main() -> None:
    parser = argparse.ArgumentParser(description=FIG_TAG)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"{FIG_TAG}：绘制 VIV 边缘密度...")
    matrix, var_names, marg = _load_payload(cfg, CLASS_ID)
    fig = plot_class_marginal_grid(
        matrix, var_names, marg, CLASS_ID, fig_tag=FIG_TAG
    )
    web_push(
        fig,
        page=WEB_PAGE,
        slot=0,
        title="VIV边缘 PDF",
        page_cols=1,
    )
    plt.close(fig)
    print(f"OK 已推送：{WEB_PAGE}")


if __name__ == "__main__":
    main()
