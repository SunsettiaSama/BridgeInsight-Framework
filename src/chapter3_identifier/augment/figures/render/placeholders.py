from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.chapter3_identifier.augment.settings import load_config

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_CFG = load_config()
_PLACEHOLDER_DPI = int(_CFG.get("figure_sample_dpi", 96))
_USE_TIGHT_BBOX = bool(_CFG.get("figure_export_tight_bbox", False))


def render_placeholder_figure(message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
        color="#666666",
        parse_math=False,
    )
    buf = io.BytesIO()
    if _USE_TIGHT_BBOX:
        fig.savefig(buf, format="png", dpi=_PLACEHOLDER_DPI, bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", dpi=_PLACEHOLDER_DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
