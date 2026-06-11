from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def render_placeholder_figure(message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#666666")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
