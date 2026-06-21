from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FigureService:
    def render_png(self, record: dict[str, Any], figure_name: str = "risk_curve", metric_name: str = "rms") -> bytes:
        horizons = record.get("horizons", [])
        x = [int(row.get("horizon_hours", 0)) for row in horizons]
        fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=110)
        if figure_name == "metric_curve":
            for class_name in ("VIV", "RWIV", "Others"):
                y = [
                    float(row.get("metrics_by_class", {}).get(class_name, {}).get(metric_name, 0.0))
                    for row in horizons
                ]
                ax.plot(x, y, marker="o", label=class_name)
            ax.set_ylabel(metric_name)
        else:
            for class_name in ("VIV", "RWIV", "Others"):
                y = [float(row.get("event_proba", {}).get(class_name, 0.0)) for row in horizons]
                ax.plot(x, y, marker="o", label=class_name)
            ax.set_ylabel("event probability")
            ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("horizon (h)")
        ax.set_title(f"sample {record.get('sample_idx')}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

