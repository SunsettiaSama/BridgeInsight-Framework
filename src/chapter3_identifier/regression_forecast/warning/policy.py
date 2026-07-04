from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WARNING_LEVELS = ("normal", "attention", "warning", "severe")

WARNING_LEVEL_LABELS = {
    "normal": "正常",
    "attention": "关注",
    "warning": "预警",
    "severe": "严重",
}


@dataclass(frozen=True)
class WarningPolicy:
    attention: float = 0.30
    warning: float = 0.50
    severe: float = 0.75
    monitor_classes: tuple[str, ...] = ("VIV", "RWIV", "Others")
    show_normal_in_queue: bool = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "WarningPolicy":
        raw = cfg.get("warning_thresholds") or {}
        monitor = tuple(str(x) for x in cfg.get("monitor_classes", ["VIV", "RWIV", "Others"]))
        return cls(
            attention=float(raw.get("attention", 0.30)),
            warning=float(raw.get("warning", 0.50)),
            severe=float(raw.get("severe", 0.75)),
            monitor_classes=monitor,
            show_normal_in_queue=bool(cfg.get("show_normal_in_queue", False)),
        )

    def level_from_score(self, score: float) -> str:
        value = float(score)
        if value >= self.severe:
            return "severe"
        if value >= self.warning:
            return "warning"
        if value >= self.attention:
            return "attention"
        return "normal"

    def level_label(self, level: str) -> str:
        return WARNING_LEVEL_LABELS.get(level, level)

    def levels_payload(self) -> list[dict[str, Any]]:
        return [
            {"id": "normal", "label": self.level_label("normal"), "min_score": 0.0},
            {"id": "attention", "label": self.level_label("attention"), "min_score": self.attention},
            {"id": "warning", "label": self.level_label("warning"), "min_score": self.warning},
            {"id": "severe", "label": self.level_label("severe"), "min_score": self.severe},
        ]
