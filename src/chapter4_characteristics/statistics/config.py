from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from src.chapter4_characteristics._bootstrap import resolve_path

_DEFAULT = Path(__file__).resolve().parent.parent / "config" / "statistics.yaml"


@dataclass
class StatisticsConfig:
    class_label: str = "class_0_normal"
    n_modes: int = 24
    candidate_dists_freq: List[str] = field(default_factory=lambda: ["gamma", "lognorm"])
    candidate_dists_energy: List[str] = field(default_factory=lambda: ["gamma", "lognorm"])
    enable_gmm: bool = True
    gmm_max_components: int = 3
    min_valid_samples: int = 30
    corr_max_n: int = 5000
    corr_rng_seed: int = 42
    output_subdir: str = "chapter4_characteristics/copula"
    output_filename: str = "marginals.json"

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(yaml_path: Optional[str] = None) -> StatisticsConfig:
    path = resolve_path(yaml_path) if yaml_path else _DEFAULT
    if not path.exists():
        return StatisticsConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return StatisticsConfig(**{k: v for k, v in raw.items() if k in StatisticsConfig.__dataclass_fields__})
