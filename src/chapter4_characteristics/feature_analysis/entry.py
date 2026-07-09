from __future__ import annotations

from pathlib import Path

from src.chapter4_characteristics.enrich.run import run_enrichment
from src.chapter4_characteristics.feature_analysis._compactor import ensure_class_dir_compacted
from src.chapter4_characteristics.settings import CLASS_DIRS, get_enriched_dir, load_config

EXCLUDED_C34_ANOMALY_SENSOR_IDS = {
    "ST-VIC-C34-201-01",
    "ST-VIC-C34-201-02",
    "ST-VIC-C34-202-01",
    "ST-VIC-C34-202-02",
    "ST-VIC-C34-301-01",
    "ST-VIC-C34-301-02",
}


def class_enriched_exists(class_id: int, config_path: str | None = None, output_dir: str | None = None) -> bool:
    cfg = load_config(config_path)
    enriched_dir = Path(output_dir) if output_dir else get_enriched_dir(cfg)
    class_dir = enriched_dir / CLASS_DIRS[class_id]
    flat_file = enriched_dir / f"{CLASS_DIRS[class_id]}.json"
    if flat_file.exists():
        return True
    return class_dir.exists() and any(class_dir.glob("*.json"))


def ensure_enriched_for_figures(
    class_id: int = 0,
    config_path: str | None = None,
    output_dir: str | None = None,
    batch_size: int = 512,
    exclude_c34_anomalies: bool = True,
) -> Path:
    cfg = load_config(config_path)
    enriched_dir = Path(output_dir) if output_dir else get_enriched_dir(cfg)
    class_dir = enriched_dir / CLASS_DIRS[class_id]
    excluded = EXCLUDED_C34_ANOMALY_SENSOR_IDS if exclude_c34_anomalies else set()

    if not class_enriched_exists(class_id, config_path=config_path, output_dir=str(enriched_dir)):
        run_enrichment(
            config_path=config_path,
            output_dir=str(enriched_dir),
            exclude_c34_anomalies=exclude_c34_anomalies,
            batch_size=batch_size,
            skip_artifacts=True,
        )

    ensure_class_dir_compacted(
        class_dir,
        cfg=cfg,
        excluded_sensor_ids=excluded,
    )

    if not class_enriched_exists(class_id, config_path=config_path, output_dir=str(enriched_dir)):
        raise FileNotFoundError(f"feature_analysis 未生成 class {class_id} enriched 数据：{enriched_dir}")
    return enriched_dir
