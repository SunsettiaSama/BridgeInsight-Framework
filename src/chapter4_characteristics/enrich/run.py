from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.chapter4_characteristics._bootstrap import ensure_paths
from src.chapter4_characteristics.analysis.index_builder import build_others_index
from src.chapter4_characteristics.analysis.reference_builder import post_enrich_artifacts
from src.chapter4_characteristics.settings import (
    get_enriched_dir,
    get_predictions_enriched_path,
    load_config,
)

ensure_paths()

from src.identifier.feature_analysis.run import run as feature_analysis_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _apply_limit_to_enriched(data: dict, limit: int | None) -> dict:
    if limit is None or limit <= 0:
        return data
    preds = data.get("predictions", {})
    keys = sorted(preds.keys(), key=lambda k: int(k))[: int(limit)]
    key_set = set(keys)
    data["predictions"] = {k: preds[k] for k in keys}
    meta = data.get("sample_metadata", {})
    data["sample_metadata"] = {k: meta[k] for k in keys if k in meta}
    data["metadata"]["limit"] = limit
    return data


def run_enrichment(
    limit: int | None = None,
    config_path: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    if limit is None and int(cfg.get("dev_limit_samples", 0)) > 0:
        limit = int(cfg["dev_limit_samples"])

    enriched_src = get_predictions_enriched_path(cfg)
    if not enriched_src.exists():
        raise FileNotFoundError(f"识别结果不存在，请先 infer：{enriched_src}")

    with open(enriched_src, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)

    enriched_data = _apply_limit_to_enriched(enriched_data, limit)
    tmp_path = enriched_src.parent / "predictions_enriched_work.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False, indent=2)

    out_dir = get_enriched_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始特征归档 → {out_dir}")
    feature_config = cfg.get("feature_analysis_config")
    feature_analysis_run(
        result_path=str(tmp_path),
        wind_metadata_path=str(cfg["wind_metadata_path"]),
        output_dir=str(out_dir),
        config_yaml=str(feature_config) if feature_config else None,
    )

    post_enrich_artifacts(cfg)
    build_others_index(cfg)

    logger.info("归档完成")
    return {"output_dir": str(out_dir), "limit": limit}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chapter4 特征归档")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    run_enrichment(limit=args.limit, config_path=args.config)


if __name__ == "__main__":
    main()
