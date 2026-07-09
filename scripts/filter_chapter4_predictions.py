from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / "results" / "chapter4_characteristics" / "inference"
SOURCE_PATH = INFERENCE_DIR / "predictions_enriched.json"
OUTPUT_PATH = INFERENCE_DIR / "predictions_enriched_exclude_c34_201_202_301.json"
EXCLUDED_SENSOR_IDS = {
    "ST-VIC-C34-201-01",
    "ST-VIC-C34-201-02",
    "ST-VIC-C34-202-01",
    "ST-VIC-C34-202-02",
    "ST-VIC-C34-301-01",
    "ST-VIC-C34-301-02",
}


def _is_excluded(meta: dict) -> bool:
    return (
        meta.get("inplane_sensor_id") in EXCLUDED_SENSOR_IDS
        or meta.get("outplane_sensor_id") in EXCLUDED_SENSOR_IDS
    )


def main() -> None:
    print(f"loading {SOURCE_PATH}", flush=True)
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", {})
    sample_metadata = data.get("sample_metadata", {})
    keep_keys: list[str] = []
    removed = 0
    for key in predictions:
        if _is_excluded(sample_metadata.get(str(key), {})):
            removed += 1
            continue
        keep_keys.append(key)

    print(
        f"filtering: total={len(predictions)} keep={len(keep_keys)} removed={removed}",
        flush=True,
    )
    data["predictions"] = {key: predictions[key] for key in keep_keys}
    data["sample_metadata"] = {
        key: sample_metadata[key] for key in keep_keys if key in sample_metadata
    }
    metadata = dict(data.get("metadata", {}))
    metadata["num_samples_raw"] = len(predictions)
    metadata["num_samples"] = len(data["predictions"])
    metadata["excluded_sensor_ids"] = sorted(EXCLUDED_SENSOR_IDS)
    metadata["excluded_sample_count"] = removed
    metadata["source_result"] = SOURCE_PATH.name
    data["metadata"] = metadata

    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    print(f"writing {OUTPUT_PATH}", flush=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    tmp_path.replace(OUTPUT_PATH)
    print("done", flush=True)


if __name__ == "__main__":
    main()
