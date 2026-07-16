"""Chapter4 绘图统一数据加载：按 data_config.DATA_SOURCE 在 legacy / chapter4 间切换。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_chapter4_dir = Path(__file__).resolve().parent
_project_root = _chapter4_dir.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()

from src.figure_paintings.figs_for_thesis.Chapter4 import data_config


def _load_chapter4_cfg() -> dict | None:
    if data_config.DATA_SOURCE != "chapter4":
        return None
    from src.chapter4_characteristics.settings import load_config

    runtime_path = data_config.CHAPTER4.get("runtime_config_path")
    return load_config(str(runtime_path) if runtime_path else None)


def _resolve(path_str: str) -> Path:
    return data_config.PROJECT_ROOT / path_str


def get_dl_result_path() -> Path:
    path = _resolve(data_config.CHAPTER4["predictions_enriched"])
    if not path.exists():
        raise FileNotFoundError(
            f"DL 识别结果不存在：{path}\n"
            "请先运行：python scripts/filter_chapter4_predictions.py"
        )
    return path


def iter_enriched_json_files(class_dir: Path) -> list[Path]:
    excluded = data_config.EXCLUDED_SENSOR_IDS
    if not class_dir.exists():
        return []

    if data_config.DATA_SOURCE == "chapter4":
        from src.chapter4_characteristics.feature_analysis._compactor import (
            ensure_class_dir_compacted,
            list_batch_json_files,
            list_canonical_json_files,
        )

        cfg = _load_chapter4_cfg()
        if cfg is not None and list_batch_json_files(class_dir):
            ensure_class_dir_compacted(
                class_dir,
                cfg=cfg,
                excluded_sensor_ids=excluded,
            )
        return list_canonical_json_files(class_dir, excluded_sensor_ids=excluded)

    return sorted(jf for jf in class_dir.glob("*.json") if jf.stem not in excluded)


def filter_sensor_groups(groups: dict[str, str]) -> dict[str, str]:
    excluded = data_config.EXCLUDED_SENSOR_IDS
    return {
        name: fname
        for name, fname in groups.items()
        if fname.replace(".json", "") not in excluded
    }


def _is_excluded_meta(meta: dict) -> bool:
    excluded = data_config.EXCLUDED_SENSOR_IDS
    return (
        meta.get("inplane_sensor_id") in excluded
        or meta.get("outplane_sensor_id") in excluded
    )


def _filter_excluded_result(result: dict) -> dict:
    """剔除第三章已排除的 C34-201/202 样本。"""
    predictions = result.get("predictions", {})
    sample_metadata = result.get("sample_metadata", {})
    keep_keys = [
        k for k in predictions
        if not _is_excluded_meta(sample_metadata.get(str(k), {}))
    ]
    removed = len(predictions) - len(keep_keys)
    if removed <= 0:
        return result

    result = dict(result)
    result["predictions"] = {k: predictions[k] for k in keep_keys}
    result["sample_metadata"] = {
        k: sample_metadata[k] for k in keep_keys if k in sample_metadata
    }
    metadata = dict(result.get("metadata", {}))
    metadata["num_samples"] = len(result["predictions"])
    metadata["excluded_sensor_ids"] = sorted(data_config.EXCLUDED_SENSOR_IDS)
    metadata["excluded_sample_count"] = removed
    result["metadata"] = metadata
    print(f"  已剔除 C34-201/202 样本：{removed} 个")
    return result


def _mecc_glob_path() -> Path:
    if data_config.DATA_SOURCE == "legacy":
        return _resolve(data_config.LEGACY["mecc_result_glob"])
    return _resolve(data_config.CHAPTER4["mecc_result_glob"])


def _load_latest_glob(full_glob: Path) -> dict:
    from src.chapter3_identifier.identifier.dl.runner import FullDatasetRunner

    parent = full_glob.parent
    pattern = full_glob.name
    if not parent.exists():
        raise FileNotFoundError(f"结果目录不存在：{parent}")
    files = sorted(parent.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在 {parent} 中未找到匹配 {pattern!r} 的文件")
    print(f"  加载识别结果：{files[-1].name}")
    return FullDatasetRunner.load_result(str(files[-1]))


def load_predictions_result(config_key: str = "predictions_enriched") -> dict:
    """加载指定 config_key 对应的 enriched 识别结果。"""
    source = data_config.DATA_SOURCE
    if source == "legacy":
        if config_key != "predictions_enriched":
            raise ValueError(f"legacy 数据源不支持 config_key={config_key!r}")
        glob_path = _resolve(data_config.LEGACY["dl_result_glob"])
        return _load_latest_glob(glob_path)

    if source != "chapter4":
        raise ValueError(f"未知 DATA_SOURCE：{source!r}，应为 'legacy' 或 'chapter4'")

    path_value = data_config.CHAPTER4.get(config_key)
    if not path_value:
        raise KeyError(f"data_config.CHAPTER4 缺少 {config_key!r}")

    runtime_path = data_config.CHAPTER4.get("runtime_config_path")
    if config_key == "predictions_enriched" and runtime_path:
        from src.chapter4_characteristics.settings import (
            get_predictions_enriched_path,
            load_config,
        )

        cfg = load_config(str(runtime_path))
        path = get_predictions_enriched_path(cfg)
    else:
        path = _resolve(path_value)
        if (
            config_key == "predictions_enriched"
            and not path.exists()
            and data_config.CHAPTER4.get("predictions_enriched_raw")
        ):
            raw_path = _resolve(data_config.CHAPTER4["predictions_enriched_raw"])
            if raw_path.exists():
                path = raw_path

    if not path.exists():
        raise FileNotFoundError(
            f"Chapter4 识别结果不存在：{path}\n"
            "请先运行：python -m src.chapter4_characteristics infer"
        )

    print(f"  加载识别结果：{path.name}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if config_key == "predictions_enriched_any_side":
        return payload
    return _filter_excluded_result(payload)


def load_dl_result() -> dict:
    """加载 DL 全量识别结果，统一返回含 predictions / sample_metadata 的字典。"""
    return load_predictions_result("predictions_enriched")


def get_enriched_class_dir(class_id: int) -> Path:
    """返回指定类别的 enriched 特征目录。"""
    if class_id not in data_config.CLASS_DIR_NAMES:
        raise KeyError(f"未知 class_id：{class_id}")

    source = data_config.DATA_SOURCE
    if source == "legacy":
        root = _resolve(data_config.LEGACY["enriched_root"])
    elif source == "chapter4":
        runtime_path = data_config.CHAPTER4.get("runtime_config_path")
        if runtime_path:
            from src.chapter4_characteristics.settings import get_enriched_dir, load_config

            cfg = load_config(str(runtime_path))
            root = get_enriched_dir(cfg)
        else:
            root = _resolve(data_config.CHAPTER4["enriched_root"])
    else:
        raise ValueError(f"未知 DATA_SOURCE：{source!r}，应为 'legacy' 或 'chapter4'")

    return root / data_config.CLASS_DIR_NAMES[class_id]


def load_mecc_result() -> dict:
    """加载最新 MECC 物理识别结果。"""
    return _load_latest_glob(_mecc_glob_path())
