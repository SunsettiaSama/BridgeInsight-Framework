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


def _resolve(path_str: str) -> Path:
    return data_config.PROJECT_ROOT / path_str


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


def load_dl_result() -> dict:
    """加载 DL 全量识别结果，统一返回含 predictions / sample_metadata 的字典。"""
    source = data_config.DATA_SOURCE
    if source == "legacy":
        glob_path = _resolve(data_config.LEGACY["dl_result_glob"])
        return _load_latest_glob(glob_path)

    if source != "chapter4":
        raise ValueError(f"未知 DATA_SOURCE：{source!r}，应为 'legacy' 或 'chapter4'")

    runtime_path = data_config.CHAPTER4.get("runtime_config_path")
    if runtime_path:
        from src.chapter4_characteristics.settings import (
            get_predictions_enriched_path,
            load_config,
        )

        cfg = load_config(str(runtime_path))
        path = get_predictions_enriched_path(cfg)
    else:
        path = _resolve(data_config.CHAPTER4["predictions_enriched"])

    if not path.exists():
        raise FileNotFoundError(
            f"Chapter4 识别结果不存在：{path}\n"
            "请先运行：python -m src.chapter4_characteristics infer"
        )

    print(f"  加载识别结果：{path.name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
