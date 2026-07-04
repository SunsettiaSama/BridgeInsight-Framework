from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Windows 下 torch / numpy / matplotlib 可能同时加载 Intel OpenMP。
# 该脚本只做离线绘图，允许重复运行时继续完成保存与 dashboard 推送。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.infer.dataset_loader import load_staycable_dataset
from src.chapter3_identifier.identifier.dl.identifier import DLVibrationIdentifier
from src.chapter3_identifier.identifier.dl.runner import FullDatasetRunner
from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import DEFAULT_PORT as DASHBOARD_DEFAULT_PORT
from src.visualize_tools.web_dashboard import push as web_push

ensure_paths()


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "training_result"
    / "deep_learning_module"
    / "res_cnn"
    / "checkpoints"
    / "ResCNN_20260402_111429"
    / "best_checkpoint.pth"
)
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "train" / "models" / "res_cnn.yaml"
FULL_DATASET_CONFIG = PROJECT_ROOT / "config" / "datasets" / "total_staycable_vib_202409.yaml"
GOLD_ANNOTATION_PATH = PROJECT_ROOT / "results" / "dataset_annotation" / "annotation_results.json"
DL_RESULT_GLOB = PROJECT_ROOT / "results" / "identification_result" / "res_cnn_full_dataset_*.json"
SEPT2024_RESULT_GLOB = (
    PROJECT_ROOT
    / "results"
    / "chapter1"
    / "augment"
    / "1-fullly_recognize"
    / "res_cnn_sept2024_*.json"
)

CACHE_DIR = PROJECT_ROOT / "results" / "chapter1" / "augment" / "fig3_32_initial_rescnn_confidence"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figure_paintings" / "figs_for_thesis" / "Chapter3"

WEBUI_PORT = DASHBOARD_DEFAULT_PORT
WEBUI_PAGE = "fig3_32 初始ResCNN置信度差异"
LABEL_NAMES = ("Normal", "VIV", "RWIV", "Others")
_PALETTE = get_full_color_map(style="discrete").colors
COLORS = {
    "gold": VIV_INPLANE_COLOR,
    "full": VIV_OUTPLANE_COLOR,
    "gold_light": _PALETTE[2],
    "full_light": _PALETTE[8],
}
FIGURE_SPECS = (
    ("fig3_32_initial_rescnn_uncertainty_distribution.png", "不确定度分布"),
    ("fig3_32_initial_rescnn_uncertainty_boxplot.png", "不确定度箱线图"),
    ("fig3_32_initial_rescnn_uncertainty_ecdf.png", "不确定度累计分布"),
    ("fig3_32_initial_rescnn_uncertainty_ratio.png", "高不确定比例"),
)
FS = 50.0
WINDOW_SIZE = 3000
BATCH_SIZE = 256
NUM_WORKERS = 4
UNCERTAINTY_THRESHOLDS = (0.01, 0.03, 0.05, 0.08, 0.10)


def log_section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def log_kv(key: str, value) -> None:
    print(f"  {key}: {value}")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_identifier() -> DLVibrationIdentifier:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"初始 ResCNN checkpoint 不存在：{CHECKPOINT_PATH}")
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"ResCNN 模型配置不存在：{MODEL_CONFIG_PATH}")
    log_section("模型配置")
    log_kv("model", "Initial non-augment ResCNN")
    log_kv("checkpoint", CHECKPOINT_PATH)
    log_kv("checkpoint_folder", CHECKPOINT_PATH.parent.name)
    log_kv("model_config", MODEL_CONFIG_PATH)
    log_kv("num_classes", len(LABEL_NAMES))
    log_kv("label_names", ", ".join(LABEL_NAMES))
    return DLVibrationIdentifier.from_checkpoint(
        checkpoint_path=str(CHECKPOINT_PATH),
        model_type="res_cnn",
        model_config_path=str(MODEL_CONFIG_PATH),
        num_classes=len(LABEL_NAMES),
    )


def _max_confidence(probas: dict[int, np.ndarray]) -> np.ndarray:
    if not probas:
        return np.array([], dtype=np.float32)
    ordered = [np.asarray(probas[idx], dtype=np.float32) for idx in sorted(probas)]
    arr = np.vstack(ordered)
    return arr.max(axis=1).astype(np.float32)


def latest_dl_result_path() -> Path | None:
    parent = DL_RESULT_GLOB.parent
    if not parent.exists():
        return None
    files = sorted(parent.glob(DL_RESULT_GLOB.name))
    if not files:
        return None
    return files[-1]


def latest_sept2024_result_path() -> Path | None:
    pointer = SEPT2024_RESULT_GLOB.parent / "latest.json"
    if pointer.exists():
        payload = load_json(pointer)
        result_path = Path(payload.get("result_path", ""))
        if result_path.exists():
            return result_path
    parent = SEPT2024_RESULT_GLOB.parent
    if not parent.exists():
        return None
    files = sorted(parent.glob(SEPT2024_RESULT_GLOB.name))
    if not files:
        return None
    return files[-1]


def _is_202409_result(result: dict) -> bool:
    sample_metadata = result.get("sample_metadata", {})
    if not sample_metadata:
        return False
    checked = 0
    sept_hits = 0
    wrong_year_hits = 0
    for meta in sample_metadata.values():
        for key in ("inplane_file_path", "outplane_file_path"):
            file_path = str(meta.get(key) or "")
            if not file_path:
                continue
            checked += 1
            normalized = file_path.replace("\\", "/")
            if "2024September" in normalized or "/2024/09/" in normalized:
                sept_hits += 1
            if "/2023/" in normalized or "\\2023\\" in file_path:
                wrong_year_hits += 1
        if checked >= 200:
            break
    return checked > 0 and sept_hits == checked and wrong_year_hits == 0


def _result_overview(result: dict) -> dict:
    metadata = result.get("metadata", {})
    sample_metadata = result.get("sample_metadata", {})
    cable_pairs = set()
    sensors = set()
    months = set()
    path_examples = []
    for meta in sample_metadata.values():
        pair = meta.get("cable_pair")
        if pair:
            cable_pairs.add(tuple(pair))
        for key in ("inplane_sensor_id", "outplane_sensor_id"):
            sensor_id = meta.get(key)
            if sensor_id:
                sensors.add(str(sensor_id))
        timestamp = meta.get("timestamp") or []
        if timestamp:
            months.add(str(timestamp[0]))
        for key in ("inplane_file_path", "outplane_file_path"):
            file_path = meta.get(key)
            if file_path and len(path_examples) < 4:
                path_examples.append(str(file_path))
    return {
        "created_at": metadata.get("created_at", ""),
        "model_info": metadata.get("model_info", ""),
        "fingerprint": metadata.get("dataset_fingerprint_hash", ""),
        "sample_count": len(sample_metadata),
        "channel_window_count": sum(
            1
            for meta in sample_metadata.values()
            for key in ("inplane_file_path", "outplane_file_path")
            if meta.get(key)
        ),
        "cable_pairs": sorted(cable_pairs),
        "sensor_count": len(sensors),
        "sensors": sorted(sensors),
        "months": sorted(months, key=lambda x: int(x) if x.isdigit() else x),
        "path_examples": path_examples,
    }


def log_result_overview(result: dict, result_path: Path) -> None:
    overview = _result_overview(result)
    log_section("全量识别缓存概况")
    log_kv("result_path", result_path)
    log_kv("created_at", overview["created_at"])
    log_kv("model_info", overview["model_info"])
    log_kv("dataset_fingerprint_hash", overview["fingerprint"])
    log_kv("sample_count", f"{overview['sample_count']:,}")
    log_kv("channel_window_count", f"{overview['channel_window_count']:,}")
    log_kv("months", ", ".join(overview["months"]))
    log_kv("sensor_count", overview["sensor_count"])
    log_kv("sensors", ", ".join(overview["sensors"]))
    log_kv(
        "cable_pairs",
        "; ".join(f"{pair[0]} / {pair[1]}" for pair in overview["cable_pairs"]),
    )
    for idx, path in enumerate(overview["path_examples"], start=1):
        log_kv(f"path_example_{idx}", path)


def _dataset_overview(dataset) -> dict:
    sensors = set()
    cable_pairs = set()
    months = set()
    for rec in dataset._samples:
        cable_pairs.add(tuple(rec.cable_pair))
        if rec.inplane_meta:
            sensor_id = rec.inplane_meta.get("sensor_id")
            if sensor_id:
                sensors.add(str(sensor_id))
        if rec.outplane_meta:
            sensor_id = rec.outplane_meta.get("sensor_id")
            if sensor_id:
                sensors.add(str(sensor_id))
        if rec.timestamp_key:
            months.add(str(rec.timestamp_key[0]))
    return {
        "sample_count": len(dataset),
        "window_size": getattr(dataset.config, "window_size", ""),
        "enable_denoise": getattr(dataset.config, "enable_denoise", ""),
        "missing_rate_threshold": getattr(dataset.config, "missing_rate_threshold", ""),
        "vib_metadata_path": getattr(dataset.config, "vib_metadata_path", ""),
        "cache_path": getattr(dataset.config, "cache_path", ""),
        "cable_pairs": sorted(cable_pairs),
        "sensor_count": len(sensors),
        "sensors": sorted(sensors),
        "months": sorted(months, key=lambda x: int(x) if x.isdigit() else x),
    }


def log_dataset_overview(dataset) -> None:
    overview = _dataset_overview(dataset)
    log_section("202409 全量数据集概况")
    log_kv("dataset_config", FULL_DATASET_CONFIG)
    log_kv("vib_metadata_path", overview["vib_metadata_path"])
    log_kv("dataset_cache_path", overview["cache_path"])
    log_kv("sample_count", f"{overview['sample_count']:,}")
    log_kv("channel_window_count_est", f"{overview['sample_count'] * 2:,}")
    log_kv("window_size", overview["window_size"])
    log_kv("enable_denoise", overview["enable_denoise"])
    log_kv("missing_rate_threshold", overview["missing_rate_threshold"])
    log_kv("months", ", ".join(overview["months"]))
    log_kv("sensor_count", overview["sensor_count"])
    log_kv("sensors", ", ".join(overview["sensors"]))
    log_kv(
        "cable_pairs",
        "; ".join(f"{pair[0]} / {pair[1]}" for pair in overview["cable_pairs"]),
    )


def _flush_full_batch(
    identifier: DLVibrationIdentifier,
    signals: list[np.ndarray],
    confidence: list[float],
) -> None:
    if not signals:
        return
    batch = torch.from_numpy(np.stack(signals).astype(np.float32)).unsqueeze(-1)
    proba = identifier.predict_batch_proba(batch)
    confidence.extend(proba.max(axis=1).astype(float).tolist())
    signals.clear()


def _confidence_from_result_metadata(
    identifier: DLVibrationIdentifier,
    result_path: Path,
    force: bool = False,
) -> np.ndarray:
    cache_path = CACHE_DIR / f"full_202409_initial_rescnn_confidence_{result_path.stem}.npz"
    if cache_path.exists() and not force:
        log_section("全量置信度缓存命中")
        log_kv("cache_path", cache_path)
        log_kv("source_result", result_path)
        data = np.load(cache_path)
        log_kv("confidence_count", f"{len(data['confidence']):,}")
        return data["confidence"].astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = FullDatasetRunner.load_result(str(result_path))
    if not _is_202409_result(result):
        raise ValueError(f"识别缓存不是 2024 年 9 月数据，拒绝使用：{result_path}")
    log_result_overview(result, result_path)
    log_section("从识别缓存补算 softmax 置信度")
    log_kv("batch_size", BATCH_SIZE)
    log_kv("num_workers", "N/A (metadata window loader)")
    log_kv("window_size", WINDOW_SIZE)
    log_kv("fs", FS)
    log_kv("output_cache", cache_path)
    sample_metadata = result.get("sample_metadata", {})
    rows: list[tuple[str, int]] = []
    for _, meta in sorted(sample_metadata.items(), key=lambda item: int(item[0])):
        window_idx = int(meta.get("window_idx", 0))
        in_path = meta.get("inplane_file_path")
        out_path = meta.get("outplane_file_path")
        if in_path and Path(in_path).exists():
            rows.append((str(in_path), window_idx))
        if out_path and Path(out_path).exists():
            rows.append((str(out_path), window_idx))
    rows.sort(key=lambda item: (item[0], item[1]))
    if not rows:
        raise ValueError(f"识别缓存缺少可用 sample_metadata：{result_path}")
    unique_files = len({file_path for file_path, _ in rows})
    log_kv("candidate_channel_windows", f"{len(rows):,}")
    log_kv("unique_vic_files", f"{unique_files:,}")

    unpacker = UNPACK(init_path=False)
    signals: list[np.ndarray] = []
    confidence: list[float] = []
    current_file = ""
    current_data = None
    for file_path, window_idx in rows:
        if file_path != current_file:
            current_data = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float32).reshape(-1)
            current_file = file_path
        start = int(window_idx) * WINDOW_SIZE
        end = start + WINDOW_SIZE
        if current_data is None or end > len(current_data):
            continue
        signals.append(current_data[start:end])
        if len(signals) >= BATCH_SIZE:
            _flush_full_batch(identifier, signals, confidence)
        if len(confidence) and len(confidence) % (BATCH_SIZE * 200) == 0:
            log_kv("progress_channel_windows", f"{len(confidence):,}/{len(rows):,}")
    _flush_full_batch(identifier, signals, confidence)

    confidence_arr = np.asarray(confidence, dtype=np.float32)
    if len(confidence_arr) == 0:
        raise ValueError(f"识别缓存中没有可推理窗口：{result_path}")
    np.savez_compressed(
        cache_path,
        confidence=confidence_arr,
        checkpoint=str(CHECKPOINT_PATH),
        dl_result_path=str(result_path),
    )
    log_kv("saved_confidence_cache", cache_path)
    log_kv("confidence_count", f"{len(confidence_arr):,}")
    return confidence_arr


def compute_full_dataset_confidence(identifier: DLVibrationIdentifier, force: bool = False) -> np.ndarray:
    cache_path = CACHE_DIR / "full_202409_initial_rescnn_confidence.npz"
    if cache_path.exists() and not force:
        log_section("全量置信度缓存命中")
        log_kv("cache_path", cache_path)
        data = np.load(cache_path)
        log_kv("confidence_count", f"{len(data['confidence']):,}")
        return data["confidence"].astype(np.float32)

    result_path = latest_sept2024_result_path()
    if result_path is not None:
        print(f"  使用 202409 ResCNN 识别缓存补算置信度：{result_path.name}")
        return _confidence_from_result_metadata(identifier, result_path, force=force)

    result_path = latest_dl_result_path()
    if result_path is not None:
        result = FullDatasetRunner.load_result(str(result_path))
        if _is_202409_result(result):
            print(f"  使用 Chapter4 DL 识别缓存补算置信度：{result_path.name}")
            return _confidence_from_result_metadata(identifier, result_path, force=force)
        print(f"  跳过非 202409 Chapter4 识别缓存：{result_path.name}")

    print("  未找到 Chapter4 DL 识别缓存，回退为全量数据集概率推理")
    if not FULL_DATASET_CONFIG.exists():
        raise FileNotFoundError(f"全量数据集配置不存在：{FULL_DATASET_CONFIG}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_section("全量概率推理参数")
    log_kv("dataset_config", FULL_DATASET_CONFIG)
    log_kv("checkpoint", CHECKPOINT_PATH)
    log_kv("model_config", MODEL_CONFIG_PATH)
    log_kv("batch_size", BATCH_SIZE)
    log_kv("num_workers", NUM_WORKERS)
    log_kv("window_size", WINDOW_SIZE)
    log_kv("fs", FS)
    log_kv("output_cache", cache_path)
    dataset = load_staycable_dataset(str(FULL_DATASET_CONFIG))
    log_dataset_overview(dataset)
    runner = FullDatasetRunner(
        identifier=identifier,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    inplane_proba, outplane_proba = runner.run_with_proba(dataset)
    confidence = np.concatenate([
        _max_confidence(inplane_proba),
        _max_confidence(outplane_proba),
    ]).astype(np.float32)
    np.savez_compressed(
        cache_path,
        confidence=confidence,
        checkpoint=str(CHECKPOINT_PATH),
        dataset_config=str(FULL_DATASET_CONFIG),
    )
    log_kv("saved_confidence_cache", cache_path)
    log_kv("confidence_count", f"{len(confidence):,}")
    return confidence


def _annotation_label(row: dict) -> int:
    value = row.get("annotation", row.get("label", row.get("class_id")))
    if value is None:
        raise ValueError(f"黄金标注缺少 annotation/label/class_id：{row}")
    return int(value)


def load_gold_records() -> list[dict]:
    rows = load_json(GOLD_ANNOTATION_PATH)
    if not isinstance(rows, list):
        raise ValueError(f"黄金标注格式异常：期望 list，实际 {type(rows)}")
    records = []
    for row in rows:
        file_path = row.get("file_path") or row.get("metadata", {}).get("file_path")
        window_index = row.get("window_index")
        if not file_path or window_index is None:
            continue
        path = Path(file_path)
        if not path.exists():
            continue
        records.append(
            {
                "file_path": str(path),
                "window_index": int(window_index),
                "label": _annotation_label(row),
            }
        )
    records.sort(key=lambda item: (item["file_path"], item["window_index"]))
    if not records:
        raise ValueError("没有可用黄金标注窗口")
    return records


def _flush_gold_batch(
    identifier: DLVibrationIdentifier,
    signals: list[np.ndarray],
    labels: list[int],
    max_conf: list[float],
    true_conf: list[float],
) -> None:
    if not signals:
        return
    batch = torch.from_numpy(np.stack(signals).astype(np.float32)).unsqueeze(-1)
    proba = identifier.predict_batch_proba(batch)
    max_conf.extend(proba.max(axis=1).astype(float).tolist())
    true_conf.extend(float(p[label]) for p, label in zip(proba, labels))
    signals.clear()
    labels.clear()


def compute_gold_confidence(identifier: DLVibrationIdentifier, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    cache_path = CACHE_DIR / "gold_initial_rescnn_confidence.npz"
    if cache_path.exists() and not force:
        log_section("黄金集置信度缓存命中")
        log_kv("cache_path", cache_path)
        data = np.load(cache_path)
        log_kv("gold_window_count", f"{len(data['max_confidence']):,}")
        return data["max_confidence"].astype(np.float32), data["true_label_confidence"].astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = load_gold_records()
    log_section("黄金集概率推理参数")
    log_kv("gold_annotation_path", GOLD_ANNOTATION_PATH)
    log_kv("gold_window_count", f"{len(records):,}")
    log_kv("batch_size", BATCH_SIZE)
    log_kv("window_size", WINDOW_SIZE)
    log_kv("fs", FS)
    log_kv("output_cache", cache_path)
    unpacker = UNPACK(init_path=False)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    max_conf: list[float] = []
    true_conf: list[float] = []
    current_file = ""
    current_data = None

    for record in records:
        file_path = record["file_path"]
        if file_path != current_file:
            current_data = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float32).reshape(-1)
            current_file = file_path
        start = int(record["window_index"]) * WINDOW_SIZE
        end = start + WINDOW_SIZE
        if current_data is None or end > len(current_data):
            continue
        signals.append(current_data[start:end])
        labels.append(int(record["label"]))
        if len(signals) >= BATCH_SIZE:
            _flush_gold_batch(identifier, signals, labels, max_conf, true_conf)
        if len(max_conf) and len(max_conf) % (BATCH_SIZE * 50) == 0:
            log_kv("progress_gold_windows", f"{len(max_conf):,}/{len(records):,}")
    _flush_gold_batch(identifier, signals, labels, max_conf, true_conf)

    max_arr = np.asarray(max_conf, dtype=np.float32)
    true_arr = np.asarray(true_conf, dtype=np.float32)
    if len(max_arr) == 0:
        raise ValueError("黄金标注窗口全部不可用，无法计算置信度")
    np.savez_compressed(
        cache_path,
        max_confidence=max_arr,
        true_label_confidence=true_arr,
        checkpoint=str(CHECKPOINT_PATH),
        gold_annotation_path=str(GOLD_ANNOTATION_PATH),
    )
    log_kv("saved_gold_confidence_cache", cache_path)
    log_kv("gold_confidence_count", f"{len(max_arr):,}")
    return max_arr, true_arr


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "lt_070": float(np.mean(values < 0.70)),
        "lt_080": float(np.mean(values < 0.80)),
    }


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.sort(np.unique(np.concatenate([a, b])))
    if len(grid) == 0:
        return 0.0
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def setup_axis(ax, xlabel: str = "", ylabel: str = "") -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=CN_FONT, fontsize=SQUARE_FONT_SIZE - 8)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=CN_FONT, fontsize=SQUARE_FONT_SIZE - 8)
    ax.grid(True, linestyle="--", alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=SQUARE_FONT_SIZE - 10)


def uncertainty(values: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(values, dtype=np.float32)


def uncertainty_axis_limit(*arrays: np.ndarray) -> float:
    combined = np.concatenate([np.asarray(arr, dtype=float) for arr in arrays if len(arr)])
    if len(combined) == 0:
        return 0.1
    return float(min(1.0, max(0.08, np.quantile(combined, 0.995) * 1.2)))


def plot_distribution(gold_conf: np.ndarray, full_conf: np.ndarray) -> plt.Figure:
    gold_unc = uncertainty(gold_conf)
    full_unc = uncertainty(full_conf)
    x_max = uncertainty_axis_limit(gold_unc, full_unc)
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    bins = np.linspace(0.0, x_max, 51)
    ax.hist(
        full_unc,
        bins=bins,
        density=True,
        alpha=0.55,
        color=COLORS["full"],
        label=f"2024年9月全量通道（n={len(full_conf):,}）",
    )
    ax.hist(
        gold_unc,
        bins=bins,
        density=True,
        alpha=0.60,
        color=COLORS["gold"],
        label=f"黄金标注集（n={len(gold_conf):,}）",
    )
    ax.axvline(np.median(full_unc), color=COLORS["full"], linestyle="--", linewidth=1.6)
    ax.axvline(np.median(gold_unc), color=COLORS["gold"], linestyle="--", linewidth=1.6)
    ax.set_xlim(0.0, x_max)
    setup_axis(ax, "预测不确定度（1 - 预测类别概率）", "密度")
    ax.legend(frameon=False, prop=CN_FONT, fontsize=SQUARE_FONT_SIZE - 9)
    fig.tight_layout()
    return fig


def plot_boxplot(gold_conf: np.ndarray, full_conf: np.ndarray, gold_true_conf: np.ndarray) -> plt.Figure:
    gold_unc = uncertainty(gold_conf)
    full_unc = uncertainty(full_conf)
    gold_true_unc = uncertainty(gold_true_conf)
    y_max = uncertainty_axis_limit(gold_unc, full_unc, gold_true_unc)
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    box = ax.boxplot(
        [gold_unc, full_unc, gold_true_unc],
        tick_labels=["黄金集\n预测不确定度", "全量集\n预测不确定度", "黄金集\n真类不确定度"],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, color in zip(box["boxes"], [COLORS["gold"], COLORS["full"], COLORS["gold_light"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    setup_axis(ax, "", "不确定度")
    ax.set_ylim(0.0, y_max)
    for label in ax.get_xticklabels():
        label.set_fontproperties(CN_FONT)
        label.set_fontsize(SQUARE_FONT_SIZE - 8)
    fig.tight_layout()
    return fig


def plot_ecdf(gold_conf: np.ndarray, full_conf: np.ndarray) -> plt.Figure:
    gold_unc = uncertainty(gold_conf)
    full_unc = uncertainty(full_conf)
    x_max = uncertainty_axis_limit(gold_unc, full_unc)
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    for values, color, label in (
        (gold_unc, COLORS["gold"], "黄金集预测不确定度"),
        (full_unc, COLORS["full"], "全量集预测不确定度"),
    ):
        sorted_values = np.sort(values)
        y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.plot(sorted_values, y, color=color, linewidth=2.2, label=label)
    for threshold in (0.03, 0.05, 0.10):
        ax.axvline(threshold, color="#555555", linestyle=":", linewidth=1.2)
        ax.text(
            threshold + x_max * 0.015,
            0.05,
            f"{threshold:.2f}",
            fontsize=SQUARE_FONT_SIZE - 12,
            color="#555555",
        )
    setup_axis(ax, "预测不确定度（1 - 预测类别概率）", "累计比例")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, prop=CN_FONT, fontsize=SQUARE_FONT_SIZE - 9)
    fig.tight_layout()
    return fig


def plot_low_confidence_ratio(gold_conf: np.ndarray, full_conf: np.ndarray) -> plt.Figure:
    gold_unc = uncertainty(gold_conf)
    full_unc = uncertainty(full_conf)
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    x = np.arange(len(UNCERTAINTY_THRESHOLDS))
    width = 0.36
    gold_high = np.array([np.mean(gold_unc > threshold) for threshold in UNCERTAINTY_THRESHOLDS])
    full_high = np.array([np.mean(full_unc > threshold) for threshold in UNCERTAINTY_THRESHOLDS])
    ax.bar(x - width / 2, gold_high, width=width, color=COLORS["gold"], label="黄金标注集")
    ax.bar(x + width / 2, full_high, width=width, color=COLORS["full"], label="2024年9月全量集")
    ax.set_xticks(x)
    ax.set_xticklabels([f">{threshold:.2f}" for threshold in UNCERTAINTY_THRESHOLDS])
    y_max = max(0.05, float(max(gold_high.max(), full_high.max()) * 1.25))
    ax.set_ylim(0.0, min(1.0, y_max))
    setup_axis(ax, "不确定度阈值", "超过阈值的样本比例")
    ax.legend(frameon=False, prop=CN_FONT, fontsize=SQUARE_FONT_SIZE - 9)
    label_offset = ax.get_ylim()[1] * 0.025
    for idx, value in enumerate(gold_high):
        ax.text(
            idx - width / 2,
            min(value + label_offset, ax.get_ylim()[1] * 0.97),
            f"{value:.1%}",
            ha="center",
            fontsize=SQUARE_FONT_SIZE - 12,
        )
    for idx, value in enumerate(full_high):
        ax.text(
            idx + width / 2,
            min(value + label_offset, ax.get_ylim()[1] * 0.97),
            f"{value:.1%}",
            ha="center",
            fontsize=SQUARE_FONT_SIZE - 12,
        )
    fig.tight_layout()
    return fig


def build_figures(
    gold_conf: np.ndarray,
    full_conf: np.ndarray,
    gold_true_conf: np.ndarray,
) -> list[plt.Figure]:
    return [
        plot_distribution(gold_conf, full_conf),
        plot_boxplot(gold_conf, full_conf, gold_true_conf),
        plot_ecdf(gold_conf, full_conf),
        plot_low_confidence_ratio(gold_conf, full_conf),
    ]


def save_and_push(figures: list[plt.Figure], port: int = WEBUI_PORT) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for fig, (filename, _) in zip(figures, FIGURE_SPECS):
        output_path = OUTPUT_DIR / filename
        fig.savefig(output_path, dpi=240)
        output_paths.append(output_path)
        print(f"saved: {output_path}")

    for slot, (fig, (_, title)) in enumerate(zip(figures, FIGURE_SPECS)):
        web_push(
            fig,
            page=WEBUI_PAGE,
            slot=slot,
            title=title,
            port=port,
            page_cols=2,
        )
        plt.close(fig)
    print(f"dashboard page: {WEBUI_PAGE}, port={WEBUI_PORT}")
    return output_paths


def print_summary(gold_conf: np.ndarray, full_conf: np.ndarray, gold_true_conf: np.ndarray) -> None:
    gold_desc = describe(gold_conf)
    full_desc = describe(full_conf)
    gold_unc = uncertainty(gold_conf)
    full_unc = uncertainty(full_conf)
    gold_unc_desc = describe(gold_unc)
    full_unc_desc = describe(full_unc)
    log_section("图像摘要数据")
    log_kv("模型", "未经过 augment 的初始 ResCNN")
    log_kv("checkpoint", CHECKPOINT_PATH.parent.name)
    log_kv("黄金集预测概率中位数", f"{gold_desc['median']:.4f}")
    log_kv("全量集预测概率中位数", f"{full_desc['median']:.4f}")
    log_kv("黄金集不确定度中位数", f"{gold_unc_desc['median']:.4f}")
    log_kv("全量集不确定度中位数", f"{full_unc_desc['median']:.4f}")
    for threshold in UNCERTAINTY_THRESHOLDS:
        log_kv(
            f"不确定度 > {threshold:.2f}",
            f"黄金集 {np.mean(gold_unc > threshold):.2%} / 全量集 {np.mean(full_unc > threshold):.2%}",
        )
    log_kv("预测概率 KS 距离", f"{ks_distance(gold_conf, full_conf):.4f}")
    log_kv("不确定度 KS 距离", f"{ks_distance(gold_unc, full_unc):.4f}")

    log_section("概率与不确定度统计 CSV")
    rows = [
        ("gold_pmax", describe(gold_conf)),
        ("full_pmax", describe(full_conf)),
        ("gold_ptrue", describe(gold_true_conf)),
        ("gold_uncertainty", describe(gold_unc)),
        ("full_uncertainty", describe(full_unc)),
    ]
    print("dataset,n,mean,median,p10,p25,p75,p90,lt_070,lt_080")
    for name, stats in rows:
        print(
            f"{name},{int(stats['n'])},{stats['mean']:.4f},{stats['median']:.4f},"
            f"{stats['p10']:.4f},{stats['p25']:.4f},{stats['p75']:.4f},"
            f"{stats['p90']:.4f},{stats['lt_070']:.4f},{stats['lt_080']:.4f}"
        )
    print(f"ks_gold_vs_full,{ks_distance(gold_conf, full_conf):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="初始 ResCNN 黄金集/全量集置信度差异图")
    parser.add_argument("--force", action="store_true", help="强制重算置信度缓存")
    parser.add_argument("--port", type=int, default=WEBUI_PORT, help="VibDash 服务端口")
    args = parser.parse_args()

    print("=" * 80)
    print("fig3_32 初始 ResCNN 置信度差异")
    print("=" * 80)
    identifier = build_identifier()
    gold_conf, gold_true_conf = compute_gold_confidence(identifier, force=bool(args.force))
    full_conf = compute_full_dataset_confidence(identifier, force=bool(args.force))
    print_summary(gold_conf, full_conf, gold_true_conf)
    figures = build_figures(gold_conf, full_conf, gold_true_conf)
    save_and_push(figures, port=int(args.port))


if __name__ == "__main__":
    main()
