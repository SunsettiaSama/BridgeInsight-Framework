from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chapter3_identifier.augment.annotation.split import load_saved_split_key_sets
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import DEFAULT_PORT as DASHBOARD_DEFAULT_PORT
from src.visualize_tools.web_dashboard import push as web_push


DEFAULT_FINAL_ROOT = PROJECT_ROOT / "results" / "augment" / "final"
DEFAULT_FINAL_ROUNDS_ROOT = DEFAULT_FINAL_ROOT / "rounds"
DEFAULT_FINAL_MANIFEST = DEFAULT_FINAL_ROOT / "final_manifest.json"
DEFAULT_SPLIT_INDICES = PROJECT_ROOT / "results" / "augment" / "split_indices.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figure_paintings" / "figs_for_thesis" / "Chapter3"
WEBUI_PAGE = "fig3_37-42 多轮迭代困惑度与数据集变动"
FIGURE_SPECS = (
    ("fig3_37_iterative_perplexity.png", "终盘重训困惑度"),
    ("fig3_38_iterative_median_perplexity.png", "终盘重训中位数困惑度"),
    ("fig3_39_iterative_sample_mean_f1.png", "终盘重训样本平均F1"),
    ("fig3_40_iterative_annotation_scale.png", "训练+验证标注集规模"),
    ("fig3_41_iterative_annotation_delta.png", "相邻轮次标注集变动"),
    ("fig3_42_iterative_inference_prediction_dist.png", "全量识别预测类别占比"),
)
DATASET_START_ROUND = 2
EPS = 1e-12
CLASS_NAMES = ("Normal", "VIV", "RWIV", "Others")
HIGH_UNCERTAINTY_THRESHOLD = 0.5


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_index(round_dir: Path) -> int:
    return int(round_dir.name.rsplit("_", 1)[1])


def discover_final_round_dirs(rounds_root: Path, max_round: int | None = None) -> list[Path]:
    if not rounds_root.exists():
        raise FileNotFoundError(f"final 轮次目录不存在：{rounds_root}")
    round_dirs = [
        path
        for path in rounds_root.iterdir()
        if path.is_dir()
        and path.name.startswith("round_")
        and (path / "merged_training.json").exists()
        and (path / "inference.json").exists()
    ]
    round_dirs.sort(key=round_index)
    if max_round is not None:
        round_dirs = [path for path in round_dirs if round_index(path) <= int(max_round)]
    if not round_dirs:
        raise FileNotFoundError(f"没有找到 final 轮次（需 merged_training.json 与 inference.json）：{rounds_root}")
    return round_dirs


def load_final_round_reports(manifest_path: Path) -> dict[int, dict[str, Any]]:
    manifest = load_json(manifest_path)
    reports: dict[int, dict[str, Any]] = {}
    for report in manifest.get("round_artifact_reports", []):
        if not isinstance(report, dict):
            continue
        idx = int(report["round_idx"])
        if report.get("copy_only") and idx in reports:
            continue
        reports[idx] = report
    if not reports:
        raise ValueError(f"final_manifest 缺少 round_artifact_reports：{manifest_path}")
    return reports


def artifact_rows(report: dict[str, Any] | None, name: str) -> int:
    if not report:
        return 0
    artifact = report.get("artifacts", {}).get(name, {})
    rows = artifact.get("rows")
    return int(rows) if rows is not None else 0


def record_perplexity(proba: list[float]) -> float:
    arr = np.asarray(proba, dtype=np.float64)
    arr = np.clip(arr, EPS, 1.0)
    arr = arr / arr.sum()
    entropy = -float(np.sum(arr * np.log(arr)))
    return float(math.exp(entropy))


def joint_label_from_pair(row: dict[str, Any]) -> int:
    inplane = int(row["inplane_annotation"])
    outplane = int(row["outplane_annotation"])
    if inplane == outplane:
        return inplane
    return max(inplane, outplane)


def pair_key_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row["pair_key"])


def pair_label_state(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["inplane_annotation"]), int(row["outplane_annotation"]))


def summarize_pair_key_delta(
    current: dict[tuple[Any, ...], tuple[int, int]],
    previous: dict[tuple[Any, ...], tuple[int, int]] | None,
) -> dict[str, int | float]:
    if previous is None:
        return {
            "pair_added_total": len(current),
            "pair_modified_total": 0,
            "pair_removed_total": 0,
            "pair_modified_ratio": 0.0,
        }
    current_keys = set(current)
    previous_keys = set(previous)
    added = len(current_keys - previous_keys)
    removed = len(previous_keys - current_keys)
    modified = sum(1 for key in current_keys & previous_keys if current[key] != previous[key])
    total = max(1, len(current_keys))
    return {
        "pair_added_total": added,
        "pair_modified_total": modified,
        "pair_removed_total": removed,
        "pair_modified_ratio": modified / total,
    }


def summarize_pair_key_dataset(
    path: Path,
    train_keys: set[tuple[Any, ...]],
    val_keys: set[tuple[Any, ...]],
) -> tuple[dict[str, int | float], dict[tuple[Any, ...], tuple[int, int]]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"merged_training_pair_key.json 必须是列表：{path}")

    labels_by_key: dict[tuple[Any, ...], tuple[int, int]] = {}
    class_counts = [0, 0, 0, 0]
    train_total = 0
    val_total = 0
    manual_total = 0
    gold_total = 0
    abnormal_total = 0
    for row in rows:
        key = pair_key_tuple(row)
        labels_by_key[key] = pair_label_state(row)
        joint_label = joint_label_from_pair(row)
        class_counts[joint_label] += 1
        abnormal_total += int(joint_label != 0)
        manual_total += int(bool(row.get("is_manual", False)))
        gold_total += int(bool(row.get("is_gold", False)))
        if key in train_keys:
            train_total += 1
        elif key in val_keys:
            val_total += 1

    total = len(rows)
    denom = max(1, total)
    return {
        "pair_key_total": total,
        "train_pair_total": train_total,
        "val_pair_total": val_total,
        "pair_outside_split_total": total - train_total - val_total,
        "manual_pair_total": manual_total,
        "gold_pair_total": gold_total,
        "abnormal_pair_ratio": abnormal_total / denom,
        "pair_normal_ratio": class_counts[0] / denom,
        "pair_viv_ratio": class_counts[1] / denom,
        "pair_rwiv_ratio": class_counts[2] / denom,
        "pair_others_ratio": class_counts[3] / denom,
    }, labels_by_key


def inference_record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["inplane_file_path"]),
        str(record["outplane_file_path"]),
        int(record["window_index"]),
    )


def prediction_flip_ratio(
    current: dict[tuple[str, str, int], int],
    previous: dict[tuple[str, str, int], int] | None,
) -> float:
    if previous is None:
        return float("nan")
    common = set(current) & set(previous)
    if not common:
        return float("nan")
    flipped = sum(1 for key in common if current[key] != previous[key])
    return flipped / len(common)


def summarize_inference(
    path: Path,
) -> tuple[dict[str, float | int], dict[tuple[str, str, int], int]]:
    if not path.exists():
        return {
            "inference_total": 0,
            "mean_perplexity": np.nan,
            "median_perplexity": np.nan,
            "p90_perplexity": np.nan,
            "mean_inplane_perplexity": np.nan,
            "median_inplane_perplexity": np.nan,
            "mean_outplane_perplexity": np.nan,
            "median_outplane_perplexity": np.nan,
            "mean_uncertainty": np.nan,
            "high_uncertainty_ratio": np.nan,
            "pred_normal_ratio": np.nan,
            "pred_viv_ratio": np.nan,
            "pred_rwiv_ratio": np.nan,
            "pred_others_ratio": np.nan,
            "pred_abnormal_ratio": np.nan,
        }, {}

    merged_values: list[float] = []
    inplane_values: list[float] = []
    outplane_values: list[float] = []
    uncertainties: list[float] = []
    pred_counts = [0, 0, 0, 0]
    predictions: dict[tuple[str, str, int], int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("{") or '"prediction"' not in stripped:
                continue
            if stripped.endswith(","):
                stripped = stripped[:-1]
            record = json.loads(stripped)
            prediction = int(record["prediction"])
            pred_counts[prediction] += 1
            uncertainties.append(float(record.get("uncertainty", np.nan)))
            predictions[inference_record_key(record)] = prediction
            merged_values.append(record_perplexity(record.get("proba", [])))
            inplane_values.append(record_perplexity(record.get("inplane_proba", record.get("proba", []))))
            outplane_values.append(record_perplexity(record.get("outplane_proba", record.get("proba", []))))

    merged = np.asarray(merged_values, dtype=float)
    inplane = np.asarray(inplane_values, dtype=float)
    outplane = np.asarray(outplane_values, dtype=float)
    uncertainty = np.asarray(uncertainties, dtype=float)
    if merged.size == 0:
        return {
            "inference_total": 0,
            "mean_perplexity": np.nan,
            "median_perplexity": np.nan,
            "p90_perplexity": np.nan,
            "mean_inplane_perplexity": np.nan,
            "median_inplane_perplexity": np.nan,
            "mean_outplane_perplexity": np.nan,
            "median_outplane_perplexity": np.nan,
            "mean_uncertainty": np.nan,
            "high_uncertainty_ratio": np.nan,
            "pred_normal_ratio": np.nan,
            "pred_viv_ratio": np.nan,
            "pred_rwiv_ratio": np.nan,
            "pred_others_ratio": np.nan,
            "pred_abnormal_ratio": np.nan,
        }, {}

    total = int(merged.size)
    denom = max(1, total)
    finite_uncertainty = uncertainty[np.isfinite(uncertainty)]
    return {
        "inference_total": total,
        "mean_perplexity": float(np.mean(merged)),
        "median_perplexity": float(np.median(merged)),
        "p90_perplexity": float(np.percentile(merged, 90)),
        "mean_inplane_perplexity": float(np.mean(inplane)),
        "median_inplane_perplexity": float(np.median(inplane)),
        "mean_outplane_perplexity": float(np.mean(outplane)),
        "median_outplane_perplexity": float(np.median(outplane)),
        "mean_uncertainty": float(np.mean(finite_uncertainty)) if finite_uncertainty.size else np.nan,
        "high_uncertainty_ratio": float(np.mean(finite_uncertainty >= HIGH_UNCERTAINTY_THRESHOLD))
        if finite_uncertainty.size
        else np.nan,
        "pred_normal_ratio": pred_counts[0] / denom,
        "pred_viv_ratio": pred_counts[1] / denom,
        "pred_rwiv_ratio": pred_counts[2] / denom,
        "pred_others_ratio": pred_counts[3] / denom,
        "pred_abnormal_ratio": (pred_counts[1] + pred_counts[2] + pred_counts[3]) / denom,
    }, predictions


def direction_macro_f1(metrics: dict[str, Any], direction: str) -> float:
    values = [float(metrics.get(f"{direction}_{name}_f1", np.nan)) for name in CLASS_NAMES]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def legacy_macro_f1(metrics: dict[str, Any]) -> float:
    values = [float(metrics.get(f"{name}_f1", np.nan)) for name in CLASS_NAMES]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def sample_mean_f1(metrics: dict[str, Any]) -> float:
    inplane = direction_macro_f1(metrics, "inplane")
    outplane = direction_macro_f1(metrics, "outplane")
    if np.isfinite(inplane) or np.isfinite(outplane):
        return float(np.nanmean([inplane, outplane]))
    return legacy_macro_f1(metrics)


def metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, np.nan)
    return float(value) if value is not None else float("nan")


def best_f1_summary(metrics_path: Path) -> dict[str, float | int]:
    if not metrics_path.exists():
        return {
            "best_epoch": 0,
            "sample_mean_f1": np.nan,
            "inplane_macro_f1": np.nan,
            "outplane_macro_f1": np.nan,
            "joint_viv_rwiv_mean_f1": np.nan,
            "best_val_loss": np.nan,
        }
    rows = load_json(metrics_path)
    if not isinstance(rows, list) or not rows:
        return {
            "best_epoch": 0,
            "sample_mean_f1": np.nan,
            "inplane_macro_f1": np.nan,
            "outplane_macro_f1": np.nan,
            "joint_viv_rwiv_mean_f1": np.nan,
            "best_val_loss": np.nan,
        }

    best_row: dict[str, Any] | None = None
    best_score = float("-inf")
    for row in rows:
        metrics = row.get("val_metrics", {})
        score = sample_mean_f1(metrics)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return {
            "best_epoch": 0,
            "sample_mean_f1": np.nan,
            "inplane_macro_f1": np.nan,
            "outplane_macro_f1": np.nan,
            "joint_viv_rwiv_mean_f1": np.nan,
            "best_val_loss": np.nan,
        }

    best_metrics = best_row.get("val_metrics", {})
    inplane_f1 = direction_macro_f1(best_metrics, "inplane")
    outplane_f1 = direction_macro_f1(best_metrics, "outplane")
    if not np.isfinite(inplane_f1) and not np.isfinite(outplane_f1):
        legacy = legacy_macro_f1(best_metrics)
        inplane_f1 = legacy
        outplane_f1 = legacy
    val_losses = [float(row.get("val_loss", np.nan)) for row in rows]
    finite_losses = [value for value in val_losses if np.isfinite(value)]
    return {
        "best_epoch": int(best_row.get("epoch", 0)),
        "sample_mean_f1": sample_mean_f1(best_metrics),
        "inplane_macro_f1": inplane_f1,
        "outplane_macro_f1": outplane_f1,
        "joint_viv_rwiv_mean_f1": metric_value(
            best_metrics,
            "joint_viv_rwiv_mean_f1" if "joint_viv_rwiv_mean_f1" in best_metrics else "viv_rwiv_mean_f1",
        ),
        "best_val_loss": min(finite_losses) if finite_losses else np.nan,
    }


def collect_round_summaries(
    round_dirs: list[Path],
    final_reports: dict[int, dict[str, Any]],
    split_path: Path,
) -> list[dict[str, Any]]:
    train_keys, val_keys = load_saved_split_key_sets(str(split_path))
    summaries: list[dict[str, Any]] = []
    previous_pair_labels: dict[tuple[Any, ...], tuple[int, int]] | None = None
    previous_predictions: dict[tuple[str, str, int], int] | None = None

    for round_dir in round_dirs:
        idx = round_index(round_dir)
        report = final_reports.get(idx)
        print(f"collect final round {idx}: {round_dir}")

        pair_key_path = round_dir / "merged_training_pair_key.json"
        direction_window_total = artifact_rows(report, "merged_training")
        if pair_key_path.exists():
            pair_summary, pair_labels = summarize_pair_key_dataset(pair_key_path, train_keys, val_keys)
            pair_delta = summarize_pair_key_delta(pair_labels, previous_pair_labels)
            previous_pair_labels = pair_labels
        else:
            pair_summary = {
                "pair_key_total": 0,
                "train_pair_total": 0,
                "val_pair_total": 0,
                "pair_outside_split_total": 0,
                "manual_pair_total": artifact_rows(report, "manual_edits"),
                "gold_pair_total": 0,
                "abnormal_pair_ratio": np.nan,
                "pair_normal_ratio": np.nan,
                "pair_viv_ratio": np.nan,
                "pair_rwiv_ratio": np.nan,
                "pair_others_ratio": np.nan,
            }
            pair_delta = {
                "pair_added_total": 0,
                "pair_modified_total": 0,
                "pair_removed_total": 0,
                "pair_modified_ratio": 0.0,
            }

        inference_summary, predictions = summarize_inference(round_dir / "inference.json")
        f1_summary = best_f1_summary(round_dir / "metrics.json")
        manifest = load_json(round_dir / "round_manifest.json") if (round_dir / "round_manifest.json").exists() else {}
        flip_ratio = prediction_flip_ratio(predictions, previous_predictions)
        if predictions:
            previous_predictions = predictions

        summaries.append(
            {
                "round_idx": idx,
                **pair_summary,
                **pair_delta,
                **inference_summary,
                **f1_summary,
                "prediction_flip_ratio": flip_ratio,
                "manifest_record_count": int(
                    manifest.get("record_count", artifact_rows(report, "inference") or inference_summary["inference_total"])
                ),
                "manual_edits_rows": artifact_rows(report, "manual_edits"),
                "manual_history_rows": artifact_rows(report, "manual_edits_history"),
                "direction_window_total": direction_window_total,
            }
        )
    return summaries


def setup_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str | None = None,
    xlabel: str = "迭代轮次",
) -> None:
    ax.set_title(title, fontproperties=CN_FONT, fontsize=SQUARE_FONT_SIZE - 6, pad=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=CN_FONT, fontsize=SQUARE_FONT_SIZE - 8)
    ax.set_xlabel(xlabel, fontproperties=CN_FONT, fontsize=SQUARE_FONT_SIZE - 8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=SQUARE_FONT_SIZE - 10)


def annotate_points(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, fmt: str = "{:.2f}") -> None:
    finite = np.isfinite(ys)
    for x, y in zip(xs[finite], ys[finite]):
        ax.text(
            x,
            y,
            fmt.format(y),
            ha="center",
            va="bottom",
            fontsize=SQUARE_FONT_SIZE - 12,
            fontproperties=ENG_FONT,
        )


def extract_plot_arrays(summaries: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    rounds = np.asarray([row["round_idx"] for row in summaries], dtype=int)
    return {
        "rounds": rounds,
        "mean_perplexity": np.asarray([row["mean_perplexity"] for row in summaries], dtype=float),
        "median_perplexity": np.asarray([row["median_perplexity"] for row in summaries], dtype=float),
        "p90_perplexity": np.asarray([row["p90_perplexity"] for row in summaries], dtype=float),
        "inplane_perplexity": np.asarray([row["mean_inplane_perplexity"] for row in summaries], dtype=float),
        "inplane_median_perplexity": np.asarray([row["median_inplane_perplexity"] for row in summaries], dtype=float),
        "outplane_perplexity": np.asarray([row["mean_outplane_perplexity"] for row in summaries], dtype=float),
        "outplane_median_perplexity": np.asarray([row["median_outplane_perplexity"] for row in summaries], dtype=float),
        "train_pair_total": np.asarray([row["train_pair_total"] for row in summaries], dtype=float),
        "val_pair_total": np.asarray([row["val_pair_total"] for row in summaries], dtype=float),
        "pair_key_total": np.asarray([row["pair_key_total"] for row in summaries], dtype=float),
        "pair_added_total": np.asarray([row["pair_added_total"] for row in summaries], dtype=float),
        "pair_modified_total": np.asarray([row["pair_modified_total"] for row in summaries], dtype=float),
        "pair_removed_total": np.asarray([row["pair_removed_total"] for row in summaries], dtype=float),
        "abnormal_pair_ratio": np.asarray([row["abnormal_pair_ratio"] for row in summaries], dtype=float),
        "pair_normal_ratio": np.asarray([row["pair_normal_ratio"] for row in summaries], dtype=float),
        "pair_viv_ratio": np.asarray([row["pair_viv_ratio"] for row in summaries], dtype=float),
        "pair_rwiv_ratio": np.asarray([row["pair_rwiv_ratio"] for row in summaries], dtype=float),
        "pair_others_ratio": np.asarray([row["pair_others_ratio"] for row in summaries], dtype=float),
        "sample_mean_f1": np.asarray([row["sample_mean_f1"] for row in summaries], dtype=float),
        "inplane_macro_f1": np.asarray([row["inplane_macro_f1"] for row in summaries], dtype=float),
        "outplane_macro_f1": np.asarray([row["outplane_macro_f1"] for row in summaries], dtype=float),
        "joint_viv_rwiv_mean_f1": np.asarray([row["joint_viv_rwiv_mean_f1"] for row in summaries], dtype=float),
        "pred_normal_ratio": np.asarray([row["pred_normal_ratio"] for row in summaries], dtype=float),
        "pred_viv_ratio": np.asarray([row["pred_viv_ratio"] for row in summaries], dtype=float),
        "pred_rwiv_ratio": np.asarray([row["pred_rwiv_ratio"] for row in summaries], dtype=float),
        "pred_others_ratio": np.asarray([row["pred_others_ratio"] for row in summaries], dtype=float),
        "pred_abnormal_ratio": np.asarray([row["pred_abnormal_ratio"] for row in summaries], dtype=float),
        "prediction_flip_ratio": np.asarray([row["prediction_flip_ratio"] for row in summaries], dtype=float),
        "mean_uncertainty": np.asarray([row["mean_uncertainty"] for row in summaries], dtype=float),
        "high_uncertainty_ratio": np.asarray([row["high_uncertainty_ratio"] for row in summaries], dtype=float),
    }


def slice_arrays(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    sliced = dict(arrays)
    for key, values in arrays.items():
        if key == "rounds":
            sliced[key] = values[mask]
        elif isinstance(values, np.ndarray) and values.shape == arrays["rounds"].shape:
            sliced[key] = values[mask]
    return sliced


def dataset_scale_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = arrays["rounds"] >= DATASET_START_ROUND
    return slice_arrays(arrays, mask)


def dataset_delta_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    total_delta = arrays["pair_added_total"] + arrays["pair_modified_total"] + arrays["pair_removed_total"]
    mask = (arrays["rounds"] >= DATASET_START_ROUND) & (total_delta > 0)
    return slice_arrays(arrays, mask)


def plot_perplexity(arrays: dict[str, np.ndarray]) -> plt.Figure:
    rounds = arrays["rounds"]
    palette = get_full_color_map(style="discrete").colors
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    ax.plot(rounds, arrays["mean_perplexity"], marker="o", linewidth=3.0, markersize=8, color=palette[0], label="融合预测均值")
    ax.plot(rounds, arrays["p90_perplexity"], marker="s", linewidth=2.6, markersize=7, color=palette[9], label="融合预测90分位")
    ax.plot(rounds, arrays["inplane_perplexity"], marker="^", linewidth=2.4, markersize=7, color=VIV_INPLANE_COLOR, label="面内预测均值")
    ax.plot(rounds, arrays["outplane_perplexity"], marker="v", linewidth=2.4, markersize=7, color=VIV_OUTPLANE_COLOR, label="面外预测均值")
    ax.set_ylim(1.0, 4.05)
    ax.set_xticks(rounds)
    setup_axis(ax, "终盘重训后多轮模型困惑度变化", "困惑度")
    ax.legend(frameon=False, fontsize=SQUARE_FONT_SIZE - 9, prop=CN_FONT, ncol=2, loc="upper right")
    annotate_points(ax, rounds, arrays["mean_perplexity"])
    fig.tight_layout()
    return fig


def plot_median_perplexity(arrays: dict[str, np.ndarray]) -> plt.Figure:
    rounds = arrays["rounds"]
    palette = get_full_color_map(style="discrete").colors
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    ax.plot(rounds, arrays["median_perplexity"], marker="o", linewidth=3.0, markersize=8, color=palette[0], label="融合预测中位数")
    ax.plot(rounds, arrays["inplane_median_perplexity"], marker="^", linewidth=2.4, markersize=7, color=VIV_INPLANE_COLOR, label="面内预测中位数")
    ax.plot(rounds, arrays["outplane_median_perplexity"], marker="v", linewidth=2.4, markersize=7, color=VIV_OUTPLANE_COLOR, label="面外预测中位数")
    ax.set_ylim(1.0, 4.05)
    ax.set_xticks(rounds)
    setup_axis(ax, "终盘重训后全量数据中位数困惑度", "困惑度")
    ax.legend(frameon=False, fontsize=SQUARE_FONT_SIZE - 9, prop=CN_FONT, ncol=2, loc="upper right")
    annotate_points(ax, rounds, arrays["median_perplexity"])
    fig.tight_layout()
    return fig


def plot_sample_mean_f1(arrays: dict[str, np.ndarray]) -> plt.Figure:
    rounds = arrays["rounds"]
    palette = get_full_color_map(style="discrete").colors
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    ax.plot(rounds, arrays["sample_mean_f1"], marker="o", linewidth=3.0, markersize=8, color=palette[0], label="样本平均F1")
    ax.plot(rounds, arrays["inplane_macro_f1"], marker="^", linewidth=2.4, markersize=7, color=VIV_INPLANE_COLOR, label="面内平均F1")
    ax.plot(rounds, arrays["outplane_macro_f1"], marker="v", linewidth=2.4, markersize=7, color=VIV_OUTPLANE_COLOR, label="面外平均F1")
    ax.plot(rounds, arrays["joint_viv_rwiv_mean_f1"], marker="s", linewidth=2.4, markersize=7, color=palette[9], label="VIV/RWIV平均F1")
    ax.set_ylim(0.55, 1.02)
    ax.set_xticks(rounds)
    setup_axis(ax, "终盘重训后每轮样本平均F1变化", "F1")
    ax.legend(frameon=False, fontsize=SQUARE_FONT_SIZE - 9, prop=CN_FONT, ncol=2, loc="lower right")
    annotate_points(ax, rounds, arrays["sample_mean_f1"], "{:.3f}")
    fig.tight_layout()
    return fig


def plot_annotation_scale(arrays: dict[str, np.ndarray]) -> plt.Figure:
    data = dataset_scale_arrays(arrays)
    rounds = data["rounds"]
    palette = get_full_color_map(style="discrete").colors
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    width = 0.22
    ax.bar(rounds - width, data["train_pair_total"], width=width, color=palette[0], label="训练 pair 标注")
    ax.bar(rounds, data["val_pair_total"], width=width, color=palette[7], label="验证 pair 标注")
    ax.bar(rounds + width, data["pair_key_total"], width=width, color=palette[2], label="面内面外 pair 合计")
    ax.set_xticks(rounds)
    setup_axis(ax, "训练+验证标注集规模", "pair 标注数")
    ax.legend(frameon=False, fontsize=SQUARE_FONT_SIZE - 9, prop=CN_FONT, loc="upper left")
    fig.tight_layout()
    return fig


def plot_annotation_delta(arrays: dict[str, np.ndarray]) -> plt.Figure:
    data = dataset_delta_arrays(arrays)
    rounds = data["rounds"]
    palette = get_full_color_map(style="discrete").colors
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    ax.bar(rounds, data["pair_added_total"], color=palette[1], label="新增 pair")
    ax.bar(
        rounds,
        data["pair_modified_total"],
        bottom=data["pair_added_total"],
        color=palette[8],
        label="标签修改 pair",
    )
    ax.bar(
        rounds,
        data["pair_removed_total"],
        bottom=data["pair_added_total"] + data["pair_modified_total"],
        color=palette[11],
        label="移除 pair",
    )
    ax.set_xticks(rounds)
    setup_axis(ax, "相邻轮次标注集变动", "pair 数")
    ax.legend(frameon=False, fontsize=SQUARE_FONT_SIZE - 9, prop=CN_FONT, loc="upper right")
    fig.tight_layout()
    return fig


def plot_inference_prediction_dist(arrays: dict[str, np.ndarray]) -> plt.Figure:
    rounds = arrays["rounds"]
    palette = get_full_color_map(style="discrete").colors
    class_colors = [palette[2], VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR, palette[10]]
    fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
    bottom = np.zeros_like(rounds, dtype=float)
    class_ratios = (
        arrays["pred_normal_ratio"],
        arrays["pred_viv_ratio"],
        arrays["pred_rwiv_ratio"],
        arrays["pred_others_ratio"],
    )
    for ratio, name, color in zip(class_ratios, CLASS_NAMES, class_colors):
        ax.bar(rounds, ratio, bottom=bottom, color=color, label=name, width=0.62)
        bottom = bottom + ratio
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(rounds)
    setup_axis(ax, "全量识别预测类别占比", "比例")
    ax.legend(frameon=False, fontsize=SQUARE_FONT_SIZE - 9, prop=CN_FONT, ncol=2, loc="upper right")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    return fig


def plot_summaries(summaries: list[dict[str, Any]]) -> list[plt.Figure]:
    arrays = extract_plot_arrays(summaries)
    return [
        plot_perplexity(arrays),
        plot_median_perplexity(arrays),
        plot_sample_mean_f1(arrays),
        plot_annotation_scale(arrays),
        plot_annotation_delta(arrays),
        plot_inference_prediction_dist(arrays),
    ]


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    print(f"saved: {path}")
    return path


def save_summary(summaries: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fig3_37_39_iterative_dataset_change_summary.json"
    cleaned = [
        {
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in row.items()
        }
        for row in summaries
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"saved: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 final rounds 绘制多轮困惑度与数据集变动图")
    parser.add_argument("--rounds-root", type=Path, default=DEFAULT_FINAL_ROUNDS_ROOT)
    parser.add_argument("--final-manifest", type=Path, default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--split-indices", type=Path, default=DEFAULT_SPLIT_INDICES)
    parser.add_argument("--max-round", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=DASHBOARD_DEFAULT_PORT)
    parser.add_argument("--no-web", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round_dirs = discover_final_round_dirs(args.rounds_root, max_round=args.max_round)
    final_reports = load_final_round_reports(args.final_manifest)
    summaries = collect_round_summaries(round_dirs, final_reports, args.split_indices)
    figs = plot_summaries(summaries)
    output_paths: list[Path] = []
    for slot, (fig, (name, title)) in enumerate(zip(figs, FIGURE_SPECS)):
        output_paths.append(save_figure(fig, args.output_dir, name))
        if not args.no_web:
            web_push(fig, page=WEBUI_PAGE, slot=slot, title=title, port=args.port, page_cols=3)
        plt.close(fig)
    output_paths.append(save_summary(summaries, args.output_dir))
    if not args.no_web:
        print(f"pushed to VibDash page: {WEBUI_PAGE}")
    for path in output_paths:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
