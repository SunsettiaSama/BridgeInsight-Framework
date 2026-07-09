from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.train.visualize import save_confusion_matrix
from src.chapter3_identifier.augment_eval_compare.infer.run_compare import load_predictions
from src.chapter3_identifier.augment_eval_compare.metrics.pair_metrics import compute_delta, compute_pair_metrics
from src.chapter3_identifier.augment_eval_compare.settings import get_compare_output_dir


def _extract_labels(rows: list[dict[str, Any]]) -> tuple[list[int], list[int], list[int], list[int]]:
    in_true = [int(r["inplane_annotation"]) for r in rows]
    out_true = [int(r["outplane_annotation"]) for r in rows]
    in_pred = [int(r["inplane_prediction"]) for r in rows]
    out_pred = [int(r["outplane_prediction"]) for r in rows]
    return in_true, out_true, in_pred, out_pred


def _model_summary(rows: list[dict[str, Any]], label_names: list[str]) -> dict[str, Any]:
    in_true, out_true, in_pred, out_pred = _extract_labels(rows)
    metrics = compute_pair_metrics(
        in_true,
        out_true,
        in_pred,
        out_pred,
        label_names=label_names,
    )
    first = rows[0] if rows else {}
    return {
        "checkpoint": first.get("checkpoint"),
        "model_type": first.get("model_type"),
        **metrics,
    }


def _validate_same_eval_pairs(dec_rows: list[dict[str, Any]], joint_rows: list[dict[str, Any]]) -> None:
    if len(dec_rows) != len(joint_rows):
        raise ValueError(
            f"预测条数不一致：decoupled={len(dec_rows)} joint={len(joint_rows)}"
        )

    compare_keys = (
        "pair_key",
        "inplane_annotation",
        "outplane_annotation",
        "window_index",
    )
    for idx, (dec, joint) in enumerate(zip(dec_rows, joint_rows)):
        for key in compare_keys:
            if dec.get(key) != joint.get(key):
                raise ValueError(
                    "两组预测不是同一评估样本，"
                    f"idx={idx} key={key} decoupled={dec.get(key)!r} joint={joint.get(key)!r}"
                )


def build_compare_report(
    cfg: dict,
    round_idx: int,
    *,
    decoupled_predictions_path: Path | None = None,
    joint_predictions_path: Path | None = None,
    save_confusion: bool = True,
) -> Path:
    output_dir = get_compare_output_dir(cfg, round_idx)
    dec_path = decoupled_predictions_path or (output_dir / "predictions_decoupled.json")
    joint_path = joint_predictions_path or (output_dir / "predictions_joint.json")
    if not dec_path.exists():
        raise FileNotFoundError(f"缺少 decoupled 预测：{dec_path}")
    if not joint_path.exists():
        raise FileNotFoundError(f"缺少 joint 预测：{joint_path}")

    dec_rows = load_predictions(dec_path)
    joint_rows = load_predictions(joint_path)
    _validate_same_eval_pairs(dec_rows, joint_rows)

    label_names = get_label_names()
    decoupled = _model_summary(dec_rows, label_names)
    joint = _model_summary(joint_rows, label_names)
    delta = compute_delta(decoupled, joint)

    report = {
        "round_idx": int(round_idx),
        "eval_split": str(cfg.get("eval_split", "val")),
        "eval_pair_count": len(dec_rows),
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "comparison_scope": {
            "same_eval_pairs": True,
            "eval_pair_source": str(dec_path.parent / "eval_manifest.json"),
        },
        "decoupled": decoupled,
        "joint": joint,
        "delta": delta,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "compare_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if save_confusion:
        in_true, out_true, _, _ = _extract_labels(dec_rows)
        _, _, in_pred_dec, out_pred_dec = _extract_labels(dec_rows)
        _, _, in_pred_joint, out_pred_joint = _extract_labels(joint_rows)
        save_confusion_matrix(
            in_true,
            in_pred_dec,
            output_dir / "confusion_inplane_decoupled.png",
            label_names=label_names,
        )
        save_confusion_matrix(
            out_true,
            out_pred_dec,
            output_dir / "confusion_outplane_decoupled.png",
            label_names=label_names,
        )
        save_confusion_matrix(
            in_true,
            in_pred_joint,
            output_dir / "confusion_inplane_joint.png",
            label_names=label_names,
        )
        save_confusion_matrix(
            out_true,
            out_pred_joint,
            output_dir / "confusion_outplane_joint.png",
            label_names=label_names,
        )

    return report_path
