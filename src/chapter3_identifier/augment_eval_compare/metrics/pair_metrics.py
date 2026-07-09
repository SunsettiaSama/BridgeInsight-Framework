from __future__ import annotations

from typing import Any

import numpy as np

from src.chapter3_identifier.augment.train.visualize import compute_class_metrics


def pair_joint_accuracy(
    in_true: list[int],
    out_true: list[int],
    in_pred: list[int],
    out_pred: list[int],
) -> float:
    if not in_true:
        return 0.0
    correct = [
        (yt_in == yp_in) and (yt_out == yp_out)
        for yt_in, yt_out, yp_in, yp_out in zip(in_true, out_true, in_pred, out_pred)
    ]
    return float(np.mean(correct))


def pair_error_breakdown(
    in_true: list[int],
    out_true: list[int],
    in_pred: list[int],
    out_pred: list[int],
) -> dict[str, int]:
    in_only = 0
    out_only = 0
    both = 0
    for yt_in, yt_out, yp_in, yp_out in zip(in_true, out_true, in_pred, out_pred):
        in_ok = yt_in == yp_in
        out_ok = yt_out == yp_out
        if in_ok and out_ok:
            continue
        if not in_ok and not out_ok:
            both += 1
        elif not in_ok:
            in_only += 1
        else:
            out_only += 1
    return {"in_only": in_only, "out_only": out_only, "both": both}


def compute_pair_metrics(
    in_true: list[int],
    out_true: list[int],
    in_pred: list[int],
    out_pred: list[int],
    *,
    label_names: list[str] | None = None,
    num_classes: int = 4,
) -> dict[str, Any]:
    inplane_metrics = compute_class_metrics(
        in_true,
        in_pred,
        label_names=label_names,
        num_classes=num_classes,
    )
    outplane_metrics = compute_class_metrics(
        out_true,
        out_pred,
        label_names=label_names,
        num_classes=num_classes,
    )
    prefixed_in = {f"inplane_{k}": v for k, v in inplane_metrics.items()}
    prefixed_out = {f"outplane_{k}": v for k, v in outplane_metrics.items()}
    return {
        "pair_joint_accuracy": pair_joint_accuracy(in_true, out_true, in_pred, out_pred),
        "inplane_accuracy": inplane_metrics["accuracy"],
        "outplane_accuracy": outplane_metrics["accuracy"],
        "pair_error_breakdown": pair_error_breakdown(in_true, out_true, in_pred, out_pred),
        "direction_metrics": {
            **prefixed_in,
            **prefixed_out,
        },
    }


def compute_delta(decoupled: dict[str, Any], joint: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key in ("pair_joint_accuracy", "inplane_accuracy", "outplane_accuracy"):
        if key in decoupled and key in joint:
            delta[key] = float(joint[key]) - float(decoupled[key])
    in_dec = decoupled.get("direction_metrics", {}).get("inplane_viv_rwiv_mean_f1")
    in_joint = joint.get("direction_metrics", {}).get("inplane_viv_rwiv_mean_f1")
    out_dec = decoupled.get("direction_metrics", {}).get("outplane_viv_rwiv_mean_f1")
    out_joint = joint.get("direction_metrics", {}).get("outplane_viv_rwiv_mean_f1")
    if in_dec is not None and in_joint is not None:
        delta["inplane_viv_rwiv_mean_f1"] = float(in_joint) - float(in_dec)
    if out_dec is not None and out_joint is not None:
        delta["outplane_viv_rwiv_mean_f1"] = float(out_joint) - float(out_dec)
    return delta
