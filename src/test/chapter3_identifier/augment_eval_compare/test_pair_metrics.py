from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.chapter3_identifier.augment_eval_compare.metrics.pair_metrics import (
    compute_delta,
    compute_pair_metrics,
    pair_error_breakdown,
    pair_joint_accuracy,
)


def test_pair_joint_accuracy_all_correct():
    in_true = [0, 1, 2]
    out_true = [0, 1, 3]
    in_pred = [0, 1, 2]
    out_pred = [0, 1, 3]
    assert pair_joint_accuracy(in_true, out_true, in_pred, out_pred) == 1.0


def test_pair_joint_accuracy_partial():
    in_true = [0, 1]
    out_true = [0, 1]
    in_pred = [0, 1]
    out_pred = [1, 1]
    assert pair_joint_accuracy(in_true, out_true, in_pred, out_pred) == 0.5


def test_pair_error_breakdown():
    in_true = [0, 1, 2, 3]
    out_true = [0, 1, 2, 3]
    in_pred = [1, 1, 2, 0]
    out_pred = [0, 2, 0, 3]
    breakdown = pair_error_breakdown(in_true, out_true, in_pred, out_pred)
    assert breakdown == {"in_only": 2, "out_only": 2, "both": 0}


def test_compute_pair_metrics_and_delta():
    in_true = [0, 1, 1, 2]
    out_true = [0, 1, 2, 2]
    in_pred = [0, 1, 2, 2]
    out_pred = [0, 1, 1, 3]
    dec = compute_pair_metrics(in_true, out_true, in_pred, out_pred)
    joint = compute_pair_metrics(in_true, out_true, in_true, out_true)
    assert dec["pair_joint_accuracy"] == 0.5
    assert joint["pair_joint_accuracy"] == 1.0
    delta = compute_delta(dec, joint)
    assert delta["pair_joint_accuracy"] == 0.5
    assert "direction_metrics" in dec
