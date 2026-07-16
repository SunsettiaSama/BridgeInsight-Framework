"""面内/面外任一侧命中特殊振动时的窗口类别合并规则。"""

from __future__ import annotations

SPECIAL_CLASSES = (1, 2, 3)

PREDICTION_OVERRIDE_RULE = "any_side_special"
PREDICTION_OVERRIDE_NOTE = (
    "面内或面外任一方向识别为 1（涡激共振）、2（风雨振）、3（其他振动）时，"
    "即将该窗口计入对应特殊振动类别；两个方向均为 0 时计入随机振动。"
    "若两面同时命中不同特殊类，按 1 > 2 > 3 优先级取类。"
)


def merge_any_side_prediction(in_pred: int, out_pred: int) -> int:
    in_label = int(in_pred)
    out_label = int(out_pred)
    for cls_id in SPECIAL_CLASSES:
        if in_label == cls_id or out_label == cls_id:
            return cls_id
    return 0
