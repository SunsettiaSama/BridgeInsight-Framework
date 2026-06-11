from __future__ import annotations

from typing import Dict, List, Optional

_DEFAULT_LABEL_NAMES = ["Normal", "VIV", "RWIV", "Others"]


def get_label_names(cfg: Optional[dict] = None) -> List[str]:
    if cfg is None:
        return list(_DEFAULT_LABEL_NAMES)
    num_classes = int(cfg.get("num_classes", len(_DEFAULT_LABEL_NAMES)))
    names = cfg.get("label_names")
    if names is None:
        if num_classes <= len(_DEFAULT_LABEL_NAMES):
            return list(_DEFAULT_LABEL_NAMES[:num_classes])
        return [f"Class_{i}" for i in range(num_classes)]
    names = [str(x) for x in names]
    if len(names) != num_classes:
        raise ValueError(
            f"label_names 长度 {len(names)} 与 num_classes={num_classes} 不一致，"
            "请在 default.yaml 中同步扩充两者"
        )
    return names


def label_name(class_id: int, cfg: Optional[dict] = None) -> str:
    names = get_label_names(cfg)
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def label_name_map(cfg: dict) -> Dict[int, str]:
    return {i: name for i, name in enumerate(get_label_names(cfg))}
