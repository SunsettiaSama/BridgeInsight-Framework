import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

CLASS_LABELS: Dict[int, str] = {
    0: "normal",
    1: "viv",
    2: "rwiv",
    3: "transition",
}

CLASS_NAMES_CN: Dict[int, str] = {
    0: "正常振动",
    1: "涡激振动",
    2: "随机风致振动",
    3: "过渡状态",
}


def split_samples_by_class(
    enriched_samples: List[Dict],
) -> Dict[int, List[Dict]]:
    """
    按预测类别将样本分组。
    enriched_samples 中每条记录需含 "predicted_class" 键。
    返回 {class_id: [sample_dict, ...]}
    """
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for sample in enriched_samples:
        cls = sample.get("predicted_class", -1)
        groups[cls].append(sample)
    return dict(groups)


def split_samples_by_class_and_sensor(
    enriched_samples: List[Dict],
) -> Dict[int, Dict[str, List[Dict]]]:
    """
    按预测类别再按面内传感器 ID 二级分组。
    返回 {class_id: {sensor_id: [sample_dict, ...]}}
    """
    by_class_sensor: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for sample in enriched_samples:
        cls = sample.get("predicted_class", -1)
        sid = sample.get("inplane_sensor_id", "unknown")
        by_class_sensor[cls][sid].append(sample)
    return {cls: dict(sensors) for cls, sensors in by_class_sensor.items()}


def save_class_results(
    enriched_samples: List[Dict],
    output_dir: str,
    source_result_name: str = "",
    split_by_sensor: bool = True,
) -> Dict[int, Path]:
    """
    将 enriched_samples 按类别（可选再按传感器）分割后保存为独立 JSON 文件。

    文件命名规则
    -----------
    - 按类别：  class_{id}_{label}.json
    - 按传感器：class_{id}_{label}/{sensor_id}.json

    返回 {class_id: output_path}（仅记录各类根文件路径）
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_class = split_samples_by_class(enriched_samples)
    saved_paths: Dict[int, Path] = {}

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for cls_id, samples in sorted(by_class.items()):
        label = CLASS_LABELS.get(cls_id, f"class_{cls_id}")
        label_cn = CLASS_NAMES_CN.get(cls_id, f"未知类别{cls_id}")

        file_meta = {
            "created_at":        created_at,
            "source_result":     source_result_name,
            "class_id":          cls_id,
            "class_label":       label,
            "class_label_cn":    label_cn,
            "num_samples":       len(samples),
        }

        if split_by_sensor:
            cls_dir = out_dir / f"class_{cls_id}_{label}"
            cls_dir.mkdir(parents=True, exist_ok=True)

            by_sensor: Dict[str, List[Dict]] = defaultdict(list)
            for s in samples:
                sid = s.get("inplane_sensor_id", "unknown")
                by_sensor[sid].append(s)

            for sensor_id, sensor_samples in sorted(by_sensor.items()):
                sensor_path = cls_dir / f"{sensor_id}.json"
                payload = {
                    "metadata": {**file_meta, "sensor_id": sensor_id, "num_samples": len(sensor_samples)},
                    "samples":  sensor_samples,
                }
                with open(sensor_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                logger.info(f"  → {sensor_path.relative_to(out_dir)}  ({len(sensor_samples)} 条)")

            saved_paths[cls_id] = cls_dir
        else:
            cls_path = out_dir / f"class_{cls_id}_{label}.json"
            payload = {
                "metadata": file_meta,
                "samples":  samples,
            }
            with open(cls_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            logger.info(f"  → {cls_path.name}  ({len(samples)} 条)")
            saved_paths[cls_id] = cls_path

    return saved_paths
