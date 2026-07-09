from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from src.chapter4_characteristics.settings import CLASS_DIRS, get_enriched_dir

logger = logging.getLogger(__name__)

_BATCH_STEM_RE = re.compile(r"^(.+)_batch_(\d{5})$")


def is_batch_json(path: Path) -> bool:
    return _BATCH_STEM_RE.match(path.stem) is not None


def is_canonical_json(path: Path) -> bool:
    return path.suffix == ".json" and not is_batch_json(path)


def parse_batch_sensor_id(stem: str) -> str:
    match = _BATCH_STEM_RE.match(stem)
    if match is None:
        return stem
    return match.group(1)


def list_batch_json_files(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        path for path in class_dir.glob("*.json")
        if is_batch_json(path)
    )


def list_canonical_json_files(class_dir: Path, excluded_sensor_ids: Optional[Set[str]] = None) -> List[Path]:
    if not class_dir.exists():
        return []
    excluded = excluded_sensor_ids or set()
    return sorted(
        path for path in class_dir.glob("*.json")
        if is_canonical_json(path) and path.stem not in excluded
    )


def _group_batch_files_by_sensor(batch_files: Iterable[Path]) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for batch_path in batch_files:
        sensor_id = parse_batch_sensor_id(batch_path.stem)
        grouped[sensor_id].append(batch_path)
    for sensor_id in grouped:
        grouped[sensor_id] = sorted(grouped[sensor_id], key=lambda path: path.name)
    return dict(grouped)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_samples(batch_files: List[Path]) -> List[dict]:
    merged: Dict[int, dict] = {}
    for batch_path in batch_files:
        payload = _read_json(batch_path)
        for sample in payload.get("samples", []):
            sample_idx = sample.get("sample_idx")
            if sample_idx is None:
                raise ValueError(f"batch 样本缺少 sample_idx：{batch_path}")
            merged[int(sample_idx)] = sample
    return [merged[idx] for idx in sorted(merged)]


def _stored_batch_names(canonical_path: Path) -> Set[str]:
    payload = _read_json(canonical_path)
    metadata = payload.get("metadata", {})
    stored = metadata.get("compacted_batch_files")
    if stored is None:
        return set()
    return {str(name) for name in stored}


def _needs_rebuild(
    canonical_path: Path,
    current_batch_names: List[str],
    force: bool,
) -> bool:
    if force or not canonical_path.exists():
        return True
    stored_names = _stored_batch_names(canonical_path)
    current_names = set(current_batch_names)
    return stored_names != current_names


def _write_canonical_json(
    canonical_path: Path,
    batch_files: List[Path],
    samples: List[dict],
) -> None:
    first_payload = _read_json(batch_files[0])
    base_meta = dict(first_payload.get("metadata", {}))
    batch_names = [batch_path.name for batch_path in batch_files]

    base_meta.update({
        "sensor_id": parse_batch_sensor_id(batch_files[0].stem),
        "num_samples": len(samples),
        "compacted_from_batch_count": len(batch_files),
        "compacted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "compacted_batch_files": batch_names,
    })

    payload = {
        "metadata": base_meta,
        "samples": samples,
    }
    tmp_path = canonical_path.with_name(f"{canonical_path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp_path.replace(canonical_path)


def compact_class_dir(
    class_dir: Path,
    force: bool = False,
    excluded_sensor_ids: Optional[Set[str]] = None,
) -> dict:
    excluded = excluded_sensor_ids or set()
    batch_files = list_batch_json_files(class_dir)
    if not batch_files:
        return {"class_dir": str(class_dir), "written": 0, "skipped": 0, "sensors": 0}

    grouped = _group_batch_files_by_sensor(batch_files)
    written = 0
    skipped = 0

    for sensor_id, sensor_batches in sorted(grouped.items()):
        if sensor_id in excluded:
            continue

        batch_names = [batch_path.name for batch_path in sensor_batches]
        canonical_path = class_dir / f"{sensor_id}.json"
        if not _needs_rebuild(canonical_path, batch_names, force=force):
            skipped += 1
            continue

        samples = _merge_samples(sensor_batches)
        _write_canonical_json(canonical_path, sensor_batches, samples)
        written += 1
        logger.info(
            "  compact %s ← %d batch, %d samples",
            canonical_path.name,
            len(sensor_batches),
            len(samples),
        )

    return {
        "class_dir": str(class_dir),
        "written": written,
        "skipped": skipped,
        "sensors": len(grouped),
    }


def compact_enriched_dir(
    enriched_dir: Path | str,
    force: bool = False,
    class_ids: Optional[Iterable[int]] = None,
    excluded_sensor_ids: Optional[Set[str]] = None,
) -> dict:
    root = Path(enriched_dir)
    if not root.exists():
        raise FileNotFoundError(f"enriched 目录不存在：{root}")

    target_class_ids = list(class_ids) if class_ids is not None else sorted(CLASS_DIRS)
    summary = {
        "enriched_dir": str(root),
        "force": force,
        "classes": {},
        "written": 0,
        "skipped": 0,
    }

    for class_id in target_class_ids:
        class_name = CLASS_DIRS[class_id]
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        result = compact_class_dir(
            class_dir,
            force=force,
            excluded_sensor_ids=excluded_sensor_ids,
        )
        summary["classes"][class_name] = result
        summary["written"] += result["written"]
        summary["skipped"] += result["skipped"]

    logger.info(
        "enriched compact 完成：written=%d skipped=%d dir=%s",
        summary["written"],
        summary["skipped"],
        root,
    )
    return summary


def ensure_class_dir_compacted(
    class_dir: Path,
    cfg: dict,
    force: bool = False,
    excluded_sensor_ids: Optional[Set[str]] = None,
) -> None:
    batch_files = list_batch_json_files(class_dir)
    if not batch_files:
        return

    auto_compact = bool(cfg.get("auto_compact_on_read", True))
    canonical_files = list_canonical_json_files(class_dir, excluded_sensor_ids=excluded_sensor_ids)
    if canonical_files and not force:
        compact_class_dir(
            class_dir,
            force=False,
            excluded_sensor_ids=excluded_sensor_ids,
        )
        return

    if not auto_compact and not canonical_files:
        raise FileNotFoundError(
            f"类别目录仅有 batch 文件，未找到 canonical JSON：{class_dir}\n"
            "请先运行：python -m src.chapter4_characteristics enrich --compact-only"
        )

    compact_class_dir(
        class_dir,
        force=force or bool(cfg.get("compact_enriched_force", False)),
        excluded_sensor_ids=excluded_sensor_ids,
    )

    canonical_files = list_canonical_json_files(class_dir, excluded_sensor_ids=excluded_sensor_ids)
    if not canonical_files:
        raise FileNotFoundError(
            f"compact 后仍无 canonical JSON：{class_dir}"
        )


def ensure_enriched_compacted(
    cfg: dict,
    class_id: Optional[int] = None,
    force: bool = False,
) -> Path:
    enriched_dir = get_enriched_dir(cfg)
    excluded = {
        str(sensor_id)
        for sensor_id in (cfg.get("infer_exclude_sensor_ids") or cfg.get("exclude_sensor_ids") or [])
    }

    if class_id is None:
        compact_enriched_dir(
            enriched_dir,
            force=force or bool(cfg.get("compact_enriched_force", False)),
            excluded_sensor_ids=excluded,
        )
        return enriched_dir

    class_dir = enriched_dir / CLASS_DIRS[class_id]
    ensure_class_dir_compacted(
        class_dir,
        cfg=cfg,
        force=force,
        excluded_sensor_ids=excluded,
    )
    return enriched_dir
