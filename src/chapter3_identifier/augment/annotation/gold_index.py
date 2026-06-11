from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_PATH_NORM_CACHE: Dict[str, str] = {}


def annotation_key(file_path: str, window_index: int) -> Tuple[str, int]:
    norm = _PATH_NORM_CACHE.get(file_path)
    if norm is None:
        norm = os.path.normcase(str(Path(file_path).resolve()))
        _PATH_NORM_CACHE[file_path] = norm
    return (norm, int(window_index))


def build_gold_index(entries: Iterable[dict]) -> Dict[Tuple[str, int], dict]:
    index: Dict[Tuple[str, int], dict] = {}
    for entry in entries:
        fp = entry.get("file_path")
        wi = entry.get("window_index", 0)
        if fp is None:
            continue
        index[annotation_key(fp, wi)] = entry
    return index
