from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def annotation_key(file_path: str, window_index: int) -> Tuple[str, int]:
    return (os.path.normcase(str(Path(file_path).resolve())), int(window_index))


def build_gold_index(entries: Iterable[dict]) -> Dict[Tuple[str, int], dict]:
    index: Dict[Tuple[str, int], dict] = {}
    for entry in entries:
        fp = entry.get("file_path")
        wi = entry.get("window_index", 0)
        if fp is None:
            continue
        index[annotation_key(fp, wi)] = entry
    return index
