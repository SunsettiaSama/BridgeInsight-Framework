from __future__ import annotations

import logging
import os
import sys
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


def plain_progress_enabled() -> bool:
    if os.environ.get("AUGMENT_PLAIN_LOG") == "1":
        return True
    err = sys.stderr
    return not (hasattr(err, "isatty") and err.isatty())


class _PlainProgressBar:
    def __init__(self, total: int, desc: str = "", unit: str = "", log_every_pct: float = 5.0):
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.log_every_pct = log_every_pct
        self.n = 0
        self._last_logged_pct = -log_every_pct

    def update(self, n: int = 1) -> None:
        if n <= 0:
            return
        self.n = min(self.n + n, self.total) if self.total else self.n + n
        self._maybe_log()

    def _maybe_log(self, force: bool = False) -> None:
        if self.total <= 0:
            return
        pct = 100.0 * self.n / self.total
        if force or pct - self._last_logged_pct >= self.log_every_pct or self.n >= self.total:
            logger.info(f"{self.desc}：{self.n}/{self.total} ({pct:.1f}%)")
            self._last_logged_pct = pct

    def __enter__(self) -> "_PlainProgressBar":
        return self

    def __exit__(self, *args) -> None:
        if self.total > 0 and self.n < self.total:
            self.n = self.total
            self._maybe_log(force=True)


def progress_bar(
    total: int,
    *,
    desc: str = "",
    unit: str = "",
    log_every_pct: float = 5.0,
):
    if plain_progress_enabled():
        return _PlainProgressBar(total, desc=desc, unit=unit, log_every_pct=log_every_pct)
    from tqdm import tqdm

    return tqdm(total=total, desc=desc, unit=unit)


def iter_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "",
    log_every_pct: float = 5.0,
) -> Iterator[T]:
    if not plain_progress_enabled():
        from tqdm import tqdm

        yield from tqdm(iterable, total=total, desc=desc, unit=unit)
        return

    if total is None:
        total = len(iterable) if hasattr(iterable, "__len__") else None

    last_logged_pct = -log_every_pct
    for i, item in enumerate(iterable):
        yield item
        if total is None or total <= 0:
            continue
        done = i + 1
        pct = 100.0 * done / total
        if pct - last_logged_pct >= log_every_pct or done == total:
            logger.info(f"{desc}：{done}/{total} ({pct:.1f}%)")
            last_logged_pct = pct
