from __future__ import annotations

import threading
import time

from src.chapter3_identifier.augment.figures.cache import PngCache


class RenderDispatchCenter:
    """Render dispatch center: cache notifications + waiter coordination."""

    def __init__(self, cache: PngCache) -> None:
        self._cache = cache
        self._cv = threading.Condition()
        self._epoch = 0
        self._cache.subscribe(self._on_cache_put)

    def _on_cache_put(self, _key: str) -> None:
        with self._cv:
            self._cv.notify_all()

    def epoch(self) -> int:
        with self._cv:
            return self._epoch

    def mark_jump_reset(self) -> int:
        with self._cv:
            self._epoch += 1
            self._cv.notify_all()
            return self._epoch

    def wait_for_keys(
        self,
        keys: list[str],
        timeout_ms: int,
        *,
        expected_epoch: int | None = None,
    ) -> bool:
        if not keys:
            return True
        deadline = time.monotonic() + max(0.0, timeout_ms) / 1000.0
        with self._cv:
            while True:
                if expected_epoch is not None and expected_epoch != self._epoch:
                    return False
                if self._cache.has_all(keys):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=min(0.2, remaining))
