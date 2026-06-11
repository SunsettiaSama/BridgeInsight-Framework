from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional


class PngCache:
    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            png = self._items.get(key)
            if png is None:
                return None
            self._items.move_to_end(key)
            return png

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)

    def has_all(self, keys: list[str]) -> bool:
        with self._lock:
            return all(key in self._items for key in keys)

    def size(self) -> int:
        with self._lock:
            return len(self._items)
