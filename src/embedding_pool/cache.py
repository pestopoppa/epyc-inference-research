"""sha256-keyed LRU cache of embedding vectors.

The key binds the text to the embedding MODEL (invariant 4: an index belongs to one model),
so two pools of different models can never serve each other's vectors. Only vectors that
passed the degenerate-vector guard are stored; they are stored read-only.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np


def chunk_key(model_id: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class EmbeddingLRUCache:
    def __init__(self, max_entries: int = 20000) -> None:
        self.max_entries = max(0, int(max_entries))
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            vec = self._data.get(key)
            if vec is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        if self.max_entries == 0:
            return
        stored = np.array(vec, dtype=np.float32, copy=True)
        stored.setflags(write=False)
        with self._lock:
            self._data[key] = stored
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


__all__ = ["EmbeddingLRUCache", "chunk_key"]
