#!/usr/bin/env python3
"""CS-45 drop-in for the Smart Turn upstream ``record_and_predict.SileroVAD`` / ``ensure_model``.

``ensure_model`` returns the pinned local path after a SHA-256 check (it never downloads);
``SileroVAD(path).prob(chunk512)`` is the streaming probability of the single pinned Silero.
"""
from __future__ import annotations

import numpy as np

import tt_silero as _s


def ensure_model(_ignored_path: str | None = None) -> str:
    got = _s.sha256_file(_s.DEFAULT_MODEL)
    if got != _s.PINNED_SHA256:
        raise _s.PinMismatch(f"pinned Silero at {_s.DEFAULT_MODEL}: {got} != {_s.PINNED_SHA256}")
    return str(_s.DEFAULT_MODEL)


class SileroVAD:
    def __init__(self, path: str | None = None):
        self._m = _s.PinnedSilero(path or None)
        self._st = self._m.stream()

    def reset(self):
        self._st = self._m.stream()

    def prob(self, chunk: np.ndarray) -> float:
        return self._st.feed(np.asarray(chunk, dtype=np.float32))
