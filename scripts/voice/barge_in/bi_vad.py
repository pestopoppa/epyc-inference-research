#!/usr/bin/env python3
"""CS-48 access to the ONE pinned Silero VAD from CS-45 (``tt_silero``): hash-checked, refuses on mismatch.

``TT_PINS_DIR`` names the directory holding the CS-45 pinned modules (``tt_silero.py``, ``tt_asr_words.py``); default is
``scripts/voice/turn_taking`` of this repo. ``BI_SILERO_MODEL`` overrides the model file path (used by the wrong-hash
mutation case; the expected sha256 stays the CS-45 pin, so an override with different bytes is refused).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

import bi_audio as A

HERE = Path(__file__).resolve().parent


def pins_dir() -> Path:
    env = os.environ.get("TT_PINS_DIR")
    return Path(env) if env else HERE.parent / "turn_taking"


def tt(modname: str):
    d = str(pins_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    return __import__(modname)


@lru_cache(maxsize=4)
def get_vad(model_path: str | None = None):
    m = tt("tt_silero")
    return m.PinnedSilero(model_path or os.environ.get("BI_SILERO_MODEL") or None)


def speech_segments_s(x_int16: np.ndarray, *, vad=None) -> list[tuple[float, float]]:
    """Pinned-Silero speech segments (FDB-style: seconds) with the CS-45 silero defaults."""
    m = tt("tt_silero")
    v = vad or get_vad()
    return [(t["start"] / A.SR, t["end"] / A.SR) for t in m.get_speech_timestamps(A.to_float(x_int16), v)]
