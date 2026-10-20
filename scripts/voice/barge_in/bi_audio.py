#!/usr/bin/env python3
"""CS-48 audio helpers: stdlib WAV I/O (mono 16-bit), exact-rational resampling, PCM hashing."""
from __future__ import annotations

import hashlib
import wave
from math import gcd
from pathlib import Path

import numpy as np

SR = 16_000


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """-> (int16 mono samples, sample rate). Refuses anything but mono 16-bit PCM."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: mono 16-bit PCM only (got {w.getnchannels()} ch, {w.getsampwidth() * 8} bit)")
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    return x, sr


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    x = np.asarray(x, dtype="<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())


def to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768.0


def resample_int16(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype="<i2")
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    y = resample_poly(x.astype(np.float64), sr_out // g, sr_in // g)
    return np.clip(np.rint(y), -32768, 32767).astype("<i2")


def pcm_sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype="<i2").tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
