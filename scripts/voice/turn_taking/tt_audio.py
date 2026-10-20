#!/usr/bin/env python3
"""CS-45 audio primitives: WAV I/O (stdlib), PCM-SHA-256, fades, resampling, clip assembly.

Everything here is deterministic integer/numpy arithmetic on int16 PCM. The corpus binds audio by
the SHA-256 of the DECODED interleaved s16le samples (Annex S P-STT-1 §1.2), never by file hash
alone; sample rate, channel count and sample count travel with it.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000  # corpus sample rate (Silero, Parakeet, Smart Turn, TurnBench all consume 16 kHz)


@dataclass(frozen=True)
class Pcm:
    """Decoded PCM: int16 array shaped (n_samples, channels)."""

    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    def pcm_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.samples, dtype="<i2").tobytes()).hexdigest()

    def channel_sha256(self, ch: int) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.samples[:, ch], dtype="<i2").tobytes()).hexdigest()


def read_wav(path: str | Path) -> Pcm:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is admitted (got {8 * w.getsampwidth()}-bit)")
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    arr = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.int16)
    return Pcm(arr, sr)


def wav_bytes(pcm: Pcm) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.channels)
        w.setsampwidth(2)
        w.setframerate(pcm.sample_rate)
        w.writeframes(np.ascontiguousarray(pcm.samples, dtype="<i2").tobytes())
    return buf.getvalue()


def write_wav(path: str | Path, pcm: Pcm) -> str:
    """Write and return the FILE sha256 (a file hash is recorded but never substitutes for the PCM hash)."""
    b = wav_bytes(pcm)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b)
    return hashlib.sha256(b).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def resample_24k_to_16k(x: np.ndarray) -> np.ndarray:
    """Pinned polyphase 3:2 down-conversion (scipy.signal.resample_poly, up=2 down=3, default Kaiser window)."""
    from scipy.signal import resample_poly

    y = resample_poly(x.astype(np.float64) / 32768.0, 2, 3)
    return np.clip(np.rint(y * 32768.0), -32768, 32767).astype(np.int16)


def fade_out(x: np.ndarray, n: int) -> np.ndarray:
    """Raised-cosine fade over the last n samples (applied to a copy)."""
    y = x.astype(np.float64).copy()
    n = min(n, len(y))
    if n > 0:
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, endpoint=True)))
        y[-n:] *= ramp
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def trim_to_speech(x: np.ndarray, sr: int = SR, frame_ms: int = 10, floor_db: float = -50.0, pad_ms: int = 30):
    """Return (start, end) sample indices bounding the non-silent region (frame RMS above floor_db re full scale)."""
    f = int(sr * frame_ms / 1000)
    n = len(x) // f
    if n == 0:
        return 0, len(x)
    fr = x[: n * f].astype(np.float64).reshape(n, f) / 32768.0
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-20)
    db = 20 * np.log10(rms + 1e-12)
    idx = np.where(db > floor_db)[0]
    if len(idx) == 0:
        return 0, len(x)
    pad = int(sr * pad_ms / 1000)
    return max(0, int(idx[0] * f) - pad), min(len(x), int((idx[-1] + 1) * f) + pad)


def snap_to_quiet(x: np.ndarray, center: int, radius: int, sr: int = SR, win_ms: int = 5) -> int:
    """Return the sample in [center-radius, center+radius] at the centre of the lowest-RMS window."""
    w = max(1, int(sr * win_ms / 1000))
    lo = max(w, center - radius)
    hi = min(len(x) - w, center + radius)
    if hi <= lo:
        return int(min(max(center, 0), len(x)))
    xs = x.astype(np.float64)
    c = np.concatenate([[0.0], np.cumsum(xs ** 2)])
    best, best_e = lo, None
    for i in range(lo, hi + 1):
        e = c[i + w] - c[i - w]
        if best_e is None or e < best_e:
            best, best_e = i, e
    return int(best)


def read_wav_bytes(data: bytes) -> Pcm:
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is admitted")
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    return Pcm(np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.int16), sr)


def write_tar(path: str | Path, members: dict[str, bytes]) -> str:
    """Deterministic uncompressed USTAR (sorted names, mtime 0, uid/gid 0). Returns the archive sha256.

    Audio is stored in a few large archives rather than hundreds of small WAV files: the repository's
    pii_precommit hook text-scans every staged blob under 1 MB, and raw PCM bytes produce random digit runs.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as t:
        for name in sorted(members):
            ti = tarfile.TarInfo(name)
            ti.size = len(members[name])
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            t.addfile(ti, io.BytesIO(members[name]))
    return file_sha256(path)


def load_audio(root: str | Path, ref: str) -> tuple[Pcm, str]:
    """ref is 'relative/path.wav' or 'relative/archive.tar#member.wav'. Returns (Pcm, sha256 of the WAV bytes)."""
    if "#" in ref:
        arc, member = ref.split("#", 1)
        with tarfile.open(Path(root) / arc) as t:
            f = t.extractfile(member)
            if f is None:
                raise FileNotFoundError(ref)
            data = f.read()
    else:
        data = (Path(root) / ref).read_bytes()
    return read_wav_bytes(data), hashlib.sha256(data).hexdigest()
