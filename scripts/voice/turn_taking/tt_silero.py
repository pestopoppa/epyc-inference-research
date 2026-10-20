#!/usr/bin/env python3
"""CS-45 single pinned Silero VAD (ONNX, CPU, no torch, no torch.hub, no network).

The pinned file is the faster-whisper 1.2.1 asset ``silero_vad_v6.onnx`` (sha256 recorded in
``data/turn-taking-cs45-v1/pins.json`` and copied into the repo so the pin is durable). Every
scorer in the CS-22/CS-23 programme obtains Silero through THIS module; loading a file whose
SHA-256 differs from the pin raises ``PinMismatch`` (refusal, not a warning).

Model interface (this export, not the PyPI ``silero_vad.onnx``): input ``[seq, 576]`` float32
(64 samples of left context + a 512-sample window at 16 kHz), ``h``/``c`` ``[1,1,128]``;
outputs ``speech_probs [seq]``, ``hn``, ``cn``. The LSTM state is carried across rows.

``get_speech_timestamps`` is a port of ``faster_whisper.vad.get_speech_timestamps`` (1.2.1) with
the SILERO scorer defaults made explicit (threshold 0.5, min_speech 250 ms, min_silence 100 ms,
speech_pad 30 ms, neg_threshold = threshold - 0.15). The max-speech splitting branch is omitted
because every scorer uses the default ``max_speech_duration_s = inf``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent.parent.parent / "data" / "turn-taking-cs45-v1" / "pins" / "silero_vad_v6.onnx"
PINNED_SHA256 = "4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2"
WINDOW = 512
CONTEXT = 64
SR = 16000


class PinMismatch(RuntimeError):
    """A pinned instrument file does not hash to its recorded SHA-256."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class PinnedSilero:
    def __init__(self, path: str | Path | None = None, expected_sha256: str = PINNED_SHA256):
        self.path = Path(path) if path else DEFAULT_MODEL
        got = sha256_file(self.path)
        if got != expected_sha256:
            raise PinMismatch(f"Silero VAD {self.path}: sha256 {got} != pinned {expected_sha256}")
        self.sha256 = got
        import onnxruntime as ort  # imported after the pin check so a bad file never reaches ORT

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"], sess_options=opts)

    def speech_probs(self, audio: np.ndarray) -> np.ndarray:
        """Offline: 1-D float32 in [-1, 1] at 16 kHz -> one probability per 32 ms window."""
        assert audio.ndim == 1
        a = audio.astype(np.float32)
        a = np.pad(a, (0, WINDOW - a.shape[0] % WINDOW))
        batched = a.reshape(-1, WINDOW)
        ctx = batched[..., -CONTEXT:].copy()
        ctx[-1] = 0
        ctx = np.roll(ctx, 1, 0)
        x = np.concatenate([ctx, batched], 1).astype(np.float32)
        h = np.zeros((1, 1, 128), dtype=np.float32)
        c = np.zeros((1, 1, 128), dtype=np.float32)
        out, _, _ = self.session.run(None, {"input": x, "h": h, "c": c})
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def stream(self) -> "SileroStream":
        return SileroStream(self)


class SileroStream:
    """Streaming: one 512-sample window per call; LSTM state and left context carried here."""

    def __init__(self, model: PinnedSilero):
        self.m = model
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((CONTEXT,), dtype=np.float32)

    def feed(self, window: np.ndarray) -> float:
        assert window.shape == (WINDOW,)
        x = np.concatenate([self.ctx, window.astype(np.float32)])[None, :]
        out, self.h, self.c = self.m.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = window[-CONTEXT:].astype(np.float32).copy()
        return float(np.asarray(out).reshape(-1)[0])


def get_speech_timestamps(
    audio: np.ndarray,
    model: PinnedSilero,
    *,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    sampling_rate: int = SR,
    return_seconds: bool = False,
):
    """Offline segmentation. Returns [{'start','end'}] in samples (or seconds)."""
    if sampling_rate != SR:
        raise ValueError("pinned Silero scorer path is 16 kHz only")
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    n = len(audio)
    probs = model.speech_probs(audio)
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    triggered = False
    speeches: list[dict] = []
    cur: dict = {}
    temp_end = 0
    for i, p in enumerate(probs):
        if p >= threshold and temp_end:
            temp_end = 0
        if p >= threshold and not triggered:
            triggered = True
            cur["start"] = WINDOW * i
            continue
        if p < neg_threshold and triggered:
            if not temp_end:
                temp_end = WINDOW * i
            if WINDOW * i - temp_end < min_silence_samples:
                continue
            cur["end"] = temp_end
            if cur["end"] - cur["start"] > min_speech_samples:
                speeches.append(cur)
            cur = {}
            temp_end = 0
            triggered = False
            continue
    if cur and (n - cur["start"]) > min_speech_samples:
        cur["end"] = n
        speeches.append(cur)
    for i, s in enumerate(speeches):
        if i == 0:
            s["start"] = int(max(0, s["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            gap = speeches[i + 1]["start"] - s["end"]
            if gap < 2 * speech_pad_samples:
                s["end"] += int(gap // 2)
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - gap // 2))
            else:
                s["end"] = int(min(n, s["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
        else:
            s["end"] = int(min(n, s["end"] + speech_pad_samples))
    if return_seconds:
        return [{"start": round(s["start"] / sampling_rate, 3), "end": round(s["end"] / sampling_rate, 3)} for s in speeches]
    return speeches
