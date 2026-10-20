#!/usr/bin/env python3
"""CS-48 scripted fake agent: the REAL ``CascadeBackend`` composed from deterministic, clock-driven fakes.

The synthesizer streams a known PCM signal in fixed chunks at a fixed multiple of real time through the injected
clock, so the exact instant model audio starts and (after a cancel) stops is known to the sample. No model, no network,
no service. ``cancel_log`` records which fake hooks actually fired, independently of the harness's own timestamps.
"""
from __future__ import annotations

import numpy as np

import bi_audio as A
import bi_orch


class FakeParts:
    def __init__(self, clock, log, pcm: np.ndarray, model_sr: int, *, chunk_ms: int = 200, pace: float = 2.0,
                 first_chunk_delay_ms: int = 0) -> None:
        self.clock, self.log, self.pcm, self.model_sr = clock, log, pcm, model_sr
        self.chunk_ms, self.pace, self.first_delay_ms = chunk_ms, pace, first_chunk_delay_ms
        self.cancel_log: list[str] = []
        self.cancelled = False

    # Transcriber
    def transcribe(self, audio: bytes, *, language=None) -> str:
        return "fake transcript"

    # VoiceTurnService
    def stream_turn(self, transcript, *, session_id, turn_id, response_mode, must_preserve):
        _, _, _, VoiceEvent = bi_orch.load_voice()
        self.cancelled = False
        yield VoiceEvent("text_delta", "This is a long scripted answer.")
        yield VoiceEvent("end")

    def inject(self, text: str) -> None:
        pass

    def cancel_generation(self) -> None:
        self.cancel_log.append("generation")

    def cancel_orchestrator(self) -> None:
        self.cancel_log.append("orchestrator")

    # SpeechSynthesizer
    def stream_pcm(self, text, *, language=None):
        n = int(self.model_sr * self.chunk_ms / 1000)
        t0 = self.clock.now_ns() + int(self.first_delay_ms * 1e6)
        for i, k in enumerate(range(0, len(self.pcm), n)):
            self.clock.sleep_until_ns(t0 + int(i * self.chunk_ms * 1e6 / self.pace))
            if self.cancelled:
                return
            yield self.pcm[k : k + n].astype("<i2").tobytes()

    def cancel(self) -> None:
        self.cancelled = True
        self.cancel_log.append("vocoder")

    def health(self) -> bool:
        return True


def agent_pcm(wav_paths: list[str], model_sr: int, gap_s: float = 0.0) -> np.ndarray:
    parts = []
    for p in wav_paths:
        x, sr = A.read_wav(p)
        parts.append(A.resample_int16(x, sr, model_sr))
        if gap_s:
            parts.append(np.zeros(int(gap_s * model_sr), dtype="<i2"))
    return np.concatenate(parts)


def make_factory(pcm: np.ndarray, model_sr: int, holder: list | None = None, **kw):
    """-> backend_factory(clock, log) for Session; the FakeParts instance is appended to ``holder``."""

    def factory(clock, log):
        _, CascadeBackend, _, _ = bi_orch.load_voice()
        parts = FakeParts(clock, log, pcm, model_sr, **kw)
        if holder is not None:
            holder.append(parts)
        return CascadeBackend(parts, parts, parts)

    return factory
