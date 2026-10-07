"""Transport-neutral contracts for the voice-turn controller.

This module contains data shapes and protocols only. It has no audio-device,
model, network, or orchestrator imports, so controller behavior can be checked
with deterministic in-process fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Literal, Protocol


VoiceEventKind = Literal[
    "text_delta", "display", "preserve", "audio_chunk", "tool_call", "end", "error"
]


class RetainCancelChoice(str, Enum):
    """Explicit caller-selected disposition for an in-flight turn."""

    RETAIN = "retain"
    CANCEL = "cancel"


@dataclass(frozen=True)
class VoiceTurn:
    turn_id: str
    session_id: str
    audio_chunk: bytes
    language: str | None = None
    transcript: str | None = None
    conversation_context: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    response_mode: Literal["normal", "verbatim"] = "normal"
    must_preserve: tuple[str, ...] = ()


@dataclass(frozen=True)
class VoiceEvent:
    kind: VoiceEventKind
    payload: Any = None


class InterlocutorBackend(Protocol):
    def respond(self, turn: VoiceTurn) -> Iterable[VoiceEvent]: ...
    def inject(self, text: str) -> None: ...
    def cancel_generation(self) -> None: ...
    def cancel_vocoder(self) -> None: ...
    def cancel_orchestrator(self) -> None: ...
    def health(self) -> bool: ...


class VoiceBackend(Protocol):
    def ingest(self, audio_chunk: bytes) -> str: ...
    def respond(self, turn: VoiceTurn) -> Iterable[VoiceEvent]: ...
    def cancel_generation(self) -> None: ...
    def cancel_vocoder(self) -> None: ...
    def cancel_orchestrator(self) -> None: ...
    def health(self) -> bool: ...
    def inject(self, text: str) -> None: ...


class AudioQueue(Protocol):
    def stop_queued_audio(self, session_id: str, turn_id: str) -> None: ...
