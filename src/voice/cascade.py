"""Injected whisper → voice-turn → qwentts cascade backend contract."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from src.voice.contracts import VoiceEvent, VoiceTurn


MAX_ROUTE_EVENTS = 4096
MAX_ANSWER_CHARACTERS = 262_144


class Transcriber(Protocol):
    def transcribe(self, audio: bytes, *, language: str | None = None) -> str: ...
    def health(self) -> bool: ...


class VoiceTurnService(Protocol):
    def stream_turn(
        self, transcript: str, *, session_id: str, turn_id: str,
        response_mode: str, must_preserve: tuple[str, ...],
    ) -> Iterable[VoiceEvent]: ...
    def inject(self, text: str) -> None: ...
    def cancel_generation(self) -> None: ...
    def cancel_orchestrator(self) -> None: ...
    def health(self) -> bool: ...


class SpeechSynthesizer(Protocol):
    def stream_pcm(self, text: str, *, language: str | None = None) -> Iterable[bytes]: ...
    def cancel(self) -> None: ...
    def health(self) -> bool: ...


class CascadeBackend:
    """Compose existing service boundaries; concrete clients are caller supplied."""

    def __init__(
        self,
        transcriber: Transcriber,
        voice_turn: VoiceTurnService,
        synthesizer: SpeechSynthesizer,
    ) -> None:
        self._transcriber = transcriber
        self._voice_turn = voice_turn
        self._synthesizer = synthesizer

    def ingest(self, audio_chunk: bytes) -> str:
        if not isinstance(audio_chunk, bytes):
            raise TypeError("audio_chunk must be bytes")
        return self._transcriber.transcribe(audio_chunk)

    def respond(self, turn: VoiceTurn) -> Iterator[VoiceEvent]:
        # An explicitly empty transcript is valid and must not trigger a second
        # transcription. Only None means that the route must transcribe audio.
        transcript = (
            turn.transcript
            if turn.transcript is not None
            else self._transcriber.transcribe(turn.audio_chunk, language=turn.language)
        )
        answer_parts: list[str] = []
        answer_size = 0
        terminal_seen = False
        route_failed = False
        for index, event in enumerate(self._voice_turn.stream_turn(
            transcript, session_id=turn.session_id, turn_id=turn.turn_id,
            response_mode=turn.response_mode, must_preserve=turn.must_preserve,
        )):
            if index >= MAX_ROUTE_EVENTS:
                yield VoiceEvent("error", "voice-turn stream exceeded event limit")
                return
            if (
                not isinstance(event, VoiceEvent)
                or not isinstance(event.kind, str)
                or event.kind not in {
                    "text_delta", "display", "preserve", "tool_call", "end", "error"
                }
            ):
                yield VoiceEvent("error", "voice-turn stream emitted malformed event")
                return
            if event.kind == "text_delta" and isinstance(event.payload, str):
                answer_size += len(event.payload)
                if answer_size > MAX_ANSWER_CHARACTERS:
                    yield VoiceEvent("error", "voice-turn answer exceeded size limit")
                    return
                answer_parts.append(event.payload)
                yield event
            elif event.kind == "text_delta":
                yield VoiceEvent("error", "voice-turn emitted malformed text event")
                return
            elif event.kind in {"display", "preserve", "tool_call"}:
                if event.kind == "display" and not isinstance(event.payload, dict):
                    yield VoiceEvent("error", "voice-turn emitted malformed display event")
                    return
                if event.kind == "preserve" and not (
                    isinstance(event.payload, (tuple, list))
                    and all(isinstance(item, str) for item in event.payload)
                ):
                    yield VoiceEvent("error", "voice-turn emitted malformed preserve event")
                    return
                if event.kind == "tool_call" and not isinstance(event.payload, dict):
                    yield VoiceEvent("error", "voice-turn emitted malformed tool event")
                    return
                yield event
            elif event.kind == "error":
                if event.payload is not None and not isinstance(event.payload, str):
                    yield VoiceEvent("error", "voice-turn emitted malformed error event")
                    return
                yield event
                terminal_seen = True
                route_failed = True
                break
            elif event.kind == "end":
                if event.payload is not None:
                    yield VoiceEvent("error", "voice-turn emitted malformed end event")
                    return
                terminal_seen = True
                break
        if not terminal_seen:
            # Exhausting the route stream without a terminal marker is an error;
            # never synthesize a potentially incomplete answer.
            yield VoiceEvent("error", "voice-turn stream ended without terminal event")
            return
        if route_failed:
            return
        answer = "".join(answer_parts)
        if not answer.strip():
            yield VoiceEvent("error", "voice-turn stream contained no speakable text")
            return
        if any(value not in answer for value in turn.must_preserve):
            yield VoiceEvent("error", "voice-turn stream omitted a protected value")
            return
        for index, chunk in enumerate(self._synthesizer.stream_pcm(answer, language=turn.language)):
            if index >= MAX_ROUTE_EVENTS:
                yield VoiceEvent("error", "speech backend exceeded chunk limit")
                return
            if not isinstance(chunk, bytes):
                yield VoiceEvent("error", "speech backend emitted non-byte PCM")
                return
            yield VoiceEvent("audio_chunk", chunk)
        yield VoiceEvent("end")

    def inject(self, text: str) -> None:
        self._voice_turn.inject(text)

    def cancel_generation(self) -> None:
        self._voice_turn.cancel_generation()

    def cancel_vocoder(self) -> None:
        self._synthesizer.cancel()

    def cancel_orchestrator(self) -> None:
        self._voice_turn.cancel_orchestrator()

    def health(self) -> bool:
        return all((self._transcriber.health(), self._voice_turn.health(),
                    self._synthesizer.health()))
