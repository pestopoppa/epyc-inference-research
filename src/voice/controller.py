"""Transport-neutral voice controller with fail-closed cascade fallback."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import threading

from src.voice.contracts import (
    AudioQueue,
    InterlocutorBackend,
    RetainCancelChoice,
    VoiceBackend,
    VoiceEvent,
    VoiceTurn,
)


MAX_CONTROLLER_EVENTS = 4096
_EVENT_KINDS = frozenset(
    {"text_delta", "display", "preserve", "audio_chunk", "tool_call", "end", "error"}
)


@dataclass
class _ActiveTurn:
    session_id: str
    turn_id: str
    backend_name: str
    backend: InterlocutorBackend | VoiceBackend
    cancelled: bool = False
    stream_done: bool = False
    cleanup_started: bool = False
    cleanup_done: bool = True
    operations_in_flight: int = 0


class VoiceController:
    """Route one active turn to a healthy, language-capable backend.

    Concrete transports and engines are injected. A controller admits one
    active turn at a time; every mutation or cancellation names both session
    and turn so one client cannot take over another client's generation.
    """

    def __init__(
        self,
        cascade: VoiceBackend,
        audio_queue: AudioQueue,
        interlocutor: InterlocutorBackend | None = None,
        *,
        interlocutor_languages: frozenset[str] = frozenset(),
    ) -> None:
        self._cascade = cascade
        self._audio_queue = audio_queue
        self._interlocutor = interlocutor
        self._interlocutor_languages = frozenset(
            language.casefold() for language in interlocutor_languages
        )
        self._state_lock = threading.Lock()
        self._active: _ActiveTurn | None = None

    def ingest(self, audio_chunk: bytes) -> str:
        if not isinstance(audio_chunk, bytes):
            raise TypeError("audio_chunk must be bytes")
        return self._cascade.ingest(audio_chunk)

    def health(self) -> dict[str, bool]:
        return {
            "cascade": bool(self._cascade.health()),
            "interlocutor": bool(self._interlocutor and self._interlocutor.health()),
        }

    def respond(self, turn: VoiceTurn) -> Iterator[VoiceEvent]:
        self._validate_turn(turn)
        use_interlocutor = self._can_use_interlocutor(turn)
        if not use_interlocutor and not self._cascade.health():
            yield VoiceEvent("error", "cascade backend unavailable")
            return
        backend: InterlocutorBackend | VoiceBackend = (
            self._interlocutor if use_interlocutor else self._cascade
        )
        assert backend is not None
        owner = _ActiveTurn(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            backend_name="interlocutor" if use_interlocutor else "cascade",
            backend=backend,
        )
        with self._state_lock:
            busy = self._active is not None
            if not busy:
                self._active = owner
        if busy:
            yield VoiceEvent("error", "another voice turn is already active")
            return
        terminal_seen = False
        try:
            for index, event in enumerate(backend.respond(turn)):
                with self._state_lock:
                    if owner.cancelled:
                        return
                if index >= MAX_CONTROLLER_EVENTS:
                    yield VoiceEvent("error", "voice backend exceeded event limit")
                    return
                if not self._valid_event(event):
                    yield VoiceEvent("error", "voice backend emitted malformed event")
                    return
                yield event
                if event.kind in {"end", "error"}:
                    terminal_seen = True
                    return
            if not terminal_seen:
                with self._state_lock:
                    if owner.cancelled:
                        return
                yield VoiceEvent("error", "voice backend ended without terminal event")
        finally:
            with self._state_lock:
                owner.stream_done = True
                self._retire_if_finished(owner)

    def inject(self, session_id: str, turn_id: str, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        with self._state_lock:
            owner = self._matching_active(session_id, turn_id)
            if owner is None or owner.cancelled:
                raise ValueError("no active matching voice turn")
            owner.operations_in_flight += 1
        try:
            # Backend code is caller supplied and may re-enter cancel/respond.
            # The admission decision is made while locked; the hook runs outside
            # the controller lock, and remains part of this owner's lifecycle.
            owner.backend.inject(text)
        finally:
            with self._state_lock:
                owner.operations_in_flight -= 1
                self._retire_if_finished(owner)

    def cancel(self, session_id: str, turn_id: str) -> None:
        """Cancel only the named turn and attempt every ordered cleanup hook."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("turn_id must be a non-empty string")
        run_cleanup = False
        with self._state_lock:
            owner = self._matching_active(session_id, turn_id)
            if owner is not None and not owner.cleanup_started:
                owner.cancelled = True
                owner.cleanup_started = True
                owner.cleanup_done = False
                run_cleanup = True
        errors: list[Exception] = []
        operations = [("queued audio", lambda: self._audio_queue.stop_queued_audio(
            session_id, turn_id
        ))]
        if owner is not None and run_cleanup:
            operations.extend((
                ("generation", owner.backend.cancel_generation),
                ("vocoder", owner.backend.cancel_vocoder),
                ("orchestrator", owner.backend.cancel_orchestrator),
            ))
        for label, operation in operations:
            try:
                operation()
            except Exception as exc:
                errors.append(RuntimeError(f"{label} cancellation failed: {exc}"))
        if owner is not None and run_cleanup:
            with self._state_lock:
                owner.cleanup_done = True
                self._retire_if_finished(owner)
        if errors:
            raise RuntimeError("voice cancellation incomplete: " + "; ".join(map(str, errors))) from errors[0]

    def apply_retain_cancel_choice(
        self, session_id: str, turn_id: str, choice: RetainCancelChoice
    ) -> RetainCancelChoice:
        """Apply an explicit typed disposition; language/intent classification is external."""
        if not isinstance(choice, RetainCancelChoice):
            raise TypeError("choice must be a RetainCancelChoice")
        with self._state_lock:
            owner = self._matching_active(session_id, turn_id)
            if owner is None or owner.cancelled:
                raise ValueError("no active matching voice turn")
        if choice is RetainCancelChoice.CANCEL:
            self.cancel(session_id, turn_id)
        return choice

    def _retire_if_finished(self, owner: _ActiveTurn) -> None:
        if (self._active is owner and owner.stream_done and owner.cleanup_done
                and owner.operations_in_flight == 0):
            self._active = None

    @staticmethod
    def _validate_turn(turn: VoiceTurn) -> None:
        if not isinstance(turn, VoiceTurn):
            raise TypeError("turn must be a VoiceTurn")
        if not isinstance(turn.session_id, str) or not turn.session_id.strip():
            raise ValueError("turn.session_id must be a non-empty string")
        if not isinstance(turn.turn_id, str) or not turn.turn_id.strip():
            raise ValueError("turn.turn_id must be a non-empty string")
        if not isinstance(turn.audio_chunk, bytes):
            raise TypeError("turn.audio_chunk must be bytes")
        if turn.language is not None and not isinstance(turn.language, str):
            raise TypeError("turn.language must be str or None")
        if turn.transcript is not None and not isinstance(turn.transcript, str):
            raise TypeError("turn.transcript must be str or None")
        if (not isinstance(turn.conversation_context, tuple)
                or any(not isinstance(item, dict) for item in turn.conversation_context)):
            raise TypeError("turn.conversation_context must be a tuple of dictionaries")
        if not isinstance(turn.metadata, dict):
            raise TypeError("turn.metadata must be a dictionary")
        if (not isinstance(turn.response_mode, str)
                or turn.response_mode not in {"normal", "verbatim"}):
            raise ValueError("turn.response_mode must be normal or verbatim")
        if (not isinstance(turn.must_preserve, tuple)
                or any(not isinstance(item, str) or not item.strip()
                       for item in turn.must_preserve)):
            raise TypeError("turn.must_preserve must be a tuple of non-empty strings")

    @staticmethod
    def _valid_event(event: VoiceEvent) -> bool:
        if (not isinstance(event, VoiceEvent) or not isinstance(event.kind, str)
                or event.kind not in _EVENT_KINDS):
            return False
        if event.kind in {"text_delta"}:
            return isinstance(event.payload, str)
        if event.kind in {"display", "tool_call"}:
            return isinstance(event.payload, dict)
        if event.kind == "preserve":
            return (isinstance(event.payload, (tuple, list))
                    and all(isinstance(item, str) for item in event.payload))
        if event.kind == "audio_chunk":
            return isinstance(event.payload, bytes)
        if event.kind == "end":
            return event.payload is None
        if event.kind == "error":
            return event.payload is None or isinstance(event.payload, str)
        return False

    def _matching_active(self, session_id: str, turn_id: str) -> _ActiveTurn | None:
        owner = self._active
        if owner is None or owner.session_id != session_id or owner.turn_id != turn_id:
            return None
        return owner

    def _can_use_interlocutor(self, turn: VoiceTurn) -> bool:
        backend = self._interlocutor
        if backend is None or not backend.health():
            return False
        if turn.language is None:
            return False
        return turn.language.casefold() in self._interlocutor_languages
