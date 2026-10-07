"""Synthetic controls for transport-neutral voice routing and cascade order."""

from src.voice.cascade import CascadeBackend
from src.voice.contracts import VoiceEvent, VoiceTurn
from src.voice.controller import VoiceController


class _AudioQueue:
    def __init__(self, calls):
        self.calls = calls

    def stop_queued_audio(self, session_id, turn_id):
        self.calls.append(("stop_queued_audio", session_id, turn_id))


class _Interlocutor:
    def __init__(self, *, healthy=True):
        self.is_healthy = healthy
        self.calls = []

    def respond(self, turn):
        self.calls.append(("respond", turn.turn_id))
        yield VoiceEvent("text_delta", "interlocutor")
        yield VoiceEvent("end")

    def inject(self, text):
        self.calls.append(("inject", text))

    def cancel_generation(self):
        self.calls.append(("cancel_generation",))

    def cancel_vocoder(self):
        self.calls.append(("cancel_vocoder",))

    def cancel_orchestrator(self):
        self.calls.append(("cancel_orchestrator",))

    def health(self):
        return self.is_healthy


class _Transcriber:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, *, language=None):
        self.calls.append((audio, language))
        return "synthetic transcript"

    def health(self):
        return True


class _VoiceTurn:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def stream_turn(self, transcript, *, session_id, turn_id):
        self.calls.append((transcript, session_id, turn_id))
        yield from self.events

    def inject(self, text):
        self.calls.append(("inject", text))

    def cancel_generation(self):
        self.calls.append(("cancel_generation",))

    def cancel_orchestrator(self):
        self.calls.append(("cancel_orchestrator",))

    def health(self):
        return True


class _Synthesizer:
    def __init__(self):
        self.calls = []

    def stream_pcm(self, text, *, language=None):
        self.calls.append((text, language))
        yield b"pcm-1"
        yield b"pcm-2"

    def cancel(self):
        self.calls.append(("cancel",))

    def health(self):
        return True


def test_interlocutor_falls_back_for_unsupported_language_and_unhealthy_backend():
    transcriber = _Transcriber()
    voice_turn = _VoiceTurn([VoiceEvent("text_delta", "cascade"), VoiceEvent("end")])
    synth = _Synthesizer()
    cascade = CascadeBackend(transcriber, voice_turn, synth)
    interlocutor = _Interlocutor()
    controller = VoiceController(
        cascade, _AudioQueue([]), interlocutor,
        interlocutor_languages=frozenset({"en"}),
    )

    unsupported = VoiceTurn("t1", "s1", b"wav", language="fr")
    events = list(controller.respond(unsupported))

    assert [event.kind for event in events] == ["text_delta", "audio_chunk", "audio_chunk", "end"]
    assert transcriber.calls == [(b"wav", "fr")]
    assert voice_turn.calls[0] == ("synthetic transcript", "s1", "t1")
    assert interlocutor.calls == []

    interlocutor.is_healthy = False
    assert [event.payload for event in controller.respond(
        VoiceTurn("t2", "s1", b"wav2", language="en")
    ) if event.kind == "text_delta"] == ["cascade"]
    assert len(interlocutor.calls) == 0


def test_healthy_supported_interlocutor_is_used_and_cancel_order_is_explicit():
    transcriber = _Transcriber()
    cascade = CascadeBackend(transcriber, _VoiceTurn([]), _Synthesizer())
    cancellation_order = []
    interlocutor = _Interlocutor()
    interlocutor.calls = cancellation_order
    controller = VoiceController(
        cascade, _AudioQueue(cancellation_order), interlocutor,
        interlocutor_languages=frozenset({"en"}),
    )
    stream = controller.respond(VoiceTurn("t1", "s1", b"wav", language="EN"))

    assert next(stream) == VoiceEvent("text_delta", "interlocutor")
    controller.cancel("s1", "t1")

    assert cancellation_order[-4:] == [
        ("stop_queued_audio", "s1", "t1"), ("cancel_generation",),
        ("cancel_vocoder",), ("cancel_orchestrator",)
    ]
    # The fake backend ignores its cancellation signal and yields its terminal
    # event anyway; the controller suppresses every post-cancel event.
    assert list(stream) == []
    assert transcriber.calls == []


def test_controller_rejects_a_second_active_turn_and_idle_cancel_only_stops_queue():
    class OpenBackend(_Interlocutor):
        def __init__(self):
            super().__init__()
            self.release = False

        def respond(self, turn):
            yield VoiceEvent("text_delta", turn.turn_id)
            while not self.release:
                yield VoiceEvent("text_delta", "backend ignored cancellation")
            yield VoiceEvent("end")

    backend = OpenBackend()
    queue_calls = []
    controller = VoiceController(
        CascadeBackend(_Transcriber(), _VoiceTurn([]), _Synthesizer()),
        _AudioQueue(queue_calls), backend,
        interlocutor_languages=frozenset({"en"}),
    )
    first = controller.respond(VoiceTurn("t1", "s1", b"a", language="en"))
    assert next(first) == VoiceEvent("text_delta", "t1")
    second = list(controller.respond(VoiceTurn("t2", "s2", b"b", language="en")))
    assert second == [VoiceEvent("error", "another voice turn is already active")]

    controller.cancel("other-session", "other-turn")
    assert queue_calls == [("stop_queued_audio", "other-session", "other-turn")]
    backend.release = True
    assert list(first) == [VoiceEvent("text_delta", "backend ignored cancellation"), VoiceEvent("end")]


def test_cancel_marks_active_turn_before_cleanup_and_attempts_all_hooks_on_errors():
    class FailingBackend(_Interlocutor):
        def respond(self, turn):
            yield VoiceEvent("text_delta", "first")
            yield VoiceEvent("text_delta", "must be suppressed")

        def cancel_generation(self):
            self.calls.append(("cancel_generation",))
            raise RuntimeError("generation hook failed")

    calls = []
    backend = FailingBackend()
    backend.calls = calls
    controller = VoiceController(
        CascadeBackend(_Transcriber(), _VoiceTurn([]), _Synthesizer()),
        _AudioQueue(calls), backend,
        interlocutor_languages=frozenset({"en"}),
    )
    stream = controller.respond(VoiceTurn("t1", "s1", b"a", language="en"))
    assert next(stream) == VoiceEvent("text_delta", "first")
    try:
        controller.cancel("s1", "t1")
    except RuntimeError as exc:
        assert "generation cancellation failed" in str(exc)
    else:
        raise AssertionError("cancel should report failed cleanup")
    assert calls == [
        ("stop_queued_audio", "s1", "t1"), ("cancel_generation",),
        ("cancel_vocoder",), ("cancel_orchestrator",),
    ]
    assert list(stream) == []


def test_inject_hook_can_reenter_cancel_without_holding_controller_lock():
    class ReentrantBackend(_Interlocutor):
        controller = None

        def inject(self, text):
            self.calls.append(("inject", text))
            self.controller.cancel("s1", "t1")

    calls = []
    backend = ReentrantBackend()
    backend.calls = calls
    controller = VoiceController(
        CascadeBackend(_Transcriber(), _VoiceTurn([]), _Synthesizer()),
        _AudioQueue(calls), backend,
        interlocutor_languages=frozenset({"en"}),
    )
    backend.controller = controller
    stream = controller.respond(VoiceTurn("t1", "s1", b"a", language="en"))
    assert next(stream) == VoiceEvent("text_delta", "interlocutor")

    controller.inject("s1", "t1", "synthetic interrupt")

    assert calls == [
        ("inject", "synthetic interrupt"),
        ("stop_queued_audio", "s1", "t1"),
        ("cancel_generation",), ("cancel_vocoder",), ("cancel_orchestrator",),
    ]
    assert list(stream) == []


def test_owner_remains_reserved_until_stream_and_cancel_hooks_finish():
    class ReentrantCancelBackend(_Interlocutor):
        controller = None
        replacement_result = None

        def cancel_generation(self):
            self.calls.append(("cancel_generation",))
            self.replacement_result = list(self.controller.respond(
                VoiceTurn("t2", "s2", b"b", language="en")
            ))

    calls = []
    backend = ReentrantCancelBackend()
    backend.calls = calls
    controller = VoiceController(
        CascadeBackend(_Transcriber(), _VoiceTurn([]), _Synthesizer()),
        _AudioQueue(calls), backend,
        interlocutor_languages=frozenset({"en"}),
    )
    backend.controller = controller
    stream = controller.respond(VoiceTurn("t1", "s1", b"a", language="en"))
    assert next(stream) == VoiceEvent("text_delta", "interlocutor")

    controller.cancel("s1", "t1")

    assert backend.replacement_result == [
        VoiceEvent("error", "another voice turn is already active")
    ]
    assert list(stream) == []
    replacement = controller.respond(VoiceTurn("t2", "s2", b"b", language="en"))
    assert next(replacement) == VoiceEvent("text_delta", "interlocutor")
    replacement.close()


def test_controller_rejects_malformed_turns_events_and_missing_terminal():
    class EventsBackend(_Interlocutor):
        def __init__(self, events):
            super().__init__()
            self.events = events

        def respond(self, turn):
            yield from self.events

    def controller_for(events):
        backend = EventsBackend(events)
        return VoiceController(
            CascadeBackend(_Transcriber(), _VoiceTurn([]), _Synthesizer()),
            _AudioQueue([]), backend, interlocutor_languages=frozenset({"en"}),
        )

    malformed_turns = (
        VoiceTurn(1, "s1", b"a", language="en"),
        VoiceTurn("t1", "s1", "not bytes", language="en"),
        VoiceTurn("t1", "s1", b"a", language=4),
        VoiceTurn("t1", "s1", b"a", conversation_context=[]),
    )
    for turn in malformed_turns:
        try:
            list(controller_for([VoiceEvent("end")]).respond(turn))
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("malformed VoiceTurn field should be rejected")

    malformed = list(controller_for([VoiceEvent("display", "not a mapping")]).respond(
        VoiceTurn("t1", "s1", b"a", language="en")
    ))
    assert malformed == [VoiceEvent("error", "voice backend emitted malformed event")]
    unterminated = list(controller_for([VoiceEvent("text_delta", "partial")]).respond(
        VoiceTurn("t1", "s1", b"a", language="en")
    ))
    assert unterminated == [
        VoiceEvent("text_delta", "partial"),
        VoiceEvent("error", "voice backend ended without terminal event"),
    ]


def test_cascade_synthesizes_only_a_complete_text_turn_and_emits_pcm_order():
    transcriber = _Transcriber()
    voice_turn = _VoiceTurn([
        VoiceEvent("text_delta", "hello "),
        VoiceEvent("display", {"url": "https://example.invalid"}),
        VoiceEvent("text_delta", "there"),
        VoiceEvent("end"),
    ])
    synth = _Synthesizer()
    cascade = CascadeBackend(transcriber, voice_turn, synth)

    events = list(cascade.respond(VoiceTurn("t1", "s1", b"wav", language="en")))

    assert voice_turn.calls == [("synthetic transcript", "s1", "t1")]
    assert synth.calls == [("hello there", "en")]
    assert [event.kind for event in events] == [
        "text_delta", "display", "text_delta", "audio_chunk", "audio_chunk", "end"
    ]
    assert [event.payload for event in events if event.kind == "audio_chunk"] == [b"pcm-1", b"pcm-2"]


def test_incomplete_or_failed_voice_turn_never_reaches_synthesizer():
    for events, expected in (
        ([VoiceEvent("text_delta", "partial")], "without terminal"),
        ([VoiceEvent("text_delta", "partial"), VoiceEvent("error", "cancelled")], "cancelled"),
    ):
        synth = _Synthesizer()
        cascade = CascadeBackend(_Transcriber(), _VoiceTurn(events), synth)

        observed = list(cascade.respond(VoiceTurn("t1", "s1", b"wav")))

        assert synth.calls == []
        assert observed[-1].kind == "error"
        assert expected in observed[-1].payload


def test_empty_explicit_transcript_is_valid_and_malformed_route_data_fails_closed():
    transcriber = _Transcriber()
    empty_transcript_route = _VoiceTurn([VoiceEvent("text_delta", "hello"), VoiceEvent("end")])
    empty_transcript_synth = _Synthesizer()
    backend = CascadeBackend(transcriber, empty_transcript_route, empty_transcript_synth)
    list(backend.respond(VoiceTurn("t1", "s1", b"wav", transcript="")))
    assert empty_transcript_route.calls == [("", "s1", "t1")]
    assert transcriber.calls == []

    for malformed in (
        VoiceEvent("text_delta", {"not": "text"}),
        VoiceEvent("unexpected"),
        VoiceEvent([], "unhashable event kind"),
    ):
        synth = _Synthesizer()
        output = list(CascadeBackend(
            _Transcriber(), _VoiceTurn([malformed, VoiceEvent("end")]), synth
        ).respond(VoiceTurn("t2", "s1", b"wav", transcript="synthetic")))
        assert output[-1].kind == "error"
        assert synth.calls == []


def test_route_answer_accumulation_has_a_fixed_upper_bound():
    synth = _Synthesizer()
    route = _VoiceTurn([
        VoiceEvent("text_delta", "x" * (262_144 + 1)),
        VoiceEvent("end"),
    ])
    output = list(CascadeBackend(_Transcriber(), route, synth).respond(
        VoiceTurn("t1", "s1", b"wav", transcript="synthetic")
    ))
    assert output[-1] == VoiceEvent("error", "voice-turn answer exceeded size limit")
    assert synth.calls == []
