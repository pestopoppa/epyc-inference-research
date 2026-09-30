"""RI-18 — review-gate score accessor (C2), verdict question cap (C3), telemetry (C1).

C2: ``review_gate_score`` must reproduce ``_should_review`` exactly — same decision, same
KNN key — over a fixture grid, because the review call sites now decide through it.
C3: ``build_review_verdict_prompt``'s defaults keep the production prompt byte-identical.
C1: every review call site writes one ``review_gate`` tap event per gate evaluation, and
an unavailable verdict is recorded as ``unavailable``, never as ``ok``.

Mocked only: no network, no model.
"""

from __future__ import annotations

import itertools
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import ChatRequest
from src.api.routes import chat_review
from src.api.routes.chat_review import (
    GATE_SCORED,
    GateScore,
    _architect_verdict_with_status,
    _should_review,
    evaluate_review_gate,
    record_review_gate,
    review_gate_event,
    review_gate_score,
    review_gate_skipped,
)
from src.api.routes.chat_utils import RoutingResult
from src.prompt_builders import build_review_verdict_prompt
from src.prompt_builders.review import (
    REVIEW_VERDICT_ANSWER_CAP,
    REVIEW_VERDICT_QUESTION_CAP,
    _REVIEW_VERDICT_FALLBACK,
)
from src.runtime import routing_stage_timing as rst

LONG_ANSWER = "The answer is B) Nitrogen, which is about 78 percent of the atmosphere."
assert len(LONG_ANSWER) >= 50


# ── fixtures ────────────────────────────────────────────────────────────────


def _row(action: str, q: float) -> SimpleNamespace:
    return SimpleNamespace(memory=SimpleNamespace(action=action), q_value=q)


class _Retriever:
    """Records every KNN key; returns a fixed result list (or raises)."""

    def __init__(self, results=None, exc: Exception | None = None) -> None:
        self.results = results
        self.exc = exc
        self.keys: list[dict] = []

    def retrieve_for_routing(self, task_ir):
        self.keys.append(task_ir)
        if self.exc is not None:
            raise self.exc
        return self.results


def _state(retriever: _Retriever | None) -> SimpleNamespace:
    router = SimpleNamespace(retriever=retriever) if retriever is not None else None
    return SimpleNamespace(hybrid_router=router)


RESULT_SETS = {
    "none": None,
    "empty": [],
    "other_role_only": [_row("coder_escalation", 0.1), _row("worker_general", 0.2)],
    "low": [_row("frontdoor", 0.1), _row("frontdoor", 0.3), _row("coder_escalation", 0.9)],
    "high": [_row("frontdoor", 0.95), _row("frontdoor", 0.99)],
    "at_threshold": [_row("frontdoor", 0.6)],
    "just_below": [_row("frontdoor", 0.5999999)],
    "mixed": [_row("frontdoor", 0.2), _row("frontdoor", 1.0), _row("frontdoor", 0.7)],
    "raises": RuntimeError("retriever exploded"),
}
ROLES = ["frontdoor", "architect_general", "architect_critic", "coder_escalation"]
ANSWERS = ["", "B", "x" * 49, "y" * 50, LONG_ANSWER, "z" * 900]
THRESHOLDS = [0.6, 0.35, 1.01, 0.0]


def _with_threshold(threshold: float):
    cfg = MagicMock()
    cfg.chat.review_low_q_threshold = threshold
    return patch("src.api.routes.chat_review._get_config", return_value=cfg)


# ── C2: the accessor is _should_review's decision, exactly ──────────────────


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_accessor_reproduces_should_review_over_the_grid(threshold: float) -> None:
    checked = 0
    with _with_threshold(threshold):
        for (name, spec), role, answer in itertools.product(
            list(RESULT_SETS.items()) + [("no_router", "NO_ROUTER")], ROLES, ANSWERS
        ):
            def build() -> SimpleNamespace:
                if spec == "NO_ROUTER":
                    return _state(None)
                if isinstance(spec, Exception):
                    return _state(_Retriever(exc=spec))
                return _state(_Retriever(results=spec))

            old_state, new_state = build(), build()
            old = _should_review(old_state, "task-1", role, answer)
            score = review_gate_score(new_state, role, answer)
            assert score.triggered is old, (name, role, len(answer), threshold, score)
            # the same KNN key reached the retriever, the same number of times
            if old_state.hybrid_router is not None:
                assert old_state.hybrid_router.retriever.keys == (
                    new_state.hybrid_router.retriever.keys
                ), (name, role, len(answer))
            checked += 1
    assert checked == (len(RESULT_SETS) + 1) * len(ROLES) * len(ANSWERS)


def test_accessor_exposes_inputs_and_skip_reasons() -> None:
    with _with_threshold(0.6):
        assert review_gate_score(_state(None), "frontdoor", LONG_ANSWER).skip_reason == "no_router"
        r = _Retriever(results=RESULT_SETS["low"])
        assert review_gate_score(_state(r), "architect_critic", LONG_ANSWER).skip_reason == (
            "architect_role"
        )
        assert review_gate_score(_state(r), "frontdoor", "x" * 49).skip_reason == "short"
        assert r.keys == []  # none of the structural skips touched the KNN
        assert (
            review_gate_score(_state(_Retriever(results=[])), "frontdoor", LONG_ANSWER).skip_reason
            == "no_results"
        )
        s = review_gate_score(
            _state(_Retriever(results=RESULT_SETS["other_role_only"])), "frontdoor", LONG_ANSWER
        )
        assert (s.skip_reason, s.n_results, s.n_role_rows, s.avg_q) == ("no_role_rows", 2, 0, None)
        s = review_gate_score(
            _state(_Retriever(exc=RuntimeError("x"))), "frontdoor", LONG_ANSWER
        )
        assert (s.skip_reason, s.avg_q, s.triggered) == ("error", None, False)
        s = review_gate_score(_state(_Retriever(results=RESULT_SETS["low"])), "frontdoor", LONG_ANSWER)
        assert s.skip_reason == GATE_SCORED
        assert (s.n_results, s.n_role_rows) == (3, 2)
        assert s.avg_q == pytest.approx(0.2)
        assert s.threshold == 0.6 and s.answer_chars == len(LONG_ANSWER)
        assert s.triggered is True


def test_gate_controls_fire_at_1_01_and_never_at_0() -> None:
    """RI-18 §2.5: gate@1.01 is True on every scored item, gate@0.0 False on all."""
    with _with_threshold(0.6):
        for spec in RESULT_SETS.values():
            state = (
                _state(_Retriever(exc=spec))
                if isinstance(spec, Exception)
                else _state(_Retriever(results=spec))
            )
            s = review_gate_score(state, "frontdoor", LONG_ANSWER)
            assert s.fires_at(0.0) is False
            assert s.fires_at(1.01) is (s.skip_reason == GATE_SCORED)


def test_key_text_replaces_only_the_knn_key() -> None:
    r = _Retriever(results=RESULT_SETS["low"])
    with _with_threshold(0.6):
        s = review_gate_score(_state(r), "frontdoor", LONG_ANSWER, key_text="What gas? " * 40)
        assert r.keys[0]["objective"].startswith("What gas?")
        assert s.answer_chars == len(LONG_ANSWER)
        # the short guard is on the answer, not the key
        r2 = _Retriever(results=RESULT_SETS["low"])
        assert review_gate_score(_state(r2), "frontdoor", "B", key_text="long " * 40).skip_reason == (
            "short"
        )
        assert r2.keys == []


def test_caller_skip_never_triggers() -> None:
    g = review_gate_skipped("force_role", LONG_ANSWER)
    assert g.skip_reason == "caller:force_role" and g.triggered is False
    assert g.answer_chars == len(LONG_ANSWER)
    assert review_gate_skipped("no_answer", None).answer_chars == 0


def test_evaluate_review_gate_times_into_ri16_stage() -> None:
    rst.begin("t-eval", rst.PATH_CHAT)
    try:
        with _with_threshold(0.6):
            g = evaluate_review_gate(
                _state(_Retriever(results=RESULT_SETS["high"])), "frontdoor", LONG_ANSWER
            )
        assert g.triggered is False and g.gate_ms is not None and g.gate_ms >= 0
        assert isinstance(rst.telemetry_for("t-eval")["stage_ms"]["review_gate"], float)
    finally:
        rst.clear()


# ── C3: the default verdict prompt is byte-identical ────────────────────────


def _pre_ri18_verdict_prompt(question, answer, context_digest="", worker_digests=None) -> str:
    """The builder as it was before the cap parameters (orch 78847544), verbatim logic."""
    from src.prompt_builders.review import resolve_prompt

    digest_section = ""
    if worker_digests:
        try:
            from src.services.toon_encoder import encode, is_available

            if is_available():
                digest_section = f"\nEvidence:\n{encode(worker_digests)}\n"
            else:
                digest_section = f"\nEvidence:\n{json.dumps(worker_digests)}\n"
        except Exception:
            digest_section = f"\nEvidence:\n{json.dumps(worker_digests)}\n"
    elif context_digest:
        digest_section = f"\nContext: {context_digest[:800]}\n"
    return resolve_prompt(
        "review_verdict",
        _REVIEW_VERDICT_FALLBACK,
        digest_section=digest_section,
        question=question[:300],
        answer=answer[:1500],
    )


@pytest.mark.parametrize(
    "question,answer,kwargs",
    [
        ("What is 2+2?", "4", {}),
        ("Q" * 299, "A" * 1499, {}),
        ("Q" * 300, "A" * 1500, {}),
        ("Which option? " * 80, "Because... " * 400, {}),
        ("q", "a", {"context_digest": "C" * 1200}),
        ("q", "a", {"worker_digests": [{"role": "coder", "output": "code"}]}),
    ],
)
def test_default_verdict_prompt_is_byte_identical(question, answer, kwargs) -> None:
    assert REVIEW_VERDICT_QUESTION_CAP == 300 and REVIEW_VERDICT_ANSWER_CAP == 1500
    new = build_review_verdict_prompt(question, answer, **kwargs)
    assert new.encode() == _pre_ri18_verdict_prompt(question, answer, **kwargs).encode()
    explicit = build_review_verdict_prompt(
        question, answer, question_cap=300, answer_cap=1500, **kwargs
    )
    assert explicit == new


def test_question_cap_widens_only_the_question() -> None:
    question = "Stem. " + " ".join(f"({c}) option {c}" for c in "ABCDEFGHIJ") * 4
    assert len(question) > 300
    capped = build_review_verdict_prompt(question, "Answer: B")
    full = build_review_verdict_prompt(question, "Answer: B", question_cap=1500)
    assert question[:1500] in full and question[:1500] not in capped
    assert question[:300] in capped
    with pytest.raises(ValueError):
        build_review_verdict_prompt("q", "a", question_cap=0)


def test_verdict_with_status_default_sends_the_production_prompt() -> None:
    primitives = MagicMock()
    primitives.llm_call.return_value = "OK"
    question = "Which option? " * 60
    _architect_verdict_with_status(question, LONG_ANSWER, primitives)
    sent = primitives.llm_call.call_args.args[0]
    assert sent == _pre_ri18_verdict_prompt(question, LONG_ANSWER)
    _architect_verdict_with_status(question, LONG_ANSWER, primitives, question_cap=1500)
    assert primitives.llm_call.call_args.args[0] == build_review_verdict_prompt(
        question, LONG_ANSWER, question_cap=1500
    )


# ── C1: the review_gate event ───────────────────────────────────────────────


@pytest.fixture
def tap_events(monkeypatch, tmp_path):
    """A real (tmp) tap: the event must reach disk, not just a mock."""
    tap = tmp_path / "tap.log"
    events = tmp_path / "tap_events.jsonl"
    monkeypatch.setenv("INFERENCE_TAP_FILE", str(tap))
    monkeypatch.setenv("INFERENCE_TAP_EVENTS_FILE", str(events))

    def read() -> list[dict]:
        if not events.exists():
            return []
        rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
        return [r for r in rows if r.get("event") == "review_gate"]

    return read


def _scored(triggered: bool) -> GateScore:
    return GateScore(
        avg_q=0.2 if triggered else 0.9,
        n_results=5,
        n_role_rows=2,
        skip_reason=GATE_SCORED,
        threshold=0.6,
        answer_chars=len(LONG_ANSWER),
        gate_ms=1.5,
    )


def test_record_writes_one_event_with_the_schema(tap_events) -> None:
    rst.clear()
    assert record_review_gate(
        "t1",
        "frontdoor",
        gate=_scored(True),
        path="direct",
        verdict_status="unavailable",
        revision_applied=False,
        verdict_ms=12.0,
    )
    (event,) = tap_events()
    assert event["schema"] == "review_gate/v1"
    assert event["task_id"] == "t1" and event["path"] == "direct" and event["role"] == "frontdoor"
    assert event["triggered"] is True and event["avg_q"] == pytest.approx(0.2)
    assert event["threshold"] == 0.6 and event["skip_reason"] == "scored"
    assert event["n_results"] == 5 and event["n_role_rows"] == 2
    assert event["answer_chars"] == len(LONG_ANSWER)
    assert event["verdict_status"] == "unavailable"  # never recorded as ok
    assert event["reviewer_role"]  # defaults to the reviewer binding once a verdict ran
    assert event["revision_applied"] is False
    assert (event["gate_ms"], event["verdict_ms"], event["revision_ms"]) == (1.5, 12.0, None)
    assert event["timing_source"] == "local"
    assert "request_id" not in event


def test_event_prefers_ri16_stage_ms() -> None:
    timing = rst.begin("t-stage", rst.PATH_CHAT)
    try:
        timing.record("review_gate", 3.25)
        timing.record("review_verdict", 950.0)
        fields = review_gate_event(
            "t-stage", "frontdoor", gate=_scored(True), path="repl",
            verdict_status="ok", verdict_ms=999.0,
        )
        assert fields["timing_source"] == "stage_ms"
        assert (fields["gate_ms"], fields["verdict_ms"]) == (3.25, 950.0)
        other = review_gate_event("t-other", "frontdoor", gate=_scored(False), path="repl")
        assert other["timing_source"] == "local" and other["gate_ms"] == 1.5
        assert other["verdict_status"] is None and other["reviewer_role"] is None
    finally:
        rst.clear()


def test_record_is_silent_with_the_tap_off_and_never_raises(monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
    monkeypatch.delenv("INFERENCE_TAP_EVENTS_FILE", raising=False)
    assert record_review_gate("t", "frontdoor", gate=_scored(False), path="direct") is False
    with patch("src.runtime.inference_tap.emit_request_event", side_effect=OSError("disk")):
        assert record_review_gate("t", "frontdoor", gate=_scored(False), path="direct") is False


# ── C1: one event per path ──────────────────────────────────────────────────


def _direct_run(
    primitives, state, request: ChatRequest, gate: GateScore | None, verdict, revised="Revised"
):
    from src.api.routes.chat_pipeline.direct_stage import _execute_direct

    primitives.llm_call.return_value = LONG_ANSWER
    routing = RoutingResult(
        task_id="direct-ri18",
        task_ir={},
        use_mock=False,
        routing_decision=["frontdoor"],
        routing_strategy="deterministic",
    )
    patches = [
        patch(
            "src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer",
            return_value=LONG_ANSWER,
        ),
        patch(
            "src.api.routes.chat_pipeline.direct_stage._should_formalize",
            return_value=(False, None),
        ),
        patch("src.api.routes.chat_pipeline.stages.features"),
        patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"),
        patch(
            "src.api.routes.chat_pipeline.direct_stage._architect_verdict_with_status",
            return_value=verdict,
        ),
        patch("src.api.routes.chat_pipeline.direct_stage._fast_revise", return_value=revised),
    ]
    if gate is not None:
        patches.append(
            patch("src.api.routes.chat_pipeline.direct_stage.evaluate_review_gate", return_value=gate)
        )
    for p in patches:
        p.start()
    try:
        from src.api.routes.chat_pipeline import stages

        stages.features.return_value.generation_monitor = False
        return _execute_direct(
            request, routing, primitives, state, time.perf_counter(), initial_role="frontdoor"
        )
    finally:
        for p in reversed(patches):
            p.stop()


def test_direct_path_records_wrong_and_revision(
    tap_events, mock_llm_primitives, mock_app_state
) -> None:
    result = _direct_run(
        mock_llm_primitives,
        mock_app_state,
        ChatRequest(prompt="Q?", real_mode=True), _scored(True), ("WRONG: it is B", "wrong")
    )
    assert result.answer == "Revised"
    (event,) = tap_events()
    assert (event["path"], event["triggered"], event["verdict_status"]) == ("direct", True, "wrong")
    assert event["revision_applied"] is True and event["revision_ms"] is not None


def test_direct_path_records_unavailable_not_ok(
    tap_events, mock_llm_primitives, mock_app_state
) -> None:
    result = _direct_run(
        mock_llm_primitives,
        mock_app_state,
        ChatRequest(prompt="Q?", real_mode=True), _scored(True), (None, "unavailable")
    )
    assert result.answer == LONG_ANSWER
    (event,) = tap_events()
    assert event["verdict_status"] == "unavailable"
    assert event["revision_applied"] is False


def test_direct_path_records_no_trigger_and_force_role_skip(
    tap_events, mock_llm_primitives, mock_app_state
) -> None:
    run = (mock_llm_primitives, mock_app_state)
    _direct_run(*run, ChatRequest(prompt="Q?", real_mode=True), _scored(False), ("x", "ok"))
    _direct_run(
        *run,
        ChatRequest(prompt="Q?", real_mode=True, force_role="frontdoor"),
        None,
        ("x", "ok"),
    )
    no_trigger, forced = tap_events()
    assert no_trigger["triggered"] is False and no_trigger["verdict_status"] is None
    assert no_trigger["skip_reason"] == "scored" and no_trigger["avg_q"] == pytest.approx(0.9)
    assert forced["skip_reason"] == "caller:force_role" and forced["avg_q"] is None


@pytest.mark.asyncio
async def test_unified_stream_path_records_wrong_and_revision(tap_events) -> None:
    from src.api.routes.chat_pipeline.stream_adapter import _stream_repl

    SA = "src.api.routes.chat_pipeline.stream_adapter"
    request = ChatRequest(prompt="What gas?", real_mode=True, max_turns=2)
    routing = RoutingResult(
        task_id="stream-ri18",
        task_ir={"task_type": "chat"},
        use_mock=False,
        routing_decision=["frontdoor"],
        routing_strategy="deterministic",
    )
    state = MagicMock()
    state.hybrid_router = None
    primitives = MagicMock()
    primitives.llm_call.return_value = "FINAL('x')"
    final = MagicMock(is_final=True, error=None)
    repl = MagicMock()
    repl.artifacts = {}
    repl._invoked_tools = []
    repl.execute.return_value = final
    with (
        patch(f"{SA}.REPLEnvironment", return_value=repl),
        patch(f"{SA}.build_root_lm_prompt", return_value="prompt"),
        patch(f"{SA}.build_corpus_context", return_value=""),
        patch(f"{SA}.extract_code_from_response", side_effect=lambda c: c),
        patch(f"{SA}.auto_wrap_final", side_effect=lambda c: c),
        patch(f"{SA}._resolve_answer", return_value=LONG_ANSWER),
        patch(f"{SA}.evaluate_review_gate", return_value=_scored(True)),
        patch(
            f"{SA}._architect_verdict_with_status",
            return_value=("WRONG: it is nitrogen", "wrong"),
        ) as verdict,
        patch(f"{SA}._fast_revise", return_value="Revised stream answer"),
    ):
        events = [e async for e in _stream_repl(request, routing, primitives, state, 0.0)]
    verdict.assert_called_once()
    assert any("Revised stream answer" in json.dumps(e) for e in events)
    (event,) = tap_events()
    assert (event["path"], event["task_id"]) == ("unified_stream", "stream-ri18")
    assert event["verdict_status"] == "wrong" and event["revision_applied"] is True


@pytest.mark.asyncio
async def test_legacy_stream_path_records_unavailable(tap_events, mock_app_state) -> None:
    """The legacy ``/chat/stream`` generator, driven to its FINAL turn.

    Also pins that the review block's names do not collide with the generator's
    locals (the generator assigns a local ``elapsed_ms``).
    """
    from src.api.routes.chat import chat_stream

    C = "src.api.routes.chat"
    mock_app_state.hybrid_router = None
    primitives = MagicMock()
    primitives.llm_call.return_value = "FINAL('x')"
    final = MagicMock(is_final=True, error=None)
    repl = MagicMock()
    repl.artifacts = {}
    repl._invoked_tools = []
    repl.execute.return_value = final
    feats = MagicMock()
    feats.unified_streaming = False
    with (
        patch(f"{C}.features", return_value=feats),
        patch(f"{C}.ensure_memrl_initialized"),
        patch(f"{C}.LLMPrimitives", return_value=primitives),
        patch(f"{C}.REPLEnvironment", return_value=repl),
        patch(f"{C}.build_root_lm_prompt", return_value="prompt"),
        patch(f"{C}.build_corpus_context", return_value=""),
        patch(f"{C}.extract_code_from_response", side_effect=lambda c: c),
        patch(f"{C}.auto_wrap_final", side_effect=lambda c: c),
        patch(f"{C}._resolve_answer", return_value=LONG_ANSWER),
        patch(f"{C}.score_completed_task"),
        patch(f"{C}.evaluate_review_gate", return_value=_scored(True)),
        patch(f"{C}._architect_verdict_with_status", return_value=(None, "unavailable")),
        patch(f"{C}._fast_revise") as revise,
    ):
        response = await chat_stream(
            ChatRequest(prompt="What gas?", real_mode=True, max_turns=2), mock_app_state
        )
        chunks = [
            c.decode() if isinstance(c, bytes) else str(c)
            async for c in response.body_iterator
        ]
    revise.assert_not_called()
    assert LONG_ANSWER in "".join(chunks)
    (event,) = tap_events()
    assert event["path"] == "legacy_stream" and event["task_id"].startswith("stream-")
    assert event["verdict_status"] == "unavailable" and event["revision_applied"] is False
    assert event["verdict_ms"] is not None


@pytest.mark.parametrize(
    "module_path,path_value",
    [
        ("src.api.routes.chat_pipeline.stream_adapter", "unified_stream"),
        ("src.api.routes.chat", "legacy_stream"),
    ],
)
def test_stream_sites_import_the_status_api(module_path: str, path_value: str) -> None:
    """Both streaming sites call the ``_with_status`` verdict and record their path.

    Both are also driven end to end above; this pins the import-level wiring.
    """
    import importlib
    import inspect

    module = importlib.import_module(module_path)
    source = inspect.getsource(module)
    assert "_architect_verdict_with_status(" in source
    assert "evaluate_review_gate(" in source and "record_review_gate(" in source
    assert f"REVIEW_PATH_{path_value.upper()}" in source
    assert getattr(chat_review, f"REVIEW_PATH_{path_value.upper()}") == path_value
    # the old status-less call is gone from the review block
    assert "verdict = _architect_verdict(" not in source


@pytest.mark.asyncio
async def test_repl_path_records_ok(tap_events) -> None:
    from src.api.routes.chat_pipeline.repl_executor import _execute_repl
    from src.graph.state import TaskResult
    from src.llm_primitives import LLMPrimitives

    request = ChatRequest(prompt="What is 6*7?", context="", real_mode=True, max_turns=5)
    routing = RoutingResult(
        task_id="repl-ri18",
        task_ir={},
        use_mock=False,
        routing_decision=["frontdoor"],
        routing_strategy="direct",
    )
    primitives = MagicMock(spec=LLMPrimitives)
    primitives.mock_mode = False
    primitives.total_tokens_generated = 100
    primitives.total_prompt_eval_ms = 50
    primitives.total_generation_ms = 200
    primitives.total_http_overhead_ms = 10
    primitives._last_predicted_tps = 15.0
    primitives._backends = {"frontdoor": MagicMock()}
    primitives.get_cache_stats.return_value = {"hits": 0, "misses": 0}
    state = MagicMock()
    state.hybrid_router = None
    result = TaskResult(answer=LONG_ANSWER, success=True, turns=1, role_history=["frontdoor"])
    with (
        patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as repl_cls,
        patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=result),
        patch(
            "src.api.routes.chat_pipeline.repl_executor.evaluate_review_gate",
            return_value=_scored(True),
        ),
        patch(
            "src.api.routes.chat_pipeline.repl_executor._architect_verdict_with_status",
            return_value=(None, "ok"),
        ),
        patch("src.api.routes.chat_pipeline.repl_executor._fast_revise") as revise,
    ):
        repl = MagicMock()
        repl.artifacts = {}
        repl._tool_invocations = 0
        repl.tool_registry = None
        repl_cls.return_value = repl
        response = await _execute_repl(
            request=request,
            routing=routing,
            primitives=primitives,
            state=state,
            start_time=time.perf_counter(),
            initial_role="frontdoor",
        )
    revise.assert_not_called()
    assert response.answer == LONG_ANSWER
    events = [e for e in tap_events() if e["path"] == "repl"]
    assert len(events) == 1
    assert events[0]["verdict_status"] == "ok" and events[0]["revision_applied"] is False
    assert events[0]["task_id"] == "repl-ri18"
