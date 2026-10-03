"""RI-16: per-stage routing-decision latency on the live /chat path.

All mocked — no network, no model. Covers: every stage timer is populated on a
routed request, ``total`` >= the sum of the sequential stages, a stage that did not
run records ``None`` (never 0) — including ``review_gate`` / ``review_verdict``, which
are always ``None`` since RI-18c removed the review gate — and the emitted schema
(routing_decision event, task_completed event, ChatResponse) is stable.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_pipeline.routing import _route_request
from src.api.routes.chat_utils import RoutingResult
from src.runtime import routing_stage_timing as rst

# Golden: the stage_ms schema. Changing it is a telemetry-schema change — update
# scripts/analysis/routing_stage_latency.py and this list together.
GOLDEN_STAGE_KEYS = [
    "memrl_init",
    "priors",
    "route",
    "xmas",
    "factual_risk",
    "failure_veto",
    "difficulty",
    "trinity",
    "route_total",
    "mode",
    "routing_context",
    "review_gate",
    "review_verdict",
    "total",
]
PRE_EXECUTION_STAGES = (
    "memrl_init",
    "priors",
    "route",
    "xmas",
    "factual_risk",
    "failure_veto",
    "difficulty",
    "trinity",
)


@pytest.fixture(autouse=True)
def _clear_timing():
    rst.clear()
    yield
    rst.clear()


def _routed_state(progress_logger=None) -> MagicMock:
    state = MagicMock()
    state.hybrid_router.route.return_value = (["frontdoor"], "learned")
    state.hybrid_router.last_decision_meta = {}
    state.failure_graph = None
    state.llm_primitives = None
    state.registry = None
    state.progress_logger = progress_logger
    return state


def _route(state, prompt: str = "Explain the difference between TCP and UDP.") -> RoutingResult:
    request = ChatRequest(prompt=prompt, real_mode=True)
    with (
        patch("src.api.routes.chat_pipeline.routing.ensure_memrl_initialized"),
        patch("src.api.routes.chat_pipeline.routing_decision.features") as feats,
    ):
        feats.return_value.skillbank = False
        return _route_request(request, state)


# ── module contract ─────────────────────────────────────────────────────


def test_stage_keys_golden() -> None:
    assert list(rst.STAGE_KEYS) == GOLDEN_STAGE_KEYS
    assert set(rst.TOTAL_COMPONENTS) <= set(rst.STAGE_KEYS)
    assert "review_verdict" not in rst.TOTAL_COMPONENTS


def test_fresh_timing_has_every_key_as_none() -> None:
    timing = rst.begin("t-1", rst.PATH_CHAT)
    assert timing.telemetry() == {
        "routing_path": "chat",
        "stage_ms": dict.fromkeys(GOLDEN_STAGE_KEYS),
    }


def test_record_accumulates_and_recomputes_total() -> None:
    timing = rst.begin("t-1", rst.PATH_CHAT)
    timing.record("route_total", 10.0)
    timing.record("review_gate", 2.0, accumulate=True)
    timing.record("review_gate", 3.0, accumulate=True)
    timing.record("review_verdict", 500.0)
    timing.record("total", 1.0)  # ignored: total is derived
    timing.record("not_a_stage", 1.0)  # ignored: schema is fixed
    ms = timing.stage_ms()
    assert ms["review_gate"] == 5.0
    assert ms["total"] == 15.0  # verdict excluded
    assert set(ms) == set(GOLDEN_STAGE_KEYS)


def test_untimed_helpers_are_passthrough() -> None:
    assert rst.current() is None
    with rst.timed("route"):
        pass
    assert rst.call_timed("route", lambda x: x + 1, 1) == 2
    assert rst.telemetry_for("anything") is None


def test_telemetry_is_task_id_matched() -> None:
    rst.begin("t-1", rst.PATH_CHAT)
    assert rst.telemetry_for("t-1") is not None
    assert rst.telemetry_for("t-2") is None


def test_snapshot_is_a_copy() -> None:
    timing = rst.begin("t-1", rst.PATH_CHAT)
    snap = timing.stage_ms()
    timing.record("mode", 4.0)
    assert snap["mode"] is None


# ── the live /chat routing path ─────────────────────────────────────────


def test_route_request_populates_every_pre_execution_stage() -> None:
    progress_logger = MagicMock()
    result = _route(_routed_state(progress_logger))

    meta = progress_logger.log_task_started.call_args.kwargs["routing_meta"]
    assert meta["routing_path"] == "chat"
    stage_ms = meta["stage_ms"]
    assert list(stage_ms) == GOLDEN_STAGE_KEYS
    for stage in (*PRE_EXECUTION_STAGES, "route_total"):
        assert isinstance(stage_ms[stage], float), stage
        assert stage_ms[stage] >= 0.0
    # Post-execution stages have not run yet.
    for stage in ("mode", "routing_context", "review_gate", "review_verdict"):
        assert stage_ms[stage] is None, stage
    assert rst.telemetry_for(result.task_id)["routing_path"] == "chat"


def test_route_total_covers_its_sequential_stages() -> None:
    progress_logger = MagicMock()
    _route(_routed_state(progress_logger))
    stage_ms = progress_logger.log_task_started.call_args.kwargs["routing_meta"]["stage_ms"]
    inner = sum(stage_ms[s] for s in PRE_EXECUTION_STAGES)
    # rounding to 3 dp per stage: allow 1 µs per stage
    assert stage_ms["route_total"] + 0.001 * len(PRE_EXECUTION_STAGES) >= inner
    assert stage_ms["total"] == stage_ms["route_total"]


def test_full_request_stages_and_total(tmp_path: Path) -> None:
    """Route -> mode -> routing context, one request.

    RI-18c removed the answer review gate, so ``review_gate`` and ``review_verdict``
    stay ``None`` on a full request and ``total`` is the remaining decision stages.
    """
    from src.api.routes.chat_routing import _select_mode
    from src.prompt_builders import build_routing_context

    state = _routed_state()
    result = _route(state)

    router = MagicMock()
    router.route_with_mode.return_value = (["frontdoor"], "learned", "repl")
    router.retriever.retrieve_for_routing.return_value = []
    mode_state = MagicMock(hybrid_router=router)
    assert _select_mode("Refactor this module into two files please", "", mode_state) == "repl"

    build_routing_context(role="frontdoor", hybrid_router=router, task_description="x")

    stage_ms = rst.telemetry_for(result.task_id)["stage_ms"]
    retired = ("review_gate", "review_verdict")
    for stage in GOLDEN_STAGE_KEYS:
        if stage in retired:
            assert stage_ms[stage] is None, stage
        else:
            assert isinstance(stage_ms[stage], float), stage
    sequential = sum(stage_ms[s] or 0.0 for s in rst.TOTAL_COMPONENTS)
    assert stage_ms["total"] == pytest.approx(sequential, abs=0.005)


def test_forced_mode_and_skipped_review_gate_record_none() -> None:
    """A stage that never ran is None — consistently, never 0."""
    result = _route(_routed_state())
    stage_ms = rst.telemetry_for(result.task_id)["stage_ms"]
    assert stage_ms["mode"] is None
    assert stage_ms["review_gate"] is None
    assert stage_ms["review_verdict"] is None
    assert stage_ms["total"] == stage_ms["route_total"]


def test_direct_stage_records_no_review_gate(mock_app_state, mock_llm_primitives) -> None:
    """RI-18c: the direct stage no longer runs the review gate or its verdict, so both
    RI-16 stages stay ``None`` whether or not ``force_role`` is set."""
    from src.api.routes.chat_pipeline.direct_stage import _execute_direct

    def run(request: ChatRequest, task_id: str) -> dict:
        rst.begin(task_id, rst.PATH_CHAT)
        routing = RoutingResult(
            task_id=task_id,
            task_ir={},
            use_mock=False,
            routing_decision=["frontdoor"],
            routing_strategy="deterministic",
        )
        mock_llm_primitives.llm_call.reset_mock()
        mock_llm_primitives.llm_call.return_value = "Direct answer that is longer than fifty chars."
        with (
            patch(
                "src.api.routes.chat_pipeline.direct_stage._truncate_looped_answer",
                side_effect=lambda answer, _prompt: answer,
            ),
            patch(
                "src.api.routes.chat_pipeline.direct_stage._should_formalize",
                return_value=(False, None),
            ),
            patch("src.api.routes.chat_pipeline.stages.features") as feats,
            patch("src.api.routes.chat_pipeline.direct_stage.score_completed_task"),
        ):
            feats.return_value.generation_monitor = False
            response = _execute_direct(
                request, routing, mock_llm_primitives, mock_app_state, time.perf_counter(),
                initial_role="frontdoor",
            )
        return {
            "answer": response.answer,
            "llm_calls": mock_llm_primitives.llm_call.call_count,
            **rst.telemetry_for(task_id),
        }

    for request, task_id in (
        (ChatRequest(prompt="Direct question", real_mode=True), "direct-1"),
        (ChatRequest(prompt="Direct question", real_mode=True, force_role="frontdoor"), "direct-2"),
    ):
        out = run(request, task_id)
        assert out["answer"] == "Direct answer that is longer than fifty chars."
        assert out["llm_calls"] == 1  # the answer only: no verdict, no revision
        assert out["stage_ms"]["review_gate"] is None
        assert out["stage_ms"]["review_verdict"] is None


# ── where the telemetry lands ───────────────────────────────────────────


def _read_events(log_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return rows


def test_progress_log_carries_stage_ms_on_decision_and_completion(tmp_path: Path) -> None:
    from orchestration.repl_memory.progress_logger import ProgressLogger

    logger = ProgressLogger(log_dir=tmp_path, buffer_size=1000)
    result = _route(_routed_state(logger))
    rst.current().record("mode", 1.5)
    logger.log_task_completed(result.task_id, success=True, details="ok")
    logger.log_task_completed("some-other-task", success=True, details="ok")
    logger.flush()

    events = _read_events(tmp_path)
    decision = next(e for e in events if e["event_type"] == "routing_decision")
    completed = [e for e in events if e["event_type"] == "task_completed"]
    assert decision["data"]["routing_path"] == "chat"
    assert list(decision["data"]["stage_ms"]) == GOLDEN_STAGE_KEYS
    assert decision["data"]["stage_ms"]["mode"] is None  # snapshot, not a live reference

    ours = next(e for e in completed if e["task_id"] == result.task_id)
    assert ours["data"]["routing_path"] == "chat"
    assert ours["data"]["stage_ms"]["mode"] == 1.5
    assert ours["data"]["stage_ms"]["total"] >= ours["data"]["stage_ms"]["route_total"]
    assert "task_record_v1" in ours["data"]
    assert "stage_ms" not in ours["data"]["task_record_v1"]

    other = next(e for e in completed if e["task_id"] == "some-other-task")
    assert "stage_ms" not in other["data"]
    assert "routing_path" not in other["data"]


def test_chat_response_schema_has_optional_stage_ms() -> None:
    field = ChatResponse.model_fields["routing_stage_ms"]
    assert field.default is None
    resp = ChatResponse(answer="a", turns=1, elapsed_seconds=0.1, mock_mode=False)
    assert resp.routing_stage_ms is None


@pytest.mark.asyncio
async def test_handle_chat_stamps_stage_ms_on_response(mock_app_state, mock_llm_primitives):
    """Non-streaming /chat: the response carries the request's stage_ms, mode included."""
    from src.api.routes.chat import _handle_chat

    mock_app_state.progress_logger = None
    mock_llm_primitives.request_context = MagicMock(return_value=nullcontext())
    executed = ChatResponse(
        answer="done", turns=1, elapsed_seconds=0.1, mock_mode=False, real_mode=True,
        routed_to="frontdoor", mode="repl",
    )
    request = ChatRequest(prompt="Refactor this module into two files please", real_mode=True)

    def fake_route(req, state):
        return _route(_routed_state(), prompt=req.prompt)

    with (
        patch("src.api.routes.chat._route_request", side_effect=fake_route),
        patch("src.api.routes.chat._preprocess", return_value=None),
        patch("src.api.routes.chat._init_primitives", return_value=mock_llm_primitives),
        patch("src.api.routes.chat._plan_review_gate", return_value=None),
        patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)),
        patch("src.api.routes.chat._execute_vision_multimodal", new=AsyncMock(return_value=None)),
        patch("src.api.routes.chat._execute_proactive", new=AsyncMock(return_value=None)),
        patch("src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)),
        patch("src.api.routes.chat._execute_repl", new=AsyncMock(return_value=executed)),
    ):
        result = await _handle_chat(request, mock_app_state)

    stage_ms = result.routing_stage_ms
    assert stage_ms is not None
    assert list(stage_ms) == GOLDEN_STAGE_KEYS
    assert isinstance(stage_ms["route_total"], float)
    assert isinstance(stage_ms["mode"], float)  # _select_mode ran (not forced)
    assert stage_ms["review_gate"] is None  # execution stage was mocked out
    assert stage_ms["total"] >= stage_ms["route_total"] + stage_ms["mode"] - 0.002
