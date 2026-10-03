"""UFH14-B1 in the REPL graph: session compaction vs the rendered prompt, the
context-overflow path, and the F2 wall-time forced-answer turn.

Real REPLEnvironment, mocked LLM primitives (tests/integration/conftest.py).
"""

from __future__ import annotations

import time

import pytest
from pydantic_graph import GraphRunContext

from src.exceptions import ContextOverflowError
from src.features import Features, reset_features, set_features
from src.graph.helpers import ANSWER_FORCE_MESSAGE, _execute_turn, answer_force_at
from src.roles import Role

pytestmark = pytest.mark.integration

_NOPROGRESS = "```python\nfor i in range(3):\n    print(f'step {i}')\n```"
BIG = "x" * 50_000   # well past the 12000-char compaction trigger


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    from src.backends.context_limits import ContextLimitResolver, set_context_limit_resolver

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
    set_context_limit_resolver(
        ContextLimitResolver(live=False, registry_facts=lambda: {}, role_urls=lambda: {}))
    yield
    set_context_limit_resolver(None)
    reset_features()


def _ctx(graph_ctx, *, context_override: str | None = None, responses=None):
    state, deps = graph_ctx(task_id="ufh14-b1", responses=responses or [_NOPROGRESS])
    deps.primitives.registry = None
    deps.primitives.server_urls = {}
    deps.primitives._count_tokens = lambda text: len(text) // 4
    if context_override is not None:
        state.context = context_override   # TaskState.context only, not the REPL variable
    return GraphRunContext(state=state, deps=deps), state, deps


def _calls(deps, role=None):
    calls = deps.primitives.llm_call.call_args_list
    return [c for c in calls if role is None or c.kwargs.get("role") == role]


async def _run(ctx, turns: int) -> None:
    for _ in range(turns):
        await _execute_turn(ctx, Role.FRONTDOOR)


@pytest.mark.asyncio
async def test_task_state_context_never_reaches_the_turn_prompt(graph_ctx):
    """The DESIGN-NOTE finding, as a test: two runs that differ ONLY in
    TaskState.context (one compacting from turn 5) send byte-identical turn prompts."""
    set_features(Features(session_compaction=True))
    small_ctx, _, small_deps = _ctx(graph_ctx, context_override="short")
    big_ctx, big_state, big_deps = _ctx(graph_ctx, context_override=BIG)
    await _run(small_ctx, 6)
    await _run(big_ctx, 6)
    assert big_state.compaction_count >= 1            # compaction did run on the big one
    small = [c.args[0] for c in _calls(small_deps, str(Role.FRONTDOOR))]
    big = [c.args[0] for c in _calls(big_deps, str(Role.FRONTDOOR))]
    assert len(small) == len(big) == 6
    assert small == big


@pytest.mark.asyncio
async def test_compaction_makes_no_index_llm_call_by_default(graph_ctx):
    """The fix: an index of a field no prompt renders is not worth a worker_general
    call (which carried the whole unbounded context as input)."""
    set_features(Features(session_compaction=True))
    ctx, state, deps = _ctx(graph_ctx, context_override=BIG)
    await _run(ctx, 6)
    assert state.compaction_count >= 1
    assert "[Fallback Index]" in state.context
    assert _calls(deps, "worker_general") == []


@pytest.mark.asyncio
async def test_llm_index_flag_restores_the_call_with_capped_input(graph_ctx):
    from src.graph import compaction

    set_features(Features(session_compaction=True, session_compaction_llm_index=True))
    huge = "z" * 400_000
    ctx, state, deps = _ctx(graph_ctx, context_override=huge)
    await _run(ctx, 5)
    (call,) = _calls(deps, "worker_general")[:1]
    window = compaction.UNKNOWN_CONTEXT_FALLBACK_TOKENS     # no live/registry window here
    cap = int(window * compaction.INDEX_INPUT_WINDOW_SHARE) * compaction.INDEX_CHARS_PER_TOKEN
    assert len(call.args[0]) <= cap
    assert len(call.args[0]) < len(huge)


@pytest.mark.asyncio
async def test_context_overflow_no_longer_claims_a_compaction(graph_ctx):
    """The overflow path used to force-compact TaskState.context (never rendered)
    and reply 'context compacted; retry the turn' with nothing that overflowed shrunk."""
    set_features(Features(session_compaction=True, session_compaction_llm_index=True))
    ctx, state, deps = _ctx(graph_ctx, context_override=BIG)

    def _overflow(prompt, role="worker", **kwargs):
        raise ContextOverflowError("too big", kind=ContextOverflowError.REQUEST_TOO_LARGE,
                                   role=str(role), n_prompt_tokens=70_000, n_ctx=65_536)

    deps.primitives.llm_call.side_effect = _overflow
    output, error, is_final, _ = await _execute_turn(ctx, Role.FRONTDOOR)
    assert (output, is_final) == ("", False)
    assert error.startswith("LLM call failed:") and "compacted" not in error
    assert state.context == BIG                          # nothing pretended to shrink
    assert state.compaction_count == 0
    assert _calls(deps, "worker_general") == []          # no extra call to the overflowed server


# ── F2: wall-time forced-answer turn ─────────────────────────────────────────


class _Prims:
    def __init__(self, deadline):
        self._deadline = deadline

    def get_request_deadline_s(self):
        return self._deadline


def test_answer_force_at_is_off_by_default_and_needs_a_deadline():
    now = time.perf_counter()
    set_features(Features(repl_answer_force=False))
    assert answer_force_at(_Prims(now + 100), now=now) is None
    set_features(Features(repl_answer_force=True))
    assert answer_force_at(_Prims(None), now=now) is None
    assert answer_force_at(_Prims(now - 1), now=now) is None
    assert answer_force_at(_Prims(now + 100), now=now) == pytest.approx(now + 65)


def test_answer_force_frac_env(monkeypatch):
    set_features(Features(repl_answer_force=True))
    monkeypatch.setenv("ORCHESTRATOR_REPL_ANSWER_FORCE_FRAC", "0.5")
    assert answer_force_at(_Prims(100.0), now=0.0) == pytest.approx(50.0)
    monkeypatch.setenv("ORCHESTRATOR_REPL_ANSWER_FORCE_FRAC", "1.5")   # out of range: default
    assert answer_force_at(_Prims(100.0), now=0.0) == pytest.approx(65.0)


@pytest.mark.asyncio
async def test_forced_answer_turn_demands_final_and_skips_compaction(graph_ctx):
    set_features(Features(session_compaction=True))
    ctx, state, deps = _ctx(graph_ctx, context_override=BIG)
    state.turns = 8                                       # past the compaction min-turns gate
    state.answer_force_at_s = time.perf_counter() - 1.0   # budget point already passed
    await _execute_turn(ctx, Role.FRONTDOOR)
    (prompt,) = [c.args[0] for c in _calls(deps, str(Role.FRONTDOOR))]
    assert ANSWER_FORCE_MESSAGE in prompt
    assert state.answer_forced_turns == 1
    assert state.compaction_count == 0                    # no compaction on the answer turn


@pytest.mark.asyncio
async def test_no_forced_answer_before_the_point_or_when_unset(graph_ctx):
    ctx, state, deps = _ctx(graph_ctx)
    state.answer_force_at_s = time.perf_counter() + 3600
    await _execute_turn(ctx, Role.FRONTDOOR)
    state.answer_force_at_s = None
    await _execute_turn(ctx, Role.FRONTDOOR)
    prompts = [c.args[0] for c in _calls(deps, str(Role.FRONTDOOR))]
    assert len(prompts) == 2 and not any(ANSWER_FORCE_MESSAGE in p for p in prompts)
    assert state.answer_forced_turns == 0


def test_answer_force_fields_survive_the_langgraph_round_trip():
    from src.graph.langgraph.state import lg_to_task_state, task_state_to_lg
    from src.graph.state import TaskState

    src_state = TaskState(task_id="t", answer_force_at_s=12.5, answer_forced_turns=2)
    dst = TaskState(task_id="t")
    lg_to_task_state(task_state_to_lg(src_state), dst)
    assert (dst.answer_force_at_s, dst.answer_forced_turns) == (12.5, 2)
