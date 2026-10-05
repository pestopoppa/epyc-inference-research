"""INF-78 integration: OAB-7 context bundle + OAB-8 scouts in ONE /chat request.

Offline and inference-free. The request goes through the real `/chat` route (task scope +
quiescence carrier), the real `_handle_chat` (proactive guard, scout stage 6.8) and the
real REPL executor/loop; only the model endpoints are doubles: the scouts talk to a
scripted transport, the REPL root role to a recording model.

Acceptance:
  * the proactive stage does not intercept (prod default parallel_execution=True): the
    guard is the UNION (task_root OR context_bundle) and both are set here;
  * the scouts run and their block HEADS request.prompt; the bundle index is APPENDED by
    the REPL -- so the planner's root prompt reads: scout block, caller prompt, bundle index;
  * the bundle's bytes stay out of every prompt sent to any role, and the pulls are
    accounted (ChatResponse.context_pulls) alongside the scout provenance (ChatResponse.scouts);
  * quiescence holds: the carrier is installed and echoed, no scout is still in flight,
    and the process does no measurable CPU work after the reply. (The turn-1 architect
    prewarm this test used to inject was deleted in UFH14-B4e; OAB-3's own acceptance test
    covers suppression of an injected launch site.)

Run: taskset -c 72-79 .venv/bin/python -m pytest tests/unit/test_inf78_oab7_oab8_composition.py -q
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline import scout_stage as S
from src.repl_environment import task_root as TR
from src.runtime import quiescence as Q
from src.runtime import trailing_work_witness as W
from tests.unit.test_oab7_context_bundle import (
    PULLS_SCHEMA,
    SENTINEL,
    _payload,
    _RecordingPrimitives,
)
from tests.unit.test_oab8_scouts import KERNEL_C, FakeResolver, ScriptedTransport, _summary

SCRATCH_BASE = Path("/mnt/raid0/llm/tmp")
SCOUT_HEADER = "## Orchestrator scout findings (INF-78 OAB-8)"
BUNDLE_HEADER = "[Context bundle: the REPL variable `context`]"
CALLER_PROMPT = "PROPOSE ONE HYPOTHESIS for the hot kernel."


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.delenv(TR.ENV_VAR, raising=False)
    hold_dir = SCRATCH_BASE / f"test_inf78_comp_{os.getpid()}_{time.monotonic_ns()}"
    monkeypatch.setenv(Q.HOLD_DIR_ENV, str(hold_dir))
    TR.clear_request_scope()
    Q.clear()
    yield
    TR.clear_request_scope()
    Q.clear()
    shutil.rmtree(hold_dir, ignore_errors=True)


@pytest.fixture()
def lane():
    base = SCRATCH_BASE / f"test_inf78_comp_{uuid.uuid4().hex[:10]}"
    root = base / "lane"
    (root / "ggml" / "src").mkdir(parents=True)
    (root / "ggml" / "src" / "quants.c").write_text(KERNEL_C)
    (root / "ggml" / "src" / "ops.cpp").write_text(
        "void ggml_compute_forward_mul_mat(void) {\n    // matmul driver\n}\n")
    yield root
    shutil.rmtree(base, ignore_errors=True)


class _Primitives(_RecordingPrimitives):
    """The OAB-7 recording model, plus the server map the scout stage resolves its URL from."""

    def __init__(self, script):
        super().__init__(script)
        self.server_urls = {"frontdoor": "http://127.0.0.1:8070"}


class _FakeHttpRequest:
    async def is_disconnected(self) -> bool:
        return False


def _state():
    from src.api.state import AppState

    state = MagicMock(spec=AppState)
    for attr in ("progress_logger", "hybrid_router", "tool_registry", "script_registry",
                 "registry", "session_store"):
        setattr(state, attr, None)
    return state


@pytest.mark.asyncio
async def test_bundle_and_scouts_compose_in_one_chat_request(lane):
    from src.api.routes.chat import chat
    from src.api.routes.chat_utils import RoutingResult
    from src.backends import context_limits as CL

    transport = ScriptedTransport({"T-dot": [_summary("dot")], "T-ops": [_summary("ops")]})
    prims = _Primitives([
        # turn 1: pull the whole `big` section (sentinel included) into a variable; print a
        # count. (Only counts are bound to names: get_state's 80-char repr preview of each
        # user variable is OAB-7's second, bounded + counted route into the root prompt, so
        # binding the grep hits themselves would legitimately show the sentinel.)
        "```python\nbig = context['big']\nn_hits = len(context.grep('needle'))\n"
        "print(len(big), n_hits)\n```",
        "```python\nFINAL('{\"abstain\": \"nothing to change\"}')\n```",
    ])
    routing = RoutingResult(task_id="inf78-comp", task_ir={"task_type": "chat", "objective": "x"},
                            use_mock=False, routing_decision=["frontdoor"],
                            routing_strategy="rules", skill_ids=[])
    request = ChatRequest(
        prompt=CALLER_PROMPT, real_mode=True, mock_mode=False, force_mode="repl", max_turns=6,
        task_root=str(lane), quiescent_after=True, request_id="inf78-comp-1",
        context_bundle=_payload(), context_print_cap_bytes=1024,
        scouts={"enabled": True, "max": 4, "max_turns": 4, "budget_s": 20.0,
                "targets": [
                    {"symbol": "ggml_vec_dot_q4_K_q8_K", "label": "T-dot", "share": 0.31},
                    {"file": "ggml/src/ops.cpp", "label": "T-ops"}]},
    )
    captured: dict = {}
    proactive_feats = SimpleNamespace(parallel_execution=True)
    CL.set_context_limit_resolver(FakeResolver(total=4, busy=0))
    try:
        with patch("src.api.routes.chat._route_request", return_value=routing), \
                patch("src.api.routes.chat._preprocess", return_value=None), \
                patch("src.api.routes.chat._init_primitives", return_value=prims), \
                patch("src.api.routes.chat._plan_review_gate", return_value=None), \
                patch("src.api.routes.chat._execute_vision", new=AsyncMock(return_value=None)), \
                patch("src.api.routes.chat._try_cheap_first", new=AsyncMock(return_value=None)), \
                patch("src.api.routes.chat_pipeline.proactive_stage.features",
                      return_value=proactive_feats), \
                patch("src.proactive_delegation.classify_task_complexity",
                      side_effect=AssertionError("proactive stage intercepted the request")), \
                patch.object(S, "ChatCompletionsTransport",
                             side_effect=lambda url, **kw: (captured.setdefault("url", url),
                                                            transport)[1]):
            resp = await chat(request, _FakeHttpRequest(), _state())
            # ── the reply has been produced; witness this process from here ──
            verdict = await asyncio.to_thread(
                W.witness, {os.getpid(): "orchestrator(test)"}, window_s=2.0,
                threshold_core_s=0.5)
    finally:
        CL.set_context_limit_resolver(None)

    assert resp.error_code is None, resp.error_detail
    assert resp.answer == '{"abstain": "nothing to change"}'
    assert request.prompt == CALLER_PROMPT, "the caller's request object is not mutated"

    # scouts ran, concurrently capped by free slots, against the routed role's server
    sc = resp.scouts
    assert sc is not None and sc["launched"] == 2 and sc["completed"] == 2, sc
    assert captured["url"] == "http://127.0.0.1:8070"
    assert transport.active == 0, "no scout call is still in flight after the reply"
    assert {label for label, _ in transport.calls} == {"T-dot", "T-ops"}

    # the planner's root prompt: scout block, then the caller prompt, then the bundle index
    root = prims.root_prompts()
    assert len(root) == 2
    first = root[0]
    i_scout, i_prompt, i_index = (first.find(SCOUT_HEADER), first.find(CALLER_PROMPT),
                                  first.find(BUNDLE_HEADER))
    assert -1 not in (i_scout, i_prompt, i_index), (i_scout, i_prompt, i_index)
    assert i_scout < i_prompt < i_index, (i_scout, i_prompt, i_index)
    assert "dot: the loop at ggml/src/quants.c:9" in first and "ops: the loop" in first
    assert "| big | text |" in first
    for role, prompt in prims.calls:
        assert SENTINEL not in prompt, f"pulled bundle bytes leaked into a {role} prompt"
        assert "row 399: filler" not in prompt
    for _label, messages in transport.calls:
        assert all(SENTINEL not in m["content"] for m in messages), "scouts never see the bundle"

    # pulls accounted
    acc = resp.context_pulls
    assert acc is not None and acc["schema"] == PULLS_SCHEMA
    assert acc["totals"]["pull_calls"] >= 2
    big = acc["sections"]["big"]
    assert big["bytes_pulled"] >= big["offered_bytes"] and big["coverage"] == 1.0
    assert [t["turn"] for t in acc["turns"]] == [1, 2]

    # quiescence: the carrier was installed and echoed, it did not outlive the request,
    # and nothing burned CPU after the reply
    assert resp.quiescence is not None and resp.quiescence["quiescent_after"] is True
    assert Q.current() is None
    assert verdict["quiescent"] is True, verdict
    assert resp.task_scope is not None


def test_cli_sends_bundle_and_scouts_together(tmp_path):
    """The actor CLI carries both flag sets in one request, and the sidecar keeps both
    echoes (context_pulls AND scouts)."""
    import json

    from tests.unit.test_autokernel_actor_cli import (
        OK_SCHEMA,
        MockChat,
        _schema_file,
        chat_response,
        run_main,
    )

    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps(_payload()), encoding="utf-8")
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps([{"symbol": "ggml_vec_dot_q4_K_q8_K", "share": 0.31},
                                   {"file": "ggml/src/ops.cpp", "label": "ops"}]))
    pulls = {"schema": PULLS_SCHEMA, "totals": {"bytes_pulled": 123}}
    scouts_echo = {"schema": S.SCHEMA, "launched": 2, "completed": 2, "scouts": []}
    sidecar = tmp_path / "prov.json"
    with MockChat(body=chat_response('{"ok": true}', context_pulls=pulls,
                                     scouts=scouts_echo)) as mock:
        code, _out, err = run_main(
            ["--root", str(tmp_path), "--read-only",
             "--schema", str(_schema_file(tmp_path, OK_SCHEMA)), "--url", mock.url,
             "--context-bundle", str(bundle), "--context-print-cap-bytes", "2048",
             "--scout-targets", str(targets), "--scouts-max", "2",
             "--provenance-out", str(sidecar)], prompt="index only")
    assert code == 0, err
    body = mock.requests[0]["body"]
    assert body["context_bundle"] == _payload() and body["context_print_cap_bytes"] == 2048
    assert body["scouts"]["enabled"] is True and body["scouts"]["max"] == 2
    assert len(body["scouts"]["targets"]) == 2 and body["force_mode"] == "repl"
    record = json.loads(sidecar.read_text())
    assert {"context_bundle", "scouts"} <= set(record["request"]["fields_sent"])
    assert record["request"]["context_bundle"]["sections"] == 4
    assert record["request"]["scouts"]["targets"] == 2
    assert record["context_bundle_acknowledged"] is True
    assert record["response"]["context_pulls"] == pulls
    assert record["response"]["scouts"] == scouts_echo
