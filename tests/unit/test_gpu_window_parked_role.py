"""Parked roles: a GPU server lent to AutoKernel fails fast with 503 role_parked.

Offline and inference-free. Every test points ``ORCHESTRATOR_GPU_WINDOW_FILE`` at
its own tmp file (the suite default is ``off``, see ``tests/conftest.py``).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.exceptions import RoleParkedError
from src.runtime import gpu_window as gw

PARKED_ROLES = ["architect_critic", "coder_escalation", "ingest_long_context"]


@pytest.fixture
def window(tmp_path, monkeypatch):
    path = tmp_path / "gpu-window" / "mi210.json"
    monkeypatch.setenv(gw.PATH_ENV, str(path))
    log = tmp_path / "serving_calls.jsonl"
    monkeypatch.setenv("ORCHESTRATOR_SERVING_CALLS_LOG", str(log))
    return SimpleNamespace(path=path, log=log)


def _park(window, **overrides) -> None:
    gw.park(roles=PARKED_ROLES, ports=[8083], holder="autokernel",
            expected_end=gw.parse_expected_end("+1h"), path=window.path)
    if overrides:
        data = json.loads(window.path.read_text())
        data.update(overrides)
        window.path.write_text(json.dumps(data))


def _records(log) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


# ── window file ──────────────────────────────────────────────────────────────


def test_missing_file_is_not_parked(window):
    assert not window.path.exists()
    assert gw.read_window() is None
    assert not gw.is_parked("architect_critic")
    assert not gw.is_parked(8083)
    gw.refuse_if_parked("architect_critic", 8083)  # returns, no raise


@pytest.mark.parametrize("content", ["{not json", "[]", '{"holder": "martians"}',
                                     '{"holder": "autokernel", "parked_roles": "x"}', ""])
def test_garbled_file_is_not_parked_and_logged(window, content, caplog):
    window.path.parent.mkdir(parents=True)
    window.path.write_text(content)
    with caplog.at_level("WARNING", logger=gw.__name__):
        assert not gw.is_parked("architect_critic")
    assert "garbled" in caplog.text


def test_disabled_env_is_not_parked(window, monkeypatch):
    _park(window)
    monkeypatch.setenv(gw.PATH_ENV, "off")
    assert not gw.is_parked("architect_critic")


def test_parked_by_role_by_port_and_production_releases(window):
    _park(window)
    assert gw.is_parked("architect_critic")
    assert gw.is_parked(role="coder_escalation")
    assert gw.is_parked(8083)
    assert not gw.is_parked("frontdoor")
    assert not gw.is_parked(8074)
    info = gw.parked_info(role="architect_critic")
    assert info.holder == "autokernel" and info.matched_by == "role"
    assert 3000 <= info.retry_after_s <= 3600
    gw.restore(path=window.path)
    assert not gw.is_parked("architect_critic")
    assert not gw.is_parked(8083)


def test_released_holder_is_still_parked_with_short_retry(window):
    _park(window, holder="released")
    info = gw.parked_info(port=8083)
    assert info is not None and info.holder == "released"
    assert info.retry_after_s == gw.RETRY_RELEASED_S


def test_alias_on_a_parked_port_is_parked(window, monkeypatch):
    gw.park(roles=[], ports=[8083], expected_end=None, path=window.path)
    monkeypatch.setattr(gw, "_role_ports", lambda role: (8083,) if role == "some_alias" else ())
    assert gw.is_parked("some_alias")
    assert not gw.is_parked("frontdoor")


# ── preempt ──────────────────────────────────────────────────────────────────


def test_request_preempt_is_idempotent(window):
    _park(window)
    first = gw.request_preempt("architect_critic request r1", role="architect_critic",
                               request_id="r1")
    assert first["status"] == "requested"
    data = json.loads(window.path.read_text())
    stamp = data["preempt_requested_at"]
    assert stamp and data["preempt_reason"] == "architect_critic request r1"
    assert data["preempt_requested_by"]["request_id"] == "r1"
    assert data["preempt_requested_by"]["role"] == "architect_critic"
    second = gw.request_preempt("coder_escalation request r2", role="coder_escalation",
                                request_id="r2")
    assert second["status"] == "already_requested"
    data = json.loads(window.path.read_text())
    assert data["preempt_requested_at"] == stamp
    assert data["preempt_requested_by"]["request_id"] == "r1"
    # The rest of the window survives the rewrite.
    assert data["holder"] == "autokernel" and data["parked_ports"] == [8083]
    assert not list(window.path.parent.glob(".mi210.json.*.tmp"))


def test_request_preempt_never_writes_when_not_held(window):
    assert gw.request_preempt("x")["status"] == "not_held"  # no file
    assert not window.path.exists()
    _park(window, holder="released")
    before = window.path.read_text()
    assert gw.request_preempt("x")["status"] == "not_held"
    assert window.path.read_text() == before


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_park_status_restore(window, capsys, monkeypatch):
    from src.runtime import gpu_window_executor as gwe

    # restore = executor close; prove serving against a stub, never a live port/lock.
    monkeypatch.setattr(gwe, "live_ops", lambda: gwe.StackOps(
        stop=lambda c: True, reload=lambda c: True, get_json=lambda u, t: (0, None),
        post_json=lambda u, b, t: (0, None), pids_on_port=lambda p: [],
        proc_maps=lambda pid: [], proc_exe=lambda pid: "", device_held=lambda d: False))
    monkeypatch.setattr(gwe, "serving_proof", lambda ops, port: (True, "ok"))
    rc = gw.main(["park", "--roles", ",".join(PARKED_ROLES), "--ports", "8083",
                  "--holder", "autokernel", "--expected-end", "+45m"])
    assert rc == 0
    data = json.loads(window.path.read_text())
    assert data["holder"] == "autokernel"
    assert data["parked_roles"] == sorted(PARKED_ROLES)
    assert data["parked_ports"] == [8083]
    assert data["preempt_requested_at"] is None and data["since"] and data["expected_end"]
    capsys.readouterr()

    assert gw.main(["status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["active"] is True and status["parked_ports"] == [8083]
    assert gw.is_parked("ingest_long_context")

    assert gw.main(["restore"]) == 0
    data = json.loads(window.path.read_text())
    assert data["holder"] == "production" and data["parked_roles"] == []
    assert data["previous_holder"] == "autokernel"
    assert not gw.is_parked("ingest_long_context")


def test_cli_rejects_bad_input(window, capsys):
    assert gw.main(["park", "--roles", "architect_critic", "--expected-end", "soon"]) == 2
    assert not window.path.exists()


# ── refusal: fast, explicit, recorded ────────────────────────────────────────


def test_refusal_is_fast_explicit_and_recorded(window):
    _park(window)
    gw.read_window()  # warm cache, as on a live process
    t0 = time.perf_counter()
    with pytest.raises(RoleParkedError) as exc_info:
        gw.refuse_if_parked("architect_critic", request_id="req-1",
                            base_url="http://localhost:8083")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 100.0, elapsed_ms
    exc = exc_info.value
    assert exc.error_type == "role_parked" and exc.holder == "autokernel"
    assert exc.refusal["gate"] == "role_parked" and exc.refusal["preempt"] == "requested"
    assert not isinstance(exc, RuntimeError)  # no silent same-tier model fallback
    body = exc.to_dict()
    assert body["type"] == "role_parked" and body["error_code"] == 503
    assert body["retry_after_s"] == exc.retry_after_s
    # The real request started the drain.
    assert json.loads(window.path.read_text())["preempt_requested_at"]
    (record,) = _records(window.log)
    assert record["outcome"] == "refused" and record["dispatched"] is False
    assert record["refusal"]["gate"] == "role_parked"
    assert record["refusal"]["holder"] == "autokernel"
    assert record["role"] == "architect_critic"
    assert record["caller"]["request_id"] == "req-1"


def test_not_parked_check_is_cheap(window):
    _park(window)
    gw.read_window()
    t0 = time.perf_counter()
    for _ in range(1000):
        gw.refuse_if_parked("frontdoor", 8074)
    per_call_ms = (time.perf_counter() - t0)  # 1000 calls, seconds == ms per call
    assert per_call_ms < 1.0, per_call_ms


def test_sentinel_round_trip_and_chat_annotation(window):
    from src.api.models import ChatResponse
    from src.api.routes.chat_pipeline.stages import _annotate_error

    _park(window)
    with pytest.raises(RoleParkedError) as exc_info:
        gw.refuse_if_parked("coder_escalation", preempt=False, record=False)
    sentinel = f"[ERROR: {exc_info.value}]"  # what llm_call returns
    parsed = gw.parse_parked_sentinel(sentinel)
    assert parsed["gate"] == "role_parked" and parsed["holder"] == "autokernel"
    assert parsed["role"] == "coder_escalation" and parsed["port"] is None
    assert parsed["retry_after_s"] == exc_info.value.retry_after_s

    from src.graph.helpers import _backend_infra_sentinel

    assert _backend_infra_sentinel(sentinel) is not None  # /chat infra failure, no nudge
    response = _annotate_error(ChatResponse(answer=sentinel, turns=1, tokens_used=0,
                                            elapsed_seconds=0.0, mock_mode=False))
    assert response.error_code == 503 and "role_parked" in response.error_detail


# ── choke points ─────────────────────────────────────────────────────────────


def test_primitives_refuse_before_any_gate(window, monkeypatch):
    """/chat stages and /v1 go through ``_real_call``: refused before the
    contention gate, the role semaphore and the region locks."""
    from src.llm_primitives import LLMPrimitives

    _park(window)
    prims = LLMPrimitives(mock_mode=False, server_urls={
        "architect_critic": "http://localhost:8083", "frontdoor": "http://localhost:8070",
    })
    import src.scheduling.contention_gate as cg

    monkeypatch.setattr(cg, "get_gate", lambda: pytest.fail("contention gate reached"))
    t0 = time.perf_counter()
    with pytest.raises(RoleParkedError):
        prims._real_call("hello", "architect_critic")
    assert (time.perf_counter() - t0) * 1000.0 < 100.0
    out = prims.llm_call("hello", role="architect_critic")
    assert out.startswith("[ERROR: role_parked")
    assert any(r["outcome"] == "refused" and r["caller"].get("source") == "primitives"
               for r in _records(window.log))


def test_backend_layer_refuses_parked_port(window):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.model_server import InferenceRequest

    _park(window)
    backend = LlamaServerBackend(ServerConfig(base_url="http://localhost:8083"))
    role_config = SimpleNamespace(name="whatever_alias", model=None)
    t0 = time.perf_counter()
    with pytest.raises(RoleParkedError):
        backend.infer(role_config, InferenceRequest(role="whatever_alias", prompt="hi"))
    assert (time.perf_counter() - t0) * 1000.0 < 100.0
    with pytest.raises(RoleParkedError):
        backend.infer_stream_text(role_config, InferenceRequest(role="whatever_alias",
                                                                prompt="hi"))
    records = _records(window.log)
    assert len(records) == 2
    assert all(r["outcome"] == "refused" and r["refusal"]["gate"] == "role_parked"
               and r["dispatched"] is False and r["server"]["port"] == 8083
               for r in records)
    backend.close()


def test_v1_chat_completions_returns_503_role_parked(window, monkeypatch):
    from src.api import app
    from src.api.state import get_state, reset_state
    from src.features import reset_features
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    reset_features()
    reset_state()
    get_state()
    _park(window)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            state = get_state()
            if state.registry is None:
                state.registry = MagicMock()
            built: list = []
            monkeypatch.setattr(llm_primitives_module, "LLMPrimitives",
                                lambda **kw: built.append(kw) or MagicMock())
            t0 = time.perf_counter()
            resp = client.post("/v1/chat/completions", json={
                "model": "architect_critic",
                "messages": [{"role": "user", "content": "hello"}],
                "x_disable_repl": True,
            }, headers={"x-request-id": "v1-req"})
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        reset_features()
    assert resp.status_code == 503, resp.text
    assert built == []  # refused before any per-request work
    body = resp.json()
    assert body["type"] == "role_parked" and body["holder"] == "autokernel"
    assert int(resp.headers["Retry-After"]) == body["retry_after_s"]
    assert elapsed_ms < 1000.0  # in-process HTTP round trip; the check itself is µs
    data = json.loads(window.path.read_text())
    assert data["preempt_requested_by"]["request_id"] == "v1-req"


def test_passthrough_returns_503_role_parked_with_refusal_gate(window, monkeypatch):
    from src.api import app
    from src.api.routes import passthrough as pt

    monkeypatch.setattr(pt, "_server_urls", lambda: {"architect_critic": "http://127.0.0.1:8083"})
    monkeypatch.delenv(pt.PASSTHROUGH_ENV, raising=False)
    monkeypatch.delenv(pt.PASSTHROUGH_ROLES_ENV, raising=False)
    _park(window)
    with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as client:
        resp = client.post("/v1/passthrough/architect_critic/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}]},
                           headers={"x-request-id": "pt-req"})
        models = client.get("/v1/passthrough/architect_critic/models")
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["error"] == "role_parked" and body["refusal"]["gate"] == "role_parked"
    assert body["refusal"]["holder"] == "autokernel"
    assert resp.headers["Retry-After"] == str(body["retry_after_s"])
    assert models.status_code == 503 and models.json()["type"] == "role_parked"
    (record,) = _records(window.log)
    assert record["outcome"] == "refused" and record["dispatched"] is False
    assert record["refusal"]["gate"] == "role_parked"
    assert record["caller"]["source"] == "passthrough"
    assert json.loads(window.path.read_text())["preempt_requested_by"]["request_id"] == "pt-req"


def test_passthrough_refusals_carry_structured_gate():
    """UFH14-B6a: every passthrough refusal names its gate structurally."""
    from src.api.routes.passthrough import _Refused

    refused = _Refused(503, "admission_queue_full", "[ERROR: admission] full", retry_after=5)
    assert refused.refusal == {"gate": "admission_queue_full", "http_status": 503,
                               "retry_after_s": 5}
    assert json.loads(refused.response().body)["refusal"]["gate"] == "admission_queue_full"
