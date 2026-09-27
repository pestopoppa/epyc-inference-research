"""Unit tests for benchmark seeding_orchestrator helper module."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import httpx
import pytest


_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark"
sys.path.insert(0, str(_ROOT))
# resilient_http (watcher path) lives under scripts/autopilot; make it importable
# so the eval reconnect-backoff watcher-path test can patch resilient_post.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "autopilot"))
_SPEC = importlib.util.spec_from_file_location("seeding_orchestrator_test", _ROOT / "seeding_orchestrator.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seeding_orchestrator_test"] = _MOD
_SPEC.loader.exec_module(_MOD)


class _Resp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_normalize_tool_telemetry_shapes_fields_consistently():
    _MOD._normalize_tool_telemetry(None)

    data = {
        "tools_called": "web_search",
        "tool_timings": "bad-shape",
        "tools_used": "abc",
    }
    _MOD._normalize_tool_telemetry(data)
    assert data["tools_called"] == ["web_search"]
    assert data["tools_used"] == 1
    assert data["tool_timings"] == [
        {"tool_name": "web_search", "elapsed_ms": 0.0, "success": True}
    ]

    data2 = {"tools_called": [], "tool_timings": [{"tool_name": "x"}], "tools_used": 0}
    _MOD._normalize_tool_telemetry(data2)
    assert data2["tools_used"] == 1
    assert data2["tools_called"] == ["x"]


def test_busy_heavy_ports_and_read_slot_progress_cover_mixed_paths():
    with patch.object(_MOD, "HEAVY_PORTS", [8080, 8081, 8082]):
        with patch(
            "httpx.get",
            side_effect=[
                _Resp(200, [{"is_processing": True}]),
                _Resp(200, [{"is_processing": False}]),
                RuntimeError("down"),
            ],
        ):
            assert _MOD._busy_heavy_ports(timeout_s=1.0) == [8080]

    with patch("httpx.get", return_value=_Resp(503, {})):
        assert _MOD._read_slot_progress(8080) is None
    with patch("httpx.get", return_value=_Resp(200, [])):
        assert _MOD._read_slot_progress(8080) is None

    payload = [
        {"id_task": "x", "is_processing": False, "next_token": [{"n_decoded": "1", "n_remain": "2"}]},
        {"id_task": "7", "is_processing": True, "next_token": [{"n_decoded": "9", "n_remain": "3"}]},
    ]
    with patch("httpx.get", return_value=_Resp(200, payload)):
        prog = _MOD._read_slot_progress(8080)
    assert prog == {"is_processing": True, "task_id": 7, "n_decoded": 9, "n_remain": 3}


def test_read_slot_progress_coercion_failures_fallback_to_zero():
    payload = [
        {"id_task": "bad", "is_processing": False, "next_token": [{"n_decoded": "x", "n_remain": "y"}]},
    ]
    with patch("httpx.get", return_value=_Resp(200, payload)):
        prog = _MOD._read_slot_progress(8080)
    assert prog == {"is_processing": False, "task_id": 0, "n_decoded": 0, "n_remain": 0}


def test_erase_slots_and_force_erase_paths():
    _MOD._SLOT_ERASE_CAPABILITY.clear()
    slot_payload = [{"id": 1, "is_processing": True}]
    with (
        patch("httpx.get", side_effect=[_Resp(200, slot_payload), _Resp(200, {})]),
        patch("httpx.post", side_effect=[_Resp(404, {}), _Resp(200, {})]),
    ):
        _MOD._erase_slots(8080)
    assert _MOD._SLOT_ERASE_CAPABILITY[8080] == "GET_QUERY"

    _MOD._SLOT_ERASE_CAPABILITY.clear()
    with (
        # First GET returns slot list; second GET (erase attempt) also fails
        patch("httpx.get", side_effect=[_Resp(200, slot_payload), _Resp(404, {})]),
        patch("httpx.post", return_value=_Resp(404, {})),
    ):
        _MOD._erase_slots(8081)
    assert _MOD._SLOT_ERASE_CAPABILITY[8081] is False

    with (
        patch.object(_MOD, "_erase_slots"),
        patch.object(_MOD.time, "sleep"),
        patch("httpx.get", return_value=_Resp(200, [{"is_processing": False}])),
    ):
        assert _MOD._force_erase_and_verify(8080, max_attempts=2) is True

    with (
        patch.object(_MOD, "_erase_slots"),
        patch.object(_MOD.time, "sleep"),
        patch("httpx.get", return_value=_Resp(200, [{"is_processing": True}])),
    ):
        assert _MOD._force_erase_and_verify(8080, max_attempts=2) is False


def test_erase_slots_non200_cached_reset_and_exception_paths():
    # Non-200 /slots probe: early return.
    with patch("httpx.get", return_value=_Resp(503, {})):
        _MOD._erase_slots(8080)

    # Cached strategy failure resets cache to unknown for reprobe.
    _MOD._SLOT_ERASE_CAPABILITY.clear()
    _MOD._SLOT_ERASE_CAPABILITY[8080] = "POST_QUERY"
    with (
        patch("httpx.get", return_value=_Resp(200, [{"id": 1, "is_processing": True}])),
        patch("httpx.post", return_value=_Resp(500, {})),
    ):
        _MOD._erase_slots(8080)
    assert _MOD._SLOT_ERASE_CAPABILITY[8080] is None

    # Explicit unsupported cache disables attempts for this port.
    _MOD._SLOT_ERASE_CAPABILITY[8081] = False
    with patch("httpx.get", return_value=_Resp(200, [{"id": 1, "is_processing": True}])):
        _MOD._erase_slots(8081)

    # Outer probe exception is swallowed/logged.
    with patch("httpx.get", side_effect=RuntimeError("probe failed")):
        _MOD._erase_slots(8082)


def test_recover_heavy_ports_if_stuck_handles_disabled_fail_and_success(tmp_path: Path):
    assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", []) is True

    os.environ["SEEDING_ENABLE_TARGETED_RELOAD"] = "0"
    try:
        assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8080]) is False
    finally:
        os.environ.pop("SEEDING_ENABLE_TARGETED_RELOAD", None)

    os.environ["SEEDING_ENABLE_TARGETED_RELOAD"] = "1"
    try:
        assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [9999]) is False

        with patch("subprocess.run", return_value=SimpleNamespace(returncode=2, stderr="reload failed")) as run:
            assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8070]) is False
            assert "frontdoor" in run.call_args.args[0]

        with (
            patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")) as run,
            patch.object(_MOD, "_wait_for_heavy_models_idle"),
            patch.object(_MOD, "_busy_heavy_ports", return_value=[]),
        ):
            assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8070, 8083]) is True
            cmd = run.call_args.args[0]
            assert "frontdoor" in cmd
            # 2026-09-27 ARCHITECT SWAP: :8083 is the architect_critic process.
            assert "architect_critic" in cmd

        with (
            patch("subprocess.run", side_effect=RuntimeError("reload boom")),
            patch.object(_MOD, "_wait_for_heavy_models_idle"),
            patch.object(_MOD, "_busy_heavy_ports", return_value=[]),
        ):
            assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8070]) is False

        with (
            patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="")),
            patch.object(_MOD, "_wait_for_heavy_models_idle"),
            patch.object(_MOD, "_busy_heavy_ports", return_value=[8083]),
        ):
            assert _MOD._recover_heavy_ports_if_stuck("http://localhost:8000", [8070]) is False
    finally:
        os.environ.pop("SEEDING_ENABLE_TARGETED_RELOAD", None)


def test_call_orchestrator_forced_normalizes_tool_data_and_handles_errors():
    client = Mock()
    client.post.return_value = _Resp(
        200,
        {
            "answer": "ok",
            "error_code": 503,
            "error_detail": "backend down",
            "tools_called": "web_search",
            "tool_timings": "bad",
            "tools_used": "x",
        },
    )
    data = _MOD.call_orchestrator_forced(
        prompt="hello",
        force_role="worker",
        client=client,
    )
    assert data["error"] == "backend down"
    assert data["tools_used"] == 1
    assert data["tools_called"] == ["web_search"]
    assert data["tool_timings"][0]["tool_name"] == "web_search"

    with patch("httpx.post", side_effect=RuntimeError("network down")):
        err = _MOD.call_orchestrator_forced(prompt="hello", force_role="worker", client=None)
    assert "network down" in err["error"]


def test_surface_inband_error_sets_error_and_preserves_structured():
    # HTTP-200 in-band banner with error=None -> surfaced into data["error"].
    banner = "[ERROR: Backend unavailable (circuit open): http://localhost:8082]"
    data = {"answer": banner}
    _MOD._surface_inband_error(data)
    assert data["error"] == banner
    assert data["failure_reason"] == "inband_error"
    # answer left untouched (still re-scorable offline)
    assert data["answer"] == banner

    # An already-present structured error wins (idempotent, no clobber).
    data2 = {"answer": banner, "error": "HTTP 502"}
    _MOD._surface_inband_error(data2)
    assert data2["error"] == "HTTP 502"
    assert "failure_reason" not in data2

    # A normal answer is untouched.
    data3 = {"answer": "The final answer is 42."}
    _MOD._surface_inband_error(data3)
    assert data3.get("error") is None


def test_call_orchestrator_forced_surfaces_inband_error_answer():
    # 2026-07-21 EV-11c: circuit breaker returns an HTTP-200 body whose answer
    # IS an error banner with error=None. Without surfacing, seeding scores it
    # as a WRONG answer (0.0 MemRL reward). Assert it is surfaced to error.
    client = Mock()
    banner = "[ERROR: Backend unavailable (circuit open): http://localhost:8082]"
    client.post.return_value = _Resp(200, {"answer": banner})
    data = _MOD.call_orchestrator_forced(prompt="q", force_role="worker_math", client=client)
    assert data["answer"] == banner
    assert data["error"] == banner
    assert data["failure_reason"] == "inband_error"


def test_call_orchestrator_forced_preserves_structured_http_error_body():
    client = Mock()
    client.post.return_value = _Resp(
        502,
        {
            "answer": "[ERROR: Direct LLM call failed after retry: model returned no answer]",
            "error_code": 502,
            "error_detail": "Direct LLM call failed after retry: model returned no answer",
            "tools_called": "web_search",
        },
    )

    data = _MOD.call_orchestrator_forced(
        prompt="hello",
        force_role="worker",
        client=client,
    )

    assert data["answer"].startswith("[ERROR: Direct LLM call failed")
    assert data["error"] == "Direct LLM call failed after retry: model returned no answer"
    assert data["error_code"] == 502
    assert data["tools_called"] == ["web_search"]


def test_call_orchestrator_forced_preserves_server_failure_provenance():
    provenance = {
        "schema": "epyc.failure_provenance.v1",
        "class": "admission_timeout",
        "code": "race_lost",
        "phase": "admission",
        "generation_started": False,
        "tokens_generated": 0,
        "partial": False,
        "degraded": False,
        "role": "frontdoor",
        "workload_class": "eval_batch",
        "max_queue_wait_ms": 90_000,
    }
    client = Mock()
    client.post.return_value = _Resp(
        503,
        {
            "error": "contention_denied",
            "detail": "placement timeout",
            "retry_after_s": 5,
            "error_code": 503,
            "error_detail": "placement timeout",
            "failure_provenance": provenance,
        },
    )

    data = _MOD.call_orchestrator_forced(
        prompt="q",
        force_role="frontdoor",
        client=client,
        workload_class="eval_batch",
    )

    assert data["error"] == "contention_denied"
    assert data["detail"] == "placement timeout"
    assert data["retry_after_s"] == 5
    assert data["error_code"] == 503
    assert data["error_detail"] == "placement timeout"
    assert data["failure_provenance"] == provenance


def test_call_orchestrator_forced_includes_optional_payload_fields():
    client = Mock()
    client.post.return_value = _Resp(200, {"answer": "ok"})

    _MOD.call_orchestrator_forced(
        prompt="hello",
        force_role="worker",
        client=client,
        image_path="/tmp/img.png",
        cache_prompt=False,
        allow_delegation=True,
        session_id="sess-1",
        scoring_method="score-x",
        stop_sequences=["</answer>"],
        request_priority="background",
        workload_class="eval_batch",
        batch_id="evaltower-T1-123-43q",
        batch_placement_mode="mixed_role_split",
        tools=[
            {
                "type": "function",
                "function": {"name": "web_search", "parameters": {"type": "object"}},
            }
        ],
        tool_choice={"type": "function", "function": {"name": "web_search"}},
        max_tokens=1024,
        n_probs=7,
        output_schema={"type": "object", "required": ["decision"]},
        prompt_root="/tmp/gepa-root",
    )

    payload = client.post.call_args.kwargs["json"]
    assert payload["image_path"] == "/tmp/img.png"
    assert payload["cache_prompt"] is False
    assert payload["allow_delegation"] is True
    assert payload["session_id"] == "sess-1"
    assert payload["scoring_method"] == "score-x"
    assert payload["stop_sequences"] == ["</answer>"]
    assert payload["request_priority"] == "background"
    assert payload["workload_class"] == "eval_batch"
    assert payload["batch_id"] == "evaltower-T1-123-43q"
    assert payload["batch_placement_mode"] == "mixed_role_split"
    assert payload["tools"][0]["function"]["name"] == "web_search"
    assert payload["tool_choice"]["function"]["name"] == "web_search"
    assert payload["max_tokens"] == 1024
    assert payload["n_probs"] == 7
    assert payload["output_schema"] == {"type": "object", "required": ["decision"]}
    assert payload["x_orchestrator_prompt_root"] == "/tmp/gepa-root"


def test_call_orchestrator_forced_omits_native_tool_payload_by_default():
    client = Mock()
    client.post.return_value = _Resp(200, {"answer": "ok"})

    _MOD.call_orchestrator_forced(prompt="hello", force_role="worker", client=client)

    payload = client.post.call_args.kwargs["json"]
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "output_schema" not in payload
    assert "x_orchestrator_prompt_root" not in payload


# ── Eval reconnect backoff (REL-2) ───────────────────────────────────


def test_eval_reconnect_recovers_after_transient_connection_failure():
    # Mid-run API reload: first two POSTs are refused (ConnectError), third
    # succeeds. Eval traffic must survive it instead of burning the question.
    client = Mock()
    client.post.side_effect = [
        httpx.ConnectError("connection refused"),
        httpx.ConnectError("connection refused"),
        _Resp(200, {"answer": "recovered"}),
    ]
    with patch.object(_MOD.time, "sleep") as sleep_mock:
        data = _MOD.call_orchestrator_forced(
            prompt="q",
            force_role="worker_math",
            client=client,
            workload_class="eval_batch",
        )
    assert data["answer"] == "recovered"
    assert data.get("error") is None
    assert client.post.call_count == 3
    # Exponential backoff: 1s then 2s before the successful third attempt.
    assert sleep_mock.call_args_list == [call(1.0), call(2.0)]


def test_eval_reconnect_persistent_failure_honest_error_row():
    # Persistent unreachability must still become an EXCLUDED REL-1 error row,
    # tagged with the api_unreachable_after_backoff reason, after the bounded
    # backoff budget is spent.
    client = Mock()
    client.post.side_effect = httpx.ConnectError("connection refused")
    with patch.dict(os.environ, {"AUTOPILOT_EVAL_RECONNECT_MAX_S": "3"}):
        with patch.object(_MOD.time, "sleep") as sleep_mock:
            data = _MOD.call_orchestrator_forced(
                prompt="q",
                force_role="worker_math",
                client=client,
                workload_class="eval_batch",
            )
    assert data["answer"] == ""
    assert data["failure_reason"] == "api_unreachable_after_backoff"
    assert data["error"].startswith("api_unreachable_after_backoff:")
    # Budget=3s → sleeps 1s + 2s (=3), the next (4s) would overrun → honest row.
    assert sleep_mock.call_args_list == [call(1.0), call(2.0)]
    assert client.post.call_count == 3


def test_eval_reconnect_does_not_retry_timeouts():
    # 04411baf timeout semantics preserved: a ReadTimeout is NOT connection-level
    # and must fall straight through to a terminal error with zero backoff.
    client = Mock()
    client.post.side_effect = httpx.ReadTimeout("read timed out")
    with patch.object(_MOD.time, "sleep") as sleep_mock:
        data = _MOD.call_orchestrator_forced(
            prompt="q",
            force_role="worker_math",
            client=client,
            workload_class="eval_batch",
        )
    assert data["answer"] == ""
    assert data.get("failure_reason") != "api_unreachable_after_backoff"
    assert "api_unreachable_after_backoff" not in str(data.get("error"))
    assert data["failure_provenance"] == {
        "schema": "epyc.failure_provenance.v1",
        "class": "client_transport_timeout",
        "code": "read_timeout",
        "phase": "client_transport",
        "exception_class": "httpx.ReadTimeout",
        "exception_reason": "read_timeout",
        "role": "worker_math",
        "workload_class": "eval_batch",
        "max_queue_wait_ms": 300_000,
    }
    assert "generation_started" not in data["failure_provenance"]
    assert "tokens_generated" not in data["failure_provenance"]
    assert "partial" not in data["failure_provenance"]
    assert "degraded" not in data["failure_provenance"]
    sleep_mock.assert_not_called()
    assert client.post.call_count == 1


def test_non_eval_connection_error_preserves_legacy_terminal_semantics():
    # The 14 non-eval callers keep the EXACT legacy CONTROL FLOW: a ConnectError
    # is a terminal error dict with no backoff/retry (reconnect is eval-scoped).
    #
    # This test previously also asserted `"failure_reason" not in data`. That
    # assertion was retired on 2026-08-12: the non-eval branch is the branch the
    # SEEDING path takes (it sets no workload_class), and seeding is where an
    # unreachable/refusing endpoint was classified from the exception's MESSAGE
    # TEXT and scored as a WRONG answer. `failure_reason` is the structural
    # disposition signal that fixes it. It is purely ADDITIVE — the
    # answer/error/retry shape asserted below is unchanged — so the test now
    # pins what it actually protects instead of the absence of a key.
    client = Mock()
    client.post.side_effect = httpx.ConnectError("connection refused")
    with patch.object(_MOD.time, "sleep") as sleep_mock:
        data = _MOD.call_orchestrator_forced(
            prompt="q",
            force_role="worker",
            client=client,
        )  # no workload_class → non-eval path
    assert data["answer"] == ""
    assert "error" in data
    # Terminal, not retried, and no timeout provenance invented for a connect
    # failure (a ConnectError is not evidence the server started generating).
    assert "failure_provenance" not in data
    # Additive structural disposition signal (see comment above).
    assert data["failure_reason"] == "connect_error"
    sleep_mock.assert_not_called()
    assert client.post.call_count == 1


def test_non_eval_timeout_adds_only_typed_transport_provenance():
    client = Mock()
    client.post.side_effect = httpx.ReadTimeout("read timed out")

    data = _MOD.call_orchestrator_forced(
        prompt="q",
        force_role="worker",
        client=client,
    )

    assert data["answer"] == ""
    assert data["error"] == "read timed out"
    assert data["failure_provenance"]["class"] == "client_transport_timeout"
    assert data["failure_provenance"]["code"] == "read_timeout"
    assert data["failure_provenance"]["workload_class"] == ""
    assert set(data["failure_provenance"]).isdisjoint(
        {"generation_started", "tokens_generated", "partial", "degraded"}
    )


@pytest.mark.parametrize(
    ("exc", "reason", "exception_class"),
    [
        (httpx.ConnectTimeout("connect"), "connect_timeout", "httpx.ConnectTimeout"),
        (httpx.ReadTimeout("read"), "read_timeout", "httpx.ReadTimeout"),
        (httpx.WriteTimeout("write"), "write_timeout", "httpx.WriteTimeout"),
        (httpx.PoolTimeout("pool"), "pool_timeout", "httpx.PoolTimeout"),
        (socket.timeout("socket"), "socket_timeout", "builtins.TimeoutError"),
    ],
)
def test_transport_timeout_preserves_typed_exception_class_and_reason(
    exc, reason, exception_class
):
    client = Mock()
    client.post.side_effect = exc
    data = _MOD.call_orchestrator_forced(
        prompt="q", force_role="worker", client=client, workload_class="eval_batch"
    )
    provenance = data["failure_provenance"]
    assert provenance["class"] == "client_transport_timeout"
    assert provenance["code"] == reason
    assert provenance["exception_reason"] == reason
    assert provenance["exception_class"] == exception_class


def _reconnect_meta(**overrides):
    meta = {
        "clean": False,
        "exogenous_recovered": False,
        "exogenous_unrecovered": False,
        "external_restart": False,
        "real_failure": False,
        "retry_count": 0,
        "wait_s": 0.0,
        "marker_changes": {},
        "reason": "",
        "detail": "",
    }
    meta.update(overrides)
    return meta


def test_eval_reconnect_watcher_path_backs_off_and_recovers():
    # Watcher path: resilient_post can report a connection-level real_failure
    # when the watcher itself couldn't read markers during the reload. The
    # reconnect-backoff wraps it: back off, then retry until it recovers.
    import resilient_http

    calls = [
        (
            {"answer": "", "error": "ConnectError: connection refused"},
            _reconnect_meta(real_failure=True, reason="connect_error",
                            detail="ConnectError: connection refused"),
        ),
        ({"answer": "recovered"}, _reconnect_meta(clean=True)),
    ]
    rp_mock = Mock(side_effect=calls)
    with patch.object(resilient_http, "resilient_post", rp_mock):
        with patch.object(_MOD.time, "sleep") as sleep_mock:
            data = _MOD.call_orchestrator_forced(
                prompt="q",
                force_role="worker_math",
                workload_class="eval_batch",
                watcher=object(),  # any non-None watcher engages the watcher path
            )
    assert data["answer"] == "recovered"
    assert rp_mock.call_count == 2
    assert sleep_mock.call_args_list == [call(1.0)]


def test_watcher_path_preserves_server_failure_provenance_unchanged():
    import resilient_http

    provenance = {
        "schema": "epyc.failure_provenance.v1",
        "class": "admission_timeout",
        "code": "race_lost",
        "phase": "admission",
        "generation_started": False,
        "tokens_generated": 0,
        "partial": False,
        "degraded": False,
        "role": "frontdoor",
        "workload_class": "eval_batch",
        "max_queue_wait_ms": 90_000,
    }
    response = {
        "error": "contention_denied",
        "error_code": 503,
        "error_detail": "placement timeout",
        "failure_provenance": provenance,
    }
    with patch.object(
        resilient_http,
        "resilient_post",
        return_value=(response, _reconnect_meta(clean=True)),
    ):
        data = _MOD.call_orchestrator_forced(
            prompt="q",
            force_role="frontdoor",
            workload_class="eval_batch",
            watcher=object(),
        )

    assert data["failure_provenance"] == provenance
    assert data["error"] == "contention_denied"


def test_watcher_client_timeout_omits_unobserved_server_state():
    import resilient_http

    with patch.object(
        resilient_http,
        "resilient_post",
        return_value=(
            {"answer": "", "error": "ReadTimeout: timed out"},
            _reconnect_meta(
                real_failure=True,
                reason="read_timeout",
                detail="ReadTimeout: timed out",
            ),
        ),
    ):
        data = _MOD.call_orchestrator_forced(
            prompt="q",
            force_role="frontdoor",
            workload_class="eval_batch",
            watcher=object(),
        )

    provenance = data["failure_provenance"]
    assert provenance["class"] == "client_transport_timeout"
    assert provenance["code"] == "read_timeout"
    assert set(provenance).isdisjoint(
        {"generation_started", "tokens_generated", "partial", "degraded"}
    )


class _Future:
    def __init__(self, results):
        self._results = list(results)

    def result(self, timeout=None):
        if not self._results:
            return {"answer": "done"}
        nxt = self._results.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _Executor:
    def __init__(self, future):
        self._future = future

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn):
        return self._future


def test_call_orchestrator_with_slot_poll_success_after_timeout():
    fut = _Future([concurrent.futures.TimeoutError(), {"answer": "ok"}])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._read_slot_progress", return_value={"n_decoded": 12, "n_remain": 3, "task_id": 9}),
        # Provide enough perf_counter values for all timing calls in the function
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=60,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert resp["answer"] == "ok"
    assert progress["max_decoded"] == 12
    assert progress["task_id"] == 9
    assert elapsed >= 0.0


def test_call_orchestrator_with_slot_poll_timeout_erase_branch():
    fut = _Future(
        [
            concurrent.futures.TimeoutError(),
            concurrent.futures.TimeoutError(),
        ]
    )
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._force_erase_and_verify"),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=10,  # timeout-erase path triggers immediately (10-15 < 0)
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert "timeout after slot erase" in resp["error"]
    assert resp["failure_provenance"]["class"] == "slot_erase_timeout"
    assert resp["failure_provenance"]["code"] == "timeout_after_slot_erase"
    assert resp["failure_provenance"]["elapsed_ms"] == 2000
    assert elapsed >= 0.0
    assert progress["max_decoded"] == 0


def test_call_orchestrator_with_slot_poll_timeout_erase_then_success():
    fut = _Future([concurrent.futures.TimeoutError(), {"answer": "ok-after-erase"}])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._force_erase_and_verify"),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 2.0]),
    ):
        resp, elapsed, _progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=10,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert resp["answer"] == "ok-after-erase"
    assert elapsed >= 0.0


def test_call_orchestrator_with_slot_poll_slot_stall_watchdog(monkeypatch):
    monkeypatch.setenv("SEEDING_SLOT_STALL_WATCHDOG_S", "2")
    fut = _Future([
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
    ])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._force_erase_and_verify") as erase,
        patch(
            "seeding_orchestrator_test._read_slot_progress",
            return_value={"is_processing": True, "n_decoded": 10, "n_remain": 5, "task_id": 3},
        ),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 4.0, 5.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=300,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    erase.assert_called_once()
    assert resp["failure_reason"] == "slot_stalled_no_progress"
    assert "slot stalled" in resp["error"]
    assert progress["max_decoded"] == 10
    assert elapsed >= 0.0


def test_call_orchestrator_with_slot_poll_idle_orphan_watchdog(monkeypatch):
    monkeypatch.setenv("SEEDING_SLOT_IDLE_ORPHAN_WATCHDOG_S", "2")
    fut = _Future([
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
    ])
    slot_progress = [
        {"is_processing": True, "n_decoded": 4, "n_remain": 10, "task_id": 8},
        {"is_processing": False, "n_decoded": 4, "n_remain": 10, "task_id": 8},
        {"is_processing": False, "n_decoded": 4, "n_remain": 10, "task_id": 8},
    ]
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._read_slot_progress", side_effect=slot_progress),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 2.0, 5.0, 6.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=300,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert resp["failure_reason"] == "slot_idle_orphan"
    assert "slot idle while request pending" in resp["error"]
    assert progress["task_id"] == 8
    assert elapsed >= 0.0


def test_call_orchestrator_with_slot_poll_idle_slot_allows_completion_grace(monkeypatch):
    monkeypatch.setenv("SEEDING_SLOT_IDLE_ORPHAN_WATCHDOG_S", "2")
    fut = _Future([
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        concurrent.futures.TimeoutError(),
        {"answer": "ok-after-idle"},
    ])
    slot_progress = [
        {"is_processing": True, "n_decoded": 4, "n_remain": 10, "task_id": 8},
        {"is_processing": False, "n_decoded": 4, "n_remain": 10, "task_id": 8},
        {"is_processing": False, "n_decoded": 4, "n_remain": 10, "task_id": 8},
    ]
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._read_slot_progress", side_effect=slot_progress),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 2.0, 5.0, 6.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=300,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert resp["answer"] == "ok-after-idle"
    assert "failure_reason" not in resp
    assert progress["task_id"] == 8
    assert elapsed >= 0.0


def test_call_orchestrator_with_slot_poll_port_zero_heartbeat_path():
    fut = _Future([concurrent.futures.TimeoutError(), {"answer": "ok"}])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 10.0, 130.0, 131.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=300,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=0,
        )
    assert resp["answer"] == "ok"
    assert elapsed >= 0.0
    assert progress["max_decoded"] == 0


def test_call_orchestrator_with_slot_poll_future_exception_path():
    fut = _Future([RuntimeError("worker crash")])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 0.2]),
    ):
        resp, elapsed, _progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=30,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert "worker crash" in resp["error"]
    assert elapsed >= 0.0


def test_erase_slots_covers_skip_invalid_strategy_and_transient_exception_paths():
    # all_slots=False skips idle slots.
    with (
        patch("httpx.get", return_value=_Resp(200, [{"id": 1, "is_processing": False}])),
        patch("httpx.post") as post,
    ):
        _MOD._erase_slots(8080, all_slots=False)
    post.assert_not_called()

    # Unknown cached strategy returns None from _erase_slot_with_strategy.
    _MOD._SLOT_ERASE_CAPABILITY.clear()
    _MOD._SLOT_ERASE_CAPABILITY[8081] = "INVALID"
    with patch("httpx.get", return_value=_Resp(200, [{"id": 2, "is_processing": True}])):
        _MOD._erase_slots(8081)
    assert _MOD._SLOT_ERASE_CAPABILITY[8081] is None

    # Transient strategy exception is swallowed and probing continues.
    _MOD._SLOT_ERASE_CAPABILITY.clear()
    with (
        patch("httpx.get", side_effect=[_Resp(200, [{"id": 3, "is_processing": True}]), _Resp(404, {})]),
        patch("httpx.post", side_effect=[RuntimeError("transient"), _Resp(404, {})]),
    ):
        _MOD._erase_slots(8082)


def test_force_erase_and_verify_short_circuit_and_verify_exception():
    assert _MOD._force_erase_and_verify(0) is True

    with (
        patch.object(_MOD, "_erase_slots"),
        patch.object(_MOD.time, "sleep"),
        patch("httpx.get", side_effect=RuntimeError("probe failed")),
    ):
        assert _MOD._force_erase_and_verify(8080, max_attempts=1) is False


def test_busy_heavy_ports_non_200_and_read_slot_progress_exception_paths():
    with patch.object(_MOD, "HEAVY_PORTS", [8080]):
        with patch("httpx.get", return_value=_Resp(503, {})):
            assert _MOD._busy_heavy_ports() == []

    with patch("httpx.get", side_effect=RuntimeError("slots down")):
        assert _MOD._read_slot_progress(8080) is None


def test_call_orchestrator_with_slot_poll_progress_none_continues():
    fut = _Future([concurrent.futures.TimeoutError(), {"answer": "ok"}])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch("seeding_orchestrator_test._read_slot_progress", return_value=None),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 2.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=300,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert resp["answer"] == "ok"
    assert elapsed >= 0.0
    assert progress["source"] == ""


def test_call_orchestrator_with_slot_poll_logs_progress_and_heartbeat():
    fut = _Future([concurrent.futures.TimeoutError(), {"answer": "ok"}])
    with (
        patch("seeding_orchestrator_test.concurrent.futures.ThreadPoolExecutor", return_value=_Executor(fut)),
        patch(
            "seeding_orchestrator_test._read_slot_progress",
            return_value={"n_decoded": 256, "n_remain": 3, "task_id": 7},
        ),
        patch.object(_MOD.time, "perf_counter", side_effect=[0.0, 1.0, 2.0, 130.0, 131.0]),
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=300,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=8080,
        )
    assert resp["answer"] == "ok"
    assert elapsed >= 0.0
    assert progress["max_decoded"] == 256
    assert progress["source"] == "slots_poll"


def test_call_orchestrator_with_slot_poll_real_executor_invokes_call_path():
    with patch(
        "seeding_orchestrator_test.call_orchestrator_forced",
        return_value={"answer": "ok-direct"},
    ):
        resp, elapsed, progress = _MOD._call_orchestrator_with_slot_poll(
            prompt="p",
            force_role="worker",
            force_mode="direct",
            url="http://localhost:8000",
            timeout=60,
            image_path="",
            cache_prompt=None,
            client=None,
            allow_delegation=None,
            log_label="test",
            poll_port=0,
        )
    assert resp["answer"] == "ok-direct"
    assert elapsed >= 0.0
    assert progress["max_decoded"] == 0
