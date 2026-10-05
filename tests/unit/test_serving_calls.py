"""Per-call serving record (``src/backends/serving_calls.py``).

One JSONL record per backend call, carrying llama.cpp's timings, the caller the
primitives layer staged, the queue wait before dispatch, an outcome for every way a
call can end, and the provenance a ClaimTuple projection needs. No network.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.backends import serving_calls as sc
from src.model_server import InferenceRequest, InferenceResult
from src.runtime.git_head import resolve_git_head

SERVER_TIMINGS = {
    "cache_n": 1200,
    "prompt_n": 340,
    "prompt_ms": 812.5,
    "prompt_per_second": 418.4,
    "predicted_n": 64,
    "predicted_ms": 1500.0,
    "predicted_per_second": 42.6,
    "draft_n": 40,
    "draft_n_accepted": 30,
}


@pytest.fixture
def log_file(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "serving_calls" / "serving_calls.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(path))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    sc._SIDECAR_CACHE.clear()
    sc.clear_staged()
    yield path
    sc.clear_staged()


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _role_config(name: str = "frontdoor"):
    return types.SimpleNamespace(name=name, model=types.SimpleNamespace(name="Qwen-test"))


def _ok_result(**kw) -> InferenceResult:
    base = dict(role="frontdoor", output="hi", tokens_generated=64, generation_speed=42.6,
                elapsed_time=2.4, success=True, prompt_eval_ms=812.5, generation_ms=1500.0)
    base.update(kw)
    return InferenceResult(**base)


class _Backend:
    """Minimal stand-in for LlamaServerBackend: same decorator, scripted behaviour."""

    def __init__(self, behaviour, base_url: str = "http://localhost:8070"):
        self.config = types.SimpleNamespace(base_url=base_url)
        self._behaviour = behaviour
        self.calls = 0

    @sc.recorded_call("infer")
    def infer(self, role_config, request):
        self.calls += 1
        return self._behaviour(self, role_config, request, None)

    @sc.recorded_call("infer_stream_text")
    def infer_stream_text(self, role_config, request, on_chunk=None):
        self.calls += 1
        return self._behaviour(self, role_config, request, on_chunk)


def _server_ok(self, role_config, request, on_chunk):
    sc.note_timings(SERVER_TIMINGS, endpoint="/completion", stream=on_chunk is not None)
    if on_chunk is not None:
        on_chunk("hi")
    return _ok_result()


def test_one_record_per_call_with_timings_caller_and_wait(log_file):
    backend = _Backend(_server_ok)
    sc.stage_caller(role="frontdoor", request_id="api-1:ab", task_id="chat-1",
                    session_id="sess-9", workload_class="interactive")
    backend.infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x", n_tokens=64))

    [rec] = _records(log_file)
    assert rec["schema"] == sc.SCHEMA
    assert rec["outcome"] == "ok" and rec["dispatched"] is True
    assert rec["role"] == "frontdoor" and rec["model_registry"] == "Qwen-test"
    assert rec["server"]["port"] == 8070
    assert rec["timings"] == SERVER_TIMINGS and rec["timings_source"] == "server"
    assert rec["caller"]["task_id"] == "chat-1" and rec["caller"]["session_id"] == "sess-9"
    assert rec["caller"]["source"] == "primitives"
    assert rec["queue"]["pre_dispatch_wait_ms"] >= 0.0
    assert rec["ts_start"].endswith("+00:00") and rec["ts_end"] >= rec["ts_start"]
    assert rec["notes"]["endpoint"] == "/completion"
    prov = rec["provenance"]
    assert prov["run_id"] and prov["pid"] and prov["started_at"]
    assert "orch_commit" in prov


@pytest.mark.parametrize("client_class", ["opencode", None])
def test_explicit_client_class_reaches_native_record_without_inference(
    log_file, client_class,
):
    from src.api.models.openai import OpenAIChatRequest
    from src.api.routes.openai_compat import _request_keys
    from src.llm_primitives import LLMPrimitives

    request = OpenAIChatRequest(
        model="orchestrator",
        messages=[{"role": "user", "content": "metadata-only fixture"}],
        x_session_id="ses_native_record",
        x_client_class=client_class,
    )
    primitives = LLMPrimitives(mock_mode=True)
    request_keys = _request_keys(request)
    primitives.set_request_trace_keys(request_keys)
    primitives._stage_serving_caller(
        "frontdoor", request, "http://localhost:8070", 8070
    )

    _Backend(_server_ok).infer(
        _role_config(), InferenceRequest(role="frontdoor", prompt="fixture")
    )

    [record] = _records(log_file)
    assert record["caller"]["trace_keys"]["x_session_id"] == "ses_native_record"
    if client_class is None:
        assert "client_class" not in record["caller"]
        assert "client_class_provenance" not in record["caller"]
    else:
        assert record["caller"]["client_class"] == client_class
        assert record["caller"]["client_class_provenance"] == "caller_supplied:x_client_class"
        assert record["caller"]["trace_keys"]["x_client_class"] == client_class
    assert record["caller"]["workload_class"] == "interactive"


def test_every_line_is_self_hashed(log_file):
    _Backend(_server_ok).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    [rec] = _records(log_file)
    assert rec["record_sha256"] == sc.record_digest(rec)
    tampered = dict(rec, wall_ms=rec["wall_ms"] + 1)
    assert sc.record_digest(tampered) != rec["record_sha256"]


def test_stage_is_consumed_once(log_file):
    backend = _Backend(_server_ok)
    sc.stage_caller(role="frontdoor", task_id="chat-1")
    backend.infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    backend.infer(_role_config(), InferenceRequest(role="frontdoor", prompt="y"))
    first, second = _records(log_file)
    assert first["caller"]["task_id"] == "chat-1"
    assert second["caller"] == {"source": "unstaged"}
    assert "pre_dispatch_wait_ms" not in second["queue"]


def test_nested_entry_points_write_one_record(log_file):
    """infer_stream_text -> self.infer (the chat-payload backstop) is ONE call."""

    def _delegates(self, role_config, request, on_chunk):
        if on_chunk is not None:
            return self.infer(role_config, request)
        return _server_ok(self, role_config, request, None)

    backend = _Backend(_delegates)
    backend.infer_stream_text(_role_config(), InferenceRequest(role="frontdoor", prompt="x"),
                              on_chunk=lambda c: None)
    assert backend.calls == 2
    [rec] = _records(log_file)
    assert rec["method"] == "infer_stream_text"


def test_early_stop_and_first_chunk_recorded(log_file):
    def _streams(self, role_config, request, on_chunk):
        try:
            on_chunk("FINAL(")
        except StopIteration:
            pass
        return _ok_result(completion_reason="stop")

    def _stopper(content):
        raise StopIteration

    _Backend(_streams).infer_stream_text(_role_config(), InferenceRequest(role="frontdoor", prompt="x"),
                                         on_chunk=_stopper)
    [rec] = _records(log_file)
    assert rec["outcome"] == "early_stop"
    assert rec["notes"]["first_chunk_ms"] >= 0.0
    assert rec["timings"] is None and rec["timings_source"] == "absent"


def test_positional_on_chunk_is_observed(log_file):
    def _streams(self, role_config, request, on_chunk):
        on_chunk("a")
        return _ok_result()

    seen: list[str] = []
    _Backend(_streams).infer_stream_text(_role_config(), InferenceRequest(role="frontdoor", prompt="x"),
                                         seen.append)
    assert seen == ["a"]
    [rec] = _records(log_file)
    assert "first_chunk_ms" in rec["notes"]


@pytest.mark.parametrize(
    "result_kw, expected",
    [
        (dict(success=False, completion_reason="read_timeout", error_message="Request timed out"), "timeout"),
        (dict(success=False, failure_reason="server_error", completion_reason="server_error"), "failed"),
        (dict(success=False, context_overflow={"n_ctx": 4096}), "context_overflow"),
        (dict(success=False, error_message="request cancelled by client"), "cancelled"),
    ],
)
def test_failed_results_are_classified(log_file, result_kw, expected):
    _Backend(lambda *a: _ok_result(**result_kw)).infer(
        _role_config(), InferenceRequest(role="frontdoor", prompt="x")
    )
    [rec] = _records(log_file)
    assert rec["outcome"] == expected
    assert rec["result"]["success"] is False


def test_exception_is_recorded_and_reraised(log_file):
    def _boom(*a):
        raise TimeoutError("read timeout after 600s")

    with pytest.raises(TimeoutError):
        _Backend(_boom).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    [rec] = _records(log_file)
    assert rec["outcome"] == "timeout"
    assert rec["error"]["type"] == "TimeoutError"


def test_undispatched_call_is_recorded_by_abandon(log_file):
    """A call that dies queued (lock timeout / cancel) used to leave no trace."""
    sc.stage_caller(role="architect_general", task_id="chat-7", backend_url="http://localhost:8083")
    sc.abandon_staged(RuntimeError("Inference lock timeout for architect_general"))
    [rec] = _records(log_file)
    assert rec["dispatched"] is False and rec["method"] == "undispatched"
    assert rec["outcome"] == "timeout"
    assert rec["role"] == "architect_general" and rec["caller"]["task_id"] == "chat-7"
    assert rec["server"]["port"] == 8083
    assert rec["queue"]["pre_dispatch_wait_ms"] >= 0.0
    # A dispatched call consumed its stage, so abandon after it is a no-op.
    sc.abandon_staged(RuntimeError("late"))
    assert len(_records(log_file)) == 1


def test_placement_annotation_lands_in_queue_not_caller(log_file):
    sc.stage_caller(role="frontdoor", task_id="chat-1")
    sc.annotate_staged(instance_idx=2, instance_full=False, placement_wait_ms=12.5)
    _Backend(_server_ok).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    [rec] = _records(log_file)
    assert rec["queue"]["instance_idx"] == 2 and rec["queue"]["placement_wait_ms"] == 12.5
    assert "instance_idx" not in rec["caller"]


def test_annotation_without_stage_claims_no_wait(log_file):
    sc.annotate_staged(instance_idx=-1, instance_full=True, placement_wait_ms=0.4)
    _Backend(_server_ok).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    [rec] = _records(log_file)
    assert rec["caller"]["source"] == "unstaged"
    assert "pre_dispatch_wait_ms" not in rec["queue"]
    assert rec["queue"]["instance_full"] is True


def test_client_synthesized_timings_are_not_recorded_as_server(log_file):
    """The /completion early-stop branch builds a stand-in from wall time."""

    def _synth(self, role_config, request, on_chunk):
        sc.note_timings({"predicted_ms": 900.0, "predicted_per_second": 3.0})
        return _ok_result()

    _Backend(_synth).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    [rec] = _records(log_file)
    assert rec["timings"] is None and rec["timings_source"] == "absent"


def test_server_identity_joins_the_stack_sidecar(log_file, tmp_path):
    sidecar = tmp_path / "server_launches" / "8070.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "launch_id": "L1", "launched_at": "2026-10-03T03:45:48.505Z", "pid": 26641,
        "roles": ["frontdoor"], "argv_sha256": "a" * 64, "binary": "/k/llama-server",
        "binary_realpath": "/k/v10/llama-server", "model_path": "/m/q.gguf", "argv": ["x"],
    }))
    _Backend(_server_ok).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    [rec] = _records(log_file)
    server = rec["server"]
    assert server["identity_source"] == "stack_sidecar"
    assert server["launch_id"] == "L1" and server["argv_sha256"] == "a" * 64
    assert server["model_path"] == "/m/q.gguf" and "argv" not in server


def test_missing_sidecar_is_reported_absent(log_file):
    _Backend(_server_ok, base_url="http://localhost:8199").infer(
        _role_config(), InferenceRequest(role="frontdoor", prompt="x")
    )
    [rec] = _records(log_file)
    assert rec["server"]["identity_source"] == "absent"


def test_disabled_log_writes_nothing_and_does_not_interfere(monkeypatch, tmp_path):
    monkeypatch.setenv(sc.LOG_ENV, "off")
    backend = _Backend(_server_ok)
    assert backend.infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x")).success
    assert sc.log_path() is None


def test_rotation_keeps_bounded_history(log_file, monkeypatch):
    monkeypatch.setattr(sc, "_rotation", lambda: (200, 2))
    backend = _Backend(_server_ok)
    for _ in range(6):
        backend.infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    assert log_file.exists()
    assert log_file.with_name(log_file.name + ".1").exists()
    assert log_file.with_name(log_file.name + ".2").exists()
    assert not log_file.with_name(log_file.name + ".3").exists()


def test_write_failure_never_breaks_the_call(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(sc.LOG_ENV, str(blocker / "sub" / "x.jsonl"))  # parent is a file
    result = _Backend(_server_ok).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    assert result.success


def test_resolve_git_head_plain_worktree_and_packed(tmp_path):
    sha = "1" * 40
    repo = tmp_path / "repo"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".git" / "refs" / "heads" / "main").write_text(sha + "\n")
    assert resolve_git_head(repo) == sha

    # linked worktree: .git file -> private gitdir with commondir -> shared refs
    wt = tmp_path / "wt"
    wt.mkdir()
    private = repo / ".git" / "worktrees" / "wt"
    private.mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {private}\n")
    (private / "HEAD").write_text("ref: refs/heads/feat\n")
    (private / "commondir").write_text("../..\n")
    (repo / ".git" / "packed-refs").write_text(f"# pack\n{'2' * 40} refs/heads/feat\n")
    assert resolve_git_head(wt) == "2" * 40

    detached = tmp_path / "det"
    (detached / ".git").mkdir(parents=True)
    (detached / ".git" / "HEAD").write_text("3" * 40 + "\n")
    assert resolve_git_head(detached) == "3" * 40
    assert resolve_git_head(tmp_path / "nope") is None


# ── the real LlamaServerBackend entry points are the choke point ─────────────


def _real_role_config(name: str = "frontdoor"):
    from src.registry_loader import (
        AccelerationConfig,
        MemoryConfig,
        ModelConfig,
        PerformanceMetrics,
        RoleConfig,
    )

    return RoleConfig(
        name=name,
        tier="A",
        description="test",
        model=ModelConfig(name="Qwen-test", path="m.gguf", quant="Q8_0", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


def test_llama_server_backend_entry_points_are_recorded(log_file):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig

    backend = LlamaServerBackend(config=ServerConfig(base_url="http://localhost:8070"))
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "content": "done", "tokens_predicted": 64, "tokens_evaluated": 1540,
        "stop_type": "eos", "timings": dict(SERVER_TIMINGS),
    }
    sc.stage_caller(role="frontdoor", task_id="chat-42")
    with patch.object(backend.client, "post", return_value=response):
        result = backend.infer(
            _real_role_config(), InferenceRequest(role="frontdoor", prompt="x", n_tokens=64)
        )
    assert result.prompt_eval_ms == 812.5
    [rec] = _records(log_file)
    assert rec["method"] == "infer" and rec["caller"]["task_id"] == "chat-42"
    assert rec["timings"]["prompt_n"] == 340 and rec["timings"]["draft_n_accepted"] == 30
    assert rec["notes"] == {"endpoint": "/completion", "stream": False}


def test_native_enqueue_is_optional_and_never_inherited_by_another_call(log_file):
    sc.stage_caller(role="frontdoor", task_id="post-hook")
    sc.annotate_staged(enqueue_ts_epoch=1234.125)
    _Backend(_server_ok).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="x"))
    sc.stage_caller(role="frontdoor", task_id="pre-hook-shaped")
    _Backend(_server_ok).infer(_role_config(), InferenceRequest(role="frontdoor", prompt="y"))
    first, second = _records(log_file)
    assert first["queue"]["enqueue_ts_epoch"] == 1234.125
    assert "enqueue_ts_epoch" not in first["caller"]
    assert "enqueue_ts_epoch" not in second["queue"]
