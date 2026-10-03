"""prompt_eval_ms = 0.0 on every progress-log completion (workspace-89, 2026-10-03).

Root cause: the streaming ``/v1/chat/completions`` path (the one every frontdoor
chat completion takes when the inference tap is on) ignored the ``timings`` object
llama-server attaches to the LAST chunk of a chat stream and hard-coded
``prompt_eval_ms=0.0`` / ``generation_ms=elapsed`` / ``http_overhead_ms=0.0``.
``LLMPrimitives.total_prompt_eval_ms`` summed those zeros into the progress row.

The final chunk shape is llama.cpp ``server-task.cpp``
``to_json_oaicompat_chat_stream``: ``if (timings.prompt_n >= 0)
deltas.back().push_back({"timings", ...})`` — the finish_reason chunk, or the
``choices: []`` usage chunk when ``stream_options.include_usage`` is set.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from src.api.routes.chat_pipeline.telemetry import llm_completion_meta

TIMINGS = {
    "cache_n": 3000,
    "prompt_n": 512,
    "prompt_ms": 1234.5,
    "prompt_per_second": 414.7,
    "predicted_n": 37,
    "predicted_ms": 890.0,
    "predicted_per_second": 41.6,
    "draft_n": 20,
    "draft_n_accepted": 15,
}


class _FakeStream:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b""


def _chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
    return "data: " + json.dumps(
        {"choices": [{"delta": delta, "finish_reason": finish, "index": 0}], **extra}
    )


def _role_config():
    from src.registry_loader import (
        AccelerationConfig,
        MemoryConfig,
        ModelConfig,
        PerformanceMetrics,
        RoleConfig,
    )

    return RoleConfig(
        name="frontdoor",
        tier="A",
        description="test",
        model=ModelConfig(name="m", path="m.gguf", quant="Q8_0", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


def _run(lines, on_chunk=None):
    from src.backends.llama_server import LlamaServerBackend, ServerConfig
    from src.model_server import InferenceRequest

    backend = LlamaServerBackend(
        config=ServerConfig(base_url="http://localhost:8070", use_chat_completions=True)
    )
    request = InferenceRequest(role="frontdoor", prompt="hello", n_tokens=256)
    with patch.object(backend.client, "stream", return_value=_FakeStream(lines)):
        return backend.infer_stream_text(_role_config(), request, on_chunk=on_chunk)


def test_final_chunk_timings_reach_the_result():
    lines = [
        _chunk({"role": "assistant", "content": "Hel"}),
        _chunk({"content": "lo"}),
        _chunk({}, "stop", timings=TIMINGS),
        "data: [DONE]",
    ]
    result = _run(lines, on_chunk=lambda c: None)
    assert result.prompt_eval_ms == pytest.approx(1234.5)
    assert result.generation_ms == pytest.approx(890.0)
    assert result.tokens_generated == 37          # server count, not the chars/4 estimate
    assert result.prompt_tokens == 512 + 3000     # processed + reused = total prompt
    assert result.cached_prompt_tokens == 3000
    assert result.n_tokens_drafted == 20 and result.n_tokens_accepted == 15
    assert result.acceptance_rate == pytest.approx(0.75)
    assert result.predicted_per_second == pytest.approx(41.6)
    assert result.generation_speed == pytest.approx(41.6)


def test_usage_chunk_carrying_timings_is_read_too():
    """With include_usage the timings ride the trailing ``choices: []`` chunk."""
    lines = [
        _chunk({"content": "ok"}),
        _chunk({}, "stop"),
        "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 3512}, "timings": TIMINGS}),
        "data: [DONE]",
    ]
    assert _run(lines).prompt_eval_ms == pytest.approx(1234.5)


def test_no_timings_keeps_the_legacy_estimate():
    """An early-stopped stream never sees the final chunk: estimates, not zeros invented."""
    lines = [_chunk({"content": "abcd"}), "data: [DONE]"]
    result = _run(lines)
    assert result.prompt_eval_ms == 0.0
    assert result.generation_ms > 0.0
    assert result.prompt_tokens is None


def test_progress_completion_meta_now_carries_prompt_eval_ms():
    """End to end at the unit level: the progress-row field is no longer 0.0."""
    lines = [_chunk({"content": "x"}), _chunk({}, "stop", timings=TIMINGS), "data: [DONE]"]
    result = _run(lines)

    class _Primitives:
        total_tokens_generated = result.tokens_generated
        total_prompt_eval_ms = result.prompt_eval_ms
        total_generation_ms = result.generation_ms
        total_http_overhead_ms = result.http_overhead_ms

    meta = llm_completion_meta(_Primitives())
    assert meta["prompt_eval_ms"] == pytest.approx(1234.5)
