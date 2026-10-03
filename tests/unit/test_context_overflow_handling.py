"""llama-server context-size failures: detection, limits, recovery, admission.

Every server payload here uses the frozen production server's REAL strings and
shape (production-consolidated-v10, /mnt/raid0/llm/llama.cpp @ ffc1bac82);
``TestFrozenServerPayloadShape`` re-derives them from that source tree so a
drift in the server strings fails loudly instead of silently un-detecting.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from src.backends.context_limits import (
    ContextLimit,
    ContextLimitResolver,
    estimate_tokens,
    limit_from_registry,
    parse_props,
    set_context_limit_resolver,
    split_urls,
)
from src.backends.context_overflow import (
    classify_error_body,
    classify_error_payload,
    classify_sse_line,
    context_overflow_error_from_info,
)
from src.backends.llama_server import LlamaServerBackend, ServerConfig
from src.exceptions import ContextOverflowError, InferenceError
from src.llm_primitives.context_recovery import recover_context_overflow
from src.model_server import InferenceRequest, InferenceResult
from src.registry_loader import (
    AccelerationConfig,
    MemoryConfig,
    ModelConfig,
    PerformanceMetrics,
    RoleConfig,
)

FROZEN_TREE = Path("/mnt/raid0/llm/llama.cpp")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "llama_server" / "context_overflow_v10.json"

# Exact frozen-server bodies (see the fixture's provenance block).
BODY_400 = (
    '{"error":{"code":400,"message":"request (120000 tokens) exceeds the available context '
    'size (98304 tokens), try increasing it","type":"exceed_context_size_error",'
    '"n_prompt_tokens":120000,"n_ctx":98304}}'
)
BODY_POOL_500 = '{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}'
SSE_POOL = f"data: {BODY_POOL_500}"


@pytest.fixture(autouse=True)
def fresh_pool(monkeypatch):
    """A fresh process-wide SharedKVPoolAdmission (ratio/queue state is global)."""
    import src.scheduling.kv_pool_admission as kpa

    pool = kpa.SharedKVPoolAdmission(occupancy=lambda url: None)
    monkeypatch.setattr(kpa, "_shared_pool_admission", pool)
    return pool


# ── Detection ────────────────────────────────────────────────────────────


class TestClassify:
    def test_http_400_exceed_context_size(self):
        info = classify_error_body(BODY_400, 400)
        assert info is not None
        assert info.kind == ContextOverflowError.REQUEST_TOO_LARGE
        assert (info.n_prompt_tokens, info.n_ctx, info.http_status) == (120000, 98304, 400)
        assert info.error_type == "exceed_context_size_error"

    def test_numbers_recovered_from_message_when_fields_missing(self):
        body = json.dumps({"error": {"code": 400, "message":
                                     "request (5000 tokens) exceeds the available context size (4096 tokens), try increasing it"}})
        info = classify_error_body(body, 400)
        assert info.kind == ContextOverflowError.REQUEST_TOO_LARGE
        assert (info.n_prompt_tokens, info.n_ctx) == (5000, 4096)

    def test_input_larger_than_max_context_variant(self):
        body = json.dumps({"error": {"code": 400, "message":
                                     "input (70000 tokens) is larger than the max context size (65536 tokens). skipping",
                                     "type": "exceed_context_size_error"}})
        info = classify_error_body(body, 400)
        assert info.kind == ContextOverflowError.REQUEST_TOO_LARGE
        assert (info.n_prompt_tokens, info.n_ctx) == (70000, 65536)

    def test_pool_exhausted_500(self):
        info = classify_error_body(BODY_POOL_500, 500)
        assert info.kind == ContextOverflowError.POOL_EXHAUSTED
        assert info.http_status == 500

    def test_pool_exhausted_in_band_sse(self):
        info = classify_sse_line(SSE_POOL)
        assert info is not None and info.kind == ContextOverflowError.POOL_EXHAUSTED

    def test_decode_failed_prefix_still_detected(self):
        # abort_all_slots("decode() failed: " + e.what()) — server-context.cpp:2949-2951
        info = classify_error_payload({"error": {"code": 500,
                                                 "message": "decode() failed: Context size has been exceeded.",
                                                 "type": "server_error"}})
        assert info.kind == ContextOverflowError.POOL_EXHAUSTED

    def test_plain_text_and_bare_object_shapes(self):
        assert classify_error_body("Context size has been exceeded.", 500).kind == ContextOverflowError.POOL_EXHAUSTED
        assert classify_error_payload({"message": "Context size has been exceeded."}).kind == \
            ContextOverflowError.POOL_EXHAUSTED

    def test_unrelated_errors_are_not_overflow(self):
        assert classify_error_body('{"error":{"code":500,"message":"Compute error.","type":"server_error"}}', 500) is None
        assert classify_error_body('{"error":{"code":400,"message":"Invalid input batch.","type":"invalid_request_error"}}', 400) is None
        assert classify_error_body("", 500) is None

    def test_model_content_quoting_the_phrase_is_not_an_error(self):
        # A normal stream chunk whose TEXT mentions the phrase has no error key.
        chunk = 'data: {"content":"Context size has been exceeded. is a llama.cpp error","stop":false}'
        assert classify_sse_line(chunk) is None
        oai = 'data: {"choices":[{"delta":{"content":"exceeds the available context size"}}]}'
        assert classify_sse_line(oai) is None

    def test_typed_error_is_both_inference_and_runtime_error(self):
        err = context_overflow_error_from_info(classify_error_body(BODY_400, 400), role="architect_general",
                                               backend_url="http://localhost:8083")
        assert isinstance(err, ContextOverflowError)
        assert isinstance(err, InferenceError) and isinstance(err, RuntimeError)
        assert not err.retryable
        assert "120000 prompt tokens > per-request n_ctx 98304" in str(err)
        assert err.to_dict()["kind"] == "request_too_large"


class TestFrozenServerPayloadShape:
    """Derive the payloads from the frozen server SOURCE and classify them."""

    @pytest.fixture(scope="class")
    def sources(self):
        ctx = FROZEN_TREE / "tools/server/server-context.cpp"
        common = FROZEN_TREE / "tools/server/server-common.cpp"
        task = FROZEN_TREE / "tools/server/server-task.cpp"
        if not (ctx.exists() and common.exists() and task.exists()):
            pytest.skip("frozen llama.cpp tree not present on this host")
        return {
            "ctx": ctx.read_text(errors="replace"),
            "common": common.read_text(errors="replace"),
            "task": task.read_text(errors="replace"),
        }

    def test_fixture_strings_match_frozen_source(self, sources):
        fixture = json.loads(FIXTURE.read_text())
        fmts = fixture["provenance"]["source_format_strings"]
        ctx_src = sources["ctx"]
        # The two admission messages are split across C++ string literals.
        joined = re.sub(r'"\s*\n\s*"', "", ctx_src)
        assert fmts["request_too_large"] in joined
        assert fmts["input_too_large"] in joined
        assert f'err = "{fmts["pool_exhausted"]}";' in ctx_src
        # exceed_context_size_error is HTTP 400 and carries n_prompt_tokens/n_ctx.
        assert re.search(r'type_str = "exceed_context_size_error";\s*code = 400;', sources["common"])
        assert 'res["n_prompt_tokens"] = n_prompt_tokens;' in sources["task"]
        assert 'res["n_ctx"]           = n_ctx;' in sources["task"]
        # The pool failure goes out with the DEFAULT error type (server_error → 500).
        assert re.search(r"send_error\(slot, err\);", ctx_src)
        assert re.search(r'type_str = "server_error";\s*code = 500;', sources["common"])

    def test_payloads_rendered_from_source_are_classified(self, sources):
        joined = re.sub(r'"\s*\n\s*"', "", sources["ctx"])
        fmt = re.search(r'"(request \(%d tokens\) exceeds the available context size \(%d tokens\)[^"]*)"', joined).group(1)
        message = fmt.replace("%d", "131072", 1).replace("%d", "98304", 1)
        # nlohmann::ordered_json field order: format_error_response then the extras.
        body = json.dumps({"error": {"code": 400, "message": message, "type": "exceed_context_size_error",
                                     "n_prompt_tokens": 131072, "n_ctx": 98304}}, separators=(",", ":"))
        info = classify_error_body(body, 400)
        assert info.kind == ContextOverflowError.REQUEST_TOO_LARGE
        assert (info.n_prompt_tokens, info.n_ctx) == (131072, 98304)

        pool_msg = re.search(r'err = "(Context size[^"]*)";', sources["ctx"]).group(1)
        pool_line = "data: " + json.dumps({"error": {"code": 500, "message": pool_msg, "type": "server_error"}},
                                          separators=(",", ":"))
        assert classify_sse_line(pool_line).kind == ContextOverflowError.POOL_EXHAUSTED

    def test_every_fixture_case_classifies_as_expected(self):
        for case in json.loads(FIXTURE.read_text())["cases"]:
            if "body" in case:
                info = classify_error_body(case["body"], case["http_status"])
            else:
                info = classify_sse_line(case["sse_line"])
            assert info is not None, case["name"]
            exp = case["expect"]
            assert info.kind == exp["kind"], case["name"]
            assert info.n_prompt_tokens == exp["n_prompt_tokens"], case["name"]
            assert info.n_ctx == exp["n_ctx"], case["name"]


# ── Backend: every transport path surfaces the overflow ──────────────────


@pytest.fixture
def role_config():
    return RoleConfig(
        name="architect_general",
        tier="B",
        description="t",
        model=ModelConfig(name="m", path="m.gguf", quant="Q8_0", size_gb=1.0),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


def _backend(handler, *, chat: bool = False) -> LlamaServerBackend:
    cfg = ServerConfig(base_url="http://ctx-test:8083", timeout=30, num_slots=2, connect_timeout=5,
                       retry_count=0, retry_backoff=0.0, use_chat_completions=chat)
    backend = LlamaServerBackend(config=cfg)
    backend.client = httpx.Client(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    return backend


def _req(prompt="hello") -> InferenceRequest:
    return InferenceRequest(role="architect_general", prompt=prompt, n_tokens=64)


class TestBackendDetection:
    @pytest.mark.parametrize("chat", [False, True])
    def test_batch_http_400_returns_overflow_result(self, role_config, chat):
        backend = _backend(lambda req: httpx.Response(400, text=BODY_400), chat=chat)
        result = backend.infer(role_config, _req())
        assert result.success is False and result.output == ""
        assert result.failure_reason == "context_overflow"
        assert result.context_overflow["kind"] == "request_too_large"
        assert result.context_overflow["n_ctx"] == 98304

    @pytest.mark.parametrize("chat", [False, True])
    def test_stream_first_error_500_returns_overflow_result(self, role_config, chat):
        backend = _backend(lambda req: httpx.Response(500, text=BODY_POOL_500), chat=chat)
        result = backend.infer_stream_text(role_config, _req())
        assert result.context_overflow["kind"] == "pool_exhausted"
        assert result.success is False

    def test_stream_in_band_error_is_not_a_silent_empty_success(self, role_config):
        # Before: the error line was ignored and "" came back as success=True.
        sse = 'data: {"content":"par","stop":false}\n\n' + SSE_POOL + "\n\n"
        backend = _backend(lambda req: httpx.Response(200, text=sse,
                                                      headers={"content-type": "text/event-stream"}))
        result = backend.infer_stream_text(role_config, _req())
        assert result.success is False
        assert result.output == ""  # partial text is never an answer
        assert result.context_overflow["kind"] == "pool_exhausted"

    def test_chat_stream_in_band_error(self, role_config):
        sse = ('data: {"choices":[{"delta":{"content":"a"}}]}\n\n' + SSE_POOL + "\n\n")
        backend = _backend(lambda req: httpx.Response(200, text=sse,
                                                      headers={"content-type": "text/event-stream"}), chat=True)
        result = backend.infer_stream_text(role_config, _req())
        assert result.context_overflow["kind"] == "pool_exhausted"
        assert result.success is False and result.output == ""

    def test_stream_in_band_other_error_fails_explicitly(self, role_config):
        sse = 'data: {"error":{"code":500,"message":"Compute error.","type":"server_error"}}\n\n'
        backend = _backend(lambda req: httpx.Response(200, text=sse,
                                                      headers={"content-type": "text/event-stream"}))
        result = backend.infer_stream_text(role_config, _req())
        assert result.success is False and result.context_overflow is None
        assert "Compute error." in result.error_message

    def test_non_overflow_http_error_in_stream_is_structured(self, role_config):
        backend = _backend(lambda req: httpx.Response(400, text='{"error":{"code":400,"message":"bad","type":"invalid_request_error"}}'))
        result = backend.infer_stream_text(role_config, _req())
        assert result.success is False and result.failure_reason == "http_status"

    def test_truncated_completion_is_not_a_clean_stop(self, role_config):
        body = json.dumps({"content": "cut off", "tokens_predicted": 3, "tokens_evaluated": 98300,
                           "stop": True, "stop_type": "limit", "truncated": True, "timings": {}})
        backend = _backend(lambda req: httpx.Response(200, text=body))
        result = backend.infer(role_config, _req())
        assert result.completion_reason == "context_limit"


# ── Knowing the limit ────────────────────────────────────────────────────


PROPS_SPLIT = {"default_generation_settings": {"n_ctx": 98304}, "total_slots": 2}
PROPS_UNIFIED = {"default_generation_settings": {"n_ctx": 196608}, "total_slots": 2}
REG_8083 = {"context_tokens": 196608, "slots": 2, "kv_unified": None}


class TestContextLimits:
    def test_parse_props_split_vs_unified(self):
        split = parse_props("http://h:8083", PROPS_SPLIT, REG_8083)
        assert (split.per_request_n_ctx, split.kv_unified, split.shared_pool, split.pool_tokens) == (98304, False, False, 196608)
        unified = parse_props("http://h:8083", PROPS_UNIFIED, REG_8083)
        assert (unified.per_request_n_ctx, unified.kv_unified, unified.shared_pool, unified.pool_tokens) == (196608, True, True, 196608)

    def test_explicit_kv_unified_in_props_wins(self):
        lim = parse_props("u", {**PROPS_SPLIT, "kv_unified": True}, REG_8083)
        assert lim.kv_unified is True

    def test_live_evidence_beats_declared_but_unreloaded_flag(self):
        lim = parse_props("u", PROPS_SPLIT, {**REG_8083, "kv_unified": True})
        assert lim.kv_unified is False

    def test_registry_fallback_split_and_declared_unified(self):
        assert limit_from_registry("u", REG_8083).per_request_n_ctx == 98304
        lim = limit_from_registry("u", {**REG_8083, "kv_unified": True})
        assert (lim.per_request_n_ctx, lim.shared_pool) == (196608, True)

    def test_resolver_live_then_cache_then_registry_on_failure(self):
        calls = []
        now = [0.0]

        def fetch(url, timeout):
            calls.append(url)
            if len(calls) > 1:
                raise httpx.ConnectError("down")
            return PROPS_UNIFIED

        r = ContextLimitResolver(live=True, ttl_s=60, fetch_props=fetch,
                                 registry_facts=lambda: {8083: REG_8083}, role_urls=lambda: {},
                                 clock=lambda: now[0])
        assert r.limit_for_url("http://localhost:8083").source == "live_props"
        assert r.limit_for_url("http://localhost:8083").per_request_n_ctx == 196608
        assert len(calls) == 1  # cached
        now[0] = 61.0
        lim = r.limit_for_url("http://localhost:8083")
        assert lim.source == "registry" and lim.per_request_n_ctx == 98304

    def test_role_limit_is_min_across_instances(self):
        facts = {8070: {"context_tokens": 262144, "slots": 4}, 8080: {"context_tokens": 262144, "slots": 1}}
        r = ContextLimitResolver(live=False, registry_facts=lambda: facts,
                                 role_urls=lambda: {"frontdoor": ["http://localhost:8070", "http://localhost:8080"]})
        assert r.limit_for_role("frontdoor").per_request_n_ctx == 65536
        assert r.limit_for_role("frontdoor", "full:http://localhost:8080").per_request_n_ctx == 262144

    def test_observe_updates_cached_limit(self):
        r = ContextLimitResolver(live=False, registry_facts=lambda: {8083: REG_8083}, role_urls=lambda: {})
        r.limit_for_url("http://localhost:8083")
        r.observe("http://localhost:8083", n_ctx=65536)
        lim = r.limit_for_url("http://localhost:8083")
        assert (lim.per_request_n_ctx, lim.source) == (65536, "observed")

    def test_live_default_off_in_test_suite(self):
        assert ContextLimitResolver().live_enabled() is False

    def test_helpers(self):
        assert split_urls("full:http://a:1,http://b:2/") == ["http://a:1", "http://b:2"]
        assert estimate_tokens("x" * 20001) == 5001
        assert ContextLimit("u", 100, 1, None, "live_props").fits(99) is True
        assert ContextLimit("u", 100, 1, None, "live_props").fits(90, 10) is False


# ── Recovery policy ──────────────────────────────────────────────────────


def _pool_exc(role="architect_general", url="http://localhost:8083"):
    return context_overflow_error_from_info(classify_error_body(BODY_POOL_500, 500), role=role, backend_url=url)


def _too_large_exc(role="architect_general", url="http://localhost:8083", n_prompt=120000, n_ctx=98304):
    body = json.dumps({"error": {"code": 400, "message":
                                 f"request ({n_prompt} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it",
                                 "type": "exceed_context_size_error", "n_prompt_tokens": n_prompt, "n_ctx": n_ctx}})
    return context_overflow_error_from_info(classify_error_body(body, 400), role=role, backend_url=url)


def _resolver(limits: dict[str, int], *, unified: bool = True) -> ContextLimitResolver:
    ports = {role: 9000 + i for i, role in enumerate(limits)}
    facts = {ports[r]: {"context_tokens": n, "slots": 2, "kv_unified": unified} for r, n in limits.items()}
    urls = {r: [f"http://localhost:{p}"] for r, p in ports.items()}
    return ContextLimitResolver(live=False, registry_facts=lambda: facts, role_urls=lambda: urls)


class TestRecovery:
    def test_a_pool_exhaustion_that_fits_alone_backs_off_and_retries(self):
        sleeps, calls = [], []

        def call(role):
            calls.append(role)
            if len(calls) == 1:
                raise _pool_exc()
            return "answer"

        out = recover_context_overflow(_pool_exc(), prompt="x" * 3000, role="architect_general", call=call,
                                       sleep=sleeps.append, resolver=_resolver({"architect_general": 196608}),
                                       max_pool_retries=2, backoff_s=2.0, reroute_candidates=[])
        assert out == "answer"
        assert calls == ["architect_general", "architect_general"]
        assert len(sleeps) == 2 and 2.0 <= sleeps[0] <= 2.5 and 4.0 <= sleeps[1] <= 5.0

    def test_a_retries_are_bounded_then_typed_error_with_history(self):
        sleeps = []

        def call(role):
            raise _pool_exc()

        with pytest.raises(ContextOverflowError) as ei:
            recover_context_overflow(_pool_exc(), prompt="x" * 3000, role="architect_general", call=call,
                                     sleep=sleeps.append, resolver=_resolver({"architect_general": 196608}),
                                     max_pool_retries=2, backoff_s=0.01, reroute_candidates=[])
        assert len(sleeps) == 2
        assert ei.value.kind == "pool_exhausted"
        assert [a["step"] for a in ei.value.recovery] == ["pool_backoff", "pool_backoff"]

    def test_a_no_backoff_past_the_deadline(self):
        sleeps = []
        with pytest.raises(ContextOverflowError) as ei:
            recover_context_overflow(_pool_exc(), prompt="x", role="architect_general",
                                     call=lambda r: pytest.fail("must not retry"), sleep=sleeps.append,
                                     resolver=_resolver({"architect_general": 196608}), deadline_s=100.5,
                                     clock=lambda: 100.0, max_pool_retries=2, backoff_s=2.0, reroute_candidates=[])
        assert sleeps == []
        assert ei.value.recovery[0]["outcome"] == "skipped_deadline"

    def test_a_admission_refusal_is_not_retried_here(self):
        exc = ContextOverflowError("busy", kind=ContextOverflowError.POOL_EXHAUSTED, source="admission",
                                   n_ctx=196608)
        with pytest.raises(ContextOverflowError):
            recover_context_overflow(exc, prompt="x", role="architect_general",
                                     call=lambda r: pytest.fail("must not retry"), sleep=lambda s: None,
                                     resolver=_resolver({"architect_general": 196608}), reroute_candidates=[])

    def test_b_too_large_reroutes_to_a_larger_context_role(self):
        calls = []

        def call(role):
            calls.append(role)
            return f"from {role}"

        resolver = _resolver({"worker_general": 65536, "ingest_long_context": 196608})
        out = recover_context_overflow(_too_large_exc(role="worker_general", n_prompt=120000, n_ctx=65536),
                                       prompt="x", role="worker_general", call=call, sleep=lambda s: None,
                                       resolver=resolver, reroute_candidates=["ingest_long_context"])
        assert out == "from ingest_long_context" and calls == ["ingest_long_context"]

    def test_b_pool_exhaustion_that_cannot_fit_alone_is_rerouted_not_retried(self):
        calls = []
        resolver = _resolver({"worker_general": 8192, "ingest_long_context": 196608})
        out = recover_context_overflow(_pool_exc(role="worker_general"), prompt="x" * 60000, role="worker_general",
                                       call=lambda r: calls.append(r) or "ok", sleep=lambda s: pytest.fail("no backoff"),
                                       resolver=resolver, reroute_candidates=["ingest_long_context"])
        assert out == "ok" and calls == ["ingest_long_context"]

    def test_c_no_larger_role_raises_clear_typed_error(self):
        resolver = _resolver({"ingest_long_context": 196608})
        with pytest.raises(ContextOverflowError) as ei:
            recover_context_overflow(_too_large_exc(role="ingest_long_context", n_prompt=250000, n_ctx=196608),
                                     prompt="x", role="ingest_long_context",
                                     call=lambda r: pytest.fail("nothing fits"), sleep=lambda s: None,
                                     resolver=resolver, reroute_candidates=["ingest_long_context"])
        err = ei.value
        assert err.kind == "request_too_large" and err.n_prompt_tokens == 250000
        assert err.recovery[-1]["outcome"] == "no_larger_role"

    def test_b_then_a_reroute_target_pool_busy_gets_bounded_retry(self):
        calls = []

        def call(role):
            calls.append(role)
            if len(calls) == 1:
                raise _pool_exc(role="ingest_long_context", url="http://localhost:9001")
            return "ok"

        resolver = _resolver({"worker_general": 65536, "ingest_long_context": 196608})
        out = recover_context_overflow(_too_large_exc(role="worker_general", n_prompt=100000, n_ctx=65536),
                                       prompt="x", role="worker_general", call=call, sleep=lambda s: None,
                                       resolver=resolver, reroute_candidates=["ingest_long_context"],
                                       max_pool_retries=1, backoff_s=0.0)
        assert out == "ok" and calls == ["ingest_long_context", "ingest_long_context"]


# ── Inference layer integration ──────────────────────────────────────────


def _overflow_result(kind="request_too_large"):
    info = classify_error_body(BODY_400 if kind == "request_too_large" else BODY_POOL_500,
                               400 if kind == "request_too_large" else 500)
    return InferenceResult(role="r", output="", tokens_generated=0, generation_speed=0.0, elapsed_time=0.1,
                           success=False, failure_reason="context_overflow", context_overflow=info.to_dict())


def _ok_result(text="fine"):
    return InferenceResult(role="r", output=text, tokens_generated=3, generation_speed=10.0,
                           elapsed_time=0.1, success=True)


class TestInferenceLayer:
    @pytest.fixture
    def prims(self, monkeypatch):
        from src.llm_primitives import LLMPrimitives

        monkeypatch.setenv("ORCHESTRATOR_CTX_POOL_BACKOFF_S", "0")
        tracker = MagicMock()
        tracker.is_available.return_value = True
        p = LLMPrimitives(mock_mode=False,
                          server_urls={"architect_general": "http://localhost:8083",
                                       "ingest_long_context": "http://localhost:8084"},
                          health_tracker=tracker)
        return p

    def _backend(self, *results):
        backend = MagicMock(spec=[])
        backend.infer = MagicMock(side_effect=list(results))
        return backend

    def test_overflow_raises_typed_error_and_never_trips_the_circuit(self, prims):
        set_context_limit_resolver(_resolver({"architect_general": 196608}, unified=False))
        prims._backends["architect_general"] = self._backend(_overflow_result())
        with pytest.raises(ContextOverflowError) as ei:
            prims._real_call("x" * 100, "architect_general")
        assert ei.value.kind == "request_too_large" and ei.value.role == "architect_general"
        prims.health_tracker.record_failure.assert_not_called()

    def test_pool_exhaustion_is_retried_once_and_succeeds(self, prims):
        set_context_limit_resolver(_resolver({"architect_general": 196608}))
        backend = self._backend(_overflow_result("pool_exhausted"), _ok_result("second try"))
        prims._backends["architect_general"] = backend
        assert prims._real_call("x" * 100, "architect_general") == "second try"
        assert backend.infer.call_count == 2

    def test_too_large_reroutes_through_real_call(self, prims, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CONTEXT_OVERFLOW_ROLES", "ingest_long_context")
        facts = {8083: {"context_tokens": 196608, "slots": 2}, 8084: {"context_tokens": 262144, "slots": 1}}
        set_context_limit_resolver(ContextLimitResolver(live=False, registry_facts=lambda: facts,
                                                        role_urls=lambda: {}))
        prims._backends["architect_general"] = self._backend(_overflow_result())
        prims._backends["ingest_long_context"] = self._backend(_ok_result("long ok"))
        assert prims._real_call("x" * 100, "architect_general") == "long ok"


# ── Shared-pool admission ────────────────────────────────────────────────


class TestSharedPoolAdmission:
    def test_lone_request_always_admitted_even_if_larger_than_pool(self):
        from src.api.admission import SharedKVPoolAdmission

        a = SharedKVPoolAdmission()
        t = a.acquire("u", 500_000, 196608, timeout_s=0)
        assert t is not None and a.in_flight_tokens("u") == 196608

    def test_second_long_request_waits_then_times_out(self, monkeypatch):
        from src.api.admission import SharedKVPoolAdmission

        # The TOKEN rule in isolation (the one-long-prefill rule is covered in
        # test_kv_pool_long_prefill.py).
        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_LONG_PREFILL_TOKENS", "0")
        a = SharedKVPoolAdmission()
        first = a.acquire("u", 120_000, 196608, timeout_s=0)
        assert a.acquire("u", 90_000, 196608, timeout_s=0.05, poll_s=0.01) is None
        assert a.acquire("u", 50_000, 196608, timeout_s=0) is not None  # fits alongside
        a.release("u", first)
        assert a.acquire("u", 90_000, 196608, timeout_s=0) is not None

    def test_waiter_is_admitted_when_in_flight_releases(self):
        import threading

        from src.api.admission import SharedKVPoolAdmission

        a = SharedKVPoolAdmission()
        first = a.acquire("u", 150_000, 196608, timeout_s=0)
        got = []
        th = threading.Thread(target=lambda: got.append(a.acquire("u", 100_000, 196608, timeout_s=5, poll_s=0.01)))
        th.start()
        a.release("u", first)
        th.join(timeout=5)
        assert got and got[0] is not None

    def test_fcfs_a_queued_long_request_is_not_starved_by_later_small_ones(self):
        import threading
        import time as _time

        from src.api.admission import SharedKVPoolAdmission

        a = SharedKVPoolAdmission()
        first = a.acquire("u", 150_000, 196608, timeout_s=0)
        got = []
        th = threading.Thread(target=lambda: got.append(a.acquire("u", 100_000, 196608, timeout_s=5, poll_s=0.01)))
        th.start()
        for _ in range(200):
            if a.queued("u") == 1:
                break
            _time.sleep(0.005)
        # 150k + 10k would fit, but the queued 100k request is ahead of it.
        assert a.acquire("u", 10_000, 196608, timeout_s=0.1, poll_s=0.01) is None
        a.release("u", first)
        th.join(timeout=5)
        assert got and got[0] is not None
        assert a.get_status()["u"]["reserved_tokens"] == 100_000

    def test_deadline_bounds_the_queue_wait(self):
        import time as _time

        from src.api.admission import SharedKVPoolAdmission

        a = SharedKVPoolAdmission()
        a.acquire("u", 190_000, 196608, timeout_s=0)
        t0 = _time.perf_counter()
        assert a.acquire("u", 50_000, 196608, deadline_s=t0 + 0.05, poll_s=0.01) is None
        assert _time.perf_counter() - t0 < 2.0

    def test_inference_raises_typed_admission_overflow_when_pool_stays_busy(self, monkeypatch):
        from src.api.admission import get_shared_pool_admission
        from src.llm_primitives import LLMPrimitives

        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_WAIT_S", "0.05")
        set_context_limit_resolver(ContextLimitResolver(
            live=False, registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 2, "kv_unified": True}},
            role_urls=lambda: {}))
        pool = get_shared_pool_admission()
        held = pool.acquire("http://localhost:8083", 190_000, 196608, timeout_s=0)
        try:
            tracker = MagicMock()
            tracker.is_available.return_value = True
            prims = LLMPrimitives(mock_mode=False, server_urls={"architect_general": "http://localhost:8083"},
                                  health_tracker=tracker)
            backend = MagicMock(spec=[])
            backend.infer = MagicMock(side_effect=AssertionError("must not reach the server"))
            prims._backends["architect_general"] = backend
            with pytest.raises(ContextOverflowError) as ei:
                prims._real_call("x" * 60_000, "architect_general", n_tokens=1000)
            assert ei.value.source == "admission" and ei.value.kind == "pool_exhausted"
        finally:
            pool.release("http://localhost:8083", held)


# ── Routing and compaction on tokens ─────────────────────────────────────


class TestRoutingAndCompaction:
    def _route(self, prompt):
        from src.api.models import ChatRequest
        from src.api.routes.chat_pipeline.routing_decision import select_initial_route

        state = MagicMock()
        state.hybrid_router = None
        return select_initial_route(ChatRequest(prompt=prompt, real_mode=True), state, {}, False, {})

    def test_fallback_threshold_when_frontdoor_limit_unknown(self):
        # No frontdoor limit known → the conservative char-knob threshold (5000 tokens) decides.
        set_context_limit_resolver(_resolver({"ingest_long_context": 196608}))
        assert self._route("x" * 20_001)[:2] == (["ingest_long_context"], "long_context_guard")
        assert self._route("x" * 20_000)[1] != "long_context_guard"

    def test_routes_by_live_frontdoor_limit_not_5k_tokens(self):
        # frontdoor 65536 per request (262144 / -np 4), ingest 196608 (unified).
        r = ContextLimitResolver(
            live=False,
            registry_facts=lambda: {9000: {"context_tokens": 262144, "slots": 4},
                                    9001: {"context_tokens": 196608, "slots": 2, "kv_unified": True}},
            role_urls=lambda: {"frontdoor": ["http://localhost:9000"],
                               "ingest_long_context": ["http://localhost:9001"]})
        set_context_limit_resolver(r)
        # 60,000 chars ≈ 20,000 conservative tokens + 4096 reserve: fits 65536 → stays off the guard
        assert self._route("x" * 60_000)[1] != "long_context_guard"
        # 180,000 chars ≈ 60,000 + 4096 = 64,096 < 65536 still fits
        assert self._route("x" * 180_000)[1] != "long_context_guard"
        # 186,000 chars ≈ 62,000 + 4096 = 66,096 ≥ 65536 → long-context role
        assert self._route("x" * 186_000)[:2] == (["ingest_long_context"], "long_context_guard")

    def test_reserved_decode_counts(self):
        from src.api.models import ChatRequest
        from src.api.routes.chat_pipeline.routing_decision import select_initial_route

        set_context_limit_resolver(ContextLimitResolver(
            live=False, registry_facts=lambda: {9000: {"context_tokens": 262144, "slots": 4}},
            role_urls=lambda: {"frontdoor": ["http://localhost:9000"]}))
        state = MagicMock()
        state.hybrid_router = None
        small = ChatRequest(prompt="x" * 150_000, real_mode=True, max_tokens=1000)
        big = ChatRequest(prompt="x" * 150_000, real_mode=True, max_tokens=20_000)
        assert select_initial_route(small, state, {}, False, {})[1] != "long_context_guard"
        assert select_initial_route(big, state, {}, False, {})[1] == "long_context_guard"

    def test_default_reroute_target_is_only_ingest_long_context(self, monkeypatch):
        from src.backends.context_limits import context_overflow_roles

        monkeypatch.delenv("ORCHESTRATOR_CONTEXT_OVERFLOW_ROLES", raising=False)
        assert context_overflow_roles() == ["ingest_long_context"]
        monkeypatch.setenv("ORCHESTRATOR_CONTEXT_OVERFLOW_ROLES", "ingest_long_context, architect_critic")
        assert context_overflow_roles() == ["ingest_long_context", "architect_critic"]

    def test_request_beyond_ingest_limit_goes_to_configured_larger_role(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CONTEXT_OVERFLOW_ROLES", "ingest_long_context,architect_critic")
        set_context_limit_resolver(_resolver({"ingest_long_context": 98304 * 2, "architect_critic": 262144 * 2},
                                             unified=False))
        # ~120k conservative tokens > ingest 98304 per request, < critic 262144
        assert self._route("x" * 360_000)[0] == ["architect_critic"]

    def test_request_beyond_every_limit_still_routes_to_ingest(self):
        set_context_limit_resolver(_resolver({"ingest_long_context": 98304}))
        assert self._route("x" * 900_000)[0] == ["ingest_long_context"]

    def test_compaction_uses_the_live_limit_not_32768(self):
        from src.graph.compaction import _get_model_max_context

        set_context_limit_resolver(_resolver({"architect_general": 196608}, unified=True))
        ctx = MagicMock()
        ctx.deps.primitives.registry = MagicMock(spec=["get_role"])  # the real loader has no get_role_config
        ctx.deps.primitives.server_urls = {}
        ctx.state.current_role = "architect_general"
        assert _get_model_max_context(ctx) == 196608

    @pytest.mark.asyncio
    async def test_forced_compaction_runs_despite_flag_and_turn_gates(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from src.graph.compaction import _maybe_compact_context

        monkeypatch.setenv("ORCHESTRATOR_PATHS_TMP_DIR", str(tmp_path))
        set_context_limit_resolver(_resolver({"worker": 65536}))
        ctx = MagicMock()
        ctx.deps.primitives._count_tokens.return_value = 100
        ctx.deps.primitives.llm_call.return_value = "- index"
        ctx.deps.primitives.server_urls = {}
        ctx.state.turns = 1
        ctx.state.context = "y" * 20000
        ctx.state.current_role = "worker"
        ctx.state.task_id = "overflow"
        ctx.state.compaction_count = 0
        ctx.state.last_compaction_turn = 0
        ctx.state.context_file_paths = []
        ctx.state.compaction_tokens_saved = 0
        with patch("src.features.features") as feat:
            feat.return_value.session_compaction = False
            feat.return_value.session_token_budget = False
            await _maybe_compact_context(ctx)
            assert ctx.state.compaction_count == 0
            await _maybe_compact_context(ctx, force=True)
        assert ctx.state.compaction_count == 1
        assert len(ctx.state.context) < 20000


# ── API mapping ──────────────────────────────────────────────────────────


class TestApiMapping:
    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.api import app
        from src.api.state import get_state, reset_state
        from src.features import reset_features

        monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
        reset_features()
        reset_state()
        get_state()
        with TestClient(app, raise_server_exceptions=False) as c:
            state = get_state()
            if state.registry is None:
                state.registry = MagicMock()
            yield c
        reset_features()

    def _install(self, monkeypatch, exc):
        import src.llm_primitives as llm_primitives_module

        primitives = MagicMock()
        primitives.llm_call.side_effect = exc
        primitives.total_tokens_generated = 0
        monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)

    def _body(self, **kw):
        return {"model": "frontdoor", "messages": [{"role": "user", "content": "hi"}], "x_disable_repl": True, **kw}

    def test_too_large_is_413_with_typed_payload(self, client, monkeypatch):
        self._install(monkeypatch, _too_large_exc())
        r = client.post("/v1/chat/completions", json=self._body())
        assert r.status_code == 413
        assert r.json()["context_overflow"]["kind"] == "request_too_large"

    def test_pool_exhausted_is_503_with_retry_after(self, client, monkeypatch):
        self._install(monkeypatch, _pool_exc())
        r = client.post("/v1/chat/completions", json=self._body())
        assert r.status_code == 503 and r.headers.get("retry-after") == "10"
        assert r.json()["error"] == "context_overflow"

    def test_streaming_emits_typed_terminal_error(self, client, monkeypatch):
        self._install(monkeypatch, _too_large_exc())
        r = client.post("/v1/chat/completions", json=self._body(stream=True))
        assert r.status_code == 200
        assert '"type": "context_overflow"' in r.text or '"type":"context_overflow"' in r.text


# ═══ Serving-engine audit follow-ups (T2, T3, T5, T6, T7) ═══════════════════


# /slots bodies in the frozen server's shape (server-context.cpp:699-722, metrics-only).
def _slot(i, *, processing, n_prompt, n_remain=-1, n_ctx=196608):
    return {"id": i, "n_ctx": n_ctx, "speculative": True, "is_processing": processing,
            "id_task": 10 + i, "n_prompt_tokens": n_prompt, "n_prompt_tokens_processed": n_prompt,
            "n_prompt_tokens_cache": 0, "params": {},
            "next_token": [{"has_next_token": processing, "has_new_line": False,
                            "n_remain": n_remain, "n_decoded": 0}]}


class TestSlotsOccupancy:
    def test_frozen_source_exposes_the_fields_we_read(self):
        src = FROZEN_TREE / "tools/server/server-context.cpp"
        if not src.exists():
            pytest.skip("frozen llama.cpp tree not present on this host")
        text = src.read_text(errors="replace")
        for key in ('{"is_processing", is_processing()}', 'res["n_prompt_tokens"]', '{"n_remain",       n_remaining}',
                    '{"n_ctx",         n_ctx}', 'res["next_token"] = {'):
            assert key in text, key
        # metrics-only by default: /slots does not detokenize prompts (cheap to poll)
        assert "slot.to_json(slots_debug == 0)" in text

    def test_parse_and_project(self):
        from src.backends.context_limits import parse_slots

        occ = parse_slots("u", [_slot(0, processing=True, n_prompt=100_000, n_remain=20_000),
                                _slot(1, processing=False, n_prompt=80_000)])
        assert occ.processing == 1
        assert occ.used_tokens == 100_000            # idle slot's warm prefix is purgeable → free
        assert occ.projected_tokens(1.0) == 120_000
        assert occ.projected_tokens(0.5) == 110_000
        assert parse_slots("u", {"error": "x"}) is None

    def test_resolver_occupancy_cached_and_degrades(self):
        calls = []
        now = [0.0]

        def fetch(url, timeout):
            calls.append(url)
            if len(calls) >= 2:
                raise httpx.HTTPStatusError("501", request=None, response=None)
            return [_slot(0, processing=True, n_prompt=5)]

        r = ContextLimitResolver(live=True, fetch_slots=fetch, fetch_props=lambda u, t: None,
                                 registry_facts=lambda: {}, role_urls=lambda: {}, clock=lambda: now[0])
        assert r.pool_occupancy("http://h:8083").used_tokens == 5
        assert r.pool_occupancy("http://h:8083").used_tokens == 5 and len(calls) == 1
        now[0] = 2.0
        assert r.pool_occupancy("http://h:8083") is None  # /slots unavailable → caller degrades
        assert ContextLimitResolver(live=False).pool_occupancy("http://h:8083") is None

    def test_admission_counts_traffic_it_did_not_admit(self, monkeypatch):
        from src.backends.context_limits import parse_slots
        from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

        # TOKEN rule in isolation: this slot is mid-prefill (n_decoded 0), which
        # the one-long-prefill rule would also (correctly) wait on.
        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_LONG_PREFILL_TOKENS", "0")

        external = parse_slots("u", [_slot(0, processing=True, n_prompt=150_000)])  # opencode → :8083
        pool = SharedKVPoolAdmission(occupancy=lambda url: external)
        assert pool.acquire("u", 60_000, 196608, timeout_s=0.05, poll_s=0.01) is None
        assert pool.acquire("u", 40_000, 196608, timeout_s=0) is not None

    def test_admission_degrades_to_own_tickets_without_slots(self):
        from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

        pool = SharedKVPoolAdmission(occupancy=lambda url: None)
        assert pool.acquire("u", 180_000, 196608, timeout_s=0) is not None
        assert pool.acquire("u", 60_000, 196608, timeout_s=0.05, poll_s=0.01) is None

    def test_occupancy_read_error_degrades(self):
        from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

        def boom(url):
            raise RuntimeError("down")

        pool = SharedKVPoolAdmission(occupancy=boom)
        assert pool.acquire("u", 10, 100, timeout_s=0) is not None


class TestAdaptiveDecodeReservation:
    def test_ratio_decays_on_success_and_resets_on_exhaustion(self, monkeypatch):
        from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO_DECAY", "0.25")
        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO_MIN", "0.3")
        pool = SharedKVPoolAdmission(occupancy=lambda url: None)
        assert pool.new_token_ratio("u") == 1.0
        for _ in range(5):
            pool.release("u", pool.acquire("u", 10, 1000, timeout_s=0), success=True)
        assert pool.new_token_ratio("u") == pytest.approx(0.3)
        pool.release("u", pool.acquire("u", 10, 1000, timeout_s=0), success=False)
        assert pool.new_token_ratio("u") == pytest.approx(0.3)  # failures do not decay
        pool.report_pool_exhausted("u")
        assert pool.new_token_ratio("u") == 1.0

    def test_thinking_call_no_longer_over_serialises(self, monkeypatch):
        from src.scheduling.kv_pool_admission import SharedKVPoolAdmission

        # 120k prompt + max_tokens 32768 on a 196608 pool, then a 60k request.
        # TOKEN rule in isolation (no one-long-prefill rule).
        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_LONG_PREFILL_TOKENS", "0")
        full = SharedKVPoolAdmission(occupancy=lambda url: None)
        full.acquire("u", 120_000, 196608, max_new_tokens=32768, timeout_s=0)
        assert full.acquire("u", 60_000, 196608, timeout_s=0.05, poll_s=0.01) is None

        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO", "0.3")
        adaptive = SharedKVPoolAdmission(occupancy=lambda url: None)
        t = adaptive.acquire("u", 120_000, 196608, max_new_tokens=32768, timeout_s=0)
        assert adaptive.in_flight_tokens("u") == 120_000 + 9831
        assert t is not None and adaptive.acquire("u", 60_000, 196608, timeout_s=0) is not None

    def test_server_pool_exhaustion_resets_ratio_via_inference(self, fresh_pool, monkeypatch):
        from src.llm_primitives import LLMPrimitives

        monkeypatch.setenv("ORCHESTRATOR_CTX_POOL_BACKOFF_S", "0")
        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO_DECAY", "0.5")
        set_context_limit_resolver(ContextLimitResolver(
            live=False, registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 2, "kv_unified": True}},
            role_urls=lambda: {}))
        tracker = MagicMock()
        tracker.is_available.return_value = True
        prims = LLMPrimitives(mock_mode=False, server_urls={"architect_general": "http://localhost:8083"},
                              health_tracker=tracker)
        backend = MagicMock(spec=[])
        backend.infer = MagicMock(side_effect=[_ok_result(), _overflow_result("pool_exhausted"), _ok_result("again")])
        prims._backends["architect_general"] = backend
        prims._real_call("x" * 100, "architect_general", n_tokens=100)
        assert fresh_pool.new_token_ratio("http://localhost:8083") == pytest.approx(0.5)
        assert prims._real_call("x" * 100, "architect_general", n_tokens=100) == "again"
        # reset to 1.0 on exhaustion, then decayed once by the successful retry
        assert fresh_pool.new_token_ratio("http://localhost:8083") == pytest.approx(0.5)
        assert fresh_pool.in_flight_tokens("http://localhost:8083") == 0


class TestBoundedQueue:
    def test_queue_full_is_immediate(self):
        import threading
        import time as _time

        from src.scheduling.kv_pool_admission import KVPoolQueueFull, SharedKVPoolAdmission

        pool = SharedKVPoolAdmission(occupancy=lambda url: None)
        held = pool.acquire("u", 190_000, 196608, timeout_s=0)
        th = threading.Thread(target=lambda: pool.acquire("u", 100_000, 196608, timeout_s=1, poll_s=0.01))
        th.start()
        for _ in range(200):
            if pool.queued("u") == 1:
                break
            _time.sleep(0.005)
        t0 = _time.perf_counter()
        with pytest.raises(KVPoolQueueFull):
            pool.acquire("u", 10, 196608, timeout_s=5, max_queued=1)
        assert _time.perf_counter() - t0 < 0.5
        pool.release("u", held)
        th.join(timeout=5)

    def test_env_bound_and_unbounded(self, monkeypatch):
        from src.scheduling.kv_pool_admission import KVPoolQueueFull, SharedKVPoolAdmission

        pool = SharedKVPoolAdmission(occupancy=lambda url: None)
        pool.acquire("u", 190_000, 196608, timeout_s=0)
        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_MAX_QUEUED", "0")  # unbounded
        assert pool.acquire("u", 100_000, 196608, timeout_s=0.02, poll_s=0.01) is None
        # an admitted-at-once request is never refused by the bound when nobody waits
        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_MAX_QUEUED", "1")
        assert pool.acquire("v", 10, 100, timeout_s=0) is not None
        with pytest.raises(KVPoolQueueFull):
            pool._queue["u"] = [999]  # one waiter already queued
            pool.acquire("u", 1, 196608, timeout_s=0)
        pool._queue.pop("u")

    def test_queue_full_surfaces_as_retryable_typed_error(self, fresh_pool, monkeypatch):
        from src.llm_primitives import LLMPrimitives

        monkeypatch.setenv("ORCHESTRATOR_KV_POOL_MAX_QUEUED", "1")
        set_context_limit_resolver(ContextLimitResolver(
            live=False, registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 2, "kv_unified": True}},
            role_urls=lambda: {}))
        fresh_pool._queue["http://localhost:8083"] = [12345]  # someone is already waiting
        tracker = MagicMock()
        tracker.is_available.return_value = True
        prims = LLMPrimitives(mock_mode=False, server_urls={"architect_general": "http://localhost:8083"},
                              health_tracker=tracker)
        backend = MagicMock(spec=[])
        backend.infer = MagicMock(side_effect=AssertionError("must not dispatch"))
        prims._backends["architect_general"] = backend
        with pytest.raises(ContextOverflowError) as ei:
            prims._real_call("hi", "architect_general", n_tokens=10)
        assert ei.value.retryable and ei.value.source == "admission"
        assert "queue full" in str(ei.value)


class TestMaxTokensClamp:
    def test_clamp_arithmetic(self):
        from src.llm_primitives.inference import _clamp_max_tokens

        lim = ContextLimit("u", 98304, 2, False, "live_props")
        prompt = "x" * 270_000  # ~90000 conservative tokens
        n, note = _clamp_max_tokens(prompt, 32768, lim)
        assert note is not None and n == 98304 - 90000 - 64 and note["from"] == 32768
        assert _clamp_max_tokens("x" * 300, 32768, lim) == (32768, None)       # fits
        assert _clamp_max_tokens(prompt, -1, lim) == (-1, None)                 # unbounded untouched
        assert _clamp_max_tokens("x" * 294_000, 4096, lim)[1] is None           # no useful room left
        assert _clamp_max_tokens(prompt, 32768, None) == (32768, None)          # limit unknown

    def test_inference_clamps_before_dispatch_and_annotates(self, fresh_pool):
        from src.llm_primitives import LLMPrimitives

        set_context_limit_resolver(ContextLimitResolver(
            live=False, registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 2}},
            role_urls=lambda: {}))
        tracker = MagicMock()
        tracker.is_available.return_value = True
        prims = LLMPrimitives(mock_mode=False, server_urls={"architect_general": "http://localhost:8083"},
                              health_tracker=tracker)
        seen = {}

        def infer(role_config, request):
            seen["n_tokens"] = request.n_tokens
            r = _ok_result("cut")
            r.completion_reason = "limit"
            r.tokens_generated = request.n_tokens
            return r

        backend = MagicMock(spec=[])
        backend.infer = MagicMock(side_effect=infer)
        prims._backends["architect_general"] = backend
        assert prims._real_call("x" * 270_000, "architect_general", n_tokens=32768) == "cut"
        assert seen["n_tokens"] == 98304 - 90000 - 64
        meta = prims._last_inference_meta
        assert meta["completion_reason"] == "context_limit"
        assert meta["max_tokens_clamped"]["from"] == 32768

    def test_clamp_annotation_lands_on_this_calls_meta_not_a_concurrent_one(self, fresh_pool):
        """TD-21.33c: a concurrent call on the SAME primitives (asyncio.to_thread) can replace the
        shared `_last_inference_meta` attribute between this call's set and its follow-up writes;
        the clamp annotation must land on this call's own dict (the per-call getter), never on
        the other call's."""
        from src.llm_primitives import LLMPrimitives

        set_context_limit_resolver(ContextLimitResolver(
            live=False, registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 2}},
            role_urls=lambda: {}))
        tracker = MagicMock()
        tracker.is_available.return_value = True
        prims = LLMPrimitives(mock_mode=False, server_urls={"architect_general": "http://localhost:8083"},
                              health_tracker=tracker)

        def infer(role_config, request):
            r = _ok_result("cut")
            r.completion_reason = "limit"
            r.tokens_generated = request.n_tokens
            return r

        backend = MagicMock(spec=[])
        backend.infer = MagicMock(side_effect=infer)
        prims._backends["architect_general"] = backend
        foreign = {"role": "other_request"}
        real_set = prims._set_last_inference_meta

        def set_then_interleave(meta):
            real_set(meta)
            prims._last_inference_meta = foreign  # another thread's call lands right here

        prims._set_last_inference_meta = set_then_interleave
        assert prims._real_call("x" * 270_000, "architect_general", n_tokens=32768) == "cut"
        assert foreign == {"role": "other_request"}
        mine = prims.get_last_inference_meta()
        assert mine["completion_reason"] == "context_limit"
        assert mine["max_tokens_clamped"]["from"] == 32768


class TestModelsAdvertiseContextLength:
    def test_models_list_and_get_carry_context_length(self, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api.routes.openai_compat as oc
        from src.api import app

        monkeypatch.setattr(oc, "available_roles", lambda: ["architect_general", "mystery_role"])
        set_context_limit_resolver(ContextLimitResolver(
            live=False, registry_facts=lambda: {8083: {"context_tokens": 196608, "slots": 2, "kv_unified": True}},
            role_urls=lambda: {"architect_general": ["http://localhost:8083"]}))
        with TestClient(app, raise_server_exceptions=False) as c:
            data = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
            one = c.get("/v1/models/architect_general").json()
        assert data["architect_general"]["context_length"] == 196608
        assert data["architect_general"]["max_model_len"] == 196608
        assert "context_length" not in data["mystery_role"]  # unknown → omitted, never invented
        assert one["context_length"] == 196608


# ═══ Measured 2026-09-24: spec-decode sub-batch failure + split-KV truncation ═══

SPEC_SUBBATCH_MSG = "got exception: speculative batch index 8 is not inside the current sub-batch [0, 8)"
BODY_SPEC_500 = json.dumps({"error": {"code": 500, "message": SPEC_SUBBATCH_MSG, "type": "server_error"}},
                           separators=(",", ":"))
STUDY = Path("/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_kvu_study_20260924/q38_27b_q8")


class TestSpecSubBatchAndTruncation:
    def test_exact_measured_string_is_pool_exhausted(self):
        info = classify_error_body(BODY_SPEC_500, 500)
        assert info is not None and info.kind == ContextOverflowError.POOL_EXHAUSTED
        assert classify_sse_line("data: " + BODY_SPEC_500).kind == ContextOverflowError.POOL_EXHAUSTED
        assert context_overflow_error_from_info(info).retryable

    def test_loose_match_needs_both_fragments(self):
        assert classify_error_payload({"error": {"message": "speculative batch index 3 bad"}}) is None
        assert classify_error_payload({"error": {"message": "sub-batch mismatch"}}) is None
        assert classify_error_payload({"error": {"message": "Speculative batch index 12 is not inside the "
                                                            "current sub-batch [4, 12)"}}).kind == "pool_exhausted"

    def test_string_matches_frozen_source_and_the_measured_log(self):
        src = FROZEN_TREE / "tools/server/server-context.cpp"
        if not src.exists():
            pytest.skip("frozen llama.cpp tree not present on this host")
        text = src.read_text(errors="replace")
        assert '"speculative batch index %d is not inside the current sub-batch [%d, %d)"' in text
        assert 'send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);' in text
        log = STUDY / "unified/np2_L2048/server.stderr"
        if log.exists():
            assert f"error: {SPEC_SUBBATCH_MSG}" in log.read_text(errors="replace")

    def test_backend_turns_spec_subbatch_500_into_pool_overflow(self, role_config):
        backend = _backend(lambda req: httpx.Response(500, text=BODY_SPEC_500), chat=True)
        result = backend.infer(role_config, _req())
        assert result.context_overflow["kind"] == "pool_exhausted"
        assert result.success is False and result.output == ""

    def test_spec_subbatch_gets_the_pool_recovery_path(self):
        exc = context_overflow_error_from_info(classify_error_body(BODY_SPEC_500, 500), role="architect_general")
        calls = []
        out = recover_context_overflow(exc, prompt="x" * 300, role="architect_general",
                                       call=lambda r: calls.append(r) or "ok", sleep=lambda s: None,
                                       resolver=_resolver({"architect_general": 4096}),
                                       max_pool_retries=1, backoff_s=0.0, reroute_candidates=[])
        assert out == "ok" and calls == ["architect_general"]

    def test_split_kv_chat_length_stop_below_max_tokens_is_context_limit(self, role_config):
        # Measured split -np 2 -c 4096: prompt 271 + completion 1777 = 2048 = n_ctx/np, finish_reason length.
        body = json.dumps({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
                           "usage": {"prompt_tokens": 271, "completion_tokens": 1777}})
        backend = _backend(lambda req: httpx.Response(200, text=body), chat=True)
        result = backend.infer(role_config, InferenceRequest(role="architect_general", prompt="q", n_tokens=4096))
        assert result.completion_reason == "context_limit"

    def test_length_stop_at_max_tokens_is_still_length(self, role_config):
        body = json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 64}})
        backend = _backend(lambda req: httpx.Response(200, text=body), chat=True)
        assert backend.infer(role_config, _req()).completion_reason == "length"

    def test_completion_stream_truncated_is_context_limit(self, role_config):
        sse = ('data: {"content":"par","stop":false}\n\n'
               'data: {"content":"","stop":true,"stop_type":"limit","truncated":true,'
               '"tokens_predicted":1777,"tokens_evaluated":271,"timings":{}}\n\n')
        backend = _backend(lambda req: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))
        result = backend.infer_stream_text(role_config, _req())
        assert result.completion_reason == "context_limit" and result.output == "par"

    def test_measured_split_run_shape(self):
        pq = STUDY / "split/np2_L2048/pq.jsonl"
        if not pq.exists():
            pytest.skip("study artifacts not present")
        rows = [json.loads(line) for line in pq.read_text().splitlines() if line.strip()]
        # The shape our chat-path rule keys on: a length stop with completion far below any max_tokens
        # budget, because prompt + completion reached n_ctx/np (2048; the server logged n_tokens = 2047).
        assert rows and all(r["finish_reason"] == "length" and r["completion_tokens"] < 4096
                            and 2040 <= r["prompt_tokens"] + r["completion_tokens"] <= 2048 for r in rows)
