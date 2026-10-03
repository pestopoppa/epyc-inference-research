"""Unit tests for llama_server backend."""

import json
from unittest.mock import Mock, patch

import httpx
import pytest

from src.backends.llama_server import (
    CacheStats,
    LlamaServerBackend,
    LlamaServerError,
    ServerConfig,
)
from src.model_server import InferenceRequest
from src.registry_loader import (
    AccelerationConfig,
    GenerationDefaults,
    MemoryConfig,
    ModelConfig,
    PerformanceMetrics,
    RoleConfig,
)


@pytest.fixture
def role_config():
    """Create a minimal role config for testing."""
    return RoleConfig(
        name="test_role",
        tier="C",
        description="Test role",
        model=ModelConfig(
            name="test-model",
            path="test-model.gguf",
            quant="Q4_K_M",
            size_gb=1.0,
        ),
        acceleration=AccelerationConfig(type="baseline", temperature=0.0),
        performance=PerformanceMetrics(baseline_tps=10.0),
        memory=MemoryConfig(residency="hot"),
    )


@pytest.fixture
def server_config():
    """Create a server config for testing."""
    return ServerConfig(
        base_url="http://localhost:8080",
        timeout=120,
        num_slots=4,
        connect_timeout=5,
    )


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values from config module."""
        with patch("src.backends.llama_server._server_cfg") as mock_cfg:
            mock_cfg.return_value = Mock(
                default_url="http://test:8080",
                timeout=300,
                num_slots=8,
                connect_timeout=10,
                retry_count=3,
                retry_backoff=0.5,
            )
            config = ServerConfig()

            assert config.base_url == "http://test:8080"
            assert config.timeout == 300
            assert config.num_slots == 8

    def test_config_creation_with_overrides(self):
        """Test creating config with explicit values."""
        config = ServerConfig(
            base_url="http://custom:9090",
            timeout=60,
            num_slots=2,
        )

        assert config.base_url == "http://custom:9090"
        assert config.timeout == 60
        assert config.num_slots == 2


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        stats = CacheStats(total_requests=100, cache_hits=75, cache_misses=25)
        assert stats.hit_rate == 75.0

    def test_hit_rate_zero_requests(self):
        """Test hit rate with zero requests (division by zero edge case)."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_token_savings_rate(self):
        """Test token savings rate calculation."""
        stats = CacheStats(
            total_prompt_tokens=1000,
            cached_prompt_tokens=600,
        )
        assert stats.token_savings_rate == 60.0

    def test_token_savings_zero_tokens(self):
        """Test token savings with zero tokens (division by zero edge case)."""
        stats = CacheStats()
        assert stats.token_savings_rate == 0.0


class TestLlamaServerBackend:
    """Tests for LlamaServerBackend."""

    def test_initialization_with_config(self, server_config):
        """Test backend initialization with ServerConfig."""
        backend = LlamaServerBackend(config=server_config)

        assert backend.config.base_url == "http://localhost:8080"
        assert backend.config.timeout == 120
        assert isinstance(backend.cache_stats, CacheStats)
        assert backend._healthy is False

    def test_initialization_with_base_url(self):
        """Test backend initialization with just base_url."""
        backend = LlamaServerBackend(base_url="http://custom:9090")

        assert backend.config.base_url == "http://custom:9090"
        assert isinstance(backend.client, httpx.Client)

    def test_build_payload_minimal(self, role_config):
        """Test building payload with minimal request."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", n_tokens=128)

        payload = backend._build_payload(role_config, request)

        assert payload["prompt"] == "Hello"
        assert payload["n_predict"] == 128
        assert payload["cache_prompt"] is True  # Default
        assert payload["temperature"] == 0.0
        assert "top_k" in payload
        assert "top_p" in payload

    def test_build_payload_with_stop_sequences(self, role_config):
        """Test building payload with stop sequences."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(
            role="test",
            prompt="Hello",
            stop_sequences=["END", "STOP"],
        )

        payload = backend._build_payload(role_config, request)

        assert payload["stop"] == ["END", "STOP"]

    def test_build_payload_cache_prompt_override(self, role_config):
        """Test cache_prompt can be overridden per-request."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(
            role="test",
            prompt="Hello",
            cache_prompt=False,  # Override default
        )

        payload = backend._build_payload(role_config, request)

        assert payload["cache_prompt"] is False

    def test_build_payload_temperature_from_role(self, role_config):
        """Role temperature should apply when request omits temperature."""
        role_config.acceleration.temperature = 0.7
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello")

        payload = backend._build_payload(role_config, request)

        assert payload["temperature"] == 0.7

    def test_build_payload_temperature_from_generation_defaults(self, role_config):
        """Role generation defaults should apply when request and acceleration omit temp."""
        role_config.acceleration.temperature = None
        role_config.generation_defaults = GenerationDefaults(temperature=0.3)
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello")

        payload = backend._build_payload(role_config, request)

        assert payload["temperature"] == 0.3

    def test_build_payload_explicit_temperature_overrides_registry(self, role_config):
        """Caller-supplied temperature is an explicit override."""
        role_config.acceleration.temperature = 0.7
        role_config.generation_defaults = GenerationDefaults(temperature=0.3)
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", temperature=0.2)

        payload = backend._build_payload(role_config, request)

        assert payload["temperature"] == 0.2

    def test_build_payload_pins_sampling_seed_and_allows_request_override(self, role_config):
        """Sampling params should be reproducible, with per-request seed override."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        default_payload = backend._build_payload(
            role_config,
            InferenceRequest(role="test", prompt="Hello"),
        )
        override_payload = backend._build_payload(
            role_config,
            InferenceRequest(role="test", prompt="Hello", seed=1234, top_p=0.8, top_k=64),
        )

        assert default_payload["top_k"] == 40
        assert default_payload["top_p"] == 0.95
        assert default_payload["repeat_penalty"] == 1.1
        assert default_payload["seed"] == 42
        assert override_payload["seed"] == 1234
        assert override_payload["top_p"] == 0.8
        assert override_payload["top_k"] == 64

    def test_build_payload_adds_logit_probe_probs_for_frontdoor_only(self, role_config):
        """Logit probe should request top-k probabilities only for frontdoor."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="frontdoor", prompt="Hello")
        role_config.name = "frontdoor"

        with patch("src.features.features", return_value=Mock(logit_probe=True)):
            frontdoor_payload = backend._build_payload(role_config, request)
            role_config.name = "worker_general"
            worker_payload = backend._build_payload(role_config, request)

        assert frontdoor_payload["n_probs"] == 64
        assert "n_probs" not in worker_payload

    def test_build_payload_omits_logit_probe_probs_when_disabled(self, role_config):
        """The default-off logit probe flag should leave payloads unchanged."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="frontdoor", prompt="Hello")
        role_config.name = "frontdoor"

        with patch("src.features.features", return_value=Mock(logit_probe=False)):
            payload = backend._build_payload(role_config, request)

        assert "n_probs" not in payload

    def test_build_payload_honors_explicit_n_probs(self, role_config):
        """Explicit calibration probability capture should be role-independent."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="worker_general", prompt="Hello", n_probs=7)
        role_config.name = "worker_general"

        with patch("src.features.features", return_value=Mock(logit_probe=False)):
            payload = backend._build_payload(role_config, request)

        assert payload["n_probs"] == 7

    def test_build_payload_forwards_post_sampling_probs(self, role_config):
        """TD-1d.2: opt-in only — absent unless the request explicitly sets it."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        role_config.name = "worker_general"

        with patch("src.features.features", return_value=Mock(logit_probe=False)):
            on = backend._build_payload(
                role_config,
                InferenceRequest(
                    role="worker_general", prompt="Hello", n_probs=7, post_sampling_probs=True
                ),
            )
            off = backend._build_payload(
                role_config, InferenceRequest(role="worker_general", prompt="Hello", n_probs=7)
            )

        assert on["post_sampling_probs"] is True
        assert "post_sampling_probs" not in off

    def test_infer_success(self, role_config):
        """Test successful inference with mocked HTTP response."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", n_tokens=64)

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Hello world",
            "tokens_predicted": 5,
            "tokens_evaluated": 10,
            "timings": {
                "prompt_ms": 100.0,
                "predicted_ms": 50.0,
                "predicted_per_second": 33.0,
                # v9: cache_n is the true KV-reuse hit count (tokens_cached is
                # the total slot occupancy — counting it as hits inflated
                # cache_hits on every request).
                "cache_n": 3,
            },
        }

        with patch.object(backend.client, "post", return_value=mock_response):
            result = backend.infer(role_config, request)

        assert result.success is True
        assert result.output == "Hello world"
        assert result.tokens_generated == 5
        assert result.generation_speed == 33.0
        assert result.partial is False
        assert result.degraded is False
        assert result.prompt_eval_ms == 100.0
        assert result.generation_ms == 50.0
        assert result.predicted_per_second == 33.0
        assert backend.cache_stats.cache_hits == 1  # cache_n > 0

    def test_infer_returns_completion_probabilities_when_present(self, role_config):
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", n_tokens=64, n_probs=3)
        rows = [{"content": "H", "probs": [{"tok_str": "H", "prob": 0.9}]}]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Hello world",
            "tokens_predicted": 5,
            "tokens_evaluated": 10,
            "completion_probabilities": rows,
            "timings": {
                "prompt_ms": 100.0,
                "predicted_ms": 50.0,
                "predicted_per_second": 33.0,
                "cache_n": 0,
            },
        }

        with patch.object(backend.client, "post", return_value=mock_response):
            result = backend.infer(role_config, request)

        assert result.completion_probabilities == rows


    def test_infer_writes_logit_probe_for_frontdoor_completion_probs(
        self,
        role_config,
        tmp_path,
    ):
        """Frontdoor logit probe writes hashed first-token top-k probabilities."""
        from src.backends import llama_server

        probe_path = tmp_path / "logit_probe.jsonl"
        backend = LlamaServerBackend(base_url="http://test:8080")
        role_config.name = "frontdoor"
        request = InferenceRequest(role="frontdoor", prompt="Sensitive prompt", n_tokens=64)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "OK",
            "tokens_predicted": 1,
            "tokens_evaluated": 5,
            "completion_probabilities": [
                {
                    "content": "O",
                    "probs": [
                        {"tok_str": "O", "prob": 0.7},
                        {"tok_str": "K", "prob": 0.2},
                    ],
                }
            ],
            "timings": {
                "prompt_ms": 100.0,
                "predicted_ms": 50.0,
                "predicted_per_second": 20.0,
                "cache_n": 0,
            },
        }

        with (
            patch.object(backend.client, "post", return_value=mock_response),
            patch("src.features.features", return_value=Mock(logit_probe=True)),
            patch.object(llama_server, "_LOGIT_PROBE_PATH", str(probe_path)),
        ):
            result = backend.infer(role_config, request)

        assert result.success is True
        rows = probe_path.read_text().splitlines()
        assert len(rows) == 1
        entry = json.loads(rows[0])
        assert entry["prompt_len"] == len("Sensitive prompt")
        assert entry["prompt_hash"]
        assert "Sensitive prompt" not in rows[0]
        assert entry["first_token"] == "O"
        assert entry["top_k_probs"] == [
            {"tok": "O", "prob": 0.7},
            {"tok": "K", "prob": 0.2},
        ]

    def test_infer_empty_long_generation_is_failure(self, role_config):
        """Long-running empty /completion responses should not look successful."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", n_tokens=64)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "",
            "tokens_predicted": 0,
            "tokens_evaluated": 10,
            "timings": {
                "prompt_ms": 100.0,
                "predicted_ms": 40_000.0,
                "predicted_per_second": 0.0,
            },
            "stop": True,
        }

        with (
            patch.object(backend.client, "post", return_value=mock_response),
            patch(
                "src.backends.llama_server.time.time",
                side_effect=[100.0, 140.0, 140.0, 140.0, 140.0, 140.0],
            ),
            patch("src.backends.llama_server.time.perf_counter", side_effect=[0.0, 40.0]),
        ):
            result = backend.infer(role_config, request)

        assert result.success is False
        assert result.degraded is True
        assert result.failure_stage == "generation"
        assert result.failure_reason == "empty_generation"
        assert result.completion_reason == "empty_generation"

    def test_infer_timeout(self, role_config):
        """Test inference timeout handling."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", timeout=10)

        with patch.object(backend.client, "post", side_effect=httpx.TimeoutException("Timeout")):
            result = backend.infer(role_config, request)

        assert result.success is False
        assert "timed out" in result.error_message.lower()
        assert result.failure_reason == "timeout"
        assert result.tokens_generated == 0

    def test_infer_http_error(self, role_config):
        """Test inference with HTTP request error."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello")

        # Simulate network error
        with patch.object(
            backend.client, "post", side_effect=httpx.RequestError("Connection failed")
        ):
            result = backend.infer(role_config, request)

        assert result.success is False
        assert "Server request failed" in result.error_message
        assert result.failure_reason == "request_error"

    def test_infer_http_status_error_returns_structured_failure(self, role_config):
        """A 4xx/5xx from the backend (e.g. a VL server rejecting a misrouted
        text /completion with HTTP 400) becomes a structured degraded result,
        never an uncaught exception surfaced as a raw in-band error string.

        HTTPStatusError is a SIBLING of RequestError (not a subclass), so the
        legacy /completion path previously let it escape infer() uncaught.
        """
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello")

        mock_request = httpx.Request("POST", "http://test:8080/completion")
        mock_response = httpx.Response(400, request=mock_request)
        status_error = httpx.HTTPStatusError(
            "Bad Request", request=mock_request, response=mock_response
        )
        resp = Mock()
        resp.status_code = 400
        resp.raise_for_status = Mock(side_effect=status_error)

        with patch.object(backend.client, "post", return_value=resp):
            result = backend.infer(role_config, request)

        assert result.success is False
        assert result.failure_stage == "transport"
        assert result.failure_reason == "http_status"
        assert result.completion_reason == "http_error"
        assert "HTTP 400" in result.error_message
        assert result.output == ""

    def test_infer_stream_text_partial_timeout_sets_partial_flags(self, role_config):
        """Streaming read timeout with chunks should be marked partial/degraded."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", timeout=10)

        class _StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"content":"Hel"}'
                raise httpx.ReadTimeout("timed out")

        with patch.object(backend.client, "stream", return_value=_StreamResponse()):
            result = backend.infer_stream_text(role_config, request)

        assert result.success is False
        assert result.partial is True
        assert result.degraded is True
        assert result.failure_reason == "read_timeout"
        assert result.completion_reason == "read_timeout_partial"

    def test_infer_stream_text_read_timeout_covers_full_budget(self, role_config):
        """/completion streaming read MUST equal the request budget, not min(_,120).

        The 5th sibling of c12484fb's read caps was missed here: a
        ``min(_overall, 120)`` read cap killed every >120s generation on
        /completion-streaming roles (worker_vision, worker_explore, ...) while
        the eval budget was 420s — the EV-BASELINE-E7 119-120s timeouts. Under
        4-wide fan-out the server can withhold the first SSE byte past 120s.
        """
        config = ServerConfig(base_url="http://test:8080", use_chat_completions=False)
        backend = LlamaServerBackend(config=config)
        request = InferenceRequest(role="test", prompt="Hello", timeout=420)
        captured = {}

        class _StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"content":"OK"}'
                yield 'data: {"stop":true}'

        def _stream(_method, _path, json, timeout):
            captured["timeout"] = timeout
            captured["path"] = _path
            return _StreamResponse()

        with patch.object(backend.client, "stream", side_effect=_stream):
            backend.infer_stream_text(role_config, request)

        assert captured["path"] == "/completion"
        # read must cover the whole 420s budget — NOT capped to 120.
        assert captured["timeout"].read == 420
        assert captured["timeout"].read != 120

    def test_infer_stream_text_empty_long_generation_is_failure(self, role_config):
        """Long-running empty streams should be infrastructure failures."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", timeout=60)

        class _StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield (
                    'data: {"content":"","stop":true,"tokens_predicted":0,'
                    '"tokens_evaluated":10,"timings":{"predicted_ms":40000.0}}'
                )

        with (
            patch.object(backend.client, "stream", return_value=_StreamResponse()),
            patch("src.backends.llama_server.time.time", side_effect=[100.0, 140.0, 140.0]),
            patch("src.backends.llama_server.time.perf_counter", side_effect=[0.0, 40.0]),
        ):
            result = backend.infer_stream_text(role_config, request)

        assert result.success is False
        assert result.degraded is True
        assert result.failure_stage == "generation"
        assert result.failure_reason == "empty_generation"
        assert result.completion_reason == "empty_generation"

    def test_chat_completions_stream_forwards_registry_chat_template_kwargs(self, role_config):
        """Streaming chat-completions should preserve per-role template kwargs."""
        config = ServerConfig(base_url="http://test:8080", use_chat_completions=True)
        backend = LlamaServerBackend(config=config)
        role_config.name = "frontdoor"
        request = InferenceRequest(role="frontdoor", prompt="Hello", n_tokens=64)
        captured_payload = {}

        class _StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"OK"}}]}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
                yield "data: [DONE]"

        def _stream(_method, _path, json, timeout):
            captured_payload.update(json)
            return _StreamResponse()

        with (
            patch.object(backend.client, "stream", side_effect=_stream),
            patch(
                "src.registry.registry_loader.chat_template_kwargs_for_role",
                return_value={"enable_thinking": False},
            ),
        ):
            result = backend.infer_stream_text(role_config, request)

        assert result.success is True
        assert result.output == "OK"
        assert captured_payload["stream"] is True
        assert captured_payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_chat_completions_stream_token_estimate_respects_request_cap(self, role_config):
        """Streaming chat-completions telemetry must not exceed max_tokens."""
        config = ServerConfig(base_url="http://test:8080", use_chat_completions=True)
        backend = LlamaServerBackend(config=config)
        role_config.name = "frontdoor"
        request = InferenceRequest(role="frontdoor", prompt="Hello", n_tokens=8)
        chunk = "This streamed response is much longer than eight token estimate units."

        class _StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                content = json.dumps(chunk)
                yield f'data: {{"choices":[{{"delta":{{"content":{content}}}}}]}}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}'
                yield "data: [DONE]"

        with (
            patch.object(backend.client, "stream", return_value=_StreamResponse()),
            patch("src.backends.llama_server.time.time", side_effect=[100.0, 101.0]),
        ):
            result = backend.infer_stream_text(role_config, request)

        assert result.output == chunk
        assert result.completion_reason == "length"
        assert result.tokens_generated == 8

    def test_health_check_success(self):
        """Test health check with healthy server."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(backend.client, "get", return_value=mock_response):
            healthy = backend.health_check(0)

        assert healthy is True
        assert backend._healthy is True

    def test_health_check_failure(self):
        """Test health check with unreachable server."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        with patch.object(
            backend.client, "get", side_effect=httpx.RequestError("Connection failed")
        ):
            healthy = backend.health_check(0)

        assert healthy is False
        assert backend._healthy is False

    def test_health_check_rate_limiting(self):
        """Test health check is rate limited (< 1s between checks)."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        backend._healthy = True
        backend._last_health_check = 999999999.0  # Recent

        with patch("time.time", return_value=999999999.5):  # 0.5s later
            with patch.object(backend.client, "get") as mock_get:
                healthy = backend.health_check(0)

        # Should not make HTTP call (rate limited)
        mock_get.assert_not_called()
        assert healthy is True

    def test_get_slots(self):
        """Test fetching slot information."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            # llama.cpp v9 field names (n_prompt_tokens / n_prompt_tokens_cache)
            {"id": 0, "state": "idle", "n_prompt_tokens": 100, "n_prompt_tokens_cache": 50},
            {"id": 1, "state": "processing", "n_prompt_tokens": 200, "n_prompt_tokens_cache": 0},
        ]

        with patch.object(backend.client, "get", return_value=mock_response):
            slots = backend.get_slots()

        assert len(slots) == 2
        assert slots[0].slot_id == 0
        assert slots[0].state == "idle"
        assert slots[0].prompt_tokens == 100
        assert slots[0].cache_tokens == 50
        assert slots[1].slot_id == 1
        assert slots[1].state == "processing"
        assert slots[1].prompt_tokens == 200
        assert slots[1].cache_tokens == 0

    def test_get_slots_legacy_field_fallback(self):
        """Legacy n_past/n_cache fields fall back to zero (no crash) on v9."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 0, "state": "idle", "n_past": 100, "n_cache": 50},
        ]

        with patch.object(backend.client, "get", return_value=mock_response):
            slots = backend.get_slots()

        assert len(slots) == 1
        assert slots[0].prompt_tokens == 0
        assert slots[0].cache_tokens == 0

    def test_save_slot_success(self):
        """Test saving slot state."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(backend.client, "post", return_value=mock_response):
            success = backend.save_slot(0, "/tmp/slot_0.cache")

        assert success is True

    def test_save_slot_failure(self):
        """Test saving slot with HTTP error."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        with patch.object(backend.client, "post", side_effect=httpx.RequestError("Network error")):
            success = backend.save_slot(0, "/tmp/slot_0.cache")

        assert success is False

    def test_load_raises_error_on_unhealthy_server(self, role_config):
        """Test load() raises error if server not reachable."""
        backend = LlamaServerBackend(base_url="http://test:8080")

        with patch.object(backend, "health_check", return_value=False):
            with pytest.raises(LlamaServerError, match="Cannot reach"):
                backend.load(role_config)

    def test_unload_always_succeeds(self):
        """Test unload() is a no-op that returns True."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        assert backend.unload(12345) is True


class TestEarlyStopTiming:
    """Tests for timing telemetry when early-stop breaks the SSE stream."""

    def test_early_stop_produces_timing(self, role_config, server_config):
        """When on_chunk raises StopIteration, timings should still be set."""
        backend = LlamaServerBackend(config=server_config)

        # Simulate SSE stream with 5 chunks before early-stop
        sse_lines = [
            'data: {"content": "Hello"}',
            'data: {"content": " world"}',
            'data: {"content": "!"}',
            'data: {"content": " FINAL"}',
            'data: {"content": "(42)"}',
            # stop event would follow but early-stop breaks before it
        ]

        call_count = 0

        def on_chunk(content):
            nonlocal call_count
            call_count += 1
            if call_count >= 4:
                raise StopIteration

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def iter_lines(self):
                return iter(sse_lines)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with patch.object(backend.client, "stream", return_value=FakeResponse()):
            request = InferenceRequest(
                role="test_role",
                prompt="test",
                n_tokens=100,
                timeout=30,
            )
            result = backend.infer_stream_text(role_config, request, on_chunk=on_chunk)

        # Before the fix: generation_ms == 0 because timings dict was empty
        # After the fix: generation_ms > 0 computed from wall clock
        assert result.tokens_generated == 4  # chunks before stop
        assert result.generation_ms > 0, "Early-stop should still produce timing"
        assert result.predicted_per_second > 0, "Early-stop should still produce TPS"


class TestChatCompletionsLogprobs:
    """Chat-path Option A (2026-07-22): n_probs → logprobs/top_logprobs translation.

    llama's OpenAI-compat endpoint ignores the native n_probs param, so every
    chat_completions role returned empty completion_probabilities and
    calibration fell back to the binary proxy (EV-4b/EV-11c confidence void).
    """

    def _backend(self):
        from src.backends.llama_server import LlamaServerBackend, ServerConfig

        config = ServerConfig(base_url="http://test:8080", use_chat_completions=True)
        return LlamaServerBackend(config=config)

    def _chat_response(self, *, logprobs=None):
        mock_response = Mock()
        mock_response.status_code = 200
        choice = {
            "message": {"content": "4"},
            "finish_reason": "stop",
        }
        if logprobs is not None:
            choice["logprobs"] = logprobs
        mock_response.json.return_value = {
            "choices": [choice],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            "timings": {"prompt_ms": 5.0, "predicted_ms": 5.0, "predicted_per_second": 30.0},
        }
        mock_response.raise_for_status = Mock()
        return mock_response

    def test_n_probs_translates_to_openai_logprobs_params(self, role_config):
        backend = self._backend()
        request = InferenceRequest(role="frontdoor", prompt="2+2?", n_tokens=16, n_probs=5)
        captured = {}

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            return self._chat_response(
                logprobs={"content": [{"token": "4", "logprob": -0.05, "top_logprobs": []}]}
            )

        with patch.object(backend.client, "post", side_effect=_post):
            result = backend.infer(role_config, request)

        assert "n_probs" not in captured, "native param must not leak to the OAI endpoint"
        assert captured["logprobs"] is True
        assert captured["top_logprobs"] == 5
        assert result.success is True
        assert result.completion_probabilities == [
            {"token": "4", "logprob": -0.05, "top_logprobs": []}
        ]

    def test_post_sampling_probs_translates_to_openai_chat_params(self, role_config):
        """TD-1d.2: forwarded on the /v1 lane the same way as /completion."""
        backend = self._backend()
        request = InferenceRequest(
            role="frontdoor", prompt="2+2?", n_tokens=16, n_probs=5, post_sampling_probs=True
        )
        captured = {}

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            return self._chat_response(
                logprobs={"content": [{"token": "4", "prob": 0.95, "top_probs": []}]}
            )

        with patch.object(backend.client, "post", side_effect=_post):
            backend.infer(role_config, request)

        assert captured["post_sampling_probs"] is True

    def test_no_n_probs_leaves_payload_and_result_clean(self, role_config):
        backend = self._backend()
        request = InferenceRequest(role="frontdoor", prompt="2+2?", n_tokens=16)
        captured = {}

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            return self._chat_response()

        with patch.object(backend.client, "post", side_effect=_post):
            result = backend.infer(role_config, request)

        assert "logprobs" not in captured
        assert "top_logprobs" not in captured
        assert result.completion_probabilities == []


class TestChatCompletionsSchemaForwarding:
    """TD-21.0: json_schema/grammar must reach the /v1 wire.

    Before this fix, ``_infer_chat_completions`` and
    ``_infer_stream_text_chat_completions`` built the OAI payload with no
    schema field at all, so a caller's ``json_schema``/``grammar`` was
    silently dropped on the /v1 lane (audit finding X1 / TD-1d.1) even
    though ``_build_payload`` (the native /completion lane) has forwarded
    both since HS-4. See ``_apply_schema_constraint``.
    """

    def _backend(self, use_chat_completions=True):
        config = ServerConfig(base_url="http://test:8080", use_chat_completions=use_chat_completions)
        return LlamaServerBackend(config=config)

    def _chat_response(self):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            "timings": {"prompt_ms": 5.0, "predicted_ms": 5.0, "predicted_per_second": 30.0},
        }
        mock_response.raise_for_status = Mock()
        return mock_response

    def test_json_schema_becomes_response_format_non_streaming(self, role_config):
        backend = self._backend()
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        request = InferenceRequest(role="frontdoor", prompt="hi", n_tokens=16, json_schema=schema)
        captured = {}

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            return self._chat_response()

        with patch.object(backend.client, "post", side_effect=_post):
            result = backend.infer(role_config, request)

        assert result.success is True
        assert captured["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
        assert "json_schema" not in captured  # never sent as a bare top-level key

    def test_json_schema_becomes_response_format_streaming(self, role_config):
        backend = self._backend()
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        request = InferenceRequest(role="frontdoor", prompt="hi", n_tokens=16, json_schema=schema)
        captured = {}

        class _StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"{}"}}]}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
                yield "data: [DONE]"

        def _stream(_method, _path, json, timeout):
            captured.update(json)
            return _StreamResponse()

        with patch.object(backend.client, "stream", side_effect=_stream):
            result = backend.infer_stream_text(role_config, request)

        assert result.success is True
        assert captured["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
        assert "json_schema" not in captured

    def test_grammar_forwarded_non_streaming(self, role_config):
        backend = self._backend()
        request = InferenceRequest(
            role="frontdoor", prompt="hi", n_tokens=16, grammar='root ::= "yes" | "no"'
        )
        captured = {}

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            return self._chat_response()

        with patch.object(backend.client, "post", side_effect=_post):
            result = backend.infer(role_config, request)

        assert result.success is True
        assert captured["grammar"] == 'root ::= "yes" | "no"'
        assert "response_format" not in captured

    def test_grammar_forwarded_streaming(self, role_config):
        backend = self._backend()
        request = InferenceRequest(
            role="frontdoor", prompt="hi", n_tokens=16, grammar='root ::= "yes" | "no"'
        )
        captured = {}

        class _StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"yes"}}]}'
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
                yield "data: [DONE]"

        def _stream(_method, _path, json, timeout):
            captured.update(json)
            return _StreamResponse()

        with patch.object(backend.client, "stream", side_effect=_stream):
            result = backend.infer_stream_text(role_config, request)

        assert result.success is True
        assert captured["grammar"] == 'root ::= "yes" | "no"'

    def test_json_schema_and_grammar_both_forwarded_independently(self, role_config):
        """Mirrors ``_build_payload``: no client-side precedence, the server decides."""
        backend = self._backend()
        schema = {"type": "object"}
        request = InferenceRequest(
            role="frontdoor",
            prompt="hi",
            n_tokens=16,
            json_schema=schema,
            grammar='root ::= "x"',
        )
        captured = {}

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            return self._chat_response()

        with patch.object(backend.client, "post", side_effect=_post):
            result = backend.infer(role_config, request)

        assert result.success is True
        assert captured["response_format"]["json_schema"]["schema"] == schema
        assert captured["grammar"] == 'root ::= "x"'

    def test_no_schema_leaves_v1_payload_byte_identical_to_before(self, role_config):
        """No json_schema/grammar on the request -> no new keys on the wire.

        Uses a role with no registry chat_template_kwargs override (unlike
        frontdoor) so the captured payload is a fixed, fully known set of
        keys — the same set built before this change (TD-21.0) existed.
        """
        backend = self._backend()
        request = InferenceRequest(role="worker_general", prompt="hi", n_tokens=16)
        captured = {}

        def _post(_path, json=None, timeout=None):
            captured.update(json or {})
            return self._chat_response()

        with (
            patch.object(backend.client, "post", side_effect=_post),
            patch(
                "src.registry.registry_loader.chat_template_kwargs_for_role",
                return_value=None,
            ),
        ):
            result = backend.infer(role_config, request)

        assert result.success is True
        assert "response_format" not in captured
        assert "json_schema" not in captured
        assert "grammar" not in captured
        assert captured == {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": False,
            "cache_prompt": True,  # UFH14-B4: explicit on the chat lane, as on /completion
            "temperature": 0.0,
            "top_k": 40,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "seed": 42,
        }

    def test_completion_lane_json_schema_and_grammar_unchanged(self, role_config):
        """/completion (``_build_payload``) forwarding is untouched by this change."""
        backend = LlamaServerBackend(base_url="http://test:8080")  # use_chat_completions=False
        schema = {"type": "object"}
        request = InferenceRequest(
            role="test", prompt="Hello", json_schema=schema, grammar='root ::= "x"'
        )

        payload = backend._build_payload(role_config, request)

        assert payload["json_schema"] == schema
        assert payload["grammar"] == 'root ::= "x"'
        assert "response_format" not in payload

    def test_completion_lane_no_schema_unchanged(self, role_config):
        """/completion payload with no schema/grammar carries neither key (pre-existing behaviour)."""
        backend = LlamaServerBackend(base_url="http://test:8080")
        request = InferenceRequest(role="test", prompt="Hello", n_tokens=128)

        payload = backend._build_payload(role_config, request)

        assert "json_schema" not in payload
        assert "grammar" not in payload
        assert "response_format" not in payload
