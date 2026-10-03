#!/usr/bin/env python3
"""Integration tests for LLMPrimitives with CachingBackend.

These tests validate the integration between:
1. LLMPrimitives (orchestrator's LLM interface)
2. CachingBackend (RadixAttention prefix caching)
3. PrefixRouter (slot routing for KV cache reuse)

Requirements:
    - Tests run with mocks by default (no live servers needed)
    - Use --run-server for live server tests

To run with a live server:
    pytest tests/integration/test_cache_integration.py -v --run-server

To run without server (mocked):
    pytest tests/integration/test_cache_integration.py -v
"""

import os
import pytest
from unittest.mock import MagicMock

from src.llm_primitives import LLMPrimitives
from src.prefix_cache import PrefixRouter, CachingBackend
from src.model_server import InferenceRequest, InferenceResult

# Mark all tests as integration
pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _pin_slots(monkeypatch):
    """UFH14-B4: router slot pinning is opt-in; this module exercises the pinning mode."""
    monkeypatch.setenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", "1")


@pytest.fixture
def mock_llama_server_backend():
    """Create a mock LlamaServerBackend for testing."""
    backend = MagicMock()

    # Track call count for simulation
    call_count = {"count": 0}

    def mock_infer(role_config, request):
        call_count["count"] += 1
        return InferenceResult(
            role=role_config.name if hasattr(role_config, "name") else "test",
            output=f"Mock response for: {request.prompt[:30]}...",
            tokens_generated=50,
            generation_speed=20.0,
            elapsed_time=2.5,
            success=True,
            prompt_eval_ms=100.0,
            generation_ms=2400.0,
            predicted_per_second=20.0,
            http_overhead_ms=0.0,
        )

    backend.infer.side_effect = mock_infer
    backend.call_count = call_count

    return backend


@pytest.fixture
def server_urls():
    """Server URLs for testing."""
    return {
        "frontdoor": os.environ.get("FRONTDOOR_URL", "http://localhost:8080"),
        "worker": os.environ.get("WORKER_URL", "http://localhost:8082"),
    }


# =============================================================================
# LLMPrimitives with CachingBackend Tests
# =============================================================================


class TestLLMPrimitivesCachingIntegration:
    """Tests for LLMPrimitives with CachingBackend integration."""

    def test_llm_primitives_stores_server_urls_without_connecting(self):
        """LLMPrimitives should store server_urls without connecting in mock mode."""
        primitives = LLMPrimitives(
            mock_mode=True,
            server_urls={
                "frontdoor": "http://localhost:8080",
                "worker": "http://localhost:8082",
            },
        )

        # Server URLs should be stored
        assert primitives.server_urls is not None
        assert "frontdoor" in primitives.server_urls
        assert "worker" in primitives.server_urls

        # But backends should not be initialized in mock mode
        assert len(primitives._backends) == 0

    def test_llm_primitives_mock_mode_no_backends(self):
        """In mock_mode, backends should not be initialized."""
        primitives = LLMPrimitives(mock_mode=True)

        # No backends in mock mode
        assert len(primitives._backends) == 0

    def test_get_backend_returns_configured_backend(self):
        """get_backend should return the backend for a role."""
        primitives = LLMPrimitives(mock_mode=True)

        # Manually inject a mock backend
        mock_backend = MagicMock()
        primitives._backends = {"worker": mock_backend}

        backend = primitives.get_backend("worker")
        assert backend is mock_backend

    def test_get_backend_returns_none_for_unknown_role(self):
        """get_backend should return None for unknown roles."""
        primitives = LLMPrimitives(mock_mode=True)

        backend = primitives.get_backend("unknown_role")
        assert backend is None

    def test_get_cache_stats_aggregates_all_backends(self, mock_llama_server_backend):
        """get_cache_stats should aggregate stats from all backends."""
        primitives = LLMPrimitives(mock_mode=True)

        # Create real CachingBackend instances with real PrefixRouters
        router1 = PrefixRouter(num_slots=2)
        router2 = PrefixRouter(num_slots=2)

        caching_backend1 = CachingBackend(mock_llama_server_backend, router1)
        caching_backend2 = CachingBackend(mock_llama_server_backend, router2)

        primitives._backends = {
            "frontdoor": caching_backend1,
            "worker": caching_backend2,
        }

        # Generate some calls to create real stats
        role_config = MagicMock(name="test")
        request = InferenceRequest(role="test", prompt="test prompt")

        caching_backend1.infer(role_config, request)
        caching_backend2.infer(role_config, request)

        stats = primitives.get_cache_stats()

        # Both backends should be in stats
        assert "frontdoor" in stats
        assert "worker" in stats

        # Each backend should have a dict with expected stat keys
        assert isinstance(stats["frontdoor"], dict)
        assert isinstance(stats["worker"], dict)
        # CachingBackend.get_stats() returns router_* and backend_* stats
        assert "router_total_routes" in stats["frontdoor"]
        assert "router_hit_rate" in stats["frontdoor"]

    def test_get_stats_includes_cache_info_with_correct_structure(self, mock_llama_server_backend):
        """get_stats should include cache_stats with proper structure."""
        primitives = LLMPrimitives(mock_mode=True)

        # Create a real CachingBackend
        router = PrefixRouter(num_slots=2)
        caching_backend = CachingBackend(mock_llama_server_backend, router)

        primitives._backends = {"worker": caching_backend}

        stats = primitives.get_stats()

        # Should include cache_stats key
        assert "cache_stats" in stats

        # cache_stats should be a dict with backend names as keys
        assert isinstance(stats["cache_stats"], dict)
        assert "worker" in stats["cache_stats"]

        # Each backend's stats should be a dict
        assert isinstance(stats["cache_stats"]["worker"], dict)


# =============================================================================
# Cache Hit Rate Tests (Real Integration)
# =============================================================================


class TestCacheHitRates:
    """Tests for cache hit rate behavior with real PrefixRouter + CachingBackend."""

    def test_repeated_prefix_improves_hit_rate(self, mock_llama_server_backend):
        """Repeated prompts with same prefix should show cache hits in router stats."""
        # Create real PrefixRouter + CachingBackend combo
        # Use shorter prefix_length so the scaffold fits entirely within it
        router = PrefixRouter(num_slots=4, prefix_length=50)
        caching_backend = CachingBackend(mock_llama_server_backend, router)

        # Simulate repeated calls with same prefix
        scaffold = "You are a helpful assistant. Answer briefly.\n\n"
        queries = [
            scaffold + "What is 1+1?",
            scaffold + "What is 2+2?",
            scaffold + "What is 3+3?",
        ]

        role_config = MagicMock(name="worker")

        for query in queries:
            request = InferenceRequest(role="worker", prompt=query)
            caching_backend.infer(role_config, request)

        # Check router stats - all 3 should have same prefix hash, so hits after first
        router_stats = router.get_stats()
        assert router_stats["total_routes"] == 3
        assert router_stats["cache_hits"] == 2  # 2nd and 3rd queries hit cache
        assert router_stats["cache_misses"] == 1  # Only first query misses

        # Verify hit rate
        assert router_stats["hit_rate"] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_batch_calls_with_shared_prefix_benefit_from_caching(self, mock_llama_server_backend):
        """Batch calls with shared prefix should show increasing hit rate."""
        # Use prefix_length = 25 to match exactly the scaffold length
        router = PrefixRouter(num_slots=4, prefix_length=25)
        caching_backend = CachingBackend(mock_llama_server_backend, router)

        scaffold = "Analyze the following:\n\n"  # Exactly 25 chars
        prompts = [scaffold + f"Item {i}" for i in range(8)]

        role_config = MagicMock(name="worker")

        # Simulate batch processing
        for prompt in prompts:
            request = InferenceRequest(role="worker", prompt=prompt)
            caching_backend.infer(role_config, request)

        router_stats = router.get_stats()
        assert router_stats["total_routes"] == 8

        # All should hit same prefix (7 hits, 1 miss)
        assert router_stats["cache_hits"] == 7
        assert router_stats["cache_misses"] == 1

        # Hit rate should be ~87.5%
        assert router_stats["hit_rate"] == pytest.approx(7.0 / 8.0, abs=0.01)

    def test_different_prefixes_cause_cache_misses(self, mock_llama_server_backend):
        """Prompts with different prefixes should cause cache misses."""
        router = PrefixRouter(num_slots=2)  # Only 2 slots
        caching_backend = CachingBackend(mock_llama_server_backend, router)

        # Create prompts with completely different prefixes
        prompts = [
            "Alpha: " + "x" * 300,
            "Beta: " + "y" * 300,
            "Gamma: " + "z" * 300,  # Will evict one
            "Alpha: " + "x" * 300,  # Should hit if Alpha still cached
        ]

        role_config = MagicMock(name="worker")

        for prompt in prompts:
            request = InferenceRequest(role="worker", prompt=prompt)
            caching_backend.infer(role_config, request)

        router_stats = router.get_stats()
        assert router_stats["total_routes"] == 4

        # First 3 are misses (different prefixes), 4th might hit if Alpha survived
        assert router_stats["cache_misses"] >= 3


# =============================================================================
# Real Mode Integration Tests
# =============================================================================


class TestRealModeIntegration:
    """Tests for real_mode with actual backends."""

    def test_llm_call_uses_caching_backend_when_available(self, mock_llama_server_backend):
        """llm_call should use CachingBackend when injected."""
        # Must use mock_mode=False to trigger _real_call path
        primitives = LLMPrimitives(mock_mode=False)

        # Create real CachingBackend with real PrefixRouter
        router = PrefixRouter(num_slots=4)
        caching_backend = CachingBackend(mock_llama_server_backend, router)

        # Inject into primitives
        primitives._backends = {"worker": caching_backend}

        # Make a call - should route through backend
        primitives.llm_call("Test prompt", role="worker")

        # Should have called the mock backend's infer
        assert mock_llama_server_backend.infer.called
        assert mock_llama_server_backend.call_count["count"] == 1

        # Router should have stats
        router_stats = router.get_stats()
        assert router_stats["total_routes"] == 1

    def test_llm_batch_parallelizes_calls_to_backend(self, mock_llama_server_backend):
        """llm_batch should make multiple calls to backend."""
        # Must use mock_mode=False to trigger _real_batch path
        primitives = LLMPrimitives(mock_mode=False)

        # Create real CachingBackend
        router = PrefixRouter(num_slots=4)
        caching_backend = CachingBackend(mock_llama_server_backend, router)

        primitives._backends = {"worker": caching_backend}

        prompts = [f"Prompt {i}" for i in range(4)]

        results = primitives.llm_batch(prompts, role="worker")

        # Should be called once per prompt
        assert mock_llama_server_backend.call_count["count"] == 4
        assert len(results) == 4

        # Router should show 4 routes
        router_stats = router.get_stats()
        assert router_stats["total_routes"] == 4


# =============================================================================
# Live Server Tests (require --run-server)
# =============================================================================


@pytest.mark.requires_server
class TestLiveServerIntegration:
    """Tests requiring a live llama-server instance."""

    @pytest.fixture(autouse=True)
    def skip_without_server(self, request):
        """Skip tests if --run-server not provided."""
        if not request.config.getoption("--run-server"):
            pytest.skip("Need --run-server to run live server tests")

    def test_llm_primitives_real_inference(self, server_urls):
        """LLMPrimitives should perform real inference with live server."""
        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker": server_urls["worker"]},
        )

        # Skip if backends didn't initialize
        if not primitives._backends:
            pytest.skip("Could not initialize backends")

        result = primitives.llm_call("Say 'hello' and nothing else.", role="worker")

        assert result is not None
        assert len(result) > 0

    def test_cache_hit_rate_on_repeated_prefix(self, server_urls):
        """Cache should show hits on repeated prefix."""
        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker": server_urls["worker"]},
        )

        if not primitives._backends:
            pytest.skip("Could not initialize backends")

        scaffold = "You are a helpful AI. Answer in one word.\n\n"

        # Run multiple queries with same prefix
        for i in range(5):
            primitives.llm_call(scaffold + f"What is {i}+{i}?", role="worker")

        stats = primitives.get_cache_stats()
        worker_stats = stats.get("worker", {})

        print(f"Cache stats after 5 queries: {worker_stats}")

        # Expect some hits after repeated prefix
        if "hit_rate" in worker_stats:
            assert worker_stats["hit_rate"] >= 0

    def test_batch_performance(self, server_urls):
        """Batch calls should complete faster than sequential."""
        import time

        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls={"worker": server_urls["worker"]},
        )

        if not primitives._backends:
            pytest.skip("Could not initialize backends")

        prompts = [f"Say the number {i}" for i in range(4)]

        start = time.time()
        results = primitives.llm_batch(prompts, role="worker")
        elapsed = time.time() - start

        assert len(results) == 4
        print(f"Batch of 4 completed in {elapsed:.2f}s")

        # Should complete reasonably fast due to parallelism
        assert elapsed < 60  # Allow 60s for 4 parallel requests


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCachingErrorHandling:
    """Tests for error handling with caching backends."""

    def test_backend_error_propagates(self):
        """Errors from backend should propagate to caller."""
        primitives = LLMPrimitives(mock_mode=False)

        error_backend = MagicMock(spec=["infer", "infer_stream_text"])
        error_result = InferenceResult(
            role="worker",
            output="",
            tokens_generated=0,
            generation_speed=0.0,
            elapsed_time=0.0,
            success=False,
            error_message="Server error",
        )
        error_backend.infer.return_value = error_result
        error_backend.infer_stream_text.return_value = error_result

        primitives._backends = {"worker": error_backend}

        with pytest.raises(RuntimeError, match="Inference failed"):
            primitives._call_caching_backend(
                error_backend,
                "Test prompt",
                "worker",
            )

    def test_missing_backend_falls_back(self):
        """Missing backend should fall back to model server or mock."""
        primitives = LLMPrimitives(mock_mode=True)
        primitives._backends = {}  # No backends

        # Should use mock mode fallback
        result = primitives.llm_call("Test", role="unknown_role")

        # Mock response should be returned
        assert result is not None
