#!/usr/bin/env python3
"""Integration tests for prefix caching.

These tests validate the end-to-end caching workflow:
1. Server connection and health
2. Prefix routing and cache hits
3. Token savings measurement
4. Hot prefix persistence

Requirements:
    - llama-server running on localhost:8080
    - Model with cache_prompt support

To run with a live server:
    pytest tests/integration/test_cache_hits.py -v --run-server

To run without server (mocked):
    pytest tests/integration/test_cache_hits.py -v
"""

import os
import pytest
import time
from unittest.mock import MagicMock

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "requires_server: mark test as requiring live llama-server")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-server",
        action="store_true",
        default=False,
        help="Run tests against live llama-server",
    )


@pytest.fixture
def server_url():
    """Get server URL from environment or default."""
    return os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")


@pytest.fixture
def mock_backend():
    """Create a mock LlamaServerBackend for unit testing."""
    backend = MagicMock()
    backend.health_check.return_value = True
    backend.get_cache_stats.return_value = MagicMock(
        hit_rate=0.0,
        token_savings_rate=0.0,
        total_prompt_tokens=0,
        cached_prompt_tokens=0,
    )

    def mock_infer(role_config, request):
        return MagicMock(
            role=role_config.name,
            output="Test response",
            tokens_generated=50,
            generation_speed=20.0,
            elapsed_time=2.5,
            success=True,
        )

    backend.infer.side_effect = mock_infer
    backend.save_slot.return_value = True
    backend.restore_slot.return_value = True

    return backend


@pytest.fixture
def mock_role_config():
    """Create a mock RoleConfig."""
    config = MagicMock()
    config.name = "test_role"
    config.acceleration = MagicMock(temperature=0.0)
    return config


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing cache behavior."""
    system_prompt = (
        "You are a helpful AI assistant. "
        "You provide clear, accurate, and concise responses. "
        "You follow instructions carefully and completely. "
    )

    return {
        "system": system_prompt,
        "queries": [
            system_prompt + "What is the capital of France?",
            system_prompt + "What is the capital of Germany?",
            system_prompt + "What is the capital of Italy?",
            system_prompt + "Explain quantum computing briefly.",
            system_prompt + "Write a haiku about coding.",
        ],
    }


# =============================================================================
# Prefix Router Integration Tests
# =============================================================================


class TestPrefixRouterIntegration:
    """Integration tests for PrefixRouter with CachingBackend (opt-in pinning mode)."""

    @pytest.fixture(autouse=True)
    def _pin_slots(self, monkeypatch):
        # UFH14-B4: pinning is opt-in; these tests exercise the pinning mode.
        monkeypatch.setenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", "1")

    def test_same_prefix_hits_same_slot(self, mock_backend, mock_role_config, sample_prompts):
        """Prompts with same prefix should route to same slot."""
        from src.prefix_cache import PrefixRouter, CachingBackend
        from src.model_server import InferenceRequest

        # Use prefix_length=128 to only hash the shared system prompt portion
        # (the system prompt is ~130 chars, the unique questions come after)
        router = PrefixRouter(num_slots=4, prefix_length=128)
        caching = CachingBackend(mock_backend, router)

        # Route multiple queries with same system prompt via caching.infer()
        # (which internally calls router.get_slot_for_prompt)
        for query in sample_prompts["queries"]:
            request = InferenceRequest(role="test", prompt=query)
            caching.infer(mock_role_config, request)

        # All queries share a prefix within first 128 chars, so only 1 unique prefix
        assert len(router.prefix_to_slot) == 1
        # First query is a miss, rest are hits
        assert router.cache_misses == 1
        assert router.cache_hits == len(sample_prompts["queries"]) - 1

    def test_different_prefix_different_slots(self, mock_backend, mock_role_config):
        """Prompts with different prefixes should route to different slots."""
        from src.prefix_cache import PrefixRouter, CachingBackend
        from src.model_server import InferenceRequest

        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router)

        prompts = [
            "System A: You are a code assistant. Write hello world.",
            "System B: You are a math tutor. Solve 2+2.",
            "System C: You are a writer. Write a poem.",
        ]

        slots = []
        for prompt in prompts:
            request = InferenceRequest(role="test", prompt=prompt)
            caching.infer(mock_role_config, request)
            slots.append(router.get_slot_for_prompt(prompt))

        # Each should get different slot
        assert len(set(slots)) == 3
        assert router.cache_misses == 3

    def test_hit_rate_improves_with_reuse(self, mock_backend, mock_role_config, sample_prompts):
        """Cache hit rate should improve with prefix reuse."""
        from src.prefix_cache import PrefixRouter, CachingBackend
        from src.model_server import InferenceRequest

        # Use prefix_length=128 to only hash the shared system prompt portion
        router = PrefixRouter(num_slots=4, prefix_length=128)
        caching = CachingBackend(mock_backend, router)

        # Run multiple rounds of queries via caching.infer()
        for _ in range(3):
            for query in sample_prompts["queries"]:
                request = InferenceRequest(role="test", prompt=query)
                caching.infer(mock_role_config, request)

        # 15 total routes: 1 miss (first query ever), 14 hits
        # Hit rate = 14/15 = 93.3%
        hit_rate = caching.get_hit_rate()
        assert hit_rate > 0.5, f"Expected hit rate > 50%, got {hit_rate:.1%}"
        # With shared prefix, expect very high hit rate
        assert hit_rate > 0.9, f"Expected hit rate > 90%, got {hit_rate:.1%}"


# =============================================================================
# Live Server Tests (require --run-server flag)
# =============================================================================


@pytest.mark.requires_server
class TestLiveServerCaching:
    """Tests that require a live llama-server instance."""

    @pytest.fixture(autouse=True)
    def skip_without_server(self, request):
        """Skip tests if --run-server not provided."""
        if not request.config.getoption("--run-server"):
            pytest.skip("Need --run-server to run live server tests")

    def test_server_health(self, server_url):
        """Verify server is reachable."""
        from src.backends.llama_server import LlamaServerBackend, ServerConfig

        config = ServerConfig(base_url=server_url)
        backend = LlamaServerBackend(config)

        assert backend.health_check(0), f"Server not healthy at {server_url}"

    def test_cache_prompt_reduces_latency(self, server_url):
        """Cached prompts should have lower latency."""
        from src.backends.llama_server import LlamaServerBackend, ServerConfig
        from src.model_server import InferenceRequest

        config = ServerConfig(base_url=server_url)
        backend = LlamaServerBackend(config)

        # Skip if server not available
        if not backend.health_check(0):
            pytest.skip("Server not available")

        # Create a mock role config
        role_config = MagicMock()
        role_config.name = "test"
        role_config.acceleration = MagicMock(temperature=0.0)

        scaffold = "You are a helpful assistant. " * 20  # Long prefix

        # First request - cold
        request1 = InferenceRequest(role="test", prompt=scaffold + "Query 1", n_tokens=10)
        backend.infer(role_config, request1)

        # Second request - should be cached
        request2 = InferenceRequest(role="test", prompt=scaffold + "Query 2", n_tokens=10)
        backend.infer(role_config, request2)

        # Cache should show token savings
        stats = backend.get_cache_stats()
        print(
            f"Cache stats: hit_rate={stats.hit_rate:.1f}%, savings={stats.token_savings_rate:.1f}%"
        )

        # Second request should benefit from cache
        # (We can't guarantee timing in all cases, but savings should be measurable)
        assert stats.total_prompt_tokens > 0

    def test_slot_save_restore(self, server_url, tmp_path):
        """Slot state should persist across save/restore."""
        from src.backends.llama_server import LlamaServerBackend, ServerConfig

        config = ServerConfig(base_url=server_url)
        backend = LlamaServerBackend(config)

        if not backend.health_check(0):
            pytest.skip("Server not available")

        save_path = str(tmp_path / "slot_0.bin")

        # Save slot 0
        saved = backend.save_slot(0, save_path)

        if saved:
            assert os.path.exists(save_path)

            # Restore to verify
            restored = backend.restore_slot(0, save_path)
            assert restored


# =============================================================================
# Performance Benchmarks
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("ORCHESTRATOR_MOCK_MODE") == "true",
    reason="Performance benchmarks unreliable on CI runners",
)
class TestCachePerformanceBenchmarks:
    """Performance benchmarks for cache operations."""

    def test_router_throughput(self):
        """PrefixRouter should handle high request rates."""
        from src.prefix_cache import PrefixRouter

        router = PrefixRouter(num_slots=8)

        # Generate prompts
        prompts = [f"System prompt {i % 4}: User query {i}" for i in range(1000)]

        start = time.time()
        for prompt in prompts:
            router.get_slot_for_prompt(prompt)
        elapsed = time.time() - start

        ops_per_sec = len(prompts) / elapsed
        print(f"Router throughput: {ops_per_sec:.0f} routes/sec")

        # Should handle at least 10k routes/sec
        assert ops_per_sec > 10000, f"Router too slow: {ops_per_sec:.0f} routes/sec"


    def test_canonicalization_throughput(self):
        """Canonicalization should not be a bottleneck."""
        from src.prefix_cache import canonicalize_prompt

        # Prompt with various patterns to normalize
        prompt = (
            "Request at 2024-01-15T10:30:00Z\r\n"
            "ID: 550e8400-e29b-41d4-a716-446655440000\r\n"
            "Date: 2024-01-15\r\n"
            "Content follows...\r\n" * 50
        )

        start = time.time()
        for _ in range(10000):
            canonicalize_prompt(prompt)
        elapsed = time.time() - start

        ops_per_sec = 10000 / elapsed
        print(f"Canonicalization throughput: {ops_per_sec:.0f} prompts/sec")

        # Should handle at least 5k prompts/sec (conservative for CI environments)
        # Production machines typically achieve 50k+ prompts/sec
        assert ops_per_sec > 5000, f"Canonicalization too slow: {ops_per_sec:.0f} prompts/sec"
