#!/usr/bin/env python3
"""Unit tests for prefix caching infrastructure.

Tests cover:
- Prompt canonicalization (Phase C)
- PrefixRouter with LRU eviction (Phase B)
- CachingBackend integration
"""

import pytest
from unittest.mock import MagicMock

from src.prefix_cache import (
    canonicalize_prompt,
    create_prefix_filter,
    PrefixRouter,
    SlotState,
    CachingBackend,
)
from src.model_server import InferenceRequest


# =============================================================================
# Prompt Canonicalization Tests (Phase C)
# =============================================================================


class TestCanonicalizePrompt:
    """Tests for canonicalize_prompt function."""

    def test_strips_trailing_whitespace(self):
        """Should strip trailing whitespace."""
        assert canonicalize_prompt("hello   ") == "hello"
        assert canonicalize_prompt("hello\n\n") == "hello"
        assert canonicalize_prompt("hello\t  \n") == "hello"

    def test_normalizes_crlf(self):
        """Should normalize CRLF to LF."""
        assert canonicalize_prompt("line1\r\nline2") == "line1\nline2"
        assert canonicalize_prompt("line1\rline2") == "line1\nline2"

    def test_normalizes_iso_timestamps(self):
        """Should normalize ISO timestamps to [TIMESTAMP]."""
        assert canonicalize_prompt("Time: 2024-01-15T10:30:00Z") == "Time: [TIMESTAMP]"
        assert canonicalize_prompt("Time: 2024-01-15T10:30:00.123Z") == "Time: [TIMESTAMP]"
        assert canonicalize_prompt("Time: 2024-01-15T10:30:00+05:30") == "Time: [TIMESTAMP]"

    def test_normalizes_dates(self):
        """Should normalize dates to [DATE]."""
        assert canonicalize_prompt("Date: 2024-01-15") == "Date: [DATE]"
        assert canonicalize_prompt("From 2024-01-15 to 2024-02-20") == "From [DATE] to [DATE]"

    def test_normalizes_uuids(self):
        """Should normalize UUIDs to [UUID]."""
        assert canonicalize_prompt("ID: 550e8400-e29b-41d4-a716-446655440000") == "ID: [UUID]"
        assert canonicalize_prompt("ids: ABC12345-DEF6-7890-ABCD-EF1234567890") == "ids: [UUID]"

    def test_collapses_multiple_blank_lines(self):
        """Should collapse multiple blank lines to single."""
        assert canonicalize_prompt("a\n\n\n\nb") == "a\n\nb"
        assert canonicalize_prompt("a\n\n\n\n\n\nb") == "a\n\nb"

    def test_combined_normalization(self):
        """Should apply all normalizations together."""
        input_prompt = (
            "Request at 2024-01-15T10:30:00Z\r\n"
            "ID: 550e8400-e29b-41d4-a716-446655440000\r\n"
            "\r\n"
            "\r\n"
            "\r\n"
            "Done   "
        )
        expected = "Request at [TIMESTAMP]\nID: [UUID]\n\nDone"
        assert canonicalize_prompt(input_prompt) == expected


class TestCreatePrefixFilter:
    """Tests for create_prefix_filter function."""

    def test_single_pattern(self):
        """Should filter single pattern."""
        filter_fn = create_prefix_filter([r"\d{4}-\d{4}-\d{4}"])
        assert filter_fn("Card: 1234-5678-9012") == "Card: [FILTERED]"

    def test_multiple_patterns(self):
        """Should filter multiple patterns."""
        filter_fn = create_prefix_filter(
            [
                r"password=\S+",
                r"token=\S+",
            ]
        )
        result = filter_fn("password=secret123 token=abc456")
        assert result == "[FILTERED] [FILTERED]"

    def test_no_match(self):
        """Should leave non-matching text unchanged."""
        filter_fn = create_prefix_filter([r"\d{16}"])
        assert filter_fn("no numbers here") == "no numbers here"


# =============================================================================
# PrefixRouter Tests (Phase B)
# =============================================================================


class TestSlotState:
    """Tests for SlotState dataclass."""

    def test_hit_rate_calculation(self):
        """Should calculate hit rate correctly."""
        slot = SlotState(slot_id=0, hit_count=3, miss_count=1)
        assert slot.hit_rate == 0.75

    def test_hit_rate_zero_total(self):
        """Should return 0 for zero total requests."""
        slot = SlotState(slot_id=0)
        assert slot.hit_rate == 0.0


class TestPrefixRouter:
    """Tests for PrefixRouter class."""

    def test_initialization(self):
        """Should initialize with correct number of slots."""
        router = PrefixRouter(num_slots=4)
        assert router.num_slots == 4
        assert len(router.slots) == 4

    def test_first_request_allocates_slot(self):
        """First request should allocate a slot."""
        router = PrefixRouter(num_slots=4)
        slot_id = router.get_slot_for_prompt("Hello world")

        assert 0 <= slot_id < 4
        assert router.total_routes == 1
        assert router.cache_misses == 1

    def test_same_prefix_hits_same_slot(self):
        """Same prefix should hit same slot."""
        router = PrefixRouter(num_slots=4, prefix_length=10)
        prefix = "System: You are a helpful assistant."

        slot1 = router.get_slot_for_prompt(prefix + " First request")
        slot2 = router.get_slot_for_prompt(prefix + " Second request")

        assert slot1 == slot2
        assert router.cache_hits == 1
        assert router.cache_misses == 1

    def test_different_prefix_different_slot(self):
        """Different prefixes should get different slots."""
        router = PrefixRouter(num_slots=4, prefix_length=10)

        slot1 = router.get_slot_for_prompt("Prefix A: content")
        slot2 = router.get_slot_for_prompt("Prefix B: content")

        assert slot1 != slot2
        assert router.cache_misses == 2

    def test_lru_eviction(self):
        """Should evict LRU slot when all are full."""
        router = PrefixRouter(num_slots=2, prefix_length=8)

        # Fill both slots
        slot_a = router.get_slot_for_prompt("AAAAAAAA first")
        router.get_slot_for_prompt("BBBBBBBB second")

        # Third prompt should evict LRU (slot_a)
        slot_c = router.get_slot_for_prompt("CCCCCCCC third")

        assert slot_c == slot_a  # Evicted slot_a

    def test_lru_order_updates_on_access(self):
        """Accessing a slot should update its LRU position."""
        router = PrefixRouter(num_slots=2, prefix_length=8)

        # Fill both slots
        router.get_slot_for_prompt("AAAAAAAA first")
        slot_b = router.get_slot_for_prompt("BBBBBBBB second")

        # Access slot_a again (moves to end of LRU)
        router.get_slot_for_prompt("AAAAAAAA again")

        # New prompt should evict slot_b (now LRU)
        slot_c = router.get_slot_for_prompt("CCCCCCCC third")

        assert slot_c == slot_b  # Evicted slot_b

    def test_canonicalization_improves_hits(self):
        """Canonicalization should increase cache hits."""
        router = PrefixRouter(num_slots=4, prefix_length=50)

        # Same logical prompt with different timestamps
        prompt1 = "Request at 2024-01-15T10:30:00Z: Help me"
        prompt2 = "Request at 2024-01-15T11:45:00Z: Help me"

        slot1 = router.get_slot_for_prompt(prompt1, canonicalize=True)
        slot2 = router.get_slot_for_prompt(prompt2, canonicalize=True)

        assert slot1 == slot2  # Should hit same slot
        assert router.cache_hits == 1

    def test_get_stats(self):
        """Should return correct statistics."""
        router = PrefixRouter(num_slots=4)

        router.get_slot_for_prompt("first")
        router.get_slot_for_prompt("first")  # Hit
        router.get_slot_for_prompt("second")

        stats = router.get_stats()
        assert stats["total_routes"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["hit_rate_pct"] == pytest.approx(33.33, rel=0.01)

    def test_reset_stats(self):
        """Should reset all statistics."""
        router = PrefixRouter(num_slots=4)
        router.get_slot_for_prompt("test")

        router.reset_stats()

        stats = router.get_stats()
        assert stats["total_routes"] == 0
        assert stats["cache_hits"] == 0

    def test_clear(self):
        """Should clear all slot assignments."""
        router = PrefixRouter(num_slots=4)
        router.get_slot_for_prompt("test")

        router.clear()

        assert len(router.prefix_to_slot) == 0
        assert all(s.prefix_hash == "" for s in router.slots.values())


class TestCachingBackend:
    """Tests for CachingBackend class (in its opt-in slot-pinning mode)."""

    @pytest.fixture(autouse=True)
    def _pin_slots(self, monkeypatch):
        # UFH14-B4: pinning is opt-in; these tests exercise the pinning mode.
        monkeypatch.setenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", "1")

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend for testing CachingBackend."""
        backend = MagicMock()
        backend.infer.return_value = MagicMock(
            role="test",
            output="response",
            tokens_generated=10,
        )
        backend.infer_stream_text.return_value = MagicMock(
            role="test",
            output="response",
            tokens_generated=10,
        )
        backend.get_cache_stats.return_value = MagicMock(
            hit_rate=50.0,
            token_savings_rate=30.0,
            total_prompt_tokens=100,
            cached_prompt_tokens=30,
        )
        return backend

    def test_routes_to_slot(self, mock_backend):
        """Should route prompts through PrefixRouter."""
        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router)

        # Create request and mock role_config
        request = InferenceRequest(role="test", prompt="Test prompt")
        mock_role_config = MagicMock()

        caching.infer(mock_role_config, request)

        # Router should have tracked the request
        assert router.total_routes == 1

    def test_canonicalizes_prompt(self, mock_backend):
        """Should canonicalize prompts when enabled."""
        caching = CachingBackend(mock_backend, canonicalize=True)

        request = InferenceRequest(role="test", prompt="Time: 2024-01-15T10:00:00Z")
        mock_role_config = MagicMock()

        caching.infer(mock_role_config, request)

        # Canonicalization is ONLY used for cache key routing (get_slot_for_prompt),
        # NOT for mutating the prompt sent to the backend. The original prompt
        # must be preserved to avoid [DATE]/[TIMESTAMP] contamination in output.
        call_args = mock_backend.infer.call_args
        assert call_args[0][1].prompt == "Time: 2024-01-15T10:00:00Z"

    def test_get_hit_rate(self, mock_backend):
        """Should return correct hit rate."""
        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router)

        # Simulate some requests
        router.get_slot_for_prompt("test1")
        router.get_slot_for_prompt("test1")  # Hit

        assert caching.get_hit_rate() == 0.5

    def test_get_stats(self):
        """Should combine router and backend stats including slot_stats and token_savings_pct."""
        # Create a specialized mock backend with specific stats for this test
        backend = MagicMock()
        backend.get_cache_stats.return_value = MagicMock(
            hit_rate=75.0,
            token_savings_rate=50.0,
            total_prompt_tokens=1000,
            cached_prompt_tokens=500,
        )

        router = PrefixRouter(num_slots=4)
        router.get_slot_for_prompt("test")

        caching = CachingBackend(backend, router)
        stats = caching.get_stats()

        assert stats["router_total_routes"] == 1
        assert stats["backend_hit_rate"] == 0.75
        assert stats["total_prompt_tokens"] == 1000
        # C4: slot_stats and token_savings_pct
        assert "slot_stats" in stats
        assert isinstance(stats["slot_stats"], list)
        assert len(stats["slot_stats"]) == 4  # num_slots
        assert "token_savings_pct" in stats
        assert stats["token_savings_pct"] == 50.0

    def test_bypass_slot_for_frontdoor_repl_default_on(self, mock_backend, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_PREFIX_CACHE_BYPASS_FRONTDOOR_REPL", raising=False)
        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router)

        request = InferenceRequest(
            role="frontdoor",
            prompt="repl prompt",
            stop_sequences=["\n```\n"],
        )
        mock_role_config = MagicMock()
        caching.infer(mock_role_config, request)

        assert router.total_routes == 0
        call_args = mock_backend.infer.call_args
        assert call_args[0][1].slot_id is None

    def test_bypass_slot_can_be_disabled(self, mock_backend, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_PREFIX_CACHE_BYPASS_FRONTDOOR_REPL", "0")
        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router)

        request = InferenceRequest(
            role="frontdoor",
            prompt="repl prompt",
            stop_sequences=["\n```\n"],
        )
        mock_role_config = MagicMock()
        caching.infer(mock_role_config, request)

        assert router.total_routes == 1
        call_args = mock_backend.infer.call_args
        assert call_args[0][1].slot_id is not None


# =============================================================================
# Slot-allocation coherence across roles sharing one server
# =============================================================================


class TestSharedServerSlotAllocation:
    """Roles sharing one physical llama-server must share one PrefixRouter.

    Regression guard for the ``shared_with`` id_slot collision: a per-role router
    allocates id_slot in [0, num_slots) from its own private LRU, so two roles on
    the same base_url both hand llama-server id_slot=0 and silently evict each
    other's KV. Neither router can observe it, because each sees only its own
    counters. See src/llm_primitives/backend.py::_router_for.
    """

    def test_one_router_gives_distinct_slots_to_distinct_prompts(self):
        """The fixed shape: one shared router hands out distinct slots."""
        shared = PrefixRouter(num_slots=2)

        slot_role_a = shared.get_slot_for_prompt("role A prompt " + "a" * 400)
        slot_role_b = shared.get_slot_for_prompt("role B prompt " + "b" * 400)

        assert slot_role_a != slot_role_b, (
            "two roles on one server were handed the same id_slot; "
            "they would evict each other"
        )

    def test_separate_routers_collide(self):
        """Characterizes the bug: independent routers both start at slot 0."""
        router_a = PrefixRouter(num_slots=2)
        router_b = PrefixRouter(num_slots=2)

        slot_a = router_a.get_slot_for_prompt("role A prompt " + "a" * 400)
        slot_b = router_b.get_slot_for_prompt("role B prompt " + "b" * 400)

        # Both allocate from their own empty pool, so both return slot 0.
        assert slot_a == slot_b == 0

    def test_router_is_shared_per_url_not_per_role(self):
        """`_router_for` returns the same instance for one URL, distinct across URLs."""
        routers: dict[str, PrefixRouter] = {}

        def router_for(url: str) -> PrefixRouter:
            r = routers.get(url)
            if r is None:
                r = PrefixRouter(num_slots=2)
                routers[url] = r
            return r

        same_server_role_1 = router_for("http://127.0.0.1:8081")
        same_server_role_2 = router_for("http://127.0.0.1:8081")
        other_server = router_for("http://127.0.0.1:8082")

        assert same_server_role_1 is same_server_role_2
        assert other_server is not same_server_role_1


class TestSlotPinningDefaultOff:
    """UFH14-B4: by default the router never pins id_slot; the server picks the slot."""

    @pytest.fixture
    def backend(self):
        backend = MagicMock()
        backend.infer.return_value = MagicMock(role="test", output="r", tokens_generated=1)
        backend.infer_stream_text.return_value = MagicMock(role="test", output="r", tokens_generated=1)
        return backend

    def test_infer_sends_no_slot_and_skips_router(self, backend, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", raising=False)
        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(backend, router)
        caching.infer(MagicMock(), InferenceRequest(role="architect_critic", prompt="p" * 600))
        assert router.total_routes == 0
        assert backend.infer.call_args[0][1].slot_id is None

    def test_stream_sends_no_slot(self, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", raising=False)
        seen = []

        class _StreamBackend:
            def infer(self, role_config, request):  # pragma: no cover - not the path under test
                raise AssertionError("stream call fell back to infer")

            def infer_stream_text(self, role_config, request, on_chunk=None):
                seen.append(request.slot_id)
                return MagicMock()

        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(_StreamBackend(), router)
        caching.infer_stream_text(MagicMock(), InferenceRequest(role="x", prompt="q" * 600), on_chunk=None)
        assert router.total_routes == 0
        assert seen == [None]

    def test_same_system_prompt_no_longer_serializes_on_one_slot(self, backend, monkeypatch):
        """The defect: a 256-char key maps every root-LM prompt of a role to one slot."""
        shared_head = "You are the root LM. " * 20  # > 256 chars, identical across turns
        monkeypatch.setenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", "1")
        pinned = CachingBackend(backend, PrefixRouter(num_slots=4))
        slots = set()
        for i in range(4):
            pinned.infer(MagicMock(), InferenceRequest(role="r", prompt=shared_head + f"task {i}"))
            slots.add(backend.infer.call_args[0][1].slot_id)
        assert len(slots) == 1  # all four pinned to ONE slot
        monkeypatch.setenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", "0")
        for i in range(4):
            pinned.infer(MagicMock(), InferenceRequest(role="r", prompt=shared_head + f"task {i}"))
            assert backend.infer.call_args[0][1].slot_id is None

    def test_explicit_caller_slot_is_preserved(self, backend, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", raising=False)
        caching = CachingBackend(backend, PrefixRouter(num_slots=4))
        caching.infer(MagicMock(), InferenceRequest(role="x", prompt="p", slot_id=3))
        assert backend.infer.call_args[0][1].slot_id == 3

    @pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
    def test_opt_in_values(self, value, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", value)
        assert CachingBackend._pin_slots_enabled() is True

    @pytest.mark.parametrize("value", ["0", "", "off", "false", "no"])
    def test_opt_out_values(self, value, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", value)
        assert CachingBackend._pin_slots_enabled() is False
