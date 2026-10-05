#!/usr/bin/env python3
"""Prefix caching infrastructure for RadixAttention-style KV reuse.

This module provides:
- PrefixRouter: Routes requests to slots based on prefix matching
- Prompt canonicalization for improved cache hit rates
- Cache metrics and monitoring

The design is inspired by SGLang's RadixAttention but adapted for
CPU inference via llama-server's slot-based caching.

Usage:
    from src.prefix_cache import PrefixRouter, canonicalize_prompt

    router = PrefixRouter(num_slots=4)

    # Route a prompt to an optimal slot
    slot_id = router.get_slot_for_prompt(prompt)

    # Canonicalize for better cache hits
    normalized = canonicalize_prompt(prompt)

See research/radix_attention_handoff.md for implementation plan.

RTG-58 P2 / KPF-23 decision (2026-10-04). ``PrefixRouter`` is kept as is, as
the legacy opt-in pin path (``ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS=1``, UFH14-B4).
It is neither replaced nor wrapped by the prefix index
(``src/inference/prefix_index.py``, flag ``ORCHESTRATOR_PREFIX_INDEX``): the index
is a separate per-physical-server structure that the KV pool gate and the
serving record drive. With the index on and ``ORCHESTRATOR_PREFIX_INDEX_PIN=idle``,
``CachingBackend`` asks the index first and pins only a VERIFIED IDLE slot that
holds the request's longest prefix; otherwise this module behaves exactly as
before. Retiring the 256-char hash routing on ``--kv-unified`` servers waits on
the index's shadow metric (``slot_prediction_hits`` against the server's own
``id_slot``) and on INF-05 KV-6's measurement of ``canonicalize_prompt``.

UFH14-B4h (2026-10-05): with pinning off (the default) ``CachingBackend`` returns
before ``get_slot_for_prompt``, so ``canonicalize_prompt`` never runs on the
production path and costs nothing there. It is kept for the opt-in pin path; whether
it survives is KPF-23's decision, on KV-6's measurement.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Canonicalization (Phase C)
# =============================================================================


def canonicalize_prompt(prompt: str) -> str:
    """Normalize a prompt to maximize cache hits.

    Applies transformations that preserve semantic meaning while
    removing variation that would cause cache misses.

    Transformations:
    - Strip trailing whitespace
    - Normalize line endings (CRLF -> LF)
    - Normalize timestamps to [TIMESTAMP] placeholder
    - Normalize UUIDs to [UUID] placeholder
    - Collapse multiple blank lines

    Args:
        prompt: Raw prompt text.

    Returns:
        Canonicalized prompt.
    """
    # Strip trailing whitespace
    prompt = prompt.rstrip()

    # Normalize line endings
    prompt = prompt.replace("\r\n", "\n")
    prompt = prompt.replace("\r", "\n")

    # Normalize ISO timestamps: 2024-01-15T10:30:00 -> [TIMESTAMP]
    prompt = re.sub(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
        "[TIMESTAMP]",
        prompt,
    )

    # Normalize date-only: 2024-01-15 -> [DATE]
    prompt = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", prompt)

    # Normalize UUIDs: 550e8400-e29b-41d4-a716-446655440000 -> [UUID]
    prompt = re.sub(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "[UUID]",
        prompt,
    )

    # Collapse multiple blank lines to single
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    return prompt


# =============================================================================
# Prefix Router (Phase B)
# =============================================================================


@dataclass
class SlotState:
    """State tracked for a server slot."""

    slot_id: int
    prefix_hash: str = ""
    prefix_length: int = 0  # Chars in cached prefix
    last_access: float = field(default_factory=time.time)
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Hit rate for this slot."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


class PrefixRouter:
    """Routes prompts to slots for optimal cache utilization.

    Uses prefix hashing to map prompts with similar beginnings to the
    same slot, enabling KV cache reuse. Implements LRU eviction when
    all slots are occupied with different prefixes.

    Algorithm:
    1. Hash the first N tokens/chars of the prompt
    2. If a slot exists with that prefix hash, return it (cache hit)
    3. Otherwise, allocate the LRU slot (cache miss)
    4. Update slot's prefix hash for future matching

    Attributes:
        num_slots: Number of server slots available.
        prefix_length: Characters to use for prefix hashing.
        slots: Mapping of slot_id to SlotState.
        prefix_to_slot: Mapping of prefix_hash to slot_id.
    """

    def __init__(
        self,
        num_slots: int = 4,
        prefix_length: int = 256,
    ):
        """Initialize the prefix router.

        Args:
            num_slots: Number of server slots to manage.
            prefix_length: Number of characters to use for prefix hashing.
        """
        self.num_slots = num_slots
        self.prefix_length = prefix_length

        # Slot state tracking
        self.slots: dict[int, SlotState] = {i: SlotState(slot_id=i) for i in range(num_slots)}

        # LRU order for eviction (slot_id -> None, ordered by access time)
        self._lru_order: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_slots))

        # Prefix -> slot mapping for O(1) lookup
        self.prefix_to_slot: dict[str, int] = {}

        # Statistics
        self.total_routes = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def get_slot_for_prompt(
        self,
        prompt: str,
        canonicalize: bool = True,
    ) -> int:
        """Get the optimal slot for a prompt.

        Routes the prompt to a slot that:
        1. Has the same prefix cached (cache hit), or
        2. Is the least recently used (cache miss with eviction)

        Args:
            prompt: The prompt to route.
            canonicalize: Whether to canonicalize the prompt first.

        Returns:
            Slot ID (0 to num_slots-1).
        """
        self.total_routes += 1

        # Canonicalize if requested
        if canonicalize:
            prompt = canonicalize_prompt(prompt)

        # Hash the prefix
        prefix_hash = self._hash_prefix(prompt)

        # Check for existing slot with this prefix
        if prefix_hash in self.prefix_to_slot:
            slot_id = self.prefix_to_slot[prefix_hash]
            slot = self.slots[slot_id]
            slot.hit_count += 1
            slot.last_access = time.time()
            self._touch_lru(slot_id)
            self.cache_hits += 1
            logger.debug(f"Cache HIT: slot {slot_id} for prefix {prefix_hash[:8]}...")
            return slot_id

        # Cache miss - allocate LRU slot
        slot_id = self._allocate_slot(prefix_hash)
        self.cache_misses += 1
        logger.debug(f"Cache MISS: allocated slot {slot_id} for prefix {prefix_hash[:8]}...")
        return slot_id

    def _hash_prefix(self, prompt: str) -> str:
        """Hash the prefix of a prompt for slot matching.

        Args:
            prompt: The prompt to hash.

        Returns:
            SHA-256 hex digest of the prefix.
        """
        prefix = prompt[: self.prefix_length]
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    def _allocate_slot(self, prefix_hash: str) -> int:
        """Allocate a slot for a new prefix (LRU eviction).

        Args:
            prefix_hash: Hash of the new prefix.

        Returns:
            Allocated slot ID.
        """
        # Get LRU slot (first in OrderedDict)
        slot_id = next(iter(self._lru_order))

        # Evict old prefix mapping if exists
        old_slot = self.slots[slot_id]
        if old_slot.prefix_hash and old_slot.prefix_hash in self.prefix_to_slot:
            del self.prefix_to_slot[old_slot.prefix_hash]

        # Update slot state
        old_slot.prefix_hash = prefix_hash
        old_slot.last_access = time.time()
        old_slot.miss_count += 1

        # Update mappings
        self.prefix_to_slot[prefix_hash] = slot_id
        self._touch_lru(slot_id)

        return slot_id

    def _touch_lru(self, slot_id: int) -> None:
        """Move a slot to the end of the LRU order (most recently used).

        Args:
            slot_id: Slot to touch.
        """
        self._lru_order.move_to_end(slot_id)

    def get_stats(self) -> dict[str, float | int]:
        """Get routing statistics.

        Returns:
            Dictionary with hit rate, miss rate, and counts.
        """
        hit_rate = self.cache_hits / self.total_routes if self.total_routes > 0 else 0.0
        return {
            "total_routes": self.total_routes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "hit_rate_pct": hit_rate * 100,
        }

    def get_slot_stats(self) -> list[dict[str, int | float | str]]:
        """Get per-slot statistics.

        Returns:
            List of stats for each slot.
        """
        return [
            {
                "slot_id": slot.slot_id,
                "prefix_hash": slot.prefix_hash[:8] + "..." if slot.prefix_hash else "",
                "hit_count": slot.hit_count,
                "miss_count": slot.miss_count,
                "hit_rate": slot.hit_rate,
                "last_access": slot.last_access,
            }
            for slot in self.slots.values()
        ]

    def reset_stats(self) -> None:
        """Reset all routing statistics."""
        self.total_routes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        for slot in self.slots.values():
            slot.hit_count = 0
            slot.miss_count = 0

    def clear(self) -> None:
        """Clear all slot assignments and statistics."""
        self.reset_stats()
        self.prefix_to_slot.clear()
        for slot in self.slots.values():
            slot.prefix_hash = ""
            slot.prefix_length = 0


# =============================================================================
# Caching Backend Integration
# =============================================================================


class CachingBackend:
    """Wrapper that adds prefix caching to a LlamaServerBackend.

    Integrates PrefixRouter with the backend to automatically:
    - Route requests to optimal slots
    - Track cache performance metrics
    - Report savings from prefix reuse

    Usage:
        from src.backends.llama_server import LlamaServerBackend, ServerConfig
        from src.prefix_cache import CachingBackend, PrefixRouter

        backend = LlamaServerBackend(ServerConfig(base_url="http://localhost:8080"))
        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(backend, router)

        result = caching.infer(role_config, request)
        print(f"Cache hit rate: {caching.get_hit_rate():.1%}")

    The Phase E hot-prefix persistence (``save_hot_prefixes`` / ``restore_hot_prefixes``
    / ``clear_saved_prefixes``, slot files under the server's ``--slot-save-path``) was
    deleted in UFH14-B4h (2026-10-05): it had no production caller, and a slot file
    carries no context checkpoints, so a restored prefix on a hybrid model cannot be
    partially reused. The server's own host-RAM prompt cache (``--cache-ram``) is the
    prefix store. ``--slot-save-path`` itself stays on the launch argv: KV migration
    (``src/backends/concurrency_aware.py``) saves and restores slot files under it.
    """

    def __init__(
        self,
        backend: "LlamaServerBackend",  # noqa: F821 - forward reference
        router: PrefixRouter | None = None,
        canonicalize: bool = True,
    ):
        """Initialize the caching wrapper.

        Args:
            backend: The LlamaServerBackend to wrap.
            router: PrefixRouter instance. Creates default if None.
            canonicalize: Whether to canonicalize prompts (pin path only).
        """
        self.backend = backend
        self.router = router if router is not None else PrefixRouter()
        self.canonicalize = canonicalize
        self.frontdoor_repl_bypass_count = 0

    def _frontdoor_repl_bypass_enabled(self) -> bool:
        raw = os.environ.get("ORCHESTRATOR_PREFIX_CACHE_BYPASS_FRONTDOOR_REPL", "1")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _pin_slots_enabled() -> bool:
        """Whether the client-side PrefixRouter may pin ``id_slot`` (default: NO).

        UFH14-B4 (2026-10-03). Pinning is off by default because, against llama-server
        v10, it can only lose:

        * The router keys on the first 256 characters, and every root-LM prompt for a
          role starts with the same static system prompt, so all of a role's calls
          map to ONE slot. A pinned task whose slot is busy is DEFERRED by the server
          (``server-context.cpp:2444-2449``) even while other slots are free — the
          admission gates above, which count slots and tokens, cannot see that wait.
        * ``num_slots`` is one process-wide value (``ORCHESTRATOR_SERVER_NUM_SLOTS``,
          default 2), not each server's ``-np``.
        * The server already does token-level affinity on its own: LCP-similarity
          slot selection, then its host-RAM prompt cache (``--cache-ram``). Under
          ``--kv-unified`` idle slots are cleared after every launch
          (``server-context.cpp:2469-2484``), so a pinned slot holds nothing anyway.

        Set ``ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS=1`` to restore the old behaviour.
        """
        raw = os.environ.get("ORCHESTRATOR_PREFIX_ROUTER_PIN_SLOTS", "0")
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _prefix_index_pin(self, request: "InferenceRequest") -> int | None:  # noqa: F821
        """RTG-58 P2 (c): the slot to pin from the prefix index, or None.

        Only with ``ORCHESTRATOR_PREFIX_INDEX=1`` and
        ``ORCHESTRATOR_PREFIX_INDEX_PIN=idle`` (default ``off``). Reconciled with
        UFH14-B4, which made the 256-char-hash pin opt-in because it sent every
        call of a role to ONE slot and v10 deferred it while other slots were
        free. This pin is never that: it names a slot only when the index has
        VERIFIED (via ``/slots``, within ``ORCHESTRATOR_PREFIX_INDEX_PIN_FRESH_S``)
        that the slot is IDLE and holds this request's longest prefix of at least
        ``ORCHESTRATOR_PREFIX_INDEX_PIN_MIN_TOKENS``. A caller's explicit slot_id
        wins; the chat lane never sends ``id_slot``, so it is never pinned.
        Never raises."""
        try:
            from src.inference import prefix_index

            if prefix_index.pin_policy() != "idle":
                return None
            if getattr(request, "slot_id", None) is not None:
                return None
            if getattr(request, "chat_payload", None) is not None:
                return None
            config = getattr(self.backend, "config", None)
            if config is None or getattr(config, "use_chat_completions", False):
                return None
            url = getattr(config, "base_url", "") or ""
            idx = prefix_index.peek_index(url)
            if idx is None:
                return None
            # Server truth first: the resolver's cached /slots (no extra HTTP
            # inside its TTL); None (live reads off) only ages the view.
            from src.backends.context_limits import get_context_limit_resolver

            idx.reconcile(get_context_limit_resolver().pool_occupancy(url))
            return idx.pin_candidate(
                prefix_index.key_text_for_request(request),
                min_tokens=prefix_index._env_int(prefix_index.PIN_MIN_TOKENS_ENV,
                                                 prefix_index.DEFAULT_PIN_MIN_TOKENS),
                fresh_s=prefix_index._env_float(prefix_index.PIN_FRESH_S_ENV,
                                                prefix_index.DEFAULT_PIN_FRESH_S),
            )
        except Exception:
            return None

    def _should_bypass_slot_routing(self, request: "InferenceRequest") -> bool:  # noqa: F821
        """Return True when slot routing should be skipped for this request."""
        if not self._frontdoor_repl_bypass_enabled():
            return False
        role = (request.role or "").strip().lower()
        if role not in {"frontdoor", "role.frontdoor"}:
            return False
        stop_sequences = request.stop_sequences or []
        return "\n```\n" in stop_sequences

    def _backend_supports_streaming(self) -> bool:
        """Return True when wrapped backend provides a real stream API."""
        try:
            attr = inspect.getattr_static(self.backend, "infer_stream_text")
        except AttributeError:
            return False
        except Exception:
            attr = None
        if attr is None:
            return False
        return callable(getattr(self.backend, "infer_stream_text", None))

    def _request_with_slot(self, request: "InferenceRequest", slot_id: int | None):  # noqa: F821
        """Attach slot_id to dataclass or test-double request objects."""
        if is_dataclass(request):
            return replace(request, slot_id=slot_id)
        try:
            setattr(request, "slot_id", slot_id)
        except Exception:
            pass
        return request

    def infer(
        self,
        role_config: "RoleConfig",  # noqa: F821 - forward reference
        request: "InferenceRequest",  # noqa: F821 - forward reference
    ) -> "InferenceResult":  # noqa: F821 - forward reference
        """Run inference with automatic slot routing.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.

        Returns:
            InferenceResult with output and metrics.
        """
        if self._should_bypass_slot_routing(request):
            self.frontdoor_repl_bypass_count += 1
            return self.backend.infer(role_config, self._request_with_slot(request, None))

        pinned = self._prefix_index_pin(request)
        if pinned is not None:
            return self.backend.infer(role_config, self._request_with_slot(request, pinned))

        if not self._pin_slots_enabled():
            # Server-side affinity (LCP + --cache-ram); an explicit caller slot_id is kept.
            return self.backend.infer(role_config, request)

        # Get optimal slot from prefix router
        prompt = request.prompt or ""
        slot_id = self.router.get_slot_for_prompt(prompt, canonicalize=self.canonicalize)

        # Pass computed slot_id to backend via request (id_slot in llama-server)
        routed_request = self._request_with_slot(request, slot_id)

        # Forward to backend
        # NOTE: Canonicalization is intentionally NOT applied to the actual prompt.
        # It is only used for cache key computation in get_slot_for_prompt() above.
        # Applying it here was a bug — it replaced ISO dates with "[DATE]" in the
        # prompt sent to the model, contaminating inference output.
        return self.backend.infer(role_config, routed_request)

    def infer_stream_text(
        self,
        role_config: "RoleConfig",  # noqa: F821
        request: "InferenceRequest",  # noqa: F821
        on_chunk=None,
    ) -> "InferenceResult":  # noqa: F821
        """Stream inference with prefix caching (delegates to backend)."""
        if not self._backend_supports_streaming():
            # Test doubles may expose dynamic attributes but no real streaming API.
            return self.infer(role_config, request)
        if self._should_bypass_slot_routing(request):
            self.frontdoor_repl_bypass_count += 1
            return self.backend.infer_stream_text(
                role_config,
                self._request_with_slot(request, None),
                on_chunk=on_chunk,
            )

        pinned = self._prefix_index_pin(request)
        if pinned is not None:
            return self.backend.infer_stream_text(
                role_config, self._request_with_slot(request, pinned), on_chunk=on_chunk)

        if not self._pin_slots_enabled():
            return self.backend.infer_stream_text(role_config, request, on_chunk=on_chunk)

        prompt = request.prompt or ""
        slot_id = self.router.get_slot_for_prompt(prompt, canonicalize=self.canonicalize)
        routed_request = self._request_with_slot(request, slot_id)
        return self.backend.infer_stream_text(role_config, routed_request, on_chunk=on_chunk)

    def get_hit_rate(self) -> float:
        """Get the current cache hit rate.

        Returns:
            Hit rate as a float (0.0 to 1.0).
        """
        stats = self.router.get_stats()
        return stats["hit_rate"]

    def get_stats(self) -> dict[str, float | int | list]:
        """Get combined statistics including per-slot details.

        Returns:
            Dictionary with router stats, backend stats, per-slot stats,
            and token savings percentage.
        """
        router_stats = self.router.get_stats()
        backend_stats = self.backend.get_cache_stats()

        return {
            # Router stats
            "router_total_routes": router_stats["total_routes"],
            "router_hit_rate": router_stats["hit_rate"],
            # Backend stats (actual cache performance)
            "backend_hit_rate": backend_stats.hit_rate / 100,
            "backend_token_savings": backend_stats.token_savings_rate / 100,
            "total_prompt_tokens": backend_stats.total_prompt_tokens,
            "cached_prompt_tokens": backend_stats.cached_prompt_tokens,
            # Per-slot stats (C4)
            "slot_stats": self.router.get_slot_stats(),
            # Convenience: token savings as a percentage (0-100)
            "token_savings_pct": backend_stats.token_savings_rate,
            # Bypass diagnostics for WS3A validation
            "frontdoor_repl_bypass_enabled": self._frontdoor_repl_bypass_enabled(),
            "frontdoor_repl_bypass_count": self.frontdoor_repl_bypass_count,
        }

# =============================================================================
# Utility Functions
# =============================================================================


def create_prefix_filter(
    patterns: list[str],
) -> Callable[[str], str]:
    """Create a custom prompt filter for domain-specific canonicalization.

    Args:
        patterns: List of regex patterns to normalize (replaced with [FILTERED]).

    Returns:
        Function that applies all filters to a prompt.
    """
    compiled = [(re.compile(p), "[FILTERED]") for p in patterns]

    def filter_prompt(prompt: str) -> str:
        result = prompt
        for pattern, replacement in compiled:
            result = pattern.sub(replacement, result)
        return result

    return filter_prompt
