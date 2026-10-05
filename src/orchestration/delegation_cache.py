"""Delegation result cache — content-hash keyed reuse of specialist reports.

When the architect delegates the same brief to the same target role, skip
specialist execution and return the cached compressed report. This saves
the full specialist loop (REPL turns, tool calls, LLM inference).

Key: SHA-256(brief_normalized[:200] + "|" + delegate_to)
Value: compressed report text, optional original report, and metadata

Design:
- In-memory dict (no disk persistence — delegation results are session-scoped)
- TTL: 1 hour default (prompts may change between sessions)
- Max entries: 200 (bounded memory, LRU eviction)
- Cache is per-process (not shared across workers)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600  # 1 hour
DEFAULT_MAX_ENTRIES = 200


@dataclass
class DelegationCacheEntry:
    """A cached delegation result."""

    key: str
    report: str                # compressed report text
    delegate_to: str           # target role
    created_at: float = 0.0
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    tokens_used: int = 0       # tokens consumed in original execution
    report_handle: dict[str, str] | None = None  # if a handle was stored
    full_report: str | None = None  # original report for user-facing returns

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class DelegationCache:
    """In-memory content-hash keyed cache for delegation results.

    Thread-safe via GIL (dict operations are atomic in CPython).
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._store: dict[str, DelegationCacheEntry] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(
        brief: str,
        delegate_to: str,
        *,
        interaction_type: str = "delegate",
        skill: str = "",
        schema_hash: str = "",
        policy_version: str = "1.0",
    ) -> str:
        """Generate cache key from brief, role, and interaction namespace.

        Normalizes the brief (lowercase, strip, truncate to 200 chars)
        to match the semantic dedup key in the delegation loop.
        """
        normalized = brief.strip().lower()[:200]
        if (
            interaction_type == "delegate"
            and not skill
            and not schema_hash
            and policy_version == "1.0"
        ):
            payload = f"{normalized}|{delegate_to}"
        else:
            payload = (
                f"{interaction_type}|{skill}|{normalized}|{delegate_to}|"
                f"{schema_hash}|{policy_version}"
            )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(
        self,
        key: str,
        *,
        require_full_report: bool = False,
    ) -> DelegationCacheEntry | None:
        """Look up a cached delegation result.

        Returns None on miss, expiry, or when a required original report is absent.
        """
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired:
            del self._store[key]
            self._misses += 1
            return None
        if require_full_report and not entry.full_report:
            self._misses += 1
            return None
        self._hits += 1
        return entry

    def put(
        self,
        key: str,
        report: str,
        delegate_to: str,
        tokens_used: int = 0,
        report_handle: dict[str, str] | None = None,
        full_report: str | None = None,
    ) -> None:
        """Store a delegation result."""
        if not report or not report.strip():
            return

        # Evict expired entries first
        self._evict_expired()

        # LRU eviction if at capacity (evict oldest)
        if len(self._store) >= self._max_entries:
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]

        self._store[key] = DelegationCacheEntry(
            key=key,
            report=report,
            delegate_to=delegate_to,
            created_at=time.time(),
            ttl_seconds=self._ttl,
            tokens_used=tokens_used,
            report_handle=report_handle,
            full_report=full_report,
        )

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        self._evict_expired()
        return {
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
        }

    def clear(self) -> None:
        """Clear all entries."""
        self._store.clear()


# Module-level singleton (lazy init)
_delegation_cache: DelegationCache | None = None


def get_delegation_cache() -> DelegationCache:
    """Get or create the global delegation cache singleton."""
    global _delegation_cache
    if _delegation_cache is None:
        _delegation_cache = DelegationCache()
    return _delegation_cache
