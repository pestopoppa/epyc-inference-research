"""Tests for delegation_cache.py — content-hash keyed delegation result reuse."""

import time

from src.delegation_cache import DelegationCache


def test_make_key_deterministic():
    c = DelegationCache()
    k1 = c.make_key("solve x^2 = 4", "coder")
    k2 = c.make_key("solve x^2 = 4", "coder")
    assert k1 == k2


def test_make_key_differs_by_role():
    c = DelegationCache()
    k1 = c.make_key("solve x^2 = 4", "coder")
    k2 = c.make_key("solve x^2 = 4", "architect")
    assert k1 != k2


def test_make_key_normalizes():
    c = DelegationCache()
    k1 = c.make_key("  Solve X^2 = 4  ", "coder")
    k2 = c.make_key("solve x^2 = 4", "coder")
    assert k1 == k2


def test_make_key_namespaces_consult_skill_without_changing_legacy_delegate_key():
    c = DelegationCache()
    legacy = c.make_key("brief", "architect_general")
    explicit_legacy = c.make_key(
        "brief",
        "architect_general",
        interaction_type="delegate",
        policy_version="1.0",
    )
    consult = c.make_key(
        "brief",
        "architect_general",
        interaction_type="consult",
        skill="review_before_commit",
        schema_hash="abc123",
    )

    assert explicit_legacy == legacy
    assert consult != legacy


def test_put_and_get():
    c = DelegationCache()
    key = c.make_key("brief", "coder")
    c.put(key, "result report", "coder", tokens_used=100)
    entry = c.get(key)
    assert entry is not None
    assert entry.report == "result report"
    assert entry.tokens_used == 100


def test_get_miss():
    c = DelegationCache()
    assert c.get("nonexistent") is None


def test_ttl_expiration():
    c = DelegationCache(ttl_seconds=0.01)
    key = c.make_key("brief", "coder")
    c.put(key, "result", "coder")
    time.sleep(0.02)
    assert c.get(key) is None


def test_max_entries_eviction():
    c = DelegationCache(max_entries=3)
    for i in range(4):
        key = c.make_key(f"brief_{i}", "coder")
        c.put(key, f"result_{i}", "coder")
    # Oldest should be evicted
    assert c.stats()["entries"] == 3


def test_empty_report_not_cached():
    c = DelegationCache()
    key = c.make_key("brief", "coder")
    c.put(key, "", "coder")
    assert c.get(key) is None
    c.put(key, "   ", "coder")
    assert c.get(key) is None


def test_stats():
    c = DelegationCache()
    key = c.make_key("brief", "coder")
    c.put(key, "result", "coder")
    c.get(key)  # hit
    c.get("missing")  # miss
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1


def test_clear():
    c = DelegationCache()
    key = c.make_key("brief", "coder")
    c.put(key, "result", "coder")
    c.clear()
    assert c.get(key) is None
    assert c.stats()["entries"] == 0


def test_report_handle_preserved():
    c = DelegationCache()
    key = c.make_key("brief", "coder")
    handle = {"id": "coder-123-abc", "path": "/tmp/x.txt", "chars": "500", "sha16": "abc"}
    c.put(key, "result", "coder", report_handle=handle)
    entry = c.get(key)
    assert entry is not None
    assert entry.report_handle == handle


def test_full_report_is_preserved_separately_from_loop_report():
    c = DelegationCache()
    key = c.make_key("brief", "coder")
    c.put(key, "compact handle and summary", "coder", full_report="original full report")

    entry = c.get(key, require_full_report=True)

    assert entry is not None
    assert entry.report == "compact handle and summary"
    assert entry.full_report == "original full report"


def test_full_report_required_cache_lookup_misses_legacy_entry():
    c = DelegationCache()
    key = c.make_key("brief", "coder")
    c.put(key, "legacy compact summary", "coder")

    assert c.get(key, require_full_report=True) is None
    # General lookups retain the old entry/API behavior.
    assert c.get(key).report == "legacy compact summary"
