"""The model cap on per-request context (STACKCHG-KVPOOL-20261003).

:8083 runs a UNIFIED KV pool of -c 393216 at -np 4 on a model trained to
262144. The v10 server caps every slot at n_ctx_train
(``tools/server/server-context.cpp:1316-1322``), so ``/props`` reports
``n_ctx 262144`` while the pool the four slots share is 393216. These tests pin
what the orchestrator's context limits must make of that:

* one request may use 262144, never 393216 (live, registry fallback, observe);
* the server is still read as UNIFIED, so ``shared_pool`` stays True and
  SharedKVPoolAdmission stays armed — the pre-fix inference compared the live
  n_ctx with the raw -c (262144 >= 393216 is False), read the server as SPLIT,
  reported a 1,048,576-token pool and turned pool admission OFF;
* the pool is the whole -c.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.backends.context_limits import (
    ContextLimitResolver,
    limit_from_registry,
    parse_props,
    registry_facts_by_port,
)

FROZEN_TREE = Path("/mnt/raid0/llm/llama.cpp")

TRAIN = 262144
POOL = 393216
REG = {"context_tokens": POOL, "slots": 4, "kv_unified": True, "ctx_max": TRAIN}
PROPS_CAPPED = {"default_generation_settings": {"n_ctx": TRAIN}, "total_slots": 4}


def _shape(lim):
    return (lim.per_request_n_ctx, lim.kv_unified, lim.shared_pool, lim.pool_tokens)


class TestLiveProps:
    def test_capped_unified_server_stays_unified_with_the_whole_pool(self):
        lim = parse_props("http://localhost:8083", PROPS_CAPPED, REG)
        assert _shape(lim) == (TRAIN, True, True, POOL)
        assert lim.request_cap == TRAIN and not lim.cap_binding

    def test_regression_raw_c_comparison_would_disarm_pool_admission(self):
        # The exact pre-fix arithmetic: n_ctx >= -c decides "unified".
        assert not (TRAIN >= POOL)
        lim = parse_props("u", PROPS_CAPPED, {**REG, "kv_unified": None})
        assert lim.kv_unified is True and lim.shared_pool is True
        assert lim.pool_tokens == POOL  # not 4 x 262144 = 1,048,576

    def test_split_server_at_the_same_declaration_is_still_split(self):
        lim = parse_props("u", {"default_generation_settings": {"n_ctx": POOL // 4},
                                "total_slots": 4}, REG)
        assert _shape(lim) == (POOL // 4, False, False, POOL)

    def test_not_yet_reloaded_196608_unified_server_reads_its_own_pool(self):
        # API reloaded onto the new priors before :8083 is relaunched: the live
        # server is uncapped (196608 < 262144), so its slot context IS its pool.
        lim = parse_props("u", {"default_generation_settings": {"n_ctx": 196608},
                                "total_slots": 4}, REG)
        assert _shape(lim) == (196608, True, True, 196608)

    def test_a_build_that_does_not_clamp_is_capped_by_config(self):
        lim = parse_props("u", {"default_generation_settings": {"n_ctx": POOL},
                                "total_slots": 4}, REG)
        assert _shape(lim) == (TRAIN, True, True, POOL)
        assert lim.cap_binding  # step-1 pre-dispatch refusal applies (the server would accept)

    def test_no_cap_known_is_the_pre_fix_behaviour(self):
        reg = {"context_tokens": 196608, "slots": 2, "kv_unified": None}
        uni = parse_props("u", {"default_generation_settings": {"n_ctx": 196608}, "total_slots": 2}, reg)
        spl = parse_props("u", {"default_generation_settings": {"n_ctx": 98304}, "total_slots": 2}, reg)
        assert _shape(uni) == (196608, True, True, 196608)
        assert _shape(spl) == (98304, False, False, 196608)

    def test_cap_above_every_slot_changes_nothing(self):
        # frontdoor :8070 shape: -c 262144 -np 4 split, model ctx_max 262144.
        reg = {"context_tokens": 262144, "slots": 4, "kv_unified": None, "ctx_max": TRAIN}
        lim = parse_props("u", {"default_generation_settings": {"n_ctx": 65536}, "total_slots": 4}, reg)
        assert _shape(lim) == (65536, False, False, 262144)

    def test_single_slot_server_is_untouched(self):
        reg = {"context_tokens": 262144, "slots": 1, "kv_unified": None, "ctx_max": TRAIN}
        lim = parse_props("u", {"default_generation_settings": {"n_ctx": 262144}, "total_slots": 1}, reg)
        assert (lim.per_request_n_ctx, lim.kv_unified, lim.shared_pool) == (262144, None, False)


class TestRegistryFallback:
    def test_declared_unified_pool_above_the_cap(self):
        assert _shape(limit_from_registry("u", REG)) == (TRAIN, True, True, POOL)

    def test_declared_unified_pool_below_the_cap_is_unchanged(self):
        lim = limit_from_registry("u", {**REG, "context_tokens": 196608})
        assert _shape(lim) == (196608, True, True, 196608)
        assert lim.pool_n_ctx is None

    def test_without_a_cap_the_whole_pool_is_one_request(self):
        lim = limit_from_registry("u", {**REG, "ctx_max": None})
        assert _shape(lim) == (POOL, True, True, POOL)


class TestResolver:
    def test_registry_path_of_the_resolver_applies_the_cap(self):
        r = ContextLimitResolver(live=False, registry_facts=lambda: {8083: REG}, role_urls=lambda: {})
        assert _shape(r.limit_for_url("http://localhost:8083")) == (TRAIN, True, True, POOL)

    def test_live_path_of_the_resolver_applies_the_cap(self):
        r = ContextLimitResolver(live=True, fetch_props=lambda u, t: PROPS_CAPPED,
                                 registry_facts=lambda: {8083: REG}, role_urls=lambda: {})
        assert _shape(r.limit_for_url("http://localhost:8083")) == (TRAIN, True, True, POOL)

    def test_observe_caps_a_reported_n_ctx_and_keeps_the_pool(self):
        r = ContextLimitResolver(live=False, registry_facts=lambda: {8083: REG}, role_urls=lambda: {})
        r.limit_for_url("http://localhost:8083")
        r.observe("http://localhost:8083", n_ctx=POOL)
        assert _shape(r.limit_for_url("http://localhost:8083")) == (TRAIN, True, True, POOL)

    def test_observe_below_the_cap_means_the_pool_is_that_size(self):
        r = ContextLimitResolver(live=False, registry_facts=lambda: {8083: REG}, role_urls=lambda: {})
        r.limit_for_url("http://localhost:8083")
        r.observe("http://localhost:8083", n_ctx=196608)
        lim = r.limit_for_url("http://localhost:8083")
        assert lim.source == "observed" and _shape(lim) == (196608, True, True, 196608)

    def test_observe_of_the_clamped_value_keeps_the_clamped_pool(self):
        r = ContextLimitResolver(live=True, fetch_props=lambda u, t: PROPS_CAPPED,
                                 registry_facts=lambda: {8083: REG}, role_urls=lambda: {})
        r.limit_for_url("http://localhost:8083")
        r.observe("http://localhost:8083", n_ctx=TRAIN)  # a 400 body from the clamped server
        assert _shape(r.limit_for_url("http://localhost:8083")) == (TRAIN, True, True, POOL)


class TestRegistryFacts:
    def test_compiled_priors_of_this_tree_carry_the_cap_for_8083(self):
        facts = registry_facts_by_port()
        if 8083 not in facts:
            pytest.skip("no :8083 role in this tree's compiled priors")
        cap = facts[8083]["ctx_max"]
        assert cap == TRAIN
        lim = limit_from_registry("http://localhost:8083", facts[8083])
        assert lim.per_request_n_ctx <= cap
        assert lim.pool_tokens == facts[8083]["context_tokens"]


class TestFrozenServerCapsSlotsAtTrainingContext:
    """Re-derive the server behaviour this module mirrors from the frozen source."""

    def test_server_clamps_slot_context_to_n_ctx_train(self):
        src = FROZEN_TREE / "tools/server/server-context.cpp"
        if not src.exists():
            pytest.skip("frozen llama.cpp tree not present on this host")
        text = src.read_text(errors="replace")
        assert re.search(r"int n_ctx_slot = llama_n_ctx_seq\(ctx_tgt\);\s*"
                         r"if \(n_ctx_slot > n_ctx_train\) \{", text)
        assert re.search(r"n_ctx_slot = n_ctx_train;", text)
        assert re.search(r"slot\.n_ctx\s*=\s*n_ctx_slot;", text)
        assert '{ "n_ctx",  meta->slot_n_ctx }' in text
