"""UFH14-B4: the --cache-ram derivation, pinned to measured :8083 entry costs.

Evidence for the 27B numbers: 489 ``prompt_save`` lines in logs/llama-server-8083.log
(MTP era, np 4 unified, q8_0 K/V) fit ``149.6 MiB + 38,960 B/token`` exactly, and
checkpoints log at 150.2-152.3 MiB. Design: /mnt/raid0/llm/tmp/ufh14-b4-ec/DESIGN.md.
"""

from __future__ import annotations

import pytest

from src.registry.prefix_cache_policy import (
    SERVER_DEFAULT_CACHE_RAM_MIB,
    EntryCost,
    PrefixCachePolicyError,
    PrefixWorkload,
    cache_reuse_for,
    numa_domain_cap_mib,
    recommend,
)

Q38_27B_MTP = EntryCost(memory_kind="hybrid", fixed_mib=149.6, per_token_bytes=38960, checkpoint_mib=152.0)


def test_entry_cost_matches_the_measured_save_sizes():
    no_ckpt = EntryCost(memory_kind="hybrid", fixed_mib=149.6, per_token_bytes=38960)
    # 8083 log: "saving prompt with length 276, total state size = 159.881 MiB"
    assert no_ckpt.entry_mib(276) == pytest.approx(159.88, abs=0.05)
    # 8083 log: "length 577, total state size = 171.065 MiB"
    assert no_ckpt.entry_mib(577) == pytest.approx(171.07, abs=0.05)


def test_checkpoints_dominate_short_entries_and_double_long_ones():
    # A 120-token prompt with one checkpoint logged at 303.8 MiB: the fixed parts dominate.
    assert Q38_27B_MTP.entry_mib(120) > 300
    # Agentic length: ~70-80 MiB per 1k tokens in the log, about 2x the raw KV rate.
    per_k = Q38_27B_MTP.entry_mib(70_000) / 70
    assert 50 < per_k < 80
    assert Q38_27B_MTP.checkpoints(10**7) == 32  # bounded by --ctx-checkpoints


def test_27b_today_keeps_65536():
    """np 4 unified, ~8 warm agentic sessions + 2 warmed shared prefixes at p90 86,988."""
    wl = PrefixWorkload(slots=4, kv_unified=True, warm_sessions=8, p90_prompt_tokens=86_988, shared_prefixes=2)
    cap = numa_domain_cap_mib(283 * 1024, resident_mib=(44 + 20) * 1024, tenants=1)
    rec = recommend("architect_critic", Q38_27B_MTP, wl, cap_mib=cap, current_mib=65536)
    assert rec.entries == 10
    assert rec.need_mib < 65536 * 1.0 + 1  # today's value already covers the derived need
    assert rec.verdict == "keep" and rec.cache_ram_mib == 65536
    assert rec.cache_reuse == 0  # hybrid
    assert any("--no-cache-idle-slots they stay in the pool" in n for n in rec.notes)


def test_more_sessions_raise_and_round_to_4gib():
    wl = PrefixWorkload(slots=4, kv_unified=True, warm_sessions=16, p90_prompt_tokens=86_988)
    rec = recommend("architect_critic", Q38_27B_MTP, wl, cap_mib=200_000, current_mib=65536)
    assert rec.verdict == "raise"
    assert rec.cache_ram_mib % 4096 == 0 and rec.cache_ram_mib >= rec.need_mib


def test_cap_binds_and_says_so():
    wl = PrefixWorkload(slots=4, kv_unified=True, warm_sessions=40, p90_prompt_tokens=150_000)
    rec = recommend("x", Q38_27B_MTP, wl, cap_mib=100_000, current_mib=65536)
    assert rec.cache_ram_mib <= 100_000
    assert any("exceeds the NUMA-domain cap" in n for n in rec.notes)


def test_split_kv_counts_idle_slot_copies():
    small = EntryCost(memory_kind="hybrid", fixed_mib=63, per_token_bytes=12_000, checkpoint_mib=63)
    split = PrefixWorkload(slots=4, kv_unified=False, warm_sessions=4, p90_prompt_tokens=1_013)
    unified = PrefixWorkload(slots=4, kv_unified=True, warm_sessions=4, p90_prompt_tokens=1_013)
    assert recommend("a", small, split, cap_mib=1e6).entries == 8
    assert recommend("a", small, unified, cap_mib=1e6).entries == 4


def test_overprovisioned_short_prompt_server_is_kept_not_churned():
    """Frontdoor-like: tiny prompts, need ~ a few GiB; 32768 is inside the keep band."""
    small = EntryCost(memory_kind="hybrid", fixed_mib=63, per_token_bytes=12_000, checkpoint_mib=63)
    wl = PrefixWorkload(slots=4, kv_unified=False, warm_sessions=16, p90_prompt_tokens=1_013)
    rec = recommend("frontdoor", small, wl, cap_mib=40_000, current_mib=32768)
    assert rec.need_mib < 8192
    assert rec.verdict == "keep" and rec.cache_ram_mib == 32768


def test_absurd_overprovision_is_lowered():
    small = EntryCost(memory_kind="hybrid", fixed_mib=63, per_token_bytes=12_000, checkpoint_mib=63)
    wl = PrefixWorkload(slots=1, kv_unified=False, warm_sessions=2, p90_prompt_tokens=1_000)
    rec = recommend("x", small, wl, cap_mib=20_000, current_mib=262_144)
    assert rec.verdict == "lower" and rec.cache_ram_mib == SERVER_DEFAULT_CACHE_RAM_MIB


def test_unset_server_gets_a_value():
    wl = PrefixWorkload(slots=1, kv_unified=False, warm_sessions=4, p90_prompt_tokens=159_354)
    flash = EntryCost(memory_kind="hybrid", fixed_mib=115, per_token_bytes=27_000, checkpoint_mib=115)
    rec = recommend("architect_general", flash, wl, cap_mib=1e6, current_mib=None)
    assert rec.verdict == "set" and rec.flags()[:2] == ["--cache-ram", str(rec.cache_ram_mib)]


def test_huge_entry_is_flagged_as_never_cached():
    wl = PrefixWorkload(slots=1, kv_unified=False, warm_sessions=1, p90_prompt_tokens=250_000)
    rec = recommend("x", Q38_27B_MTP, wl, cap_mib=8192, current_mib=8192)
    assert any("never cached" in n for n in rec.notes)


def test_no_sessions_on_a_single_slot_server_disables():
    wl = PrefixWorkload(slots=1, kv_unified=False, warm_sessions=0, p90_prompt_tokens=500)
    rec = recommend("embedder", Q38_27B_MTP, wl, cap_mib=1e6, current_mib=8192)
    assert rec.verdict == "disable" and rec.cache_ram_mib == 0


def test_unified_multislot_server_without_sessions_is_refused():
    wl = PrefixWorkload(slots=4, kv_unified=True, warm_sessions=0, p90_prompt_tokens=500)
    with pytest.raises(PrefixCachePolicyError, match="ONLY place"):
        recommend("x", Q38_27B_MTP, wl, cap_mib=1e6)


@pytest.mark.parametrize("kind,mm,expected", [
    ("attention", False, 256),
    ("attention", True, 0),
    ("swa", False, 0),
    ("hybrid", False, 0),
    ("recurrent", False, 0),
])
def test_cache_reuse_only_for_pure_attention_text(kind, mm, expected):
    cost = EntryCost(memory_kind=kind, fixed_mib=0, per_token_bytes=1000)
    wl = PrefixWorkload(slots=1, kv_unified=False, warm_sessions=1, p90_prompt_tokens=10, multimodal=mm)
    assert cache_reuse_for(cost, wl) == expected


def test_flags_include_cache_reuse_only_when_nonzero():
    cost = EntryCost(memory_kind="attention", fixed_mib=0, per_token_bytes=100_000)
    wl = PrefixWorkload(slots=2, kv_unified=False, warm_sessions=2, p90_prompt_tokens=8000)
    assert "--cache-reuse" in recommend("x", cost, wl, cap_mib=1e6).flags()
    assert "--cache-reuse" not in recommend("y", Q38_27B_MTP, wl, cap_mib=1e6).flags()


@pytest.mark.parametrize("kw", [
    dict(memory_kind="hybrid", fixed_mib=-1, per_token_bytes=1),
    dict(memory_kind="hybrid", fixed_mib=1, per_token_bytes=0),
    dict(memory_kind="hybrid", fixed_mib=1, per_token_bytes=1, checkpoint_every_tokens=0),
])
def test_non_physical_costs_are_refused(kw):
    with pytest.raises(PrefixCachePolicyError):
        EntryCost(**kw)


def test_numa_cap_arithmetic():
    assert numa_domain_cap_mib(1000, resident_mib=350, reserve_fraction=0.15, tenants=2) == pytest.approx(250)
    assert numa_domain_cap_mib(1000, resident_mib=2000) == 0.0
    with pytest.raises(PrefixCachePolicyError):
        numa_domain_cap_mib(1000, resident_mib=0, tenants=0)
