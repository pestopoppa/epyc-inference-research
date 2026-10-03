"""Prefix-cache policy per llama-server: derive ``--cache-ram`` and its companions.

UFH14-B4 (2026-10-03). Design: ``/mnt/raid0/llm/tmp/ufh14-b4-ec/DESIGN.md``.

This module is PURE: it turns two declared facts into launch-flag recommendations and
never reads the registry, the topology or a live server itself. The compiler wiring
(``stack_topology.yaml -> prefix_cache_selection.<server>``, a DRAFT-SEL-1-shaped rule)
is a stack change and is proposed in the design doc, not done here.

The two facts
-------------
* A **model fact** (:class:`EntryCost`): what one llama-server prompt-cache entry costs
  for that GGUF + K/V type + drafter. llama.cpp v10 stores each entry as a full
  serialized sequence state (no prefix sharing, ``server-task.cpp:1677-1757``):

      entry(L) = fixed + per_token * L + checkpoints(L) * checkpoint

  ``fixed`` is the recurrent (Gated-DeltaNet) state of a hybrid model plus any fixed
  draft state; ``checkpoint`` is one context checkpoint (a partial, recurrent-only
  state, ``server-context.cpp:2365-2416``). Measured on Qwen3.8-27B-Q8_0, MTP era:
  149.6 MiB + 38,960 B/token, checkpoints ~152 MiB (8083 log, exact fit over 489 saves).
* A **workload fact** (:class:`PrefixWorkload`): how many conversations must stay warm
  between their turns on that server, and how long they are (p90 prompt tokens).

The derivation
--------------
``entries = warm_sessions + shared_prefixes + (slots if split KV else 0)``. Under
``--kv-unified`` an idle slot is MOVED into the cache after every launch
(``server-context.cpp:2469-2484``), so the sessions already count it; with split KV
idle slots keep their KV and the cache additionally holds a COPY of each.

``need = entries * entry(p90) * headroom``; the recommendation is ``need`` rounded up
to 4 GiB, floored at the server default (8192 MiB) and capped at the server's share
of its NUMA domain (``--cache-ram`` is heap under the process's memory policy; a
``bind:N`` GPU-lane server competes for ONE node).

A current value inside ``[need, max(cap, 2 * need)]`` is KEPT: over-provisioning a
host-RAM cap that is allocated on demand costs nothing until it is used, and changing
a launch flag costs a relaunch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

MIB = 1024 * 1024

#: llama.cpp v10 server defaults (common/common.h:630-635).
SERVER_DEFAULT_CACHE_RAM_MIB = 8192
SERVER_DEFAULT_CTX_CHECKPOINTS = 32
SERVER_DEFAULT_CHECKPOINT_MIN_STEP = 8192

#: Round recommendations to this granularity so they do not churn on noise.
ROUND_MIB = 4096

MemoryKind = Literal["attention", "swa", "hybrid", "recurrent"]
Verdict = Literal["keep", "raise", "lower", "set", "disable"]


class PrefixCachePolicyError(ValueError):
    """A declaration that cannot produce a sane prefix-cache configuration."""


@dataclass(frozen=True)
class EntryCost:
    """What one prompt-cache entry costs for one model (GGUF + K/V type + drafter)."""

    memory_kind: MemoryKind
    fixed_mib: float
    per_token_bytes: float
    checkpoint_mib: float = 0.0
    #: Observed spacing of checkpoints in long agentic prompts. They are created at
    #: user-message starts no closer than --checkpoint-min-step (8192) plus two near the
    #: prompt end; the 27B's cache held ~1 per 12k tokens (+1.5) on agentic traffic.
    checkpoint_every_tokens: float = 12000.0
    checkpoint_base: float = 1.5
    max_checkpoints: int = SERVER_DEFAULT_CTX_CHECKPOINTS

    def __post_init__(self) -> None:
        if self.fixed_mib < 0 or self.per_token_bytes <= 0 or self.checkpoint_mib < 0:
            raise PrefixCachePolicyError(f"non-physical entry cost: {self}")
        if self.checkpoint_every_tokens <= 0 or self.max_checkpoints < 0:
            raise PrefixCachePolicyError(f"bad checkpoint model: {self}")

    def checkpoints(self, tokens: int) -> float:
        if self.checkpoint_mib == 0 or self.max_checkpoints == 0:
            return 0.0
        return min(float(self.max_checkpoints), self.checkpoint_base + tokens / self.checkpoint_every_tokens)

    def entry_mib(self, tokens: int) -> float:
        if tokens < 0:
            raise PrefixCachePolicyError("tokens must be >= 0")
        return self.fixed_mib + tokens * self.per_token_bytes / MIB + self.checkpoints(tokens) * self.checkpoint_mib

    def tokens_per_gib(self, tokens: int) -> float:
        """Cached tokens held per GiB at entries of this length (an efficiency read-out)."""
        return 1024.0 * tokens / self.entry_mib(tokens) if tokens > 0 else 0.0


@dataclass(frozen=True)
class PrefixWorkload:
    """How a server is used, as far as the prefix cache is concerned."""

    slots: int
    kv_unified: bool
    warm_sessions: int
    p90_prompt_tokens: int
    shared_prefixes: int = 0
    headroom: float = 1.25
    multimodal: bool = False

    def __post_init__(self) -> None:
        if self.slots < 1 or self.warm_sessions < 0 or self.shared_prefixes < 0:
            raise PrefixCachePolicyError(f"bad workload: {self}")
        if self.p90_prompt_tokens < 0 or self.headroom < 1.0:
            raise PrefixCachePolicyError(f"bad workload: {self}")


@dataclass(frozen=True)
class Recommendation:
    server: str
    cache_ram_mib: int
    verdict: Verdict
    need_mib: float
    cap_mib: float
    entries: int
    entry_mib_p90: float
    cache_reuse: int
    cache_idle_slots: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def flags(self) -> list[str]:
        """The launch flags this recommendation implies (cache-reuse only when non-zero)."""
        out = ["--cache-ram", str(self.cache_ram_mib)]
        if self.cache_reuse > 0:
            out += ["--cache-reuse", str(self.cache_reuse)]
        if not self.cache_idle_slots:
            out.append("--no-cache-idle-slots")
        return out


def _round_up(mib: float) -> int:
    return int(math.ceil(mib / ROUND_MIB) * ROUND_MIB)


def cache_reuse_for(cost: EntryCost, workload: PrefixWorkload, chunk: int = 256) -> int:
    """``--cache-reuse`` (KV-shift chunk reuse) is only sound for pure attention.

    v10 accepts it on hybrids because ``llama_memory_hybrid::get_can_shift`` reports the
    attention part (``src/llama-memory-hybrid.cpp:133-136``), but shifting attention KV
    leaves the recurrent state describing other tokens; SWA caches lose the shifted
    window; multimodal disables it (``server-context.cpp:1286-1302``).
    """
    if cost.memory_kind != "attention" or workload.multimodal:
        return 0
    return chunk


def recommend(
    server: str,
    cost: EntryCost,
    workload: PrefixWorkload,
    *,
    cap_mib: float,
    current_mib: int | None = None,
) -> Recommendation:
    """Derive the ``--cache-ram`` (and companion) recommendation for one server."""
    if cap_mib <= 0:
        raise PrefixCachePolicyError(f"{server}: NUMA-domain cap must be positive")
    notes: list[str] = []
    resumed = workload.warm_sessions + workload.shared_prefixes
    # Split KV also stores a copy of each idle slot — worth paying only if anything resumes.
    entries = resumed + (0 if workload.kv_unified or resumed == 0 else workload.slots)
    entry_p90 = cost.entry_mib(workload.p90_prompt_tokens)
    need = entries * entry_p90 * workload.headroom

    if entries == 0:
        if workload.kv_unified and workload.slots > 1:
            raise PrefixCachePolicyError(
                f"{server}: a unified multi-slot server with no warm sessions declared — under "
                "--kv-unified the prompt cache is the ONLY place an idle conversation survives"
            )
        return Recommendation(server, 0, "disable" if current_mib not in (None, 0) else "keep",
                              0.0, cap_mib, 0, entry_p90, 0, False,
                              ("no conversation is ever resumed here; the cache only costs copies",))

    target = max(SERVER_DEFAULT_CACHE_RAM_MIB, _round_up(need))
    if target > cap_mib:
        notes.append(
            f"need {need:.0f} MiB exceeds the NUMA-domain cap {cap_mib:.0f} MiB: capped; "
            "expect evictions, or cut entry cost (fewer checkpoints, shorter retained prompts)"
        )
        target = int(cap_mib // ROUND_MIB * ROUND_MIB) or int(cap_mib)

    if entry_p90 > target:
        notes.append(
            f"a p90 entry ({entry_p90:.0f} MiB) exceeds the cache ({target} MiB): such prompts "
            "are never cached ('exceeds cache size limit, skipping')"
        )

    if current_mib is None:
        verdict: Verdict = "set"
        chosen = target
    elif current_mib == 0:
        verdict, chosen = "raise", target
    elif current_mib < need:
        verdict, chosen = "raise", target
    elif current_mib > max(cap_mib, 2.0 * need, SERVER_DEFAULT_CACHE_RAM_MIB):
        verdict, chosen = "lower", target
    else:
        verdict, chosen = "keep", int(current_mib)

    if workload.kv_unified and workload.slots > 1:
        notes.append("kv_unified: --cache-idle-slots must stay on (idle slots survive only in the cache)")

    return Recommendation(
        server=server,
        cache_ram_mib=chosen,
        verdict=verdict,
        need_mib=need,
        cap_mib=cap_mib,
        entries=entries,
        entry_mib_p90=entry_p90,
        cache_reuse=cache_reuse_for(cost, workload),
        cache_idle_slots=True,
        notes=tuple(notes),
    )


def numa_domain_cap_mib(
    domain_total_mib: float,
    *,
    resident_mib: float,
    reserve_fraction: float = 0.15,
    tenants: int = 1,
) -> float:
    """A server's share of its NUMA domain for prompt cache.

    ``resident_mib`` is everything that must stay resident in the domain regardless
    (locked weights, KV pools, other tenants' anon memory); ``reserve_fraction`` of the
    domain is left for the kernel and page cache; the rest is split over ``tenants``
    prompt-cache users of that domain.
    """
    if domain_total_mib <= 0 or tenants < 1 or not 0 <= reserve_fraction < 1:
        raise PrefixCachePolicyError("bad NUMA domain declaration")
    free = domain_total_mib * (1.0 - reserve_fraction) - resident_mib
    return max(0.0, free / tenants)
