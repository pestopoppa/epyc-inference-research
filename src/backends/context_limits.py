"""Per-role REAL per-request context limits, read from the live llama-server.

Why this exists: the orchestrator never asked a server how much context a
request may use. Routing used a 20,000-character threshold, compaction used a
registry accessor that does not exist (so it always fell back to 32768), and
admission counted slots only. The server's own answer is ``GET /props``:

* ``default_generation_settings.n_ctx`` — the per-slot context, i.e. the largest
  request (prompt + generation) one request may use
  (``tools/server/server-context.cpp:4878-4881``, value ``slot_n_ctx`` =
  ``llama_n_ctx_seq``: ``-c / -np`` under split KV, ``-c`` under unified KV —
  ``src/llama-context.cpp:289-302``);
* ``total_slots`` — ``-np`` (:4888);
* ``kv_unified`` — NOT exposed by the v10 server. Read if a future build adds it,
  else taken from the registry declaration, else inferred: a multi-slot server
  whose per-slot n_ctx equals the registry's whole ``-c`` must be unified.

Fallback order per URL: live ``/props`` (cached, TTL) → registry/stack priors
(``runtime.cache.context_tokens`` / ``slots_by_port`` / ``kv_unified``) → None.

Per-request cap (2026-10-03, :8083 KV-pool decision step 1): under unified KV the
server lets ONE request use the whole ``-c``. The orchestrator caps
``per_request_n_ctx`` at the model's ``ctx_max`` from the compiled stack priors
(``roles.<role>.model.ctx_max``, sourced from the research registry). Nothing is
hardcoded: today ``min(196608, 262144) = 196608``; after a relaunch at a larger
``-c`` the same rule yields 262144. ``server_n_ctx`` keeps the server's own slot
n_ctx.

THE v10 SERVER CAPS TOO (STACKCHG-KVPOOL-20261003). Above ``n_ctx_train`` the
server does not only warn: it clamps every slot, ``n_ctx_slot = min(n_ctx_seq,
n_ctx_train)`` (llama.cpp ffc1bac82 ``tools/server/server-context.cpp:1316-1322``,
WARN "the slot context (393216) exceeds the training context of the model
(262144) - capping"), and ``/props`` reports the CLAMPED 262144. So for a capping
server the slot n_ctx no longer equals ``-c`` under unified KV, and two things
must not be read from it:

* the unified/split inference — comparing the live n_ctx with the raw ``-c``
  (262144 >= 393216 is False) reads the server as SPLIT, which turns
  ``shared_pool`` off and DISARMS SharedKVPoolAdmission for :8083. It compares
  against the midpoint of the two CAPPED expectations instead;
* the pool — ``pool_n_ctx`` carries the declared ``-c`` (393216) when the server
  has clamped its slots, so ``pool_tokens`` is the real pool, not 262144.

A build that does not clamp (``/props`` n_ctx 393216) keeps the step-1 path:
``per_request_n_ctx`` 262144 from the config cap, ``cap_binding`` True.
There is no invented default here: a caller that gets None decides (and logs)
its own degraded behaviour.

Set ``ORCHESTRATOR_CONTEXT_LIMITS_LIVE=off`` to disable the live read (the test
suite does, so unit tests never touch the live stack).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

log = logging.getLogger(__name__)

LIVE_ENV = "ORCHESTRATOR_CONTEXT_LIMITS_LIVE"
TTL_ENV = "ORCHESTRATOR_CONTEXT_LIMITS_TTL_S"
DEFAULT_TTL_S = 60.0
FAILURE_TTL_S = 10.0
DEFAULT_PROPS_TIMEOUT_S = 1.0
DEFAULT_OCCUPANCY_TTL_S = 1.5

# Token estimation without a tokenizer round-trip. 4 chars/token is the
# codebase-wide rough ratio (chat_utils._estimate_tokens, LlamaTokenizer's
# fallback); for FIT decisions a conservative 3 chars/token over-estimates
# English/code prompts, so a "fits" verdict errs toward rerouting/compacting.
ROUGH_CHARS_PER_TOKEN = 4.0
CONSERVATIVE_CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str | None, *, chars_per_token: float = ROUGH_CHARS_PER_TOKEN) -> int:
    """Ceil-estimate the token count of ``text`` from its length."""
    if not text:
        return 0
    return int(math.ceil(len(text) / max(0.5, float(chars_per_token))))


def estimate_tokens_conservative(text: str | None) -> int:
    return estimate_tokens(text, chars_per_token=CONSERVATIVE_CHARS_PER_TOKEN)


@dataclass(frozen=True)
class ContextLimit:
    """What one server lets one request use."""

    url: str
    per_request_n_ctx: int
    total_slots: int | None
    kv_unified: bool | None
    source: str  # "live_props" | "registry" | "observed"
    registry_context_tokens: int | None = None
    # The server's own per-slot n_ctx (n_ctx_seq) before the model cap; None
    # means "same as per_request_n_ctx" (no cap applied / not known).
    server_n_ctx: int | None = None
    # The model's trained context (``model.ctx_max``) when config declares it.
    request_cap: int | None = None
    # The whole unified pool (-c) when the SERVER clamped its slots below it
    # (n_ctx_slot = min(-c, n_ctx_train)); None = the slot n_ctx is the pool.
    pool_n_ctx: int | None = None

    @property
    def slot_n_ctx(self) -> int:
        """The server's per-slot n_ctx."""
        return int(self.server_n_ctx or self.per_request_n_ctx)

    @property
    def cap_binding(self) -> bool:
        """True when the model cap, not the server, sets the per-request limit."""
        return self.per_request_n_ctx < self.slot_n_ctx

    @property
    def shared_pool(self) -> bool:
        """True when concurrent requests draw on ONE KV pool (unified, >1 slot)."""
        return bool(self.kv_unified) and (self.total_slots or 1) > 1

    @property
    def pool_tokens(self) -> int:
        """Total KV cells requests on this server compete for."""
        if self.kv_unified:
            return max(self.slot_n_ctx, self.pool_n_ctx or 0)
        return self.slot_n_ctx * max(1, self.total_slots or 1)

    def fits(self, prompt_tokens: int, max_new_tokens: int = 0) -> bool:
        """True when the request fits one slot on its own.

        The server rejects ``n_prompt >= n_ctx`` (server-context.cpp:3303) and,
        with context shift disabled, stops generation at ``n_ctx - 1``.
        """
        return int(prompt_tokens) + max(0, int(max_new_tokens)) < self.per_request_n_ctx

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "per_request_n_ctx": self.per_request_n_ctx,
            "total_slots": self.total_slots,
            "kv_unified": self.kv_unified,
            "shared_pool": self.shared_pool,
            "pool_tokens": self.pool_tokens,
            "server_n_ctx": self.slot_n_ctx,
            "request_cap": self.request_cap,
            "source": self.source,
        }


def capped_n_ctx(n_ctx: int, request_cap: int | None) -> int:
    """``min(server n_ctx, model ctx_max)``; the cap only ever lowers the limit."""
    cap = _positive_int(request_cap)
    return min(int(n_ctx), cap) if cap else int(n_ctx)


def split_urls(url_value: str | None) -> list[str]:
    """``"full:http://h:8070,http://h:8080"`` → ``["http://h:8070", "http://h:8080"]``."""
    if not url_value:
        return []
    out: list[str] = []
    for part in str(url_value).split(","):
        part = part.strip()
        if part.startswith("full:"):
            part = part[len("full:"):]
        if part:
            out.append(part.rstrip("/"))
    return out


def _port(url: str) -> int | None:
    try:
        return urlparse(url).port
    except ValueError:
        return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def parse_props(
    url: str,
    props: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> ContextLimit | None:
    """Build a ContextLimit from a ``/props`` body (+ optional registry facts)."""
    settings = props.get("default_generation_settings")
    n_ctx = _positive_int(settings.get("n_ctx")) if isinstance(settings, dict) else None
    if n_ctx is None:
        n_ctx = _positive_int(props.get("n_ctx"))
    if n_ctx is None:
        return None
    total_slots = _positive_int(props.get("total_slots"))
    reg_ctx = _positive_int(registry.get("context_tokens")) if registry else None
    request_cap = _positive_int(registry.get("ctx_max")) if registry else None
    kv_unified: bool | None
    if isinstance(props.get("kv_unified"), bool):
        kv_unified = props["kv_unified"]
    elif total_slots == 1:
        kv_unified = None  # no sharing possible either way
    elif total_slots and reg_ctx:
        # Live evidence beats the declaration (a declared-but-not-yet-reloaded
        # server is still split): split KV gives each slot -c/np, unified gives
        # each slot all of -c — and a capping server clamps BOTH at n_ctx_train.
        # Decide on the midpoint of the two CAPPED expectations. The raw-(-c)
        # comparison read a clamped unified server (262144 < 393216) as split and
        # disarmed SharedKVPoolAdmission (STACKCHG-KVPOOL-20261003).
        unified_slot = capped_n_ctx(reg_ctx, request_cap)
        split_slot = capped_n_ctx(max(1, reg_ctx // total_slots), request_cap)
        if unified_slot > split_slot:
            kv_unified = 2 * n_ctx > unified_slot + split_slot
        elif registry and isinstance(registry.get("kv_unified"), bool):
            kv_unified = registry["kv_unified"]
        else:
            kv_unified = None  # both layouts give the same slot context
    elif registry and isinstance(registry.get("kv_unified"), bool):
        kv_unified = registry["kv_unified"]
    else:
        kv_unified = None
    # A unified server whose slot n_ctx sits AT the cap below the declared -c has
    # clamped its slots; the pool is still the declared -c.
    clamped_pool = (
        reg_ctx
        if kv_unified and request_cap and n_ctx >= request_cap and reg_ctx and reg_ctx > n_ctx
        else None
    )
    return ContextLimit(
        url=url,
        per_request_n_ctx=capped_n_ctx(n_ctx, request_cap),
        total_slots=total_slots,
        kv_unified=kv_unified,
        source="live_props",
        registry_context_tokens=reg_ctx,
        server_n_ctx=n_ctx,
        request_cap=request_cap,
        pool_n_ctx=clamped_pool,
    )


def registry_facts_by_port(priors_path: Path | None = None) -> dict[int, dict[str, Any]]:
    """Per-port ``{context_tokens, slots, kv_unified, ctx_max}`` from the compiled stack priors.

    ``ctx_max`` is the served model's trained context (``model.ctx_max``), the
    per-request cap; None when the record does not declare it.
    """
    try:
        from src.registry.stack_priors import (
            DEFAULT_OUTPUT,
            live_stack_role_records,
            stack_prior_launch,
            stack_prior_serving,
            stack_prior_serving_ports,
        )
    except Exception:  # pragma: no cover - import guard
        return {}
    facts: dict[int, dict[str, Any]] = {}
    for record in live_stack_role_records(priors_path or DEFAULT_OUTPUT).values():
        serving = stack_prior_serving(record)
        runtime = stack_prior_launch(record).get("runtime")
        cache = runtime.get("cache") if isinstance(runtime, dict) else None
        if not isinstance(cache, dict):
            continue
        context_tokens = _positive_int(cache.get("context_tokens"))
        if context_tokens is None:
            continue
        role_slots = _positive_int(cache.get("slots")) or _positive_int(serving.get("slots"))
        kv_unified = cache.get("kv_unified")
        if not isinstance(kv_unified, bool):
            kv_unified = serving.get("kv_unified") if isinstance(serving.get("kv_unified"), bool) else None
        model = record.get("model") if isinstance(record.get("model"), dict) else {}
        ctx_max = _positive_int(model.get("ctx_max"))
        by_port = cache.get("slots_by_port") if isinstance(cache.get("slots_by_port"), dict) else {}
        ports = set(stack_prior_serving_ports(serving))
        for raw in by_port:
            p = _positive_int(raw)
            if p:
                ports.add(p)
        for port in ports:
            slots = _positive_int(by_port.get(port)) or _positive_int(by_port.get(str(port))) or role_slots
            facts.setdefault(
                port,
                {"context_tokens": context_tokens, "slots": slots, "kv_unified": kv_unified,
                 "ctx_max": ctx_max},
            )
    return facts


def registry_role_urls(priors_path: Path | None = None) -> dict[str, list[str]]:
    """Role → serving URLs from the compiled stack priors."""
    try:
        from src.registry.stack_priors import DEFAULT_OUTPUT, live_stack_serving_url_values
    except Exception:  # pragma: no cover - import guard
        return {}
    return {
        role: split_urls(value)
        for role, value in live_stack_serving_url_values(priors_path or DEFAULT_OUTPUT).items()
    }


def limit_from_registry(url: str, facts: dict[str, Any] | None) -> ContextLimit | None:
    if not facts:
        return None
    context_tokens = _positive_int(facts.get("context_tokens"))
    if context_tokens is None:
        return None
    slots = _positive_int(facts.get("slots")) or 1
    kv_unified = facts.get("kv_unified")
    # The launcher always passes -np explicitly, which makes the server's
    # default (split) stand unless kv_unified is declared (kvu PACKAGE §1).
    unified = bool(kv_unified) if isinstance(kv_unified, bool) else False
    per_slot = context_tokens if unified else max(1, context_tokens // slots)
    request_cap = _positive_int(facts.get("ctx_max"))
    return ContextLimit(
        url=url,
        per_request_n_ctx=capped_n_ctx(per_slot, request_cap),
        total_slots=slots,
        kv_unified=unified if slots > 1 else None,
        source="registry",
        registry_context_tokens=context_tokens,
        server_n_ctx=per_slot,
        request_cap=request_cap,
    )


@dataclass(frozen=True)
class SlotState:
    """One slot as ``GET /slots`` reports it (server-context.cpp:699-722)."""

    slot_id: int
    n_ctx: int | None
    is_processing: bool
    n_prompt_tokens: int  # prompt.tokens.size(): prompt + tokens generated so far
    n_remain: int | None  # remaining decode budget; None/-1 = unbounded
    # Prefill progress (server-context.cpp:714-722, updated per batch :3510/:3599):
    # tokens decoded so far (0 while still in prefill) and uncached prompt tokens
    # processed so far. None when the server did not report them.
    n_decoded: int | None = None
    n_prompt_tokens_processed: int | None = None

    @property
    def prefilling(self) -> bool:
        """In flight and has not produced a token yet."""
        return self.is_processing and self.n_decoded == 0


@dataclass(frozen=True)
class PoolOccupancy:
    """The server's REAL KV occupancy, including load the orchestrator never
    admitted (opencode straight to :8083, scripts, other processes)."""

    url: str
    slots: tuple[SlotState, ...]
    source: str = "live_slots"

    @property
    def processing(self) -> int:
        return sum(1 for s in self.slots if s.is_processing)

    @property
    def used_tokens(self) -> int:
        """Cells held by in-flight requests right now. Idle slots' cached
        prefixes are purgeable (and restorable from --cache-ram), so free."""
        return sum(s.n_prompt_tokens for s in self.slots if s.is_processing)

    def long_prefills(self, min_processed_tokens: int) -> int:
        """Slots still in prefill that have already processed at least
        ``min_processed_tokens`` uncached prompt tokens — a long prefill in
        flight, whoever sent it (another API worker, a client that bypasses the
        orchestrator). 0 when the server does not report prefill progress."""
        if min_processed_tokens <= 0:
            return 0
        return sum(
            1 for s in self.slots
            if s.prefilling and (s.n_prompt_tokens_processed or 0) >= min_processed_tokens
        )

    def projected_tokens(self, new_token_ratio: float = 1.0) -> int:
        """In-flight cells plus the (ratio-weighted) decode they may still add."""
        total = 0
        for s in self.slots:
            if not s.is_processing:
                continue
            remain = s.n_remain if isinstance(s.n_remain, int) and s.n_remain > 0 else 0
            total += s.n_prompt_tokens + int(math.ceil(remain * max(0.0, new_token_ratio)))
        return total


def parse_slots(url: str, body: Any) -> PoolOccupancy | None:
    """Build a PoolOccupancy from a ``GET /slots`` body (a JSON array)."""
    if not isinstance(body, list):
        return None
    slots: list[SlotState] = []
    for i, raw in enumerate(body):
        if not isinstance(raw, dict):
            continue
        nxt = raw.get("next_token")
        if isinstance(nxt, list):
            nxt = nxt[0] if nxt and isinstance(nxt[0], dict) else {}
        n_remain = nxt.get("n_remain") if isinstance(nxt, dict) else None
        n_remain = n_remain if isinstance(n_remain, int) and not isinstance(n_remain, bool) else None
        n_decoded = nxt.get("n_decoded") if isinstance(nxt, dict) else None
        n_decoded = n_decoded if isinstance(n_decoded, int) and not isinstance(n_decoded, bool) else None
        n_processed = raw.get("n_prompt_tokens_processed")
        n_processed = (n_processed if isinstance(n_processed, int)
                       and not isinstance(n_processed, bool) else None)
        n_prompt = raw.get("n_prompt_tokens")
        n_prompt = n_prompt if isinstance(n_prompt, int) and n_prompt > 0 else 0
        slots.append(SlotState(
            slot_id=raw.get("id") if isinstance(raw.get("id"), int) else i,
            n_ctx=_positive_int(raw.get("n_ctx")),
            is_processing=bool(raw.get("is_processing")),
            n_prompt_tokens=n_prompt,
            n_remain=n_remain,
            n_decoded=n_decoded,
            n_prompt_tokens_processed=n_processed,
        ))
    return PoolOccupancy(url=url, slots=tuple(slots))


def _default_fetch_slots(url: str, timeout_s: float) -> Any:
    import httpx

    resp = httpx.get(f"{url.rstrip('/')}/slots", timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _default_fetch(url: str, timeout_s: float) -> dict[str, Any] | None:
    import httpx

    resp = httpx.get(f"{url.rstrip('/')}/props", timeout=timeout_s)
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, dict) else None


class ContextLimitResolver:
    """Cached per-URL / per-role context limits (live → registry)."""

    def __init__(
        self,
        *,
        ttl_s: float | None = None,
        props_timeout_s: float = DEFAULT_PROPS_TIMEOUT_S,
        fetch_props: Callable[[str, float], dict[str, Any] | None] | None = None,
        fetch_slots: Callable[[str, float], Any] | None = None,
        occupancy_ttl_s: float = DEFAULT_OCCUPANCY_TTL_S,
        registry_facts: Callable[[], dict[int, dict[str, Any]]] | None = None,
        role_urls: Callable[[], dict[str, list[str]]] | None = None,
        live: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_s is None:
            try:
                ttl_s = float(os.environ.get(TTL_ENV, DEFAULT_TTL_S))
            except ValueError:
                ttl_s = DEFAULT_TTL_S
        self._ttl_s = max(0.0, ttl_s)
        self._props_timeout_s = props_timeout_s
        self._fetch = fetch_props or _default_fetch
        self._fetch_slots = fetch_slots or _default_fetch_slots
        self._occupancy_ttl_s = max(0.0, occupancy_ttl_s)
        self._occupancy_cache: dict[str, tuple[float, PoolOccupancy | None]] = {}
        self._registry_facts_fn = registry_facts or registry_facts_by_port
        self._role_urls_fn = role_urls or registry_role_urls
        self._live = live
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, ContextLimit | None]] = {}
        self._registry_cache: dict[int, dict[str, Any]] | None = None
        self._role_url_cache: dict[str, list[str]] | None = None

    # -- configuration ----------------------------------------------------
    def live_enabled(self) -> bool:
        if self._live is not None:
            return self._live
        return os.environ.get(LIVE_ENV, "on").strip().lower() not in {"0", "off", "false", "no"}

    def _registry_facts(self) -> dict[int, dict[str, Any]]:
        if self._registry_cache is None:
            try:
                self._registry_cache = self._registry_facts_fn() or {}
            except Exception:
                log.debug("context limits: registry facts unavailable", exc_info=True)
                self._registry_cache = {}
        return self._registry_cache

    def urls_for_role(self, role: str) -> list[str]:
        if self._role_url_cache is None:
            try:
                self._role_url_cache = self._role_urls_fn() or {}
            except Exception:
                log.debug("context limits: role urls unavailable", exc_info=True)
                self._role_url_cache = {}
        return list(self._role_url_cache.get(str(role), []))

    # -- lookups ----------------------------------------------------------
    def limit_for_url(self, url: str) -> ContextLimit | None:
        url = (split_urls(url) or [""])[0]
        if not url:
            return None
        now = self._clock()
        with self._lock:
            cached = self._cache.get(url)
        if cached is not None and cached[0] > now:
            return cached[1]

        port = _port(url)
        reg = self._registry_facts().get(port) if port is not None else None
        limit: ContextLimit | None = None
        ttl = self._ttl_s
        if self.live_enabled():
            try:
                props = self._fetch(url, self._props_timeout_s)
                limit = parse_props(url, props, reg) if props else None
            except Exception as exc:
                log.debug("context limits: GET %s/props failed: %s", url, exc)
                limit = None
            if limit is None:
                ttl = min(ttl, FAILURE_TTL_S)
        if limit is None:
            limit = limit_from_registry(url, reg)
            if limit is not None:
                log.info(
                    "context limits: %s from registry (live /props unavailable): "
                    "per_request_n_ctx=%d slots=%s kv_unified=%s",
                    url, limit.per_request_n_ctx, limit.total_slots, limit.kv_unified,
                )
        with self._lock:
            self._cache[url] = (now + ttl, limit)
        return limit

    def pool_occupancy(self, url: str) -> PoolOccupancy | None:
        """Live ``/slots`` occupancy for ``url``, cached ``occupancy_ttl_s``
        (default 1.5 s). None when live reads are off or /slots is unavailable
        (``--no-slots``, down, timeout) — callers then fall back to their own
        accounting. A failure is cached as briefly as a success."""
        url = (split_urls(url) or [""])[0]
        if not url or not self.live_enabled():
            return None
        now = self._clock()
        with self._lock:
            cached = self._occupancy_cache.get(url)
        if cached is not None and cached[0] > now:
            return cached[1]
        try:
            occ = parse_slots(url, self._fetch_slots(url, self._props_timeout_s))
        except Exception as exc:
            log.debug("context limits: GET %s/slots failed: %s", url, exc)
            occ = None
        with self._lock:
            self._occupancy_cache[url] = (now + self._occupancy_ttl_s, occ)
        return occ

    def limit_for_role(self, role: str, urls: list[str] | str | None = None) -> ContextLimit | None:
        """The binding (smallest) per-request limit across the role's instances.

        A role served by several instances (frontdoor: -np 4 on :8070, -np 1 on
        :8080/:8180) can land on any of them, so the safe answer is the minimum.
        """
        if isinstance(urls, str):
            urls = split_urls(urls)
        candidates = list(urls or []) or self.urls_for_role(role)
        limits = [lim for lim in (self.limit_for_url(u) for u in candidates) if lim is not None]
        if not limits:
            return None
        return min(limits, key=lambda lim: lim.per_request_n_ctx)

    def observe(self, url: str, *, n_ctx: int | None) -> None:
        """Record a server-reported per-slot n_ctx (e.g. from a 400 body).

        The model cap still applies on top of what the server reports."""
        url = (split_urls(url) or [""])[0]
        n_ctx = _positive_int(n_ctx)
        if not url or n_ctx is None:
            return
        with self._lock:
            cached = self._cache.get(url)
            base = cached[1] if cached else None
            if base is not None and base.slot_n_ctx == n_ctx:
                return
            if base is not None:
                # A clamped pool survives only while the server still reports the
                # clamped value; below the cap the slot n_ctx IS the unified pool.
                clamped = (base.pool_n_ctx if base.request_cap and n_ctx >= base.request_cap
                           else None)
                limit = replace(base, per_request_n_ctx=capped_n_ctx(n_ctx, base.request_cap),
                                server_n_ctx=n_ctx, pool_n_ctx=clamped, source="observed")
            else:
                limit = ContextLimit(url=url, per_request_n_ctx=n_ctx, total_slots=None,
                                     kv_unified=None, source="observed")
            self._cache[url] = (self._clock() + self._ttl_s, limit)

    def invalidate(self, url: str | None = None) -> None:
        with self._lock:
            if url is None:
                self._cache.clear()
                self._occupancy_cache.clear()
                self._registry_cache = None
                self._role_url_cache = None
            else:
                self._cache.pop((split_urls(url) or [""])[0], None)

    def larger_context_role(
        self,
        needed_tokens: int,
        *,
        candidates: list[str],
        exclude: set[str] | None = None,
        url_for_role: Callable[[str], list[str] | str | None] | None = None,
    ) -> tuple[str, ContextLimit] | None:
        """First candidate role (in order) whose per-request limit fits ``needed_tokens``."""
        exclude = set(exclude or ())
        for role in candidates:
            if role in exclude:
                continue
            urls = url_for_role(role) if url_for_role else None
            limit = self.limit_for_role(role, urls)
            if limit is not None and limit.fits(needed_tokens):
                return role, limit
        return None


_resolver: ContextLimitResolver | None = None
_resolver_lock = threading.Lock()


def get_context_limit_resolver() -> ContextLimitResolver:
    global _resolver
    with _resolver_lock:
        if _resolver is None:
            _resolver = ContextLimitResolver()
        return _resolver


def set_context_limit_resolver(resolver: ContextLimitResolver | None) -> None:
    """Install (or with None, reset) the process-wide resolver. For tests."""
    global _resolver
    with _resolver_lock:
        _resolver = resolver


def context_overflow_roles() -> list[str]:
    """Ordered roles a too-large request may be rerouted to.

    ``ORCHESTRATOR_CONTEXT_OVERFLOW_ROLES`` (comma list) overrides. The default
    is only the long-context specialist; widening it (e.g. to the Flash-Next
    full-CPU process, 262144 per request at -np 1 — architect_general since the
    2026-09-27 ARCHITECT SWAP, architect_critic before) is a routing-policy choice
    for the operator.
    """
    raw = os.environ.get("ORCHESTRATOR_CONTEXT_OVERFLOW_ROLES")
    if raw is not None:
        return [r.strip() for r in raw.split(",") if r.strip()]
    return ["ingest_long_context"]
