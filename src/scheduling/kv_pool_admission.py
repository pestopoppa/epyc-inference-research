"""Shared (unified) KV pool admission — queue, never oversubscribe.

See the block comment below for the failure it prevents and the design.
"""

from __future__ import annotations

import logging
import os
import math
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Shared (unified) KV pool admission ────────────────────────────────────
#
# The per-backend semaphores in src/api/admission.py count REQUESTS (== -np).
# Under --kv-unified that is not the binding constraint: every slot may take a
# request as large as the whole -c, and the server admits each one against slot
# n_ctx only
# (server-context.cpp:3303-3310) — it never checks the pool's occupancy. Two
# long requests (e.g. 120k + 90k on a 196608 pool) are both admitted, and when
# llama_decode runs out of cells the server purges idle slots, halves the batch
# to 1 and then fails EVERY in-flight request with "Context size has been
# exceeded." (:3759-3764, :2949-2951).
#
# So this is the PRIMARY mechanism, modelled on how vLLM/SGLang schedule a
# shared KV pool: never oversubscribe it. A request whose token reservation
# (prompt + generation budget) does not fit alongside the in-flight ones is
# QUEUED, not dispatched, strictly first-come-first-served per server (a
# stream of small requests cannot starve a waiting long one), until it fits or
# its own deadline / cancellation ends the wait. Retry-on-exhaustion in the
# inference layer (src/llm_primitives/context_recovery.py) is only the
# FALLBACK, for load this process cannot see (clients that bypass the
# orchestrator, e.g. opencode straight to :8083) and estimate error.
#
# A lone request is always admitted whatever its size — the server decides
# whether it fits one slot (HTTP 400 → reroute/typed error).
#
# ── One long prefill at a time, per server (2026-10-03, :8083 decision step 1) ─
#
# A long prefill shares the GPU with its neighbours' decode: on :8083 a slot
# decoding at 28.6 tok/s solo fell to 7.8 tok/s while a neighbour prefilled, and
# two concurrent cold prefills of 46,230 tokens took 395 s and 665 s instead of
# finishing one after the other (tmp/prefill-share-20261003/report.md §2, §4).
# So a request whose estimated prompt is at least LONG_PREFILL_TOKENS takes a
# per-server "long prefill" lease when admitted, and a second long request waits
# while the lease is held. The lease ends at the first sign the prefill is over
# (``prefill_done``: the caller saw the first output chunk), at ``release``, or
# when the prefill must be over at a conservative rate floor (PREFILL_FLOOR_TPS;
# a batch-transport caller never reports a first chunk, and holding the lease
# through its whole decode would serialise long decodes, which is not the rule).
#
# The wait is the same wait as the token wait above: one FCFS queue, bounded by
# the request's deadline / ORCHESTRATOR_KV_POOL_WAIT_S / cancellation, and the
# same queue bound. One refinement keeps short traffic moving: while EVERY
# request ahead of a ticket is a long one held back by the lease, a SHORT request
# may pass them if it fits with their reservations counted as already taken —
# so passing can never starve the long requests of pool tokens.
#
# ── KVU-15a: the lease is HOST-WIDE, and sized on NEW tokens (2026-10-03) ──────
#
# Cross-process. The API runs six uvicorn workers; a per-process lease let two
# of them start long prefills on one server seconds apart. The lease is now an
# exclusive flock on ``{tmp_dir}/kv_prefill_lease.{host}_{port}.lock``
# (src/runtime/long_prefill_lease.py — the same primitive and tmp dir as the
# per-region inference locks). The in-process record (``_prefill_lease``) still
# maps the lease to its ticket and expiry; the open file description held next
# to it is what other workers see. It is released on prefill_done (first chunk),
# release (completion, error, cancel, timeout — the caller's ``finally``), at the
# rate-floor expiry (a timer in the holding process, so an idle holder still
# lets go on time), and by the kernel when the holding worker dies. Lease
# acquisition is non-blocking and sits between the request semaphore and the
# inference/region locks in the lock order (see long_prefill_lease.py), so it
# cannot deadlock with them. ORCHESTRATOR_KV_POOL_CROSS_PROCESS_LEASE=0 returns
# to the per-process lease.
#
# New tokens, not the whole prompt. A long prompt that is mostly a cache hit
# prefills only its suffix, so holding it behind the lease cost a wait for
# nothing. When the caller passes the prompt text, ``/slots`` (which already
# returns each slot's detokenized ``prompt``) gives the longest prefix an IDLE
# slot still holds; the request is sized at
#     min(whole, ceil(uncached_chars / 3) + ORCHESTRATOR_KV_POOL_CACHE_CREDIT_MARGIN_TOKENS)
# The margin (default 4096 tokens, two -ub 2048 batches) covers what the server
# may re-process anyway: a hybrid/recurrent model can only resume from a context
# checkpoint at or before the divergence point, and tokenizer boundaries move a
# little. The credit is a heuristic and stays conservative: no prompt text (the
# chat-payload path — the server's chat template renders a text the orchestrator
# never sees), no /slots, no prompt in /slots, or no matching idle slot all fall
# back to the whole-prompt estimate; only an idle slot counts (a busy one cannot
# take the request); a negative margin disables the credit. Residual risk: the
# matched slot is purged or taken by another request between admission and
# dispatch, and a long cold prefill runs without the lease — the /slots
# observation below then catches it for everyone else. The KV RESERVATION is
# never credited: cached cells still occupy the pool.
#
# ── KVU-15c: the credit's sources — prefix HISTORY first, /slots text second ──
#
# The /slots text above exists only under LLAMA_SERVER_SLOTS_DEBUG (v10
# server-context.cpp to_json :699-730, ``slot.to_json(slots_debug == 0)`` at
# :2541), so in production that credit never applied. The primary source is now
# the orchestrator's own history of what it served on the server
# (src/scheduling/prefix_history.py): the caller passes ``prefix_ladder`` (the
# UFH14-B4 ``prefix_fp`` fingerprint at a denser depth ladder,
# ``serving_calls.prefix_ladder``) and the LONGEST depth at which it equals a
# call that finished on the same server — within that server's ``--cache-ram``
# survival, a time window and the same server launch — is credited:
#     min(whole, ceil((ladder_chars - matched_chars) * rate) + margin)
# ``rate`` is the larger of 1/3 token per char (the conservative estimator) and
# the matched call's MEASURED prompt tokens per fingerprinted char, so the suffix
# is never under-sized. The ladder of a chat payload fingerprints the JSON of
# ``tools`` + ``messages`` (the same text on the passthrough and client-tool
# lanes), not the rendered template, so its suffix is measured in JSON chars —
# longer than the content, i.e. again conservative. The same margin env var
# applies (< 0 disables both sources). When /slots does carry text (a debug
# window) both sources are computed and the larger credit wins; the decision is
# recorded per ticket (``admission_record``) with ``cache_credit_source``
# (``fp_history`` | ``slots_text`` | None), the credited tokens, and — when no
# credit applied — ``cache_credit_unavailable`` naming why for each source (e.g.
# ``slots_no_prompt``), so an inert credit never looks like zero hits (KVU-15b).
#
# Clients that bypass the orchestrator hold no lease at all. So the rule also
# reads the server: a slot that is still in prefill (``n_decoded == 0``) and has
# already processed LONG_PREFILL_OBSERVE_TOKENS uncached tokens (``/slots``
# ``n_prompt_tokens_processed``) counts as a long prefill in flight, whoever sent
# it.

KV_POOL_WAIT_ENV = "ORCHESTRATOR_KV_POOL_WAIT_S"
# Used only when the request carries no deadline of its own. Long on purpose:
# queueing is the correct behaviour under pool pressure, and a long in-flight
# request can legitimately hold the pool for many minutes.
DEFAULT_KV_POOL_WAIT_S = 1800.0

# Bounded queue (vLLM `max_num_queued_reqs`, SGLang `--max-queued-requests`,
# TGI `--max-concurrent-requests`): past this many waiters per server a new
# request is refused at once (503 + Retry-After) instead of waiting silently.
# <= 0 disables the bound.
KV_POOL_MAX_QUEUED_ENV = "ORCHESTRATOR_KV_POOL_MAX_QUEUED"
DEFAULT_KV_POOL_MAX_QUEUED = 8

# Adaptive decode reservation (SGLang `new_token_ratio`): reserve
# prompt + ceil(max_new_tokens * ratio). The ratio starts at INIT, decays by
# DECAY per request that completes without pool trouble, never below MIN, and
# snaps back to INIT on any POOL_EXHAUSTED for that server. Reserving the full
# max_tokens over-serialises thinking-mode calls (max_tokens 32768 that finish
# far below it); a ratio that is too low shows up as POOL_EXHAUSTED, which
# resets it and is absorbed by the retry fallback.
KV_POOL_RATIO_INIT_ENV = "ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO"
KV_POOL_RATIO_MIN_ENV = "ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO_MIN"
KV_POOL_RATIO_DECAY_ENV = "ORCHESTRATOR_KV_POOL_NEW_TOKEN_RATIO_DECAY"
DEFAULT_NEW_TOKEN_RATIO = 1.0
DEFAULT_MIN_NEW_TOKEN_RATIO = 0.3
DEFAULT_NEW_TOKEN_RATIO_DECAY = 0.05

# One-long-prefill rule (see the block comment above). <= 0 disables the rule.
# Default 16384: on :8083 the new-token prefill p90 is 20,702, so the rule bites
# on roughly the heaviest decile of prefills, which run >= ~25 s even solo
# (8-32k solo prefill ~658 tok/s), and leaves ordinary agentic turns alone.
KV_POOL_LONG_PREFILL_ENV = "ORCHESTRATOR_KV_POOL_LONG_PREFILL_TOKENS"
DEFAULT_LONG_PREFILL_TOKENS = 16384
# Prefill-rate floor (tok/s) that bounds how long a lease is held when the caller
# never reports the first output. 250 is below every measured :8083 long-prefill
# rate (all-traffic 32-64k: 257, >=64k: 325). <= 0: hold until prefill_done or
# release.
KV_POOL_PREFILL_FLOOR_TPS_ENV = "ORCHESTRATOR_KV_POOL_PREFILL_FLOOR_TPS"
DEFAULT_PREFILL_FLOOR_TPS = 250.0
# A /slots prefill that has already processed this many uncached tokens is a
# long prefill in flight (capped at the threshold; <= 0 disables observation).
# 4096 = two -ub 2048 batches, ~6 s of :8083 prefill.
KV_POOL_LONG_PREFILL_OBSERVE_ENV = "ORCHESTRATOR_KV_POOL_LONG_PREFILL_OBSERVE_TOKENS"
DEFAULT_LONG_PREFILL_OBSERVE_TOKENS = 4096
# KVU-15a: host-wide lease (flock per server). "0"/"false"/"off" = per-process.
KV_POOL_CROSS_PROCESS_LEASE_ENV = "ORCHESTRATOR_KV_POOL_CROSS_PROCESS_LEASE"
# KVU-15a: tokens added back to the uncached-suffix estimate when a cached
# prefix is credited (< 0 disables the credit: whole-prompt sizing).
KV_POOL_CACHE_CREDIT_MARGIN_ENV = "ORCHESTRATOR_KV_POOL_CACHE_CREDIT_MARGIN_TOKENS"
DEFAULT_CACHE_CREDIT_MARGIN_TOKENS = 4096
# Same ratio as context_limits.CONSERVATIVE_CHARS_PER_TOKEN (the estimator the
# caller sizes the whole prompt with), so whole and suffix estimates agree.
_SUFFIX_CHARS_PER_TOKEN = 3.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class KVPoolQueueFull(RuntimeError):
    """The per-server admission queue is at its bound; retry later (503)."""

    def __init__(self, url: str, queued: int, limit: int) -> None:
        super().__init__(
            f"KV pool admission queue for {url} is full ({queued} waiting, limit {limit})"
        )
        self.url = url
        self.queued = queued
        self.limit = limit


def _default_occupancy(url: str) -> Any:
    from src.backends.context_limits import get_context_limit_resolver

    return get_context_limit_resolver().pool_occupancy(url)


class SharedKVPoolAdmission:
    """FCFS token reservations against each server's shared KV pool.

    ``_fits`` counts the larger of (a) this process's own reservations and
    (b) the server's REAL occupancy from ``GET /slots`` (in-flight cells plus
    their ratio-weighted remaining decode), so load that bypasses the
    orchestrator is seen. When /slots is unavailable it degrades to (a).
    """

    def __init__(
        self,
        occupancy: Callable[[str], Any] | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        cross_process: bool | None = None,
        history: Any = None,
    ) -> None:
        self._cond = threading.Condition()
        self._inflight: dict[str, dict[int, int]] = {}
        self._queue: dict[str, list[int]] = {}
        self._ratio: dict[str, float] = {}
        self._next_ticket = 0
        self._occupancy_fn = occupancy if occupancy is not None else _default_occupancy
        self._clock = clock
        # ticket -> (reservation tokens, is_long) for every queued ticket.
        self._waiting: dict[int, tuple[int, bool]] = {}
        # url -> (holder ticket, lease expiry on self._clock or None = no expiry).
        self._prefill_lease: dict[str, tuple[int, float | None]] = {}
        # KVU-15a: url -> the host-wide flock behind that lease, and the timer
        # that drops it at its expiry. None = decided per call from the env.
        self._cross_process = cross_process
        self._lease_handles: dict[str, Any] = {}
        self._lease_timers: dict[str, threading.Timer] = {}
        # url -> queue-wait accounting (see get_status).
        self._stats: dict[str, dict[str, float]] = {}
        # KVU-15c: the prefix-history source (None = the process-wide one) and
        # each admitted ticket's admission decision, until release.
        self._history = history
        self._admission_info: dict[int, dict[str, Any]] = {}

    # -- long-prefill rule -----------------------------------------------------
    @staticmethod
    def long_prefill_threshold() -> int:
        """Estimated prompt tokens at/above which a request is a long prefill (0 = off)."""
        return max(0, _env_int(KV_POOL_LONG_PREFILL_ENV, DEFAULT_LONG_PREFILL_TOKENS))

    def is_long_prefill(self, prompt_tokens: int) -> bool:
        threshold = self.long_prefill_threshold()
        return threshold > 0 and int(prompt_tokens) >= threshold

    def cross_process_lease(self) -> bool:
        if self._cross_process is not None:
            return bool(self._cross_process)
        return os.environ.get(KV_POOL_CROSS_PROCESS_LEASE_ENV, "1").strip().lower() not in {
            "0", "false", "no", "off"}

    def _drop_lease(self, url: str) -> None:
        """End ``url``'s lease here: forget the ticket, release the host-wide
        flock, stop the expiry timer. Caller holds the lock."""
        self._prefill_lease.pop(url, None)
        handle = self._lease_handles.pop(url, None)
        if handle is not None:
            handle.release()
        timer = self._lease_timers.pop(url, None)
        if timer is not None:
            timer.cancel()

    def _expire_lease(self, url: str, ticket: int) -> None:
        """Timer callback: the rate-floor expiry, in real time, so a holder whose
        process has no other admission traffic still lets the lease go."""
        with self._cond:
            lease = self._prefill_lease.get(url)
            if lease is not None and lease[0] == ticket:
                self._drop_lease(url)
                self._cond.notify_all()

    def _take_lease(self, url: str, ticket: int, expires: float | None) -> bool:
        """Take ``url``'s lease for ``ticket`` (host-wide flock first when
        enabled). False = another holder has it. Caller holds the lock."""
        handle = None
        if self.cross_process_lease():
            from src.runtime import long_prefill_lease

            handle = long_prefill_lease.try_acquire(url, payload={
                "ticket": ticket,
                "expires_in_s": None if expires is None else round(expires - self._clock(), 3),
            })
            if handle is None:
                return False
        self._drop_lease(url)  # defensive: never two records for one url
        self._prefill_lease[url] = (ticket, expires)
        if handle is not None:
            self._lease_handles[url] = handle
        if expires is not None:
            timer = threading.Timer(max(0.0, expires - self._clock()), self._expire_lease,
                                    args=(url, ticket))
            timer.daemon = True
            self._lease_timers[url] = timer
            timer.start()
        return True

    def _lease_holder(self, url: str) -> int | None:
        """The ticket holding ``url``'s long-prefill lease IN THIS PROCESS,
        dropping an expired one."""
        lease = self._prefill_lease.get(url)
        if lease is None:
            return None
        holder, expires = lease
        if expires is not None and self._clock() >= expires:
            self._drop_lease(url)
            return None
        return holder

    def _lease_busy(self, url: str) -> bool:
        """Is ``url``'s lease held by anyone — this process or another worker?
        Caller holds the lock."""
        if self._lease_holder(url) is not None:
            return True
        if self.cross_process_lease():
            from src.runtime import long_prefill_lease

            return long_prefill_lease.is_held(url)
        return False

    def long_prefill_holder(self, url: str) -> int | None:
        """This process's ticket holding ``url``'s lease, or None."""
        with self._cond:
            return self._lease_holder(url)

    def long_prefill_lease_busy(self, url: str) -> bool:
        """Whether any process holds ``url``'s long-prefill lease."""
        with self._cond:
            return self._lease_busy(url)

    # -- new-token estimate (KVU-15a) -------------------------------------------
    @staticmethod
    def cache_credit_margin() -> int:
        return _env_int(KV_POOL_CACHE_CREDIT_MARGIN_ENV, DEFAULT_CACHE_CREDIT_MARGIN_TOKENS)

    def prefill_tokens_estimate(self, prompt_tokens: int, prompt_text: str | None,
                                cached_chars: int) -> int:
        """Tokens the server will actually prefill: the whole-prompt estimate,
        lowered to the uncached suffix + margin when an idle slot's cached prompt
        covers a prefix of ``prompt_text`` (see the KVU-15a block comment)."""
        prompt_tokens = int(prompt_tokens)
        margin = self.cache_credit_margin()
        if margin < 0 or not prompt_text or cached_chars <= 0:
            return prompt_tokens
        suffix_chars = max(0, len(prompt_text) - int(cached_chars))
        suffix_tokens = int(math.ceil(suffix_chars / _SUFFIX_CHARS_PER_TOKEN))
        return max(0, min(prompt_tokens, suffix_tokens + margin))

    def _history_source(self) -> Any:
        if self._history is not None:
            return self._history
        from src.scheduling.prefix_history import get_prefix_history

        return get_prefix_history()

    def history_prefill_estimate(self, url: str, prompt_tokens: int,
                                 prefix_ladder: dict[str, Any] | None
                                 ) -> tuple[int, dict[str, Any]]:
        """``(new-token estimate, facts)`` from the KVU-15c prefix history.
        ``facts`` carries ``matched_chars`` / ``prefix_tokens_est`` on a match,
        else ``reason``."""
        prompt_tokens = int(prompt_tokens)
        margin = self.cache_credit_margin()
        if margin < 0:
            return prompt_tokens, {"reason": "disabled"}
        try:
            match, reason = self._history_source().lookup(url, prefix_ladder)
        except Exception:
            logger.debug("KV pool admission: prefix history lookup failed", exc_info=True)
            match, reason = None, "no_history"
        if match is None or match.matched_chars <= 0:
            return prompt_tokens, {"reason": reason or "no_match"}
        measured = match.tokens_per_char()
        rate = max(1.0 / _SUFFIX_CHARS_PER_TOKEN, measured or 0.0)
        ladder_chars = int((prefix_ladder or {}).get("chars") or 0)
        suffix_chars = max(0, ladder_chars - int(match.matched_chars))
        suffix_tokens = int(math.ceil(suffix_chars * rate))
        estimate = max(0, min(prompt_tokens, suffix_tokens + margin))
        # Best estimate of the cached prefix in tokens (what the server's
        # ``cache_n`` should reach): the measured ratio, else ~4 chars/token.
        prefix_tokens = int(match.matched_chars * measured) if measured else (
            int(match.matched_chars) // 4)
        return estimate, {
            "matched_chars": int(match.matched_chars),
            "prefix_tokens_est": prefix_tokens,
            "history_age_s": match.age_s,
            "history_tokens_since": match.tokens_since,
        }

    def _credit(self, url: str, tokens: int, prompt_text: str | None,
                prefix_ladder: dict[str, Any] | None, observed: tuple | None
                ) -> tuple[int, dict[str, Any]]:
        """The new-token estimate and the admission-record credit fields: the
        better of the history and /slots-text sources (KVU-15c)."""
        tokens = int(tokens)
        hist_tokens, hist = self.history_prefill_estimate(url, tokens, prefix_ladder)
        cached_chars = int(observed[3]) if observed is not None and len(observed) > 3 else 0
        slots_tokens = self.prefill_tokens_estimate(tokens, prompt_text, cached_chars)
        if self.cache_credit_margin() < 0:
            slots_reason: str | None = "disabled"
        elif observed is None:
            slots_reason = "no_prompt_text" if not prompt_text else "slots_unavailable"
        else:
            slots_reason = observed[4] if len(observed) > 4 else None
            if slots_reason is None and cached_chars <= 0:
                slots_reason = "no_slot_match"
        info: dict[str, Any] = {
            "prompt_tokens_est": tokens,
            "cache_credit_source": None,
            "cache_credited_tokens": 0,
            "cache_credit_prefix_tokens_est": None,
            "cache_credit_matched_chars": None,
            "cache_credit_unavailable": None,
        }
        best = tokens
        if "matched_chars" in hist and hist_tokens <= slots_tokens:
            best = hist_tokens
            info.update(cache_credit_source="fp_history",
                        cache_credit_prefix_tokens_est=hist["prefix_tokens_est"],
                        cache_credit_matched_chars=hist["matched_chars"],
                        cache_credit_history_age_s=hist["history_age_s"],
                        cache_credit_history_tokens_since=hist["history_tokens_since"])
        elif cached_chars > 0 and slots_tokens < tokens:
            best = slots_tokens
            info.update(cache_credit_source="slots_text",
                        cache_credit_prefix_tokens_est=cached_chars // 4,
                        cache_credit_matched_chars=cached_chars)
        if info["cache_credit_source"] is None:
            info["cache_credit_unavailable"] = {
                "fp_history": hist.get("reason") or "no_credit",
                "slots_text": slots_reason or "no_credit",
            }
        info["cache_credited_tokens"] = max(0, tokens - best)
        info["prefill_tokens_est"] = best
        return best, info

    def admission_record(self, ticket: int | None) -> dict[str, Any] | None:
        """The admission decision for an admitted ``ticket`` (KVU-15c): the
        long-prefill verdict and the cached-prefix credit with its source, for
        the serving record. None once released or for an unknown ticket."""
        if ticket is None:
            return None
        with self._cond:
            info = self._admission_info.get(ticket)
            return dict(info) if info is not None else None

    def prefill_done(self, url: str, ticket: int | None) -> None:
        """The request holding ``ticket`` finished its prefill (first output seen):
        hand the long-prefill lease on. Harmless for any other ticket."""
        if ticket is None:
            return
        with self._cond:
            lease = self._prefill_lease.get(url)
            if lease is not None and lease[0] == ticket:
                self._drop_lease(url)
                self._cond.notify_all()

    def _stat(self, url: str) -> dict[str, float]:
        return self._stats.setdefault(url, {
            "admitted": 0, "queued_admissions": 0, "queue_wait_s_total": 0.0,
            "queue_wait_s_max": 0.0, "long_prefill_admitted": 0,
            "long_prefill_waits": 0, "long_prefill_wait_s_total": 0.0,
            "short_passed_long": 0, "abandoned": 0, "cache_credited": 0,
            "cache_credit_fp_history": 0, "cache_credit_slots_text": 0,
            "cache_credit_unavailable": 0,
        })

    # -- adaptive decode reservation -----------------------------------------
    def new_token_ratio(self, url: str) -> float:
        with self._cond:
            return self._ratio.get(url, self._ratio_init())

    @staticmethod
    def _ratio_init() -> float:
        return max(0.0, _env_float(KV_POOL_RATIO_INIT_ENV, DEFAULT_NEW_TOKEN_RATIO))

    def _decay(self, url: str) -> None:
        floor = max(0.0, _env_float(KV_POOL_RATIO_MIN_ENV, DEFAULT_MIN_NEW_TOKEN_RATIO))
        step = max(0.0, _env_float(KV_POOL_RATIO_DECAY_ENV, DEFAULT_NEW_TOKEN_RATIO_DECAY))
        current = self._ratio.get(url, self._ratio_init())
        self._ratio[url] = max(min(floor, current), current - step)

    def report_pool_exhausted(self, url: str) -> None:
        """The server ran out of KV cells: stop being optimistic about decode."""
        with self._cond:
            before = self._ratio.get(url, self._ratio_init())
            self._ratio[url] = self._ratio_init()
        logger.warning(
            "KV pool exhausted on %s: new_token_ratio %.2f -> %.2f", url, before, self._ratio_init()
        )

    def reservation_tokens(self, url: str, prompt_tokens: int, max_new_tokens: int) -> int:
        ratio = self.new_token_ratio(url)
        return int(prompt_tokens) + int(math.ceil(max(0, int(max_new_tokens)) * ratio))

    # -- accounting ------------------------------------------------------------
    def in_flight_tokens(self, url: str) -> int:
        with self._cond:
            return sum(self._inflight.get(url, {}).values())

    def queued(self, url: str) -> int:
        with self._cond:
            return len(self._queue.get(url, []))

    def _observe_min_tokens(self) -> int:
        threshold = self.long_prefill_threshold()
        observe = _env_int(KV_POOL_LONG_PREFILL_OBSERVE_ENV, DEFAULT_LONG_PREFILL_OBSERVE_TOKENS)
        if threshold <= 0 or observe <= 0:
            return 0
        return min(observe, threshold)

    def _observed(self, url: str, prompt_text: str | None = None
                  ) -> tuple[int, int, int, int, str | None] | None:
        """(projected in-flight tokens, processing slots, long prefills in flight,
        chars of ``prompt_text`` an idle slot has cached, why that is 0 or None)
        from /slots, or None."""
        try:
            occ = self._occupancy_fn(url)
        except Exception:
            logger.debug("KV pool admission: occupancy read failed for %s", url, exc_info=True)
            return None
        if occ is None:
            return None
        ratio = self.new_token_ratio(url)
        long_prefills = 0
        min_tokens = self._observe_min_tokens()
        if min_tokens > 0 and hasattr(occ, "long_prefills"):
            try:
                long_prefills = int(occ.long_prefills(min_tokens))
            except Exception:
                long_prefills = 0
        cached_chars = 0
        slots_reason: str | None = None
        if not prompt_text:
            slots_reason = "no_prompt_text"
        elif hasattr(occ, "best_cached_prefix_chars"):
            try:
                cached_chars = int(occ.best_cached_prefix_chars(prompt_text))
            except Exception:
                cached_chars = 0
            if cached_chars <= 0:
                # v10 reports slot text only under LLAMA_SERVER_SLOTS_DEBUG.
                try:
                    has_text = any(not s.is_processing and s.prompt_text
                                   for s in getattr(occ, "slots", ()))
                except Exception:
                    has_text = False
                slots_reason = "no_slot_match" if has_text else "slots_no_prompt"
        else:
            slots_reason = "slots_no_prompt"
        return (int(occ.projected_tokens(ratio)), int(occ.processing), long_prefills,
                cached_chars, slots_reason)

    def _fits(self, url: str, tokens: int, pool_tokens: int, observed: tuple | None) -> bool:
        reserved = self._inflight.get(url, {})
        own = sum(reserved.values())
        seen, processing = (observed[0], observed[1]) if observed is not None else (0, 0)
        if not reserved and processing == 0:
            return True  # a lone request is always admitted; the server decides
        return max(own, seen) + tokens <= pool_tokens

    def acquire(
        self,
        url: str,
        tokens: int,
        pool_tokens: int,
        *,
        max_new_tokens: int = 0,
        deadline_s: float | None = None,
        timeout_s: float | None = None,
        cancel_check=None,
        poll_s: float = 0.25,
        max_queued: int | None = None,
        prompt_text: str | None = None,
        prefix_ladder: dict[str, Any] | None = None,
    ) -> int | None:
        """Queue for ``tokens`` (+ ratio-weighted ``max_new_tokens``) of ``url``'s
        pool; return a ticket, or None when the wait ends (deadline, timeout,
        cancellation) before it fits. Raises KVPoolQueueFull when ``max_queued``
        (default ``ORCHESTRATOR_KV_POOL_MAX_QUEUED``) requests already wait.

        ``tokens`` is the estimated prompt; when the request's NEW tokens (the
        whole prompt, less a prefix an idle slot has cached when ``prompt_text``
        is given — KVU-15a) are at or above
        ``ORCHESTRATOR_KV_POOL_LONG_PREFILL_TOKENS`` the request is a long
        prefill and also waits for ``url``'s host-wide long-prefill lease (one
        long prefill in flight per server, across every worker process). Call
        ``prefill_done`` on the first output chunk.

        ``prefix_ladder`` (``serving_calls.prefix_ladder``) is the KVU-15c
        primary credit source: the longest prefix a call that finished on this
        server shares with the request. ``admission_record(ticket)`` returns the
        decision (verdict, credit, source) for the serving record.

        ``deadline_s`` is a ``time.perf_counter`` deadline (the primitives clock).
        ``timeout_s`` defaults to ``ORCHESTRATOR_KV_POOL_WAIT_S`` only when there
        is no deadline; with a deadline the deadline alone bounds the wait.
        """
        pool_tokens = max(1, int(pool_tokens))
        want = self.reservation_tokens(url, tokens, max_new_tokens)
        want = max(1, min(want, pool_tokens))
        is_long = self.is_long_prefill(tokens)
        prefill_tokens = int(tokens)
        if timeout_s is None and deadline_s is None:
            timeout_s = _env_float(KV_POOL_WAIT_ENV, DEFAULT_KV_POOL_WAIT_S)
        if max_queued is None:
            max_queued = _env_int(KV_POOL_MAX_QUEUED_ENV, DEFAULT_KV_POOL_MAX_QUEUED)
        start = time.perf_counter()
        with self._cond:
            queue = self._queue.setdefault(url, [])
            if max_queued > 0 and len(queue) >= max_queued:
                raise KVPoolQueueFull(url, len(queue), max_queued)
            self._next_ticket += 1
            ticket = self._next_ticket
            queue.append(ticket)
            self._waiting[ticket] = (want, is_long)
        logged = False
        waited_on_lease = False
        admitted = False
        try:
            while True:
                # Read the server outside the lock (an HTTP GET, cached ~1.5 s).
                observed = self._observed(url, prompt_text)
                prefill_tokens, credit = self._credit(url, tokens, prompt_text,
                                                      prefix_ladder, observed)
                with self._cond:
                    is_long = self.is_long_prefill(prefill_tokens)
                    self._waiting[ticket] = (want, is_long)
                    verdict = self._admissible(url, ticket, queue, want, is_long,
                                               pool_tokens, observed)
                    if verdict == "head" and is_long and not self._take_lease(
                            url, ticket, self._lease_expiry(prefill_tokens)):
                        verdict = "lease"  # another worker took it first
                    if verdict in ("head", "pass"):
                        self._inflight.setdefault(url, {})[ticket] = want
                        admitted = True
                        waited = time.perf_counter() - start
                        stat = self._stat(url)
                        stat["admitted"] += 1
                        if logged:
                            stat["queued_admissions"] += 1
                        stat["queue_wait_s_total"] += waited
                        stat["queue_wait_s_max"] = max(stat["queue_wait_s_max"], waited)
                        if is_long:
                            stat["long_prefill_admitted"] += 1
                        if waited_on_lease:
                            stat["long_prefill_waits"] += 1
                            stat["long_prefill_wait_s_total"] += waited
                        if verdict == "pass":
                            stat["short_passed_long"] += 1
                        cache_credited = not is_long and self.is_long_prefill(tokens)
                        if cache_credited:
                            stat["cache_credited"] += 1
                        source = credit.get("cache_credit_source")
                        if source:
                            stat[f"cache_credit_{source}"] += 1
                        else:
                            stat["cache_credit_unavailable"] += 1
                        self._admission_info[ticket] = dict(
                            credit,
                            long_prefill=bool(is_long),
                            cache_credited=bool(cache_credited),
                            long_prefill_threshold=self.long_prefill_threshold(),
                            queue_wait_ms=round(waited * 1000.0, 3),
                        )
                        if logged:
                            logger.info(
                                "KV pool admission: %s request of %d tokens admitted after %.1fs%s",
                                url, want, waited,
                                " (waited for the long-prefill lease)" if waited_on_lease else "",
                            )
                        return ticket
                    if verdict == "lease":
                        waited_on_lease = True
                    now = time.perf_counter()
                    if deadline_s is not None and now >= deadline_s:
                        return None
                    if timeout_s is not None and now - start >= max(0.0, timeout_s):
                        return None
                    if cancel_check is not None:
                        try:
                            if cancel_check():
                                return None
                        except Exception:
                            pass
                    if not logged:
                        logger.warning(
                            "KV pool admission: %s own=%d observed=%s pool=%d, %d queued ahead; "
                            "request of %d tokens%s queued (not dispatched)%s",
                            url, sum(self._inflight.get(url, {}).values()),
                            observed[0] if observed else "n/a", pool_tokens,
                            queue.index(ticket), want,
                            " (long prefill)" if is_long else "",
                            " — another long prefill is in flight" if verdict == "lease" else "",
                        )
                        logged = True
                    self._cond.wait(timeout=poll_s)
        finally:
            # Leave the queue whether admitted or abandoned, and wake the next
            # waiter (the head may have changed).
            with self._cond:
                try:
                    queue.remove(ticket)
                except ValueError:
                    pass
                self._waiting.pop(ticket, None)
                if not admitted:
                    self._stat(url)["abandoned"] += 1
                if not queue and self._queue.get(url) is queue:
                    self._queue.pop(url, None)
                self._cond.notify_all()

    def _lease_expiry(self, prompt_tokens: int) -> float | None:
        floor = _env_float(KV_POOL_PREFILL_FLOOR_TPS_ENV, DEFAULT_PREFILL_FLOOR_TPS)
        if floor <= 0:
            return None
        return self._clock() + max(1, int(prompt_tokens)) / floor

    def _admissible(
        self,
        url: str,
        ticket: int,
        queue: list[int],
        want: int,
        is_long: bool,
        pool_tokens: int,
        observed: tuple | None,
    ) -> str:
        """``head``/``pass`` = admit now; ``lease`` = blocked only by the
        long-prefill rule; ``wait`` = anything else. Caller holds the lock."""
        observed_long = int(observed[2]) if observed is not None and len(observed) > 2 else 0
        lease_held = observed_long > 0 or self._lease_busy(url)
        position = queue.index(ticket)
        if position == 0:
            if not self._fits(url, want, pool_tokens, observed):
                return "wait"
            if is_long and lease_held:
                return "lease"
            return "head"
        if is_long or not lease_held:
            return "wait"
        # A short request behind long ones that only the lease holds back may
        # pass them, with their reservations counted as already taken.
        ahead = [self._waiting.get(t, (0, False)) for t in queue[:position]]
        if not all(long_ for _, long_ in ahead):
            return "wait"
        held_for_ahead = sum(w for w, _ in ahead)
        if self._fits(url, want + held_for_ahead, pool_tokens, observed):
            return "pass"
        return "wait"

    def release(self, url: str, ticket: int | None, *, success: bool = True) -> None:
        """Return a reservation. ``success`` = the request finished without pool
        trouble, which decays the decode ratio toward its floor."""
        if ticket is None:
            return
        with self._cond:
            self._admission_info.pop(ticket, None)
            reserved = self._inflight.get(url)
            if reserved is not None:
                reserved.pop(ticket, None)
                if not reserved:
                    self._inflight.pop(url, None)
            lease = self._prefill_lease.get(url)
            if lease is not None and lease[0] == ticket:
                self._drop_lease(url)
            if success:
                self._decay(url)
            self._cond.notify_all()

    def get_status(self) -> dict[str, dict[str, Any]]:
        with self._cond:
            urls = (set(self._inflight) | set(self._queue) | set(self._ratio)
                    | set(self._stats) | set(self._prefill_lease))
            out: dict[str, dict[str, Any]] = {}
            for url in urls:
                stat = dict(self._stats.get(url, {}))
                for key in ("queue_wait_s_total", "queue_wait_s_max", "long_prefill_wait_s_total"):
                    if key in stat:
                        stat[key] = round(stat[key], 3)
                out[url] = {
                    "reserved_tokens": sum(self._inflight.get(url, {}).values()),
                    "in_flight": len(self._inflight.get(url, {})),
                    "queued": len(self._queue.get(url, [])),
                    "new_token_ratio": round(self._ratio.get(url, self._ratio_init()), 4),
                    "long_prefill_threshold": self.long_prefill_threshold(),
                    # Held by any worker process (host-wide flock) / by this one.
                    "long_prefill_lease_held": self._lease_busy(url),
                    "long_prefill_lease_held_here": self._lease_holder(url) is not None,
                    "long_prefill_lease_cross_process": self.cross_process_lease(),
                    "queued_long_prefills": sum(
                        1 for t in self._queue.get(url, []) if self._waiting.get(t, (0, False))[1]
                    ),
                    "admission_stats": stat,
                }
            return out


_shared_pool_admission = SharedKVPoolAdmission()


def get_shared_pool_admission() -> SharedKVPoolAdmission:
    return _shared_pool_admission
